#pragma once

#include "portable_native_ple_transaction.hpp"
#include "portable_native_ple_key_batch.hpp"
#include "portable_native_ple_value.hpp"
#include "portable_native_ple_copy_graph.hpp"
#include "portable_native_ple_gate.hpp"
#include "portable_native_ple_gated_value.hpp"
#include "portable_native_ple_conv_copy_graph.hpp"
#include "strata/core/portable_ple.hpp"

namespace strata::core::portable_detail {
// Standalone candidate executor. All roles and the backend are borrowed and
// must outlive the executor. Neither this object nor its owners commit state.
class PortableNativePleExecutor {
public:
    // Limit source paths before any owner factory or directory creation.
    static constexpr size_t max_path_bytes = 4096;
    struct Roles {
        ggml_backend_t backend = nullptr;
        const PortablePleTable* table = nullptr;
        ggml_tensor* key_weight = nullptr;
        const void* key_role = nullptr;
        size_t key_bytes = 0;
        ggml_tensor* value_weight = nullptr;
        const uint16_t* value_role = nullptr;
        size_t value_elements = 0;
        std::array<const float*, 3> gamma{};
        const void* conv_role = nullptr;
        size_t conv_bytes = 0;
        uint32_t conv_type = 0;
        std::string directory;
    };
private:
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    static constexpr size_t channels = 10240, width = 2560, metadata = 16384;
    struct Set {
        Context ctx{nullptr, ggml_free};
        Buffer buffer{nullptr, ggml_backend_buffer_free};
        ggml_tensor *history = nullptr, *residual = nullptr, *gated_leaf = nullptr;
        std::unique_ptr<PortableNativePleValue> value;
        std::unique_ptr<PortableNativePleGate> gate;
        std::unique_ptr<PortableNativePleGatedValue> gated;
        std::unique_ptr<PortableNativePleCopyGraph> norm;
        std::unique_ptr<PortableNativePleConvCopyGraph> conv;
        std::array<ggml_tensor*,3> norm_sources{};
        std::array<ggml_tensor*,4> conv_sources{};
    };
    Roles roles_;
    size_t budget_;
    std::unique_ptr<PortableNativePleKeyBatch> key_;
    std::array<Set,2> sets_;
    static void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
    static size_t add(size_t a, size_t b) { need(a <= SIZE_MAX-b, "PLE executor: budget overflow"); return a+b; }
    static Context make_context() {
        Context c(ggml_init({metadata,nullptr,true}),ggml_free);
        need(bool(c),"PLE executor: metadata"); return c;
    }
    static void leaves(ggml_context* c, size_t rows, ggml_tensor*& h, ggml_tensor*& r) {
        h=ggml_new_tensor_1d(c,GGML_TYPE_F32,9*channels);
        r=ggml_new_tensor_1d(c,GGML_TYPE_F32,rows*channels);
        need(h && r,"PLE executor: scratch tensors");
    }
    static size_t scratch_estimate(ggml_backend_t b,size_t rows) {
        auto c=make_context(); ggml_tensor *h,*r; leaves(c.get(),rows,h,r);
        return add(metadata,ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b)));
    }
    static size_t set_estimate(ggml_backend_t b,size_t rows) {
        size_t n=scratch_estimate(b,rows);
        n=add(n,PortableNativePleValue::estimate(b,rows));
        n=add(n,PortableNativePleGate::estimate(b,rows));
        n=add(n,PortableNativePleGatedValue::estimate(b,rows));
        n=add(n,PortableNativePleCopyGraph::estimate(b,rows));
        return add(n,PortableNativePleConvCopyGraph::estimate(b,rows));
    }
    void create_set(size_t rows, const std::filesystem::path& root) {
        auto& s=sets_[rows-1]; auto b=roles_.backend;
        s.ctx=make_context(); leaves(s.ctx.get(),rows,s.history,s.residual);
        s.buffer.reset(ggml_backend_alloc_ctx_tensors(s.ctx.get(),b));
        need(bool(s.buffer) && ggml_backend_buffer_get_size(s.buffer.get()) <= scratch_estimate(b,rows)-metadata,"PLE executor: scratch allocation");
        auto dir=root / (rows==1 ? "t1" : "t2");
        need(std::filesystem::create_directory(dir),"PLE executor: fresh row directory required");
        std::vector<uint16_t> weight(roles_.value_role, roles_.value_role+roles_.value_elements);
        s.value=std::make_unique<PortableNativePleValue>(b,roles_.value_weight,std::move(weight),(dir/"value").string(),PortableNativePleValue::estimate(b,rows),rows);
        // The grouped norm bank is allocated later, but its estimate bounds the
        // shared buffer subsequently bound to the gate.
        s.gate=std::make_unique<PortableNativePleGate>(b,(dir/"gate").string(),PortableNativePleGate::estimate(b,rows),PortableNativePleGroupedNorm::estimate(b,rows),rows);
        s.gated=std::make_unique<PortableNativePleGatedValue>(b,s.value->output(),s.gate->gate(),PortableNativePleGatedValue::estimate(b,rows),rows);
        // Copy connectors expand dependencies. Bind a leaf alias, not the
        // gated graph node, so those connectors cannot execute it again.
        s.gated_leaf=ggml_new_tensor_1d(s.ctx.get(),GGML_TYPE_F32,rows*channels);
        need(s.gated_leaf,"PLE executor: gated alias");
        s.gated_leaf->buffer=s.gated->output_bank()->buffer;
        s.gated_leaf->data=s.gated->output_bank()->data;
        s.norm_sources={key_->output_bank(rows),s.residual,s.gated_leaf};
        auto norm_module=(dir/"norm.spv").string();
        s.norm=std::make_unique<PortableNativePleCopyGraph>(b,s.norm_sources,roles_.gamma,norm_module.c_str(),PortableNativePleCopyGraph::estimate(b,rows),rows);
        s.gate->bind_direct(s.norm->output_bank(0),s.norm->output_bank(1));
        s.conv_sources={s.history,s.norm->output_bank(2),s.residual,s.gated_leaf};
        auto conv_module=(dir/"conv.spv").string();
        s.conv=std::make_unique<PortableNativePleConvCopyGraph>(b,s.conv_sources,roles_.conv_role,roles_.conv_bytes,roles_.conv_type,conv_module.c_str(),PortableNativePleConvCopyGraph::estimate(b,rows),rows);
    }
public:
    struct EstimateBreakdown {
        size_t owner_estimates = 0, key_host = 0, value_host = 0, gate_host = 0;
        size_t copy_host = 0, conv_host = 0, gated_host = 0, gather_host = 0, total = 0;
    };
    // Deliberately sum mixed owner estimates and complete host bounds. Graph
    // metadata and some diagnostics are counted twice rather than subtracting
    // overlapping terms whose lifetimes may change. Borrowed roles/table,
    // transaction storage, allocator overhead and caller copies are excluded.
    static EstimateBreakdown estimate_breakdown(ggml_backend_t b) {
        need(b,"PLE executor: backend");
        EstimateBreakdown x;
        x.owner_estimates=add(add(PortableNativePleKeyBatch::estimate(b),set_estimate(b,1)),set_estimate(b,2));
        x.key_host=PortableNativePleKeyBatch::host_bound(2);
        for (size_t rows : {size_t(1),size_t(2)}) {
            x.value_host=add(x.value_host,PortableNativePleValue::host_bound(rows));
            x.gate_host=add(x.gate_host,PortableNativePleGate::host_bound(rows));
            x.copy_host=add(x.copy_host,PortableNativePleCopyGraph::host_bound(rows));
            x.conv_host=add(x.conv_host,PortableNativePleConvCopyGraph::host_bound(rows));
            x.gated_host=add(x.gated_host,PortableNativePleGatedValue::host_bound(rows));
        }
        // gather_batch retains the batch embedding while gather decodes one
        // additional row, before inserting it into that embedding.
        x.gather_host=ple_host_add(ple_host_mul(3*width,sizeof(float)),ple_host_mul(2,sizeof(std::array<uint32_t,16>)));
        x.total=add(add(add(x.owner_estimates,x.key_host),add(x.value_host,x.gate_host)),
                    add(add(x.copy_host,x.conv_host),add(x.gated_host,x.gather_host)));
        return x;
    }
    static size_t estimate(ggml_backend_t b) { return estimate_breakdown(b).total; }
    // Descriptor-only checks: no owners, backend buffers, or receipt directories
    // are constructed. Resident payload equality is checked by the key owner.
    static void validate_roles(const Roles& roles) {
        need(roles.backend && roles.table && roles.key_weight && roles.key_role && roles.value_weight && roles.value_role &&
             roles.value_elements==width*width && roles.gamma[0] && roles.gamma[1] && roles.gamma[2] && roles.conv_role &&
             !roles.directory.empty(),"PLE executor: missing role");
        if (roles.directory.size() > max_path_bytes)
            throw std::length_error("PLE executor: directory too long");
        need(roles.key_weight->type==GGML_TYPE_Q8_0 && roles.key_weight->ne[0]==width &&
             roles.key_weight->ne[1]==channels && roles.key_weight->buffer && ggml_is_contiguous(roles.key_weight) &&
             roles.key_bytes==ggml_nbytes(roles.key_weight), "PLE executor: key descriptor");
        need(roles.value_weight->buffer && roles.value_weight->type==GGML_TYPE_I32 &&
             ggml_is_contiguous(roles.value_weight) && ggml_nbytes(roles.value_weight)==width*width*2,
             "PLE executor: value descriptor");
        need((roles.conv_type==0 && roles.conv_bytes==40960*4) ||
             (roles.conv_type==1 && roles.conv_bytes==40960*2), "PLE executor: convolution descriptor");
    }
private:
    // Validate the borrowed descriptor before copying its potentially oversized path.
    static const Roles& validated_roles(const Roles& roles) {
        validate_roles(roles);
        return roles;
    }
public:
    explicit PortableNativePleExecutor(const Roles& roles, size_t budget) : roles_(validated_roles(roles)), budget_(budget) {
        need(estimate(roles.backend)<=budget_,"PLE executor: total budget");
        need(std::filesystem::create_directory(roles.directory),"PLE executor: fresh directory required");
        key_=std::make_unique<PortableNativePleKeyBatch>(roles.backend,roles.key_weight,roles.key_role,roles.key_bytes,
                    (std::filesystem::path(roles.directory)/"key").string(),PortableNativePleKeyBatch::estimate(roles.backend));
        create_set(1,roles.directory); create_set(2,roles.directory);
    }
    PortableNativePleExecutor(const PortableNativePleExecutor&)=delete;
    PortableNativePleExecutor& operator=(const PortableNativePleExecutor&)=delete;
    void operator()(const PortableNativePleTransaction::Request& request, const PortableNativePleTransaction::OutputView& out) {
        const size_t rows=request.tokens.size();
        need((rows==1||rows==2) && request.predecessors.size()==rows && request.history.size()==9*channels &&
             request.residual_input.size()==rows*channels && out.normalized.size()==rows*channels &&
             out.history.size()==9*channels && out.residual.size()==rows*channels,"PLE executor: request/output shape");
        auto& s=sets_[rows-1];
        s.norm->reset(); s.gated->reset(); s.gate->begin(); s.conv->abort();
        std::vector<std::array<uint32_t,16>> gathered_rows;
        std::vector<float> embedding; std::string error;
        if (!roles_.table->gather_batch(request.tokens,request.predecessors,gathered_rows,embedding,error))
            throw std::runtime_error("PLE executor: gather: "+error);
        need(gathered_rows.size()==rows && embedding.size()==rows*width,
             "PLE executor: gather extent");
        ggml_backend_tensor_set(s.history,request.history.data(),0,request.history.size()*sizeof(float));
        ggml_backend_tensor_set(s.residual,request.residual_input.data(),0,request.residual_input.size()*sizeof(float));
        key_->run(embedding,rows);
        s.value->run(embedding);
        s.norm->run_key_query(s.norm_sources);
        s.gate->run_direct(s.norm->output(0),s.norm->output(1));
        s.gated->run(s.value->output(),s.gate->gate());
        s.norm->run_gated_value(s.norm_sources);
        s.conv->run(s.conv_sources,request.residual_input);
        ggml_backend_tensor_get(s.norm->output(2),out.normalized.data(),0,out.normalized.size()*sizeof(float));
        ggml_backend_tensor_get(s.conv->history(),out.history.data(),0,out.history.size()*sizeof(float));
        ggml_backend_tensor_get(s.conv->result(),out.residual.data(),0,out.residual.size()*sizeof(float));
    }
};
}
