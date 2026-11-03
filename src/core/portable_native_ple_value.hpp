#pragma once
#include "native_qsa_indexer_bf16.hpp"
#include "strata_vk_native_shared.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace strata::core::portable_detail {
// The gathered n-gram embedding is host produced; this owner supplies the
// device input and the actual output consumed by the existing PLE graph.
class PortableNativePleValue {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Context ctx_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeShared, decltype(&strata_vk_native_shared_destroy)> component_{nullptr, strata_vk_native_shared_destroy};
    ggml_tensor *weight_ = nullptr, *input_ = nullptr, *output_ = nullptr, *aux_ = nullptr;
    std::vector<uint16_t> host_weight_;
    std::vector<float> last_input_, last_output_;
    std::filesystem::path directory_;
    bool recorded_ = false;
    size_t bytes_ = 0;
    static constexpr size_t width = 2560;
    static Context context() {
        Context c(ggml_init({32768, nullptr, true}), ggml_free);
        native_shared_need(bool(c), "PLE value: metadata allocation");
        return c;
    }
    static void leaves(ggml_context* c, ggml_tensor*& input, ggml_tensor*& output, ggml_tensor*& aux) {
        input = ggml_new_tensor_2d(c, GGML_TYPE_F32, width, 1);
        output = ggml_new_tensor_2d(c, GGML_TYPE_F32, width, 1);
        aux = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);
        ggml_set_input(input);
        ggml_set_output(output);
    }
public:
    static size_t estimate(ggml_backend_t backend) {
        auto c = context(); ggml_tensor *i, *o, *a; leaves(c.get(), i, o, a);
        auto bank = ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(), ggml_backend_get_default_buffer_type(backend));
        auto m = context(); ggml_new_tensor_1d(m.get(), GGML_TYPE_I32, 128);
        return bank + ggml_backend_alloc_ctx_tensors_from_buft_size(m.get(), ggml_backend_get_default_buffer_type(backend));
    }
    PortableNativePleValue(ggml_backend_t backend, ggml_tensor* weight, std::vector<uint16_t> host_weight,
                           const std::string& directory, size_t budget)
        : weight_(weight), host_weight_(std::move(host_weight)), directory_(directory) {
        native_shared_need(backend && weight && weight->buffer && weight->type == GGML_TYPE_I32 &&
                           ggml_is_contiguous(weight) && ggml_nbytes(weight) == width * width * 2 &&
                           host_weight_.size() == width * width && estimate(backend) <= budget &&
                           !directory.empty() && std::filesystem::create_directory(directory),
                           "PLE value: role, backend, budget or fresh receipt directory");
        const uint16_t endian = 1;
        native_shared_need(*reinterpret_cast<const uint8_t*>(&endian) == 1, "PLE value: little-endian host required");
        native_bf16_matrix_raw_weights(host_weight_.data(), host_weight_.size());
        std::vector<uint8_t> actual(65536);
        const auto* expected = reinterpret_cast<const uint8_t*>(host_weight_.data());
        for (size_t offset = 0; offset < width * width * 2; offset += actual.size()) {
            size_t count = std::min(actual.size(), width * width * 2 - offset);
            ggml_backend_tensor_get(weight, actual.data(), offset, count);
            native_shared_need(std::memcmp(actual.data(), expected + offset, count) == 0, "PLE value: resident role differs");
        }
        ctx_ = context(); leaves(ctx_.get(), input_, output_, aux_);
        const size_t borrowed = ggml_backend_buffer_get_size(weight->buffer);
        native_shared_need(borrowed <= SIZE_MAX - budget, "PLE value: budget overflow");
        component_.reset(strata_vk_ple_value_bf16_create(backend, (std::filesystem::path(directory) / "effective.spv").c_str(), budget + borrowed));
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend));
        native_shared_need(bool(buffer_), "PLE value: bank allocation");
        bytes_ = ggml_backend_buffer_get_size(buffer_.get()) + strata_vk_native_shared_bytes(component_.get());
        native_shared_need(bytes_ <= estimate(backend) && bytes_ <= budget, "PLE value: allocation exceeds admission");
    }
    size_t bytes() const { return bytes_; }
    ggml_tensor* input() const { return input_; }
    ggml_tensor* output() const { return output_; }
    void run(const std::vector<float>& embedding) {
        native_shared_need(embedding.size() == width, "PLE value: embedding size");
        auto expected = native_bf16_matrix_control(embedding.data(), host_weight_.data(), width);
        ggml_backend_tensor_set(input_, embedding.data(), 0, width * 4);
        strata_vk_native_shared_run(component_.get(), 1, weight_, input_, output_, aux_, width, width, 1);
        std::vector<float> actual(width);
        ggml_backend_tensor_get(output_, actual.data(), 0, width * 4);
        native_shared_need(std::memcmp(actual.data(), expected.data(), width * 4) == 0,
                           "PLE value: device output differs from ordered control");
        last_input_ = embedding;
        last_output_ = std::move(actual);
    }
    // Records a graph-completed candidate, not an FFN or sequence commit.
    // The first receipt is lifetime-scoped; reset does not overwrite it.
    void verify_and_record(const ggml_tensor* consumed, uint64_t position, int32_t token,
                           const std::array<int32_t,2>& previous, const std::array<uint32_t,16>& rows) {
        native_shared_need(consumed == output_ && consumed->buffer && ggml_is_contiguous(consumed) &&
                           ggml_nbytes(consumed) == width*4 && last_input_.size() == width && last_output_.size() == width,
                           "PLE value: actual consumer bank differs");
        std::vector<float> input(width), value(width);
        ggml_backend_tensor_get(input_, input.data(), 0, width*4);
        ggml_backend_tensor_get(consumed, value.data(), 0, width*4);
        native_shared_need(std::memcmp(input.data(),last_input_.data(),width*4)==0 &&
                           std::memcmp(value.data(),last_output_.data(),width*4)==0,
                           "PLE value: graph changed input or producer bank");
        if(recorded_)return;
        auto path=directory_/"first-candidate";
        native_shared_need(std::filesystem::create_directory(path),"PLE value: first candidate already exists");
        auto save=[&](const char* name,const void* bytes,size_t n){std::ofstream f(path/name,std::ios::binary);f.write(static_cast<const char*>(bytes),n);f.flush();native_shared_need(bool(f),"PLE value: candidate receipt write");};
        save("input.f32le",input.data(),width*4);
        save("value.f32le",value.data(),width*4);
        std::ofstream meta(path/"metadata.json");
        meta<<"{\"position\":"<<position<<",\"token\":"<<token<<",\"previous\":["<<previous[0]<<","<<previous[1]<<"],\"rows\":[";
        for(size_t i=0;i<rows.size();++i)meta<<(i?",":"")<<rows[i];
        meta<<"],\"direct_producer_bank\":true,\"graph_completed\":true,\"committed\":false,\"qualified\":false}\n";
        meta.flush();
        native_shared_need(bool(meta),"PLE value: candidate metadata write");
        recorded_=true;
    }
};
}
