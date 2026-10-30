#pragma once
#include "portable_qsa_fixture.hpp"
#include "portable_layer_chunk_fixture.hpp"
#include "../../src/core/portable_memory.hpp"
#include <array>
namespace qsa_fixture {
struct AttentionPlanFault {
    static inline decltype(ggml_backend_device_i::supports_op) original = nullptr;
    static bool supports(ggml_backend_dev_t device, const ggml_tensor *tensor) {
        return tensor->op != GGML_OP_LOG && original(device, tensor);
    }
};
inline void shared_attention_tests(PortableQsaOptions options, bool gpu = false) {
    Fixture fixture;
    std::string error;
    options.capacity = 256;
    options.reuse_attention = true;
    options.production = true;
    auto owner = PortableQsaAttention::create(options.execution, error);
    check(bool(owner), error);
    check(owner->memory_usage().host_extra_bound == options.execution.graph_budget + PortableQsaAttention::host_metadata_bound(),
          "shared metadata bound is separate from scheduler cap");
    auto overflow = options.execution; overflow.graph_budget = SIZE_MAX;
    check(!PortableQsaAttention::create(overflow,error), "shared metadata plus graph cap overflow rejected");
    options.attention = owner;
    auto make = [&](int64_t id, PortableQsaOptions opts) {
        auto views = fixture.views;
        for (auto &v : views) v.name.replace(0, 6, "blk." + std::to_string(id) + ".");
        auto layer = PortableQsaLayer::create(id, views, opts, error);
        check(bool(layer), error);
        return layer;
    };
    auto a = make(3, options);
    const auto primed = owner->memory_usage();
    const auto highwater = owner->graph_bytes();
    auto b = make(7, options);
    check(a->attention_owner() == owner.get() && b->attention_owner() == owner.get(), "shared attention identity");
    check(owner->graph_bytes() == highwater && owner->memory_usage().host_graph == primed.host_graph &&
          owner->memory_usage().device_graph == primed.device_graph, "two borrowers retain one primed allocation");
    auto private_options = options; private_options.attention.reset();
    auto pa = make(3, private_options), pb = make(7, private_options);
    check(!pa->attention_owner() && !pb->attention_owner(), "standalone layers retain private owners");
    const auto ma = a->memory_usage(), mb = b->memory_usage(), mp = pa->memory_usage(), mq = pb->memory_usage();
    const size_t private_buffers = mp.host_graph + mp.device_graph + mq.host_graph + mq.device_graph;
    const size_t shared_buffers = ma.host_graph + ma.device_graph + mb.host_graph + mb.device_graph + primed.host_graph + primed.device_graph;
    check(private_buffers > shared_buffers && private_buffers - shared_buffers == primed.host_graph + primed.device_graph,
          "actual scheduler backing bytes removed, rather than ledger-only savings");
    check(ma.device_state == mp.device_state && mb.device_state == mq.device_state && ma.device_state > 0,
          "both borrowers retain full distinct aligned state");
    auto verify = [&](PortableQsaLayer &layer, PortableQsaLayer &reference) {
        const auto got = layer_chunk_fixture::snapshot(layer, true), expected = layer_chunk_fixture::snapshot(reference, true);
        check(got.tokens == expected.tokens && got.history == expected.history &&
              layer_chunk_fixture::exact(got.last, expected.last), "shared/private byte-identical outputs and all seven state tensors");
    };
    const PortableExpertRows provider = [](int64_t, const PortableRoute &route, std::vector<float> &rows, std::string &e) {
        return gdn_fixture::experts()(0, route, rows, e);
    };
    auto step = [&](PortableQsaLayer &layer, PortableQsaLayer &reference, int seed) {
        PortableRoute got, expected;
        const auto position = layer.tokens();
        const bool ok = layer.step(position, fixture.input(seed), provider, got, error);
        check(ok, error);
        check(reference.step(position, fixture.input(seed), provider, expected, error), error);
        check(got.ids == expected.ids && layer_chunk_fixture::exact(got.weights, expected.weights), "exact shared routing");
        verify(layer, reference);
    };
    // A grows to the whole-model context limit. B repeatedly shrinks to zero,
    // so each consecutive borrow binds a different state and active history.
    for (int i = 0; i < 256; ++i) {
        step(*a, *pa, i);
        if (i % 17 == 0) check(b->reset(error) && pb->reset(error), error);
        step(*b, *pb, i + 31);
    }
    PortableRoute route;
    check(!a->step(256, fixture.input(0), provider, route, error) && a->poisoned(), "context256 boundary refuses and poisons");
    check(a->reset(error) && pa->reset(error), error);
    for (int i = 0; i < 9; ++i) step(*a, *pa, i);
    check(layer_chunk_fixture::snapshot(*a, true).history != layer_chunk_fixture::snapshot(*b, true).history,
          "different KV/indexer state survives alternating resource borrows");
    // Layer-major chunk and T1 tails keep the same borrow through FFN/provider.
    for (size_t width = 1; width <= 8; ++width) {
        check(a->reset(error) && pa->reset(error), error);
        for (size_t position = 0; position < 9;) {
            const size_t n = std::min(width, size_t(9) - position);
            std::vector<float> rows, expected;
            for (size_t t = 0; t < n; ++t) {
                auto row = fixture.input(int(position + t));
                rows.insert(rows.end(), row.begin(), row.end());
            }
            expected = rows;
            std::vector<PortableRoute> routes, controls;
            auto dispatch = [&](int64_t l, const auto &rr, auto &out, auto &e) {
                // Reentrant use must refuse even though attention math completed.
                PortableRoute rejected;
                check(!b->step(b->tokens(), fixture.input(0), provider, rejected, e) &&
                      e.find("already borrowed") != std::string::npos, "borrow spans provider completion and commit");
                for (const auto &r : rr) { std::vector<float> token; check(provider(l,r,token,e),e); out.insert(out.end(),token.begin(),token.end()); }
                return true;
            };
            auto control_dispatch = [&](int64_t l, const auto &rr, auto &out, auto &e) {
                for (const auto &r : rr) { std::vector<float> token; check(provider(l,r,token,e),e); out.insert(out.end(),token.begin(),token.end()); }
                return true;
            };
            check(a->run_chunk(std::vector<int32_t>(n,5),position,rows,dispatch,routes,error),error);
            check(pa->run_chunk(std::vector<int32_t>(n,5),position,expected,control_dispatch,controls,error),error);
            check(layer_chunk_fixture::exact(rows, expected) && routes.size() == controls.size(), "shared/private exact chunk outputs");
            for (size_t t = 0; t < n; ++t) check(routes[t].ids == controls[t].ids &&
                layer_chunk_fixture::exact(routes[t].weights, controls[t].weights), "shared/private exact chunk routes");
            verify(*a,*pa);
            position += n;
        }
    }
    // Failure releases metadata/state views; another borrower remains usable.
    for (int fault = 0; fault < (gpu ? 3 : 6); ++fault) {
        check(a->reset(error) && b->reset(error),error);
        const auto before = layer_chunk_fixture::snapshot(*a, true);
        PortableExpertRows failing = provider;
        auto input = fixture.input(0);
        if (fault == 0) input[0] = NAN;
        if (fault == 1) failing = [](int64_t,const auto&,auto&,auto &e) { e="controlled provider rejection"; return false; };
        if (fault == 2) failing = [](int64_t,const auto&,auto&,auto&) -> bool { throw std::runtime_error("controlled provider throw"); };
        const auto original = options.execution.accelerator->iface.graph_compute;
        if (fault >= 3) {
            layer_chunk_fixture::PostFault::original = original;
            layer_chunk_fixture::PostFault::calls = 0;
            // Projection, attention, or FFN post after the provider completes.
            layer_chunk_fixture::PostFault::fail_at = fault == 3 ? 1 : fault == 4 ? 3 : 6;
            options.execution.accelerator->iface.graph_compute = layer_chunk_fixture::PostFault::compute;
        }
        const bool ok = a->step(0,input,failing,route,error);
        options.execution.accelerator->iface.graph_compute = original;
        check(!ok && a->poisoned() && a->tokens() == 0 && a->last_residual() == before.last && route.ids.empty(),
              "failure retains poison and public commit semantics");
        check(b->step(0,fixture.input(11),provider,route,error),error);
        check(a->reset(error) && a->step(0,fixture.input(0),provider,route,error),error);
    }
    check(a->reset(error), error);
    auto rows = fixture.input(0); auto next_row = fixture.input(1);
    rows.insert(rows.end(),next_row.begin(),next_row.end());
    const auto uncommitted = rows;
    std::vector<PortableRoute> failed_routes;
    check(!a->run_chunk({5,6},0,rows,[](int64_t,const auto&,auto&,auto&) { return false; },failed_routes,error) &&
          a->poisoned() && a->tokens()==0 && layer_chunk_fixture::exact(rows,uncommitted) && failed_routes.empty(),
          "shared chunk provider failure leaves receipts and rows uncommitted");
    check(b->reset(error) && b->step(0,fixture.input(4),provider,route,error),error);
    // Constructor planning failures leave the shared owner safely reusable.
    auto invalid = options; invalid.execution.threads += 1;
    check(!PortableQsaLayer::create(3,fixture.views,invalid,error), "incompatible companion configuration rejected");
    invalid = options; invalid.capacity = 262145;
    check(!PortableQsaLayer::create(3,fixture.views,invalid,error), "state schema failure rejects before scratch binding");
    if (!gpu) {
        // LOG appears in attention masking, after persistent-state views have
        // been bound, but not in projection/pooling/combine/FFN graphs.
        auto *device = ggml_backend_get_device(options.execution.accelerator);
        AttentionPlanFault::original = device->iface.supports_op;
        device->iface.supports_op = AttentionPlanFault::supports;
        auto rejected = PortableQsaLayer::create(3, fixture.views, options, error);
        device->iface.supports_op = AttentionPlanFault::original;
        check(!rejected && error.find("declines") != std::string::npos, "attention planning failure after state-view binding");
        check(b->reset(error) && b->step(0,fixture.input(3),provider,route,error),error);
    }
    {
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> other(ggml_backend_cpu_init(),ggml_backend_free);
        invalid = options; invalid.execution.accelerator = other.get();
        check(!PortableQsaLayer::create(3,fixture.views,invalid,error), "different backend cannot reuse another device's owner");
    }
    auto second_owner = PortableQsaAttention::create(options.execution,error);
    check(bool(second_owner) && second_owner.get() != owner.get() && second_owner->graph_bytes() == 0,
          "separate session has isolated scratch even on same physical backend");
    auto isolated_options = options; isolated_options.attention = second_owner;
    auto isolated = make(3, isolated_options);
    check(isolated->attention_owner() != a->attention_owner(), "separate session identity");
    const auto storage = owner->memory_usage();
    const auto bytes = storage.device_graph + storage.device_extra_bound;
    PortableModelReport report;
    report.device_admission={0}; report.device_scheduler={0}; report.device_workspace_bound={0};
    report.device_token_temporary_bound={0}; report.device_shared_conversion_bound={0};
    report.device_state={0}; report.device_weights={0};
    portable_detail::account_device_memory(report,0,bytes,storage);
    check(report.device_admission[0]==bytes,"shared exact-cap admission");
    bool refused=false;
    try { portable_detail::account_device_memory(report,0,bytes,storage); } catch(const std::exception&) { refused=true; }
    check(refused,"shared cap cannot charge a second owner without capacity");
    report.device_admission[0]=0; refused=false;
    try { portable_detail::account_device_memory(report,0,bytes-1,storage); } catch(const std::exception&) { refused=true; }
    check(refused && report.device_admission[0]==0,"one-byte-over shared admission rejects before ledger commit");
    a.reset(); b.reset();
    check(owner->memory_usage().host_graph==storage.host_graph && owner->memory_usage().device_graph==storage.device_graph,
          "destroying borrowers retains only session scratch, with no dangling state views");
    auto replacement = make(3,options);
    check(replacement->step(0,fixture.input(2),provider,route,error),error);
    std::printf("QSA shared attention: private_scheduler_bytes=%zu shared_scheduler_bytes=%zu removed_backing_bytes=%zu retained_attention_bytes=%zu host=%zu device=%zu cap=%zu metadata_bound=%zu state_per_layer=%zu backend=%s; context256, alternating states, T1-8/tails, failures, replay passed\n",
        private_buffers,shared_buffers,private_buffers-shared_buffers,owner->graph_bytes(),storage.host_graph,storage.device_graph,options.execution.graph_budget,PortableQsaAttention::host_metadata_bound(),ma.device_state,ggml_backend_name(options.execution.accelerator));
}
}
