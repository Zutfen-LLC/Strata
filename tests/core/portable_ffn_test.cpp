#include "portable_ffn_fixture.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "gguf_fixture.hpp"
#include <chrono>
#include <limits>
#include <cstring>
#include <cstdlib>
#ifndef _WIN32
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
// CPU execution with observed submit/join boundaries. This proves coordinator
// ordering and failure draining, not asynchronous hardware execution.
struct ObservedBackend {
    static inline ObservedBackend* active = nullptr;
    ggml_backend_t backend = ggml_backend_cpu_init();
    decltype(backend->iface.graph_compute) compute = backend->iface.graph_compute;
    decltype(backend->iface.synchronize) synchronize = backend->iface.synchronize;
    bool pending = false;
    int joins = 0, matrices = 0, sync_failures = 0;
    bool unknown_sync_failure = false, record_owner_destruction = false;
    std::string fail;
    ObservedBackend() {
        active = this;
        backend->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph* graph) {
            auto& s = *active;
            bool shared = false, route = false;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto* t = ggml_graph_node(graph, i);
                shared |= std::string(ggml_get_name(t)) == "shared";
                route |= std::string(ggml_get_name(t)) == "router_logits";
                if (t->op == GGML_OP_MUL_MAT) ++s.matrices;
            }
            if (shared) {
                check(!s.pending, "no outstanding shared reuse");
                // The shared graph has four matmuls and a leaf with no producer edges.
                int products = 0;
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
                    products += ggml_graph_node(graph,i)->op == GGML_OP_MUL_MAT;
                check(products == 4, "shared graph must not repeat routing/HC producers");
                s.pending = true;
            }
            if (shared && s.fail == "shared-throw") throw std::runtime_error("shared submission exception");
            if ((shared && s.fail == "shared") || (route && s.fail == "route")) return GGML_STATUS_FAILED;
            if (s.fail == "post-sync")
                for(int i=0;i<ggml_graph_n_nodes(graph);++i)
                    if(std::string(ggml_get_name(ggml_graph_node(graph,i)))=="residual_out") s.sync_failures=1;
            return s.compute(b, graph);
        };
        backend->iface.synchronize = [](ggml_backend_t b) {
            auto& s = *active;
            if (s.sync_failures > 0) {
                --s.sync_failures;
                if(s.unknown_sync_failure) throw 17;
                throw std::runtime_error("injected synchronization failure");
            }
            if (s.pending) { s.pending = false; ++s.joins; }
            if (s.synchronize) s.synchronize(b);
        };
    }
    ~ObservedBackend() {
        check(!pending, "destruction must drain shared submission");
        backend->iface.graph_compute = compute;
        backend->iface.synchronize = synchronize;
        if(record_owner_destruction) std::fputs("UNSAFE_BACKEND_OWNER_DESTRUCTION\n",stderr);
        ggml_backend_free(backend); active = nullptr;
    }
};
// Run before any CPU workers are created. The child owns the backend just as
// the CLI owns it outside the model. Unconfirmed FFN destruction must fail-stop
// before that owner's destructor, return-to-caller or atexit cleanup can run.
void fatal_ownership_test() {
#ifndef _WIN32
    int channel[2]; check(pipe(channel)==0,"fatal owner pipe");
    const pid_t child=fork(); check(child>=0,"fatal owner child");
    if(child==0) {
        ::close(channel[0]);
        if(dup2(channel[1],STDERR_FILENO)<0) std::_Exit(98);
        ::close(channel[1]);
        std::atexit([] { std::fputs("UNSAFE_ATEXIT\n",stderr); });
        try {
            Fixture f; ObservedBackend backend; std::string error;
            backend.record_owner_destruction=true;
            PortableFfnOptions o; o.accelerator=backend.backend; o.request_accelerator=true;
            o.overlap_shared_cpu=true; o.resident_dense_weights=true;
            auto terminal=PortableFfn::create(f.g,0,f.views,o,error);
            check(bool(terminal) && terminal->reset(std::vector<float>(128,.2f),error),error);
            PortableRoute route;
            check(!terminal->step([&](int64_t,const PortableRoute&,std::vector<float>&,std::string& e) {
                backend.sync_failures=9; e="terminal provider failure"; return false;
            },route,error) && terminal->poisoned() && backend.pending,"terminal failed-work state");
            terminal.reset(); // must exit before borrowed backend owner's scope ends
            std::fputs("UNSAFE_RETURN_FROM_FFN_DESTRUCTION\n",stderr);
            backend.pending=false; // only let the old unsafe path unwind in this regression
        } catch (...) { std::_Exit(98); }
        std::exit(99); // exercises atexit and owner teardown if fail-stop was omitted
    }
    ::close(channel[1]);
    int status=0; pid_t waited;
    do { waited=waitpid(child,&status,0); } while(waited<0 && errno==EINTR);
    std::string diagnostic; char bytes[512]; ssize_t count;
    while((count=::read(channel[0],bytes,sizeof(bytes)))>0) diagnostic.append(bytes,size_t(count));
    ::close(channel[0]);
    check(waited==child && WIFEXITED(status) && WEXITSTATUS(status)==PortableFfn::fatal_drain_exit_code,
          "unconfirmed drain must fail-stop owned process with explicit nonzero status");
    check(diagnostic.find("completion unconfirmed")!=std::string::npos &&
          diagnostic.find("refusing borrowed backend teardown/reuse")!=std::string::npos &&
          diagnostic.find("UNSAFE_")==std::string::npos,
          "fatal outcome reports incomplete cleanup and prevents backend-owner/atexit teardown");
#else
    std::puts("fatal borrowed-backend process regression not run: POSIX child test");
#endif
}
void scheduling_test() {
    Fixture f; ObservedBackend backend; std::string error;
    PortableFfnOptions o; o.accelerator = backend.backend; o.request_accelerator = true;
    o.overlap_shared_cpu = true; o.resident_dense_weights = true;
    auto runner = PortableFfn::create(f.g,0,f.views,o,error);
    check(bool(runner),error);
    std::vector<float> initial(128, .2f);
    PortableRoute route;
    auto callback = [&](int64_t, const PortableRoute& r, std::vector<float>& rows, std::string& e) {
        check(backend.pending, "shared submitted before CPU callback without join");
        if (backend.fail == "cpu" || backend.fail == "gpu") { e = backend.fail; return false; }
        if (backend.fail == "throw") throw std::runtime_error("callback exception");
        rows.resize(64);
        for (size_t k=0;k<2;++k) for(size_t h=0;h<32;++h)
            rows[k*32+h] = float(r.ids[k]+1) * (.03f + float(h)*.001f);
        return true;
    };
    for (const std::string stage : {"", "route", "shared", "shared-throw", "cpu", "gpu", "throw"}) {
        backend.fail = stage; backend.matrices = 0;
        check(runner->reset(initial,error),error);
        bool dispatched = false;
        const bool ok = runner->step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e) {
            dispatched = true; return callback(l,r,rows,e);
        },route,error);
        check(ok == stage.empty(), "injected stage outcome: " + stage);
        check(!backend.pending, "join before return, including failed submit");
        if (!ok) {
            close(runner->residual(),initial,0);
            check(route.ids.empty(), "no partial route publication");
            if(stage == "shared-throw") check(runner->poisoned(),"submission exception preserves poison");
            if(stage == "route" || stage == "shared" || stage == "shared-throw") check(!dispatched,"failed graph blocks dispatch");
        } else {
            check(backend.matrices == 8,"eight executed matmuls without duplicate producer work");
            PortableRoute ref; close(runner->residual(),f.oracle(initial,ref));
            close(route.activation,ref.activation); close(route.weights,ref.weights);
            check(route.ids==ref.ids,"identical stable routing");
        }
    }
    backend.fail.clear();
    auto serial = PortableFfn::create(f.g,0,f.views,{},error);
    check(bool(serial),error);
    check(serial->reset(initial,error) && runner->reset(initial,error),error);
    for (int token=0; token<3; ++token) {
        PortableRoute serial_route;
        check(serial->step([](int64_t,const PortableRoute& r,std::vector<float>& rows,std::string&) {
            rows.resize(64);
            for(size_t k=0;k<2;++k) for(size_t h=0;h<32;++h)
                rows[k*32+h]=float(r.ids[k]+1)*(.03f+float(h)*.001f);
            return true;
        },serial_route,error),error);
        check(runner->step(callback,route,error),error);
        close(runner->residual(),serial->residual(),0);
        close(route.activation,serial_route.activation,0);
        close(route.weights,serial_route.weights,0);
        check(route.ids==serial_route.ids,"serial/split exact routes");
    }
    check(runner->reset(initial,error) && runner->step(callback,route,error),error);
    // Failed joins preserve the primary error, poison execution even after a
    // later successful drain, and never publish a candidate residual/route.
    for (bool unknown : {false,true}) {
        check(runner->reset(initial,error),error);
        backend.unknown_sync_failure=unknown;
        check(!runner->step([&](int64_t,const PortableRoute&,std::vector<float>&,std::string& e) {
            backend.sync_failures=3; e="primary provider rejection"; return false;
        },route,error),"provider plus cleanup failure");
        check(error==std::string("primary provider rejection; cleanup synchronization failed: ")+
                     (unknown?"unknown backend exception":"injected synchronization failure"),
              "cleanup preserves original provider failure and reports unconfirmed drain");
        check(runner->poisoned() && backend.pending && route.ids.empty(),"pending work remains poisoned");
        close(runner->residual(),initial,0);
        check(!runner->step(callback,route,error) && error.find("poisoned")!=std::string::npos,
              "poison blocks execution even if cleanup subsequently drains");
        check(!backend.pending && runner->poisoned(),"successful cleanup does not erase failed-work status");
        backend.sync_failures=1;
        check(!runner->reset(initial,error) && runner->poisoned() &&
              error.find("reset could not drain")!=std::string::npos,"failed reset cannot clear poison");
        check(runner->reset(initial,error) && !runner->poisoned(),"confirmed reset recovers poisoned execution");
        backend.unknown_sync_failure=false;
    }
    for(const std::string point : {"route-sync","shared-sync","post-sync"}) {
        check(runner->reset(initial,error),error);
        backend.fail = point;
        if(point=="route-sync") backend.sync_failures=1;
        check(!runner->step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e) {
            if(point=="shared-sync") backend.sync_failures=1;
            return callback(l,r,rows,e);
        },route,error),"graph synchronization failure");
        check(error=="injected synchronization failure" && runner->poisoned() && route.ids.empty(),
              "route/shared/post sync failures stay failed after cleanup recovery");
        close(runner->residual(),initial,0);
        check(!backend.pending,"transient graph sync failure drained before return");
        backend.fail.clear();
    }
    check(runner->reset(initial,error) && runner->step(callback,route,error),error);
    backend.sync_failures=1; // destructor retries a transient wait failure without throwing
    runner.reset(); check(!backend.pending,"cleanup after recovery");

}
}
int main() {
    try {
        fatal_ownership_test();
        scheduling_test();
        for (bool overlap : {false, true}) {
        Fixture f;
        std::string error;
        PortableFfnOptions options;
        options.overlap_shared_cpu = overlap;
        auto runner = PortableFfn::create(f.g, 0, f.views, options, error);
        check(bool(runner), error);
        size_t matrices = 0;
        for (const auto& p : runner->placements()) if (p.operation == "MUL_MAT") ++matrices;
        check(matrices == 8, "exactly eight original pre-FFN matrix products");
        PortableRoute route, expected_route;
        const PortableExpertRows experts = [&](int64_t layer, const PortableRoute& r, std::vector<float>& rows, std::string&) {
            check(layer == 0 && r.ids.size() == 2 && r.activation.size() == 32, "callback boundary");
            rows.resize(64);
            for (size_t k = 0; k < 2; ++k)
                for (size_t h = 0; h < 32; ++h) rows[k*32+h] = float(r.ids[k]+1) * (0.03f + float(h)*0.001f);
            return true;
        };
        check(!runner->step(experts, route, error), "reset required");
        std::vector<float> initial(128);
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0.2f + float(int(i % 19)-9) * 0.06f;
        check(runner->reset(initial, error), error);
        auto expected = f.oracle(initial, expected_route);
        check(runner->step(experts, route, error), error);
        close(runner->residual(), expected);
        close(route.activation, expected_route.activation);
        close(route.weights, expected_route.weights);
        check(route.ids == expected_route.ids, "routing IDs");
        expected = f.oracle(expected, expected_route);
        check(runner->step(experts, route, error), error);
        close(runner->residual(), expected);
        check(runner->reset(initial, error) && runner->step(experts, route, error), error);
        close(runner->residual(), f.oracle(initial, expected_route));
        const auto saved = runner->residual();
        check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>&, std::string& e) { e="dispatch rejected"; return false; }, route, error), "dispatch rejection");
        close(runner->residual(), saved, 0);
        check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>& rows, std::string&) { rows={1}; return true; }, route, error), "row count rejection");
        close(runner->residual(), saved, 0);
        check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>& rows, std::string&) { rows.assign(64, std::numeric_limits<float>::quiet_NaN()); return true; }, route, error), "nonfinite rejection");
        close(runner->residual(), saved, 0);
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
            for (size_t at : {size_t(0), initial.size() / 2, initial.size() - 1}) {
                auto bad = initial;
                bad[at] = invalid;
                check(!runner->reset(bad, error) && error == "portable FFN: nonfinite activation/result",
                      "FFN residual finite diagnostic");
                close(runner->residual(), saved, 0);
            }
            for (size_t at : {size_t(0), size_t(32), size_t(63)}) {
                check(!runner->step([&](int64_t, const PortableRoute&, std::vector<float>& rows, std::string&) {
                    rows.assign(64, 1);
                    rows[at] = invalid;
                    return true;
                }, route, error) && error == "portable FFN: nonfinite activation/result" && route.ids.empty(),
                      "FFN row finite diagnostic/route rollback");
                close(runner->residual(), saved, 0);
            }
        }
        {
            auto bad_views = f.views;
            auto bad_norm = f.data[0];
            bad_norm.back() = std::numeric_limits<float>::quiet_NaN();
            bad_views[0].data = bad_norm.data();
            auto bad = PortableFfn::create(f.g, 0, bad_views, {}, error);
            check(bool(bad) && bad->reset(initial, error), error);
            bool dispatched = false;
            check(!bad->step([&](int64_t, const PortableRoute&, std::vector<float>&, std::string&) {
                dispatched = true;
                return true;
            }, route, error) && error == "portable FFN: nonfinite activation/result" && !dispatched,
                  "FFN read-side finite diagnostic");
            close(bad->residual(), initial, 0);
        }
        auto finite_edges = initial;
        finite_edges[0] = 0.0f;
        finite_edges[1] = -0.0f;
        finite_edges[finite_edges.size() / 2] = std::numeric_limits<float>::min();
        finite_edges.back() = std::numeric_limits<float>::denorm_min();
        check(runner->reset(finite_edges, error) && runner->step(experts, route, error), error);
        const auto edge_residual = runner->residual();
        const auto edge_route = route;
        check(runner->reset(finite_edges, error) && runner->step(experts, route, error), error);
        const auto bytes_equal = [](const std::vector<float>& a, const std::vector<float>& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        check(bytes_equal(edge_residual, runner->residual()) && route.ids == edge_route.ids &&
              bytes_equal(route.activation, edge_route.activation) && bytes_equal(route.weights, edge_route.weights),
              "FFN finite edge replay bytes");
        check(runner->reset(saved, error), error);
        check(runner->step(experts, route, error), error); // recovery after rejected callbacks
        check(runner->weight_bytes() > 0 && runner->graph_bytes() > 0, "allocation accounting");

        options.request_accelerator = true;
        check(!PortableFfn::create(f.g, 0, f.views, options, error), "strict unavailable device");
        options.allow_cpu_fallback = true;
        auto fallback = PortableFfn::create(f.g, 0, f.views, options, error);
        check(bool(fallback), error);
        size_t rejected = 0;
        for (const auto& p : fallback->placements()) if (!p.fallback_reason.empty()) ++rejected;
        check(rejected > 0, "fallback reporting");
        check(fallback->reset(initial, error) && fallback->step(experts, route, error), error);
        close(fallback->residual(), f.oracle(initial, expected_route));
        {
            // A real CPU execution backend with a private device descriptor that
            // declines graph ops. Exercises supports_op rejection, not GPU math.
            struct DecliningBackend {
                ggml_backend_t backend = ggml_backend_cpu_init();
                ggml_backend_device device = *backend->device;
                ggml_backend_dev_t original = backend->device;
                DecliningBackend() {
                    device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor*) { return false; };
                    backend->device = &device;
                    backend->iface.get_name = [](ggml_backend_t) { return "TEST_DECLINING_CPU"; };
                }
                ~DecliningBackend() { backend->device = original; ggml_backend_free(backend); }
            } declining;
            options.accelerator = declining.backend;
            options.allow_cpu_fallback = false;
            check(!PortableFfn::create(f.g, 0, f.views, options, error) && error.find("declines") != std::string::npos,
                  "strict operation rejection");
            options.allow_cpu_fallback = true;
            auto operation_fallback = PortableFfn::create(f.g, 0, f.views, options, error);
            check(bool(operation_fallback), error);
            bool reported = false;
            for (const auto& p : operation_fallback->placements())
                if (p.fallback_reason.find("declines") != std::string::npos) reported = true;
            check(reported, "declined operation reporting");
            check(operation_fallback->reset(initial, error) && operation_fallback->step(experts, route, error), error);
            close(operation_fallback->residual(), f.oracle(initial, expected_route));
        }
        options = {}; options.graph_budget = 1;
        check(!PortableFfn::create(f.g, 0, f.views, options, error), "graph budget");
        options = {}; options.weight_budget = 1;
        check(!PortableFfn::create(f.g, 0, f.views, options, error), "weight budget");
        auto malformed = f.views; malformed[1].shape[0] = 127;
        check(!PortableFfn::create(f.g, 0, malformed, {}, error), "bad tensor shape");
        // Exercise the actual fleet projection encoding without requiring model files.
        std::vector<std::vector<uint8_t>> q8(9);
        auto quantized = f.views;
        for (size_t i : {size_t(1),size_t(2),size_t(3),size_t(6),size_t(7),size_t(8)}) {
            q8[i].resize(f.data[i].size()/32*34);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data[i].data(), q8[i].data(), 0,
                                int64_t(f.data[i].size()/size_t(f.views[i].shape[0])), f.views[i].shape[0], nullptr);
            quantized[i].type = 8; quantized[i].bytes = q8[i].size(); quantized[i].data = q8[i].data();
        }
        auto qrunner = PortableFfn::create(f.g, 0, quantized, {}, error);
        check(bool(qrunner), error);
        check(qrunner->reset(initial, error) && qrunner->step(experts, route, error), error);
        close(qrunner->residual(), f.oracle(initial, expected_route), 0.003f);
        // Real split-reader binding: metadata-only first shard, tensors in the
        // second. The payload is synthetic but bytes/layout match the inputs.
        const auto root = std::filesystem::temp_directory_path() /
            ("strata-portable-ffn-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture directory");
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code e; std::filesystem::remove_all(p,e); } } cleanup{root};
        auto kv = fixture::split_keys(0, 2, 9);
        kv.push_back(fixture::str("general.architecture", "qwen4exp"));
        for (auto pair : {std::pair<const char*, uint64_t>{"block_count",48}, {"embedding_length",32},
                          {"expert_count",4}, {"expert_used_count",2}, {"attention.head_count",24},
                          {"attention.head_count_kv",2}, {"hyper_connection.count",4},
                          {"hyper_connection.low_rank",32}, {"expert_feed_forward_length",32},
                          {"expert_shared_feed_forward_length",32}})
            kv.push_back(fixture::u32("qwen4exp." + std::string(pair.first), pair.second));
        const auto first = root / "fixture-00001-of-00002.gguf";
        const auto second = root / "fixture-00002-of-00002.gguf";
        fixture::write(first, kv, {});
        std::vector<fixture::Tensor> tensors;
        for (const auto& v : quantized) {
            fixture::Tensor t; t.name = v.name; t.type = v.type;
            for (auto d : v.shape) t.shape.push_back(uint64_t(d));
            tensors.push_back(t);
        }
        const auto written = fixture::write(second, fixture::split_keys(1,2,9), tensors);
        {
            std::fstream file(second, std::ios::in | std::ios::out | std::ios::binary);
            for (size_t i = 0; i < quantized.size(); ++i) {
                file.seekp(std::streamoff(written.data_start + written.offsets[i]));
                file.write(static_cast<const char*>(quantized[i].data), std::streamsize(quantized[i].bytes));
            }
            check(bool(file), "fixture payload write");
        }
        std::unique_ptr<PortableFfn> bound;
        {
            auto model = strata::GgufModel::open(first.string());
            bound = PortableFfn::from_gguf(model, 0, {}, error);
            check(bool(bound), error);
            check(!PortableFfn::from_gguf(model, 48, {}, error), "GGUF layer range");
        } // mapping lifetime ends: copied weights must remain usable
        check(bound->reset(initial, error) && bound->step(experts, route, error), error);
        close(bound->residual(), qrunner->residual(), 0);
        std::puts("portable_ffn_test: scalar parity, state/reset, callback failures, placement/fallback, budgets, Q8_0, split GGUF passed");
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
