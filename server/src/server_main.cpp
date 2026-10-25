// gvs_server: simulated order book + view engine (CPU or GPU) behind a WebSocket/protobuf API.
#include "engine.h"
#include "loadviews.h"
#include "net.h"
#include "protocol.h"
#include "sim.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>

using namespace gvs;

namespace {

volatile std::sig_atomic_t gStop = 0;

struct Args {
    uint16_t port = 8080;
    SimConfig sim;
    uint32_t tickMs = 300;
    std::string backend = "gpu";
    uint32_t loadViews = 0;
    unsigned threads = 0;
    std::string www;
    bool validate = false;
    GpuConfig gpu;
};

void usage() {
    std::puts(
        "gvs_server [options]\n"
        "  --port N            WebSocket/HTTP port (8080)\n"
        "  --rows N            orders in the table (4000000, max 16777216)\n"
        "  --instruments N     instruments (5000)\n"
        "  --tick-ms N         update/push interval (300)\n"
        "  --fills N           fill attempts per tick (20000)\n"
        "  --new N             recycled orders per tick (5000)\n"
        "  --md-share X        share of instruments moving per tick (0.3)\n"
        "  --backend cpu|gpu   starting backend (gpu, falls back to cpu)\n"
        "  --load-views N      synthetic views evaluated every tick on top of client views (0)\n"
        "  --threads N         CPU engine threads (all cores)\n"
        "  --gpus N            max GPUs to use (all)\n"
        "  --slots N           GPU views per batch (8)\n"
        "  --validate          also run the other backend every tick and compare results\n"
        "  --www DIR           serve the built client from DIR (client/dist)\n"
        "  --seed N            simulation seed (42)");
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--port") a.port = uint16_t(std::atoi(val()));
        else if (k == "--rows") a.sim.rows = uint32_t(std::atoll(val()));
        else if (k == "--instruments") a.sim.instruments = uint32_t(std::atoi(val()));
        else if (k == "--tick-ms") a.tickMs = uint32_t(std::atoi(val()));
        else if (k == "--fills") a.sim.fillsPerTick = uint32_t(std::atoi(val()));
        else if (k == "--new") a.sim.newPerTick = uint32_t(std::atoi(val()));
        else if (k == "--md-share") a.sim.mdMoveShare = float(std::atof(val()));
        else if (k == "--backend") a.backend = val();
        else if (k == "--load-views") a.loadViews = uint32_t(std::atoi(val()));
        else if (k == "--threads") a.threads = unsigned(std::atoi(val()));
        else if (k == "--gpus") a.gpu.maxDevices = uint32_t(std::atoi(val()));
        else if (k == "--slots") a.gpu.slots = uint32_t(std::atoi(val()));
        else if (k == "--validate") a.validate = true;
        else if (k == "--www") a.www = val();
        else if (k == "--seed") a.sim.seed = uint64_t(std::atoll(val()));
        else {
            usage();
            return false;
        }
    }
    return true;
}

class App final : public WsServer::Handler {
public:
    App(const Args& a) : a_(a), sim_(store_, a.sim), rng_(a.sim.seed + 1) {
        cpu_ = makeCpuEngine(a.threads);
        std::string err;
        gpu_ = makeGpuEngine(a.gpu, err);
        if (!gpu_) std::printf("GPU backend unavailable: %s\n", err.c_str());
        active_ = (a.backend == "gpu" && gpu_) ? gpu_.get() : cpu_.get();
        for (uint32_t i = 0; i < a.loadViews; ++i) {
            ViewParams p;
            std::string e;
            if (compileView(randomView(rng_, store_), store_, p, e)) loadParams_.push_back(p);
        }
        std::printf("rows %u, instruments %u, cpu %s, gpu %s, backend %s, load views %zu\n", store_.rows,
                    store_.instruments, cpu_->devices()[0].c_str(),
                    gpu_ ? join(gpu_->devices()).c_str() : "none", active_->name().c_str(), loadParams_.size());
    }

    void onOpen(uint64_t id) override {
        clients_[id];
        SchemaInfo si;
        si.backends.push_back(BACKEND_CPU);
        if (gpu_) {
            si.backends.push_back(BACKEND_GPU);
            si.gpuNames = gpu_->devices();
        }
        si.backend = active_ == gpu_.get() ? BACKEND_GPU : BACKEND_CPU;
        si.tickMs = a_.tickMs;
        si.cpuThreads = uint32_t(std::atoi(cpu_->devices()[0].c_str()));
        si.loadViews = uint32_t(loadParams_.size());
        net_->send(id, encodeSchema(store_, si));
        std::printf("client %llu connected\n", (unsigned long long)id);
    }

    void onMessage(uint64_t id, const std::string& data) override {
        ClientCommand cmd;
        if (!decodeClientMsg(data, cmd)) {
            net_->send(id, encodeError("cannot decode message"));
            return;
        }
        Client& c = clients_[id];
        if (cmd.type == ClientCommand::SUBSCRIBE) {
            ClientView v;
            std::string err;
            if (!compileView(cmd.spec, store_, v.params, err)) {
                net_->send(id, encodeError("view " + std::to_string(cmd.viewId) + ": " + err));
                return;
            }
            c.views[cmd.viewId] = v;
        } else if (cmd.type == ClientCommand::UNSUBSCRIBE) {
            c.views.erase(cmd.viewId);
        } else if (cmd.type == ClientCommand::SET_BACKEND) {
            if (cmd.backend == BACKEND_GPU && !gpu_) {
                net_->send(id, encodeError("GPU backend unavailable"));
                return;
            }
            active_ = cmd.backend == BACKEND_GPU ? gpu_.get() : cpu_.get();
            std::printf("backend -> %s\n", active_->name().c_str());
        }
    }

    void onClose(uint64_t id) override {
        clients_.erase(id);
        std::printf("client %llu disconnected\n", (unsigned long long)id);
    }

    void tick() {
        ++tick_;
        updates_.clear();
        uint32_t md = 0;
        sim_.step(a_.tickMs, updates_, md);

        // Synthetic users scroll: some load views move their viewport every tick.
        for (ViewParams& p : loadParams_)
            if (rng_() % 10 == 0) p.offset = (rng_() % 10 < 7) ? 0 : uint32_t(rng_() % 20000);

        params_ = loadParams_;
        refs_.clear();
        for (auto& [cid, c] : clients_)
            for (auto& [vid, v] : c.views) {
                params_.push_back(v.params);
                refs_.push_back({cid, vid});
            }

        StatsInfo st;
        st.tick = tick_;
        st.backend = active_ == gpu_.get() ? BACKEND_GPU : BACKEND_CPU;
        active_->run(store_, tick_, updates_, params_, results_, st.run);
        if (a_.validate && gpu_) {
            Engine* other = active_ == gpu_.get() ? cpu_.get() : gpu_.get();
            RunStats ignore;
            other->run(store_, tick_, updates_, params_, check_, ignore);
            std::string first;
            st.validation = compareResults(results_, check_, &first);
            if (st.validation) std::printf("tick %llu: %d views differ, %s\n", (unsigned long long)tick_, st.validation, first.c_str());
        }

        const size_t base = loadParams_.size();
        for (size_t i = 0; i < refs_.size(); ++i) {
            const auto [cid, vid] = refs_[i];
            if (net_->queued(cid) > (32u << 20)) continue;  // slow client: skip this tick
            net_->send(cid, encodeViewUpdate(store_, vid, tick_, params_[base + i], results_[base + i]));
        }

        st.views = uint32_t(params_.size());
        st.clientViews = uint32_t(refs_.size());
        st.rows = store_.rows;
        st.updates = uint32_t(updates_.size());
        st.mdChanges = md;
        st.tickMs = a_.tickMs;
        const std::string stats = encodeStats(st);
        for (auto& [cid, c] : clients_) net_->send(cid, stats);

        if (tick_ % 10 == 0)
            std::printf("tick %llu  %s  engine %.1f ms (upload %.1f, derive %.1f, views %.1f)  views %u  updates %u%s\n",
                        (unsigned long long)tick_, active_->name().c_str(), st.run.totalMs, st.run.uploadMs,
                        st.run.deriveMs, st.run.viewsMs, st.views, st.updates,
                        st.validation == 0 ? "  validated" : "");
    }

    void setNet(WsServer* n) { net_ = n; }

private:
    struct ClientView {
        ViewParams params;
    };
    struct Client {
        std::map<uint32_t, ClientView> views;
    };

    static std::string join(const std::vector<std::string>& v) {
        std::string s;
        for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
        return s;
    }

    Args a_;
    Store store_;
    Simulator sim_;
    std::mt19937_64 rng_;
    std::unique_ptr<Engine> cpu_, gpu_;
    Engine* active_ = nullptr;
    WsServer* net_ = nullptr;
    std::map<uint64_t, Client> clients_;
    std::vector<ViewParams> loadParams_, params_;
    std::vector<std::pair<uint64_t, uint32_t>> refs_;
    std::vector<ViewResult> results_, check_;
    std::vector<Update> updates_;
    uint64_t tick_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parseArgs(argc, argv, a)) return 1;
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, [](int) { gStop = 1; });
    std::signal(SIGTERM, [](int) { gStop = 1; });

    std::printf("generating %u orders...\n", a.sim.rows);
    App app(a);
    WsServer net;
    std::string err;
    if (!net.listen(a.port, err)) {
        std::printf("listen on %u failed: %s\n", a.port, err.c_str());
        return 1;
    }
    net.setWwwRoot(a.www);
    app.setNet(&net);
    std::printf("listening on ws://0.0.0.0:%u/ws%s\n", a.port, a.www.empty() ? "" : (" and serving " + a.www).c_str());

    using Clock = std::chrono::steady_clock;
    auto next = Clock::now() + std::chrono::milliseconds(a.tickMs);
    while (!gStop) {
        const auto now = Clock::now();
        if (now >= next) {
            app.tick();
            next += std::chrono::milliseconds(a.tickMs);
            if (next < Clock::now()) next = Clock::now() + std::chrono::milliseconds(a.tickMs);  // overran the tick
            continue;
        }
        const int wait = int(std::chrono::duration_cast<std::chrono::milliseconds>(next - now).count());
        net.poll(std::max(wait, 0), app);
    }
    std::printf("bye\n");
    return 0;
}
