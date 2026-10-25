// gvs_bench: runs the same ticks and views through every engine (two CPU engines and the GPU engine),
// checks that all produce identical results, and reports engine time per tick for each view count.
// The CPU figure that counts is the best of the two CPU engines.
#include "engine.h"
#include "loadviews.h"
#include "machine.h"
#include "sim.h"
#include "thread_pool.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

using namespace gvs;

namespace {

struct Args {
    SimConfig sim;
    std::vector<uint32_t> views = {8, 32, 128};
    uint32_t ticks = 10;
    uint32_t warmup = 2;
    std::string backends = "all";
    bool validate = true;
    std::string json;
    std::string pin;
    bool randomViewport = false;
    CpuConfig cpu;
    GpuConfig gpu;
};

void usage() {
    std::puts(
        "gvs_bench [options]\n"
        "  --rows N            orders (4000000, max 33554432)\n"
        "  --instruments N     instruments (5000)\n"
        "  --views A,B,C       view counts to measure (8,32,128)\n"
        "  --ticks N           measured ticks per view count (10)\n"
        "  --warmup N          unmeasured ticks first (2)\n"
        "  --backend LIST      engines: cpu, cpu-simple, gpu, comma separated, or all (all)\n"
        "  --threads N         CPU engine threads (one per pinned CPU, or all hardware threads)\n"
        "  --pin LIST          pin CPU threads to these CPUs, e.g. 0-23 or all (Linux; needed for isolcpus)\n"
        "  --gpus N            max GPUs (all)\n"
        "  --slots N           GPU views per batch (8)\n"
        "  --groups N          GPU workgroups per view for full-table passes (64)\n"
        "  --viewport MODE     top: views mostly look at the first rows (default);\n"
        "                      random: every tick each view jumps to a random position in its result set\n"
        "  --no-validate       skip the result comparison between engines\n"
        "  --json FILE         also write results and machine details as JSON\n"
        "  --seed N            simulation seed (42)");
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--rows") a.sim.rows = uint32_t(std::atoll(val()));
        else if (k == "--instruments") a.sim.instruments = uint32_t(std::atoi(val()));
        else if (k == "--views") {
            a.views.clear();
            std::stringstream ss(val());
            for (std::string t; std::getline(ss, t, ',');)
                if (!t.empty()) a.views.push_back(uint32_t(std::atoi(t.c_str())));
        } else if (k == "--ticks") a.ticks = uint32_t(std::atoi(val()));
        else if (k == "--warmup") a.warmup = uint32_t(std::atoi(val()));
        else if (k == "--backend") a.backends = val();
        else if (k == "--threads") a.cpu.threads = unsigned(std::atoi(val()));
        else if (k == "--pin") a.pin = val();
        else if (k == "--gpus") a.gpu.maxDevices = uint32_t(std::atoi(val()));
        else if (k == "--slots") a.gpu.slots = uint32_t(std::atoi(val()));
        else if (k == "--groups") a.gpu.groupsPerSlot = uint32_t(std::atoi(val()));
        else if (k == "--viewport") a.randomViewport = std::string(val()) == "random";
        else if (k == "--no-validate") a.validate = false;
        else if (k == "--json") a.json = val();
        else if (k == "--seed") a.sim.seed = uint64_t(std::atoll(val()));
        else {
            usage();
            return false;
        }
    }
    if (!a.pin.empty()) {
        a.cpu.cpus = parseCpuList(a.pin);
        if (a.cpu.cpus.empty()) {
            std::printf("bad --pin list: %s\n", a.pin.c_str());
            return false;
        }
    }
    if (a.sim.rows > MAX_ROWS) {
        std::printf("--rows %u exceeds the maximum of %u\n", a.sim.rows, MAX_ROWS);
        return false;
    }
    return true;
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))];
}

struct Series {
    std::vector<double> total, upload, derive, views;
};

struct Row {
    uint32_t views = 0;
    std::vector<Series> eng;  // per engine
    int mismatches = 0;       // vs the first engine
    std::string firstDiff;
};

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o.push_back('\\');
        o.push_back(c);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parseArgs(argc, argv, a)) return 1;

    std::vector<std::string> wanted;
    if (a.backends == "all" || a.backends == "both") wanted = {"cpu-simple", "cpu", "gpu"};
    else {
        std::stringstream ss(a.backends);
        for (std::string t; std::getline(ss, t, ',');) wanted.push_back(t);
    }

    std::printf("generating %u orders, %u instruments...\n", a.sim.rows, a.sim.instruments);
    Store store;
    Simulator sim(store, a.sim);

    std::vector<std::unique_ptr<Engine>> engines;
    std::vector<std::string> gpuNames;
    for (const std::string& w : wanted) {
        if (w == "cpu" || w == "cpu-simple") {
            CpuConfig c = a.cpu;
            c.simple = w == "cpu-simple";
            engines.push_back(makeCpuEngine(c));
        } else if (w == "gpu") {
            std::string err;
            auto g = makeGpuEngine(a.gpu, err);
            if (!g) {
                std::printf("GPU backend unavailable: %s\n", err.c_str());
                continue;
            }
            gpuNames = g->devices();
            engines.push_back(std::move(g));
        } else {
            std::printf("unknown backend %s\n", w.c_str());
            return 1;
        }
    }
    if (engines.empty()) return 1;
    const size_t E = engines.size();
    int gpuIdx = -1;
    for (size_t e = 0; e < E; ++e)
        if (engines[e]->name() == "gpu") gpuIdx = int(e);

    std::string cpuDesc;
    for (auto& e : engines)
        if (e->name() != "gpu") cpuDesc = e->devices()[0];
    std::printf("machine: %s, %u logical CPUs, host %s\n", cpuModel().c_str(), logicalCpus(), hostName().c_str());
    if (!cpuDesc.empty()) std::printf("cpu engines: %s\n", cpuDesc.c_str());
    for (const auto& g : gpuNames) std::printf("gpu: %s\n", g.c_str());
    std::printf("build: %s\n", gitVersion());

    std::vector<Row> rows;
    std::vector<Update> updates;
    std::vector<std::vector<ViewResult>> res(E);
    uint64_t tick = 0;
    std::mt19937_64 rng(a.sim.seed + 7);

    for (uint32_t nv : a.views) {
        std::vector<ViewParams> params;
        while (params.size() < nv) {
            ViewParams p;
            std::string err;
            if (compileView(randomView(rng, store), store, p, err)) params.push_back(p);
        }
        Row row;
        row.views = nv;
        row.eng.resize(E);
        for (uint32_t t = 0; t < a.warmup + a.ticks; ++t) {
            updates.clear();
            uint32_t md = 0;
            sim.step(300, updates, md);
            ++tick;
            for (size_t e = 0; e < E; ++e) {
                RunStats st;
                engines[e]->run(store, tick, updates, params, res[e], st);
                if (t < a.warmup) continue;
                row.eng[e].total.push_back(st.totalMs);
                row.eng[e].upload.push_back(st.uploadMs);
                row.eng[e].derive.push_back(st.deriveMs);
                row.eng[e].views.push_back(st.viewsMs);
            }
            if (t >= a.warmup && a.validate) {
                for (size_t e = 1; e < E; ++e) {
                    std::string first;
                    const int bad = compareResults(res[0], res[e], &first);
                    if (bad && row.firstDiff.empty()) row.firstDiff = engines[e]->name() + " " + first;
                    row.mismatches += bad;
                }
            }
            // Next tick, each view looks at a random window anywhere in its (last known) result set.
            if (a.randomViewport)
                for (size_t i = 0; i < params.size(); ++i) {
                    const uint32_t total = res[0][i].total, lim = params[i].limit;
                    params[i].offset = total > lim ? uint32_t(rng() % (total - lim + 1)) : 0;
                }
        }
        std::printf("views %u done\n", nv);
        rows.push_back(std::move(row));
    }

    // Best CPU engine per view count.
    auto bestCpu = [&](const Row& r) {
        int best = -1;
        for (size_t e = 0; e < E; ++e)
            if (int(e) != gpuIdx && (best < 0 || pct(r.eng[e].total, 0.5) < pct(r.eng[best].total, 0.5))) best = int(e);
        return best;
    };

    std::printf("\nrows %u, ticks %u per view count; engine ms per tick (p50 / p95)\n\n", store.rows, a.ticks);
    std::printf("%7s", "views");
    for (auto& e : engines) std::printf(" | %17s", e->name().c_str());
    std::printf(" | %8s | %s\n", "speedup", "results");
    for (const Row& r : rows) {
        std::printf("%7u", r.views);
        for (size_t e = 0; e < E; ++e)
            std::printf(" | %7.1f / %7.1f", pct(r.eng[e].total, 0.5), pct(r.eng[e].total, 0.95));
        const int b = bestCpu(r);
        if (b >= 0 && gpuIdx >= 0)
            std::printf(" | %7.1fx", pct(r.eng[b].total, 0.5) / pct(r.eng[gpuIdx].total, 0.5));
        else
            std::printf(" | %8s", "-");
        std::printf(" | %s\n", !a.validate || E < 2 ? "-" : r.mismatches ? "MISMATCH" : "identical");
        if (r.mismatches) std::printf("        first difference: %s\n", r.firstDiff.c_str());
    }
    std::printf("\nspeedup = best CPU engine p50 / gpu p50\n");
    if (!rows.empty()) {
        const Row& r = rows.back();
        for (size_t e = 0; e < E; ++e)
            std::printf("last run, %s p50: upload %.1f ms, derive %.1f ms, views %.1f ms\n",
                        engines[e]->name().c_str(), pct(r.eng[e].upload, 0.5), pct(r.eng[e].derive, 0.5),
                        pct(r.eng[e].views, 0.5));
    }

    if (!a.json.empty()) {
        std::ofstream f(a.json);
        f << "{\n  \"date\": \"" << utcNow() << "\",\n  \"build\": \"" << gitVersion() << "\",\n"
          << "  \"host\": \"" << jsonEscape(hostName()) << "\",\n"
          << "  \"cpu_model\": \"" << jsonEscape(cpuModel()) << "\",\n"
          << "  \"logical_cpus\": " << logicalCpus() << ",\n"
          << "  \"cpu\": \"" << jsonEscape(cpuDesc) << "\",\n"
          << "  \"pin\": \"" << jsonEscape(a.pin) << "\",\n  \"gpus\": [";
        for (size_t i = 0; i < gpuNames.size(); ++i) f << (i ? ", " : "") << '"' << jsonEscape(gpuNames[i]) << '"';
        f << "],\n  \"gpu_slots\": " << a.gpu.slots << ",\n  \"gpu_groups\": " << a.gpu.groupsPerSlot
          << ",\n  \"rows\": " << store.rows << ",\n  \"instruments\": " << store.instruments
          << ",\n  \"viewport\": \"" << (a.randomViewport ? "random" : "top") << "\""
          << ",\n  \"ticks\": " << a.ticks << ",\n  \"validated\": " << (a.validate && E > 1 ? "true" : "false")
          << ",\n  \"results\": [\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i];
            const int b = bestCpu(r);
            f << "    {\"views\": " << r.views;
            if (b >= 0)
                f << ", \"cpu_engine\": \"" << engines[b]->name() << "\", \"cpu_p50\": " << pct(r.eng[b].total, 0.5)
                  << ", \"cpu_p95\": " << pct(r.eng[b].total, 0.95);
            if (gpuIdx >= 0)
                f << ", \"gpu_p50\": " << pct(r.eng[gpuIdx].total, 0.5)
                  << ", \"gpu_p95\": " << pct(r.eng[gpuIdx].total, 0.95);
            for (size_t e = 0; e < E; ++e)
                f << ", \"" << engines[e]->name() << "\": {\"p50\": " << pct(r.eng[e].total, 0.5)
                  << ", \"p95\": " << pct(r.eng[e].total, 0.95) << "}";
            f << ", \"mismatches\": " << r.mismatches << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
        std::printf("\nwrote %s\n", a.json.c_str());
    }
    return 0;
}
