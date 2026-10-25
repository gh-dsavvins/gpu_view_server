// gvs_bench: runs the same ticks and views through the CPU and GPU engines, checks that both produce
// identical results, and reports engine time per tick for each view count.
#include "engine.h"
#include "loadviews.h"
#include "sim.h"

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
    unsigned threads = 0;
    std::string backend = "both";
    bool validate = true;
    std::string json;
    GpuConfig gpu;
};

void usage() {
    std::puts(
        "gvs_bench [options]\n"
        "  --rows N            orders (4000000)\n"
        "  --instruments N     instruments (5000)\n"
        "  --views A,B,C       view counts to measure (8,32,128)\n"
        "  --ticks N           measured ticks per view count (10)\n"
        "  --warmup N          unmeasured ticks first (2)\n"
        "  --backend cpu|gpu|both (both)\n"
        "  --threads N         CPU engine threads (all cores)\n"
        "  --gpus N            max GPUs (all)\n"
        "  --slots N           GPU views per batch (8)\n"
        "  --no-validate       skip the CPU vs GPU result comparison\n"
        "  --json FILE         also write results as JSON\n"
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
        else if (k == "--backend") a.backend = val();
        else if (k == "--threads") a.threads = unsigned(std::atoi(val()));
        else if (k == "--gpus") a.gpu.maxDevices = uint32_t(std::atoi(val()));
        else if (k == "--slots") a.gpu.slots = uint32_t(std::atoi(val()));
        else if (k == "--no-validate") a.validate = false;
        else if (k == "--json") a.json = val();
        else if (k == "--seed") a.sim.seed = uint64_t(std::atoll(val()));
        else {
            usage();
            return false;
        }
    }
    return true;
}

struct Series {
    std::vector<double> total, derive, views, upload;
};

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))];
}

struct Row {
    uint32_t views;
    Series cpu, gpu;
    int mismatches = 0;
    std::string firstDiff;
};

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parseArgs(argc, argv, a)) return 1;

    std::printf("generating %u orders, %u instruments...\n", a.sim.rows, a.sim.instruments);
    Store store;
    Simulator sim(store, a.sim);

    std::unique_ptr<Engine> cpu, gpu;
    if (a.backend != "gpu") cpu = makeCpuEngine(a.threads);
    if (a.backend != "cpu") {
        std::string err;
        gpu = makeGpuEngine(a.gpu, err);
        if (!gpu) std::printf("GPU backend unavailable: %s\n", err.c_str());
    }
    if (!cpu && !gpu) return 1;
    std::printf("cpu: %s\n", cpu ? cpu->devices()[0].c_str() : "off");
    if (gpu)
        for (const auto& d : gpu->devices()) std::printf("gpu: %s\n", d.c_str());

    std::vector<Row> rows;
    std::vector<Update> updates;
    std::vector<ViewResult> rc, rg;
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
        for (uint32_t t = 0; t < a.warmup + a.ticks; ++t) {
            updates.clear();
            uint32_t md = 0;
            sim.step(300, updates, md);
            ++tick;
            RunStats sc, sg;
            if (cpu) cpu->run(store, tick, updates, params, rc, sc);
            if (gpu) gpu->run(store, tick, updates, params, rg, sg);
            if (t < a.warmup) continue;
            if (cpu) {
                row.cpu.total.push_back(sc.totalMs);
                row.cpu.derive.push_back(sc.deriveMs);
                row.cpu.views.push_back(sc.viewsMs);
            }
            if (gpu) {
                row.gpu.total.push_back(sg.totalMs);
                row.gpu.derive.push_back(sg.deriveMs);
                row.gpu.views.push_back(sg.viewsMs);
                row.gpu.upload.push_back(sg.uploadMs);
            }
            if (cpu && gpu && a.validate) {
                std::string first;
                const int bad = compareResults(rc, rg, &first);
                if (bad && row.firstDiff.empty()) row.firstDiff = first;
                row.mismatches += bad;
            }
        }
        std::printf("views %u done\n", nv);
        rows.push_back(std::move(row));
    }

    std::printf("\nrows %u, ticks %u per view count; engine ms per tick (p50 / p95)\n\n", store.rows, a.ticks);
    std::printf("%7s | %17s | %17s | %8s | %s\n", "views", "cpu", "gpu", "speedup", "cpu = gpu");
    std::printf("--------+-------------------+-------------------+----------+----------\n");
    for (const Row& r : rows) {
        char c[32] = "-", g[32] = "-", sp[16] = "-";
        if (cpu) std::snprintf(c, sizeof c, "%7.1f / %7.1f", pct(r.cpu.total, 0.5), pct(r.cpu.total, 0.95));
        if (gpu) std::snprintf(g, sizeof g, "%7.1f / %7.1f", pct(r.gpu.total, 0.5), pct(r.gpu.total, 0.95));
        if (cpu && gpu) std::snprintf(sp, sizeof sp, "%6.1fx", pct(r.cpu.total, 0.5) / pct(r.gpu.total, 0.5));
        const char* v = !(cpu && gpu && a.validate) ? "-" : r.mismatches ? "MISMATCH" : "identical";
        std::printf("%7u | %17s | %17s | %8s | %s\n", r.views, c, g, sp, v);
        if (r.mismatches) std::printf("        first difference: %s\n", r.firstDiff.c_str());
    }
    if (gpu && !rows.empty()) {
        const Row& r = rows.back();
        std::printf("\nlast run, gpu p50: upload %.1f ms, derive %.1f ms, views %.1f ms\n", pct(r.gpu.upload, 0.5),
                    pct(r.gpu.derive, 0.5), pct(r.gpu.views, 0.5));
    }
    if (cpu && !rows.empty()) {
        const Row& r = rows.back();
        std::printf("last run, cpu p50: derive %.1f ms, views %.1f ms\n", pct(r.cpu.derive, 0.5), pct(r.cpu.views, 0.5));
    }

    if (!a.json.empty()) {
        std::ofstream f(a.json);
        f << "{\n  \"rows\": " << store.rows << ",\n  \"instruments\": " << store.instruments
          << ",\n  \"cpu\": \"" << (cpu ? cpu->devices()[0] : "") << "\",\n  \"gpus\": [";
        if (gpu)
            for (size_t i = 0; i < gpu->devices().size(); ++i) f << (i ? ", " : "") << '"' << gpu->devices()[i] << '"';
        f << "],\n  \"ticks\": " << a.ticks << ",\n  \"results\": [\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i];
            f << "    {\"views\": " << r.views << ", \"cpu_p50\": " << pct(r.cpu.total, 0.5)
              << ", \"cpu_p95\": " << pct(r.cpu.total, 0.95) << ", \"gpu_p50\": " << pct(r.gpu.total, 0.5)
              << ", \"gpu_p95\": " << pct(r.gpu.total, 0.95) << ", \"mismatches\": " << r.mismatches << "}"
              << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
        std::printf("\nwrote %s\n", a.json.c_str());
    }
    return 0;
}
