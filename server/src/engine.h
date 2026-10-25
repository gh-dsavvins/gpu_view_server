#pragma once
#include "view.h"

#include <memory>
#include <string>
#include <vector>

namespace gvs {

struct PivotCell {
    uint32_t count = 0;
    int64_t leaves = 0, notional = 0, pnl = 0;  // sums of trunc(value)
    bool operator==(const PivotCell&) const = default;
};

struct ViewResult {
    uint32_t total = 0;            // rows passing the filter
    std::vector<uint32_t> rows;    // viewport row indices in sort order
    std::vector<PivotCell> pivot;  // indexed by pivot group id
};

struct RunStats {
    double totalMs = 0;   // wall clock of run()
    double uploadMs = 0;  // host -> GPU copies (GPU only)
    double deriveMs = 0;  // derived-column recalculation
    double viewsMs = 0;   // filter + sort/select + pivot for all views
};

class Engine {
public:
    virtual ~Engine() = default;
    virtual std::string name() const = 0;
    virtual std::vector<std::string> devices() const = 0;

    // Recalculates derived columns and evaluates every view against the store.
    // `updates` are the base-column cells changed since the previous tick.
    virtual void run(const Store& s, uint64_t tick, const std::vector<Update>& updates,
                     const std::vector<ViewParams>& views, std::vector<ViewResult>& out, RunStats& stats) = 0;
};

struct GpuConfig {
    uint32_t maxDevices = 8;
    uint32_t slots = 8;           // views processed together per batch
    uint32_t groupsPerSlot = 64;  // workgroups per view for the full-table passes
};

struct CpuConfig {
    unsigned threads = 0;   // 0 = one per CPU in `cpus`, or per hardware thread
    std::vector<int> cpus;  // pin threads to these CPUs (Linux); empty = no pinning
    bool simple = false;    // the straightforward engine instead of the tuned one
};

// "cpu" = tuned engine (cache-blocked view batches, histogram select);
// "cpu-simple" = one view per thread, full scan + nth_element. Benchmarks run both and take the best.
std::unique_ptr<Engine> makeCpuEngine(const CpuConfig& cfg);
std::unique_ptr<Engine> makeGpuEngine(const GpuConfig& cfg, std::string& err);

// Number of views whose results differ; describes the first difference in `first`.
int compareResults(const std::vector<ViewResult>& a, const std::vector<ViewResult>& b, std::string* first);

}  // namespace gvs
