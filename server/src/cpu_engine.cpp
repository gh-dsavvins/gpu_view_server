#include "engine.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>

namespace gvs {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Baseline: every view is one task on a pool of all cores. A task scans the table once
// (filter + key + pivot), then std::nth_element / partial_sort pick the viewport.
class CpuEngine final : public Engine {
public:
    explicit CpuEngine(unsigned threads) : pool_(threads), scratch_(pool_.size()) {}

    std::string name() const override { return "cpu"; }
    std::vector<std::string> devices() const override { return {std::to_string(pool_.size()) + " threads"}; }

    void run(const Store& s, uint64_t, const std::vector<Update>&, const std::vector<ViewParams>& views,
             std::vector<ViewResult>& out, RunStats& stats) override {
        const auto t0 = Clock::now();
        const uint32_t n = s.rows;
        for (auto& d : derived_) d.resize(n);

        constexpr uint32_t CHUNK = 1 << 16;
        pool_.parallelFor((n + CHUNK - 1) / CHUNK, [&](size_t chunk, unsigned) {
            const uint32_t end = std::min<uint32_t>(n, uint32_t(chunk + 1) * CHUNK);
            for (uint32_t r = uint32_t(chunk) * CHUNK; r < end; ++r) {
                const Derived d = deriveRow(s, r);
                derived_[C_MID - NUM_BASE_COLS][r] = std::bit_cast<uint32_t>(d.mid);
                derived_[C_LEAVES - NUM_BASE_COLS][r] = std::bit_cast<uint32_t>(d.leaves);
                derived_[C_NOTIONAL - NUM_BASE_COLS][r] = std::bit_cast<uint32_t>(d.notional);
                derived_[C_DIST_BPS - NUM_BASE_COLS][r] = std::bit_cast<uint32_t>(d.distBps);
                derived_[C_PNL - NUM_BASE_COLS][r] = std::bit_cast<uint32_t>(d.pnl);
            }
        });
        stats.deriveMs = msSince(t0);

        const uint32_t* col[NUM_COLS];
        for (uint32_t c = 0; c < NUM_BASE_COLS; ++c) col[c] = s.cols[c].data();
        for (uint32_t c = NUM_BASE_COLS; c < NUM_COLS; ++c) col[c] = derived_[c - NUM_BASE_COLS].data();

        const auto t1 = Clock::now();
        out.resize(views.size());
        pool_.parallelFor(views.size(),
                          [&](size_t i, unsigned tid) { runView(views[i], col, n, scratch_[tid], out[i]); });
        stats.viewsMs = msSince(t1);
        stats.uploadMs = 0;
        stats.totalMs = msSince(t0);
    }

private:
    static void runView(const ViewParams& p, const uint32_t* const* col, uint32_t n, std::vector<uint64_t>& keys,
                        ViewResult& res) {
        keys.clear();
        res.pivot.assign(p.nGroups, PivotCell{});
        for (uint32_t r = 0; r < n; ++r) {
            if (!passFilter(p, col, r)) continue;
            keys.push_back(buildKey(p, col, r));
            if (p.nGroups) {
                PivotCell& c = res.pivot[pivotGroup(p, col, r)];
                ++c.count;
                c.leaves += int64_t(std::bit_cast<float>(col[C_LEAVES][r]));
                c.notional += int64_t(std::bit_cast<float>(col[C_NOTIONAL][r]));
                c.pnl += int64_t(std::bit_cast<float>(col[C_PNL][r]));
            }
        }
        res.total = uint32_t(keys.size());
        const size_t a = std::min<size_t>(p.offset, keys.size());
        const size_t b = std::min<size_t>(size_t(p.offset) + p.limit, keys.size());
        res.rows.clear();
        if (a < b) {
            std::nth_element(keys.begin(), keys.begin() + a, keys.end());
            std::partial_sort(keys.begin() + a, keys.begin() + b, keys.end());
            for (size_t i = a; i < b; ++i) res.rows.push_back(uint32_t(keys[i] & (MAX_ROWS - 1)));
        }
    }

    ThreadPool pool_;
    std::vector<std::vector<uint64_t>> scratch_;
    std::vector<uint32_t> derived_[NUM_COLS - NUM_BASE_COLS];
};

}  // namespace

std::unique_ptr<Engine> makeCpuEngine(unsigned threads) { return std::make_unique<CpuEngine>(threads); }

}  // namespace gvs
