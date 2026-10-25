#include "engine.h"
#include "thread_pool.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>

namespace gvs {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Tuned CPU baseline, so the GPU is compared against the best we can do on a CPU:
//  - Views are processed in blocks of BLOCK that share one pass over the table: a chunk of rows is pulled
//    from DRAM once and evaluated for every view in the block while it sits in cache.
//  - Rows are split into segments too, so every core is busy whatever the number of views.
//  - Selection is a two-pass histogram select (no full sort, no nth_element over the whole table):
//    pass 0 histograms an 11-bit key digit, pass 1 keeps only the rows in the two buckets at the edges of
//    the viewport, and those few candidates are finished with nth_element.
//  - The digit starts after the leading key bits that were constant last tick (e.g. float exponents),
//    so buckets stay small. Keys with a different prefix land in a "below" or "above" bucket, so the
//    result is exact even when the data moves.
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

        pool_.parallelFor((n + CHUNK * 16 - 1) / (CHUNK * 16), [&](size_t chunk, unsigned) {
            const uint32_t begin = uint32_t(chunk) * CHUNK * 16;
            const uint32_t end = std::min<uint32_t>(n, begin + CHUNK * 16);
            for (uint32_t r = begin; r < end; ++r) {
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
        const uint32_t nv = uint32_t(views.size());
        out.resize(nv);
        learn(views);
        work_.resize(nv);

        const uint32_t blocks = (nv + BLOCK - 1) / BLOCK;
        const uint32_t maxSegs = std::max(1u, n / (CHUNK * 4));
        const uint32_t segs = blocks ? std::clamp((2 * pool_.size() + blocks - 1) / blocks, 1u, maxSegs) : 1;
        const uint32_t segRows = ((n + segs - 1) / segs + CHUNK - 1) / CHUNK * CHUNK;
        const uint32_t tasks = blocks * segs;
        if (acc_.size() < size_t(tasks) * BLOCK) acc_.resize(size_t(tasks) * BLOCK);

        auto forTask = [&](size_t t, auto&& body) {
            const uint32_t blk = uint32_t(t) / segs, seg = uint32_t(t) % segs;
            const uint32_t v0 = blk * BLOCK, v1 = std::min(nv, v0 + BLOCK);
            const uint32_t r0 = std::min(n, seg * segRows), r1 = std::min(n, r0 + segRows);
            for (uint32_t c0 = r0; c0 < r1; c0 += CHUNK) {
                const uint32_t c1 = std::min(r1, c0 + CHUNK);
                for (uint32_t v = v0; v < v1; ++v) body(v, acc_[t * BLOCK + (v - v0)], c0, c1);
            }
        };

        // Pass 0: filter, histogram, pivot.
        pool_.parallelFor(tasks, [&](size_t t, unsigned) {
            const uint32_t blk = uint32_t(t) / segs;
            for (uint32_t v = blk * BLOCK; v < std::min(nv, blk * BLOCK + BLOCK); ++v) {
                Acc& a = acc_[t * BLOCK + (v - blk * BLOCK)];
                a.hist.assign(BINS, 0);
                a.andK = ~0ull;
                a.orK = 0;
                a.piv.assign(views[v].nGroups, PivotCell{});
            }
            forTask(t, [&](uint32_t v, Acc& a, uint32_t c0, uint32_t c1) {
                const ViewParams& p = views[v];
                const Learned& l = learned_[v];
                uint32_t* hist = a.hist.data();
                uint64_t andK = a.andK, orK = a.orK;
                for (uint32_t r = c0; r < c1; ++r) {
                    if (!passFilter(p, col, r)) continue;
                    const uint64_t key = buildKey(p, col, r);
                    ++hist[binOf(key, l)];
                    andK &= key;
                    orK |= key;
                    if (p.nGroups) {
                        PivotCell& c = a.piv[pivotGroup(p, col, r)];
                        ++c.count;
                        c.leaves += int64_t(std::bit_cast<float>(col[C_LEAVES][r]));
                        c.notional += int64_t(std::bit_cast<float>(col[C_NOTIONAL][r]));
                        c.pnl += int64_t(std::bit_cast<float>(col[C_PNL][r]));
                    }
                }
                a.andK = andK;
                a.orK = orK;
            });
        });

        // Merge per view and find the buckets holding the first and last viewport rank.
        pool_.parallelFor(nv, [&](size_t v, unsigned) {
            const ViewParams& p = views[v];
            const uint32_t blk = uint32_t(v) / BLOCK, slot = uint32_t(v) % BLOCK;
            ViewWork& w = work_[v];
            ViewResult& res = out[v];
            w.hist.assign(BINS, 0);
            w.andK = ~0ull;
            w.orK = 0;
            res.pivot.assign(p.nGroups, PivotCell{});
            for (uint32_t seg = 0; seg < segs; ++seg) {
                const Acc& a = acc_[(size_t(blk) * segs + seg) * BLOCK + slot];
                for (uint32_t i = 0; i < BINS; ++i) w.hist[i] += a.hist[i];
                w.andK &= a.andK;
                w.orK |= a.orK;
                for (uint32_t g = 0; g < p.nGroups; ++g) {
                    PivotCell& c = res.pivot[g];
                    c.count += a.piv[g].count;
                    c.leaves += a.piv[g].leaves;
                    c.notional += a.piv[g].notional;
                    c.pnl += a.piv[g].pnl;
                }
            }
            uint64_t total = 0;
            for (uint32_t c : w.hist) total += c;
            res.total = uint32_t(total);
            w.a = uint32_t(std::min<uint64_t>(p.offset, total));
            w.b = uint32_t(std::min<uint64_t>(uint64_t(p.offset) + p.limit, total));
            w.lowBin = w.highBin = BINS;
            uint64_t start = 0;
            for (uint32_t i = 0; i < BINS && w.a < w.b; ++i) {
                const uint64_t end = start + w.hist[i];
                if (w.lowBin == BINS && w.a < end) w.lowBin = i, w.startLow = start;
                if (w.b - 1 < end) {
                    w.highBin = i;
                    w.startHigh = start;
                    break;
                }
                start = end;
            }
        });

        // Pass 1: collect rows strictly between the edge buckets (accepted) and the edge-bucket candidates.
        pool_.parallelFor(tasks, [&](size_t t, unsigned) {
            const uint32_t blk = uint32_t(t) / segs;
            for (uint32_t v = blk * BLOCK; v < std::min(nv, blk * BLOCK + BLOCK); ++v) {
                Acc& a = acc_[t * BLOCK + (v - blk * BLOCK)];
                a.accepted.clear();
                a.low.clear();
                a.high.clear();
            }
            forTask(t, [&](uint32_t v, Acc& a, uint32_t c0, uint32_t c1) {
                const ViewWork& w = work_[v];
                if (w.a >= w.b) return;
                const ViewParams& p = views[v];
                const Learned& l = learned_[v];
                for (uint32_t r = c0; r < c1; ++r) {
                    if (!passFilter(p, col, r)) continue;
                    const uint64_t key = buildKey(p, col, r);
                    const uint32_t bin = binOf(key, l);
                    if (bin < w.lowBin || bin > w.highBin) continue;
                    if (bin == w.lowBin) a.low.push_back(key);
                    else if (bin == w.highBin) a.high.push_back(key);
                    else a.accepted.push_back(key);
                }
            });
        });

        // Finish each view on its candidates.
        pool_.parallelFor(nv, [&](size_t v, unsigned tid) {
            ViewWork& w = work_[v];
            ViewResult& res = out[v];
            res.rows.clear();
            Learned& l = learned_[v];
            if (res.total) {
                const uint64_t varying = w.orK & ~w.andK;
                l.skip = std::min<uint32_t>(uint32_t(std::countl_zero(varying)), 64 - DIGIT);
                l.prefix = l.skip ? w.andK >> (64 - l.skip) : 0;
            }
            if (w.a >= w.b) return;
            const uint32_t blk = uint32_t(v) / BLOCK, slot = uint32_t(v) % BLOCK;
            std::vector<uint64_t>& sel = scratch_[tid].sel;
            std::vector<uint64_t>& low = scratch_[tid].low;
            std::vector<uint64_t>& high = scratch_[tid].high;
            sel.clear();
            low.clear();
            high.clear();
            for (uint32_t seg = 0; seg < segs; ++seg) {
                const Acc& a = acc_[(size_t(blk) * segs + seg) * BLOCK + slot];
                sel.insert(sel.end(), a.accepted.begin(), a.accepted.end());
                low.insert(low.end(), a.low.begin(), a.low.end());
                high.insert(high.end(), a.high.begin(), a.high.end());
            }
            const size_t from = w.a - w.startLow;
            if (w.lowBin == w.highBin) {
                const size_t to = w.b - w.startLow;
                std::nth_element(low.begin(), low.begin() + from, low.end());
                std::partial_sort(low.begin() + from, low.begin() + to, low.end());
                sel.insert(sel.end(), low.begin() + from, low.begin() + to);
            } else {
                std::nth_element(low.begin(), low.begin() + from, low.end());
                sel.insert(sel.end(), low.begin() + from, low.end());
                const size_t to = w.b - w.startHigh;
                std::nth_element(high.begin(), high.begin() + to, high.end());
                sel.insert(sel.end(), high.begin(), high.begin() + to);
            }
            std::sort(sel.begin(), sel.end());
            for (uint64_t k : sel) res.rows.push_back(uint32_t(k & (MAX_ROWS - 1)));
        });

        stats.viewsMs = msSince(t1);
        stats.uploadMs = 0;
        stats.totalMs = msSince(t0);
    }

private:
    static constexpr uint32_t DIGIT = 11;
    static constexpr uint32_t BINS = (1u << DIGIT) + 2;  // below, 2^DIGIT digits, above
    static constexpr uint32_t CHUNK = 4096;               // rows per cache block
    static constexpr uint32_t BLOCK = 8;                  // views sharing one pass

    // Per view, carried across ticks: leading key bits that were constant last tick.
    struct Learned {
        ViewParams p{};
        uint32_t skip = 0;
        uint64_t prefix = 0;
    };

    // Per (task, view) partial results.
    struct Acc {
        std::vector<uint32_t> hist;
        uint64_t andK = ~0ull, orK = 0;
        std::vector<PivotCell> piv;
        std::vector<uint64_t> accepted, low, high;
    };

    struct ViewWork {
        std::vector<uint32_t> hist;
        uint64_t andK = ~0ull, orK = 0;
        uint32_t a = 0, b = 0;  // viewport rank window
        uint32_t lowBin = BINS, highBin = BINS;
        uint64_t startLow = 0, startHigh = 0;  // rank of the first key in each edge bucket
    };

    struct Scratch {
        std::vector<uint64_t> sel, low, high;
    };

    static uint32_t binOf(uint64_t key, const Learned& l) {
        if (l.skip == 0) return uint32_t(key >> (64 - DIGIT)) + 1;
        const uint64_t top = key >> (64 - l.skip);
        if (top < l.prefix) return 0;
        if (top > l.prefix) return BINS - 1;
        return uint32_t((key << l.skip) >> (64 - DIGIT)) + 1;
    }

    // Keeps the learned prefix while a view's filter and sort stay the same (viewport moves are fine).
    void learn(const std::vector<ViewParams>& views) {
        learned_.resize(views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            ViewParams a = learned_[i].p, b = views[i];
            a.offset = b.offset = a.limit = b.limit = 0;
            if (std::memcmp(&a, &b, sizeof a) != 0) learned_[i] = Learned{};
            learned_[i].p = views[i];
        }
    }

    ThreadPool pool_;
    std::vector<Scratch> scratch_;
    std::vector<Learned> learned_;
    std::vector<Acc> acc_;
    std::vector<ViewWork> work_;
    std::vector<uint32_t> derived_[NUM_COLS - NUM_BASE_COLS];
};

}  // namespace

std::unique_ptr<Engine> makeCpuEngine(unsigned threads) { return std::make_unique<CpuEngine>(threads); }

}  // namespace gvs
