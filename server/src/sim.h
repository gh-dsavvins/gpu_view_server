#pragma once
#include "store.h"

#include <random>
#include <vector>

namespace gvs {

struct SimConfig {
    uint32_t rows = 4'000'000;
    uint32_t instruments = 5000;
    uint32_t fillsPerTick = 20000;   // fill attempts per tick (only open orders fill)
    uint32_t newPerTick = 5000;      // finished orders recycled as new orders per tick
    float mdMoveShare = 0.3f;        // share of instruments whose price moves each tick
    uint64_t seed = 42;
};

// Generates a realistic-looking order book (Zipf instrument popularity, desks/traders/strategies,
// partial fills) and evolves it: market data moves, open orders fill or cancel, finished orders recycle.
class Simulator {
public:
    Simulator(Store& s, const SimConfig& c);

    // Advances the clock by tickMs, mutates the store and appends every changed base cell to `out`
    // (one entry per cell, last write wins).
    void step(uint32_t tickMs, std::vector<Update>& out, uint32_t& mdChanges);

    uint32_t nowMs() const { return clock_; }

private:
    void init();
    void newOrder(uint32_t r, uint32_t ts);
    void fill(uint32_t r);
    void set(uint32_t r, uint32_t c, uint32_t v);
    void setF(uint32_t r, uint32_t c, float v) { set(r, c, std::bit_cast<uint32_t>(v)); }
    uint32_t sampleInstrument();
    uint32_t uniform(uint32_t n) { return uint32_t(rng_() % n); }
    double unit() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
    double normal(double sd) { return std::normal_distribution<double>(0.0, sd)(rng_); }

    Store& s_;
    SimConfig c_;
    std::mt19937_64 rng_;
    std::vector<double> instCdf_;
    std::vector<uint32_t> traderDesk_;
    std::vector<Update>* out_ = nullptr;
    uint32_t clock_ = 0;
    uint32_t nextOrderId_ = 1'000'000;
};

}  // namespace gvs
