#include "sim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <set>

namespace gvs {

namespace {

constexpr uint32_t DAY_MS = 86'400'000;
constexpr uint32_t OPEN_MS = 9 * 3'600'000 + 30 * 60'000;

const char* kDesks[] = {"CLIENT-DMA", "DERIV-HEDGE", "EQ-ASIA",  "EQ-EU-CASH",
                        "EQ-US-CASH", "EQ-US-PROG",  "ETF-MM",   "QUANT-ALPHA"};
const char* kStrategies[] = {"ARRIVAL", "CLOSE", "DARK",  "DMA",  "ICEBERG", "IS",      "LIQ-SEEK", "MOC",
                             "OPEN",    "PEG",   "POV",   "SNIPER", "SOR",   "STEALTH", "TWAP",     "VWAP"};
const char* kVenues[] = {"ARCA", "BATS", "DARKPOOL", "EDGX", "IEX", "MEMX", "NASDAQ", "NYSE"};
const char* kFirst[] = {"alex", "ben", "chen", "dana", "eli", "fay", "gus", "hana"};
const char* kLast[] = {"adams", "baker", "cho", "diaz", "evans", "fischer", "garcia", "huang"};

float roundPx(double p) { return float(std::max(0.01, std::round(p * 100.0) / 100.0)); }
float lots(double q) { return float(std::max(100.0, std::floor(q / 100.0) * 100.0)); }

// Compacts to one update per (row, col), keeping the last write.
void compactUpdates(std::vector<Update>& v) {
    std::stable_sort(v.begin(), v.end(),
                     [](const Update& a, const Update& b) { return a.row != b.row ? a.row < b.row : a.col < b.col; });
    size_t w = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i + 1 < v.size() && v[i + 1].row == v[i].row && v[i + 1].col == v[i].col) continue;
        v[w++] = v[i];
    }
    v.resize(w);
}

}  // namespace

Simulator::Simulator(Store& s, const SimConfig& c) : s_(s), c_(c), rng_(c.seed) { init(); }

void Simulator::set(uint32_t r, uint32_t c, uint32_t v) {
    s_.cols[c][r] = v;
    if (out_) out_->push_back({r, c, v});
}

uint32_t Simulator::sampleInstrument() {
    const double u = unit() * instCdf_.back();
    return uint32_t(std::upper_bound(instCdf_.begin(), instCdf_.end(), u) - instCdf_.begin());
}

void Simulator::init() {
    const uint32_t n = std::min(c_.rows, MAX_ROWS);
    const uint32_t m = std::max(1u, c_.instruments);
    s_.rows = n;
    s_.instruments = m;

    std::set<std::string> syms;
    while (syms.size() < m) {
        std::string sym(3 + uniform(2), 'A');
        for (char& ch : sym) ch = char('A' + uniform(26));
        syms.insert(sym);
    }
    s_.dict[C_INSTRUMENT].assign(syms.begin(), syms.end());
    s_.dict[C_SIDE] = {"BUY", "SELL"};
    s_.dict[C_STATUS] = {"NEW", "PARTIAL", "FILLED", "CANCELLED"};
    s_.dict[C_DESK].assign(std::begin(kDesks), std::end(kDesks));
    s_.dict[C_STRATEGY].assign(std::begin(kStrategies), std::end(kStrategies));
    s_.dict[C_VENUE].assign(std::begin(kVenues), std::end(kVenues));
    for (const char* f : kFirst)
        for (const char* l : kLast) s_.dict[C_TRADER].push_back(std::string(f) + "." + l);
    std::set<std::string> accts;
    while (accts.size() < 1024) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "ACC-%05u", uniform(100000));
        accts.insert(buf);
    }
    s_.dict[C_ACCOUNT].assign(accts.begin(), accts.end());
    traderDesk_.resize(s_.card(C_TRADER));
    for (uint32_t& d : traderDesk_) d = uniform(s_.card(C_DESK));

    s_.mid.resize(m);
    s_.invBps.resize(m);
    for (uint32_t i = 0; i < m; ++i) {
        s_.mid[i] = roundPx(std::clamp(std::exp(std::log(40.0) + normal(0.9)), 1.0, 3000.0));
        s_.invBps[i] = 10000.0f / s_.mid[i];
    }

    // Zipf-like popularity over a random ranking of instruments.
    std::vector<uint32_t> rank(m);
    std::iota(rank.begin(), rank.end(), 0);
    std::shuffle(rank.begin(), rank.end(), rng_);
    instCdf_.resize(m);
    double acc = 0;
    for (uint32_t i = 0; i < m; ++i) instCdf_[i] = acc += 1.0 / std::pow(rank[i] + 1.0, 0.9);

    for (auto& col : s_.cols) col.assign(n, 0);
    clock_ = 13 * 3'600'000;  // mid-session; orders were created since the open
    for (uint32_t r = 0; r < n; ++r) {
        newOrder(r, OPEN_MS + uniform(clock_ - OPEN_MS));
        const float qty = s_.f(C_QTY, r);
        const float mid = s_.mid[s_.u(C_INSTRUMENT, r)];
        const double u = unit();
        float filled = 0;
        uint32_t status = ST_NEW;
        if (u < 0.30) {
            filled = std::min(lots(qty * (0.05 + 0.9 * unit())), qty - 100);
            status = filled > 0 ? ST_PARTIAL : ST_NEW;
        } else if (u < 0.40) {
            filled = qty;
            status = ST_FILLED;
        } else if (u < 0.45) {
            filled = unit() < 0.5 ? std::min(lots(qty * unit()), qty - 100) : 0;
            status = ST_CANCELLED;
        }
        if (filled > 0) {
            setF(r, C_FILLED, filled);
            setF(r, C_AVG_PX, float(mid * (1.0 + normal(0.002))));
        }
        set(r, C_STATUS, status);
        const uint32_t created = s_.u(C_CREATE_TS, r);
        if (status != ST_NEW) set(r, C_UPDATE_TS, created + uniform(clock_ - created + 1));
    }
}

void Simulator::newOrder(uint32_t r, uint32_t ts) {
    const uint32_t inst = sampleInstrument();
    const uint32_t side = uniform(2);
    const uint32_t trader = uniform(s_.card(C_TRADER));
    const double mid = s_.mid[inst];
    double off = std::min(std::abs(normal(0.004)), 0.05);
    if (unit() < 0.15) off = -off * 0.2;  // aggressive: through the mid
    const double px = side == SIDE_BUY ? mid * (1.0 - off) : mid * (1.0 + off);
    const double q = std::clamp(std::round(std::exp(std::log(8.0) + normal(1.1))), 1.0, 10000.0) * 100.0;

    set(r, C_ORDER_ID, nextOrderId_++);
    set(r, C_INSTRUMENT, inst);
    set(r, C_SIDE, side);
    set(r, C_DESK, traderDesk_[trader]);
    set(r, C_TRADER, trader);
    set(r, C_ACCOUNT, uniform(s_.card(C_ACCOUNT)));
    set(r, C_STRATEGY, uniform(s_.card(C_STRATEGY)));
    set(r, C_VENUE, uniform(s_.card(C_VENUE)));
    set(r, C_STATUS, ST_NEW);
    setF(r, C_LIMIT_PX, roundPx(px));
    setF(r, C_QTY, float(q));
    setF(r, C_FILLED, 0.0f);
    setF(r, C_AVG_PX, 0.0f);
    set(r, C_CREATE_TS, ts);
    set(r, C_UPDATE_TS, ts);
}

void Simulator::fill(uint32_t r) {
    const float qty = s_.f(C_QTY, r);
    const float filled = s_.f(C_FILLED, r);
    const float q = std::min(qty - filled, lots(qty * (0.05 + 0.55 * unit())));
    const double px = s_.mid[s_.u(C_INSTRUMENT, r)] * (1.0 + normal(0.0005));
    const double avg = (double(s_.f(C_AVG_PX, r)) * filled + px * q) / (double(filled) + q);
    const float nf = filled + q;
    setF(r, C_FILLED, nf);
    setF(r, C_AVG_PX, float(avg));
    set(r, C_STATUS, nf >= qty ? ST_FILLED : ST_PARTIAL);
    set(r, C_UPDATE_TS, clock_);
}

void Simulator::step(uint32_t tickMs, std::vector<Update>& out, uint32_t& mdChanges) {
    out_ = &out;
    clock_ = (clock_ + tickMs) % DAY_MS;
    const uint32_t n = s_.rows, m = s_.instruments;

    mdChanges = uint32_t(m * c_.mdMoveShare);
    for (uint32_t i = 0; i < mdChanges; ++i) {
        const uint32_t inst = uniform(m);
        s_.mid[inst] = roundPx(s_.mid[inst] * std::exp(normal(0.0008)));
        s_.invBps[inst] = 10000.0f / s_.mid[inst];
    }

    auto open = [&](uint32_t r) {
        const uint32_t st = s_.u(C_STATUS, r);
        return st == ST_NEW || st == ST_PARTIAL;
    };
    for (uint32_t i = 0; i < c_.fillsPerTick; ++i) {
        const uint32_t r = uniform(n);
        if (open(r)) fill(r);
    }
    for (uint32_t i = 0; i < c_.fillsPerTick / 10; ++i) {
        const uint32_t r = uniform(n);
        if (open(r)) {
            set(r, C_STATUS, ST_CANCELLED);
            set(r, C_UPDATE_TS, clock_);
        }
    }
    for (uint32_t i = 0; i < c_.newPerTick; ++i) {
        for (int probe = 0; probe < 8; ++probe) {
            const uint32_t r = uniform(n);
            if (!open(r)) {
                newOrder(r, clock_);
                break;
            }
        }
    }
    compactUpdates(out);
    out_ = nullptr;
}

}  // namespace gvs
