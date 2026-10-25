#pragma once
#include "columns.h"

#include <bit>
#include <string>
#include <vector>

namespace gvs {

// Columnar order store. Base columns only; derived columns live in the engines.
struct Store {
    uint32_t rows = 0;
    uint32_t instruments = 0;
    std::vector<uint32_t> cols[NUM_BASE_COLS];  // raw 32-bit cells; float columns hold float bits
    std::vector<float> mid;                     // per instrument
    std::vector<float> invBps;                  // per instrument: 10000 / mid
    std::vector<std::string> dict[NUM_COLS];    // names of category values, ordered by id

    uint32_t card(uint32_t c) const { return uint32_t(dict[c].size()); }
    uint32_t u(uint32_t c, uint32_t r) const { return cols[c][r]; }
    float f(uint32_t c, uint32_t r) const { return std::bit_cast<float>(cols[c][r]); }
};

// One changed base-column cell, streamed to the GPU each tick.
struct Update {
    uint32_t row, col, value;
};
static_assert(sizeof(Update) == 12);

struct Derived {
    float mid, leaves, notional, distBps, pnl;
};

// Must match shaders/derive.comp bit for bit; the build uses -ffp-contract=off and the shader `precise`.
inline Derived deriveRow(float mid, float invBps, float qty, float filled, float limitPx, float avgPx,
                         uint32_t status, uint32_t side) {
    const float s = side == SIDE_BUY ? 1.0f : -1.0f;
    Derived d;
    d.mid = mid;
    d.leaves = (status == ST_FILLED || status == ST_CANCELLED) ? 0.0f : qty - filled;
    d.notional = d.leaves * mid;
    d.distBps = ((mid - limitPx) * s) * invBps;
    d.pnl = (filled * (mid - avgPx)) * s;
    return d;
}

inline Derived deriveRow(const Store& s, uint32_t r) {
    const uint32_t inst = s.u(C_INSTRUMENT, r);
    return deriveRow(s.mid[inst], s.invBps[inst], s.f(C_QTY, r), s.f(C_FILLED, r), s.f(C_LIMIT_PX, r),
                     s.f(C_AVG_PX, r), s.u(C_STATUS, r), s.u(C_SIDE, r));
}

}  // namespace gvs
