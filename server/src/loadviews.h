#pragma once
// Synthetic client views: random filters, sorts, viewports and pivots, standing in for many real users.
#include "view.h"

#include <algorithm>
#include <iterator>
#include <random>

namespace gvs {

inline ViewSpec randomView(std::mt19937_64& rng, const Store& s) {
    auto pick = [&](uint32_t n) { return uint32_t(rng() % n); };
    auto chance = [&](double p) { return std::uniform_real_distribution<double>(0, 1)(rng) < p; };
    ViewSpec v;

    const uint32_t nPred = pick(3);
    std::vector<uint32_t> used;
    while (v.filter.size() < nPred) {
        const uint32_t kind = pick(8);
        if (std::find(used.begin(), used.end(), kind) != used.end()) continue;
        used.push_back(kind);
        Pred p;
        switch (kind) {
            case 0: p = {C_STATUS, OP_IN, 0, (1u << ST_NEW) | (1u << ST_PARTIAL)}; break;
            case 1: p = {C_DESK, OP_EQ, double(pick(s.card(C_DESK))), 0}; break;
            case 2: p = {C_SIDE, OP_EQ, double(pick(2)), 0}; break;
            case 3: p = {C_STRATEGY, OP_IN, 0, uint32_t(rng()) & 0xFFFFu}; break;
            case 4: p = {C_VENUE, OP_EQ, double(pick(s.card(C_VENUE))), 0}; break;
            case 5: p = {C_NOTIONAL, OP_GT, 10000.0 * (1 + pick(100)), 0}; break;
            case 6: p = {C_DIST_BPS, OP_LT, double(int(pick(101)) - 50), 0}; break;
            default: p = {C_LEAVES, OP_GT, 0, 0}; break;
        }
        v.filter.push_back(p);
    }

    static const uint32_t sortable[] = {C_NOTIONAL, C_DIST_BPS,  C_PNL,        C_LEAVES,   C_LIMIT_PX, C_QTY,
                                        C_UPDATE_TS, C_CREATE_TS, C_INSTRUMENT, C_TRADER,  C_ACCOUNT,  C_ORDER_ID,
                                        C_FILLED,   C_MID,       C_DESK,       C_STATUS};
    const uint32_t nSort = 1 + pick(2);
    for (uint32_t i = 0; i < nSort; ++i) v.sort.push_back({sortable[pick(std::size(sortable))], chance(0.5)});

    v.offset = chance(0.7) ? 0 : pick(20000);
    v.limit = 50;

    if (chance(0.3)) {
        static const uint32_t pairs[][2] = {
            {C_DESK, C_STATUS}, {C_STRATEGY, C_VENUE}, {C_DESK, C_SIDE}, {C_TRADER, C_SIDE}, {C_VENUE, C_STATUS}};
        const auto& pr = pairs[pick(std::size(pairs))];
        v.pivotBy = {pr[0], pr[1]};
    }
    return v;
}

}  // namespace gvs
