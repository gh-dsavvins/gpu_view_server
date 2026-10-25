#pragma once
#include "store.h"

#include <bit>
#include <string>
#include <vector>

namespace gvs {

enum Op : uint32_t { OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE, OP_IN };

// View definition as it arrives from a client.
struct Pred {
    uint32_t col = 0;
    uint32_t op = OP_EQ;
    double value = 0;   // compared as float for float columns, as uint otherwise
    uint32_t mask = 0;  // OP_IN: bit i set = category id i matches
};

struct SortKey {
    uint32_t col = 0;
    bool desc = false;
};

struct ViewSpec {
    std::vector<Pred> filter;  // AND of predicates
    std::vector<SortKey> sort;
    uint32_t offset = 0;
    uint32_t limit = 50;
    std::vector<uint32_t> pivotBy;  // up to 2 category columns
};

// Compiled view, evaluated identically by both engines.
// Layout is mirrored by `struct ViewParams` in shaders/common.glsl (std430, all uint).
struct ViewParams {
    uint32_t offset, limit, nPred, nKey;
    uint32_t predCol[MAX_PRED], predOp[MAX_PRED], predKind[MAX_PRED], predVal[MAX_PRED], predMask[MAX_PRED];
    uint32_t keyCol[MAX_SORT], keyKind[MAX_SORT], keyCatBits[MAX_SORT], keyBits[MAX_SORT], keyShift[MAX_SORT],
        keyDesc[MAX_SORT];
    uint32_t nPiv, pivCol0, pivCol1, pivCard1;
    uint32_t nGroups, pad0;
};
static_assert(sizeof(ViewParams) == 48 * 4);

bool compileView(const ViewSpec& v, const Store& s, ViewParams& p, std::string& err);

template <class T>
inline bool cmpOp(T a, T b, uint32_t op) {
    switch (op) {
        case OP_EQ: return a == b;
        case OP_NE: return a != b;
        case OP_LT: return a < b;
        case OP_LE: return a <= b;
        case OP_GT: return a > b;
        default: return a >= b;
    }
}

// col[c] points at column c (base and derived).
inline bool passFilter(const ViewParams& p, const uint32_t* const* col, uint32_t r) {
    for (uint32_t i = 0; i < p.nPred; ++i) {
        const uint32_t v = col[p.predCol[i]][r];
        const uint32_t op = p.predOp[i];
        bool ok;
        if (op == OP_IN)
            ok = v < 32 && ((p.predMask[i] >> v) & 1u);
        else if (p.predKind[i] == KIND_F32)
            ok = cmpOp(std::bit_cast<float>(v), std::bit_cast<float>(p.predVal[i]), op);
        else
            ok = cmpOp(v, p.predVal[i], op);
        if (!ok) return false;
    }
    return true;
}

// Packs the sort columns into the top KEY_BITS bits and the row index into the rest.
inline uint64_t buildKey(const ViewParams& p, const uint32_t* const* col, uint32_t r) {
    uint64_t key = r;
    for (uint32_t k = 0; k < p.nKey; ++k) {
        const uint32_t v = col[p.keyCol[k]][r];
        uint32_t norm;
        if (p.keyKind[k] == KIND_CAT)
            norm = v << (32 - p.keyCatBits[k]);
        else if (p.keyKind[k] == KIND_F32)
            norm = (v & 0x80000000u) ? ~v : (v | 0x80000000u);
        else
            norm = v;
        const uint32_t bits = p.keyBits[k];
        uint32_t part = norm >> (32 - bits);
        if (p.keyDesc[k]) part ^= bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        key |= uint64_t(part) << p.keyShift[k];
    }
    return key;
}

inline uint32_t pivotGroup(const ViewParams& p, const uint32_t* const* col, uint32_t r) {
    uint32_t g = col[p.pivCol0][r];
    if (p.nPiv == 2) g = g * p.pivCard1 + col[p.pivCol1][r];
    return g;
}

}  // namespace gvs
