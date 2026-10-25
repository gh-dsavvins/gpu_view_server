#include "view.h"

#include <algorithm>

namespace gvs {

static uint32_t bitsFor(uint32_t card) {
    uint32_t b = 1;
    while ((1ull << b) < card) ++b;
    return b;
}

bool compileView(const ViewSpec& v, const Store& s, ViewParams& p, std::string& err) {
    p = {};
    if (v.filter.size() > MAX_PRED) {
        err = "too many filter predicates (max " + std::to_string(MAX_PRED) + ")";
        return false;
    }
    for (size_t i = 0; i < v.filter.size(); ++i) {
        const Pred& f = v.filter[i];
        if (f.col >= NUM_COLS || f.op > OP_IN) {
            err = "bad filter predicate";
            return false;
        }
        const ColKind kind = COLS[f.col].kind;
        if (f.op == OP_IN && kind != KIND_CAT) {
            err = std::string("IN needs a category column, got ") + COLS[f.col].name;
            return false;
        }
        p.predCol[i] = f.col;
        p.predOp[i] = f.op;
        p.predKind[i] = kind;
        p.predMask[i] = f.mask;
        p.predVal[i] = kind == KIND_F32 ? std::bit_cast<uint32_t>(float(f.value))
                                        : uint32_t(std::clamp(f.value, 0.0, 4294967295.0));
    }
    p.nPred = uint32_t(v.filter.size());

    if (v.sort.size() > MAX_SORT) {
        err = "too many sort columns (max " + std::to_string(MAX_SORT) + ")";
        return false;
    }
    uint32_t used = 0;
    for (const SortKey& k : v.sort) {
        if (k.col >= NUM_COLS) {
            err = "bad sort column";
            return false;
        }
        const ColKind kind = COLS[k.col].kind;
        const uint32_t catBits = kind == KIND_CAT ? bitsFor(std::max(2u, s.card(k.col))) : 32;
        // Later columns get whatever is left of the 40-bit budget (their low bits are dropped).
        const uint32_t bits = std::min(catBits, KEY_BITS - used);
        if (bits == 0) break;
        const uint32_t i = p.nKey++;
        p.keyCol[i] = k.col;
        p.keyKind[i] = kind;
        p.keyCatBits[i] = catBits;
        p.keyBits[i] = bits;
        p.keyShift[i] = 64 - used - bits;
        p.keyDesc[i] = k.desc ? 1 : 0;
        used += bits;
    }

    p.offset = std::min(v.offset, MAX_ROWS);
    p.limit = std::min(v.limit, MAX_LIMIT);

    if (v.pivotBy.size() > 2) {
        err = "pivot supports at most 2 columns";
        return false;
    }
    for (uint32_t c : v.pivotBy) {
        if (c >= NUM_COLS || COLS[c].kind != KIND_CAT) {
            err = "pivot columns must be category columns";
            return false;
        }
    }
    if (!v.pivotBy.empty()) {
        p.nPiv = uint32_t(v.pivotBy.size());
        p.pivCol0 = v.pivotBy[0];
        p.pivCol1 = p.nPiv == 2 ? v.pivotBy[1] : 0;
        p.pivCard1 = p.nPiv == 2 ? s.card(p.pivCol1) : 1;
        const uint64_t groups = uint64_t(s.card(p.pivCol0)) * p.pivCard1;
        if (groups > MAX_GROUPS) {
            err = "pivot has " + std::to_string(groups) + " cells (max " + std::to_string(MAX_GROUPS) + ")";
            return false;
        }
        p.nGroups = uint32_t(groups);
    }
    return true;
}

}  // namespace gvs
