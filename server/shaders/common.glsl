// Shared definitions for all kernels. Keep in sync with columns.h, view.h and gpu_engine.cpp.

#define C_INSTRUMENT 1u
#define C_SIDE 2u
#define C_STATUS 8u
#define C_LIMIT_PX 9u
#define C_QTY 10u
#define C_FILLED 11u
#define C_AVG_PX 12u
#define C_MID 15u
#define C_LEAVES 16u
#define C_NOTIONAL 17u
#define C_DIST_BPS 18u
#define C_PNL 19u

#define KIND_CAT 0u
#define KIND_F32 2u
#define SIDE_BUY 0u
#define ST_FILLED 2u
#define ST_CANCELLED 3u

#define OP_EQ 0u
#define OP_NE 1u
#define OP_LT 2u
#define OP_LE 3u
#define OP_GT 4u
#define OP_IN 6u

#define WG 256u
#define ROW_MASK 0xFFFFFFu
#define MAX_LIMIT 256u
#define MAX_GROUPS 512u
#define PIVOT_STRIDE 7u

// Per-slot selection state (uints).
#define STATE_STRIDE 528u
#define S_CAND 0u      // candidates in the current buffer
#define S_NEXT 1u      // candidates appended to the next buffer
#define S_OUT 2u       // accepted viewport keys
#define S_TOTAL 3u     // rows passing the filter
#define S_NGROUPS 4u   // open rank windows (at most 2)
#define S_A 5u         // window start per group (2 entries)
#define S_B 7u         // window end per group (2 entries)
#define S_ACTION 16u   // per group x digit: 0 drop, 1 accept, 2+g continue as group g

#define HIST_STRIDE 512u  // 2 groups x 256 digits
#define RES_ROWS 4u
#define RES_PIVOT 260u    // RES_ROWS + MAX_LIMIT
#define RES_STRIDE 3844u  // RES_PIVOT + MAX_GROUPS * PIVOT_STRIDE

struct ViewParams {
    uint vOffset; uint vLimit; uint nPred; uint nKey;
    uint predCol[4]; uint predOp[4]; uint predKind[4]; uint predVal[4]; uint predMask[4];
    uint keyCol[3]; uint keyKind[3]; uint keyCatBits[3]; uint keyBits[3]; uint keyShift[3]; uint keyDesc[3];
    uint nPiv; uint pivCol0; uint pivCol1; uint pivCard1;
    uint nGroups; uint pad0;
};

layout(std430, binding = 0) buffer Cols { uint col[]; };            // NUM_COLS x n
layout(std430, binding = 1) readonly buffer Inst { float inst[]; };  // mid[m], invBps[m]
layout(std430, binding = 2) readonly buffer Params { ViewParams vp[]; };
layout(std430, binding = 3) buffer State { uint st[]; };
layout(std430, binding = 4) buffer Hist { uint hist[]; };
layout(std430, binding = 5) buffer Cand { uint cand[]; };          // 2 halves x slots x cap x (hi, lo, group)
layout(std430, binding = 6) buffer Out { uint outKeys[]; };        // slots x MAX_LIMIT x (hi, lo)
layout(std430, binding = 7) buffer Pivot { uint piv[]; };          // slots x MAX_GROUPS x PIVOT_STRIDE
layout(std430, binding = 8) buffer Disp { uint disp[]; };          // indirect dispatch args
layout(std430, binding = 9) readonly buffer Upd { uint upd[]; };   // (row, col, value) triples
layout(std430, binding = 10) buffer Res { uint res[]; };           // views x RES_STRIDE, host visible

layout(push_constant) uniform PC {
    uint n;         // rows
    uint m;         // instruments
    uint nb;        // views in this batch
    uint slotCap;   // slots allocated
    uint viewBase;  // device-local index of the batch's first view
    uint pass;      // radix pass (digit) 0..7
    uint srcHalf;   // candidate buffer half to read
    uint cap;       // candidate capacity per slot
    uint count;     // update count
} pc;

uint colv(uint c, uint r) { return col[c * pc.n + r]; }

bool passFilter(in ViewParams P, uint r) {
    for (uint i = 0u; i < P.nPred; ++i) {
        uint v = colv(P.predCol[i], r);
        uint op = P.predOp[i];
        bool ok;
        if (op == OP_IN) {
            ok = v < 32u && ((P.predMask[i] >> v) & 1u) != 0u;
        } else if (P.predKind[i] == KIND_F32) {
            float a = uintBitsToFloat(v), b = uintBitsToFloat(P.predVal[i]);
            ok = op == OP_EQ ? a == b : op == OP_NE ? a != b : op == OP_LT ? a < b
               : op == OP_LE ? a <= b : op == OP_GT ? a > b : a >= b;
        } else {
            uint b = P.predVal[i];
            ok = op == OP_EQ ? v == b : op == OP_NE ? v != b : op == OP_LT ? v < b
               : op == OP_LE ? v <= b : op == OP_GT ? v > b : v >= b;
        }
        if (!ok) return false;
    }
    return true;
}

// 64-bit sort key as (hi, lo); same packing as buildKey() in view.h.
uvec2 buildKey(in ViewParams P, uint r) {
    uint hi = 0u, lo = r;
    for (uint k = 0u; k < P.nKey; ++k) {
        uint v = colv(P.keyCol[k], r);
        uint norm;
        if (P.keyKind[k] == KIND_CAT) norm = v << (32u - P.keyCatBits[k]);
        else if (P.keyKind[k] == KIND_F32) norm = (v & 0x80000000u) != 0u ? ~v : (v | 0x80000000u);
        else norm = v;
        uint bits = P.keyBits[k];
        uint part = norm >> (32u - bits);
        if (P.keyDesc[k] != 0u) part ^= bits == 32u ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        uint s = P.keyShift[k];  // always >= 24
        if (s >= 32u) {
            hi |= part << (s - 32u);
        } else {
            lo |= part << s;
            if (s + bits > 32u) hi |= part >> (32u - s);
        }
    }
    return uvec2(hi, lo);
}

uint digitOf(uvec2 k, uint pass) {
    return pass < 4u ? (k.x >> (24u - 8u * pass)) & 0xFFu : (k.y >> (24u - 8u * (pass - 4u))) & 0xFFu;
}

uint pivotGroup(in ViewParams P, uint r) {
    uint g = colv(P.pivCol0, r);
    if (P.nPiv == 2u) g = g * P.pivCard1 + colv(P.pivCol1, r);
    return g;
}

// trunc(x) as a two's-complement int64 (lo, hi); matches int64_t(float) on the CPU.
void toI64(float x, out uint lo, out uint hi) {
    float ax = abs(x);
    float fh = floor(ax * (1.0 / 4294967296.0));
    hi = uint(fh);
    lo = uint(ax - fh * 4294967296.0);
    if (x < 0.0) {
        lo = ~lo + 1u;
        hi = ~hi + (lo == 0u ? 1u : 0u);
    }
}
