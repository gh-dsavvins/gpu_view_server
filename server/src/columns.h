#pragma once
#include <cstdint>

namespace gvs {

enum ColKind : uint32_t { KIND_CAT = 0, KIND_U32 = 1, KIND_F32 = 2 };

// Column ids are shared with the shaders (shaders/common.glsl) and sent to clients in the Schema.
enum Col : uint32_t {
    C_ORDER_ID = 0,
    C_INSTRUMENT,
    C_SIDE,
    C_DESK,
    C_TRADER,
    C_ACCOUNT,
    C_STRATEGY,
    C_VENUE,
    C_STATUS,
    C_LIMIT_PX,
    C_QTY,
    C_FILLED,
    C_AVG_PX,
    C_CREATE_TS,
    C_UPDATE_TS,
    // Derived columns: recomputed from market data every tick.
    C_MID,
    C_LEAVES,
    C_NOTIONAL,
    C_DIST_BPS,
    C_PNL,
    NUM_COLS
};

constexpr uint32_t NUM_BASE_COLS = C_MID;

struct ColInfo {
    const char* name;
    ColKind kind;
};

inline constexpr ColInfo COLS[NUM_COLS] = {
    {"order_id", KIND_U32},  {"instrument", KIND_CAT}, {"side", KIND_CAT},     {"desk", KIND_CAT},
    {"trader", KIND_CAT},    {"account", KIND_CAT},    {"strategy", KIND_CAT}, {"venue", KIND_CAT},
    {"status", KIND_CAT},    {"limit_px", KIND_F32},   {"qty", KIND_F32},      {"filled", KIND_F32},
    {"avg_px", KIND_F32},    {"create_ts", KIND_U32},  {"update_ts", KIND_U32}, {"mid", KIND_F32},
    {"leaves", KIND_F32},    {"notional", KIND_F32},   {"dist_bps", KIND_F32}, {"pnl", KIND_F32},
};

enum Side : uint32_t { SIDE_BUY = 0, SIDE_SELL = 1 };
enum Status : uint32_t { ST_NEW = 0, ST_PARTIAL = 1, ST_FILLED = 2, ST_CANCELLED = 3 };

// A sort key is 64 bits: up to 40 bits of packed sort columns, then the row index as a tie-break.
constexpr uint32_t ROW_BITS = 24;
constexpr uint32_t MAX_ROWS = 1u << ROW_BITS;
constexpr uint32_t KEY_BITS = 64 - ROW_BITS;

constexpr uint32_t MAX_PRED = 4;
constexpr uint32_t MAX_SORT = 3;
constexpr uint32_t MAX_LIMIT = 256;   // viewport rows per view
constexpr uint32_t MAX_GROUPS = 512;  // pivot cells per view
constexpr uint32_t PIVOT_STRIDE = 7;  // count + 3 x int64 as (lo, hi)

}  // namespace gvs
