#include "protocol.h"

#include "pb.h"

namespace gvs {

using pb::Reader;
using pb::Wire;
using pb::Writer;

namespace {

// ServerMsg oneof field numbers.
enum : uint32_t { SM_SCHEMA = 1, SM_VIEW = 2, SM_STATS = 3, SM_ERROR = 4 };

std::string wrap(uint32_t field, const Writer& inner) {
    Writer w;
    w.msg(field, inner);
    return std::move(w.buf);
}

bool decodePred(std::string_view d, Pred& p) {
    Reader r(d);
    uint32_t f;
    Wire w;
    while (r.next(f, w)) {
        if (f == 1 && w == pb::VARINT) p.col = uint32_t(r.varint());
        else if (f == 2 && w == pb::VARINT) p.op = uint32_t(r.varint());
        else if (f == 3 && w == pb::FIXED64) p.value = r.f64();
        else if (f == 4 && w == pb::VARINT) p.mask = uint32_t(r.varint());
        else r.skip(w);
    }
    return r.ok();
}

bool decodeSortKey(std::string_view d, SortKey& k) {
    Reader r(d);
    uint32_t f;
    Wire w;
    while (r.next(f, w)) {
        if (f == 1 && w == pb::VARINT) k.col = uint32_t(r.varint());
        else if (f == 2 && w == pb::VARINT) k.desc = r.varint() != 0;
        else r.skip(w);
    }
    return r.ok();
}

bool decodeSpec(std::string_view d, ViewSpec& v) {
    v = {};
    v.limit = 0;
    Reader r(d);
    uint32_t f;
    Wire w;
    while (r.next(f, w)) {
        if (f == 1 && w == pb::LEN) {
            v.filter.emplace_back();
            if (!decodePred(r.bytes(), v.filter.back())) return false;
        } else if (f == 2 && w == pb::LEN) {
            v.sort.emplace_back();
            if (!decodeSortKey(r.bytes(), v.sort.back())) return false;
        } else if (f == 3 && w == pb::VARINT) {
            v.offset = uint32_t(r.varint());
        } else if (f == 4 && w == pb::VARINT) {
            v.limit = uint32_t(r.varint());
        } else if (f == 5) {
            r.u32s(w, v.pivotBy);
        } else {
            r.skip(w);
        }
    }
    return r.ok();
}

// Reads a message holding `view_id = 1` and optionally `ViewSpec spec = 2`.
bool decodeViewMsg(std::string_view d, ClientCommand& c) {
    Reader r(d);
    uint32_t f;
    Wire w;
    while (r.next(f, w)) {
        if (f == 1 && w == pb::VARINT) c.viewId = uint32_t(r.varint());
        else if (f == 2 && w == pb::LEN) {
            if (!decodeSpec(r.bytes(), c.spec)) return false;
        } else r.skip(w);
    }
    return r.ok();
}

}  // namespace

bool decodeClientMsg(std::string_view data, ClientCommand& cmd) {
    cmd = {};
    Reader r(data);
    uint32_t f;
    Wire w;
    while (r.next(f, w)) {
        if (w != pb::LEN) {
            r.skip(w);
            continue;
        }
        const std::string_view body = r.bytes();
        if (f == 1) {
            cmd.type = ClientCommand::SUBSCRIBE;
            if (!decodeViewMsg(body, cmd)) return false;
        } else if (f == 2) {
            cmd.type = ClientCommand::UNSUBSCRIBE;
            if (!decodeViewMsg(body, cmd)) return false;
        } else if (f == 3) {
            cmd.type = ClientCommand::SET_BACKEND;
            Reader b(body);
            uint32_t bf;
            Wire bw;
            while (b.next(bf, bw)) {
                if (bf == 1 && bw == pb::VARINT) cmd.backend = uint32_t(b.varint());
                else b.skip(bw);
            }
            if (!b.ok()) return false;
        }
    }
    return r.ok() && cmd.type != ClientCommand::NONE;
}

std::string encodeSchema(const Store& s, const SchemaInfo& info) {
    Writer m;
    for (uint32_t c = 0; c < NUM_COLS; ++c) {
        Writer col;
        col.u32(1, c);
        col.str(2, COLS[c].name);
        col.u32(3, COLS[c].kind);
        for (const std::string& name : s.dict[c]) col.str(4, name);
        col.boolean(5, c >= NUM_BASE_COLS);
        m.msg(1, col);
    }
    m.u32(2, s.rows);
    m.packedU32(3, info.backends);
    m.u32(4, info.backend);
    m.u32(5, info.tickMs);
    for (const std::string& g : info.gpuNames) m.str(6, g);
    m.u32(7, info.cpuThreads);
    m.u32(8, info.loadViews);
    return wrap(SM_SCHEMA, m);
}

std::string encodeViewUpdate(const Store& s, uint32_t viewId, uint64_t tick, const ViewParams& p, const ViewResult& r) {
    Writer m;
    m.u32(1, viewId);
    m.u64(2, tick);
    m.u32(3, r.total);
    m.u32(4, p.offset);
    double vals[NUM_COLS];
    for (uint32_t row : r.rows) {
        for (uint32_t c = 0; c < NUM_BASE_COLS; ++c)
            vals[c] = COLS[c].kind == KIND_F32 ? double(s.f(c, row)) : double(s.u(c, row));
        const Derived d = deriveRow(s, row);
        vals[C_MID] = d.mid;
        vals[C_LEAVES] = d.leaves;
        vals[C_NOTIONAL] = d.notional;
        vals[C_DIST_BPS] = d.distBps;
        vals[C_PNL] = d.pnl;
        Writer rw;
        rw.u32(1, row);
        rw.packedF64(2, vals, NUM_COLS);
        m.msg(5, rw);
    }
    for (uint32_t g = 0; g < r.pivot.size(); ++g) {
        const PivotCell& c = r.pivot[g];
        if (!c.count) continue;
        Writer cw;
        if (p.nPiv == 2)
            cw.packedU32(1, {g / p.pivCard1, g % p.pivCard1});
        else
            cw.packedU32(1, {g});
        cw.u64(2, c.count);
        cw.i64(3, c.leaves);
        cw.i64(4, c.notional);
        cw.i64(5, c.pnl);
        m.msg(6, cw);
    }
    return wrap(SM_VIEW, m);
}

std::string encodeStats(const StatsInfo& st) {
    Writer m;
    m.u64(1, st.tick);
    m.u32(2, st.backend);
    m.f64(3, st.run.totalMs);
    m.f64(4, st.run.deriveMs);
    m.f64(5, st.run.viewsMs);
    m.u32(6, st.views);
    m.u32(7, st.clientViews);
    m.u32(8, st.rows);
    m.u32(9, st.updates);
    m.u32(10, st.mdChanges);
    m.i32(11, st.validation);
    m.u32(12, st.tickMs);
    m.f64(13, st.run.uploadMs);
    return wrap(SM_STATS, m);
}

std::string encodeError(std::string_view text) {
    Writer m;
    m.str(1, text);
    return wrap(SM_ERROR, m);
}

}  // namespace gvs
