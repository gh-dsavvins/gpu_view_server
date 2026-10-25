#pragma once
// Minimal protobuf wire-format codec, enough for proto/gvs.proto (varint, fixed64, length-delimited).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace gvs::pb {

enum Wire : uint32_t { VARINT = 0, FIXED64 = 1, LEN = 2, FIXED32 = 5 };

class Writer {
public:
    std::string buf;

    void varint(uint64_t v) {
        while (v >= 0x80) {
            buf.push_back(char(v | 0x80));
            v >>= 7;
        }
        buf.push_back(char(v));
    }
    void tag(uint32_t field, Wire w) { varint((uint64_t(field) << 3) | w); }

    // proto3 scalars: default values are not written.
    void u32(uint32_t f, uint32_t v) { if (v) { tag(f, VARINT); varint(v); } }
    void u64(uint32_t f, uint64_t v) { if (v) { tag(f, VARINT); varint(v); } }
    void i32(uint32_t f, int32_t v) { if (v) { tag(f, VARINT); varint(uint64_t(int64_t(v))); } }
    void i64(uint32_t f, int64_t v) { if (v) { tag(f, VARINT); varint(uint64_t(v)); } }
    void boolean(uint32_t f, bool v) { if (v) { tag(f, VARINT); varint(1); } }
    void f64(uint32_t f, double v) {
        if (v == 0 && !std::signbit(v)) return;
        tag(f, FIXED64);
        raw64(v);
    }
    void str(uint32_t f, std::string_view s) {
        tag(f, LEN);
        varint(s.size());
        buf.append(s);
    }
    void msg(uint32_t f, const Writer& w) { str(f, w.buf); }
    void packedF64(uint32_t f, const double* v, size_t n) {
        if (!n) return;
        tag(f, LEN);
        varint(n * 8);
        for (size_t i = 0; i < n; ++i) raw64(v[i]);
    }
    void packedU32(uint32_t f, const std::vector<uint32_t>& v) {
        if (v.empty()) return;
        Writer w;
        for (uint32_t x : v) w.varint(x);
        str(f, w.buf);
    }

private:
    void raw64(double v) {
        char b[8];
        std::memcpy(b, &v, 8);  // little-endian hosts only
        buf.append(b, 8);
    }
};

class Reader {
public:
    explicit Reader(std::string_view d) : p_(d.data()), e_(d.data() + d.size()) {}

    bool next(uint32_t& field, Wire& wire) {
        if (!ok_ || p_ >= e_) return false;
        const uint64_t t = varint();
        field = uint32_t(t >> 3);
        wire = Wire(t & 7);
        return ok_;
    }
    uint64_t varint() {
        uint64_t v = 0;
        for (int s = 0; s < 64 && p_ < e_; s += 7) {
            const uint8_t b = uint8_t(*p_++);
            v |= uint64_t(b & 0x7F) << s;
            if (!(b & 0x80)) return v;
        }
        ok_ = false;
        return 0;
    }
    double f64() {
        if (e_ - p_ < 8) {
            ok_ = false;
            return 0;
        }
        double d;
        std::memcpy(&d, p_, 8);
        p_ += 8;
        return d;
    }
    std::string_view bytes() {
        const uint64_t n = varint();
        if (!ok_ || uint64_t(e_ - p_) < n) {
            ok_ = false;
            return {};
        }
        std::string_view v(p_, size_t(n));
        p_ += n;
        return v;
    }
    void skip(Wire w) {
        switch (w) {
            case VARINT: varint(); break;
            case FIXED64: advance(8); break;
            case LEN: bytes(); break;
            case FIXED32: advance(4); break;
            default: ok_ = false;
        }
    }
    // Repeated uint32 in either packed or unpacked encoding.
    void u32s(Wire w, std::vector<uint32_t>& out) {
        if (w == VARINT) {
            out.push_back(uint32_t(varint()));
        } else if (w == LEN) {
            Reader r(bytes());
            while (r.p_ < r.e_ && r.ok_) out.push_back(uint32_t(r.varint()));
            ok_ &= r.ok_;
        } else {
            skip(w);
        }
    }
    bool ok() const { return ok_; }

private:
    void advance(size_t n) {
        if (size_t(e_ - p_) < n)
            ok_ = false;
        else
            p_ += n;
    }

    const char* p_;
    const char* e_;
    bool ok_ = true;
};

}  // namespace gvs::pb
