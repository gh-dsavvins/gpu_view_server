#include "net.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gvs {

namespace {

constexpr size_t MAX_MESSAGE = 16u << 20;

uint32_t rol(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }

std::array<uint8_t, 20> sha1(const std::string& msg) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    std::string m = msg;
    const uint64_t bits = uint64_t(msg.size()) * 8;
    m.push_back(char(0x80));
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; --i) m.push_back(char((bits >> (i * 8)) & 0xFF));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(uint8_t(m[off + 4 * i])) << 24 | uint32_t(uint8_t(m[off + 4 * i + 1])) << 16 |
                   uint32_t(uint8_t(m[off + 4 * i + 2])) << 8 | uint32_t(uint8_t(m[off + 4 * i + 3]));
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20) f = (b & c) | (~b & d), k = 0x5A827999;
            else if (i < 40) f = b ^ c ^ d, k = 0x6ED9EBA1;
            else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC;
            else f = b ^ c ^ d, k = 0xCA62C1D6;
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    std::array<uint8_t, 20> out;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = uint8_t(h[i] >> (24 - 8 * j));
    return out;
}

std::string base64(const uint8_t* p, size_t n) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = uint32_t(p[i]) << 16 | (i + 1 < n ? uint32_t(p[i + 1]) << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(i + 1 < n ? tbl[(v >> 6) & 63] : '=');
        out.push_back(i + 2 < n ? tbl[v & 63] : '=');
    }
    return out;
}

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(uint8_t(c)));
    return s;
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r");
    const size_t b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

const char* contentType(const std::string& path) {
    auto ends = [&](const char* ext) {
        const size_t n = std::strlen(ext);
        return path.size() >= n && path.compare(path.size() - n, n, ext) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js")) return "text/javascript";
    if (ends(".css")) return "text/css";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".json")) return "application/json";
    if (ends(".proto")) return "text/plain";
    return "application/octet-stream";
}

void setNonBlocking(int fd) { fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK); }

}  // namespace

WsServer::~WsServer() {
    for (auto& [id, c] : conns_) ::close(c.fd);
    if (listenFd_ >= 0) ::close(listenFd_);
}

bool WsServer::listen(uint16_t port, std::string& err) {
    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        err = std::strerror(errno);
        return false;
    }
    int one = 1;
    setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || ::listen(listenFd_, 64) < 0) {
        err = std::strerror(errno);
        return false;
    }
    setNonBlocking(listenFd_);
    return true;
}

void WsServer::poll(int timeoutMs, Handler& h) {
    std::vector<pollfd> fds{{listenFd_, POLLIN, 0}};
    std::vector<uint64_t> ids{0};
    for (auto& [id, c] : conns_) {
        fds.push_back({c.fd, short(POLLIN | (c.out.size() > c.outPos ? POLLOUT : 0)), 0});
        ids.push_back(id);
    }
    if (::poll(fds.data(), fds.size(), timeoutMs) <= 0) return;
    if (fds[0].revents & POLLIN) acceptAll();
    for (size_t i = 1; i < fds.size(); ++i) {
        auto it = conns_.find(ids[i]);
        if (it == conns_.end()) continue;
        Conn& c = it->second;
        if (fds[i].revents & (POLLIN | POLLHUP | POLLERR)) readConn(ids[i], c, h);
        if (!c.dead && (fds[i].revents & POLLOUT)) flush(c);
    }
    for (auto it = conns_.begin(); it != conns_.end();) {
        Conn& c = it->second;
        if (c.closeAfterWrite && c.out.size() == c.outPos) c.dead = true;
        if (c.dead) {
            if (c.ws) h.onClose(it->first);
            ::close(c.fd);
            it = conns_.erase(it);
        } else {
            ++it;
        }
    }
}

void WsServer::acceptAll() {
    for (;;) {
        const int fd = ::accept(listenFd_, nullptr, nullptr);
        if (fd < 0) return;
        setNonBlocking(fd);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        conns_[nextId_++].fd = fd;
    }
}

void WsServer::readConn(uint64_t id, Conn& c, Handler& h) {
    char buf[65536];
    for (;;) {
        const ssize_t n = ::recv(c.fd, buf, sizeof buf, 0);
        if (n > 0) {
            c.in.append(buf, size_t(n));
            continue;
        }
        if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) c.dead = true;
        break;
    }
    if (!c.ws) handleHttp(id, c, h);
    if (c.ws) handleFrames(id, c, h);
}

void WsServer::handleHttp(uint64_t id, Conn& c, Handler& h) {
    const size_t end = c.in.find("\r\n\r\n");
    if (end == std::string::npos) {
        if (c.in.size() > 65536) c.dead = true;
        return;
    }
    std::istringstream req(c.in.substr(0, end));
    c.in.erase(0, end + 4);
    std::string line, method, path;
    std::getline(req, line);
    std::istringstream(line) >> method >> path;
    std::map<std::string, std::string> headers;
    while (std::getline(req, line)) {
        const size_t colon = line.find(':');
        if (colon != std::string::npos) headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }

    if (lower(headers["upgrade"]) == "websocket" && !headers["sec-websocket-key"].empty()) {
        const auto digest = sha1(headers["sec-websocket-key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
        c.out += "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " +
                 base64(digest.data(), digest.size()) + "\r\n\r\n";
        c.ws = true;
        flush(c);
        h.onOpen(id);
        return;
    }

    std::string status = "404 Not Found", type = "text/plain", body = "not found\n";
    path = path.substr(0, path.find('?'));
    if (path == "/") path = "/index.html";
    if (!www_.empty() && method == "GET" && path.find("..") == std::string::npos) {
        std::ifstream f(www_ + path, std::ios::binary);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            status = "200 OK";
            type = contentType(path);
            body = ss.str();
        }
    } else if (www_.empty()) {
        body = "gvs_server: no --www directory; run the client with `npm run dev` and connect to ws://<host>:<port>/ws\n";
    }
    c.out += "HTTP/1.1 " + status + "\r\nContent-Type: " + type + "\r\nContent-Length: " + std::to_string(body.size()) +
             "\r\nConnection: close\r\n\r\n" + body;
    c.closeAfterWrite = true;
    flush(c);
}

void WsServer::handleFrames(uint64_t id, Conn& c, Handler& h) {
    for (;;) {
        if (c.in.size() < 2) return;
        const auto* p = reinterpret_cast<const uint8_t*>(c.in.data());
        const bool fin = p[0] & 0x80;
        const uint8_t op = p[0] & 0x0F;
        const bool masked = p[1] & 0x80;
        uint64_t len = p[1] & 0x7F;
        size_t hdr = 2;
        if (len == 126) {
            if (c.in.size() < 4) return;
            len = uint64_t(p[2]) << 8 | p[3];
            hdr = 4;
        } else if (len == 127) {
            if (c.in.size() < 10) return;
            len = 0;
            for (int i = 0; i < 8; ++i) len = len << 8 | p[2 + i];
            hdr = 10;
        }
        if (len > MAX_MESSAGE) {
            c.dead = true;
            return;
        }
        const size_t maskPos = hdr;
        if (masked) hdr += 4;
        if (c.in.size() < hdr + len) return;
        std::string payload = c.in.substr(hdr, size_t(len));
        if (masked)
            for (size_t i = 0; i < payload.size(); ++i) payload[i] = char(payload[i] ^ p[maskPos + (i & 3)]);
        c.in.erase(0, hdr + size_t(len));

        if (op == 0x8) {  // close
            queueFrame(c, 0x8, "");
            c.closeAfterWrite = true;
            return;
        }
        if (op == 0x9) {  // ping
            queueFrame(c, 0xA, payload);
            continue;
        }
        if (op == 0xA) continue;  // pong
        if (op == 0x1 || op == 0x2 || op == 0x0) {
            c.frag += payload;
            if (c.frag.size() > MAX_MESSAGE) {
                c.dead = true;
                return;
            }
            if (fin) {
                h.onMessage(id, c.frag);
                c.frag.clear();
            }
        }
    }
}

void WsServer::queueFrame(Conn& c, uint8_t opcode, const std::string& payload) {
    std::string& o = c.out;
    o.push_back(char(0x80 | opcode));
    const uint64_t n = payload.size();
    if (n < 126) {
        o.push_back(char(n));
    } else if (n < 65536) {
        o.push_back(char(126));
        o.push_back(char(n >> 8));
        o.push_back(char(n & 0xFF));
    } else {
        o.push_back(char(127));
        for (int i = 7; i >= 0; --i) o.push_back(char((n >> (8 * i)) & 0xFF));
    }
    o += payload;
    flush(c);
}

void WsServer::send(uint64_t id, const std::string& payload) {
    auto it = conns_.find(id);
    if (it == conns_.end() || !it->second.ws || it->second.dead) return;
    queueFrame(it->second, 0x2, payload);
}

size_t WsServer::queued(uint64_t id) const {
    auto it = conns_.find(id);
    return it == conns_.end() ? 0 : it->second.out.size() - it->second.outPos;
}

void WsServer::flush(Conn& c) {
    while (c.outPos < c.out.size()) {
        const ssize_t n = ::send(c.fd, c.out.data() + c.outPos, c.out.size() - c.outPos, 0);
        if (n > 0) {
            c.outPos += size_t(n);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) break;
        c.dead = true;
        return;
    }
    if (c.outPos == c.out.size()) {
        c.out.clear();
        c.outPos = 0;
    } else if (c.outPos > (1u << 20)) {
        c.out.erase(0, c.outPos);
        c.outPos = 0;
    }
}

}  // namespace gvs
