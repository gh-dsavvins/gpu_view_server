#pragma once
// Single-threaded poll() HTTP + WebSocket server (RFC 6455, binary frames). Serves static files from an
// optional directory; any request with "Upgrade: websocket" becomes a WebSocket connection.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gvs {

class WsServer {
public:
    struct Handler {
        virtual ~Handler() = default;
        virtual void onOpen(uint64_t id) = 0;
        virtual void onMessage(uint64_t id, const std::string& data) = 0;
        virtual void onClose(uint64_t id) = 0;
    };

    ~WsServer();
    bool listen(uint16_t port, std::string& err);
    void setWwwRoot(std::string dir) { www_ = std::move(dir); }

    // Waits up to timeoutMs for socket activity and dispatches it to the handler.
    void poll(int timeoutMs, Handler& h);

    // Queues a binary WebSocket message.
    void send(uint64_t id, const std::string& payload);
    // Bytes waiting to be written to the connection.
    size_t queued(uint64_t id) const;

private:
    struct Conn {
        int fd = -1;
        std::string in, out, frag;
        size_t outPos = 0;
        bool ws = false;
        bool closeAfterWrite = false;
        bool dead = false;
    };

    void acceptAll();
    void readConn(uint64_t id, Conn& c, Handler& h);
    void handleHttp(uint64_t id, Conn& c, Handler& h);
    void handleFrames(uint64_t id, Conn& c, Handler& h);
    void flush(Conn& c);
    void queueFrame(Conn& c, uint8_t opcode, const std::string& payload);

    int listenFd_ = -1;
    uint64_t nextId_ = 1;
    std::map<uint64_t, Conn> conns_;
    std::string www_;
};

}  // namespace gvs
