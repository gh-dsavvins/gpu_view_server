#pragma once
// Encoding/decoding of proto/gvs.proto messages.
#include "engine.h"

#include <string>
#include <string_view>
#include <vector>

namespace gvs {

enum BackendId : uint32_t { BACKEND_CPU = 0, BACKEND_GPU = 1 };

struct ClientCommand {
    enum Type { NONE, SUBSCRIBE, UNSUBSCRIBE, SET_BACKEND } type = NONE;
    uint32_t viewId = 0;
    ViewSpec spec;
    uint32_t backend = 0;
};

bool decodeClientMsg(std::string_view data, ClientCommand& cmd);

struct SchemaInfo {
    std::vector<uint32_t> backends;
    uint32_t backend = 0;
    uint32_t tickMs = 0;
    std::vector<std::string> gpuNames;
    uint32_t cpuThreads = 0;
    uint32_t loadViews = 0;
};

struct StatsInfo {
    uint64_t tick = 0;
    uint32_t backend = 0;
    RunStats run;
    uint32_t views = 0, clientViews = 0, rows = 0, updates = 0, mdChanges = 0, tickMs = 0;
    int32_t validation = -1;
};

std::string encodeSchema(const Store& s, const SchemaInfo& info);
std::string encodeViewUpdate(const Store& s, uint32_t viewId, uint64_t tick, const ViewParams& p, const ViewResult& r);
std::string encodeStats(const StatsInfo& st);
std::string encodeError(std::string_view text);

}  // namespace gvs
