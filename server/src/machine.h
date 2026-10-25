#pragma once
// Facts about the machine and build, recorded with every benchmark so results can be checked.
#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifndef GVS_GIT
#define GVS_GIT "unknown"
#endif

namespace gvs {

inline std::string cpuModel() {
#ifdef __APPLE__
    char buf[256];
    size_t len = sizeof buf;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) return buf;
#else
    std::ifstream f("/proc/cpuinfo");
    for (std::string line; std::getline(f, line);)
        if (line.rfind("model name", 0) == 0) {
            const size_t c = line.find(':');
            if (c != std::string::npos) return line.substr(line.find_first_not_of(' ', c + 1));
        }
#endif
    return "unknown";
}

inline unsigned logicalCpus() { return std::thread::hardware_concurrency(); }

inline std::string hostName() {
    std::ifstream f("/proc/sys/kernel/hostname");
    std::string h;
    std::getline(f, h);
    return h.empty() ? "unknown" : h;
}

inline std::string utcNow() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

inline const char* gitVersion() { return GVS_GIT; }

}  // namespace gvs
