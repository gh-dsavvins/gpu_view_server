#include "engine.h"

namespace gvs {

int compareResults(const std::vector<ViewResult>& a, const std::vector<ViewResult>& b, std::string* first) {
    if (a.size() != b.size()) {
        if (first) *first = "view counts differ";
        return int(std::max(a.size(), b.size()));
    }
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        std::string why;
        if (a[i].total != b[i].total)
            why = "total " + std::to_string(a[i].total) + " vs " + std::to_string(b[i].total);
        else if (a[i].rows != b[i].rows)
            why = "viewport rows differ";
        else if (a[i].pivot != b[i].pivot)
            why = "pivot cells differ";
        if (why.empty()) continue;
        if (!bad && first) *first = "view " + std::to_string(i) + ": " + why;
        ++bad;
    }
    return bad;
}

#ifndef GVS_GPU
std::unique_ptr<Engine> makeGpuEngine(const GpuConfig&, std::string& err) {
    err = "built without GPU support (GVS_GPU=OFF)";
    return nullptr;
}
#endif

}  // namespace gvs
