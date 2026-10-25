// Vulkan compute backend. The whole order table lives in GPU memory; each tick uploads only the changed
// cells and the instrument prices, recalculates the derived columns once, then evaluates every view with
// a batched 8-pass radix select (see shaders/). Views are split round-robin across all GPUs.
#include "engine.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "advance.spv.h"
#include "clear.spv.h"
#include "decide.spv.h"
#include "derive.spv.h"
#include "final.spv.h"
#include "hist.spv.h"
#include "hist0.spv.h"
#include "scatter.spv.h"
#include "scatter0.spv.h"
#include "upd.spv.h"

namespace gvs {

namespace {

#define VK_CHECK(expr)                                                                                  \
    do {                                                                                                \
        VkResult r_ = (expr);                                                                           \
        if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(#expr) + " failed: " + std::to_string(int(r_))); \
    } while (0)

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Keep in sync with shaders/common.glsl.
constexpr uint32_t WG = 256;
constexpr uint32_t STATE_STRIDE = 528;
constexpr uint32_t HIST_STRIDE = 512;
constexpr uint32_t RES_ROWS = 4;
constexpr uint32_t RES_PIVOT = RES_ROWS + MAX_LIMIT;
constexpr uint32_t RES_STRIDE = RES_PIVOT + MAX_GROUPS * PIVOT_STRIDE;
static_assert(RES_PIVOT == 260 && RES_STRIDE == 3844);
constexpr uint32_t PASSES = 8;
constexpr uint32_t DERIVE_GROUPS = 4096;

enum Bind : uint32_t { B_COLS, B_INST, B_PARAMS, B_STATE, B_HIST, B_CAND, B_OUT, B_PIVOT, B_DISP, B_UPD, B_RES, B_COUNT };
enum Kernel : uint32_t { K_UPD, K_DERIVE, K_CLEAR, K_HIST0, K_DECIDE, K_SCATTER0, K_ADVANCE, K_HIST, K_SCATTER, K_FINAL, K_COUNT };

struct Spv {
    const uint32_t* code;
    size_t bytes;
};
const Spv kSpv[K_COUNT] = {
    {spv_upd, sizeof(spv_upd)},           {spv_derive, sizeof(spv_derive)},   {spv_clear, sizeof(spv_clear)},
    {spv_hist0, sizeof(spv_hist0)},       {spv_decide, sizeof(spv_decide)},   {spv_scatter0, sizeof(spv_scatter0)},
    {spv_advance, sizeof(spv_advance)},   {spv_hist, sizeof(spv_hist)},       {spv_scatter, sizeof(spv_scatter)},
    {spv_final, sizeof(spv_final)},
};

struct PC {
    uint32_t n, m, nb, slotCap, viewBase, pass, srcHalf, cap, count;
};

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* map = nullptr;
};

struct Device {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::string name;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkDeviceSize maxRange = 0, maxAlloc = 0, heap = 0;
    double tsPeriodNs = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipeline pipes[K_COUNT]{};
    Buffer b[B_COUNT];
    Buffer staging;  // host visible: instrument arrays + view params, copied to the GPU each tick
    uint32_t n = 0, m = 0, slotCap = 0;
    std::vector<uint32_t> views;  // global indices of the views this device runs this tick
};

class GpuEngine final : public Engine {
public:
    explicit GpuEngine(const GpuConfig& cfg) : cfg_(cfg) {
        createInstance();
        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
        std::vector<VkPhysicalDevice> phys(count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, phys.data()));
        for (VkPhysicalDevice p : phys) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(p, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;  // e.g. llvmpipe
            if (devs_.size() >= cfg_.maxDevices) break;
            devs_.emplace_back();
            initDevice(devs_.back(), p);
        }
        if (devs_.empty()) throw std::runtime_error("no Vulkan GPU found");
    }

    ~GpuEngine() override {
        for (Device& d : devs_) {
            if (!d.dev) continue;
            vkDeviceWaitIdle(d.dev);
            for (Buffer& b : d.b) destroy(d, b);
            destroy(d, d.staging);
            for (VkPipeline p : d.pipes)
                if (p) vkDestroyPipeline(d.dev, p, nullptr);
            vkDestroyDescriptorPool(d.dev, d.dpool, nullptr);
            vkDestroyPipelineLayout(d.dev, d.layout, nullptr);
            vkDestroyDescriptorSetLayout(d.dev, d.dsl, nullptr);
            if (d.queries) vkDestroyQueryPool(d.dev, d.queries, nullptr);
            vkDestroyFence(d.dev, d.fence, nullptr);
            vkDestroyCommandPool(d.dev, d.cmdPool, nullptr);
            vkDestroyDevice(d.dev, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    std::string name() const override { return "gpu"; }

    std::vector<std::string> devices() const override {
        std::vector<std::string> v;
        for (const Device& d : devs_) v.push_back(d.name);
        return v;
    }

    void run(const Store& s, uint64_t tick, const std::vector<Update>& updates, const std::vector<ViewParams>& views,
             std::vector<ViewResult>& out, RunStats& stats) override {
        const auto t0 = Clock::now();
        const uint32_t D = uint32_t(devs_.size());
        for (Device& d : devs_) d.views.clear();
        for (uint32_t i = 0; i < views.size(); ++i) devs_[i % D].views.push_back(i);

        // A skipped tick (engine was inactive) or a reallocation means the GPU copy is stale.
        bool full = !synced_ || tick != lastTick_ + 1;
        for (Device& d : devs_) full |= ensure(d, s, uint32_t(d.views.size()), uint32_t(updates.size()));
        if (full)
            for (Device& d : devs_) fullUpload(d, s);

        for (Device& d : devs_) {
            char* stg = static_cast<char*>(d.staging.map);
            std::memcpy(stg, s.mid.data(), size_t(d.m) * 4);
            std::memcpy(stg + size_t(d.m) * 4, s.invBps.data(), size_t(d.m) * 4);
            auto* vp = reinterpret_cast<ViewParams*>(stg + size_t(d.m) * 8);
            for (size_t j = 0; j < d.views.size(); ++j) vp[j] = views[d.views[j]];
            const uint32_t nUpd = full ? 0 : uint32_t(updates.size());
            if (nUpd) std::memcpy(d.b[B_UPD].map, updates.data(), size_t(nUpd) * sizeof(Update));
            record(d, nUpd);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &d.cmd;
            VK_CHECK(vkResetFences(d.dev, 1, &d.fence));
            VK_CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
        }
        stats.uploadMs = msSince(t0);

        stats.deriveMs = stats.viewsMs = 0;
        out.resize(views.size());
        for (Device& d : devs_) {
            VK_CHECK(vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX));
            if (d.queries) {
                uint64_t ts[3];
                if (vkGetQueryPoolResults(d.dev, d.queries, 0, 3, sizeof ts, ts, sizeof(uint64_t),
                                          VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                    stats.deriveMs = std::max(stats.deriveMs, double(ts[1] - ts[0]) * d.tsPeriodNs / 1e6);
                    stats.viewsMs = std::max(stats.viewsMs, double(ts[2] - ts[1]) * d.tsPeriodNs / 1e6);
                }
            }
            const auto* res = static_cast<const uint32_t*>(d.b[B_RES].map);
            for (size_t j = 0; j < d.views.size(); ++j) {
                const uint32_t* r = res + j * RES_STRIDE;
                const ViewParams& p = views[d.views[j]];
                ViewResult& o = out[d.views[j]];
                o.total = r[0];
                o.rows.assign(r + RES_ROWS, r + RES_ROWS + std::min(r[1], MAX_LIMIT));
                o.pivot.resize(p.nGroups);
                for (uint32_t g = 0; g < p.nGroups; ++g) {
                    const uint32_t* c = r + RES_PIVOT + g * PIVOT_STRIDE;
                    auto i64 = [](const uint32_t* w) { return int64_t(uint64_t(w[0]) | (uint64_t(w[1]) << 32)); };
                    o.pivot[g] = {c[0], i64(c + 1), i64(c + 3), i64(c + 5)};
                }
            }
        }
        synced_ = true;
        lastTick_ = tick;
        stats.totalMs = msSince(t0);
    }

private:
    void createInstance() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "gpu_view_server";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
#ifdef __APPLE__
        const char* exts[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = exts;
#endif
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    void initDevice(Device& d, VkPhysicalDevice phys) {
        d.phys = phys;
        VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &m3;
        vkGetPhysicalDeviceProperties2(phys, &props2);
        const VkPhysicalDeviceProperties& props = props2.properties;
        d.name = props.deviceName;
        d.maxRange = props.limits.maxStorageBufferRange;
        d.maxAlloc = m3.maxMemoryAllocationSize;
        vkGetPhysicalDeviceMemoryProperties(phys, &d.memProps);
        for (uint32_t i = 0; i < d.memProps.memoryHeapCount; ++i)
            if (d.memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                d.heap = std::max(d.heap, d.memProps.memoryHeaps[i].size);

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qf.data());
        uint32_t fam = UINT32_MAX;
        for (uint32_t i = 0; i < qn && fam == UINT32_MAX; ++i)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) fam = i;
        if (fam == UINT32_MAX) throw std::runtime_error(d.name + ": no compute queue");

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = fam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
#ifdef __APPLE__
        const char* dexts[] = {"VK_KHR_portability_subset"};
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dexts;
#endif
        VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &d.dev));
        vkGetDeviceQueue(d.dev, fam, 0, &d.queue);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = fam;
        VK_CHECK(vkCreateCommandPool(d.dev, &pci, nullptr, &d.cmdPool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = d.cmdPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(d.dev, &cai, &d.cmd));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(d.dev, &fci, nullptr, &d.fence));

        if (qf[fam].timestampValidBits > 0 && props.limits.timestampPeriod > 0) {
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = 3;
            VK_CHECK(vkCreateQueryPool(d.dev, &qpi, nullptr, &d.queries));
            d.tsPeriodNs = props.limits.timestampPeriod;
        }

        VkDescriptorSetLayoutBinding binds[B_COUNT];
        for (uint32_t i = 0; i < B_COUNT; ++i)
            binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = B_COUNT;
        dli.pBindings = binds;
        VK_CHECK(vkCreateDescriptorSetLayout(d.dev, &dli, nullptr, &d.dsl));

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &d.dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(d.dev, &pli, nullptr, &d.layout));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, B_COUNT};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(d.dev, &dpi, nullptr, &d.dpool));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = d.dpool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &d.dsl;
        VK_CHECK(vkAllocateDescriptorSets(d.dev, &dai, &d.set));

        for (uint32_t k = 0; k < K_COUNT; ++k) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = kSpv[k].bytes;
            smi.pCode = kSpv[k].code;
            VkShaderModule mod;
            VK_CHECK(vkCreateShaderModule(d.dev, &smi, nullptr, &mod));
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = mod;
            cpi.stage.pName = "main";
            cpi.layout = d.layout;
            VK_CHECK(vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &d.pipes[k]));
            vkDestroyShaderModule(d.dev, mod, nullptr);
        }
    }

    uint32_t memType(const Device& d, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) {
        for (VkMemoryPropertyFlags f : {want, fallback})
            for (uint32_t i = 0; i < d.memProps.memoryTypeCount; ++i)
                if ((bits & (1u << i)) && (d.memProps.memoryTypes[i].propertyFlags & f) == f) return i;
        throw std::runtime_error(d.name + ": no suitable memory type");
    }

    enum Mem { DEVICE, HOST_WRITE, HOST_READ };

    void create(Device& d, Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, Mem mem) {
        destroy(d, b);
        size = std::max<VkDeviceSize>(size, 256);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(d.dev, &bi, nullptr, &b.buf));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(d.dev, b.buf, &req);
        constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            mem == DEVICE      ? memType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            : mem == HOST_READ ? memType(d, req.memoryTypeBits, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, host)
                               : memType(d, req.memoryTypeBits, host, host);
        if (vkAllocateMemory(d.dev, &ai, nullptr, &b.mem) != VK_SUCCESS) {
            vkDestroyBuffer(d.dev, b.buf, nullptr);
            b.buf = VK_NULL_HANDLE;
            throw std::runtime_error(d.name + ": out of memory allocating " + std::to_string(size >> 20) + " MB");
        }
        VK_CHECK(vkBindBufferMemory(d.dev, b.buf, b.mem, 0));
        if (mem != DEVICE) VK_CHECK(vkMapMemory(d.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
        b.size = size;
    }

    void destroy(Device& d, Buffer& b) {
        if (b.map) vkUnmapMemory(d.dev, b.mem);
        if (b.buf) vkDestroyBuffer(d.dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(d.dev, b.mem, nullptr);
        b = {};
    }

    // Grows buffers as needed. Returns true when the order table was (re)allocated and must be uploaded.
    bool ensure(Device& d, const Store& s, uint32_t nViews, uint32_t nUpd) {
        constexpr VkBufferUsageFlags storage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bool rebind = false, realloc = false;
        if (d.n != s.rows) {
            const uint32_t n = s.rows;
            const VkDeviceSize colsBytes = VkDeviceSize(NUM_COLS) * n * 4;
            if (colsBytes > d.maxRange || colsBytes > d.maxAlloc)
                throw std::runtime_error(d.name + ": " + std::to_string(n) + " rows exceed one buffer");
            create(d, d.b[B_COLS], colsBytes, storage, DEVICE);
            const VkDeviceSize perSlot = 2ull * n * 3 * 4;
            const VkDeviceSize budget = d.heap * 85 / 100 > colsBytes ? d.heap * 85 / 100 - colsBytes : 0;
            VkDeviceSize slots = std::min<VkDeviceSize>(cfg_.slots, budget / perSlot);
            slots = std::min(slots, std::min(d.maxRange, d.maxAlloc) / perSlot);
            if (slots == 0) throw std::runtime_error(d.name + ": not enough memory for " + std::to_string(n) + " rows");
            d.slotCap = uint32_t(slots);
            create(d, d.b[B_CAND], perSlot * slots, storage, DEVICE);
            create(d, d.b[B_STATE], VkDeviceSize(slots) * STATE_STRIDE * 4, storage, DEVICE);
            create(d, d.b[B_HIST], VkDeviceSize(slots) * HIST_STRIDE * 4, storage, DEVICE);
            create(d, d.b[B_OUT], VkDeviceSize(slots) * MAX_LIMIT * 2 * 4, storage, DEVICE);
            create(d, d.b[B_PIVOT], VkDeviceSize(slots) * MAX_GROUPS * PIVOT_STRIDE * 4, storage, DEVICE);
            create(d, d.b[B_DISP], 16, storage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, DEVICE);
            d.n = n;
            rebind = realloc = true;
        }
        if (d.m != s.instruments) {
            d.m = s.instruments;
            create(d, d.b[B_INST], VkDeviceSize(d.m) * 8, storage, DEVICE);
            rebind = true;
        }
        auto grow = [&](Buffer& b, VkDeviceSize need, VkBufferUsageFlags usage, Mem mem) {
            if (b.size >= need) return false;
            create(d, b, need + need / 2, usage, mem);
            return true;
        };
        const VkDeviceSize views = std::max(1u, nViews);
        grow(d.staging, VkDeviceSize(d.m) * 8 + views * sizeof(ViewParams), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST_WRITE);
        rebind |= grow(d.b[B_PARAMS], views * sizeof(ViewParams), storage, DEVICE);
        rebind |= grow(d.b[B_UPD], VkDeviceSize(std::max(1u, nUpd)) * sizeof(Update), storage, HOST_WRITE);
        rebind |= grow(d.b[B_RES], views * RES_STRIDE * 4, storage, HOST_READ);
        if (rebind) {
            VkDescriptorBufferInfo infos[B_COUNT];
            VkWriteDescriptorSet writes[B_COUNT];
            for (uint32_t i = 0; i < B_COUNT; ++i) {
                infos[i] = {d.b[i].buf, 0, VK_WHOLE_SIZE};
                writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                writes[i].dstSet = d.set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(d.dev, B_COUNT, writes, 0, nullptr);
        }
        return realloc;
    }

    template <class F>
    void oneShot(Device& d, F&& fn) {
        VK_CHECK(vkResetCommandBuffer(d.cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(d.cmd, &bi));
        fn(d.cmd);
        VK_CHECK(vkEndCommandBuffer(d.cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d.cmd;
        VK_CHECK(vkResetFences(d.dev, 1, &d.fence));
        VK_CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
        VK_CHECK(vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX));
    }

    void fullUpload(Device& d, const Store& s) {
        constexpr VkDeviceSize CHUNK = 64ull << 20;
        Buffer tmp;
        create(d, tmp, CHUNK, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST_WRITE);
        const VkDeviceSize bytes = VkDeviceSize(d.n) * 4;
        for (uint32_t c = 0; c < NUM_BASE_COLS; ++c) {
            for (VkDeviceSize off = 0; off < bytes; off += CHUNK) {
                const VkDeviceSize len = std::min(CHUNK, bytes - off);
                std::memcpy(tmp.map, reinterpret_cast<const char*>(s.cols[c].data()) + off, len);
                oneShot(d, [&](VkCommandBuffer cb) {
                    VkBufferCopy region{0, VkDeviceSize(c) * bytes + off, len};
                    vkCmdCopyBuffer(cb, tmp.buf, d.b[B_COLS].buf, 1, &region);
                });
            }
        }
        destroy(d, tmp);
    }

    static void barrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }

    void dispatch(Device& d, Kernel k, const PC& pc, uint32_t x, uint32_t y) {
        vkCmdBindPipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipes[k]);
        vkCmdPushConstants(d.cmd, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(d.cmd, x, y, 1);
        barrier(d.cmd);
    }

    void dispatchIndirect(Device& d, Kernel k, const PC& pc) {
        vkCmdBindPipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipes[k]);
        vkCmdPushConstants(d.cmd, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatchIndirect(d.cmd, d.b[B_DISP].buf, 0);
        barrier(d.cmd);
    }

    void record(Device& d, uint32_t nUpd) {
        const uint32_t nViews = uint32_t(d.views.size());
        VkCommandBuffer cb = d.cmd;
        VK_CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        if (d.queries) vkCmdResetQueryPool(cb, d.queries, 0, 3);

        const VkDeviceSize instBytes = VkDeviceSize(d.m) * 8;
        VkBufferCopy instCopy{0, 0, instBytes};
        vkCmdCopyBuffer(cb, d.staging.buf, d.b[B_INST].buf, 1, &instCopy);
        if (nViews) {
            VkBufferCopy pCopy{instBytes, 0, VkDeviceSize(nViews) * sizeof(ViewParams)};
            vkCmdCopyBuffer(cb, d.staging.buf, d.b[B_PARAMS].buf, 1, &pCopy);
        }
        barrier(cb);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.layout, 0, 1, &d.set, 0, nullptr);
        if (d.queries) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, d.queries, 0);

        PC pc{d.n, d.m, 0, d.slotCap, 0, 0, 0, d.n, nUpd};
        if (nUpd) dispatch(d, K_UPD, pc, (nUpd + WG - 1) / WG, 1);
        dispatch(d, K_DERIVE, pc, std::min((d.n + WG - 1) / WG, DERIVE_GROUPS), 1);
        if (d.queries) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d.queries, 1);

        for (uint32_t base = 0; base < nViews; base += d.slotCap) {
            pc.nb = std::min(d.slotCap, nViews - base);
            pc.viewBase = base;
            pc.pass = 0;
            dispatch(d, K_CLEAR, pc, 1, pc.nb);
            // x = slot, y = row group, so the batch's views walk the same rows together.
            dispatch(d, K_HIST0, pc, pc.nb, cfg_.groupsPerSlot);
            dispatch(d, K_DECIDE, pc, 1, pc.nb);
            dispatch(d, K_SCATTER0, pc, pc.nb, cfg_.groupsPerSlot);
            for (uint32_t p = 1; p < PASSES; ++p) {
                pc.pass = p;
                pc.srcHalf = (p - 1) & 1;
                dispatch(d, K_ADVANCE, pc, 1, 1);
                dispatchIndirect(d, K_HIST, pc);
                dispatch(d, K_DECIDE, pc, 1, pc.nb);
                dispatchIndirect(d, K_SCATTER, pc);
            }
            dispatch(d, K_FINAL, pc, 1, pc.nb);
        }
        if (d.queries) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d.queries, 2);

        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0,
                             nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(cb));
    }

    GpuConfig cfg_;
    VkInstance instance_ = VK_NULL_HANDLE;
    std::vector<Device> devs_;
    bool synced_ = false;
    uint64_t lastTick_ = 0;
};

}  // namespace

std::unique_ptr<Engine> makeGpuEngine(const GpuConfig& cfg, std::string& err) {
    try {
        return std::make_unique<GpuEngine>(cfg);
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
}

}  // namespace gvs
