#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace gvs {

// Pins the calling thread to one CPU. Works on isolated CPUs (isolcpus) too. Linux only.
inline bool pinCurrentThread(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

// "0-11,24-35" -> {0..11, 24..35}; "all" -> every configured CPU. Empty on a parse error.
inline std::vector<int> parseCpuList(const std::string& s) {
    std::vector<int> out;
    if (s == "all") {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) out.push_back(int(i));
        return out;
    }
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string part = s.substr(i, j - i);
        const size_t dash = part.find('-');
        try {
            const int a = std::stoi(part.substr(0, dash));
            const int b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
            for (int c = a; c <= b; ++c) out.push_back(c);
        } catch (...) {
            return {};
        }
        i = j + 1;
    }
    return out;
}

class ThreadPool {
public:
    // threads = 0: one per CPU in `cpus`, or per hardware thread if `cpus` is empty.
    // With `cpus`, thread i (the caller is thread 0) is pinned to cpus[i % cpus.size()].
    explicit ThreadPool(unsigned threads, std::vector<int> cpus = {})
        : n_(threads ? threads
                     : cpus.empty() ? std::max(1u, std::thread::hardware_concurrency()) : unsigned(cpus.size())),
          cpus_(std::move(cpus)) {
        if (!cpus_.empty()) pinned_ = pinCurrentThread(cpus_[0]);
        for (unsigned i = 1; i < n_; ++i)
            workers_.emplace_back([this, i] {
                if (!cpus_.empty()) pinCurrentThread(cpus_[i % cpus_.size()]);
                loop(i);
            });
    }

    bool pinned() const { return pinned_; }

    ~ThreadPool() {
        {
            std::lock_guard l(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    unsigned size() const { return n_; }

    // Runs fn(index, threadId) for every index in [0, count). The caller works as thread 0.
    void parallelFor(size_t count, const std::function<void(size_t, unsigned)>& fn) {
        if (count == 0) return;
        {
            std::lock_guard l(m_);
            fn_ = &fn;
            count_ = count;
            next_.store(0);
            active_ = n_ - 1;
            ++gen_;
        }
        cv_.notify_all();
        work(0);
        std::unique_lock l(m_);
        done_.wait(l, [&] { return active_ == 0; });
        fn_ = nullptr;
    }

private:
    void work(unsigned tid) {
        for (size_t i; (i = next_.fetch_add(1)) < count_;) (*fn_)(i, tid);
    }

    void loop(unsigned tid) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock l(m_);
                cv_.wait(l, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
            }
            work(tid);
            std::lock_guard l(m_);
            if (--active_ == 0) done_.notify_one();
        }
    }

    unsigned n_;
    std::vector<int> cpus_;
    bool pinned_ = false;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(size_t, unsigned)>* fn_ = nullptr;
    size_t count_ = 0;
    std::atomic<size_t> next_{0};
    unsigned active_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace gvs
