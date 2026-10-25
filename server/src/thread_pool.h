#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gvs {

class ThreadPool {
public:
    explicit ThreadPool(unsigned threads)
        : n_(threads ? threads : std::max(1u, std::thread::hardware_concurrency())) {
        for (unsigned i = 1; i < n_; ++i) workers_.emplace_back([this, i] { loop(i); });
    }

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
