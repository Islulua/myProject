#include "Support/ThreadPool/ThreadPool.h"

// ...existing code...

namespace my_project {
namespace support {

ThreadPool::ThreadPool(size_t numThreads) : stop_(false), joined_(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queueMutex_);
                    this->condition_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                    if (this->stop_ && this->tasks_.empty())
                        return;
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        if (stop_)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks_.emplace(std::move(task));
    }
    condition_.notify_one();
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        if (joined_) return; // 如果已经 join 过，直接返回
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_)
        worker.join();
    joined_ = true; // 设置标志
}

} // namespace support
} // namespace my_project

// ...existing code...
