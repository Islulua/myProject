/**
 * @file ThreadPool.h
 * @brief Header file for the ThreadPool class.
 *
 * This file contains the declaration of a high-efficiency thread pool.
 */
// 实现一个高效线程池

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace my_project {
namespace support {

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);
    void stop();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queueMutex_;
    std::condition_variable condition_;
    bool stop_;
    bool joined_; // 新增的标志
};

} // namespace support
} // namespace my_project

#endif // THREAD_POOL_H