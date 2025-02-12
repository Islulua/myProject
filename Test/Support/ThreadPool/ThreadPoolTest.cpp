#include <gtest/gtest.h>
#include "Support/ThreadPool/ThreadPool.h"

using namespace my_project::support;

TEST(ThreadPoolTest, EnqueueTask) {
    ThreadPool pool(4);
    std::atomic<int> counter(0);

    auto task = [&counter]() {
        counter++;
    };

    for (int i = 0; i < 10; ++i) {
        pool.enqueue(task);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    EXPECT_EQ(counter.load(), 10);
}

TEST(ThreadPoolTest, EnqueueAfterStop) {
    ThreadPool pool(4);
    pool.stop();

    EXPECT_THROW(pool.enqueue([] {}), std::runtime_error);
}
