#include "../include/singleton.h"

// 1. Basic Meyer's Singleton Implementation
// 优点: 简单，线程安全
// 缺点: 无法控制构造时机
// 使用场景: 一般场景，不需要特殊控制的情况
BasicSingleton& BasicSingleton::getInstance() {
    static BasicSingleton instance;
    return instance;
}

void BasicSingleton::doSomething() {
    std::cout << "BasicSingleton doing something" << std::endl;
}

// 2. Double-Checked Locking Pattern Implementation
// 优点: 可以控制构造时机，性能好
// 缺点: 实现复杂，需要小心处理内存序
// 使用场景: 性能关键场景，需要延迟初始化
std::atomic<DCLSingleton*> DCLSingleton::instance{nullptr};
std::mutex DCLSingleton::mutex;

DCLSingleton* DCLSingleton::getInstance() {
    DCLSingleton* tmp = instance.load(std::memory_order_acquire);
    if (!tmp) {
        std::lock_guard<std::mutex> lock(mutex);
        tmp = instance.load(std::memory_order_relaxed);
        if (!tmp) {
            tmp = new DCLSingleton();
            instance.store(tmp, std::memory_order_release);
        }
    }
    return tmp;
}

void DCLSingleton::doSomething() {
    std::cout << "DCLSingleton doing something" << std::endl;
}

// 3. Call-Once Pattern Implementation
// 优点: 保证初始化只执行一次，线程安全，自动内存管理
// 缺点: 可能比DCL稍慢
// 使用场景: 需要确保初始化只执行一次的场景
std::unique_ptr<CallOnceSingleton> CallOnceSingleton::instance;
std::once_flag CallOnceSingleton::flag;

CallOnceSingleton& CallOnceSingleton::getInstance() {
    std::call_once(flag, []() {
        instance.reset(new CallOnceSingleton());
    });
    return *instance;
}

void CallOnceSingleton::doSomething() {
    std::cout << "CallOnceSingleton doing something" << std::endl;
}

// 4. Atomic Singleton Implementation
// 优点: 可以精确控制内存序
// 缺点: 实现相对复杂
// 使用场景: 需要细粒度内存序控制的场景
AtomicSingleton& AtomicSingleton::getInstance() {
    static AtomicSingleton instance;
    std::atomic_thread_fence(std::memory_order_acquire);
    return instance;
}

void AtomicSingleton::doSomething() {
    std::cout << "AtomicSingleton doing something" << std::endl;
} 