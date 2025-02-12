#ifndef SINGLETON_H
#define SINGLETON_H

#include <iostream>
#include <mutex>
#include <atomic>
#include <memory>

// 1. 基础的Meyer's Singleton - 最简单的实现
class BasicSingleton {
public:
    static BasicSingleton& getInstance();
    void doSomething();
    
    BasicSingleton(const BasicSingleton&) = delete;
    BasicSingleton& operator=(const BasicSingleton&) = delete;

private:
    BasicSingleton() noexcept = default;
    ~BasicSingleton() noexcept = default;
};

// 2. Double-Checked Locking Pattern - 用于需要延迟初始化且对性能敏感的场景
class DCLSingleton {
public:
    static DCLSingleton* getInstance();
    void doSomething();
    
    DCLSingleton(const DCLSingleton&) = delete;
    DCLSingleton& operator=(const DCLSingleton&) = delete;

private:
    DCLSingleton() noexcept = default;
    ~DCLSingleton() noexcept = default;
    static std::atomic<DCLSingleton*> instance;
    static std::mutex mutex;
};

// 3. Call-Once Singleton - 用于需要保证初始化只执行一次的场景
class CallOnceSingleton {
private:
    friend struct std::default_delete<CallOnceSingleton>; // 允许unique_ptr访问析构函数
    
public:
    static CallOnceSingleton& getInstance();
    void doSomething();
    
    CallOnceSingleton(const CallOnceSingleton&) = delete;
    CallOnceSingleton& operator=(const CallOnceSingleton&) = delete;

private:
    CallOnceSingleton() noexcept = default;
    ~CallOnceSingleton() noexcept = default;
    
    static std::unique_ptr<CallOnceSingleton> instance;
    static std::once_flag flag;
};

// 4. Atomic Singleton - 用于需要细粒度内存序控制的场景
class AtomicSingleton {
public:
    static AtomicSingleton& getInstance();
    void doSomething();
    
    AtomicSingleton(const AtomicSingleton&) = delete;
    AtomicSingleton& operator=(const AtomicSingleton&) = delete;

private:
    AtomicSingleton() noexcept = default;
    ~AtomicSingleton() noexcept = default;
};

#endif // SINGLETON_H 