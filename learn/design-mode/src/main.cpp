#include "../include/singleton.h"
#include <thread>
#include <vector>

void testSingletons() {
    // 1. Basic Meyer's Singleton
    auto& basic1 = BasicSingleton::getInstance();
    auto& basic2 = BasicSingleton::getInstance();
    std::cout << "Basic Singleton same instance: " 
              << (&basic1 == &basic2 ? "yes" : "no") << std::endl;
    
    // 2. Double-Checked Locking Pattern
    auto* dcl1 = DCLSingleton::getInstance();
    auto* dcl2 = DCLSingleton::getInstance();
    std::cout << "DCL Singleton same instance: " 
              << (dcl1 == dcl2 ? "yes" : "no") << std::endl;
    
    // 3. Call-Once Pattern
    auto& callOnce1 = CallOnceSingleton::getInstance();
    auto& callOnce2 = CallOnceSingleton::getInstance();
    std::cout << "Call-Once Singleton same instance: " 
              << (&callOnce1 == &callOnce2 ? "yes" : "no") << std::endl;
    
    // 4. Atomic Pattern
    auto& atomic1 = AtomicSingleton::getInstance();
    auto& atomic2 = AtomicSingleton::getInstance();
    std::cout << "Atomic Singleton same instance: " 
              << (&atomic1 == &atomic2 ? "yes" : "no") << std::endl;
}

int main() {
    // 单线程测试
    testSingletons();
    
    // 多线程测试
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(testSingletons);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return 0;
} 