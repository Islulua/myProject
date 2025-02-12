#include <iostream>
#include <mutex>
#include <memory>

class Singleton {
public:
    // 获取单例实例
    static Singleton& getInstance() {
        static Singleton instance;  // C++11保证这是线程安全的
        return instance;
    }

    // 删除拷贝构造和赋值操作
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    // 示例方法
    void doSomething() {
        std::cout << "Singleton is doing something" << std::endl;
    }

private:
    // 私有构造函数
    Singleton() {
        std::cout << "Singleton is constructed" << std::endl;
    }
    
    // 私有析构函数
    ~Singleton() {
        std::cout << "Singleton is destroyed" << std::endl;
    }
};

// 使用示例
int main() {
    // 获取单例实例
    Singleton& instance1 = Singleton::getInstance();
    instance1.doSomething();

    // 再次获取单例实例（实际上是同一个实例）
    Singleton& instance2 = Singleton::getInstance();
    instance2.doSomething();

    // 验证是同一个实例
    std::cout << "Are instances the same? " 
              << (&instance1 == &instance2 ? "Yes" : "No") 
              << std::endl;

    return 0;
}
