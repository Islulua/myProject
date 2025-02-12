cmake_minimum_required(VERSION 3.20)

# 添加GTest库

add_executable(ThreadPoolTest ThreadPoolTest.cpp)
target_link_libraries(ThreadPoolTest gtest gtest_main pthread ThreadPool)
