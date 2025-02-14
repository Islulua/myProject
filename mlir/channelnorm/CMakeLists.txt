# 检查是否为顶层项目
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.10)
    project(channelnorm)
endif()

# 添加可执行文件
add_executable(channelnorm main.cpp)

# 设置C++标准 (根据需要调整版本)
target_compile_features(channelnorm PRIVATE cxx_std_11)