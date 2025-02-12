cmake_minimum_required(VERSION 3.20)

project(myProject)

# 设置输出目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 设置 include 目录
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

# 添加LLVM的include目录
include_directories(${CMAKE_SOURCE_DIR}/llvm-project/llvm/include)
include_directories(${CMAKE_SOURCE_DIR}/llvm-project/mlir/include)
include_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/include)
include_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/tools/mlir/include)

# 添加Logger的include目录
include_directories(${CMAKE_SOURCE_DIR}/third_party/Logger/include)

# include_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/include)

# 使用 ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# 设置 LLVM 编译选项
set(LLVM_ENABLE_PROJECTS "llvm;mlir" CACHE STRING "LLVM projects to build")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable assertions")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI")
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard version")

# 设置clangd编译选项
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands")

# add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(llvm-project/llvm)
add_subdirectory(tools)
add_subdirectory(Test)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/Logger)
add_subdirectory(learn)
