# 添加子目录
add_subdirectory(design-mode)

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加编译选项
add_compile_options(-Wall -Wextra -Werror)

# 如果是Debug模式，添加调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
endif()

# 查找依赖包
find_package(Threads REQUIRED)

# 设置包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 如果src目录存在且包含源文件，才创建learn_examples目标
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    # 收集源文件
    file(GLOB_RECURSE LEARN_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    )
    
    if(LEARN_SOURCES)
        # 创建可执行文件
        add_executable(learn_examples ${LEARN_SOURCES})

        # 链接依赖库
        target_link_libraries(learn_examples
            PRIVATE
            Threads::Threads
            ${LLVM_LIBS}
        )
    endif()
endif()
