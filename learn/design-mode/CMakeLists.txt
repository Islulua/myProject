# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 收集源文件
file(GLOB DESIGN_PATTERN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# 排除main.cpp
list(FILTER DESIGN_PATTERN_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

# 创建设计模式库
add_library(design_patterns STATIC
    ${DESIGN_PATTERN_SOURCES}
)

# 设置包含目录
target_include_directories(design_patterns
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 添加编译选项
target_compile_options(design_patterns
    PRIVATE
    -Wall
    -Wextra
    -Werror
)

# 创建示例可执行文件
add_executable(pattern_examples
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

# 链接设计模式库
target_link_libraries(pattern_examples
    PRIVATE
    design_patterns
)

# 添加测试
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
