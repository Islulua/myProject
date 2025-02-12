project(output-replacer)

# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)
# Add the executable
add_executable(output-replacer OutputReplacer.cpp)

# Link the LLVM support library
target_link_libraries(output-replacer LLVMSupport MLIRPass MLIRSupport MLIRIR)