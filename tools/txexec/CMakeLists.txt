project(txexec)

# Add the executable
add_executable(txexec txexec.cpp)

# Link the LLVM support library
target_link_libraries(txexec LLVMSupport MLIRPass MLIRSupport MLIRIR)