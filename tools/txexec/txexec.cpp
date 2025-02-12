#include "./txexec.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>

namespace cl = llvm::cl;
namespace fs = std::filesystem;
using namespace my_project::tools;

namespace my_project {
namespace tools {
    TxExecContext::TxExecContext(std::string modulePath) : modulePath_(modulePath) {
        loadModule();
        initialize();
    }

    TxExecContext::~TxExecContext() {
        cleanup();
    }

    bool TxExecContext::initialize() {
        return true;
    }

    bool TxExecContext::execute() {
        // Execute the main functionality of the tool
        return true;
    }

    void TxExecContext::cleanup() {
        // Clean up resources
    }

    void TxExecContext::loadModule() {
        // Load module from filePath
    }

} // namespace tools
} // namespace my_project


static cl::opt<std::string> inputMlirFile(
    cl::Positional, cl::desc("<input mlir file>"), cl::Required,
    cl::init(""), cl::value_desc("filename"));



static std::vector<std::shared_ptr<TxExecContext>> getContexts(std::string modulePath) {
    // modulePath 可能是以一个文件路径或者一个以*结尾的目录，如果以*结尾需要遍历目录下的所有.mlir后缀的文件， 对每一个文件创建一个TxExecContext对象
    std::vector<std::shared_ptr<TxExecContext>> contexts;
    std::vector<std::string> files;
    if (modulePath.back() == '*') {
        // 遍历目录下的所有.mlir后缀的文件

        for (const auto &entry : fs::directory_iterator(modulePath.substr(0, modulePath.size() - 1))) {
            if (entry.path().extension() == ".mlir") {
                files.push_back(entry.path().string());
            }
        }

        for (const auto &file : files) {
            contexts.push_back(std::make_shared<TxExecContext>(file));
        }

        return contexts;
    } else {
        files.push_back(modulePath);
    }
}



int main(int argc, char *argv[]) {
    cl::ParseCommandLineOptions(argc, argv, "TxExec\n");
    llvm::errs() << "// ===----------------------------------------------------------------------===//\n";
    llvm::errs() << "//                                   [TxExec]\n";
    llvm::errs() << "// ===----------------------------------------------------------------------===//\n";

    TxExecContext exec(inputMlirFile);
    return 0;
}
