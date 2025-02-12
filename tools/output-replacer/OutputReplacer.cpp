// ===----------------------------------------------------------------------===//
//
// This file is part of the output-replacer tool.
//
// ===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include <iostream>
#include "llvm/Support/CommandLine.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace cl = llvm::cl;

cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required);

namespace output_replacer {

struct OutputReplacerPass : public mlir::PassWrapper<OutputReplacerPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() final;
    void setNameToOp();

    std::map<std::string, mlir::Operation*> nameToOp_;
};

std::unique_ptr<mlir::Pass> createOutputReplacerPass() {
    return std::unique_ptr<mlir::Pass>(new output_replacer::OutputReplacerPass());
}

int getMLIRFromFile(mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext &context, std::string file) {
    return 0;
}

void OutputReplacerPass::setNameToOp() {
    std::string prefix = "tx8be.";
    auto funcOp = getOperation();
    std::unordered_map<std::string, int> counter;
    funcOp->walk([&](mlir::Operation *op) {
        std::string name = op->getName().getStringRef().str().substr(prefix.length());
        if (counter.find(name) == counter.end()) {
            counter[name] = 0;
        }
        name += "_" + std::to_string(counter[name]++);
        nameToOp_[name] = op;
        op->setAttr("name", mlir::StringAttr::get(op->getContext(), name));
    });
}

void saveIRToFile(mlir::Operation* op, std::string name) {
    name += ".mlir";
    return;
}

void OutputReplacerPass::runOnOperation() {
    setNameToOp();
    saveIRToFile(getOperation(), "temp");

    std::unordered_map<mlir::Operation*, int> opToIndex;

    while (false) {
        std::string opName;
        std::cout << "Enter operation name (or -1 to exit): ";
        std::cin >> opName;
        if (opName == "-1") {
            break;
        }

        auto it = nameToOp_.find(opName);
        if (it == nameToOp_.end()) {
            std::cout << "Operation not found. Try again." << std::endl;
            continue;
        }

        mlir::Operation *op = it->second;
        int numResults = op->getNumResults();
        if (numResults == 1) {
            opToIndex[op] = 0;
        } else if (numResults > 1) {
            int index;
            std::cout << "Operation has " << numResults << " results. Enter result index: ";
            std::cin >> index;
            if (index >= 0 && index < numResults) {
                opToIndex[op] = index;
            } else {
                std::cout << "Invalid index. Try again." << std::endl;
                continue;
            }
        } else {
            std::cout << "Operation has no results. Try again." << std::endl;
        }
    }
    

}

} // namespace output_replacer

using namespace output_replacer;



int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "Output Replacer Tool\n");
    std::cout << "// ===----------------------------------------\n";
    std::cout << "// Output Replacer Tool\n";
    std::cout << "// ===----------------------------------------\n";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::MLIRContext context;

    if (int error = getMLIRFromFile(module, context, inputFile)) {
        return error;
    }

    mlir::PassManager pm(&context);
    pm.addPass(createOutputReplacerPass());

    if (mlir::failed(pm.run(module.get()))) {
        return -1;
    }
    return 0;
}