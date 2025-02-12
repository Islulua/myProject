#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"),
                                      cl::Required);

static cl::opt<bool> dumpAllOp("all", cl::desc("Dump all ops"), cl::init(false));

static cl::opt<std::string> outFile("output", cl::desc("Output file"), cl::init(""));

static cl::opt<int> forwardDepth("fdepth", cl::desc("Forward depth"), cl::init(0));

static cl::opt<int> backwardDepth("bdepth", cl::desc("Backward depth"), cl::init(0));

int getMLIRFromFile(mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext & context, std::string file) {
    return 0;
}

struct GraphExtractionPass : public mlir::PassWrapper<GraphExtractionPass, OperationPass<ModuleOp>> {
    void runOnOperation() override final;

    void initPatterns() {}
    void matchGraph() {}
    void createDir(std::string path) {}
    void dumpAllOp();
};

std::unique_ptr<mlir::Pass> createGraphExtractionPass() {
    return std::unique_ptr<mlir::Pass>(new GraphExtractionPass());
}

llvm::SmallVector<Operation*> getForwardOps(Operation *op, int depth) {
    llvm::SmallVector<Operation*> ret;
    if (depth == 0) {
        return ret;
    }
    for (auto operand : op->getOperands()) {
        auto definingOp = operand.getDefiningOp();
        if (definingOp) {
            ret.push_back(definingOp);
            auto sub = getForwardOps(definingOp, depth - 1);
            ret.insert(ret.end(), sub.begin(), sub.end());
        }
    }
    return ret;
}

llvm::SmallVector<Operation*> getBackwardOps(Operation *op, int depth) {
    llvm::SmallVector<Operation*> ret;
    if (depth == 0) {
        return ret;
    }
    for (auto &use : op->getUses()) {
        auto user = use.getOwner();
        ret.push_back(user);
        auto sub = getBackwardOps(user, depth - 1);
        ret.insert(ret.end(), sub.begin(), sub.end());
    }
    return ret;
}

void GraphExtractionPass::dumpAllOp() {
    std::unordered_set<OpKey> hash;
    llvm::SmallVector<llvm::SmallVector<Operation*>> ret;
    for (auto group : groups_) {
        if (group.size() != 1) {
            continue;
        }
        auto op = group.front();
        OpKey key(group.front());

        if (hash.find(key) == hash.end()) {
            ret.push_back(group);
            hash.insert(key);
        }
    }
    auto funcOp = getOperation().getOps<mlir::func::FuncOp>();
    std:::string prefix = "/";
    for (auto group : ret) {
        auto op = group.front();

        std::string name = op->getName().getStringRef().str();
        std::string filename = prefix + name + ".mlir";
        std::error_code error;
        llvm::raw_fd_ostream file(filename, error);
        if (error) {
            llvm::errs() << "Error: " << error.message() << "\n";
            return;
        }
        for (auto op : group) {
            op->print(file);
        }
    }
}

void GraphExtractionPass::runOnOperation() {
    // initPatterns();
    // matchGraph();

    createDir(outFile);

    dumpAllOp();
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "MLIR subgraph extraction tool\n");
    llvm::errs() << "// ===----------------------------------------\n";
    llvm::errs() << "// MLIR Subgraph Extraction Tool\n";
    llvm::errs() << "// ===----------------------------------------\n";


    mlir::MLIRContext context;
    OwningOpRef<ModuleOp> module;
    if (int error = getMLIRFromFile(module, context, inputFile)) {
        return error;
    }



    return 0;
}