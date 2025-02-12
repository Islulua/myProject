


#include <cstddef>
#include <map>
#include <vector>
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;


static LLVMContext context;
static Module *module = new Module("my compiler", context);

class DynamicLinearScanMemAlloc {
public:
    DynamicLinearScanMemAlloc(int start) : start_(start) {}
    void allocMem(mlir::Value value);
    void allocMem(mlir::Operation* op);
    int getMemSize(mlir::Value value);
    size_t getTotalMemSize() {
        return  (registerStatus_.size() + 1) * UNIT_SIZE;
    }
private:
    int UNIT_SIZE = 256;
    int start_;
    std::map<mlir::Value, size_t> valueSize_;
    std::map<mlir::Value, size_t> valueStart_;
    std::map<mlir::Value, size_t> valueOffset_;
    std::vector<int> registerStatus_;
};

void DynamicLinearScanMemAlloc::allocMem(mlir::Value value) {
    size_t valueSize = (getMemSize(value) + UNIT_SIZE) / UNIT_SIZE;
    valueSize_[value] = valueSize;

    int startReg = -1;
    for (size_t reg = 0; reg < registerStatus_.size(); reg++) {
        bool hasSpace = true;
        for (size_t i = 0; i < registerStatus_.size(); i ++) {
            if (reg + i >= registerStatus_.size() || registerStatus_[reg + i] != -1) {
                hasSpace = false;
                break;
            }
        }
        if (hasSpace) {
            startReg = reg;
            break;
        }
    }

    if (startReg == -1) {
        startReg = registerStatus_.size();
        registerStatus_.resize(startReg + valueSize, -1);
    }

    for (size_t i = 0; i < valueSize; i++) {
        registerStatus_[startReg + i] = 1;
    }
    valueStart_[value] = startReg;
    valueOffset_[value] = start_ + startReg * UNIT_SIZE;
}

int DynamicLinearScanMemAlloc::allocMem(mlir::Operation* op) {
    mlir::Liveness liveness(op);
    op->walk([&](mlir::Block *block) {
        for (mlir::Operation &op : *block) {
            for (mlir::Value result : op.getResults()) {
                allocMem(result);
            }
            for (auto operand : op.getOperands()) {
                if (liveness.isDeadAfter(operand, &op)) {
                    for (size_t index = valueStart_[operand]; index < valueSize_[operand]; index ++) {
                        registerStatus_[index] = -1;
                    }
                }
            }
        }
    });
    return 0;
}