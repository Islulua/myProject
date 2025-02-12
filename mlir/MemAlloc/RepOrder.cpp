// 重新排布func里面的算子顺序，最小化buffer的生命周期
// 尽量减少buffer的生命周期

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
void updateIR(mlir::func::FuncOp funcOp) {
    funcOp.walk([&](mlir::Operation* op){
        op->walk([&](mlir::Operation* op){
            // do something

        })

    });
}