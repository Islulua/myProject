#include <mlir/Dialect/Func/IR/FuncOps.h>




class NopTreeGenerator {
public:
    NopTreeGenerator(mlir::func::FuncOp func) {
        initTrees(func);
    }
    
private:
    void initTrees(mlir::func::FuncOp func) {
        func.walk([&](tx8be::GroupOp groupOp) {
            
        });
    }
    
    std::vector<std::shared_ptr<NopTree>> trees;
};