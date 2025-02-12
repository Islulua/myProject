#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <vector>

using namespace mlir;


namespace tx8be_mlir {

class IRVerifyPass : public PassWrapper<IRVerifyPass, OperationPass<tx8be::SubgraphOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IRVerifyPass)
    IRVerifyPass() = default;

    void runOnOperation() ;
};

class NopNode {

};

// add reshape : root(add)
// concatenate concatentate concatentate root: (concatenate)
// add reshape 
// add slice add
// inputy reshape concatenate slice

// 地址复用树的基类
class NopTree {
public:
    enum NopMode {
        NopMode_None,
        NopMode_Reuse,
        NopMode_Split,
        NopMode_Merge
    };
    NopTree(mlir::Operation* op) : root(op) {}
    static std::shared_ptr<NopTree> getNopTree(mlir::Operation* op) {
        
        return mlir::dyn_cast<tx8be::GroupOp>(op);
    }
private:
    
    std::vector<mlir::Operation*> nodes;
    NopMode mode_{NopMode_None};
    mlir::Operation* root{nullptr};
};

concate

// 验证器的基类
class Verifyer {
public:
    virtual ~Verifyer() = default;
    virtual mlir::LogicalResult verify(tx8be::SubGraphOp subgraphOp) = 0;
};

class DdrVerifyer : public Verifyer {
    virtual mlir::LogicalResult verify(tx8be::SubGraphOp subgraphOp) override {
        return mlir::success();
    }
};

LogicalResult DdrVerifyer::Verifyer(tx8be::SubGraphOp subgraphOp) {
    subgraphOp.walk([](tx8be::GroupOp groupOp) {
        auto ddrAttr = ddrOp.getAttrOfType<mlir::IntegerAttr>("ddr");
        if (!ddrAttr) {
            return mlir::failure();
        }
        return mlir::success();
    });
    return mlir::success();
}

using VerifyerPtrs = std::vector<std::shared_ptr<Verifyer>>;

void populateVerifers(VerifyerPtrs& verifyers) {
    verifyers.push_back(std::make_shared<DdrVerifyer>());
    verifyers.push_back(std::make_shared<ShapeVerifyer>());
}

void IRVerifyPass::runOnOperation() {
    auto subGrpahOp = getOperation();


}


} // namespace tx8be_mlir