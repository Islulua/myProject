#ifndef MY_PROJECT_PASSES_AUTOMERGE_H
#define MY_PROJECT_PASSES_AUTOMERGE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace myproject {

std::unique_ptr<Pass> createAutoMergePass();

#define GEN_PASS_REGISTRATION
#include "MyProject/Passes/Passes.h.inc"

} // namespace myproject
} // namespace mlir

#endif 