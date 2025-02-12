#ifndef MY_PROJECT_PASSES
#define MY_PROJECT_PASSES

include "mlir/Pass/PassBase.td"

def AutoMerge : Pass<"auto-merge", "ModuleOp"> {
  let summary = "执行自动合并优化";
  let description = [{
    这个 pass 用于执行自动合并优化。
    
    详细描述...
  }];
}

#endif 