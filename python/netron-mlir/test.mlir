module {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg0 : i32
    %2 = arith.addi %1, %arg1 : i32
    return %2 : i32
  }
}
