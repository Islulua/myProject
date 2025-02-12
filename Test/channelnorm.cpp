// 用于改变数据的排列顺序
// Tensor: 多维数组，顺序排列
// NTensor: 对维数组，顺序排列，第一维度是Batch
// Cx: Tensor模式的NPU对齐模式，channel优先
// NCx: NTensor模式的NPU对齐模式，channel优先

#include <cstddef>
#include <iostream>

// 1024x1024
// 1024x16x64

void TensorNorm(char* input, char* output, std::vector<size_t> shape) {
    // NCx(HWC)
    for (int n = 0; n < shape[0]; n ++) {
        for (int cx = 0; cx < shape[1]; cx ++) {
            size_t srcIdx = n * shape[1] * shape[2] + cx * shape[2];
            
        }
    }
}