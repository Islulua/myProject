#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

// 前向声明
class TensorNormalizer;
using Shape = std::vector<int>;

class TensorNormalizer {
public:
    // 修改为静态常量
    static const inline std::vector<std::string> VALID_TYPES = {"int8", "uint8"};
    static const inline std::vector<std::string> VALID_LAYOUTS = {"Tensor", "NTensor"};

    // 获取对齐基数
    static int getAlignBase(int channel, const std::string& dtype) {
        const std::vector<std::pair<int, int>> alignBases = {
            {0, 4}, {4, 8}, {8, 16}, {16, 32}, {32, 64}
        };
        int base = 64;

        if (channel > 64 && 
            std::find(VALID_TYPES.begin(), VALID_TYPES.end(), dtype) != VALID_TYPES.end()) {
            return 128;
        }

        for (const auto& [threshold, value] : alignBases) {
            if (channel > threshold) {
                base = value;
            }
        }

        return channel > 0 ? base : 0;
    }

private:
    // 合并中间维度
    static int mergeMidDims(const Shape& shape) {
        if (shape.size() <= 2) return 1;
        
        return std::accumulate(shape.begin() + 1, shape.end() - 1, 1, 
                             std::multiplies<int>());
    }

    // 辅助函数：计算张量总大小
    static int calcTotalSize(const Shape& shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, 
                             std::multiplies<int>());
    }

public:
    // 标准化张量
    template<typename T>
    std::vector<T> normalizeTensor(const std::vector<T>& inputData, 
                                  const Shape& shape, 
                                  const std::string& dtype) {
        // 添加输入验证
        if (inputData.size() != calcTotalSize(shape)) {
            throw std::runtime_error("输入数据大小与shape不匹配");
        }

        int base = getAlignBase(shape.back(), dtype);
        if (shape.back() % base != 0) {
            throw std::runtime_error("最后一维度必须是base的整数倍");
        }

        int cx = shape.back() / base;
        int mergedDims = mergeMidDims(shape);
        
        // 重塑和转置操作
        std::vector<T> result(inputData.size());
        int totalSize = calcTotalSize(shape);
        
        // 实现真正的转置操作 [N, 1, M, Cx, base] -> [N, Cx, M, 1, base]
        for (int n = 0; n < shape[0]; ++n) {
            for (int m = 0; m < mergedDims; ++m) {
                for (int c = 0; c < cx; ++c) {
                    for (int b = 0; b < base; ++b) {
                        // 源索引 [n, 1, m, c, b]
                        int srcIdx = (((n * 1 + 0) * mergedDims + m) * cx + c) * base + b;
                        
                        // 目标索引 [n, c, m, 1, b]
                        int dstIdx = (((n * cx + c) * mergedDims + m) * 1 + 0) * base + b;
                        
                        result[dstIdx] = inputData[srcIdx];
                    }
                }
            }
        }

        return result;
    }

    // 反标准化张量
    template<typename T>
    std::vector<T> denormalizeTensor(const std::vector<T>& inputData, 
                                    const Shape& shape, 
                                    const std::string& dtype) {
        // 添加输入验证
        if (inputData.size() != calcTotalSize(shape)) {
            throw std::runtime_error("输入数据大小与shape不匹配");
        }

        int base = getAlignBase(shape.back(), dtype);
        if (shape.back() % base != 0) {
            throw std::runtime_error("最后一维度必须是base的整数倍");
        }

        int cx = shape.back() / base;
        int mergedDims = mergeMidDims(shape);
        
        // 重塑和转置操作
        std::vector<T> result(inputData.size());
        int totalSize = calcTotalSize(shape);
        
        // 实现反向转置操作 [N, Cx, M, 1, base] -> [N, 1, M, Cx, base]
        for (int n = 0; n < shape[0]; ++n) {
            for (int m = 0; m < mergedDims; ++m) {
                for (int c = 0; c < cx; ++c) {
                    for (int b = 0; b < base; ++b) {
                        // 源索引 [n, c, m, 1, b]
                        int srcIdx = (((n * cx + c) * mergedDims + m) * 1 + 0) * base + b;
                        
                        // 目标索引 [n, 1, m, c, b]
                        int dstIdx = (((n * 1 + 0) * mergedDims + m) * cx + c) * base + b;
                        
                        result[dstIdx] = inputData[srcIdx];
                    }
                }
            }
        }

        return result;
    }

    // 处理张量
    template<typename T>
    std::vector<T> processTensor(const std::vector<T>& inputData,
                                const Shape& shape,
                                const std::string& dtype,
                                bool normalize = true,
                                const std::string& layout = "Tensor") {
        // 验证布局类型
        if (std::find(VALID_LAYOUTS.begin(), VALID_LAYOUTS.end(), layout) 
            == VALID_LAYOUTS.end()) {
            throw std::runtime_error("Invalid layout type");
        }

        // 处理形状
        Shape newShape = shape;
        if (layout == "Tensor") {
            newShape.insert(newShape.begin(), 1);
        }

        // 根据normalize标志选择操作
        return normalize ? 
               normalizeTensor(inputData, newShape, dtype) :
               denormalizeTensor(inputData, newShape, dtype);
    }

    // 添加打印函数
    template<typename T>
    static void printTensorAsDataFrame(const std::vector<T>& tensor,
                                     const Shape& shape,
                                     const std::string& dtype,
                                     const std::string& name = "Tensor") {
        if (shape.size() < 2) {
            throw std::runtime_error("Shape must have at least 2 dimensions");
        }

        int base = getAlignBase(shape.back(), dtype);
        int lastDim = shape.back();
        
        // 计算除最后一维外的所有维度组合
        std::vector<int> indices(shape.size() - 1, 0);
        int totalRows = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            totalRows *= shape[i];
        }

        // 打印表头
        std::cout << "\n" << name << " DataFrame:\n";
        
        // 打印列标题
        std::cout << "dims\t";
        for (int i = 0; i < lastDim; ++i) {
            std::cout << i / base << "_" << i % base << "\t";
        }
        std::cout << "\n";

        // 打印数据
        for (int row = 0; row < totalRows; ++row) {
            // 打印行索引
            std::cout << "(";
            for (size_t i = 0; i < indices.size(); ++i) {
                std::cout << indices[i];
                if (i < indices.size() - 1) std::cout << ",";
            }
            std::cout << ")\t";

            // 打印数据
            for (int col = 0; col < lastDim; ++col) {
                int index = row * lastDim + col;
                if (index < tensor.size()) {
                    std::cout << tensor[index] << "\t";
                }
            }
            std::cout << "\n";

            // 更新索引
            for (int i = indices.size() - 1; i >= 0; --i) {
                indices[i]++;
                if (indices[i] < shape[i]) break;
                indices[i] = 0;
            }
        }
        std::cout << std::endl;
    }
};

// 主函数
int main() {
    try {
        TensorNormalizer normalizer;
        std::string dataType = "fp16";

        // 创建示例数据
        std::vector<int> inputData(1024);
        std::iota(inputData.begin(), inputData.end(), 0);  // 填充0到1023
        Shape inputShape = {1, 4, 256};
        Shape tempShape = {1, 4, 4, 64};

        // 处理张量
        auto output = normalizer.processTensor(inputData, inputShape, dataType, true, "Tensor");
        auto deOutput = normalizer.processTensor(output, inputShape, dataType, false, "Tensor");

        // 打印结果
        TensorNormalizer::printTensorAsDataFrame(inputData, inputShape, dataType, "Input");
        TensorNormalizer::printTensorAsDataFrame(output, inputShape, dataType, "Output");
        TensorNormalizer::printTensorAsDataFrame(deOutput, inputShape, dataType, "DeOutput");

        // 验证结果
        bool isValid = inputData == deOutput;
        std::cout << "Normalization and denormalization " 
                  << (isValid ? "successful" : "failed") << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
