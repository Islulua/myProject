//===- TensorNormalizer.cpp - Tensor Normalization Tool -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <iomanip>

namespace tensorNorm {

using Shape = std::vector<int>;

class TensorNormalizer {
public:
    static const inline std::vector<std::string> SHORT_TYPES   = {"int8", "uint8"};
    static const inline std::vector<std::string> VALID_LAYOUTS = {"Tensor", "NTensor"};

    static int getAlignBase(int channel, const std::string& dtype) {
        if (channel <= 0) return 0;
        if (channel > 64 && std::find(SHORT_TYPES.begin(), SHORT_TYPES.end(), dtype) != SHORT_TYPES.end()) {
            return 128;
        }

        const int thresholds[] = {4, 8, 16, 32, 64};
        for (int base : thresholds) {
            if (channel <= base) return base;
        }
        return 64;
    }

    static bool isValidLayout(const std::string& layout) {
        return std::find(VALID_LAYOUTS.begin(), VALID_LAYOUTS.end(), layout)
               != VALID_LAYOUTS.end();
    }

private:
    static int mergeMidDims(const Shape& shape) {
        if (shape.size() <= 2) return 1;
        return std::reduce(shape.begin() + 1, shape.end() - 1, 1, std::multiplies<int>());
    }

    static int calcTotalSize(const Shape& shape) {
        return std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

public:
    template<typename T>
    std::vector<T> normalizeTensor(const std::vector<T>& inputData,
                                  const Shape& shape,
                                  const std::string& dtype) {
        if (inputData.size() != calcTotalSize(shape)) {
            throw std::runtime_error("输入数据大小与shape不匹配");
        }

        int base = getAlignBase(shape.back(), dtype);
        if (shape.back() % base != 0) {
            throw std::runtime_error("最后一维度必须是base的整数倍");
        }

        int cx = shape.back() / base;
        int mergedDims = mergeMidDims(shape);
        std::vector<T> result(inputData.size());
        int totalSize = calcTotalSize(shape);

        for (int n = 0; n < shape[0]; ++n) {
            for (int m = 0; m < mergedDims; ++m) {
                for (int c = 0; c < cx; ++c) {
                    for (int b = 0; b < base; ++b) {
                        int srcIdx = (((n * 1 + 0) * mergedDims + m) * cx + c) * base + b;
                        int dstIdx = (((n * cx + c) * mergedDims + m) * 1 + 0) * base + b;
                        result[dstIdx] = inputData[srcIdx];
                    }
                }
            }
        }

        return result;
    }

    template<typename T>
    std::vector<T> denormalizeTensor(const std::vector<T>& inputData,
                                    const Shape& shape,
                                    const std::string& dtype) {
        if (inputData.size() != calcTotalSize(shape)) {
            throw std::runtime_error("输入数据大小与shape不匹配");
        }

        int base = getAlignBase(shape.back(), dtype);
        if (shape.back() % base != 0) {
            throw std::runtime_error("最后一维度必须是base的整数倍");
        }

        int cx = shape.back() / base;
        int mergedDims = mergeMidDims(shape);

        std::vector<T> result(inputData.size());
        int totalSize = calcTotalSize(shape);

        for (int n = 0; n < shape[0]; ++n) {
            for (int m = 0; m < mergedDims; ++m) {
                for (int c = 0; c < cx; ++c) {
                    for (int b = 0; b < base; ++b) {
                        int srcIdx = (((n * cx + c) * mergedDims + m) * 1 + 0) * base + b;
                        int dstIdx = (((n * 1 + 0) * mergedDims + m) * cx + c) * base + b;
                        result[dstIdx] = inputData[srcIdx];
                    }
                }
            }
        }

        return result;
    }

    template<typename T>
    std::vector<T> processTensor(const std::vector<T>& inputData,
                                const Shape& shape,
                                const std::string& dtype,
                                bool normalize = true,
                                const std::string& layout = "Tensor") {
        if (!isValidLayout(layout)) {
            throw std::runtime_error("Invalid layout type");
        }

        Shape newShape = shape;
        if (layout == "Tensor") {
            newShape.insert(newShape.begin(), 1);
        }

        return normalize ?
               normalizeTensor(inputData, newShape, dtype) :
               denormalizeTensor(inputData, newShape, dtype);
    }

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

        std::vector<int> indices(shape.size() - 1, 0);
        int totalRows = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            totalRows *= shape[i];
        }

        std::cout << "\n" << name << " DataFrame:\n";

        std::cout << "dims\t";
        for (int i = 0; i < lastDim; ++i) {
            std::cout << i / base << "_" << i % base << "\t";
        }
        std::cout << "\n";

        for (int row = 0; row < totalRows; ++row) {
            std::cout << "(";
            for (size_t i = 0; i < indices.size(); ++i) {
                std::cout << indices[i];
                if (i < indices.size() - 1) std::cout << ",";
            }
            std::cout << ")\t";

            for (int col = 0; col < lastDim; ++col) {
                int index = row * lastDim + col;
                if (index < tensor.size()) {
                    std::cout << tensor[index] << "\t";
                }
            }
            std::cout << "\n";

            for (int i = indices.size() - 1; i >= 0; --i) {
                indices[i]++;
                if (indices[i] < shape[i]) break;
                indices[i] = 0;
            }
        }
        std::cout << std::endl;
    }
};

class DataTypeUtils {
public:
    static size_t getTypeSize(const std::string& dtype) {
        static const std::unordered_map<std::string, size_t> TYPE_SIZES = {
            {"fp32", sizeof(float)},
            {"fp16", sizeof(uint16_t)},
            {"bf16", sizeof(uint16_t)},
            {"int8", sizeof(uint8_t)},
            {"uint8", sizeof(uint8_t)}
        };

        auto it = TYPE_SIZES.find(dtype);
        if (it == TYPE_SIZES.end()) {
            throw std::runtime_error("Unsupported data type: " + dtype);
        }
        return it->second;
    }

    static bool isValidDataType(const std::string& dtype) {
        static const std::vector<std::string> VALID_TYPES = {
            "fp32", "fp16", "bf16", "int8", "uint8"
        };
        return std::find(VALID_TYPES.begin(), VALID_TYPES.end(), dtype) != VALID_TYPES.end();
    }
};

class FileUtils {
public:
    template<typename T>
    static std::vector<T> readBinaryFile(const std::string& filename,
                                       const std::string& dtype) {
        if (!DataTypeUtils::isValidDataType(dtype)) {
            throw std::runtime_error("Invalid data type for file reading: " + dtype);
        }
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open input file: " + filename);
        }

        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<T> data(fileSize / sizeof(T));
        file.read(reinterpret_cast<char*>(data.data()), fileSize);

        return data;
    }

    template<typename T>
    static void writeBinaryFile(const std::string& filename,
                              const std::vector<T>& data) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open output file: " + filename);
        }
        file.write(reinterpret_cast<const char*>(data.data()),
                   data.size() * sizeof(T));
    }
};

struct ProgramOptions {
    std::string shape;
    std::string dtype = "fp16";
    std::string layout = "Tensor";
    std::string displayShape;
    std::string inputFile;
    std::string outputFile;
    bool normalize = true;
    bool printTensor = false;

    static ProgramOptions parse(int argc, char* argv[]) {
        ProgramOptions opts;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--shape" && i + 1 < argc) {
                opts.shape = argv[++i];
            } else if (arg == "--dtype" && i + 1 < argc) {
                opts.dtype = argv[++i];
            } else if (arg == "--layout" && i + 1 < argc) {
                opts.layout = argv[++i];
            } else if (arg == "--display-shape" && i + 1 < argc) {
                opts.displayShape = argv[++i];
            } else if (arg == "--input" && i + 1 < argc) {
                opts.inputFile = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                opts.outputFile = argv[++i];
            } else if (arg == "--normalize") {
                opts.normalize = true;
            } else if (arg == "--denormalize") {
                opts.normalize = false;
            } else if (arg == "--print" || arg == "-p") {
                opts.printTensor = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                         << "Options:\n"
                         << "  --shape SHAPE       张量形状，例如：1x4x256 或 1,4,256\n"
                         << "  --dtype TYPE        数据类型 [fp16, fp32, int8, uint8] (default: fp16)\n"
                         << "  --layout LAYOUT     张量布局类型 [Tensor, NTensor] (default: Tensor)\n"
                         << "  --display-shape SHAPE  临时形状用于显示，例如：1x4x4x64\n"
                         << "  --input FILE        输入二进制文件路径\n"
                         << "  --output FILE       输出二进制文件路径\n"
                         << "  --normalize         进行标准化处理（默认）\n"
                         << "  --denormalize       进行反标准化处理\n"
                         << "  --print             打印张量数据\n";
                std::exit(0);
            }
        }

        if (opts.shape.empty()) {
            throw std::runtime_error("--shape parameter is required");
        }
        if (!DataTypeUtils::isValidDataType(opts.dtype)) {
            throw std::runtime_error("Invalid data type: " + opts.dtype);
        }
        if (!TensorNormalizer::isValidLayout(opts.layout)) {
            throw std::runtime_error("Invalid layout: " + opts.layout);
        }
        return opts;
    }

    static void dumpOptions(const ProgramOptions& opts) {
        const int width = 15;
        std::cout << "\n=== Program Options ===" << std::endl;
        std::cout << std::left
                  << std::setw(width) << "Shape:"          << opts.shape << '\n'
                  << std::setw(width) << "Dtype:"          << opts.dtype << '\n'
                  << std::setw(width) << "Layout:"         << opts.layout << '\n'
                  << std::setw(width) << "DisplayShape:"   << (opts.displayShape.empty() ? "(none)" : opts.displayShape) << '\n'
                  << std::setw(width) << "Input:"          << (opts.inputFile.empty() ? "(none)" : opts.inputFile) << '\n'
                  << std::setw(width) << "Output:"         << (opts.outputFile.empty() ? "(none)" : opts.outputFile) << '\n'
                  << std::setw(width) << "Mode:"           << (opts.normalize ? "Normalize" : "Denormalize") << '\n'
                  << std::setw(width) << "PrintTensor:"    << (opts.printTensor ? "Yes" : "No") << '\n'
                  << std::string(25, '=') << std::endl;
    }
};

std::vector<int> parseShape(const std::string& shapeStr) {
    std::vector<int> shape;
    std::string::size_type pos = 0;
    std::string::size_type prev = 0;

    while ((pos = shapeStr.find_first_of("x,", prev)) != std::string::npos) {
        if (pos > prev) {
            shape.push_back(std::stoi(shapeStr.substr(prev, pos - prev)));
        }
        prev = pos + 1;
    }
    if (prev < shapeStr.length()) {
        shape.push_back(std::stoi(shapeStr.substr(prev)));
    }
    return shape;
}

template<typename T>
std::vector<T> generateTestData(const Shape& shape, const std::string& dtype) {
    int totalSize = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<T> data(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        if (dtype == "int8" || dtype == "uint8") {
            data[i] = static_cast<T>(i % 256);
        } else {
            data[i] = static_cast<T>(i % 2048) / 10.0f;
        }
    }
    return data;
}

// Add forward declaration
template<typename T>
void processDataType(const tensorNorm::ProgramOptions& options,
                    const tensorNorm::Shape& inputShape,
                    const tensorNorm::Shape& displayShape);

// Add implementation of processDataType before main()
template<typename T>
void processDataType(const tensorNorm::ProgramOptions& options,
                    const tensorNorm::Shape& inputShape,
                    const tensorNorm::Shape& displayShape) {
    tensorNorm::TensorNormalizer normalizer;
    std::vector<T> inputData;

    if (options.inputFile.empty()) {
        // Generate test data if no input file specified
        inputData = generateTestData<T>(inputShape, options.dtype);
    } else {
        // Read from input file
        inputData = tensorNorm::FileUtils::readBinaryFile<T>(options.inputFile, options.dtype);
    }

    // Process the tensor
    auto result = normalizer.processTensor(inputData, inputShape, options.dtype,
                                         options.normalize, options.layout);

    // Print tensor if requested
    if (options.printTensor) {
        tensorNorm::TensorNormalizer::printTensorAsDataFrame(result, displayShape,
                                                           options.dtype, "Result");
    }

    // Write output if specified
    if (!options.outputFile.empty()) {
        tensorNorm::FileUtils::writeBinaryFile(options.outputFile, result);
    }
}

} // namespace tensorNorm

int main(int argc, char* argv[]) {
    try {
        auto options = tensorNorm::ProgramOptions::parse(argc, argv);
        tensorNorm::ProgramOptions::dumpOptions(options);

        tensorNorm::Shape inputShape = tensorNorm::parseShape(options.shape);
        tensorNorm::Shape displayShape = options.displayShape.empty() ?
            inputShape : tensorNorm::parseShape(options.displayShape);

        if (options.dtype == "int8") {
            tensorNorm::processDataType<int8_t>(options, inputShape, displayShape);
        } else if (options.dtype == "uint8") {
            tensorNorm::processDataType<uint8_t>(options, inputShape, displayShape);
        } else {
            tensorNorm::processDataType<float>(options, inputShape, displayShape);
        }

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 3;
    }
}
