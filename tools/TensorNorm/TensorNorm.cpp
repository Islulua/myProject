#ifndef DATABLOCK_H
#define DATABLOCK_H

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cstdint>

enum DataType {
    INT8 = 1,
    INT16 = 2,
    INT32 = 4,
    INT64 = 8,
    FLOAT16 = 16,
    FLOAT32 = 32,
    FLOAT64 = 64,
};

int getTypeSize(DataType type) {
    switch (type) {
        case INT8:
            return 1;
        case INT16:
            return 2;
        case INT32:
            return 4;
        case INT64:
            return 8;
        case FLOAT16:
            return 2;
        case FLOAT32:
            return 4;
        case FLOAT64:
            return 8;
        default:
            return 0;
    }
}

class SubData {
public:
    SubData(uint64_t srcOffset, uint64_t dstOffset, size_t size)
        : srcOffset_(srcOffset), dstOffset_(dstOffset), size_(size) {}

    uint64_t getSrcOffset() const { return srcOffset_; }
    uint64_t getDstOffset() const { return dstOffset_; }
    size_t getSize() const { return size_; }

    bool operator<(const SubData& other) const {
        return srcOffset_ < other.srcOffset_;
    }

private:
    uint64_t srcOffset_;
    uint64_t dstOffset_;
    size_t size_;
};

class DataBlock {
public:
    DataBlock(std::vector<size_t> shape, DataType type) : shape_(shape), type_(type) {}
    void init();
    void addSubData(const SubData& subData) {
        subDataList_.push_back(subData);
    }

    void sortSubData() {
        std::sort(subDataList_.begin(), subDataList_.end());
    }

    const std::vector<SubData>& getSubDataList() const {
        return subDataList_;
    }

private:
    std::vector<size_t> shape_;
    DataType type_;
    std::vector<SubData> subDataList_;
    std::string layout_;
    size_t bankSize;
    size_t Cx, C0;
};

size_t get_cx_align_base(size_t cx, DataType type) {
    return 64;
}

void DataBlock::init() {
    layout_   = shape_.size() > 2 ? "NTensor" : "Tensor";
    auto base = get_cx_align_base(shape_.back(), type_);
    auto Cx   = shape_.back() / base;
    auto C0   = shape_.back() % base;

    std::vector<size_t> tempCxShape;
    if (layout_ == "Tensor") tempCxShape.push_back(1);

    bankSize = Cx * base;
    tempCxShape.push_back(Cx);
    for (auto i = 1; i < shape_.size() - 2; i ++) {
        bankSize *= shape_[i];
        tempCxShape.push_back(shape_[i]);
    }
    tempCxShape.push_back(base);

    bankSize = std::
}



int main(int argc, char* argv[]) {
    std::vector<size_t> shape = {1024, 28, 128};
    DataBlock dataBlock(shape, FLOAT32);
    dataBlock.init();
    SubData subData1(0, 0, 4);
    SubData subData2(4, 4, 4);
    dataBlock.addSubData(subData1);
    dataBlock.addSubData(subData2);
    dataBlock.sortSubData();
    const std::vector<SubData>& subDataList = dataBlock.getSubDataList();
    for (size_t i = 0; i < subDataList.size(); i++) {
        std::cout << subDataList[i].getSrcOffset() << " " << subDataList[i].getDstOffset() << " " << subDataList[i].getSize() << std::endl;
    }
    return 0;
    
}

#endif // DATABLOCK_H