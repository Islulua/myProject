### 编译

```bash
mkdir build
cd build
cmake ..
make
```

### 使用

### 命令行选项

- `--shape SHAPE`：指定张量形状，例如：1x4x256 或 1,4,256（必需参数）
- `--dtype TYPE`：指定数据类型 [fp16, fp32, int8, uint8]（默认：fp16）
- `--layout LAYOUT`：指定张量布局类型 [Tensor, NTensor]（默认：Tensor）
- `--display-shape SHAPE`：用于显示的临时形状，例如：1x4x4x64
- `--input FILE`：输入二进制文件路径
- `--output FILE`：输出二进制文件路径
- `--normalize`：执行标准化操作（默认）
- `--denormalize`：执行反标准化操作
- `--print`：打印张量数据
- `--help`：显示帮助信息

### 使用示例

1. 标准化一个1x4x256形状的fp16张量：

```bash
./tensorNorm --shape 1x4x256 --dtype fp16 --normalize --input input.bin --output output.bin
```

2. 从文件读取并反标准化，同时显示结果：

```bash
./tensorNorm --shape 1x4x256 --input input.bin --output output.bin --denormalize --print
```

## 对齐规则

通道维度的对齐规则如下：
- 对于int8/uint8类型，当通道数>64时，对齐到128
- 其他情况下，按照以下阈值对齐：4、8、16、32、64
- 最后一维必须是对齐基数的整数倍

## 数据类型支持

- fp32：32位浮点数
- fp16：16位浮点数
- bf16：16位脑浮点数
- int8：8位有符号整数
- uint8：8位无符号整数

## 布局说明

- Tensor：标准张量布局
- NTensor：标准化后的张量布局，针对特定硬件优化

## 错误处理

工具会对输入参数进行严格检查，并在遇到以下情况时报错：
- 无效的形状参数
- 不支持的数据类型
- 不支持的布局类型
- 输入数据大小与形状不匹配
- 文件读写错误

## 注意事项

1. 确保输入张量的最后一维是对齐基数的整数倍
2. 使用 --display-shape 时，确保指定的形状与实际数据大小匹配
3. 二进制文件的读写使用本机字节序
