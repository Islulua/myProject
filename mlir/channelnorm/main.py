from typing import List, Tuple, Optional, Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import argparse

class TensorNormalizer:
    """张量标准化处理类"""
    
    INT8_TYPES = ['int8', 'uint8']
    VALID_LAYOUTS = ['Tensor', 'NTensor']
    
    @staticmethod
    def get_align_base(channel: int, dtype: str) -> int:
        """
        计算对齐基数
        
        Args:
            channel: 通道数
            dtype: 数据类型
        
        Returns:
            int: 对齐基数
        """
        align_bases = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
        base = 64
        
        if channel > 64 and dtype in TensorNormalizer.INT8_TYPES:
            return 128
            
        for threshold, value in align_bases:
            if channel > threshold:
                base = value
                
        return base if channel > 0 else 0

    @staticmethod
    def _merge_middle_dims(shape: Tuple[int, ...]) -> int:
        """合并中间维度"""
        return np.prod(shape[1:-1]) if len(shape) > 2 else 1

    def normalize_tensor(self, input_data: NDArray, shape: Tuple[int, ...], dtype: str) -> NDArray:
        """标准化张量"""
        base = self.get_align_base(shape[-1], dtype)
        if shape[-1] % base != 0:
            raise ValueError("最后一维度必须是base的整数倍")
            
        cx = shape[-1] // base
        merged_dims = self._merge_middle_dims(shape)
        new_shape = [shape[0], 1, merged_dims, cx, base]
        
        reshaped = np.reshape(input_data, new_shape)
        transposed = np.transpose(reshaped, (0, 3, 2, 1, 4))
        return transposed.reshape(shape)

    def denormalize_tensor(self, input_data: NDArray, shape: Tuple[int, ...], dtype: str) -> NDArray:
        """反标准化张量"""
        base = self.get_align_base(shape[-1], dtype)
        if shape[-1] % base != 0:
            raise ValueError("最后一维度必须是base的整数倍")
            
        cx = shape[-1] // base
        merged_dims = self._merge_middle_dims(shape)
        new_shape = [shape[0], cx, merged_dims, 1, base]
        
        reshaped = np.reshape(input_data, new_shape)
        transposed = np.transpose(reshaped, (0, 3, 2, 1, 4))
        return transposed.reshape(shape)

    def process_tensor(self, 
                      input_data: NDArray, 
                      shape: Tuple[int, ...], 
                      dtype: str,
                      normalize: bool = True,
                      layout: Literal['Tensor', 'NTensor'] = 'Tensor') -> NDArray:
        """
        处理张量（标准化或反标准化）
        
        Args:
            input_data: 输入张量
            shape: 张量形状
            dtype: 数据类型
            normalize: True为标准化，False为反标准化
            layout: 张量布局类型
        """
        if layout not in self.VALID_LAYOUTS:
            raise ValueError(f"layout必须是 {self.VALID_LAYOUTS} 之一")
            
        new_shape = [1] + list(shape) if layout == 'Tensor' else list(shape)
        
        if normalize:
            return self.normalize_tensor(input_data, new_shape, dtype)
        return self.denormalize_tensor(input_data, new_shape, dtype)

    @staticmethod
    def print_tensor_as_dataframe(tensor: NDArray, 
                                shape: Tuple[int, ...], 
                                dtype: str, 
                                name: str = "Tensor") -> None:
        """将张量打印为DataFrame格式"""
        tensor_2d = tensor.reshape(shape).reshape(-1, shape[-1])
        
        index_names = [f'dim{i}' for i in range(1, len(shape))]
        index = pd.MultiIndex.from_product([range(dim) for dim in shape[:-1]], 
                                         names=index_names)
        
        base = TensorNormalizer.get_align_base(shape[-1], dtype)
        columns = [f'{i//base}_{i%base}' for i in range(shape[-1])]
        
        df = pd.DataFrame(tensor_2d, index=index, columns=columns)
        print(f"\n{name} DataFrame:")
        print(df)

def parse_shape(shape_str: str) -> List[int]:
    """
    解析形状字符串，支持 '1x2x3x4' 或 '1,2,3,4' 格式
    
    Args:
        shape_str: 形状字符串，如 '1x2x3x4' 或 '1,2,3,4'
    
    Returns:
        List[int]: 解析后的形状列表
    """
    # 支持 'x' 或 ',' 作为分隔符
    separators = ['x', ',']
    for sep in separators:
        if sep in shape_str:
            return [int(dim) for dim in shape_str.split(sep)]
    return [int(shape_str)]  # 单个数字的情况

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='张量标准化处理工具')
    parser.add_argument('--shape', type=str, required=True,
                      help='张量形状，例如：1x4x256 或 1,4,256')
    parser.add_argument('--dtype', type=str, default='fp16',
                      choices=['fp16', 'fp32', 'int8', 'uint8'],
                      help='数据类型 (default: fp16)')
    parser.add_argument('--layout', type=str, default='Tensor',
                      choices=['Tensor', 'NTensor'],
                      help='张量布局类型 (default: Tensor)')
    parser.add_argument('--temp-shape', type=str,
                      help='临时形状用于显示，例如：1x4x4x64')
    parser.add_argument('--input', type=str,
                      help='输入二进制文件路径')
    parser.add_argument('--output', type=str,
                      help='输出二进制文件路径')
    parser.add_argument('--normalize', action='store_true',
                      help='进行标准化处理，默认为True；使用--no-normalize进行反标准化')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize',
                      help='进行反标准化处理')
    parser.add_argument('--print', action='store_true', default=False,
                      help='是否打印张量数据')
    
    parser.set_defaults(normalize=True)
    args = parser.parse_args()
    
    # 设置pandas显示选项
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # 创建标准化器实例
    normalizer = TensorNormalizer()
    
    # 解析形状参数
    input_shape = tuple(parse_shape(args.shape))
    display_shape = tuple(parse_shape(args.temp_shape)) if args.temp_shape else input_shape
    
    # 确定numpy数据类型
    dtype_map = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
        'uint8': np.uint8
    }
    np_dtype = dtype_map[args.dtype]
    
    # 准备输入数据
    if args.input:
        # 从二进制文件读取数据
        with open(args.input, 'rb') as f:
            input_data = np.frombuffer(f.read(), dtype=np_dtype).reshape(input_shape)
    else:
        # 计算总元素数量并生成测试数据
        total_elements = np.prod(input_shape)
        input_data = np.arange(total_elements, dtype=np_dtype).reshape(input_shape)
    
    # 处理张量
    output = normalizer.process_tensor(input_data, input_shape, args.dtype, args.normalize, args.layout)
    
    # 保存输出数据
    if args.output:
        with open(args.output, 'wb') as f:
            output.astype(np_dtype).tofile(f)
    
    # 打印结果
    if args.print:
        normalizer.print_tensor_as_dataframe(input_data, display_shape, args.dtype, "Input")
        normalizer.print_tensor_as_dataframe(output, display_shape, args.dtype, 
                                           "Normalized Output" if args.normalize else "Denormalized Output")


if __name__ == "__main__":
    main()
    
    

                        
    