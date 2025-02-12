import json
from mlir.ir import *
from mlir.dialects import builtin, func, arith, math
from mlir.ir import WalkResult

# 定义一个map， key是operand， value是int类型，value从0开始递增，不同的operand对应不同的value
operand_map = {}
ID = 0

def convert_operation_to_node(operation):
    """将单个Operation转换为节点格式"""
    node = {
        "type": {
            "name": operation.operation.name,
            "category": "Layer"
        },
        "name": str(operation.operation.name),
        "inputs": [],
        "outputs": [],
        "attributes": []
    }
    
    # 处理输入
    for i, operand in enumerate(operation.operands):
        node["inputs"].append({
            "name": f"{operation.operation.name}_input_{i}",
            "value": [operand_map[operand]]
        })
    
    # 处理输出
    for i, result in enumerate(operation.results):
        node["outputs"].append({
            "name": f"{operation.operation.name}_output_{i}",
            "value": [operand_map[result]],
        })
    
    # # 处理属性 - 修改这部分代码
    # try:
    #     # 获取操作的所有属性名称
    #     for attr_name in operation.operation.attributes:
    #         attr = operation.operation.attributes[str(attr_name)]
    #         attr_value = None
    #         attr_type = "unknown"
            
    #         try:
    #             # 尝试获取属性值并转换为Python基本类型
    #             if hasattr(attr, "value"):
    #                 raw_value = attr.value
    #                 # 根据属性类型进行转换
    #                 if isinstance(raw_value, (int, float, bool)):
    #                     attr_value = raw_value
    #                 else:
    #                     attr_value = str(raw_value)
    #                 attr_type = str(attr.type)
    #             else:
    #                 attr_value = str(attr)
                
    #             node["attributes"].append({
    #                 "name": str(attr_name),
    #                 "type": attr_type,
    #                 "value": attr_value
    #             })
    #         except Exception as e:
    #             print(f"警告: 处理属性 {attr_name} 时出错: {e}")
    #             continue
                
    # except Exception as e:
    #     print(f"警告: 获取属性列表时出错: {e}")
    
    return node

def convert_function_to_graph(func_op):
    """将FuncOp转换为图格式"""
    graph = {
        "nodes": []
    }
    
    # 遍历函数体中的所有操作
    for block in func_op.body:
        for operation in block:
            node = convert_operation_to_node(operation)
            graph["nodes"].append(node)
    
    return graph


def convert_module_to_json(module):
    """将MLIR模块转换为JSON格式"""
    # json_model['signature'] = 'netron:onnx'
    #     json_model['format'] = 'ONNX' + (' v' + str(model.ir_version) if model.ir_version else '')
    json_data = {
        "signature": "netron:onnx",
        "format": "ONNX",
        "graphs": []
    }
    
    # 遍历模块中的所有函数
    for op in module.body:
        if isinstance(op, func.FuncOp):
            graph = convert_function_to_graph(op)
            json_data["graphs"].append(graph)
    
    return json_data


with Context() as ctx:
    ctx.allow_unregistered_dialects = True
    ctx.allow_unregistered_operations = True

    module = Module.parse(open("test.mlir").read())

    # 遍历module中的所有操作, 处理每个操作的operand和result, 并生成一个map, key是operand或result, value是int类型, value从0开始递增, 不同的operand或result对应不同的value
    for op in module.body:
        if isinstance(op, func.FuncOp):
            # 处理函数参数和返回值
            for operand in op.operands:
                if operand not in operand_map:
                    operand_map[operand] = ID
                    ID += 1
            for result in op.results:
                if result not in operand_map:
                    operand_map[result] = ID
                    ID += 1
            
            # 处理函数体内的所有操作
            for block in op.body:
                for operation in block:
                    # 处理每个操作的操作数
                    for operand in operation.operands:
                        if operand not in operand_map:
                            operand_map[operand] = ID
                            ID += 1
                    # 处理每个操作的结果
                    for result in operation.results:
                        if result not in operand_map:
                            operand_map[result] = ID
                            ID += 1

        json_data = convert_module_to_json(module)
        from netron.server import serve
        text = json.dumps(json_data, indent=2, ensure_ascii=False)
        serve("model.netron", text.encode("utf-8"))
        with open("test.json", "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)










