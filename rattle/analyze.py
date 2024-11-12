#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .recover import *

logger = logging.getLogger(__name__)


class UseDefGraph(object):
    value: StackValue

    def __init__(self, value: StackValue) -> None:
        self.value = value

    def dot(self) -> str:
        rv = ''
        rv += 'digraph G {\n'

        es = self.edges(self.value)

        for reader in self.value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(self.value).replace('%', '\\%')
            es.append(f"\"{value_s}\" -> \"{reader_s}\"")

        rv += '\n'.join(list(set(es)))
        rv += '\n}'

        return rv

    def edges(self, value) -> List[str]:
        rv = []
        writer = value.writer
        if writer is None:
            return []

        value_s = str(value).replace('%', '\\%')
        writer_s = str(writer).replace('%', '\\%')
        rv.append(f"\"{writer_s}\" -> \"{value_s}\"")

        for arg in writer:
            arg_s = str(arg).replace('%', '\\%')
            writer_s = str(writer).replace('%', '\\%')
            rv.append(f"\"{arg_s}\" -> \"{writer_s}\"")
            rv.extend(self.edges(arg))

        for reader in writer.return_value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(value).replace('%', '\\%')
            rv.append(f"\"{value_s}\" -> \"{reader_s}\"")

        return rv


class DefUseGraph(object):
    value: StackValue

    def __init__(self, value: StackValue) -> None:
        self.value = value

    def dot(self, filt=None) -> str:
        if filt is None:
            filt = lambda x: True

        rv = ''
        rv += 'digraph G {\n'

        es = self.edges(self.value, filt)

        for reader in self.value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(self.value).replace('%', '\\%')
            es.append(f"\"{value_s}\" -> \"{reader_s}\"")

        rv += '\n'.join(list(set(es)))
        rv += '\n}'

        return rv

    def edges(self, value, filt) -> List[str]:
        rv = []
        writer = value.writer
        if writer is None:
            return []

        value_s = str(value).replace('%', '\\%')
        writer_s = str(writer).replace('%', '\\%')
        rv.append(f"\"{writer_s}\" -> \"{value_s}\"")

        for reader in writer.return_value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(value).replace('%', '\\%')
            rv.append(f"\"{value_s}\" -> \"{reader_s}\"")

            if filt(reader):
                rv.extend(self.edges(reader.return_value, filt))

        return rv


class ControlFlowGraph:
    def __init__(self, function: SSAFunction) -> None:
        self.function = function

    def get_edges(self):
        edges = []
        # 直接遍历 self.function 中的每个 block
        for block in self.function:  # self.function 是一个 SSAFunction 实例，支持直接迭代
            block_id = f'block_{block.offset:#x}'
            
            # 获取 fallthrough 边
            if block.fallthrough_edge:
                target_block_id = f'block_{block.fallthrough_edge.offset:#x}'
                edges.append((block_id, target_block_id))

            # 获取跳转边
            for edge in block.jump_edges:
                target_block_id = f'block_{edge.offset:#x}'
                edges.append((block_id, target_block_id))

        return edges

    def dot(self) -> str:
        rv = 'digraph G {\n'
        rv += 'graph [fontname = "consolas"];\n'
        rv += 'node [fontname = "consolas"];\n'
        rv += 'edge [fontname = "consolas"];\n'

        name = self.function.desc()
        hash = f'Hash: {self.function.hash:#x}'
        offset = f'Start: {self.function.offset:#x}'
        arguments = f'Arguments: {self.function.arguments()}'
        storage = f'Storage: {self.function.storage}'

        function_desc = [name, hash, offset, arguments, storage]
        rv += f'ff [label="{{' + '\\l'.join(function_desc) + '\\l}}", shape="record" ];\n'

        edges = []

        # 遍历 self.function 中的每个 block
        for block in self.function:
            block_id = f'block_{block.offset:#x}'
            block_body = '\\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
            block_body = block_body.replace('<', '\\<').replace('>', '\\>')
            block_dot = f'{block_id} [label="{block_body}\\l", shape="record"];'

            fallthrough_label = ''
            jump_label = ''
            if len(block.jump_edges) > 0 and block.fallthrough_edge:
                fallthrough_label = ' [label=" f", color="red"]'
                jump_label = ' [label=" t", color="darkgreen"]'

            if block.fallthrough_edge:
                target_block_id = f'block_{block.fallthrough_edge.offset:#x}'
                edges.append(f'{block_id} -> {target_block_id}{fallthrough_label};')

            for edge in block.jump_edges:
                target_block_id = f'block_{edge.offset:#x}'
                edges.append(f'{block_id} -> {target_block_id}{jump_label};')

            rv += block_dot + '\n'

        for edge in edges:
            rv += edge + '\n'

        rv += '}\n'
        return rv



class ProgramDependenceGraph:
    def __init__(self, function: SSAFunction):
        self.function = function
        self.cfg = ControlFlowGraph(function)  # 使用已有的 CFG
        self.control_edges = self._extract_control_dependence()
        self.data_edges = self._extract_data_dependence()

    def _extract_control_dependence(self):
        # 提取控制依赖关系，直接使用 CFG 中的控制流信息
        control_edges = self.cfg.get_edges()
        return control_edges

    def _extract_data_dependence(self):
        data_edges = []
        definitions = {}
        usages = {}

        for block in self.function:
            block_id = f'block_{block.offset:#x}'

            # 记录定义和使用的变量
            for insn in block:
                # 定义的变量
                if insn.return_value is not None:
                    defined_var = insn.return_value
                    definitions[defined_var] = block_id

                # 使用的变量
                used_vars = insn.arguments
                for var in used_vars:
                    if var in definitions:
                        data_edges.append((definitions[var], block_id))

        return data_edges

    def dot(self):
        rv = 'digraph PDG {\n'
        rv += 'graph [fontname = "consolas"];\n'
        rv += 'node [fontname = "consolas"];\n'
        rv += 'edge [fontname = "consolas"];\n'

        # 函数头信息
        name = self.function.desc()
        hash = f'Hash: {self.function.hash:#x}'
        offset = f'Start: {self.function.offset:#x}'
        arguments = f'Arguments: {self.function.arguments()}'
        storage = f'Storage: {self.function.storage}'
        function_desc = [name, hash, offset, arguments, storage]
        rv += f'ff [label="{{' + '\\l'.join(function_desc) + '\\l}}", shape="record" ];\n'

        # 添加基本块节点
        for block in self.function:
            block_id = f'block_{block.offset:#x}'
            block_body = '\\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
            block_body = block_body.replace('<', '\\<').replace('>', '\\>')
            rv += f'{block_id} [label="{block_body}\\l", shape="record"];\n'

        # 添加控制依赖边
        for edge in self.control_edges:
            rv += f'{edge[0]} -> {edge[1]} [label="control", color="blue"];\n'

        # 添加数据依赖边
        for edge in self.data_edges:
            rv += f'{edge[0]} -> {edge[1]} [label="data", color="green"];\n'

        rv += '}\n'
        return rv


# 创建 SystemDependenceGraph 类
class SystemDependenceGraph:
    def __init__(self, functions: List[SSAFunction]):
        # 存储每个函数的 PDG
        self.function_pdgs = {func: ProgramDependenceGraph(func) for func in functions}
        self.call_edges = self._extract_call_dependence()
        self.parameter_edges = self._extract_parameter_dependence()

    def _extract_call_dependence(self):
        call_edges = []
        for caller, caller_pdg in self.function_pdgs.items():
            for block in caller:
                for insn in block:
                    if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                        # 检查 insn.arguments[1] 是否为 ConcreteStackValue
                        if isinstance(insn.arguments[1], ConcreteStackValue):
                            callee_hash = insn.arguments[1].concrete_value
                            callee = self._find_function_by_hash(callee_hash)

                            if callee:
                                callee_entry = f'block_{callee.blocks[0].offset:#x}'
                                caller_block = f'block_{block.offset:#x}'
                                call_edges.append((caller_block, callee_entry))

        return call_edges
    def _extract_parameter_dependence(self):
        parameter_edges = []
        for caller, caller_pdg in self.function_pdgs.items():
            for block in caller:
                for insn in block:
                    if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                        # 使用 resolve 方法尝试获取 arguments[1] 的具体值
                        resolved_arg, _ = insn.arguments[1].resolve()

                        # 检查 resolved_arg 是否有 concrete_value 属性
                        if hasattr(resolved_arg, 'concrete_value'):
                            callee_hash = resolved_arg.concrete_value
                            callee = self._find_function_by_hash(callee_hash)

                            if callee:
                                # 添加输入参数边
                                for i, arg in enumerate(insn.arguments[2:]):
                                    if arg in caller_pdg.data_edges:
                                        callee_param = f'param_{i}'
                                        caller_block = f'block_{block.offset:#x}'
                                        parameter_edges.append((caller_block, callee_param))

                                # 添加输出参数边
                                if insn.return_value:
                                    callee_return = f'block_{callee.blocks[-1].offset:#x}'
                                    caller_block = f'block_{block.offset:#x}'
                                    parameter_edges.append((callee_return, caller_block))

        return parameter_edges

    def _find_function_by_hash(self, hash_value):
        for function in self.function_pdgs:
            if function.hash == hash_value:
                return function
        return None

    
    def dot(self):
        rv = 'digraph SDG {\n'
        rv += 'graph [fontname = "consolas"];\n'
        rv += 'node [fontname = "consolas"];\n'
        rv += 'edge [fontname = "consolas"];\n'

        # 遍历所有函数的 PDG
        for func, pdg in self.function_pdgs.items():
            # 函数头信息
            name = func.desc().replace('<', '\\<').replace('>', '\\>')
            hash = f'Hash: {func.hash:#x}'
            offset = f'Start: {func.offset:#x}'
            arguments = f'Arguments: {func.arguments()}'
            storage = f'Storage: {func.storage}'

            # 创建节点名称，去掉括号和其他特殊字符
            node_name = name.replace('(', '').replace(')', '').replace(' ', '_')
            
            function_desc = [name, hash, offset, arguments, storage]
            rv += f'{node_name} [label="{{' + '\\l'.join(function_desc) + '\\l}}", shape="record" ];\n'

            # 添加基本块节点
            for block in func:
                block_id = f'block_{block.offset:#x}'
                block_body = '\\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
                block_body = block_body.replace('<', '\\<').replace('>', '\\>').replace('(', '\\(').replace(')', '\\)')
                rv += f'{block_id} [label="{block_body}\\l", shape="record"];\n'

            # 添加控制依赖边
            for edge in pdg.control_edges:
                rv += f'{edge[0]} -> {edge[1]} [label="control", color="blue"];\n'

            # 添加数据依赖边
            for edge in pdg.data_edges:
                rv += f'{edge[0]} -> {edge[1]} [label="data", color="green"];\n'

            # 添加参数依赖边
            if hasattr(pdg, 'parameter_edges'):
                for edge in pdg.parameter_edges:
                    rv += f'{edge[0]} -> {edge[1]} [label="param", color="orange"];\n'

        rv += '}\n'
        return rv






class BackwardSliceForCalls:
    def __init__(self, sdg: SystemDependenceGraph):
        self.sdg = sdg
        self.slices = {}  # 存储每个 CALL 指令的切片结果

    def perform_backward_slicing_for_calls(self):
        # 遍历每个 PDG，找到带有 CALL 指令的基本块
        for func, pdg in self.sdg.function_pdgs.items():
            for block in func:
                for insn in block:
                    if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                        # 对每个 CALL 指令执行 backward slicing
                        slice_result = self._perform_slice(pdg, f'block_{block.offset:#x}')
                        # 将结果存储
                        self.slices[f'{func.desc()}_CALL_at_{insn.offset:#x}'] = slice_result

    def _perform_slice(self, pdg, start_node):
        # 初始化切片结果
        slice_result = set()
        self._slice_recursive(start_node, pdg, slice_result)
        return slice_result

    def _slice_recursive(self, node, pdg, slice_result):
        if node in slice_result:
            return

        # 添加节点到切片结果
        slice_result.add(node)

        # 追踪控制和数据依赖的上游节点
        for edge in pdg.control_edges + pdg.data_edges:
            if edge[1] == node:
                self._slice_recursive(edge[0], pdg, slice_result)

    def to_dot(self):
        # 输出每个 CALL 指令的切片到单独的子图
        rv = 'digraph Slices {\n'
        rv += 'graph [fontname = "consolas"];\n'
        rv += 'node [fontname = "consolas"];\n'
        rv += 'edge [fontname = "consolas"];\n'

        # 每个 CALL 的切片结果作为独立子图输出
        for call, slice_nodes in self.slices.items():
            # 替换 `call` 中的非法字符
            sanitized_call = call.replace("(", "_").replace(")", "_")
            
            rv += f'subgraph cluster_{sanitized_call} {{\n'
            rv += f'label="{call}";\n'
            for node in slice_nodes:
                rv += f'{node} [style=filled, fillcolor=lightblue];\n'

            # 添加不同类型的依赖边，使用不同的颜色
            for func, pdg in self.sdg.function_pdgs.items():
                # 控制依赖（红色）
                for edge in pdg.control_edges:
                    if edge[0] in slice_nodes and edge[1] in slice_nodes:
                        rv += f'{edge[0]} -> {edge[1]} [color="red", label="control"];\n'
                
                # 数据依赖（蓝色）
                for edge in pdg.data_edges:
                    if edge[0] in slice_nodes and edge[1] in slice_nodes:
                        rv += f'{edge[0]} -> {edge[1]} [color="blue", label="data"];\n'
                
                # 如果有参数依赖边，可以加上其他样式（如橙色）
                if hasattr(pdg, 'parameter_edges'):
                    for edge in pdg.parameter_edges:
                        if edge[0] in slice_nodes and edge[1] in slice_nodes:
                            rv += f'{edge[0]} -> {edge[1]} [color="orange", label="param"];\n'

            rv += '}\n'

        rv += '}\n'
        return rv
