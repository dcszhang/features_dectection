#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dill as pickle  # 替换 pickle 为 dill
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Sequence
from collections import OrderedDict
import rattle
import networkx as nx
# This might not be true, but I have a habit of running the wrong python version and this is to save me frustration
assert (sys.version_info.major >= 3 and sys.version_info.minor >= 6)
from pprint import pprint
logger = logging.getLogger(__name__)

# 定义地址变量来保存所有路径
all_trace_paths = {}
# 定义全局变量，用于存储所有变量的追踪路径
all_traces = {}  # 将列表改为字典
# 定义全局变量来保存每个地址变量的树
all_trees = {}

def save_trace_to_global(variable, trace):
    """保存回溯路径到全局变量"""
    global all_trace_paths
    all_trace_paths[variable] = trace

def main(argv: Sequence[str] = tuple(sys.argv)) -> None:  # run me with python3, fool
    parser = argparse.ArgumentParser(
        description='rattle ethereum evm binary analysis')
    parser.add_argument('--input', '-i', type=argparse.FileType('rb'), help='input evm file')
    parser.add_argument('--optimize', '-O', action='store_true', help='optimize resulting SSA form')
    parser.add_argument('--no-split-functions', '-nsf', action='store_false', help='split functions')
    parser.add_argument('--log', type=argparse.FileType('w'), default=sys.stdout,
                        help='log output file (default stdout)')
    parser.add_argument('--verbosity', '-v', type=str, default="None",
                        help='log output verbosity (None,  Critical, Error, Warning, Info, Debug)')
    parser.add_argument('--supplemental_cfg_file', type=argparse.FileType('rb'), default=None, help='optional cfg file')
    parser.add_argument('--stdout_to', type=argparse.FileType('wt'), default=None, help='redirect stdout to file')
    args = parser.parse_args(argv[1:])

    if args.input is None:
        parser.print_usage()
        sys.exit(1)

    orig_stdout = sys.stdout
    if args.stdout_to:
        sys.stdout = args.stdout_to

    edges = []
    if args.supplemental_cfg_file:
        edges = json.loads(args.supplemental_cfg_file.read())

    try:
        loglevel = getattr(logging, args.verbosity.upper())
    except AttributeError:
        loglevel = None

    logging.basicConfig(stream=args.log, level=loglevel)
    logger.info(f"Rattle running on input: {args.input.name}")

    ssa = rattle.Recover(args.input.read(), edges=edges, optimize=args.optimize,
                         split_functions=args.no_split_functions)

    # print(ssa)

    # 假设这个集合存储了所有指令，偏移量作为键
    all_instructions_by_variable = {}
    print("Identified Functions:")
    for function in sorted(ssa.functions, key=lambda f: f.offset):
        print(f'\tFunction: {function.desc()} starts at offset {function.offset}, argument offsets: {function.arguments()}')

    print("")

    print("Storage Locations: " + repr(ssa.storage))
    # print("Memory Locations: " + repr(ssa.memory))

    # for location in [x for x in ssa.memory if x > 0x20]:
    #     print(f"Analyzing Memory Location: {location}\n")
    #     for insn in sorted(ssa.memory_at(location), key=lambda i: i.offset):
    #         print(f'\t{insn.offset:#x}: {insn}')
    #     print('\n\n')

    # for function in sorted(ssa.functions, key=lambda f: f.offset):
    #     print(f"Function {function.desc()} storage:")
    #     for location in function.storage:
    #         print(f"\tAnalyzing Storage Location: {location}")
    #         for insn in sorted(ssa.storage_at(location), key=lambda i: i.offset):
    #             print(f'\t\t{insn.offset:#x}: {insn}')
    #         print('\n')

    '''
    print("Tracing SLOAD(0) (ignoring ANDs)")
    for insn in ssa.storage_at(0):
        print(insn)
        if insn.insn.name == 'SLOAD':
            g = rattle.DefUseGraph(insn.return_value)
            print(g.dot(lambda x: x.insn.name in ('AND', )))
        print('\n')
    '''


    # 遍历所有函数并保存指令
    for function in ssa.functions:
        for block in function:
            for insn in block:
                if hasattr(insn, 'return_value') and insn.return_value is not None:
                    variable = insn.return_value  # 获取指令的返回值变量
                    all_instructions_by_variable[variable] = insn  # 将变量和对应的指令保存到字典中
    # 按偏移量对指令进行排序
    # all_instructions_by_offset = OrderedDict(sorted(all_instructions_by_offset.items()))
    # 打印每个指令的结构
    # for offset, insn in all_instructions_by_offset.items():
    #     print(f"Offset: {offset}, Instruction: {insn}")
    #     print(f"Attributes of insn: {dir(insn)}")
    #     print(f"Return value: {getattr(insn, 'return_value', 'No return_value')}")
    #     print(f"Arguments: {getattr(insn, 'arguments', 'No arguments')}")

    # print_all_instructions_by_offset(all_instructions_by_offset)
    can_send, functions_that_can_send = ssa.can_send_ether()
    func_index = 0
    if can_send:
        print("[+] Contract can send ether from following functions:")
        for function in functions_that_can_send:
             # 初始化每个函数的追踪列表
            all_traces[func_index] = []
            print(f"\t- {function.desc()}")
            _, insns = function.can_send_ether()
            for insn in insns:
                print(f"\t\t{insn}")
                if insn.insn.name == 'SELFDESTRUCT':
                    address = insn.arguments[0]
                    print(f'\t\t\t{address.writer}')
                elif insn.insn.name == 'CALL':
                    gas = insn.arguments[0]
                    address = insn.arguments[1]
                    value = insn.arguments[2]
                    argOst = insn.arguments[3]
                    argLen = insn.arguments[4]
                    retOst = insn.arguments[5]
                    retLen = insn.arguments[6]
                    # 回溯分析，记录沿途的指令路径
                    writer_insn = address.writer
                    address_source = insn.arguments[1]  # AND 操作的右侧输入
                    trace = backward_analysis(address_source, all_instructions_by_variable)  # function 是包含基本块的上下文
                    tree_root = build_tree_structure(address_source, all_instructions_by_variable)
                    print(f'\t\t\tTo:\t{writer_insn}')
                    print(f'\t\t\tTrace:')
                    for t in trace:
                        print(f'\t\t\t\t{t}')
                    analyze_saved_traces(address_source,all_instructions_by_variable) # 分析保存的路径 

                    try:
                        if value.writer:
                            print(f'\t\t\tValue:\t{value.writer}')
                        else:
                            value_in_eth = int(value) * 1.0 / 10 ** 18
                            print(f'\t\t\tValue:\t{value} {value_in_eth}ETH')
                    except Exception as e:
                        print(e)

                print("")
       
    else:
        print("[+] Contract can not send ether.")

    print("[+] Contract calls:")
    for call in ssa.calls():
        print(f"\t{call}")
        if call.insn.name == 'DELEGATECALL':
            gas, to, in_offset, in_size, out_offset, out_size = call.arguments
            value = None
        else:
            gas, to, value, in_offset, in_size, out_offset, out_size = call.arguments

        print(f"\t\tGas: {gas}", end='')
        if gas.writer:
            print(f'\t\t\t{gas.writer}')
        else:
            print("\n", end='')

        print(f"\t\tTo: {to} ", end='')
        if to.writer:
            print(f'\t\t\t{to.writer}')
        else:
            print("\n", end='')

        if value:
            print(f"\t\tValue: {value}", end='')
            if value.writer:
                print(f'\t\t\t{value.writer}')
            else:
                print("\n", end='')

        print(f"\t\tIn Data Offset: {in_offset}", end='')
        if in_offset.writer:
            print(f'\t\t{in_offset.writer}')
        else:
            print("\n", end='')

        print(f"\t\tIn Data Size: {in_size}", end='')
        if in_size.writer:
            print(f'\t\t{in_size.writer}')
        else:
            print("\n", end='')

        print(f"\t\tOut Data Offset: {out_offset}", end='')
        if out_offset.writer:
            print(f'\t\t{out_offset.writer}')
        else:
            print("\n", end='')

        print(f"\t\tOut Data Size: {out_size}", end='')
        if out_size.writer:
            print(f'\t\t{out_size.writer}')
        else:
            print("\n", end='')

        print("")



    # 对 SSA 中每个函数生成PDG
    for function in sorted(ssa.functions, key=lambda f: f.offset):
        cfg = rattle.ControlFlowGraph(function)
        pdg = rattle.ProgramDependenceGraph(function)
        # 写入 PDG 的 dot 文件
        with tempfile.NamedTemporaryFile(suffix='.dot', mode='w', delete=False) as t:
            t.write(pdg.dot())
            t.flush()
            dot_path = t.name

        try:
            os.makedirs('output', exist_ok=True)
        except:
            pass

        # 生成 SDG 的 PDF 文件
        out_file_pdf = f'output/pdg_{function.offset:#x}.pdf'
        subprocess.call(['dot', '-Tpdf', '-o', out_file_pdf, dot_path])
        print(f'[+] Wrote PDG to {out_file_pdf}')


    def generate_sdg(ssa_functions):
        # 创建 SystemDependenceGraph 实例
        sdg = rattle.SystemDependenceGraph(ssa_functions)
        
        # 写入 SDG 的 dot 文件
        with tempfile.NamedTemporaryFile(suffix='.dot', mode='w', delete=False) as t:
            t.write(sdg.dot())
            t.flush()
            dot_path = t.name

        try:
            os.makedirs('output', exist_ok=True)
        except:
            pass

        # 生成 SDG 的 PDF 文件
        out_file_pdf = 'output/sdg.pdf'
        subprocess.call(['dot', '-Tpdf', '-o', out_file_pdf, dot_path])
        print(f'[+] Wrote SDG to {out_file_pdf}')
        # 使用 CallBacktracking 对 SDG 中的函数进行回溯
        call_backtracking = rattle.CallBacktracking(sdg)

        # 获取回溯结果的 DOT 内容并保存为独立的 DOT 文件和 PDF 文件
        dot_content = call_backtracking.dot()
        dot_sections = dot_content.split('digraph ')
        
        for section in dot_sections:
            if section.strip():  # 确保不是空字符串
                func_name_end = section.find(' {')
                func_name = section[:func_name_end].strip() if func_name_end != -1 else 'unknown'
                dot_content = 'digraph ' + section

                with tempfile.NamedTemporaryFile(suffix='.dot', mode='w', delete=False) as t:
                    t.write(dot_content)
                    t.flush()
                    dot_path = t.name

                # 生成每个 CALL 路径的 PDF 文件
                out_file_pdf = f'output/{func_name}_call_backtrack.pdf'
                subprocess.call(['dot', '-Tpdf', '-o', out_file_pdf, dot_path])
                print(f'[+] Wrote backtrack paths to {out_file_pdf}')
    generate_sdg(sorted(ssa.functions, key=lambda f: f.offset))
    if args.stdout_to:
        sys.stdout = orig_stdout
        args.stdout_to.close()

    if args.input:
        args.input.close()


import networkx as nx

def build_cfg_from_slice(call_slice , all_instructions_by_variable):
    """
    根据收集的指令切片构建控制流图（CFG）。
    param call_slice: 包含 CALL 指令及其前置指令的切片
    return: 表示控制流图的 networkx.DiGraph 对象
    """
    G = nx.DiGraph()

    # 为每个指令添加一个节点
    for insn in call_slice:
        G.add_node(insn)

    # 添加控制流边，根据指令间的控制依赖关系
    for insn in call_slice:
        # 遍历每个指令的前置指令
        for argument in insn.arguments:
            if argument in all_instructions_by_variable:
                prev_insn = all_instructions_by_variable[argument]
                if prev_insn in call_slice:
                    G.add_edge(prev_insn, insn)  # 添加前置指令到当前指令的边

    return G

def collect_call_slice(call_insn, all_instructions_by_variable):
    """
    收集与 CALL 指令相关的所有前置指令，形成一个切片。
    param call_insn: CALL 指令
    param all_instructions_by_variable: 包含所有指令的字典，以变量为键，指令为值
    return: 包含所有相关指令的列表（切片）
    """
    call_slice = []           # 用于存储指令切片
    visited = set()           # 用于记录访问过的指令，防止重复
    instructions_to_trace = [call_insn]  # 初始化栈，从 CALL 指令开始

    while instructions_to_trace:
        insn = instructions_to_trace.pop()  # 获取栈顶指令
        if insn in visited:
            continue                       # 如果已访问，跳过

        visited.add(insn)                  # 标记为已访问
        call_slice.append(insn)            # 将指令加入切片

        # 追溯当前指令的前置指令
        for argument in insn.arguments:
            if argument in all_instructions_by_variable:
                prev_insn = all_instructions_by_variable[argument]
                instructions_to_trace.append(prev_insn)  # 将前置指令加入栈中

    return call_slice












def find_blocks_with_call(function):
    """
    找到包含 CALL 指令的所有基本块。
    param function: 包含基本块的函数对象
    return: 包含 CALL 指令的基本块列表
    """
    blocks_with_call = []
    for block in function.blocks:
        for insn in block:
            if insn.insn.name == 'CALL':
                blocks_with_call.append(block)
                break
    return blocks_with_call


def backward_slicing_from_call(blocks_with_call):
    """
    对包含 CALL 指令的块进行 backward slicing，去除与 CALL 无关的块。
    param blocks_with_call: 包含 CALL 指令的基本块列表
    return: 与 CALL 指令相关的基本块集合
    """
    related_blocks = set(blocks_with_call)
    visited_blocks = set(blocks_with_call)

    while blocks_with_call:
        current_block = blocks_with_call.pop(0)
        
        for pred in current_block.in_edges:
            if pred not in visited_blocks:
                related_blocks.add(pred)
                visited_blocks.add(pred)
                blocks_with_call.append(pred)

    return related_blocks

def save_related_blocks_as_graph(related_blocks):
    """
    将与 CALL 相关的基本块保存为 networkx 图。
    param related_blocks: 与 CALL 相关的基本块集合
    return: networkx 图对象
    """
    G = nx.DiGraph()
    for block in related_blocks:
        block_id = block.offset
        G.add_node(block_id)
        if block.fallthrough_edge and block.fallthrough_edge in related_blocks:
            G.add_edge(block_id, block.fallthrough_edge.offset)
        for jump_block in block.jump_edges:
            if jump_block in related_blocks:
                G.add_edge(block_id, jump_block.offset)
    return G



def backward_analysis(variable, all_instructions_by_variable, visited=None):
    """
    通过变量名直接查找定义，依据操作数进行递归回溯。
    param variable: SSA中的变量（如 %1504）
    param all_instructions_by_variable: 包含所有指令的字典，键为变量名，值为指令
    param visited: 已经访问过的变量集合，避免重复回溯
    return: 返回包含所有回溯指令的列表
    """
    if visited is None:
        visited = set()

    trace = []  # 用于存储回溯路径中的所有指令
    variables_to_trace = [variable]  # 初始化要追踪的变量列表

    while variables_to_trace:
        current_variable = variables_to_trace.pop(0)

        # 如果已经访问过该变量，则跳过，防止死循环
        if current_variable in visited:
            continue

        visited.add(current_variable)  # 将当前变量标记为已访问
                # 如果当前变量是常量，直接跳过回溯
        if str(current_variable).startswith('#'):
            # print(f"Constant value encountered: {current_variable}")
            continue
        # 查找该变量的定义
        if current_variable in all_instructions_by_variable:
            insn = all_instructions_by_variable[current_variable]
            trace.append(insn)  # 记录当前指令
            # print(f"Found definition of {current_variable}: {insn}")

            # 对该指令的操作数进行回溯
            for argument in insn.arguments:
                if argument not in visited:  # 防止重复追踪
                    variables_to_trace.append(argument)
        else:
            trace.append(f"No definition found for variable {current_variable}")
    # 保存路径到全局变量
    save_trace_to_global(variable, trace)
    return trace


def build_tree_structure(variable, all_instructions_by_variable, visited=None):
    """
    从地址变量开始构建树结构，并将其保存到全局变量中。
    param variable: 当前地址变量
    param all_instructions_by_variable: 包含所有指令的字典，键为变量名，值为指令
    param visited: 已经访问过的变量集合，避免重复回溯
    return: 树的根节点
    """
    if visited is None:
        visited = set()

    tree_root = TreeNode(variable)  # 创建树的根节点
    variables_to_trace = [(variable, tree_root)]  # 初始化要追踪的变量列表，包含树节点

    while variables_to_trace:
        current_variable, current_node = variables_to_trace.pop(0)

        # 如果已经访问过该变量，则跳过，防止死循环
        if current_variable in visited:
            continue

        visited.add(current_variable)  # 将当前变量标记为已访问

        # 如果当前变量是常量，直接跳过回溯
        if str(current_variable).startswith('#'):
            continue

        # 查找该变量的定义
        if current_variable in all_instructions_by_variable:
            insn = all_instructions_by_variable[current_variable]

            # 对该指令的操作数进行回溯，并构建子节点
            for argument in insn.arguments:
                if argument not in visited:
                    child_node = TreeNode(argument)  # 为操作数创建子节点
                    current_node.add_child(child_node)  # 将子节点加入当前节点
                    variables_to_trace.append((argument, child_node))  # 将子节点加入追踪队列

    # 保存树到全局变量
    save_tree_to_global(variable, tree_root)
    return tree_root  # 返回树的根节点


def analyze_saved_traces(address_variable, all_instructions_by_variable):
    """
    遍历全局变量中保存的所有路径，查找与字符串操作相关的指令，并判断是否匹配第一个特征。
    param address_variable: 当前的地址变量
    """
    # 遍历 all_trace_paths 中的所有保存路径
    for variable, trace in all_trace_paths.items():
        print(f"\t\t\tAnalyzing trace for variable: {variable}")
        
        has_complex_calculation = False
        has_dynamic_input = False
        has_external_call = False
        depth_threshold=5
        for insn in trace:

            # 检查是否是字符串操作指令
            if insn.insn.name in ['MLOAD', 'CALLDATALOAD']:
                has_dynamic_input = True
                # print(f"\t\t\t\tFound dynamic input operation at {insn}")
            elif insn.insn.name == 'ADD':
                has_complex_calculation = True
                # print(f"\t\t\t\tFound complex calculation operation at {insn}")
            elif insn.insn.name in ['CALL', 'DELEGATECALL']:
                has_external_call = True
                # print(f"\t\t\t\tFound external call operation at {insn}")
        # 计算树的层数
        tree_root = all_trees.get(variable)
        if tree_root:
            tree_height = tree_root.get_height()
            print(f"\t\t\t\tTree height for variable {variable}: {tree_height}")
        else:
            tree_height = 0  # 如果没有树，就设置高度为0
        # 匹配条件：(1) + (2) 或者 (1) + (3) 或者 树的高度超过阈值
        if (has_complex_calculation and has_dynamic_input) or (has_complex_calculation and has_external_call)or (tree_height > depth_threshold):
            print(f"\t\t\t[+] Trace for variable {variable} matches the first feature (complex address generation)")
        else:
            print(f"\t\t\t[-] Trace for variable {variable} does not match the first feature")


def save_tree_to_global(variable, tree):
    """
    保存树结构到全局变量
    param variable: 地址变量
    param tree: 树结构
    """
    global all_trees
    all_trees[variable] = tree

class TreeNode:
    def __init__(self, variable):
        self.variable = variable  # 当前节点对应的变量
        self.children = []  # 子节点列表

    def add_child(self, child_node):
        """将子节点添加到当前节点"""
        self.children.append(child_node)

    def __repr__(self):
        """便于调试，打印树节点的变量名"""
        return f"TreeNode({self.variable})"
    
    def print_tree(self, level=0, prefix=""):
            """递归打印整个树结构，使用更加直观的树形格式"""
            # 打印当前节点
            print(f"{prefix}{self.variable}")
            # 遍历子节点，调整前缀以显示树的结构
            for i, child in enumerate(self.children):
                # 如果是最后一个子节点，使用 "└──"，否则使用 "├──"
                if i == len(self.children) - 1:
                    child_prefix = prefix + "    "  # 对齐格式
                    child.print_tree(level + 1, prefix + "└── ")
                else:
                    child_prefix = prefix + "│   "  # 对齐格式
                    child.print_tree(level + 1, prefix + "├── ")
    def get_height(self):
            """递归计算树的高度"""
            if not self.children:
                return 1  # 如果没有子节点，树的高度为1
            return 1 + max(child.get_height() for child in self.children)