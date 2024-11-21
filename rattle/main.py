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
from .similarity import process_second_feature

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
    # 第二个特征分析
    process_second_feature(ssa)
    if args.stdout_to:
        sys.stdout = orig_stdout
        args.stdout_to.close()

    if args.input:
        args.input.close()




    





































# 解决第一个特征相关的函数
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