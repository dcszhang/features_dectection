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
from .feature2 import process_second_feature
from .feature1 import analyze_contract_complex
from .feature3 import analyze_contract_external
from .feature4 import process_fourth_feature
import pydot
import networkx as nx
# This might not be true, but I have a habit of running the wrong python version and this is to save me frustration
assert (sys.version_info.major >= 3 and sys.version_info.minor >= 6)
from pprint import pprint
logger = logging.getLogger(__name__)


def main(argv: Sequence[str] = tuple(sys.argv)) -> None:  # run me with python3, fool

    # # 调用函数
    # extract_sentences_from_bytecode(
    #     bytecode_subset_file="dataset/cleaned_bytecode.pkl",
    #     output_file="dataset/training_data.pkl"
    # )
    default_args = [
        '--input', 'bytecode',  # 假设 'bytecode' 是一个可读的文件路径
        '--verbosity', 'Info'
    ]
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
    
    # args = parser.parse_args(argv[1:])
    # 使用默认参数模拟命令行输入
    args = parser.parse_args(default_args)
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

    print(ssa)

    print("--------------------------------------------------------------------------------------")
    print("------------------------Scam smart contract analysis process--------------------------")
    print("--------------------------------------------------------------------------------------")
    
    # 第一个特征分析
    analyze_contract_complex(ssa)

    # # 第二个特征分析
    # process_second_feature(ssa)
     
    # # 第三个特征分析
    # analyze_contract_external(ssa)

    # 第四个特征分析
    process_fourth_feature(ssa)

    # 第五个特征分析
    # process_fifth_feature(ssa)



    if args.stdout_to:
        sys.stdout = orig_stdout
        args.stdout_to.close()

    if args.input:
        args.input.close()




    






































from tqdm import tqdm

def extract_sentences_from_bytecode(bytecode_subset_file, output_file,max_bytecodes=20):
    """
    提取所有 bytecode 的基本块指令序列，并整合为训练 Word2Vec 的句子。
    
    :param bytecode_subset_file: 包含 bytecode 的 .pkl 文件路径。
    :param output_file: 保存整合结果的 .pkl 文件路径。
    """
    # 加载 bytecode 数据集
    with open(bytecode_subset_file, "rb") as f:
        bytecodes = pickle.load(f)

    training_data = []  # 存储所有基本块的指令序列（每个序列即一个 sentence）

    for index, bytecode in enumerate(tqdm(bytecodes[:max_bytecodes], desc="Processing bytecodes")):
        try:
            # 使用 rattle 提取指令序列
            # 检查 bytecode 类型并修正
            if isinstance(bytecode, str):
                bytecode = bytecode.encode('utf-8')  # 转换为 bytes 类型

            ssa = rattle.Recover(bytecode, edges=[], optimize=False, split_functions=False)
            instruction_sequences = {}
            for function in sorted(ssa.functions, key=lambda f: f.offset):
                cfg = rattle.ControlFlowGraph(function)
                dot_content = cfg.dot()
                function_sequences = parse_cfg_dot(dot_content)
                instruction_sequences[function.offset] = function_sequences

            # 整合所有基本块的指令序列
            for sequences in instruction_sequences.values():
                for instructions in sequences.values():
                    training_data.append(instructions)

        except Exception as e:
            print(f"Error processing bytecode {index + 1}: {e}")

    # 打印检查：打印部分指令序列
    print("Number of instruction sequences:", len(training_data))
    print("Example sequences:")
    for i, seq in enumerate(training_data[:5]):  # 打印前 5 个序列
        print(f"Sequence {i + 1}: {seq}")

    # 保存整合结果到 .pkl 文件
    with open(output_file, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} sentences to {output_file}")




def parse_cfg_dot(dot_content):
        """
        解析单个函数生成的 CFG DOT 图内容，提取基本块及其指令序列。
        :param dot_content: DOT 图字符串
        每个函数生成的 instruction_sequences 是一个嵌套字典结构，形如：
        {
            function1: {
                block_offset1: ["PUSH1 0x60", "MSTORE"],
                block_offset2: ["CALLVALUE", "DUP1", "ISZERO"]
            },
            function2: {
                block_offset3: ["PUSH2 0x40", "JUMP"],
                ...
            }
        }
        """
        graphs = pydot.graph_from_dot_data(dot_content)
        if not graphs:
            raise ValueError("Failed to parse DOT content.")
        
        # 通常每个函数的 DOT 只包含一个图，取第一个即可
        graph = graphs[0]
        nx_graph = nx.nx_pydot.from_pydot(graph)
        
        # 提取基本块指令
        instruction_sequences = {}
        for node, data in nx_graph.nodes(data=True):
            label = data.get("label", "")
            if "block_" in node and label:
                # 提取基本块偏移量
                block_offset = int(node.split("_")[1], 16)

                # 从 label 字段解析指令，仅保留操作码
                instructions = []
                for line in label.split("\\l"):
                    if ": " in line:
                        # 提取指令部分
                        raw_instruction = line.split(": ", 1)[1].strip()
                        # 移除定义部分（如 %1508 =）
                        if "=" in raw_instruction:
                            raw_instruction = raw_instruction.split("=", 1)[1].strip()
                        # 提取操作码部分（忽略等号后的内容）
                        opcode = raw_instruction.split("(", 1)[0].strip()
                        instructions.append(opcode)

                instruction_sequences[block_offset] = instructions

        return instruction_sequences