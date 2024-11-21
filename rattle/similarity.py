import numpy as np
from gensim.models import Word2Vec
import pydot
import rattle
from typing import Sequence
import tempfile
import os
import subprocess
import networkx as nx
def process_second_feature(ssa):
    # 处理第二个特征 
    instruction_sequences = {}  # 存储所有函数的指令序列
    for function in sorted(ssa.functions, key=lambda f: f.offset):
        cfg = rattle.ControlFlowGraph(function)
        # 获取 DOT 表示
        dot_content = cfg.dot()
        
        # 解析 DOT 提取基本块指令序列
        function_sequences = parse_cfg_dot(dot_content)
        
        # 存储结果
        instruction_sequences[function] = function_sequences
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

        # 生成 PDG 的 PDF 文件
        out_file_pdf = f'output/pdg_{function.offset:#x}.pdf'
        subprocess.call(['dot', '-Tpdf', '-o', out_file_pdf, dot_path])
        print(f'[+] Wrote PDG to {out_file_pdf}') #meiyige PDGshengcheng wanb 
    # 整合所有指令序列
    training_data = []
    for sequences in instruction_sequences.values():
        for instructions in sequences.values():
            training_data.append(instructions)
    # 输出提取的指令序列
    # for function, sequences in instruction_sequences.items():
    #     print(f"Function at offset {hex(function.offset)}:")
    #     for block_offset, instructions in sequences.items():
    #         print(f"  Basic Block {hex(block_offset)}:")
    #         for insn in instructions:
    #             print(f"    {insn}")
    # 检查训练数据
    print(f"Collected {len(training_data)} basic block sequences for training.")

    # 训练 Word2Vec 模型
    model = Word2Vec(
    sentences=training_data,
    vector_size=100,  # 嵌入向量维度
    window=3,         # 上下文窗口大小
    min_count=1,      # 最小频率
    sg=1,             # Skip-Gram 方法
    )

    # 保存模型
    model.save("cfg_word2vec.model")

    # 生成 SDG
    ssa_functions = sorted(ssa.functions, key=lambda f: f.offset)
    function_embeddings, adjacency_matrices = generate_sdg(ssa_functions, "cfg_word2vec.model")
    # 打印每个函数的基本块嵌入和邻接矩阵
    for function_key in function_embeddings.keys():
        print(f"Function: {function_key}")
        print("Node Embeddings:")
        for block_offset, embedding in function_embeddings[function_key].items():
            print(f"  Block {hex(block_offset)}: {embedding[:5]}...")  # 打印前5个维度
        print("Adjacency Matrix:")
        print(adjacency_matrices[function_key])
        # print(f"Function Name: {func_name}")
        # print(f"Adjacency Matrix Keys: {list(adjacency_matrices.keys())}")


    return function_embeddings

def generate_sdg(ssa_functions, model_path):
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
    # 存储基本块嵌入和边信息
    function_embeddings = {}
    adjacency_matrices = {}
    # # 加载 Word2Vec 模型
    model = Word2Vec.load("./cfg_word2vec.model")
    for section in dot_sections:
        if section.strip():  # 确保不是空字符串
            func_name_end = section.find(' {')
            func_name = section[:func_name_end].strip() if func_name_end != -1 else 'unknown'
            dot_content = 'digraph ' + section
            # 清理 DOT 图名称中的冗余部分
            func_name_cleaned = func_name.strip('"')  # 去掉引号
            
            # 提取函数名和编号
            if '_CallBacktrack_' in func_name_cleaned:
                base_name, suffix = func_name_cleaned.split('_CallBacktrack_')
                function_key = f"{base_name}:CallBacktrack_{suffix}"
            else:
                function_key = func_name_cleaned


            with tempfile.NamedTemporaryFile(suffix='.dot', mode='w', delete=False) as t:
                t.write(dot_content)
                t.flush()
                dot_path = t.name

            # 生成每个 CALL 路径的 PDF 文件
            out_file_pdf = f'output/{func_name}.pdf'
            subprocess.call(['dot', '-Tpdf', '-o', out_file_pdf, dot_path])
            print(f'[+] Wrote backtrack paths to {out_file_pdf}')
            # 提取基本块并计算嵌入
            for func, paths in call_backtracking.get_backtrack_results().items():
                func_embedding = {}
                edges = []
                for path in paths:
                    for block in path:
                        if block.offset not in func_embedding:
                            func_embedding[block.offset] = compute_block_embedding(block, model)
                    # 提取路径中的边
                    for i in range(len(path) - 1):
                        edges.append((path[i].offset, path[i + 1].offset))
                function_embeddings[function_key] = func_embedding
                adjacency_matrices[function_key] = generate_adjacency_matrix(func_embedding, edges)
    return function_embeddings , adjacency_matrices


def generate_adjacency_matrix(node_embeddings, edges):
    """
    根据边列表生成邻接矩阵。
    :param node_embeddings: {block_offset: embedding} 字典
    :param edges: [(src, dst), ...] 边列表
    :return: 邻接矩阵
    """
    num_nodes = len(node_embeddings)
    node_to_idx = {node: idx for idx, node in enumerate(node_embeddings.keys())}
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src, dst in edges:
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            adjacency_matrix[i, j] = 1.0  # 有向图
            adjacency_matrix[j, i] = 1.0  # 如果需要无向图，添加这一行

    return adjacency_matrix


def compute_block_embedding(block, model):
    """
    计算基本块的嵌入向量。
    :param block: 基本块对象
    :param model: Word2Vec 模型
    :return: 基本块的嵌入向量
    """
    embeddings = []
    for insn in block:
        sorted_arguments = sorted(insn.arguments, key=lambda x: str(x))
        arguments_repr = ", ".join([str(arg) for arg in sorted_arguments])  # 转为字符串
        instruction_repr = f"{insn.insn.name}({arguments_repr})"  # 构造指令表示
        
        # 检查指令是否在模型词汇表中
        if instruction_repr in model.wv:
            embeddings.append(model.wv[instruction_repr])
        else:
            print(f"Instruction {instruction_repr} not found in vocabulary.")
            exit(1)
    
    # 如果有嵌入向量，返回平均值；否则返回零向量
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)


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

            # 解析基本块标签中的指令列表
            instructions = []
            for line in label.split("\\l"):
                if ": " in line:
                    # 提取指令部分
                    raw_instruction = line.split(": ", 1)[1].strip()

                    # 移除定义部分（如 %1508 =）
                    if "=" in raw_instruction:
                        raw_instruction = raw_instruction.split("=", 1)[1].strip()

                    # 对指令参数排序（标准化）
                    if "(" in raw_instruction and ")" in raw_instruction:
                        opcode, raw_args = raw_instruction.split("(", 1)
                        raw_args = raw_args.rstrip(")")
                        sorted_args = ", ".join(sorted(raw_args.split(", ")))
                        instruction = f"{opcode}({sorted_args})"
                    else:
                        instruction = raw_instruction  # 无参数的指令

                    instructions.append(instruction)

            instruction_sequences[block_offset] = instructions

    return instruction_sequences