import numpy as np
from gensim.models import Word2Vec
import pydot
import rattle
from typing import Sequence
import tempfile
import os
import subprocess
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # 确保使用新版 DataLoader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .rgcn import RGCN  # 导入 RGCN 模型
from .rgcn import AttentionPooling
import torch.nn.functional as F
from .tsne import tsne_visualization
import pickle
def process_second_feature(ssa):
    # 处理第二个特征 
    instruction_sequences = {}  # 存储所有函数的指令序列
    for function in sorted(ssa.functions, key=lambda f: f.offset):
        cfg = rattle.ControlFlowGraph(function)
        # 获取 DOT 表示
        dot_content = cfg.dot()

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

    # 输出提取的指令序列
    # 打印整合后的所有指令序列
    # print("Number of instruction sequences:", len(training_data))
    # print("Example sequences:")
    # for i, seq in enumerate(training_data):  # 打印前5个序列
    #     print(f"Sequence {i + 1}: {seq}")
    # for function, sequences in instruction_sequences.items():
    #     print(f"Function at offset {hex(function.offset)}:")
    #     for block_offset, instructions in sequences.items():
    #         print(f"  Basic Block {hex(block_offset)}:")
    #         for insn in instructions:
    #             print(f"    {insn}")
    # # 生成 SDG
    ssa_functions = sorted(ssa.functions, key=lambda f: f.offset)
    function_embeddings, adjacency_matrices = generate_sdg(ssa_functions, "word2vec.model")

    # 打印函数嵌入和邻接矩阵
    print("--- Printing Function Embeddings and Adjacency Matrices ---")
    for function_key, embeddings in function_embeddings.items():
        print(f"Function: {function_key}")
        print("Node Embeddings:")
        for block_offset, embedding in embeddings.items():
            print(f"  Block {hex(block_offset)}: {embedding[:5]}...")  # 仅打印嵌入的前5个维度
        print("Adjacency Matrix:")
        edge_index, edge_type = adjacency_matrices[function_key]
        print(f"  Edge Index (2, E):\n{edge_index}")
        print(f"  Edge Types (E):\n{edge_type}")
        print("-" * 50)

    # # 准备 RGCN 数据
    rgcn_data_list = prepare_rgcn_data(function_embeddings, adjacency_matrices)
    print("--- Printing rgcn_data_list ---")
    for i, data in enumerate(rgcn_data_list):
            print(f"Graph {i}")
            print(f"  Node Features Shape: {data.x.shape}")
            print(f"  Edge Index Shape: {data.edge_index.shape}")
            print(f"  Edge Types Shape: {data.edge_type.shape}")
            print("-" * 50)
    # 创建 DataLoader
    loader = DataLoader(rgcn_data_list, batch_size=len(rgcn_data_list), shuffle=True)
    for data in loader:
        print("--- DataLoader Output ---")
        print(f"Batch Tensor (data.batch): {data.batch}")
        print(f"Node Features Shape (data.x): {data.x.shape}")
        print(f"Edge Index Shape (data.edge_index): {data.edge_index.shape}")
        print(f"Edge Types Shape (data.edge_type): {data.edge_type.shape}")
        assert data.edge_index.max().item() < data.x.shape[0], "Edge index out of node range!"

    in_channels = 200
    hidden_channels = 64
    out_channels = 200
    num_relations = 2
    num_epochs = 1000
    model = RGCN(in_channels, hidden_channels, out_channels, num_relations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # 调用训练函数
    all_node_embeddings = train_and_get_node_and_graph_embeddings(
        loader, model, optimizer, num_epochs, out_channels=out_channels
    )
    # 遍历每个图，输出节点嵌入
    # print("--- Printing Function Node Embeddings ---")
    # for i, data in enumerate(loader):
    #     for graph_idx in data.batch.unique():
    #         # 获取属于该图的节点
    #         graph_nodes = (data.batch == graph_idx).nonzero(as_tuple=True)[0]
    #         graph_node_embeddings = all_node_embeddings[graph_nodes]

    #         print(f"Function Graph {i}-{graph_idx.item()}:")
    #         print("Node Embeddings:")
    #         for node_idx, embedding in enumerate(graph_node_embeddings):
    #             print(f"  Node {node_idx}: {embedding[:5].tolist()}...")  # 仅打印前5个维度
    #         print("Adjacency Matrix:")
    #         filtered_edges_mask = torch.tensor([
    #             src in graph_nodes.tolist() and dst in graph_nodes.tolist()
    #             for src, dst in data.edge_index.T.tolist()
    #         ])
    #         filtered_edges = data.edge_index[:, filtered_edges_mask]
    #         filtered_edge_types = data.edge_type[filtered_edges_mask]
    #         print(f"  Edge Index (2, E):\n{filtered_edges}")
    #         print(f"  Edge Types (E):\n{filtered_edge_types}")
    #         print("-" * 50)
    graph_embeddings = []  # 存储图嵌入

    # 遍历批次，计算图嵌入
    for data in loader:
        for graph_idx in data.batch.unique():
            # 提取属于当前图的节点
            graph_nodes = (data.batch == graph_idx).nonzero(as_tuple=True)[0]
            graph_node_embeddings = all_node_embeddings[graph_nodes]

            # 生成图嵌入：平均池化
            graph_embedding = torch.mean(graph_node_embeddings, dim=0)
            graph_embeddings.append(graph_embedding.unsqueeze(0))  # 保持维度一致

    # 汇总所有图的嵌入
    graph_embeddings = torch.cat(graph_embeddings, dim=0)  # 形状 [num_graphs, embedding_dim]

    # 打印图嵌入信息
    print("Graph Embeddings Shape:", graph_embeddings.shape)
    print("Graph Embeddings:", graph_embeddings)

    # 计算图相似性矩阵
    similarity_matrix = compute_graph_similarity(graph_embeddings)
    print("Graph Similarity Matrix:", similarity_matrix)

    # 打印图间相似度
    num_graphs = similarity_matrix.size(0)
    print("\nGraph Pairwise Similarities:")
    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print(f"Similarity between Graph {i} and Graph {j}: {similarity_matrix[i, j].item() * 100:.4f}%")

def compute_graph_similarity(graph_embeddings):
    """
    基于图嵌入计算图相似性矩阵，并调整为非负值。
    """
    # 归一化图嵌入以计算余弦相似性
    normalized_embeddings = F.normalize(graph_embeddings, p=2, dim=1)  # L2 归一化
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)  # 余弦相似度

    # 调整相似度范围到 [0, 1]
    similarity_matrix = (similarity_matrix + 1) / 2
    return similarity_matrix

def train_and_get_node_and_graph_embeddings(loader, model, optimizer, num_epochs, out_channels):
    """
    训练 RGCN 模型并获取节点和图嵌入。
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0

        for data in loader:
            optimizer.zero_grad()

            # 数据增强：添加随机噪声和随机边
            # data.x += torch.randn_like(data.x) * 0.005  # 加入小的随机噪声
            # num_edges_to_add = data.edge_index.size(1) // 10
            # new_edges = torch.randint(0, data.x.size(0), (2, num_edges_to_add))
            # new_edge_types = torch.zeros(num_edges_to_add, dtype=torch.long)  # 新边类型为 0
            # data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
            # data.edge_type = torch.cat([data.edge_type, new_edge_types], dim=0)

            # 模型前向传播
            node_embeddings = model(data.x, data.edge_index, data.edge_type)

            # 边类型约束损失
            loss_edge = compute_edge_loss(node_embeddings, data.edge_index, data.edge_type)

            # 总损失
            loss = loss_edge
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印训练信息
        if epoch % 250 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")

    return node_embeddings



def contrastive_loss(embeddings, labels, margin=1.0):
    """
    对比学习损失：
    - 同图的节点嵌入应该更接近。
    - 不同图的节点嵌入应该远离。
    """
    distances = torch.cdist(embeddings, embeddings, p=2)  # 计算节点之间的欧几里得距离
    positive_pairs = (labels[:, None] == labels[None, :]).float()  # 同图的节点对
    negative_pairs = 1 - positive_pairs

    positive_loss = positive_pairs * torch.square(distances)  # 同图距离应该小
    negative_loss = negative_pairs * torch.clamp(margin - distances, min=0)  # 跨图距离应该大

    return positive_loss.mean() + negative_loss.mean()


def compute_edge_loss(node_embeddings, edge_index, edge_type):
    """
    边类型约束损失：
    - 控制流或数据流相邻节点应该更相似。
    """
    src, dst = edge_index
    edge_distances = torch.norm(node_embeddings[src] - node_embeddings[dst], dim=1)
    edge_weights = torch.ones_like(edge_type, dtype=torch.float)  # 默认权重为 1
    edge_weights[edge_type == 0] = 1.5  # 控制依赖边权重更高
    edge_weights[edge_type == 1] = 1  # 数据依赖边权重较低
    weighted_distances = edge_weights * edge_distances
    edge_loss = torch.mean(weighted_distances)  # 相邻节点的距离应该小
    return edge_loss


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
    edge_types = {}
    # # 加载 Word2Vec 模型
    model = Word2Vec.load(model_path)
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
                for path_idx, path in enumerate(paths):
                    func_embedding = {}
                    edges = {"control": [], "data": []}
                    # 当前路径的基本块集合
                    path_blocks = set(path)
                    # 提取当前路径的基本块嵌入
                    for block in path_blocks:
                        if block.offset not in func_embedding:  # 避免重复计算
                            func_embedding[block.offset] = compute_block_embedding(block, model)
                        # 获取控制依赖边和数据依赖边
                        pdg = call_backtracking.sdg.function_pdgs[func]
                        edges["control"].extend([
                            (edge[0].offset, edge[1].offset)
                                for edge in pdg.control_edges
                                if edge[0] in path and edge[1] in path
                            ])
                        edges["data"].extend([
                            (edge[0].offset, edge[1].offset)
                                for edge in pdg.data_edges
                                if edge[0] in path and edge[1] in path
                            ])
                        
                        # 打印调试信息
                        # print(f"Control Edges for {function_key}: {edges['control']}")
                        # print(f"Data Edges for {function_key}: {edges['data']}")

                        # 打印调试信息
                        # print(f"Control Edges for {function_key}: {edges['control']}")
                        # print(f"Data Edges for {function_key}: {edges['data']}")
                        # for i in range(len(path) - 1):
                        #     edges.append((path[i].offset, path[i + 1].offset))

                    function_key = f"{func.desc()}:CallBacktrack_{path_idx}"
                    function_embeddings[function_key] = func_embedding
                    edge_index, edge_type = generate_weighted_adjacency_matrix_with_types(
                        func_embedding, edges["control"], edges["data"]
                    )
                    adjacency_matrices[function_key] = (edge_index, edge_type)
    return function_embeddings , adjacency_matrices


def prepare_rgcn_data(function_embeddings, adjacency_matrices):
    data_list = []
    for func_key in function_embeddings.keys():
        # print(f"Processing Function: {func_key}")
        node_features = torch.tensor(
            list(function_embeddings[func_key].values()), dtype=torch.float
        )
        edge_index, edge_type = adjacency_matrices[func_key]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        # print(f"  Node Features Shape: {node_features.shape}")
        # print(f"  Edge Index Shape: {edge_index.shape}")
        # print(f"  Edge Types Shape: {edge_type.shape}")
        data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)
        data_list.append(data)
    return data_list

def generate_weighted_adjacency_matrix_with_types(node_embeddings, control_edges, data_edges):
    """
    根据控制依赖边和数据依赖边生成带类型的边表示。
    :param node_embeddings: {block_offset: embedding} 字典
    :param control_edges: [(src, dst), ...] 控制依赖边列表
    :param data_edges: [(src, dst), ...] 数据依赖边列表
    :return: (edge_index, edge_type)
        edge_index: (2, E) 表示边的源节点和目标节点。
        edge_type: (E,) 表示边的类型，0 为控制依赖，1 为数据依赖。
    """
    node_to_idx = {node: idx for idx, node in enumerate(node_embeddings.keys())}  # 映射节点到索引
    edge_set = set()  # 用于去重

    edge_index = []
    edge_type = []

    # 添加控制依赖边
    for src, dst in control_edges:
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            if (i, j, 0) not in edge_set:  # 避免重复
                edge_index.append((i, j))
                edge_type.append(0)  # 0 表示控制依赖
                edge_set.add((i, j, 0))

    # 添加数据依赖边
    for src, dst in data_edges:
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            if (i, j, 1) not in edge_set:  # 避免重复
                edge_index.append((i, j))
                edge_type.append(1)  # 1 表示数据依赖
                edge_set.add((i, j, 1))

    # 转置为 (2, E) 格式
    edge_index = np.array(edge_index).T  # shape: (2, E)

    return edge_index, edge_type


def compute_block_embedding(block, model):
    """
    计算基本块的嵌入向量。
    :param block: 基本块对象
    :param model: Word2Vec 模型
    :return: 基本块的嵌入向量
    """
    embeddings = []
    for insn in block:
        opcode = insn.insn.name
       
        # 检查操作码是否在模型词汇表中
        if opcode in model.wv:
            embeddings.append(model.wv[opcode])
        else:
            print(f"Opcode {opcode} not found in vocabulary.")  # 调试信息
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




