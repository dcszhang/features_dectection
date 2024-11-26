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

from .rgcn import RGCN  # 导入 RGCN 模型
from .rgcn import AttentionPooling
import torch.nn.functional as F
from .tsne import tsne_visualization
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
    # 检查训练数据

    # 初始化并训练 Word2Vec 模型
    model = Word2Vec(
        sentences=training_data,
        vector_size=16,    # 嵌入向量维度
        window=2,# 上下文窗口大小
        min_count=1,       # 最小频率
        sg=1,              # Skip-Gram 方法
        compute_loss=True, # 启用损失计算
        alpha=0.03,  # 初始学习率
        min_alpha=0.0007,  # 最小学习率
        negative=100,  # 增加负采样数量
        epochs=200          # 总迭代次数
    )
    tsne_visualization(model)

    # 保存模型
    model.save("cfg_word2vec.model")
    # # 生成 SDG
    ssa_functions = sorted(ssa.functions, key=lambda f: f.offset)
    function_embeddings, adjacency_matrices = generate_sdg(ssa_functions, "cfg_word2vec.model")

    # 打印函数嵌入和邻接矩阵
    print("--- Printing Function Embeddings and Adjacency Matrices ---")
    for function_key, embeddings in function_embeddings.items():
        print(f"Function: {function_key}")
        
        # 嵌入信息统计
        all_embeddings = np.array(list(embeddings.values()))
        print(f"  Total Nodes: {len(embeddings)}")
        print(f"  Embedding Mean: {all_embeddings.mean(axis=0)[:5]}...")
        print(f"  Embedding Std Dev: {all_embeddings.std(axis=0)[:5]}...")
        
        # 邻接矩阵
        edge_index, edge_type = adjacency_matrices[function_key]
        print(f"  Total Edges: {len(edge_type)}")
        print("  Edge Types Distribution:", {t: edge_type.count(t) for t in set(edge_type)})
        
        # 打印前 5 条边
        print("  Sample Edges:")
        for i in range(min(5, len(edge_type))):
            print(f"    Edge: {edge_index[0][i]} -> {edge_index[1][i]}, Type: {edge_type[i]}")
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
    # for data in loader:
    #     print("--- DataLoader Output ---")
    #     print(f"Batch Tensor (data.batch): {data.batch}")
    #     print(f"Node Features Shape (data.x): {data.x.shape}")
    #     print(f"Edge Index Shape (data.edge_index): {data.edge_index.shape}")
    #     print(f"Edge Types Shape (data.edge_type): {data.edge_type.shape}")
    #     assert data.edge_index.max().item() < data.x.shape[0], "Edge index out of node range!"

    #     break  # 检查第一个批次


    in_channels = 16
    hidden_channels = 64
    out_channels = 16
    num_relations = 2
    num_epochs = 100
    model = RGCN(in_channels, hidden_channels, out_channels, num_relations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    # 调用训练函数
    all_node_embeddings, graph_embeddings, topology_features = train_and_get_node_and_graph_embeddings(
        loader, model, optimizer, num_epochs, out_channels=out_channels
    )

    # # Check graph embeddings
    print("Graph Embeddings Shape:", graph_embeddings.shape)
    print("Graph Embeddings:", graph_embeddings)

    similarity_matrix = compute_graph_similarity(graph_embeddings, topology_features)
    print("Graph Similarity Matrix:", similarity_matrix)

    # Print pairwise similarities
    num_graphs = similarity_matrix.size(0)
    print("\nGraph Pairwise Similarities:")
    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            print(f"Similarity between Graph {i} and Graph {j}: {similarity_matrix[i, j].item() * 100:.4f}%")


def train_and_get_node_and_graph_embeddings(loader, model, optimizer, num_epochs, out_channels):
    """
    训练模型并获取节点和图嵌入。
    """
    model.train()
    attention_pool = AttentionPooling(in_features=out_channels, num_heads=4)

    all_graph_topology_features = []

    for epoch in range(num_epochs):
        total_loss = 0
        all_node_embeddings = []
        graph_embeddings = []
        graph_topology_features = []  # 每轮需要清空
        for data in loader:
            optimizer.zero_grad()
            # 前向传播
            out = model(data.x, data.edge_index, data.edge_type)

            loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("Full Batch Tensor (data.batch):", data.batch)
            for graph_idx in data.batch.unique():
                graph_nodes = (data.batch == graph_idx).nonzero(as_tuple=True)[0]
                print(f"Graph {graph_idx}: Node Count = {graph_nodes.shape[0]}")
                print(f"Graph {graph_idx}: Node Indices = {graph_nodes.tolist()}")
        
                graph_node_embeddings = out[graph_nodes]
                all_node_embeddings.append(graph_node_embeddings)

                graph_nodes_set = set(graph_nodes.tolist())
                filtered_edges_mask = torch.tensor([
                    src in graph_nodes_set and dst in graph_nodes_set
                    for src, dst in data.edge_index.T.tolist()
                ])
                filtered_edges = data.edge_index[:, filtered_edges_mask]
                print(f"Graph {graph_idx}: Edge Count = {filtered_edges.shape[1]}")
                print(f"Graph {graph_idx}: Edges = {filtered_edges.tolist()}")

                # 计算拓扑特征
                num_nodes = graph_node_embeddings.size(0)
                num_edges = filtered_edges.size(1)
                avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
                density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
                graph_features = torch.tensor([num_nodes, num_edges, avg_degree, density], dtype=torch.float)

                # 保存拓扑特征
                graph_topology_features.append(graph_features)

                # 使用注意力池化计算图嵌入
                graph_embedding = attention_pool(graph_node_embeddings, graph_features)
                graph_embeddings.append(graph_embedding.unsqueeze(0))


        # 汇总所有图嵌入
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        all_graph_topology_features = graph_topology_features  # 不追加，只保留最新值
        if(epoch % 10 == 0):
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")
    return all_node_embeddings, graph_embeddings, torch.stack(all_graph_topology_features, dim=0)


def compute_topology_features(edge_index, num_nodes):
    """
    Compute simple topology features for a graph.
    """
    degrees = torch.bincount(edge_index[0], minlength=num_nodes)
    degree_mean = degrees.float().mean()
    degree_max = degrees.float().max()
    return torch.tensor([degree_mean, degree_max], dtype=torch.float)

def aggregate_node_embeddings(node_embeddings , batch_data_list):
    """
    聚合节点嵌入为图嵌入。
    """
    graph_embeddings = []
    for idx, graph_embedding in enumerate(node_embeddings):
        # 聚合每个图的节点嵌入为单个图嵌入
        aggregated_embedding = graph_embedding.mean(dim=0)  # 使用均值聚合
        graph_embeddings.append(aggregated_embedding)
        print(f"Graph {idx} Aggregated Embedding: {aggregated_embedding}")  # 打印每个图的嵌入
    # 返回形状为 (num_graphs, embedding_dim) 的张量
    return torch.stack(graph_embeddings, dim=0)


from torch.nn.functional import cosine_similarity

import torch.nn.functional as F

def compute_graph_similarity(graph_embeddings, topology_features, alpha=0.5):
    """
    计算图相似性，结合嵌入和拓扑特征。
    :param graph_embeddings: 图嵌入 (N, embedding_dim)
    :param topology_features: 图拓扑特征 (N, num_features)
    :param alpha: 嵌入与拓扑特征的权重系数，范围 [0, 1]
    :return: final_similarity: 图相似性矩阵 (N, N)
    """
    # 确保输入维度正确
    assert graph_embeddings.size(0) == topology_features.size(0), "图嵌入和拓扑特征的图数量不匹配"
    
    # 步骤 1: 嵌入相似性
    normalized_embeddings = F.normalize(graph_embeddings, p=2, dim=1)  # 对图嵌入归一化
    embedding_similarity = torch.mm(normalized_embeddings, normalized_embeddings.T)  # 余弦相似度 (N, N)
    
    # 步骤 2: 拓扑特征相似性
    normalized_topology = F.normalize(topology_features, p=2, dim=1)  # 对拓扑特征归一化
    topology_similarity = torch.mm(normalized_topology, normalized_topology.T)  # 余弦相似度 (N, N)
    
    # 步骤 3: 结合嵌入相似性和拓扑特征相似性
    final_similarity = alpha * embedding_similarity + (1 - alpha) * topology_similarity

    return final_similarity



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

def contrastive_loss(embeddings, margin=1.0):
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    num_graphs = embeddings.size(0)
    loss = 0.0

    for i in range(num_graphs):
        for j in range(num_graphs):
            if i != j:
                target = 1.0 if i == j else 0.0
                distance = 1.0 - similarity_matrix[i, j]
                if target == 1.0:
                    loss += distance**2
                else:
                    loss += torch.clamp(margin - distance, min=0)**2
    return loss / (num_graphs * (num_graphs - 1))

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




