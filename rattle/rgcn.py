import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn as nn
# RGCN Model
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=8)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=8)
        self.conv3 = RGCNConv(hidden_channels, out_channels, num_relations, num_bases=8)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_type)
        return x

# Attention-based Pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_features, num_heads=4):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(num_heads)])
        self.graph_feature_fc = nn.Sequential(
            nn.Linear(4, in_features),  # 输入维度从 3 改为 4，支持更多拓扑特征（如密度）
            nn.ReLU(),
            nn.Linear(in_features, in_features),  # 映射到与节点嵌入相同的维度
            nn.ReLU()
        )
        self.output_fc = nn.Linear(in_features * num_heads + in_features, in_features * 2)  # 最终输出

    def forward(self, x, graph_features):
        """
        x: 节点特征，形状为 (num_nodes, in_features)
        graph_features: 图的统计特征，形状为 (4,)（节点数、边数、平均度、密度）
        返回:
        图嵌入，形状为 (in_features * 2,)
        """
        # Step 1: 多头注意力池化
        head_embeddings = []
        for attention in self.attention_heads:
            scores = attention(x)  # 注意力得分，形状为 (num_nodes, 1)
            weights = torch.softmax(scores, dim=0)  # 归一化注意力
            head_embedding = torch.sum(weights * x, dim=0)  # 加权求和，形状为 (in_features,)
            head_embedding = torch.relu(head_embedding)  # 非线性激活
            head_embeddings.append(head_embedding)
        
        # 合并所有头的嵌入
        aggregated_embedding = torch.cat(head_embeddings, dim=-1)  # 形状为 (in_features * num_heads,)

        # Step 2: 处理图级统计特征
        graph_feature_embedding = self.graph_feature_fc(graph_features)  # 形状为 (in_features,)

        # Step 3: 拼接节点嵌入和图级特征
        combined_embedding = torch.cat([aggregated_embedding, graph_feature_embedding], dim=-1)

        # Step 4: 输出图嵌入
        return self.output_fc(combined_embedding)

