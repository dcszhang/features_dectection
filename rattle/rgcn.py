import torch
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, 64, num_relations, num_bases=4)
        self.conv2 = RGCNConv(64, out_channels, num_relations, num_bases=4)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x
