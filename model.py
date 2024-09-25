import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_add


class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        super().__init__()

        # Initialize user and item embeddings
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        # Embedding layers for users and items
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        # Initialize embeddings with normal distribution
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    # Forward propagation
    def forward(self, edge_index: SparseTensor, num_users, num_items, prompt=None):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])

        if emb_0.shape[0] != num_users + num_items:
            emb_0 = torch.nn.functional.interpolate(emb_0.unsqueeze(0).unsqueeze(0), size=(num_users + num_items, emb_0.shape[1]),
                                                    mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        if prompt is not None:
            emb_0 = prompt.add(emb_0)
        embs = [emb_0]
        emb_k = emb_0
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        try:
            embs = torch.stack(embs, dim=1)
        except RuntimeError as e:
            print(f"Error in stacking tensors: {e}")
            for i, emb in enumerate(embs):
                print(f"Tensor {i} shape: {emb.shape}")

        emb_final = torch.mean(embs, dim=1)
        users_emb_final, items_emb_final = torch.split(emb_final, [num_users, num_items])

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j
    def message_and_aggregate(self, adj_t: SparseTensor, x: torch.Tensor) -> torch.Tensor:
        return matmul(adj_t, x)

# Normalization for the adjacency matrix
def gcn_norm(edge_index: SparseTensor, add_self_loops: bool = True):
    row, col, value = edge_index.coo()
    num_nodes = edge_index.sizes()[0]

    if value is None:
        value = torch.ones(row.size(0), device=row.device)

    if add_self_loops:
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=row.device)
        loop_value = torch.ones(num_nodes, dtype=value.dtype, device=value.device)
        row = torch.cat([row, loop_index])
        col = torch.cat([col, loop_index])
        value = torch.cat([value, loop_value])

    edge_weight = torch.ones((row.size(0),), dtype=value.dtype, device=value.device)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return SparseTensor(row=row, col=col, value=norm, sparse_sizes=edge_index.sizes())