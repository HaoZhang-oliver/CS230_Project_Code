import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import EdgeConv
from torch.nn import Linear


class BaselineGCN1(nn.Module):
    """
    4-layer GCN baseline:
      - GCNConv × 4 (all 256-d), each followed by ReLU + Dropout
      - global_add_pool -> graph embedding (256-d)
      - FC(256->128) + ReLU + Dropout
      - FC(256->1) -> y (e.g., logS)
    """
    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features

        self.gcn1 = GCNConv(n_features, 256, cached=False)
        self.gcn2 = GCNConv(256, 256, cached=False)
        self.gcn3 = GCNConv(256, 256, cached=False)
        self.gcn4 = GCNConv(256, 256, cached=False)

        self.act = nn.ReLU()
        self.gdrop1 = nn.Dropout(p=dropout)
        self.gdrop2 = nn.Dropout(p=dropout)
        self.gdrop3 = nn.Dropout(p=dropout)
        self.gdrop4 = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(256, 256)
        self.fc_drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  

        x = self.gdrop1(self.act(self.gcn1(x, edge_index)))
        x = self.gdrop2(self.act(self.gcn2(x, edge_index)))
        x = self.gdrop3(self.act(self.gcn3(x, edge_index)))
        x = self.gdrop4(self.act(self.gcn4(x, edge_index)))

        # pool the graph embedding
        x = global_add_pool(x, batch)  # [batch_size, 256]

        # two FC layers to output y
        x = self.fc_drop(self.act(self.fc1(x)))   # 256 -> 256
        x = self.out(x)                           # 256 -> 1
        return x

class BaselineGCN2(nn.Module):
    """
    Baseline#2: 4×GCNConv(256) + 2×EdgeConv(256), concatenate at the end for graph-level regression.
      - GCN branch: GCNConv(·,256) ×4, each layer followed by ReLU + Dropout
      - Edge branch: EdgeConv ×2, each internal MLP: [2*in -> 256 -> 256], each layer followed by ReLU + Dropout
      - Fusion: Concatenate channels ('[GCN-256 || Edge-256]' → 512) → global_add_pool
      - Head: FC(512→256) + ReLU + Dropout → FC(256→1)
    Note: As in the original code, **edge_attr is not used** (only edge_index).
    """
    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features

        # ===== GCN branch (all 256) =====
        self.gcn1 = GCNConv(n_features, 256, cached=False)
        self.gcn2 = GCNConv(256, 256, cached=False)
        self.gcn3 = GCNConv(256, 256, cached=False)
        self.gcn4 = GCNConv(256, 256, cached=False)

        self.gact = nn.ReLU()
        self.gdo1 = nn.Dropout(p=dropout)
        self.gdo2 = nn.Dropout(p=dropout)
        self.gdo3 = nn.Dropout(p=dropout)
        self.gdo4 = nn.Dropout(p=dropout)

        # ===== EdgeConv branch (two layers, both 256) =====
        # EdgeConv's nn receives [x_i || (x_j - x_i)], so the input dimension is 2 * in_dim
        self.ecn1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(n_features * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ),
            aggr='max'
        )
        self.ecn2 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(256 * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ),
            aggr='max'
        )
        self.eact1 = nn.ReLU()
        self.edo1 = nn.Dropout(p=dropout)
        self.edo2 = nn.Dropout(p=dropout)

        # ===== Head: concat → pool → 2×FC =====
        self.fc1 = nn.Linear(256 + 256, 256)   # concat after EdgeConv is 512
        self.hact1 = nn.ReLU()
        self.hdrop1 = nn.Dropout(p=dropout)
        self.out = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch 

        # ---- GCN branch ----
        xg = self.gdo1(self.gact(self.gcn1(x, edge_index)))
        xg = self.gdo2(self.gact(self.gcn2(xg, edge_index)))
        xg = self.gdo3(self.gact(self.gcn3(xg, edge_index)))
        xg = self.gdo4(self.gact(self.gcn4(xg, edge_index)))   # [N, 256]

        # ---- EdgeConv branch (two layers) ----
        xe = self.eact1(self.ecn1(x, edge_index))
        xe = self.edo1(xe)
        xe = self.eact1(self.ecn2(xe, edge_index))
        xe = self.edo2(xe)                                     # [N, 256]

        # ---- Fuse & Readout ----
        x_cat = torch.cat([xg, xe], dim=1)                     # [N, 512]
        g = global_add_pool(x_cat, batch)                      # [B, 512]

        # ---- Head ----
        g = self.hdrop1(self.hact1(self.fc1(g)))               # [B, 256]
        y = self.out(g)                                        # [B, 1]
        return y

