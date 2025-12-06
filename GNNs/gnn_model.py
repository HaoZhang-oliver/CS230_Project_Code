import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import EdgeConv
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
import torch.nn.functional as F

class BaselineGCN1(nn.Module):

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

        x = global_add_pool(x, batch)
        x = self.fc_drop(self.act(self.fc1(x)))
        x = self.out(x)
        return x

class BaselineGCN2(nn.Module):

    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features

        self.gcn1 = GCNConv(n_features, 256, cached=False)
        self.gcn2 = GCNConv(256, 256, cached=False)
        self.gcn3 = GCNConv(256, 256, cached=False)
        self.gcn4 = GCNConv(256, 256, cached=False)

        self.gact = nn.ReLU()
        self.gdo1 = nn.Dropout(p=dropout)
        self.gdo2 = nn.Dropout(p=dropout)
        self.gdo3 = nn.Dropout(p=dropout)
        self.gdo4 = nn.Dropout(p=dropout)

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

        self.fc1 = nn.Linear(256 + 256, 256)
        self.hact1 = nn.ReLU()
        self.hdrop1 = nn.Dropout(p=dropout)
        self.out = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch 

        xg = self.gdo1(self.gact(self.gcn1(x, edge_index)))
        xg = self.gdo2(self.gact(self.gcn2(xg, edge_index)))
        xg = self.gdo3(self.gact(self.gcn3(xg, edge_index)))
        xg = self.gdo4(self.gact(self.gcn4(xg, edge_index)))

        xe = self.eact1(self.ecn1(x, edge_index))
        xe = self.edo1(xe)
        xe = self.eact1(self.ecn2(xe, edge_index))
        xe = self.edo2(xe)

        x_cat = torch.cat([xg, xe], dim=1)
        g = global_add_pool(x_cat, batch)

        g = self.hdrop1(self.hact1(self.fc1(g)))
        y = self.out(g)
        return y

class BaselineGAT_Edge(nn.Module):

    def __init__(
        self,
        n_features: int,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        pool: str = "add",   
    ):
        super().__init__()
        assert num_layers >= 2
        assert pool in ["add", "mean"]

        self.pool_fn = global_add_pool if pool == "add" else global_mean_pool
        self.act = nn.ELU()
        self.drop = nn.Dropout(p=dropout)

        self.convs = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_channels=n_features,
                out_channels=hidden_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )

        self.convs.append(
            GATv2Conv(
                in_channels=hidden_dim * heads,
                out_channels=hidden_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        )

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc_drop = nn.Dropout(p=dropout)
        self.out = Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)  
            x = self.act(x)
            x = self.drop(x)

        g = self.pool_fn(x, batch)                 
        g = self.fc_drop(self.act(self.fc1(g)))    
        y = self.out(g)                            
        return y

class FGEncodedGAT(nn.Module):

    def __init__(
        self,
        n_atom_features,
        fg_vocab_size,
        edge_dim=6,
        hidden=128,
        heads=4,
        num_atom_layers=3,
        num_fg_layers=2,
        fg_hidden=128,
        fg_heads=4,
        fg_type_emb_dim=64,
        dropout=0.2,
        pool="add",
    ):
        super().__init__()
        self.dropout = dropout

        if pool == "mean":
            self.pool_fn = global_mean_pool
        else:
            self.pool_fn = global_add_pool

        self.atom_convs = nn.ModuleList()
        self.atom_convs.append(GATv2Conv(n_atom_features, hidden, heads=heads, edge_dim=edge_dim))
        for _ in range(num_atom_layers - 2):
            self.atom_convs.append(GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim))
        self.atom_convs.append(GATv2Conv(hidden * heads, hidden, heads=1, concat=False, edge_dim=edge_dim))
        self.atom_fc = nn.Linear(hidden, hidden)

        self.fg_type_emb = nn.Embedding(fg_vocab_size, fg_type_emb_dim)
        self.fg_type_proj = nn.Linear(fg_type_emb_dim, fg_hidden)
        self.atom2fg_proj = nn.Linear(n_atom_features, fg_hidden)

        self.fg_convs = nn.ModuleList()
        self.fg_convs.append(GATv2Conv(fg_hidden, fg_hidden, heads=fg_heads, edge_dim=edge_dim))
        for _ in range(num_fg_layers - 2):
            self.fg_convs.append(GATv2Conv(fg_hidden * fg_heads, fg_hidden, heads=fg_heads, edge_dim=edge_dim))
        self.fg_convs.append(GATv2Conv(fg_hidden * fg_heads, fg_hidden, heads=1, concat=False, edge_dim=edge_dim))
        self.fg_fc = nn.Linear(fg_hidden, fg_hidden)

        self.fuse_fc = nn.Linear(hidden + fg_hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = x
        for conv in self.atom_convs:
            h = conv(h, edge_index, edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        g_atom = self.pool_fn(h, batch)
        g_atom = F.elu(self.atom_fc(g_atom))
        g_atom = F.dropout(g_atom, p=self.dropout, training=self.training)

        B = int(data.num_graphs)

        if (not hasattr(data, "fg_type_ids")) or data.fg_type_ids.numel() == 0:
            g_fg = torch.zeros((B, self.fg_fc.out_features), device=x.device, dtype=x.dtype)
        else:
            fg_type_ids = data.fg_type_ids
            fg_edge_index = data.fg_edge_index
            fg_edge_attr = data.fg_edge_attr
            atom2fg = data.atom2fg_edge_index

            fg_batch = getattr(data, "fg_type_ids_batch", None)
            if fg_batch is None:
                raise RuntimeError(
                    "Missing fg_type_ids_batch. Use PyG DataLoader(..., follow_batch=['fg_type_ids'])."
                )

            N_fg = int(fg_type_ids.size(0))

            fg_x = self.fg_type_proj(self.fg_type_emb(fg_type_ids))

            if atom2fg.numel() > 0:
                atom_idx = atom2fg[0]
                fg_idx = atom2fg[1]
                atom_msg = self.atom2fg_proj(x)
                fg_atom = scatter_mean(atom_msg[atom_idx], fg_idx, dim=0, dim_size=N_fg)
                fg_x = fg_x + fg_atom

            z = fg_x
            for conv in self.fg_convs:
                z = conv(z, fg_edge_index, fg_edge_attr)
                z = F.elu(z)
                z = F.dropout(z, p=self.dropout, training=self.training)

            g_fg = self.pool_fn(z, fg_batch, size=B)
            g_fg = F.elu(self.fg_fc(g_fg))
            g_fg = F.dropout(g_fg, p=self.dropout, training=self.training)

        g = torch.cat([g_atom, g_fg], dim=1)
        g = F.elu(self.fuse_fc(g))
        g = F.dropout(g, p=self.dropout, training=self.training)

        y = self.out(g).view(-1)
        return y