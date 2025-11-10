import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import EdgeConv
from torch.nn import Linear

params = {'a1': 0, 'a2': 2, 'a3': 1, 'a4': 2, 'bs': 1, 'd1': 0.015105134306121593, 'd2': 0.3431295462686682, \
      'd3': 0.602688496976768, 'd4': 0.9532038077650021, 'e1': 256.0, 'eact1': 0, 'edo1': 0.4813038851902818,\
      'f1': 256.0, 'f2': 256.0, 'f3': 160.0, 'f4': 24.0, 'g1': 256.0, 'g2': 320.0, 'g21': 448.0,\
      'g22': 512.0, 'gact1': 2, 'gact2': 2, 'gact21': 2, 'gact22': 0, 'gact31': 2, 'gact32': 1, 'gact33': 1,\
      'gdo1': 0.9444250299450242, 'gdo2': 0.8341272742321129, 'gdo21': 0.7675340644596443,\
      'gdo22': 0.21498171859119775, 'gdo31': 0.8236003195596049, 'gdo32': 0.6040220843354102,\
      'gdo33': 0.21007469160431758, 'lr': 0, 'nfc': 0, 'ngl': 1, 'opt': 0}

my_params = params


act = {0: torch.nn.ReLU(), 1:torch.nn.SELU(), 2:torch.nn.Sigmoid()}

class GNN(torch.nn.Module):
    
    def __init__(self, n_features):
        super(GNN, self).__init__()
        self.n_features = n_features
        self.gcn1 = GCNConv(self.n_features, int(params['g1']), cached=False) 
        self.gcn2 = GCNConv( int(params['g1']), int(params['g2']), cached=False)
        self.gcn21 = GCNConv( int(params['g2']), int(params['g21']), cached=False)
        self.gcn22 = GCNConv( int(params['g21']), int(params['g22']), cached=False)

        self.gcn31 = GCNConv(int(params['g2']), int(params['e1']), cached=False)
        self.gcn32 = GCNConv(int(params['g21']), int(params['e1']), cached=False)
        self.gcn33 = GCNConv(int(params['g22']), int(params['e1']), cached=False)

        self.gdo1 = nn.Dropout(p = params['gdo1'] )
        self.gdo2 = nn.Dropout(p = params['gdo2'] )
        self.gdo31 = nn.Dropout(p = params['gdo31'] )
        self.gdo21 = nn.Dropout(p = params['gdo21'] )
        self.gdo32 = nn.Dropout(p = params['gdo32'] )
        self.gdo22 = nn.Dropout(p = params['gdo22'] )
        self.gdo33 = nn.Dropout(p = params['gdo33'] )

        self.gact1 = act[params['gact1'] ]
        self.gact2 = act[params['gact2'] ]
        self.gact31 = act[params['gact31']] 
        self.gact21 = act[params['gact21'] ]
        self.gact32 = act[params['gact32'] ]
        self.gact22 = act[params['gact22'] ]
        self.gact33 = act[params['gact33'] ]

        self.ecn1 = EdgeConv(nn = nn.Sequential(nn.Linear(n_features*2, int(params['e1']) ),
                                          nn.ReLU(), 
                                          nn.Linear( int(params['e1']) , int(params['f1'])  ),))

        self.edo1 = nn.Dropout(p = params['edo1'] )
        self.eact1 = act[params['eact1'] ]


        self.fc1 = Linear( int(params['e1'])+ int(params['f1']), int(params['f1']))
        self.dropout1 = nn.Dropout(p = params['d1'] )
        self.act1 = act[params['a1']]

        self.fc2 = Linear(int(params['f1']), int(params['f2']))
        self.dropout2 = nn.Dropout(p = params['d2'] )
        self.act2 = act[params['a2']]

        self.fc3 = Linear(int(params['f2']), int(params['f3']))
        self.dropout3 = nn.Dropout(p = params['d3'] )
        self.act3 = act[params['a3']]

        self.fc4 = Linear(int(params['f3']), int(params['f4']))
        self.dropout4 = nn.Dropout(p = params['d4'] )
        self.act4 = act[params['a4']]

        self.out2 = Linear(int(params['f2']), 1)
        self.out3 = Linear(int(params['f3']), 1)
        self.out4 = Linear(int(params['f4']), 1)


    def forward(self, data):
        node_x, edge_x, edge_index = data.x, data.edge_attr, data.edge_index

        x1 = self.gdo1(self.gact1( self.gcn1( node_x, edge_index ) ) )
        x1 = self.gdo2(self.gact2(self.gcn2(x1, edge_index)) )
        x1 = self.gdo21(self.gact21(self.gcn21(x1, edge_index)) )
        x1 = self.gdo32(self.gact32(self.gcn32(x1, edge_index)) )
        
        x2 = self.edo1(self.eact1(self.ecn1(node_x, edge_index)) )
        x3 = torch.cat((x1,x2), 1)
        x3 = global_add_pool(x3, data.batch)
        x3 = self.dropout1(self.act1(self.fc1( x3 )))
        x3 = self.dropout2(self.act2(self.fc2( x3 )))
        x3 = self.out2(x3)
        return x3

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

