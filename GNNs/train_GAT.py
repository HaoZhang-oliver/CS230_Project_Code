import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from rdkit.Chem import Draw
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pandas as pd
from random import randrange
import itertools
import random
import os
from pickle import dump, load
from sklearn.metrics import mean_absolute_error
import pickle
import gzip, pickle
from torch_geometric.loader import DataLoader
import gnn_utils
import gnn_model
from gnn_model import BaselineGAT_Edge
import config
import datetime

def run_gat_edge():

    print("start load data")
    with gzip.open(f"{config.data_dir}train.pkl.gz", "rb") as f:
        train_X = pickle.load(f)
    with gzip.open(f"{config.data_dir}val.pkl.gz", "rb") as f:
        val_X = pickle.load(f)
    with gzip.open(f"{config.data_dir}test.pkl.gz", "rb") as f:
        test_X = pickle.load(f)

    n_features = int(train_X[0].x.size(1))
    edge_dim = int(train_X[0].edge_attr.size(1))

    bs = config.bs
    train_loader = DataLoader(train_X, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_X, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_X, batch_size=bs, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("start define model")
    model = BaselineGAT_Edge(n_features=n_features,edge_dim=edge_dim,hidden_dim=128,num_layers=3,heads=4,dropout=0.1,pool="add").to(device)

    print(f"Total params: {gnn_utils.count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    early_stopping = gnn_utils.EarlyStopping(patience = config.patience, verbose=True, chkpoint_name = config.best_model_gat_edge)

    n_epochs = config.max_epochs
    hist = {"train_rmse": [], "val_rmse": []}

    print("start train")
    for epoch in range(n_epochs):
        model.train()

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data).reshape(-1,)
            loss = criterion(output, data.y)

            loss.backward()
            optimizer.step()

        train_rmse = gnn_utils.test_fn(train_loader, model, device)
        val_rmse = gnn_utils.test_fn(val_loader, model, device)

        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)
        print(f"Epoch: {epoch}, Train_rmse: {train_rmse:.3}, Val_rmse: {val_rmse:.3}")

    print(f"training completed at {datetime.datetime.now()}")
    return hist


if __name__ == "__main__":
    run_gat_edge()