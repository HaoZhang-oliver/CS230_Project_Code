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
from gnn_model import FGEncodedGAT
import config
import datetime

def run_fggat():

    gnn_utils.set_seed(config.seed)

    train_X, fg2id = gnn_utils.load_split_with_fg("train")
    val_X, _ = gnn_utils.load_split_with_fg("val", vocab_json_path=config.accfg_vocab_json)
    test_X, _ = gnn_utils.load_split_with_fg("test", vocab_json_path=config.accfg_vocab_json)

    bs = config.bs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(train_X, batch_size=bs, shuffle=True, drop_last=True, follow_batch=["fg_type_ids"])
    val_loader = DataLoader(val_X, batch_size=bs, shuffle=False, drop_last=False, follow_batch=["fg_type_ids"])
    test_loader = DataLoader(test_X, batch_size=bs, shuffle=False, drop_last=False, follow_batch=["fg_type_ids"])

    train_loader_no_shuffle = DataLoader(train_X, batch_size=bs, shuffle=False, drop_last=False, follow_batch=["fg_type_ids"])
    val_loader_no_shuffle = DataLoader(val_X, batch_size=bs, shuffle=False, drop_last=False, follow_batch=["fg_type_ids"])

    n_atom_features = train_X[0].x.size(1)
    fg_vocab_size = len(fg2id)

    model = gnn_model.FGEncodedGAT(
        n_atom_features=n_atom_features,
        fg_vocab_size=fg_vocab_size,
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
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    early_stopping = gnn_utils.EarlyStopping(
        patience=config.patience, verbose=True, chkpoint_name=config.best_model_fg
    )

    n_epochs = config.max_epochs
    hist = {"train_rmse": [], "val_rmse": []}

    for epoch in range(0, n_epochs):
        model.train()

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)           
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        train_rmse = gnn_utils.test_fn(train_loader_no_shuffle, model, device)
        val_rmse = gnn_utils.test_fn(val_loader_no_shuffle, model, device)

        early_stopping(val_rmse, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)
        print(f"Epoch: {epoch}, Train_rmse: {train_rmse:.3}, Val_rmse: {val_rmse:.3}")

    print(f"training completed at {datetime.datetime.now()}")

    model.load_state_dict(torch.load(config.best_model_fg, map_location=device))
    gnn_utils.get_results("Test", test_loader, model, device)


if __name__ == "__main__":
    run_fggat()