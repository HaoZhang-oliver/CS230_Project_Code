data_dir = "../data/"
gnn_data_dir = "../data/"
accfg_vocab_json = data_dir + "accfg_vocab.json"
best_model_fg = "best_model_fg_gat.pt"
fg_pkl_suffix_candidates = ["_fg.pkl.gz", "_withFG.pkl.gz"]

n_features = 65
bs = 128
max_epochs = 500
lr = 1e-2
l2_lambda = 1e-4
patience=25
best_model1 = 'baseline_gnn_best1.pt'
best_model2 = 'baseline_gnn_best2.pt'
best_model3 = 'baseline_gnn_best3.pt'
best_model_gat_edgefeature = 'gat_edgefeature_best.pt'
best_model_FG_gat = 'FG_gat_best.pt'