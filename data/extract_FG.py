import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from tqdm import tqdm
import pickle
import datetime

from accfg import AccFG
import gzip
import json
from collections import defaultdict
from collections import deque


def load_pkl_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def save_pkl_gz(obj, path: Path):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _is_connected_subgraph(mol: Chem.Mol, atom_ids):

    atom_ids = list(atom_ids)
    if len(atom_ids) <= 1:
        return True
    aset = set(atom_ids)
    # BFS from first atom
    q = [atom_ids[0]]
    seen = {atom_ids[0]}
    while q:
        u = q.pop()
        au = mol.GetAtomWithIdx(int(u))
        for nb in au.GetNeighbors():
            v = nb.GetIdx()
            if v in aset and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(aset)

def _normalize_accfg_indices(mol: Chem.Mol, occs):

    n = mol.GetNumAtoms()

    all_ids = [int(a) for occ in occs for a in occ]
    if len(all_ids) == 0:
        return occs

    def valid_A():
        return all(0 <= a < n for a in all_ids)

    all_ids_B = [a - 1 for a in all_ids]
    def valid_B():
        return all(0 <= a < n for a in all_ids_B)

    def score(convert_fn):
        s = 0
        for occ in occs:
            ids = convert_fn(occ)
            if _is_connected_subgraph(mol, ids):
                s += 1
        return s

    if valid_A() and not valid_B():
        return [tuple(int(a) for a in occ) for occ in occs]
    if valid_B() and not valid_A():
        return [tuple(int(a) - 1 for a in occ) for occ in occs]

    if valid_A() and valid_B():
        sA = score(lambda occ: [int(a) for a in occ])
        sB = score(lambda occ: [int(a) - 1 for a in occ])
        if sB > sA:
            return [tuple(int(a) - 1 for a in occ) for occ in occs]
        else:
            return [tuple(int(a) for a in occ) for occ in occs]

    return []


def safe_run_accfg(afg, smi: str):

    try:
        fgs_dict, fg_graph = afg.run(smi, show_atoms=True, show_graph=False)
        if fgs_dict is None:
            fgs_dict = {}
        return fgs_dict, fg_graph
    except Exception:
        return {}, None


def build_fg_struct(mol: Chem.Mol, smi_cano: str, fgs_dict_raw: dict):

    n_atoms = mol.GetNumAtoms()

    fg_dict = {}
    for fg_name, occs in (fgs_dict_raw or {}).items():
        if not occs:
            continue
        occs_norm = _normalize_accfg_indices(mol, occs)
        if occs_norm:
            fg_dict[str(fg_name)] = [tuple(int(a) for a in occ) for occ in occs_norm]

    fg_names, fg_atoms = [], []
    for fg_name, occs in fg_dict.items():
        for occ in occs:
            fg_names.append(fg_name)
            fg_atoms.append(tuple(occ))
    K = len(fg_names)

    atom2fg = [[] for _ in range(n_atoms)]
    for fg_id, atoms in enumerate(fg_atoms):
        for a in atoms:
            if 0 <= a < n_atoms:
                atom2fg[a].append(fg_id)

    a_src, a_dst = [], []
    for a in range(n_atoms):
        for fg_id in atom2fg[a]:
            a_src.append(a)
            a_dst.append(fg_id)
    atom2fg_edge_index = np.array([a_src, a_dst], dtype=np.int64) if a_src else np.zeros((2, 0), dtype=np.int64)

    pair_feat = defaultdict(lambda: np.zeros(5, dtype=np.float32))
    shared_pairs = set()

    for a in range(n_atoms):
        fgs_here = atom2fg[a]
        if len(fgs_here) > 1:
            for i in range(len(fgs_here)):
                for j in range(i + 1, len(fgs_here)):
                    shared_pairs.add(tuple(sorted((fgs_here[i], fgs_here[j]))))

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        fgu, fgv = atom2fg[u], atom2fg[v]
        if (not fgu) or (not fgv):
            continue

        for fi in fgu:
            for fj in fgv:
                if fi == fj:
                    continue
                a, b = (fi, fj) if fi < fj else (fj, fi)

                if bond.GetIsAromatic():
                    pair_feat[(a, b)][3] += 1.0
                else:
                    bt = bond.GetBondType()
                    if bt == Chem.rdchem.BondType.SINGLE:
                        pair_feat[(a, b)][0] += 1.0
                    elif bt == Chem.rdchem.BondType.DOUBLE:
                        pair_feat[(a, b)][1] += 1.0
                    elif bt == Chem.rdchem.BondType.TRIPLE:
                        pair_feat[(a, b)][2] += 1.0
                    else:
                        pair_feat[(a, b)][0] += 1.0  # fallback
                pair_feat[(a, b)][4] += 1.0

    src, dst, eattr = [], [], []

    def push(i, j, feat5, shared):
        feat6 = np.concatenate([feat5, np.array([1.0 if shared else 0.0], dtype=np.float32)], axis=0)
        src.append(i)
        dst.append(j)
        eattr.append(feat6)

    for (a, b), feat5 in pair_feat.items():
        shared = (a, b) in shared_pairs
        push(a, b, feat5, shared)
        push(b, a, feat5, shared)

    for (a, b) in shared_pairs:
        if (a, b) not in pair_feat:
            z = np.zeros(5, dtype=np.float32)
            push(a, b, z, True)
            push(b, a, z, True)

    fg_edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    fg_edge_attr = np.vstack(eattr).astype(np.float32) if eattr else np.zeros((0, 6), dtype=np.float32)

    return {
        "smiles": smi_cano,
        "num_atoms": int(n_atoms),
        "fg_dict": fg_dict,
        "fg_names": fg_names,
        "fg_atoms": fg_atoms,
        "fg_num_nodes": int(K),
        "atom2fg_edge_index": atom2fg_edge_index,
        "fg_edge_index": fg_edge_index,
        "fg_edge_attr": fg_edge_attr,
    }

def add_fg_to_split(data_dir: Path, split: str, smiles_col="smiles", y_col="log_sol", lite=True, user_defined_fgs=None):

    csv_path = data_dir / f"{split}.csv"
    pkl_path = data_dir / f"{split}.pkl.gz"
    out_path = data_dir / f"{split}_withFG.pkl.gz"

    df = pd.read_csv(csv_path)
    data_list = load_pkl_gz(pkl_path)

    assert len(df) == len(data_list), f"[{split}] csv({len(df)}) != pkl({len(data_list)})"

    afg = AccFG(lite=lite, user_defined_fgs=user_defined_fgs, print_load_info=False)

    all_fg_types = set()
    bad = 0

    for i in tqdm(range(len(df)), desc=f"AccFG patching {split}"):
        smi = str(df.loc[i, smiles_col])
        y_csv = float(df.loc[i, y_col])

        d = data_list[i]
        if hasattr(d, "y"):
            y_pkl = float(d.y.detach().cpu().numpy().reshape(-1)[0])
            if abs(y_pkl - y_csv) > 1e-6:
                pass

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            fg_pack = {
                "smiles": smi,
                "num_atoms": 0,
                "fg_dict": {},
                "fg_names": [],
                "fg_atoms": [],
                "fg_num_nodes": 0,
                "atom2fg_edge_index": np.zeros((2, 0), dtype=np.int64),
                "fg_edge_index": np.zeros((2, 0), dtype=np.int64),
                "fg_edge_attr": np.zeros((0, 6), dtype=np.float32),
            }
        else:
            smi_cano = Chem.MolToSmiles(mol)
            fgs_dict_raw, _ = safe_run_accfg(afg, smi_cano)
            fg_pack = build_fg_struct(mol, smi_cano, fgs_dict_raw)
            all_fg_types.update(fg_pack["fg_names"])

        d.smiles = fg_pack["smiles"]
        d.fg_dict = fg_pack["fg_dict"]
        d.fg_names = fg_pack["fg_names"]
        d.fg_atoms = fg_pack["fg_atoms"]
        d.fg_num_nodes = fg_pack["fg_num_nodes"]

        d.atom2fg_edge_index = fg_pack["atom2fg_edge_index"]
        d.fg_edge_index      = fg_pack["fg_edge_index"]
        d.fg_edge_attr       = fg_pack["fg_edge_attr"]

    save_pkl_gz(data_list, out_path)
    print(f"[{split}] saved -> {out_path} | bad_smiles={bad}")

    return all_fg_types


def _occ_is_connected(mol, atom_ids):
    atom_ids = [int(a) for a in atom_ids]
    if len(atom_ids) <= 1:
        return True
    aset = set(atom_ids)

    q = deque([atom_ids[0]])
    seen = {atom_ids[0]}
    while q:
        u = q.popleft()
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            v = nb.GetIdx()
            if v in aset and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(aset)


def normalize_accfg_indices(mol, fgs_dict):

    if not isinstance(fgs_dict, dict) or len(fgs_dict) == 0:
        return {}

    n = mol.GetNumAtoms()

    occs = []
    for fg_name, occ_list in fgs_dict.items():
        if not occ_list:
            continue
        for occ in occ_list:
            if isinstance(occ, (list, tuple)) and len(occ) > 0:
                occs.append(tuple(int(a) for a in occ))

    if len(occs) == 0:
        return {}

    def score(transform):
        out_of_range = 0
        connected = 0
        total = 0
        for occ in occs:
            ids = [transform(a) for a in occ]
            total += 1
            if any((a < 0 or a >= n) for a in ids):
                out_of_range += 1
                continue
            if _occ_is_connected(mol, ids):
                connected += 1
        return (out_of_range, connected, total)

    sA = score(lambda a: int(a))
    sB = score(lambda a: int(a) - 1)

    choose_shift_minus1 = False
    if sB[0] < sA[0]:
        choose_shift_minus1 = True
    elif sB[0] == sA[0] and sB[1] > sA[1]:
        choose_shift_minus1 = True

    out = {}
    for fg_name, occ_list in fgs_dict.items():
        new_occs = []
        for occ in occ_list:
            if not isinstance(occ, (list, tuple)) or len(occ) == 0:
                continue
            if choose_shift_minus1:
                ids = [int(a) - 1 for a in occ]
            else:
                ids = [int(a) for a in occ]

            if any((a < 0 or a >= n) for a in ids):
                continue
            new_occs.append(tuple(ids))

        if new_occs:
            out[str(fg_name)] = new_occs

    return out

def _coerce_to_fg_dict(obj):

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            occs = []
            for occ in v:
                if occ is None:
                    continue
                if isinstance(occ, (list, tuple)):
                    occs.append(tuple(int(a) for a in occ))
                else:
                    # single int?
                    try:
                        occs.append((int(occ),))
                    except Exception:
                        pass
            if occs:
                out[str(k)] = occs
        return out

    if isinstance(obj, str):
        s = obj.strip()
        # try json
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return _coerce_to_fg_dict(json.loads(s))
            except Exception:
                return {}
        return {}

    if isinstance(obj, (list, tuple)):
        out = {}
        for item in obj:
            if isinstance(item, dict):
                # guess keys
                name = item.get("fg") or item.get("name") or item.get("type") or item.get("fg_name")
                atoms = item.get("atoms") or item.get("atom_ids") or item.get("idx") or item.get("atom_index")
                if name is None or atoms is None:
                    continue
                if not isinstance(atoms, (list, tuple)):
                    continue
                out.setdefault(str(name), []).append(tuple(int(a) for a in atoms))
        return out

    return {}


def _safe_run_accfg(afg, smi):
    """
    Normalize AccFG output to dict[str, list[tuple[int]]].
    """
    try:
        ret = afg.run(smi, show_atoms=True, show_graph=False)

        if isinstance(ret, tuple) and len(ret) >= 1:
            fgs_raw = ret[0]
        else:
            fgs_raw = ret

        fgs = _coerce_to_fg_dict(fgs_raw)
        return fgs
    except Exception:
        return {}

def _build_fg_struct_from_fgs(mol, smi_cano, fgs_dict):
    n_atoms = mol.GetNumAtoms()

    fg_names, fg_atoms = [], []
    for fg_name, occs in fgs_dict.items():
        for occ in occs:
            occ = tuple(int(a) for a in occ)
            fg_names.append(str(fg_name))
            fg_atoms.append(occ)

    atom2fg = [[] for _ in range(n_atoms)]
    for fg_id, atoms in enumerate(fg_atoms):
        for a in atoms:
            if 0 <= a < n_atoms:
                atom2fg[a].append(fg_id)

    a_src, a_dst = [], []
    for a in range(n_atoms):
        for fg_id in atom2fg[a]:
            a_src.append(a); a_dst.append(fg_id)
    atom2fg_edge_index = np.array([a_src, a_dst], dtype=np.int64) if a_src else np.zeros((2,0), dtype=np.int64)

    pair_feat = defaultdict(lambda: np.zeros(5, dtype=np.float32))
    shared_pairs = set()

    for a in range(n_atoms):
        fgs_here = atom2fg[a]
        if len(fgs_here) > 1:
            for i in range(len(fgs_here)):
                for j in range(i+1, len(fgs_here)):
                    shared_pairs.add(tuple(sorted((fgs_here[i], fgs_here[j]))))

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        fgu, fgv = atom2fg[u], atom2fg[v]
        if (not fgu) or (not fgv):
            continue
        for fi in fgu:
            for fj in fgv:
                if fi == fj: 
                    continue
                a, b = (fi, fj) if fi < fj else (fj, fi)
                if bond.GetIsAromatic():
                    pair_feat[(a,b)][3] += 1.0
                else:
                    bt = bond.GetBondType()
                    if bt == Chem.rdchem.BondType.SINGLE: pair_feat[(a,b)][0] += 1.0
                    elif bt == Chem.rdchem.BondType.DOUBLE: pair_feat[(a,b)][1] += 1.0
                    elif bt == Chem.rdchem.BondType.TRIPLE: pair_feat[(a,b)][2] += 1.0
                    else: pair_feat[(a,b)][0] += 1.0
                pair_feat[(a,b)][4] += 1.0

    src, dst, eattr = [], [], []
    def push(i, j, feat5, shared):
        feat6 = np.concatenate([feat5, np.array([1.0 if shared else 0.0], np.float32)], axis=0)
        src.append(i); dst.append(j); eattr.append(feat6)

    for (a,b), feat5 in pair_feat.items():
        shared = (a,b) in shared_pairs
        push(a,b,feat5,shared); push(b,a,feat5,shared)

    for (a,b) in shared_pairs:
        if (a,b) not in pair_feat:
            z = np.zeros(5, np.float32)
            push(a,b,z,True); push(b,a,z,True)

    fg_edge_index = np.array([src,dst], dtype=np.int64) if src else np.zeros((2,0), dtype=np.int64)
    fg_edge_attr  = np.vstack(eattr).astype(np.float32) if eattr else np.zeros((0,6), dtype=np.float32)

    return {
        "smiles": smi_cano,
        "num_atoms": int(n_atoms),
        "fg_names": fg_names,
        "fg_atoms": fg_atoms,
        "fg_num_nodes": int(len(fg_names)),
        "atom2fg_edge_index": atom2fg_edge_index,
        "fg_edge_index": fg_edge_index,
        "fg_edge_attr": fg_edge_attr,
    }


def process_split(data_dir: Path, split: str, lite=True):
    df = pd.read_csv(data_dir / f"{split}.csv")
    smiles_list = df["smiles"].tolist()

    afg = AccFG(lite=lite, user_defined_fgs=None, print_load_info=False)

    recs = []
    vocab = set()
    bad = 0

    for smi in tqdm(smiles_list, desc=f"AccFG {split}"):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            bad += 1
            recs.append({
                "smiles": str(smi),
                "num_atoms": 0,
                "fg_names": [], "fg_atoms": [],
                "fg_num_nodes": 0,
                "atom2fg_edge_index": np.zeros((2,0), np.int64),
                "fg_edge_index": np.zeros((2,0), np.int64),
                "fg_edge_attr": np.zeros((0,6), np.float32),
            })
            continue

        smi_cano = Chem.MolToSmiles(mol)
        fgs = _safe_run_accfg(afg, smi_cano)
        fgs = normalize_accfg_indices(mol, fgs)
        rec = _build_fg_struct_from_fgs(mol, smi_cano, fgs)

        vocab.update(rec["fg_names"])
        recs.append(rec)

    out = data_dir / f"{split}_fg.pkl.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[{split}] saved {out} | n={len(recs)} | bad_smiles={bad}")
    return vocab

def main():
    data_dir = Path(".").resolve()
    vocab = set()
    for split in ["train", "val", "test"]:
        vocab |= process_split(data_dir, split, lite=True)

    with open(data_dir / "accfg_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"fg_types": sorted(vocab)}, f, ensure_ascii=False, indent=2)
    print(f"saved accfg_vocab.json (|V|={len(vocab)})")


if __name__ == "__main__":
    main()
