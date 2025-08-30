import os
import argparse
from typing import Dict, List, Optional

import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from minimol import Minimol


from minimol_triplet_model import (
    train_minimol_triplet,
    MinimolTripletModule,
    MinimolTripletDataset,
    minimol_triplet_collate,
    CachedEncoder,
)
import torch
from torch.utils.data import DataLoader

# Classifiers for different tasks

from plausibility_utils import (
        classify_logp_rationales as classify_lipophilicity_rationales,
        classify_ames_rationales)


def _canon(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m else smiles

# minimol uses a simpler fragmenter than the dmpnn function.
def _get_cleaned_fragments(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    frags = set()
    try:
        frags.update(BRICS.BRICSDecompose(m, keepNonLeafNodes=True))
    except Exception:
        pass
    try:
        scf = MurckoScaffold.GetScaffoldForMol(m)
        smi = Chem.MolToSmiles(scf) if scf else None
        if smi:
            frags.add(smi)
    except Exception:
        pass
    valid = []
    for f in frags:
        mm = Chem.MolFromSmiles(f)
        if mm is None:
            continue
        n = mm.GetNumAtoms()
        if 4 <= n <= 12:
            valid.append(f)
    cleaned = []
    for f in valid:
        g = f.replace('()', '')
        for k in ['[1*]','[2*]','[3*]','[4*]','[5*]','[6*]','[7*]','[8*]', '[9*]', '[10*]','[11*]','[12*]','[13*]','[14*]','[15*]','[16*]','[17*]','[18*]','[19*]','[20*]']:
            g = g.replace(k, '')
        cleaned.append(g)
    # Deduplicate by Tanimoto similarity: do not add fragments that are too similar
    filtered = []
    selected_fps = []
    for smi in cleaned:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        is_similar = any(DataStructs.TanimotoSimilarity(fp, prev_fp) >= 0.3 for prev_fp in selected_fps)
        if is_similar:
            continue
        selected_fps.append(fp)
        filtered.append(smi)
    return filtered




def _build_plausibility(
    train_df: pd.DataFrame,
    classifier_fn,
    max_frags_per_mol: Optional[int],
    *,
    is_regression: bool,
    min_score: Optional[float] = None,
) -> tuple[Dict[str, Dict[str, List]], int, int]:
    mapping: Dict[str, Dict[str, List]] = {}
    # Filter candidate molecules for substructure extraction
    if is_regression:
        if min_score is not None:
            cand_df = train_df[train_df['Y'] >= float(min_score)]
        else:
            cand_df = train_df
    else:
        cand_df = train_df[train_df['Y'] == 1]

    uniq_smiles = list(dict.fromkeys([str(s) for s in cand_df['SMILES'].tolist()]))
    entered = 0
    with_pair = 0
    for s in uniq_smiles:
        cands = _get_cleaned_fragments(s)
        if not cands:
            continue
        entered += 1
        if max_frags_per_mol is not None and len(cands) > max_frags_per_mol:
            cands = sorted(cands, key=lambda x: (-len(x), x))[:max_frags_per_mol]
        try:
            labels = classifier_fn(cands)
        except Exception:
            continue
        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1
    return mapping, entered, with_pair


def _evaluate_regression_module(model: MinimolTripletModule, df: pd.DataFrame, encoder: CachedEncoder, batch_size: int = 256):
    model.eval()
    ds = MinimolTripletDataset(df, smiles_to_plausibility=None, require_triplet=False)
    collate = minimol_triplet_collate(encoder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    yp, yt = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b['x'], b['y']
            p = model(x).detach().cpu().numpy()
            yp.extend(p.tolist())
            yt.extend(y.detach().cpu().numpy().tolist())
    yp = np.array(yp)
    yt = np.array(yt)
    rmse = float(np.sqrt(((yp - yt) ** 2).mean()))
    denom = float(((yt - yt.mean()) ** 2).sum())
    r2 = float(1.0 - (((yp - yt) ** 2).sum() / denom)) if denom > 0 else 0.0
    return {'rmse': rmse, 'r2': r2}


def run_minimol_triplet(
    csv_path: str,
    output_dir: str,
    cache_file: Optional[str],
    lrs: List[float],
    epochs_list: List[int],
    batch_size: int,
    margins: List[float],
    triplet_weights: List[float],
    max_frags_per_mol: Optional[int],
    replicates: int,
    task: str,
    min_score: Optional[float] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f'grid_results_{task}.csv')
    df = pd.read_csv(csv_path)
    if 'SMILES_CANON' not in df.columns:
        df['SMILES_CANON'] = df['SMILES'].apply(_canon)

    train_df = df[df['origin'] == 'train'].copy()
    val_df = df[df['origin'] == 'val'].copy()
    test_df = df[df['origin'] == 'test'].copy()

    # Select classifier based on task
    task_lc = task.lower()
    if task_lc == 'lipophilicity':
        classifier_fn = classify_lipophilicity_rationales
        is_regression = True

    elif task_lc == 'ames':
        classifier_fn = classify_ames_rationales
        is_regression = False
    else:
        raise ValueError(f"Unknown task: {task}. Expected one of: lipophilicity, solubility, ames")

    smiles_to_plaus, n_entered, n_with_pair = _build_plausibility(
        train_df,
        classifier_fn=classifier_fn,
        max_frags_per_mol=max_frags_per_mol,
        is_regression=is_regression,
        min_score=min_score,
    )
    print(
        f"[{task}] Rationale search stats: entered={n_entered}, "
        f"with_plausible_and_implausible_pair={n_with_pair}"
    )

    # Save plausibility mapping to CSV (parent SMILES, rationale fragment, label)
    plaus_rows = []
    for s, entry in smiles_to_plaus.items():
        rats = entry.get('Rationales', [])
        labs = entry.get('plausibility', [])
        for r, l in zip(rats, labs):
            plaus_rows.append({'SMILES': s, 'rationale': r, 'plausibility': int(l)})
    if len(plaus_rows) > 0:
        plaus_out_csv = os.path.join(output_dir, f'rationale_search_results_{task}.csv')
        pd.DataFrame(plaus_rows).to_csv(plaus_out_csv, index=False)
        print(f"Saved plausibility results to {plaus_out_csv}")

    mini = Minimol()
    cache_path = cache_file if cache_file else os.path.join(output_dir, 'smiles_fp_cache.pt')
    encoder = CachedEncoder(mini, cache_path)

    run_idx = 0
    for rep in range(int(max(1, replicates))):
    
        for margin in margins:
            for triplet_weight in triplet_weights:
                for lr in lrs:
                    for n_epochs in epochs_list:
                        run_idx += 1
                        # 1) Triplet with rule-based plausibility
                        model = train_minimol_triplet(
                            train_df=train_df,
                            val_df=val_df,
                            encoder=encoder,
                            is_regression=is_regression,
                            margin=margin,
                            triplet_weight=triplet_weight,
                            init_lr=lr,
                            max_epochs=n_epochs,
                            smiles_to_plausibility=smiles_to_plaus,
                            batch_size=batch_size,
                            require_triplet_for_train=False,
                        )

                        val_metrics = _evaluate_regression_module(model, val_df, encoder, batch_size=256)
                        test_metrics = _evaluate_regression_module(model, test_df, encoder, batch_size=256)

                        print(
                            f"Run {run_idx} (rep {rep+1}): lr={lr}, epochs={n_epochs}, margin={margin}, tw={triplet_weight} | "
                            f"Val RMSE={val_metrics['rmse']:.4f}, Test RMSE={test_metrics['rmse']:.4f}"
                        )

                        # 2) No-triplet baseline (triplet_weight=0) using only main loss
                        model_no_trip = train_minimol_triplet(
                            train_df=train_df,
                            val_df=val_df,
                            encoder=encoder,
                            is_regression=is_regression,
                            margin=margin,
                            triplet_weight=0.0,
                            init_lr=lr,
                            max_epochs=n_epochs,
                            smiles_to_plausibility=None,
                            batch_size=batch_size,
                            require_triplet_for_train=False,
                        )
                        val_metrics_nt = _evaluate_regression_module(model_no_trip, val_df, encoder, batch_size=256)
                        test_metrics_nt = _evaluate_regression_module(model_no_trip, test_df, encoder, batch_size=256)
                        print(
                            f"Run {run_idx} (rep {rep+1}, no_triplet): lr={lr}, epochs={n_epochs}, margin={margin} | "
                            f"Val RMSE={val_metrics_nt['rmse']:.4f}, Test RMSE={test_metrics_nt['rmse']:.4f}"
                        )

                        # 3) Random-plausibility baseline: assign one +1 and one -1 per molecule randomly
                        smiles_to_plaus_random: Dict[str, Dict[str, List]] = {}
                        for s in list(dict.fromkeys(train_df['SMILES'].tolist())):
                            cands = _get_cleaned_fragments(s)
                            if len(cands) < 2:
                                continue
                            labels = [0] * len(cands)
                            i_pos, i_neg = random.sample(range(len(cands)), 2)
                            labels[i_pos] = 1
                            labels[i_neg] = -1
                            smiles_to_plaus_random[s] = {'Rationales': cands, 'plausibility': labels}

                        model_rand = train_minimol_triplet(
                            train_df=train_df,
                            val_df=val_df,
                            encoder=encoder,
                            is_regression=is_regression,
                            margin=margin,
                            triplet_weight=triplet_weight,
                            init_lr=lr,
                            max_epochs=n_epochs,
                            smiles_to_plausibility=smiles_to_plaus_random,
                            batch_size=batch_size,
                            require_triplet_for_train=False,
                        )
                        val_metrics_rd = _evaluate_regression_module(model_rand, val_df, encoder, batch_size=256)
                        test_metrics_rd = _evaluate_regression_module(model_rand, test_df, encoder, batch_size=256)
                        print(
                            f"Run {run_idx} (rep {rep+1}, random): lr={lr}, epochs={n_epochs}, margin={margin}, tw={triplet_weight} | "
                            f"Val RMSE={val_metrics_rd['rmse']:.4f}, Test RMSE={test_metrics_rd['rmse']:.4f}"
                        )

                        # Append a single combined row to CSV incrementally
                        combined_row = {
                            'run': run_idx,
                            'replicate': rep + 1,
                            'lr': lr,
                            'epochs': n_epochs,
                            'margin': margin,
                            'triplet_weight': triplet_weight,
                            'n_entered': n_entered,
                            'n_with_pair': n_with_pair,
                            # triplet with plausibility
                            'val_rmse': val_metrics['rmse'],
                            'val_r2': val_metrics['r2'],
                            'test_rmse': test_metrics['rmse'],
                            'test_r2': test_metrics['r2'],
                            # no-triplet baseline
                            'val_rmse_no_triplet': val_metrics_nt['rmse'],
                            'val_r2_no_triplet': val_metrics_nt['r2'],
                            'test_rmse_no_triplet': test_metrics_nt['rmse'],
                            'test_r2_no_triplet': test_metrics_nt['r2'],
                            # random-plausibility baseline
                            'val_rmse_random': val_metrics_rd['rmse'],
                            'val_r2_random': val_metrics_rd['r2'],
                            'test_rmse_random': test_metrics_rd['rmse'],
                            'test_r2_random': test_metrics_rd['r2'],
                        }
                        pd.DataFrame([combined_row]).to_csv(
                            out_csv,
                            mode='a',
                            header=not os.path.exists(out_csv),
                            index=False,
                        )

    print(f"Results are being incrementally saved to {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--cache_file', type=str, default=None)
    ap.add_argument('--lrs', type=str, default='1e-3,3e-4')
    ap.add_argument('--epochs', type=str, default='25,50')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--margin', type=float, default=0.2)
    ap.add_argument('--triplet_weight', type=float, default=1.0)
    ap.add_argument('--margins', type=str, default=None, help='Comma-separated margins to sweep')
    ap.add_argument('--triplet_weights', type=str, default=None, help='Comma-separated triplet weights to sweep')
    ap.add_argument('--max_frags_per_mol', type=int, default=8)
    ap.add_argument('--replicates', type=int, default=1)
    ap.add_argument('--task', type=str, default='lipophilicity', choices=['lipophilicity', 'solubility', 'ames'])
    ap.add_argument('--min_score', type=float, default=None, help='Regression only: minimum Y to include for plausibility extraction')
    args = ap.parse_args()
    lrs = [float(x.strip()) for x in args.lrs.split(',')]
    epochs = [int(x.strip()) for x in args.epochs.split(',')]
    if args.margins is not None and args.margins.strip() != '':
        margins = [float(x.strip()) for x in args.margins.split(',')]
    else:
        margins = [float(args.margin)]
    if args.triplet_weights is not None and args.triplet_weights.strip() != '':
        triplet_weights = [float(x.strip()) for x in args.triplet_weights.split(',')]
    else:
        triplet_weights = [float(args.triplet_weight)]
    return args, lrs, epochs, margins, triplet_weights


if __name__ == '__main__':
    args, lrs, epochs_list, margins, triplet_weights = parse_args()
    run_minimol_triplet(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
        lrs=lrs,
        epochs_list=epochs_list,
        batch_size=args.batch_size,
        margins=margins,
        triplet_weights=triplet_weights,
        max_frags_per_mol=args.max_frags_per_mol,
        replicates=args.replicates,
        task=args.task,
        min_score=args.min_score,
    )


