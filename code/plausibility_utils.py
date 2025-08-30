from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
import random
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MolStandardize


def classify_logp_rationales(smiles_list):
    """
    Labels:
      +1 (positive rationale): a molecule with NO H-bond donors AND NO H-bond acceptors.
                               If multiple, pick the largest by number of heavy atoms (ties -> first).
      -1 (negative rationale): a molecule WITH H-bond donors/acceptors.
                               If multiple, pick the one with the MOST (donors + acceptors).
                               If there's a tie, select randomly among the tied.
      0  otherwise.

    If both a positive and a negative rationale cannot be selected, return all zeros.

    Raises:
      ValueError for invalid SMILES.
    """
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        mols.append(mol)

    # Compute properties needed for selection
    props = []
    for i, mol in enumerate(mols):
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        heavy = mol.GetNumHeavyAtoms()
        props.append({
            "idx": i,
            "hbd": hbd,
            "hba": hba,
            "hb_count": hbd + hba,
            "heavy": heavy
        })

    # Candidates
    pos_candidates = [p for p in props if p["hb_count"] == 0]          # no donors/acceptors
    neg_candidates = [p for p in props if p["hb_count"] > 0]           # has donors/acceptors

    labels = [0] * len(smiles_list)

    if not pos_candidates or not neg_candidates:
        return labels  # need both to assign nonzero labels

    # Positive rationale: largest by heavy atoms (ties -> first occurrence)
    max_heavy = max(p["heavy"] for p in pos_candidates)
    pos_idx = next(p["idx"] for p in pos_candidates if p["heavy"] == max_heavy)

    # Negative rationale: most H-bond donors/acceptors; ties -> random
    max_hb = max(p["hb_count"] for p in neg_candidates)
    tied_negs = [p for p in neg_candidates if p["hb_count"] == max_hb]
    neg_idx = random.choice(tied_negs)["idx"] if len(tied_negs) > 1 else tied_negs[0]["idx"]

    labels[pos_idx] = 1
    labels[neg_idx] = -1

    return labels

from rdkit import Chem
from rdkit.Chem import Lipinski


AMES_ALERTS = [
    # Nitroaromatic (nitro attached to aromatic ring)
    ("nitroaromatic", "[NX3](=O)=O-[cX3]:[cX3]"),
    ("nitroaromatic_alt", "[N+](=O)[O-]-c"),  # broad catch

    # Aromatic amines (anilines; exclude amides/ureas where N is acylated)
    ("aromatic_amine_primary_secondary", "[$([NX3H2]),$([NX3H][CX4])]-[a]"),
    
    # N-nitrosoamines
    ("N-nitrosoamine", "[NX3,NX2]-N=O"),

    # Azo / Azoxy
    ("azo", "[NX2]=[NX2]"),
    ("azoxy", "[NX2]=N-O"),

    # Epoxides / Aziridines (3-membered heterocycles)
    ("epoxide", "[OX2r3]"),
    ("aziridine", "[NX3r3]"),

    # α,β-unsaturated carbonyls (Michael acceptors), incl. acrylates/amides
    ("alpha_beta_unsat_ketone", "C=CC=O"),
    ("alpha_beta_unsat_ester", "C=CC(=O)O"),
    ("alpha_beta_unsat_amide", "C=CC(=O)N"),

    # Alkyl halides (SN1/SN2 electrophiles); include allylic/benzylic tendencies via general CX4–halogen
    ("alkyl_halide", "[CX4][Cl,Br,I]"),
    ("allylic_halide", "C=CC[Cl,Br,I]"),
]

def _has_ames_alert(mol):
    for name, smarts in AMES_ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue  # skip malformed pattern defensively
        if mol.HasSubstructMatch(patt):
            return True
    return False

def classify_ames_rationales(smiles_list):
    """
    Modified: uses Ames mutagenicity structural alerts to choose rationales.

    Labels:
      +1 (plausible rationale): fragment containing ≥1 Ames mutagenicity alert substructure.
                                If multiple, pick the LARGEST (heavy atoms; ties -> first).
      -1 (implausible rationale): fragment with NO such alerts.
                                  If multiple, pick the LARGEST (heavy atoms; ties -> first).
      0  otherwise.

    If both a plausible (+1) and an implausible (-1) fragment cannot be selected, return all zeros.

    Raises:
      ValueError for invalid SMILES.
    """
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        mols.append(mol)

    props = []
    for i, mol in enumerate(mols):
        heavy = mol.GetNumHeavyAtoms()
        alert = _has_ames_alert(mol)
        props.append({
            "idx": i,
            "heavy": heavy,
            "alert": alert
        })

    # Partition into candidates
    plausible = [p for p in props if p["alert"]]       # contains Ames alert(s)
    implausible = [p for p in props if not p["alert"]] # contains none

    labels = [0] * len(smiles_list)
    if not plausible or not implausible:
        return labels  # need both to assign nonzero labels

    # Pick largest by heavy atoms (ties -> first occurrence)
    max_heavy_pos = max(p["heavy"] for p in plausible)
    pos_idx = next(p["idx"] for p in plausible if p["heavy"] == max_heavy_pos)

    max_heavy_neg = max(p["heavy"] for p in implausible)
    neg_idx = next(p["idx"] for p in implausible if p["heavy"] == max_heavy_neg)

    labels[pos_idx] = +1
    labels[neg_idx] = -1
    return labels
