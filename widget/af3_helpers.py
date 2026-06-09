import re
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two naming conventions are supported: the AlphaFold Server / AlphaFold3 convention
# (e.g. fold_<name>_model_0.cif, fold_<name>_summary_confidences_0.json, ...) and
# Protenix's (e.g. <name>_sample_0.cif, <name>_summary_confidence_sample_0.json, ...).
# The patterns are mutually exclusive by construction (Protenix always interposes
# "sample_" between the role and the index), so a file can be classified without
# first knowing which tool produced it.
_FILE_PATTERNS = {
    ("alphafold3", "model"):     re.compile(r"model_(\d+)\.cif$", re.IGNORECASE),
    ("alphafold3", "summary"):   re.compile(r"summary_confidences_(\d+)\.json$", re.IGNORECASE),
    ("alphafold3", "full_data"): re.compile(r"full_data_(\d+)\.json$", re.IGNORECASE),
    ("protenix",   "model"):     re.compile(r"_sample_(\d+)\.cif$", re.IGNORECASE),
    ("protenix",   "summary"):   re.compile(r"summary_confidence_sample_(\d+)\.json$", re.IGNORECASE),
    ("protenix",   "full_data"): re.compile(r"full_data_sample_(\d+)\.json$", re.IGNORECASE),
    ("openfold3",  "model"):     re.compile(r"_sample_(\d+)_model\.(cif|pdb)$", re.IGNORECASE),
    ("openfold3",  "summary"):   re.compile(r"_sample_(\d+)_confidences_aggregated\.json$", re.IGNORECASE),
    ("openfold3",  "full_data"): re.compile(r"_sample_(\d+)_confidences\.(json|npz)$", re.IGNORECASE),
}


def classify_prediction_file(filename: str):
    """Identify a structure-prediction output file's source, role and index from its name.

    Returns (source, kind, index) where source is "alphafold3" or "protenix" and kind is
    "model", "summary" or "full_data", or (None, None, None) if the filename doesn't match
    either tool's output naming convention.
    """
    for (source, kind), pattern in _FILE_PATTERNS.items():
        m = pattern.search(filename)
        if m:
            return source, kind, int(m.group(1))
    return None, None, None


def detect_prediction_source(files):
    """Infer which tool ("alphafold3" or "protenix") produced a set of uploaded files.

    Returns the source name, or None if no file matches a recognized naming convention.
    """
    for f in files:
        source, _kind, _idx = classify_prediction_file(f.name)
        if source is not None:
            return source
    return None


def group_prediction_files(files, source):
    """Group uploaded files of the given source by predicted-model index.

    Returns {model_index: {"model": FileInfo, "summary": FileInfo, "full_data": FileInfo}},
    keeping only indices for which both a structure and a summary file were found, and
    ignoring any files that don't match `source`'s naming convention.
    """
    groups: dict[int, dict] = {}
    for f in files:
        f_source, kind, idx = classify_prediction_file(f.name)
        if f_source != source or kind is None:
            continue
        groups.setdefault(idx, {})[kind] = f
    return {idx: g for idx, g in groups.items() if "model" in g and "summary" in g}


def parse_prediction_json(contents: bytes) -> dict:
    return json.loads(contents.decode("utf-8"))


def _chain_letter(asym_id) -> str:
    """Render a 0-based numeric chain index the way AlphaFold3 labels chains (A, B, … Z, AA, …)."""
    n = int(asym_id)
    letters = ""
    while True:
        n, rem = divmod(n, 26)
        letters = chr(ord("A") + rem) + letters
        if n == 0:
            return letters
        n -= 1


def normalize_summary(raw: dict, source: str) -> dict:
    """Map a raw summary-confidence dict onto the AlphaFold3 schema the viewer expects.

    AlphaFold3 and Protenix report most metrics under identical names (ptm, iptm,
    ranking_score, has_clash, chain_ptm, chain_iptm, chain_pair_iptm, num_recycles, ...).
    The one rename needed is Protenix's `disorder` → AlphaFold3's `fraction_disordered`
    for the same headline metric.
    """
    summary = dict(raw)
    if source == "protenix" and "fraction_disordered" not in summary and "disorder" in summary:
        summary["fraction_disordered"] = summary["disorder"]
    elif source == "openfold3":
        if "ranking_score" not in summary and "sample_ranking_score" in summary:
            summary["ranking_score"] = summary["sample_ranking_score"]
        if "fraction_disordered" not in summary and "disorder" in summary:
            summary["fraction_disordered"] = summary["disorder"]
        if "chain_pair_iptm" in summary and isinstance(summary["chain_pair_iptm"], dict):
            # Convert dict of pair strings to a 2D matrix
            chains = sorted(summary.get("chain_ptm", {}).keys())
            n = len(chains)
            mat = [[1.0] * n for _ in range(n)]
            raw_pair = summary["chain_pair_iptm"]
            for i, c1 in enumerate(chains):
                for j, c2 in enumerate(chains):
                    if i != j:
                        val = raw_pair.get(f"({c1}, {c2})") or raw_pair.get(f"({c2}, {c1})") or 0.0
                        mat[i][j] = val
            summary["chain_pair_iptm"] = mat
    return summary


def normalize_full_data(raw: dict, source: str, atoms=None) -> dict:
    """Map a raw full-data dict onto the AlphaFold3 schema the viewer expects.

    Protenix names the PAE matrix and per-token chain assignment differently —
    `token_pair_pae` / `token_asym_id` (integer chain indices) instead of AlphaFold3's
    `pae` / `token_chain_ids` (letter chain labels) — and its per-atom pLDDT array is
    `atom_plddt` rather than `atom_plddts`. This aliases them onto AlphaFold3's names
    so `build_confidence_figure` and `compute_ipsae` work unchanged for both sources.
    """
    full_data = dict(raw)
    if source == "protenix":
        if "pae" not in full_data and "token_pair_pae" in full_data:
            full_data["pae"] = full_data["token_pair_pae"]
        if "token_chain_ids" not in full_data and "token_asym_id" in full_data:
            full_data["token_chain_ids"] = [_chain_letter(a) for a in full_data["token_asym_id"]]
        if "atom_plddts" not in full_data and "atom_plddt" in full_data:
            full_data["atom_plddts"] = full_data["atom_plddt"]
    elif source == "openfold3":
        if "token_chain_ids" not in full_data and atoms is not None:
            import biotite.structure as struc
            starts = struc.get_residue_starts(atoms)
            full_data["token_chain_ids"] = [atoms.chain_id[idx] for idx in starts]
        if "atom_plddts" not in full_data and "plddt" in full_data:
            full_data["atom_plddts"] = full_data["plddt"]
    return full_data


def chain_boundaries(chain_ids):
    """Collapse a per-token chain-id list into contiguous (chain_id, start, end) ranges."""
    bounds = []
    start = 0
    current = chain_ids[0]
    for i in range(1, len(chain_ids)):
        if chain_ids[i] != current:
            bounds.append((current, start, i))
            start = i
            current = chain_ids[i]
    bounds.append((current, start, len(chain_ids)))
    return bounds


def _ptm_score(pae, d0):
    return 1.0 / (1.0 + (pae / d0) ** 2.0)


def _d0_array(n_residues):
    """Per-residue d0 normalisation (Yang & Skolnick, Proteins 2004), array form
    matching the reference ipsae.py implementation (residue counts floored at 26)."""
    n_residues = np.maximum(26.0, np.asarray(n_residues, dtype=float))
    return np.maximum(1.0, 1.24 * (n_residues - 15.0) ** (1.0 / 3.0) - 1.8)


def compute_ipsae(pae, chain_ids, pae_cutoff=10.0):
    """Compute the ipSAE interface-confidence score for every pair of chains.

    ipSAE (Dunbrack et al., https://doi.org/10.1101/2025.02.10.637595) refines AlphaFold's
    ipTM for assessing inter-chain interfaces by (1) discarding residue pairs whose
    predicted aligned error exceeds `pae_cutoff` and (2) normalising per residue by how
    many partner-chain residues are confidently placed relative to it. This makes it more
    sensitive to confidently predicted sub-interfaces in larger or partly-disordered
    assemblies than the global ipTM. Unlike ipTM it depends only on the PAE matrix and
    chain assignment — no 3D coordinates are required.

    Returns {(chain_a, chain_b): score} for every unordered chain pair (chain_a < chain_b),
    where score is the symmetrized maximum of the two directional (A→B and B→A) values —
    matching the headline "max" value reported by the reference `ipsae.py` tool.
    """
    pae = np.asarray(pae, dtype=float)
    chains = np.asarray(chain_ids)
    unique_chains = sorted(set(chain_ids))

    def _directional(c1, c2):
        mask1 = chains == c1
        mask2 = chains == c2
        valid = np.outer(mask1, mask2) & (pae < pae_cutoff)
        d0_byres = _d0_array(valid.sum(axis=1))
        best = 0.0
        for i in np.where(mask1)[0]:
            row_valid = valid[i]
            if not row_valid.any():
                continue
            best = max(best, float(_ptm_score(pae[i, row_valid], d0_byres[i]).mean()))
        return best

    scores = {}
    for pos, c1 in enumerate(unique_chains):
        for c2 in unique_chains[pos + 1:]:
            scores[(c1, c2)] = max(_directional(c1, c2), _directional(c2, c1))
    return scores


def build_confidence_figure(model_index, summary, full_data, rank=None):
    """Build a per-model figure with the PAE heatmap and, when more than two chains
    are present, the pairwise ipTM matrix alongside it."""
    pae = np.asarray(full_data["pae"], dtype=float)
    chain_ids = full_data["token_chain_ids"]
    bounds = chain_boundaries(chain_ids)
    ticks = [(s + e) / 2 - 0.5 for _, s, e in bounds]
    labels = [c for c, _, _ in bounds]

    chain_pair_iptm = summary.get("chain_pair_iptm")
    show_matrix = chain_pair_iptm is not None and len(chain_pair_iptm) > 2

    if show_matrix:
        fig, (ax_pae, ax_iptm) = plt.subplots(1, 2, figsize=(9, 4))
    else:
        fig, ax_pae = plt.subplots(1, 1, figsize=(4.6, 4))

    im = ax_pae.imshow(pae, cmap="Greens_r", vmin=0, vmax=31.75)
    for _, _start, end in bounds[:-1]:
        ax_pae.axhline(end - 0.5, color="white", linewidth=0.7)
        ax_pae.axvline(end - 0.5, color="white", linewidth=0.7)
    ax_pae.set_xticks(ticks)
    ax_pae.set_xticklabels(labels, fontsize=8)
    ax_pae.set_yticks(ticks)
    ax_pae.set_yticklabels(labels, fontsize=8)
    ax_pae.set_title("Predicted aligned error (Å)", fontsize=9)
    fig.colorbar(im, ax=ax_pae, fraction=0.046, pad=0.04)

    if show_matrix:
        mat = np.asarray(chain_pair_iptm, dtype=float)
        im2 = ax_iptm.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax_iptm.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.5 else "black",
                )
        ax_iptm.set_xticks(range(len(labels)))
        ax_iptm.set_xticklabels(labels, fontsize=8)
        ax_iptm.set_yticks(range(len(labels)))
        ax_iptm.set_yticklabels(labels, fontsize=8)
        ax_iptm.set_title("Pairwise ipTM", fontsize=9)
        fig.colorbar(im2, ax=ax_iptm, fraction=0.046, pad=0.04)

    _rank = f"rank {rank} · " if rank is not None else ""
    _ptm = summary.get("ptm")
    _iptm = summary.get("iptm")
    _metrics = f"pTM={_ptm:.2f}" if _ptm is not None else ""
    if _iptm is not None:
        _metrics += f", ipTM={_iptm:.2f}"
    fig.suptitle(f"Model {model_index} — {_rank}{_metrics}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
