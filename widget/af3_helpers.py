import re
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FILE_PATTERNS = {
    "model":     re.compile(r"model_(\d+)\.cif$", re.IGNORECASE),
    "summary":   re.compile(r"summary_confidences_(\d+)\.json$", re.IGNORECASE),
    "full_data": re.compile(r"full_data_(\d+)\.json$", re.IGNORECASE),
}


def classify_af3_file(filename: str):
    """Identify an AlphaFold3 output file's role and model index from its name.

    Returns (kind, index) where kind is "model", "summary" or "full_data",
    or (None, None) if the filename doesn't match the AF3 server naming convention
    (e.g. fold_<name>_model_0.cif, fold_<name>_summary_confidences_0.json, ...).
    """
    for kind, pattern in _FILE_PATTERNS.items():
        m = pattern.search(filename)
        if m:
            return kind, int(m.group(1))
    return None, None


def group_af3_files(files):
    """Group uploaded files by predicted-model index.

    Returns {model_index: {"model": FileInfo, "summary": FileInfo, "full_data": FileInfo}},
    keeping only indices for which both a structure and a summary file were found.
    """
    groups: dict[int, dict] = {}
    for f in files:
        kind, idx = classify_af3_file(f.name)
        if kind is None:
            continue
        groups.setdefault(idx, {})[kind] = f
    return {idx: g for idx, g in groups.items() if "model" in g and "summary" in g}


def parse_af3_json(contents: bytes) -> dict:
    return json.loads(contents.decode("utf-8"))


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
