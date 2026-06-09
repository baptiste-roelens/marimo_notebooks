# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9",
#     "biotite>=1.6.0",
#     "numpy>=2.4.0",
#     "pandas>=2.0.0",
#     "matplotlib>=3.7.0",
#     "pyyaml>=6.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="OpenFold3 Structure Prediction")


@app.cell(hide_code=True)
def _imports():
    import glob
    import io
    import json
    import os
    import shutil
    import subprocess
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, glob, io, json, mo, os, shutil, subprocess, zipfile


@app.cell(hide_code=True)
def _viz_helpers():
    """Confidence-figure / ipSAE / structure-viewer building blocks.

    Inlined here — rather than imported from `widget.af3_helpers` /
    `widget.structure_helpers` — so this notebook stays a single self-contained
    file that runs in sandboxed environments (e.g. MoLab) where the local
    `widget` package can't be resolved.
    """
    import html as _html
    from io import StringIO

    import numpy as np
    import matplotlib.pyplot as plt
    import biotite.structure as struc
    import biotite.structure.io.pdb as bpdb
    from biotite.structure.io.pdbx import CIFFile, get_structure as _get_structure_cif

    # ── Confidence-schema normalization & scoring ───────────────────────────────

    def _chain_letter(asym_id) -> str:
        n = int(asym_id)
        letters = ""
        while True:
            n, rem = divmod(n, 26)
            letters = chr(ord("A") + rem) + letters
            if n == 0:
                return letters
            n -= 1

    def normalize_summary(raw, source):
        summary = dict(raw)
        if source == "protenix" and "fraction_disordered" not in summary and "disorder" in summary:
            summary["fraction_disordered"] = summary["disorder"]
        elif source == "openfold3":
            if "ranking_score" not in summary and "sample_ranking_score" in summary:
                summary["ranking_score"] = summary["sample_ranking_score"]
            if "fraction_disordered" not in summary and "disorder" in summary:
                summary["fraction_disordered"] = summary["disorder"]
            if "chain_pair_iptm" in summary and isinstance(summary["chain_pair_iptm"], dict):
                chains = sorted(summary.get("chain_ptm", {}).keys())
                n = len(chains)
                mat = [[1.0] * n for _ in range(n)]
                raw_pair = summary["chain_pair_iptm"]
                for i, c1 in enumerate(chains):
                    for j, c2 in enumerate(chains):
                        if i != j:
                            val = (
                                raw_pair.get(f"({c1}, {c2})")
                                or raw_pair.get(f"({c2}, {c1})")
                                or 0.0
                            )
                            mat[i][j] = val
                summary["chain_pair_iptm"] = mat
        return summary

    def normalize_full_data(raw, source, atoms=None):
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
                starts = struc.get_residue_starts(atoms)
                full_data["token_chain_ids"] = [atoms.chain_id[idx] for idx in starts]
            if "atom_plddts" not in full_data and "plddt" in full_data:
                full_data["atom_plddts"] = full_data["plddt"]
        return full_data

    def chain_boundaries(chain_ids):
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
        n_residues = np.maximum(26.0, np.asarray(n_residues, dtype=float))
        return np.maximum(1.0, 1.24 * (n_residues - 15.0) ** (1.0 / 3.0) - 1.8)

    def compute_ipsae(pae, chain_ids, pae_cutoff=10.0):
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

    # ── Structure parsing & 3D viewer ───────────────────────────────────────────

    BASIC_COLORS = {
        "Blue":   "#4e9af1",
        "Orange": "#f1a94e",
        "Red":    "#e45c3a",
        "Green":  "#4eba6f",
        "Purple": "#9b59b6",
        "Teal":   "#1abc9c",
        "Pink":   "#e91e8c",
        "Yellow": "#f1c40f",
        "Gray":   "#95a5a6",
        "Cyan":   "#00bcd4",
    }
    _color_cycle = list(BASIC_COLORS.keys())

    def default_color(index):
        return _color_cycle[index % len(_color_cycle)]

    def _is_cif(filename):
        return filename.rsplit(".", 1)[-1].lower() in ("cif", "mmcif")

    def parse_structure(contents, filename):
        text = contents.decode("utf-8", errors="replace")
        if _is_cif(filename):
            cf = CIFFile.read(StringIO(text))
            atoms = _get_structure_cif(cf, model=1, extra_fields=["b_factor"])
        else:
            pf = bpdb.PDBFile.read(StringIO(text))
            atoms = bpdb.get_structure(pf, model=1, extra_fields=["b_factor"])
        if isinstance(atoms, struc.AtomArrayStack):
            atoms = atoms[0]
        return atoms

    def atoms_to_pdb_str(atoms):
        pf = bpdb.PDBFile()
        bpdb.set_structure(pf, atoms)
        buf = StringIO()
        pf.write(buf)
        return buf.getvalue()

    def superimpose_all(structures):
        ref = structures[0]
        ca_ref = ref[(ref.atom_name == "CA") & ~ref.hetero]
        out = [ref]
        rmsds = []
        for mobile in structures[1:]:
            ca_mob = mobile[(mobile.atom_name == "CA") & ~mobile.hetero]
            n = min(len(ca_ref), len(ca_mob))
            try:
                _, tf = struc.superimpose(ca_ref[:n], ca_mob[:n])
                mobile_t = tf.apply(mobile)
                ca_mob_t = mobile_t[(mobile_t.atom_name == "CA") & ~mobile_t.hetero]
                rmsd = float(struc.rmsd(ca_ref[:n], ca_mob_t[:n]))
            except Exception:
                mobile_t = mobile
                rmsd = float("nan")
            out.append(mobile_t)
            rmsds.append(rmsd)
        return out, rmsds

    def get_b_range(atoms):
        cats = atoms.get_annotation_categories()
        if "b_factor" not in cats:
            return (0.0, 100.0)
        ca_mask = (atoms.atom_name == "CA") & ~atoms.hetero
        b = atoms.b_factor[ca_mask]
        if len(b) == 0:
            return (0.0, 100.0)
        return (float(b.min()), float(b.max()))

    def _js_esc(s):
        return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    def build_structure_html(pdb_contents, labels, visibilities, color_modes, colors, b_ranges, height=600):
        add_models_js = []
        for i, (pdb, visible, mode, color, (bmin, bmax)) in enumerate(
            zip(pdb_contents, visibilities, color_modes, colors, b_ranges)
        ):
            add_models_js.append(f"viewer.addModel(`{_js_esc(pdb)}`, 'pdb');")
            if not visible:
                add_models_js.append(f"  viewer.setStyle({{model:{i}}}, {{}});")
            elif mode == "B-factor":
                add_models_js.append(
                    "  viewer.setStyle({model:%d}, {cartoon:{colorscheme:{prop:'b',gradient:'roygb',min:%.2f,max:%.2f}}});"
                    % (i, bmin, bmax)
                )
            elif mode == "By chain":
                add_models_js.append(
                    f"  viewer.setStyle({{model:{i}}}, {{cartoon:{{colorscheme:'chainHetatm'}}}});"
                )
            else:
                add_models_js.append(
                    f"  viewer.setStyle({{model:{i}}}, {{cartoon:{{color:'{color}'}}}});"
                )

        models_js = "\n  ".join(add_models_js)

        legend_parts = []
        for label, visible, mode, color in zip(labels, visibilities, color_modes, colors):
            if not visible:
                continue
            if mode == "B-factor":
                swatch = (
                    '<span style="background:linear-gradient(to right,'
                    '#FF0000,#FFFF00,#00FF00,#0000FF);'
                    'width:28px;height:12px;display:inline-block;'
                    'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
                )
            elif mode == "By chain":
                swatch = (
                    '<span style="background:linear-gradient(to right,'
                    '#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2);'
                    'width:28px;height:12px;display:inline-block;'
                    'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
                )
            else:
                swatch = (
                    f'<span style="background:{_html.escape(color)};'
                    'width:12px;height:12px;border-radius:2px;'
                    'display:inline-block;margin-right:4px;vertical-align:middle;"></span>'
                )
            legend_parts.append(
                f'<span style="display:inline-flex;align-items:center;margin-right:12px;">'
                f'{swatch}<span style="font-size:11px;">{_html.escape(label)}</span></span>'
            )

        legend_html = (
            " ".join(legend_parts)
            if legend_parts
            else "<em style='color:#999'>No structures visible</em>"
        )

        inner = f"""<!DOCTYPE html>
<html><head>
<script src='https://3Dmol.csb.pitt.edu/build/3Dmol-min.js'></script>
<style>
  body{{margin:0;padding:0;background:#f8f8f8;font-family:sans-serif;}}
  #legend{{padding:5px 8px;font-size:11px;border-bottom:1px solid #e0e0e0;min-height:26px;}}
  #v{{position:absolute;top:30px;left:0;right:0;bottom:0;}}
</style>
</head><body>
<div id='legend'>{legend_html}</div>
<div id='v'></div>
<script>
window.addEventListener('load', function() {{
  var viewer = $3Dmol.createViewer(document.getElementById('v'), {{backgroundColor:'#f8f8f8'}});
  {models_js}
  viewer.zoomTo();
  viewer.render();
}});
</script>
</body></html>"""

        return (
            f'<iframe srcdoc="{_html.escape(inner, quote=True)}" '
            f'style="width:100%;height:{height}px;border:1px solid #ddd;'
            f'border-radius:4px;" frameborder="0"></iframe>'
        )

    return (
        BASIC_COLORS,
        atoms_to_pdb_str,
        build_confidence_figure,
        build_structure_html,
        compute_ipsae,
        default_color,
        get_b_range,
        normalize_full_data,
        normalize_summary,
        parse_structure,
        superimpose_all,
    )


@app.cell(hide_code=True)
def _header(mo):
    mo.md("""
    # OpenFold3 Structure Prediction

    [OpenFold3](https://github.com/aqlaboratory/openfold-3) is an open-source
    re-implementation of an AlphaFold3-like model for joint biomolecular structure
    prediction. Like AlphaFold3, it models any combination of **proteins, DNA, RNA,
    small-molecule ligands, and ions** in a single run, producing ranked structural models
    together with confidence metrics (pTM, ipTM, ipSAE, PAE).

    ### MSA requirement

    OpenFold3 performs best with a pre-computed **Multiple Sequence Alignment (MSA)**
    for each protein chain. Generating MSAs from scratch requires large sequence databases
    (hundreds of GB) that are not bundled with this notebook — use the MSA upload section
    below to supply `.a3m` files generated by:

    - **ColabFold** (fast, free, server-side) — run the [ColabFold MSA notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/MSA.ipynb) to get an `.a3m`
    - **jackhmmer** (HMMER3) or **HHblits** against UniRef/BFD locally

    If no MSA is provided the model runs in **single-sequence mode** (lower accuracy).

    ### Inputs
    | Type | How to specify |
    |------|---------------|
    | Protein | Amino-acid sequence (single-letter codes) |
    | DNA | Nucleotide sequence (A/T/G/C) |
    | RNA | Nucleotide sequence (A/U/G/C) |
    | Ligand | SMILES string |
    | Ion | CCD code (e.g. `MG`, `ZN`) |

    **Quick start:** enter a protein sequence, upload its `.a3m` MSA, click **▶ Run Prediction**.
    """)
    return


@app.cell(hide_code=True)
def _settings_ui(Path, mo):
    openfold_cmd_ui = mo.ui.text(
        value="run_openfold",
        label="OpenFold3 executable (path or command name)",
        full_width=True,
    )
    weights_dir_ui = mo.ui.text(
        value=str(Path.home() / ".cache/openfold3/params"),
        label="Model weights directory",
        full_width=True,
    )
    output_dir_ui = mo.ui.text(
        value=str(Path.home() / "openfold3_predictions"),
        label="Prediction output directory",
        full_width=True,
    )
    mo.vstack([
        mo.md("## Settings"),
        openfold_cmd_ui,
        weights_dir_ui,
        output_dir_ui,
    ])
    return openfold_cmd_ui, output_dir_ui, weights_dir_ui


@app.cell(hide_code=True)
def _entity_counts_ui(mo):
    job_name_ui = mo.ui.text(value="my_prediction", label="Job name")
    _opts = list(range(0, 11))
    n_protein_ui = mo.ui.dropdown(options=_opts, value=1, label="Protein chains")
    n_dna_ui = mo.ui.dropdown(options=_opts, value=0, label="DNA chains")
    n_rna_ui = mo.ui.dropdown(options=_opts, value=0, label="RNA chains")
    n_ligand_ui = mo.ui.dropdown(options=_opts, value=0, label="Ligands (SMILES)")
    n_ion_ui = mo.ui.dropdown(options=_opts, value=0, label="Ions (CCD codes)")
    mo.vstack([
        mo.md("## Input sequences"),
        job_name_ui,
        mo.md(
            "Choose how many entities of each type to include — "
            "matching fields will appear below for you to fill in."
        ),
        mo.hstack(
            [n_protein_ui, n_dna_ui, n_rna_ui, n_ligand_ui, n_ion_ui],
            justify="start",
            gap="2rem",
        ),
    ])
    return job_name_ui, n_dna_ui, n_ion_ui, n_ligand_ui, n_protein_ui, n_rna_ui


@app.cell(hide_code=True)
def _protein_forms_ui(mo, n_protein_ui):
    protein_forms_ui = mo.ui.array([
        mo.ui.dictionary({
            "sequence": mo.ui.text_area(
                placeholder="Amino acid sequence (1-letter codes)…",
                label=f"Chain {i + 1} — sequence",
                rows=3,
                full_width=True,
            ),
            "count": mo.ui.number(start=1, stop=20, value=1, label="Copies"),
        })
        for i in range(int(n_protein_ui.value))
    ])
    mo.vstack(
        [mo.md("### Protein chains")]
        + (
            [mo.vstack([f["sequence"], f["count"]]) for f in protein_forms_ui]
            if len(protein_forms_ui) > 0
            else [mo.md('*(none — increase "Protein chains" above to add one)*')]
        )
    )
    return (protein_forms_ui,)


@app.cell(hide_code=True)
def _dna_forms_ui(mo, n_dna_ui):
    dna_forms_ui = mo.ui.array([
        mo.ui.dictionary({
            "sequence": mo.ui.text(
                placeholder="e.g. ATGCATGC",
                label=f"Chain {i + 1} — sequence",
                full_width=True,
            ),
            "count": mo.ui.number(start=1, stop=20, value=1, label="Copies"),
        })
        for i in range(int(n_dna_ui.value))
    ])
    mo.vstack(
        [mo.md("### DNA chains")]
        + (
            [mo.hstack([f["sequence"], f["count"]], justify="start", gap="2rem") for f in dna_forms_ui]
            if len(dna_forms_ui) > 0
            else [mo.md("*(none)*")]
        )
    )
    return (dna_forms_ui,)


@app.cell(hide_code=True)
def _rna_forms_ui(mo, n_rna_ui):
    rna_forms_ui = mo.ui.array([
        mo.ui.dictionary({
            "sequence": mo.ui.text(
                placeholder="e.g. AUGCAUGC",
                label=f"Chain {i + 1} — sequence",
                full_width=True,
            ),
            "count": mo.ui.number(start=1, stop=20, value=1, label="Copies"),
        })
        for i in range(int(n_rna_ui.value))
    ])
    mo.vstack(
        [mo.md("### RNA chains")]
        + (
            [mo.hstack([f["sequence"], f["count"]], justify="start", gap="2rem") for f in rna_forms_ui]
            if len(rna_forms_ui) > 0
            else [mo.md("*(none)*")]
        )
    )
    return (rna_forms_ui,)


@app.cell(hide_code=True)
def _ligand_forms_ui(mo, n_ligand_ui):
    ligand_forms_ui = mo.ui.array([
        mo.ui.dictionary({
            "ligand": mo.ui.text(
                placeholder="SMILES string (e.g. NCCc1cc(O)c(O)cc1)…",
                label=f"Ligand {i + 1} — SMILES",
                full_width=True,
            ),
            "count": mo.ui.number(start=1, stop=20, value=1, label="Copies"),
        })
        for i in range(int(n_ligand_ui.value))
    ])
    mo.vstack(
        [mo.md("### Ligands")]
        + (
            [mo.hstack([f["ligand"], f["count"]], justify="start", gap="2rem") for f in ligand_forms_ui]
            if len(ligand_forms_ui) > 0
            else [mo.md("*(none)*")]
        )
    )
    return (ligand_forms_ui,)


@app.cell(hide_code=True)
def _ion_forms_ui(mo, n_ion_ui):
    ion_forms_ui = mo.ui.array([
        mo.ui.dictionary({
            "ion": mo.ui.text(
                placeholder="e.g. MG or ZN",
                label=f"Ion {i + 1} — CCD code",
                full_width=True,
            ),
            "count": mo.ui.number(start=1, stop=20, value=1, label="Copies"),
        })
        for i in range(int(n_ion_ui.value))
    ])
    mo.vstack(
        [mo.md("### Ions")]
        + (
            [mo.hstack([f["ion"], f["count"]], justify="start", gap="2rem") for f in ion_forms_ui]
            if len(ion_forms_ui) > 0
            else [mo.md("*(none)*")]
        )
    )
    return (ion_forms_ui,)


@app.cell(hide_code=True)
def _msa_ui(mo, n_protein_ui):
    use_msa_ui = mo.ui.switch(value=False, label="Provide MSA files (.a3m)")
    msa_forms_ui = mo.ui.array([
        mo.ui.file(filetypes=[".a3m"], label=f"Chain {chr(ord('A') + i)} — MSA (.a3m)")
        for i in range(int(n_protein_ui.value))
    ])
    mo.vstack([mo.md("## Multiple Sequence Alignments"), use_msa_ui])
    return msa_forms_ui, use_msa_ui


@app.cell(hide_code=True)
def _msa_detail(mo, msa_forms_ui, use_msa_ui):
    mo.stop(not use_msa_ui.value)
    mo.vstack([
        mo.callout(
            mo.md(
                "Upload one `.a3m` file per protein chain. "
                "Chains without an MSA file will run in single-sequence mode.  \n"
                "Get an MSA quickly with the free [ColabFold MSA server]"
                "(https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/MSA.ipynb)."
            ),
            kind="info",
        ),
        *[
            mo.vstack([
                mo.md(f"**Chain {chr(ord('A') + i)}**"),
                msa_forms_ui[i],
            ])
            for i in range(len(msa_forms_ui))
        ],
    ])
    return


@app.cell(hide_code=True)
def _build_inputs(
    dna_forms_ui,
    ion_forms_ui,
    job_name_ui,
    json,
    ligand_forms_ui,
    mo,
    protein_forms_ui,
    rna_forms_ui,
):
    _chains = []
    _chain_idx = 0

    def _get_chain_letter(idx):
        n = idx
        letters = ""
        while True:
            n, rem = divmod(n, 26)
            letters = chr(ord("A") + rem) + letters
            if n == 0:
                return letters
            n -= 1

    for _f in protein_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "protein",
                    "sequence": _seq,
                    "chain_ids": [_get_chain_letter(_chain_idx)],
                })
                _chain_idx += 1

    for _f in dna_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "dna",
                    "sequence": _seq,
                    "chain_ids": [_get_chain_letter(_chain_idx)],
                })
                _chain_idx += 1

    for _f in rna_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "rna",
                    "sequence": _seq,
                    "chain_ids": [_get_chain_letter(_chain_idx)],
                })
                _chain_idx += 1

    for _f in ligand_forms_ui.value:
        _lig = "".join(_f["ligand"].split())
        if _lig:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "ligand",
                    "smiles": _lig,
                    "chain_ids": [_get_chain_letter(_chain_idx)],
                })
                _chain_idx += 1

    for _f in ion_forms_ui.value:
        _ion = "".join(_f["ion"].split()).upper()
        if _ion:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "ligand",
                    "ccd_codes": [_ion],
                    "chain_ids": [_get_chain_letter(_chain_idx)],
                })
                _chain_idx += 1

    _job_name = job_name_ui.value.strip() or "prediction"
    input_data = {
        "queries": {
            _job_name: {
                "chains": _chains,
            }
        }
    }

    _json_text = json.dumps(input_data, indent=2)
    json_preview_ui = mo.ui.code_editor(
        value=_json_text,
        language="json",
        label="Input query JSON preview (MSA paths are injected at run time)",
    )
    mo.vstack([mo.md("## JSON preview"), json_preview_ui])
    return input_data, _job_name


@app.cell(hide_code=True)
def _params_ui(mo):
    seeds_ui = mo.ui.text(value="101", label="Seeds (comma-separated)", full_width=False)
    use_templates_ui = mo.ui.switch(value=True, label="Use templates")
    low_mem_ui = mo.ui.switch(value=True, label="Low-memory mode")
    output_format_ui = mo.ui.dropdown(["cif", "pdb"], value="cif", label="Output format")
    mo.vstack([
        mo.md("## Inference parameters"),
        mo.hstack([seeds_ui, output_format_ui], justify="start", gap="2rem"),
        mo.hstack([use_templates_ui, low_mem_ui], justify="start", gap="2rem"),
    ])
    return low_mem_ui, output_format_ui, seeds_ui, use_templates_ui


@app.cell(hide_code=True)
def _run_section(mo):
    run_btn = mo.ui.run_button(label="▶  Run Prediction")
    mo.vstack([
        mo.md("## Run Inference"),
        mo.callout(
            mo.md(
                "OpenFold3 must be installed and accessible via the executable path above. "
                "Model weights will be downloaded automatically to the weights directory "
                "if not already present. Prediction requires a GPU with sufficient memory."
            ),
            kind="warn",
        ),
        run_btn,
    ])
    return (run_btn,)


@app.cell(hide_code=True)
def _run_inference(
    _job_name,
    glob,
    input_data,
    json,
    low_mem_ui,
    mo,
    msa_forms_ui,
    openfold_cmd_ui,
    os,
    output_dir_ui,
    output_format_ui,
    protein_forms_ui,
    run_btn,
    seeds_ui,
    shutil,
    subprocess,
    use_msa_ui,
    use_templates_ui,
    weights_dir_ui,
):
    import yaml as _yaml

    mo.stop(not run_btn.value, mo.md("*Click **▶ Run Prediction** to start.*"))

    # ── Locate executable ───────────────────────────────────────────────────────
    _exe = shutil.which(openfold_cmd_ui.value)
    if not _exe:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"Executable `{openfold_cmd_ui.value}` not found on PATH. "
                    "Install OpenFold3 from [github.com/aqlaboratory/openfold-3]"
                    "(https://github.com/aqlaboratory/openfold-3) and set the "
                    "correct path above."
                ),
                kind="danger",
            ),
        )

    # ── Check / download model weights ─────────────────────────────────────────
    _weights_dir = weights_dir_ui.value
    os.makedirs(_weights_dir, exist_ok=True)
    _ckpt_files = (
        glob.glob(os.path.join(_weights_dir, "*.pt"))
        + glob.glob(os.path.join(_weights_dir, "*.bin"))
        + glob.glob(os.path.join(_weights_dir, "*.npz"))
    )
    if not _ckpt_files:
        with mo.status.spinner("Downloading OpenFold3 model weights…"):
            _dl = subprocess.run(
                [_exe, "download_params", "--output_dir", _weights_dir],
                text=True,
                capture_output=True,
            )
        if _dl.returncode != 0:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        "Model weights not found and automatic download failed "
                        f"(exit code {_dl.returncode}).  \n\n"
                        "Please download weights manually following the "
                        "[OpenFold3 setup instructions](https://github.com/aqlaboratory/openfold-3#model-parameters) "
                        "and set the weights directory above.\n\n"
                        f"```\n{_dl.stderr[-2000:]}\n```"
                    ),
                    kind="danger",
                ),
            )

    # ── Prepare output directory and write input JSON ──────────────────────────
    _pred_root = output_dir_ui.value
    _job_dir = os.path.join(_pred_root, _job_name)
    os.makedirs(_job_dir, exist_ok=True)

    # ── Write MSA files if uploaded; inject paths into chain dicts ────────────
    import copy as _copy
    _input = _copy.deepcopy(input_data)
    _all_chains = _input["queries"][_job_name]["chains"]
    if use_msa_ui.value:
        # Walk protein forms in order; each form may expand to count > 1 chain copies.
        # We assign the same a3m file to every copy so all instances share the MSA.
        _prot_chain_idx = 0
        for _form_idx, _pf in enumerate(protein_forms_ui.value):
            _seq = "".join(_pf["sequence"].split()).upper()
            if not _seq:
                continue
            _count = int(_pf["count"])
            _uploads = msa_forms_ui[_form_idx].value if _form_idx < len(msa_forms_ui) else []
            if _uploads:
                _msa_file = _uploads[0]
                _first_letter = _all_chains[_prot_chain_idx]["chain_ids"][0]
                _msa_dest = os.path.join(_job_dir, f"chain_{_first_letter}_msa.a3m")
                with open(_msa_dest, "wb") as _fh:
                    _fh.write(_msa_file.contents())
                for _ci in range(_count):
                    if _prot_chain_idx + _ci < len(_all_chains):
                        _all_chains[_prot_chain_idx + _ci]["msa_path"] = _msa_dest
            _prot_chain_idx += _count

    _json_path = os.path.join(_job_dir, "input_query.json")
    with open(_json_path, "w") as _fh:
        json.dump(_input, _fh, indent=2)

    # ── Build and write runner.yml ─────────────────────────────────────────────
    _seeds = [int(s.strip()) for s in seeds_ui.value.split(",") if s.strip().isdigit()]
    if not _seeds:
        _seeds = [101]

    _use_msa_server = not (use_msa_ui.value and any(
        msa_forms_ui[i].value for i in range(len(msa_forms_ui))
    ))

    _presets = ["predict"]
    if low_mem_ui.value:
        _presets.append("low_mem")

    _runner_cfg = {
        "experiment_settings": {
            "mode": "predict",
            "output_dir": _pred_root,
            "seeds": _seeds,
            "use_msa_server": _use_msa_server,
            "use_templates": use_templates_ui.value,
        },
        "model_update": {
            "presets": _presets,
        },
        "output_writer_settings": {
            "structure_format": output_format_ui.value,
            "full_confidence_output_format": "json",
            "write_full_confidence_scores": True,
        },
    }
    if weights_dir_ui.value:
        _runner_cfg["model_update"]["params_dir"] = weights_dir_ui.value

    _yaml_path = os.path.join(_job_dir, "runner.yml")
    with open(_yaml_path, "w") as _fh:
        _yaml.dump(_runner_cfg, _fh, default_flow_style=False)

    # ── Run inference ──────────────────────────────────────────────────────────
    _cmd = [_exe, "predict", "--query_json", _json_path, "--runner_yaml", _yaml_path]

    with mo.status.spinner("Running OpenFold3 prediction… (this may take several minutes)"):
        _result = subprocess.run(_cmd, text=True, capture_output=True)

    if _result.returncode != 0:
        inference_ok = False
        inference_error = (
            f"Inference failed (exit code {_result.returncode}).\n\n"
            f"**Stdout:**\n```\n{_result.stdout[-3000:]}\n```\n\n"
            f"**Stderr:**\n```\n{_result.stderr[-3000:]}\n```"
        )
    else:
        inference_ok = True
        inference_error = ""

    pred_out_root = _pred_root
    return inference_error, inference_ok, pred_out_root


@app.cell(hide_code=True)
def _find_results(
    glob,
    inference_error,
    inference_ok,
    input_data,
    json,
    mo,
    normalize_full_data,
    normalize_summary,
    os,
    parse_structure,
    pred_out_root,
):
    mo.stop(
        not inference_ok,
        mo.callout(mo.md(inference_error or "Run prediction first."), kind="danger"),
    )

    import re as _re

    _job_name = list(input_data["queries"].keys())[0]
    _pattern = os.path.join(pred_out_root, _job_name, "seed_*")
    _seed_dirs = sorted(glob.glob(_pattern))

    mo.stop(
        not _seed_dirs,
        mo.callout(
            mo.md(f"No prediction seed directory found under `{pred_out_root}/{_job_name}/seed_*/`."),
            kind="danger",
        ),
    )

    with mo.status.spinner("Loading predicted OpenFold3 models…"):
        _parsed = []
        for _sd in _seed_dirs:
            _seed_str = os.path.basename(_sd).replace("seed_", "")
            _structs = sorted(
                glob.glob(os.path.join(_sd, "*_model.cif"))
                + glob.glob(os.path.join(_sd, "*_model.pdb"))
            )
            for _struct_path in _structs:
                _m = _re.search(r"sample_(\d+)", os.path.basename(_struct_path))
                if not _m:
                    continue
                _sample_idx = int(_m.group(1))
                _prefix = _struct_path
                for _ext in ("_model.cif", "_model.pdb"):
                    if _prefix.endswith(_ext):
                        _prefix = _prefix[: -len(_ext)]
                        break

                _summary_path = _prefix + "_confidences_aggregated.json"
                _full_path = _prefix + "_confidences.json"

                if not os.path.exists(_summary_path):
                    continue

                with open(_summary_path) as _sf:
                    _summary = normalize_summary(json.load(_sf), "openfold3")

                _full_data = None
                if os.path.exists(_full_path):
                    with open(_full_path) as _ff:
                        _full_data = normalize_full_data(json.load(_ff), "openfold3")

                with open(_struct_path, "rb") as _cf:
                    _atoms = parse_structure(_cf.read(), os.path.basename(_struct_path))

                if _full_data is not None:
                    _full_data = normalize_full_data(_full_data, "openfold3", _atoms)

                _parsed.append({
                    "index": f"Seed {_seed_str} Sample {_sample_idx}",
                    "summary": _summary,
                    "full_data": _full_data,
                    "atoms": _atoms,
                })

    mo.stop(
        not _parsed,
        mo.callout(mo.md("Could not parse any predicted model from the output directory."), kind="danger"),
    )

    _parsed.sort(key=lambda m: -m["summary"].get("ranking_score", float("-inf")))
    for _rank, _m in enumerate(_parsed, start=1):
        _m["rank"] = _rank

    prediction_models = _parsed
    mo.vstack([
        mo.md("## Results"),
        mo.callout(mo.md(f"Parsed **{len(prediction_models)}** predicted model(s)."), kind="success"),
    ])
    return (prediction_models,)


@app.cell(hide_code=True)
def _confidence_table(mo, prediction_models):
    import pandas as _pd

    _rows = []
    for _m in prediction_models:
        _s = _m["summary"]
        _rows.append({
            "Rank": _m["rank"],
            "Sample": _m["index"],
            "pTM":  round(_s["ptm"], 2)  if _s.get("ptm")  is not None else None,
            "ipTM": round(_s["iptm"], 2) if _s.get("iptm") is not None else None,
            "Ranking score": round(_s["ranking_score"], 2) if _s.get("ranking_score") is not None else None,
            "Has clash": bool(_s.get("has_clash")),
            "Fraction disordered": round(_s["fraction_disordered"], 2) if _s.get("fraction_disordered") is not None else None,
        })
    _df = _pd.DataFrame(_rows).sort_values("Rank").reset_index(drop=True)

    mo.vstack([
        mo.md(
            "### Confidence metrics\n\n"
            "**pTM** estimates overall fold accuracy (0–1, higher is better). "
            "**ipTM** applies the same idea to inter-chain interfaces — above ~0.8 "
            "indicates a confident interface. Models are ranked by **ranking score**."
        ),
        mo.ui.table(_df, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _confidence_figures(build_confidence_figure, mo, prediction_models):
    _figs = [
        mo.as_html(build_confidence_figure(_m["index"], _m["summary"], _m["full_data"], rank=_m["rank"]))
        for _m in prediction_models
        if _m["full_data"] is not None
    ]

    mo.stop(
        not _figs,
        mo.callout(
            mo.md("Confidence files containing PAE matrices are missing or were not written."),
            kind="info",
        ),
    )

    _has_chain_matrix = any(
        _m["full_data"] is not None
        and _m["summary"].get("chain_pair_iptm") is not None
        and len(_m["summary"]["chain_pair_iptm"]) > 2
        for _m in prediction_models
    )

    _intro = (
        "### Predicted aligned error (PAE)\n\n"
        "Cell *(i, j)* shows the model's expected position error (Å) for residue *j* "
        "when aligned on residue *i*. Dark green blocks on the diagonal indicate "
        "well-resolved chains/domains."
    )
    if _has_chain_matrix:
        _intro += (
            "\n\nFor assemblies with more than two chains, the **pairwise ipTM matrix** "
            "shows the predicted interface accuracy for every chain pair."
        )

    mo.vstack([mo.md(_intro), *_figs])
    return


@app.cell(hide_code=True)
def _ipsae_intro(mo, prediction_models):
    _has_full_data = any(_m["full_data"] is not None for _m in prediction_models)
    mo.stop(
        not _has_full_data,
        mo.callout(mo.md("Full confidence data (PAE matrices) are needed to compute ipSAE."), kind="info"),
    )

    ipsae_cutoff = mo.ui.slider(5, 30, value=10, step=1, label="PAE cutoff for ipSAE (Å)", show_value=True)

    mo.vstack([
        mo.md(r"""
        ### ipSAE — interface confidence between chains

        **ipSAE** ([Dunbrack et al.](https://www.biorxiv.org/content/10.1101/2025.02.10.637595))
        is an alternative to ipTM for scoring inter-chain interfaces. It only counts
        residue pairs whose predicted aligned error is below the cutoff, and normalises
        each residue's contribution by how many partner-chain residues are confidently
        placed near it. Like ipTM, scores above ~0.5 generally indicate a confident interface.
        """),
        ipsae_cutoff,
    ])
    return (ipsae_cutoff,)


@app.cell(hide_code=True)
def _ipsae_table(compute_ipsae, ipsae_cutoff, mo, prediction_models):
    import pandas as _pd

    _rows = []
    for _m in prediction_models:
        if _m["full_data"] is None:
            continue
        _scores = compute_ipsae(
            _m["full_data"]["pae"],
            _m["full_data"]["token_chain_ids"],
            pae_cutoff=float(ipsae_cutoff.value),
        )
        for (_c1, _c2), _score in _scores.items():
            _rows.append({
                "Sample": f"{_m['index']} (rank {_m['rank']})",
                "Chain 1": _c1,
                "Chain 2": _c2,
                "ipSAE": round(_score, 3),
            })

    mo.stop(not _rows, mo.callout(mo.md("No chain pairs to score."), kind="info"))
    mo.ui.table(_pd.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _structures(atoms_to_pdb_str, get_b_range, mo, prediction_models, superimpose_all):
    import pandas as _pd

    _atoms_list = [m["atoms"] for m in prediction_models]
    _labels = [f"{m['index']} (rank {m['rank']})" for m in prediction_models]

    with mo.status.spinner("Superimposing predicted models on Cα atoms…"):
        _aligned, _rmsds = superimpose_all(_atoms_list)

    structure_labels = _labels
    structure_pdbs = [atoms_to_pdb_str(a) for a in _aligned]
    b_ranges = [get_b_range(a) for a in _aligned]

    _rows = [{"Sample": _labels[0], "RMSD vs top-ranked (Å)": "0.00 (reference)"}]
    for _label, _rmsd in zip(_labels[1:], _rmsds):
        _rows.append({
            "Sample": _label,
            "RMSD vs top-ranked (Å)": f"{_rmsd:.2f}" if _rmsd == _rmsd else "N/A",
        })

    mo.vstack([
        mo.md(
            "### Predicted structures\n\n"
            "All models are superimposed onto the top-ranked prediction's Cα atoms. "
            "Switch **Color mode → B-factor** to display **pLDDT** confidence "
            "(blue/green = confident, red = uncertain)."
        ),
        mo.ui.table(_pd.DataFrame(_rows), selection=None),
    ])
    return b_ranges, structure_labels, structure_pdbs


@app.cell(hide_code=True)
def _structure_controls(mo, structure_labels, BASIC_COLORS, default_color):
    _n = len(structure_labels)
    _color_names = list(BASIC_COLORS.keys())

    global_visible = mo.ui.dropdown(
        ["Show all", "Hide all", "Custom"], value="Show all", label="Visibility"
    )
    global_color_mode = mo.ui.dropdown(
        ["Individual", "Solid color", "B-factor", "By chain"],
        value="Individual",
        label="Color mode",
    )
    global_color = mo.ui.dropdown(
        ["Individual"] + _color_names, value="Individual", label="Color"
    )

    visible_controls = mo.ui.array([mo.ui.checkbox(True) for _ in range(_n)])
    color_mode_controls = mo.ui.array(
        [mo.ui.dropdown(["Solid color", "B-factor", "By chain"], value="Solid color") for _ in range(_n)]
    )
    color_controls = mo.ui.array(
        [mo.ui.dropdown(_color_names, value=default_color(i)) for i in range(_n)]
    )

    _global_row = mo.hstack(
        [mo.md("**All structures:**"), global_visible, global_color_mode, global_color],
        gap=2,
        align="center",
    )
    _hdr = mo.hstack(
        [
            mo.Html('<div style="min-width:220px;max-width:220px;font-weight:600;font-size:13px;">Sample</div>'),
            mo.Html('<div style="min-width:36px;font-weight:600;font-size:13px;">Vis.</div>'),
            mo.Html('<div style="min-width:155px;font-weight:600;font-size:13px;">Color mode</div>'),
            mo.Html('<div style="font-weight:600;font-size:13px;">Solid color</div>'),
        ],
        gap=1, align="center",
    )
    _rows = [
        mo.hstack(
            [
                mo.Html(
                    f'<div style="min-width:220px;max-width:220px;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
                    f'font-family:monospace;font-size:13px;" title="{structure_labels[i]}">'
                    f'{structure_labels[i]}</div>'
                ),
                visible_controls[i],
                color_mode_controls[i],
                color_controls[i],
            ],
            gap=1, align="center",
        )
        for i in range(_n)
    ]

    mo.vstack([
        mo.md("### Structure controls"),
        _global_row,
        mo.Html("<hr style='margin:6px 0;border:none;border-top:1px solid #e0e0e0;'>"),
        _hdr,
        *_rows,
    ])
    return color_controls, color_mode_controls, global_color, global_color_mode, global_visible, visible_controls


@app.cell(hide_code=True)
def _viewer_3d(
    BASIC_COLORS,
    b_ranges,
    build_structure_html,
    color_controls,
    color_mode_controls,
    global_color,
    global_color_mode,
    global_visible,
    mo,
    structure_labels,
    structure_pdbs,
    visible_controls,
):
    _n = len(structure_labels)

    if global_visible.value == "Show all":
        _visibilities = [True] * _n
    elif global_visible.value == "Hide all":
        _visibilities = [False] * _n
    else:
        _visibilities = visible_controls.value

    _color_modes = (
        color_mode_controls.value
        if global_color_mode.value == "Individual"
        else [global_color_mode.value] * _n
    )

    _hex_colors = (
        [BASIC_COLORS[c] for c in color_controls.value]
        if global_color.value == "Individual"
        else [BASIC_COLORS[global_color.value]] * _n
    )

    mo.Html(
        build_structure_html(
            pdb_contents=structure_pdbs,
            labels=structure_labels,
            visibilities=_visibilities,
            color_modes=_color_modes,
            colors=_hex_colors,
            b_ranges=b_ranges,
        )
    )
    return


@app.cell(hide_code=True)
def _download_section(compute_ipsae, glob, input_data, io, mo, os, pred_out_root, prediction_models, zipfile):
    import pandas as _pd

    _IPSAE_CUTOFF = 10.0
    _job_name = list(input_data["queries"].keys())[0]

    _rows = []
    for _m in prediction_models:
        if _m["full_data"] is None:
            continue
        _scores = compute_ipsae(
            _m["full_data"]["pae"],
            _m["full_data"]["token_chain_ids"],
            pae_cutoff=_IPSAE_CUTOFF,
        )
        for (_c1, _c2), _score in _scores.items():
            _rows.append({
                "Sample": f"{_m['index']} (rank {_m['rank']})",
                "Chain 1": _c1,
                "Chain 2": _c2,
                "ipSAE": round(_score, 3),
            })
    _ipsae_df = _pd.DataFrame(_rows)
    if not _ipsae_df.empty:
        _ipsae_df.to_csv(os.path.join(pred_out_root, _job_name, "ipsae_scores.csv"), index=False)

    def _make_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _fp in sorted(glob.glob(os.path.join(pred_out_root, "**"), recursive=True)):
                if os.path.isfile(_fp):
                    zf.write(_fp, os.path.relpath(_fp, pred_out_root))
        buf.seek(0)
        return buf.read()

    _ipsae_note = (
        f"ipSAE scores (PAE cutoff {_IPSAE_CUTOFF} Å) are written to `ipsae_scores.csv` inside the zip."
        if not _ipsae_df.empty
        else "No ipSAE scores (full confidence data not available)."
    )

    mo.vstack([
        mo.md("### Download results"),
        mo.md(_ipsae_note),
        mo.download(
            data=_make_zip,
            filename=f"{_job_name}_openfold3.zip",
            mimetype="application/zip",
            label="Download results (.zip)",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
