# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "protenix",
#     "biotite",
#     "matplotlib==3.10.5",
#     "numpy==2.4.1",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Protenix Structure Prediction")


@app.cell(hide_code=True)
def _imports():
    import glob
    import json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    return Path, glob, json, mo, os, shutil, subprocess, sys


@app.cell(hide_code=True)
def _viz_helpers():
    """Confidence-figure / ipSAE / structure-viewer building blocks.

    Inlined here — rather than imported from `widget.af3_helpers` /
    `widget.structure_helpers` — so this notebook stays a single self-contained
    file that runs in sandboxed environments (e.g. MoLab) where the local
    `widget` package can't be resolved. They are otherwise identical to the
    AlphaFold3 prediction viewer widget's, so once Protenix's confidence JSON is
    normalized onto AlphaFold3's schema (`normalize_summary` /
    `normalize_full_data`), the exact same visualizations apply to both.
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
        """Render a 0-based numeric chain index the way AlphaFold3 labels chains (A, B, … Z, AA, …)."""
        n = int(asym_id)
        letters = ""
        while True:
            n, rem = divmod(n, 26)
            letters = chr(ord("A") + rem) + letters
            if n == 0:
                return letters
            n -= 1

    def normalize_summary(raw, source):
        """Map Protenix's `disorder` onto AlphaFold3's `fraction_disordered` — the
        only summary-confidence key the two tools name differently."""
        summary = dict(raw)
        if source == "protenix" and "fraction_disordered" not in summary and "disorder" in summary:
            summary["fraction_disordered"] = summary["disorder"]
        return summary

    def normalize_full_data(raw, source):
        """Alias Protenix's `token_pair_pae` / `token_asym_id` / `atom_plddt` onto
        AlphaFold3's `pae` / `token_chain_ids` / `atom_plddts` so the figure and
        ipSAE helpers below work unchanged for either source."""
        full_data = dict(raw)
        if source == "protenix":
            if "pae" not in full_data and "token_pair_pae" in full_data:
                full_data["pae"] = full_data["token_pair_pae"]
            if "token_chain_ids" not in full_data and "token_asym_id" in full_data:
                full_data["token_chain_ids"] = [_chain_letter(a) for a in full_data["token_asym_id"]]
            if "atom_plddts" not in full_data and "atom_plddt" in full_data:
                full_data["atom_plddts"] = full_data["atom_plddt"]
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
            " ".join(legend_parts) if legend_parts else "<em style='color:#999'>No structures visible</em>"
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
    # Protenix Structure Prediction

    Run protein structure prediction locally using
    [Protenix](https://github.com/bytedance/protenix).
    Model weights and data cache are downloaded automatically on the first run.

    Results are explored with the same confidence-metrics, PAE/ipSAE and
    multi-model 3D structure viewer as the AlphaFold3 prediction viewer widget —
    so Protenix and AlphaFold Server outputs can be compared side by side.

    This notebook declares its own dependencies (PEP 723 inline metadata) —
    launch it in an isolated, notebook-specific environment managed by
    [`uv`](https://docs.astral.sh/uv/) with:

    ```
    uvx marimo edit --sandbox marimo/predict.py
    ```

    (or `marimo run --sandbox marimo/predict.py` to view it read-only). No
    `pip install` needed — `uv` resolves and caches the environment for you,
    separate from the rest of the repo.

    **Quick start:** enter a protein sequence, pick
    `protenix_mini_default_v0.5.0`, set 1 sample / 5 steps, click
    **▶ Run Prediction**.
    """)
    return


@app.cell(hide_code=True)
def _settings_ui(Path, mo):
    weights_dir_ui = mo.ui.text(
        value=str(Path.home() / ".cache/protenix"),
        label="Weights & data cache directory",
        full_width=True,
    )
    output_dir_ui = mo.ui.text(
        value=str(Path(__file__).parent / "output"),
        label="Prediction output directory",
        full_width=True,
    )
    mo.vstack([
        mo.md("## Settings"),
        weights_dir_ui,
        output_dir_ui,
    ])
    return output_dir_ui, weights_dir_ui


@app.cell(hide_code=True)
def _model_ui(mo):
    _MODELS = [
        "protenix-v2",
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0",
        "protenix_base_default_v0.5.0",
        "protenix_base_constraint_v0.5.0",
        "protenix_mini_default_v0.5.0",
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_ism_v0.5.0",
        "protenix_tiny_default_v0.5.0",
    ]
    _DESC = {
        "protenix-v2":                      "464 M · MSA + RNA MSA + Template  (newest, best quality)",
        "protenix_base_default_v1.0.0":     "368 M · MSA + RNA MSA + Template",
        "protenix_base_20250630_v1.0.0":    "368 M · MSA + RNA MSA + Template · 2025-06 cutoff",
        "protenix_base_default_v0.5.0":     "368 M · MSA only",
        "protenix_base_constraint_v0.5.0":  "368 M · MSA + constraint support",
        "protenix_mini_default_v0.5.0":     "134 M · MSA only · fast",
        "protenix_mini_esm_v0.5.0":         "135 M · MSA + ESM · fast",
        "protenix_mini_ism_v0.5.0":         "135 M · MSA + ISM · fast",
        "protenix_tiny_default_v0.5.0":     "110 M · MSA only · ultra-fast",
    }
    _DEFAULT_MODEL = "protenix_mini_default_v0.5.0"
    model_ui = mo.ui.dropdown(
        options={f"{m}  ({_DESC[m]})": m for m in _MODELS},
        value=f"{_DEFAULT_MODEL}  ({_DESC[_DEFAULT_MODEL]})",
        label="Model",
    )
    mo.vstack([mo.md("## Model"), model_ui])
    return (model_ui,)


@app.cell(hide_code=True)
def _entity_counts_ui(mo):
    job_name_ui = mo.ui.text(value="my_prediction", label="Job name")
    _opts = list(range(0, 11))
    n_protein_ui = mo.ui.dropdown(options=_opts, value=1, label="Protein chains")
    n_dna_ui = mo.ui.dropdown(options=_opts, value=0, label="DNA chains")
    n_rna_ui = mo.ui.dropdown(options=_opts, value=0, label="RNA chains")
    n_ligand_ui = mo.ui.dropdown(options=_opts, value=0, label="Ligands")
    n_ion_ui = mo.ui.dropdown(options=_opts, value=0, label="Ions")
    mo.vstack([
        mo.md("## Input sequences"),
        job_name_ui,
        mo.md(
            "Choose how many entities of each type to include — matching "
            "fields will appear below for you to fill in."
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
                placeholder="Amino acid sequence (1-letter codes; X for unknown)…",
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
            else [mo.md("*(none — increase “Protein chains” above to add one)*")]
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
                placeholder="CCD_ATP  or  CCD_NAG  or  SMILES string",
                label=f"Ligand {i + 1} — CCD code or SMILES",
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
                placeholder="MG  or  ZN  (CCD code without 'CCD_' prefix)",
                label=f"Ion {i + 1}",
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
def _params_header(mo):
    mo.md("""
    ## Inference parameters
    """)
    return


@app.cell(hide_code=True)
def _params_widgets(mo):
    use_msa_ui = mo.ui.switch(value=True, label="Use MSA (slower, higher quality)")
    atom_conf_ui = mo.ui.switch(
        value=True, label="Compute atom-level confidence (enables PAE/PDE plots)"
    )
    n_cycle_ui = mo.ui.slider(start=1, stop=20, value=10, step=1, label="Pairformer cycles")
    n_step_ui = mo.ui.slider(start=1, stop=400, value=200, step=1, label="Diffusion steps")
    n_sample_ui = mo.ui.slider(start=1, stop=10, value=5, step=1, label="Samples per seed")
    seeds_ui = mo.ui.text(value="101", label="Seeds (comma-separated)")
    dtype_ui = mo.ui.dropdown(options=["bf16", "fp32"], value="bf16", label="Compute dtype")
    mo.vstack([
        mo.hstack([use_msa_ui, atom_conf_ui], justify="start", gap="3rem"),
        mo.hstack([n_cycle_ui, n_step_ui, n_sample_ui], justify="start", gap="2rem"),
        mo.hstack([seeds_ui, dtype_ui], justify="start", gap="2rem"),
    ])
    return (
        atom_conf_ui,
        dtype_ui,
        n_cycle_ui,
        n_sample_ui,
        n_step_ui,
        seeds_ui,
        use_msa_ui,
    )


@app.cell(hide_code=True)
def _build_json(
    dna_forms_ui,
    ion_forms_ui,
    job_name_ui,
    json,
    ligand_forms_ui,
    mo,
    protein_forms_ui,
    rna_forms_ui,
):
    _sequences = []
    for _f in protein_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            _sequences.append({"proteinChain": {"sequence": _seq, "count": int(_f["count"])}})
    for _f in dna_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            _sequences.append({"dnaSequence": {"sequence": _seq, "count": int(_f["count"])}})
    for _f in rna_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            _sequences.append({"rnaSequence": {"sequence": _seq, "count": int(_f["count"])}})
    for _f in ligand_forms_ui.value:
        _lig = "".join(_f["ligand"].split())
        if _lig:
            _sequences.append({"ligand": {"ligand": _lig, "count": int(_f["count"])}})
    for _f in ion_forms_ui.value:
        _ion = "".join(_f["ion"].split()).upper()
        if _ion:
            _sequences.append({"ion": {"ion": _ion, "count": int(_f["count"])}})

    input_data = [
        {
            "name": job_name_ui.value.strip() or "prediction",
            "sequences": _sequences,
            "covalent_bonds": [],
        }
    ]

    _json_text = json.dumps(input_data, indent=2)
    json_preview_ui = mo.ui.code_editor(
        value=_json_text,
        language="json",
        label="Input JSON preview (edit fields above to update)",
    )
    mo.vstack([mo.md("## JSON preview"), json_preview_ui])
    return (input_data,)


@app.cell(hide_code=True)
def _run_section(mo):
    run_btn = mo.ui.run_button(label="▶  Run Prediction")
    mo.vstack([
        mo.md("## Run"),
        mo.callout(
            mo.md(
                "Inference may take **1–30 min** depending on the model and sequence length. "
                "Weights are downloaded automatically on first use (~200 MB–500 MB)."
            ),
            kind="warn",
        ),
        run_btn,
    ])
    return (run_btn,)


@app.cell(hide_code=True)
def _run_inference(
    Path,
    atom_conf_ui,
    dtype_ui,
    glob,
    input_data,
    json,
    mo,
    model_ui,
    n_cycle_ui,
    n_sample_ui,
    n_step_ui,
    os,
    output_dir_ui,
    run_btn,
    seeds_ui,
    shutil,
    subprocess,
    sys,
    use_msa_ui,
    weights_dir_ui,
):
    mo.stop(not run_btn.value, mo.md("*Click **▶ Run Prediction** to start.*"))

    pred_out_root = output_dir_ui.value
    os.makedirs(pred_out_root, exist_ok=True)

    _json_path = os.path.join(pred_out_root, "input.json")
    with open(_json_path, "w") as _f:
        json.dump(input_data, _f, indent=2)

    # Locate the protenix CLI; fall back to running batch_inference.py directly
    _exe = shutil.which("protenix")
    if _exe:
        _cmd = [_exe, "pred"]
    else:
        _bi = str(Path(__file__).parent.parent / "runner" / "batch_inference.py")
        _cmd = [sys.executable, _bi, "pred"]

    # ── Detect available CUDA facilities ────────────────────────────────────
    # Protenix defaults to cuequivariance kernels for triangle ops; those load
    # libcue_ops.so which dlopen()s libnvrtc.so.12 at runtime. On many GPU
    # sandboxes the CUDA toolkit is installed but its lib64 directory is absent
    # from LD_LIBRARY_PATH, causing an "cannot open shared object file" crash.
    # We probe for libnvrtc on disk and inject the directory if found so the
    # faster Blackwell-optimised kernels can be used.
    _cuda_lib_dirs = {
        os.path.dirname(p)
        for pat in [
            "/usr/local/cuda*/lib64/libnvrtc.so*",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
            "/usr/lib/aarch64-linux-gnu/libnvrtc.so*",
        ]
        for p in glob.glob(pat)
    }
    # Also probe for the CUDA toolkit root via nvcc — needed for FusedLayerNorm
    # JIT compilation (separate from the NVRTC runtime library above).
    _nvcc_path = shutil.which("nvcc")
    _cuda_home = None
    if _nvcc_path:
        _cuda_home = str(Path(_nvcc_path).resolve().parent.parent)
    else:
        for _cand in sorted(glob.glob("/usr/local/cuda*"), reverse=True):
            if os.path.isfile(os.path.join(_cand, "bin", "nvcc")):
                _cuda_home = _cand
                break
    if _cuda_home:
        _cuda_lib_dirs.add(os.path.join(_cuda_home, "lib64"))

    _has_nvrtc = bool(_cuda_lib_dirs)
    _ld_parts = [d for d in _cuda_lib_dirs if d]
    if os.environ.get("LD_LIBRARY_PATH"):
        _ld_parts.append(os.environ["LD_LIBRARY_PATH"])

    _env = {
        **os.environ,
        "PROTENIX_ROOT_DIR": weights_dir_ui.value,
        # FusedLayerNorm: use Protenix's fused CUDA kernel if nvcc is available,
        # else fall back to the plain PyTorch implementation (avoids the JIT
        # compile step and the CUDA_HOME requirement).
        "LAYERNORM_TYPE": "fast_layernorm" if _cuda_home else "torch_layernorm",
        **({"CUDA_HOME": _cuda_home} if _cuda_home else {}),
        **({"LD_LIBRARY_PATH": ":".join(_ld_parts)} if _ld_parts else {}),
    }
    # ────────────────────────────────────────────────────────────────────────

    _cmd += [
        "-i", _json_path,
        "-o", pred_out_root,
        "-n", model_ui.value,
        "-s", seeds_ui.value,
        "-c", str(int(n_cycle_ui.value)),
        "-p", str(int(n_step_ui.value)),
        "-e", str(int(n_sample_ui.value)),
        "-d", dtype_ui.value,
        "--use_msa", str(use_msa_ui.value),
        "--need_atom_confidence", str(atom_conf_ui.value),
        # Use cuequivariance kernels (Blackwell-optimised) when libnvrtc is
        # accessible; fall back to pure PyTorch if the CUDA toolkit is absent.
        "--trimul_kernel", "cuequivariance" if _has_nvrtc else "torch",
        "--triatt_kernel", "cuequivariance" if _has_nvrtc else "torch",
    ]

    with mo.status.spinner("Running Protenix prediction…"):
        _result = subprocess.run(_cmd, env=_env, text=True)

    if _result.returncode != 0:
        inference_ok = False
        inference_error = f"Inference failed with exit code {_result.returncode}. Check terminal output above."
    else:
        inference_ok = True
        inference_error = ""
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

    _job_name = input_data[0]["name"]
    _pattern = os.path.join(pred_out_root, _job_name, "seed_*", "predictions")
    _pred_dirs = sorted(glob.glob(_pattern))

    mo.stop(
        not _pred_dirs,
        mo.callout(
            mo.md(f"No prediction output found under `{pred_out_root}/{_job_name}/seed_*/predictions/`."),
            kind="danger",
        ),
    )

    # Parse every sample into the same {index, summary, full_data, atoms} shape
    # the AlphaFold3 prediction viewer widget uses, normalizing Protenix's
    # confidence JSON onto AlphaFold3's schema (`normalize_summary` /
    # `normalize_full_data`) so the very same visualizations apply to both.
    with mo.status.spinner("Loading predicted models…"):
        _parsed = []
        for _pd in _pred_dirs:
            for _cif in sorted(glob.glob(os.path.join(_pd, "*.cif"))):
                _stem = os.path.splitext(os.path.basename(_cif))[0]  # e.g. my_prediction_sample_0
                _summary_path = os.path.join(_pd, _stem.replace("_sample_", "_summary_confidence_sample_", 1) + ".json")
                _full_path = os.path.join(_pd, _stem.replace("_sample_", "_full_data_sample_", 1) + ".json")

                if not os.path.exists(_summary_path):
                    continue

                with open(_summary_path) as _sf:
                    _summary = normalize_summary(json.load(_sf), "protenix")

                _full_data = None
                if os.path.exists(_full_path):
                    with open(_full_path) as _ff:
                        _full_data = normalize_full_data(json.load(_ff), "protenix")

                with open(_cif, "rb") as _cf:
                    _atoms = parse_structure(_cf.read(), os.path.basename(_cif))

                _suffix = _stem.rsplit("_sample_", 1)[-1]
                _index = int(_suffix) if _suffix.isdigit() else _stem

                _parsed.append({
                    "index": _index,
                    "summary": _summary,
                    "full_data": _full_data,
                    "atoms": _atoms,
                })

    mo.stop(
        not _parsed,
        mo.callout(mo.md("Could not parse any predicted model."), kind="danger"),
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
            "pTM": round(_s["ptm"], 2) if _s.get("ptm") is not None else None,
            "ipTM": round(_s["iptm"], 2) if _s.get("iptm") is not None else None,
            "Ranking score": round(_s["ranking_score"], 2) if _s.get("ranking_score") is not None else None,
            "Has clash": bool(_s.get("has_clash")),
            "Fraction disordered": round(_s["fraction_disordered"], 2) if _s.get("fraction_disordered") is not None else None,
        })
    _df = _pd.DataFrame(_rows).sort_values("Rank").reset_index(drop=True)

    mo.vstack([
        mo.md(
            "### Confidence metrics\n\n"
            "**pTM** (predicted TM-score) estimates how well the *overall* predicted fold "
            "matches the true structure — values range from 0 to 1, higher is better, and "
            "above ~0.5 generally indicates a correct overall topology.\n\n"
            "**ipTM** (interface pTM) applies the same idea to *inter-chain interfaces* and "
            "is the key signal for multi-chain complexes — above ~0.8 indicates a confident "
            "interface, below ~0.6 suggests the chains may not actually interact this way. "
            "Models are ranked by Protenix's combined **ranking score**."
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
            mo.md("Enable **Compute atom-level confidence** and re-run to see PAE matrices."),
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
        "Cell *(i, j)* shows the model's expected position error (in Å) for residue *j* "
        "when the prediction is aligned on residue *i*. Dark green blocks on the diagonal "
        "indicate well-resolved chains/domains; dark green *off-diagonal* blocks indicate "
        "confidently predicted relative arrangements between chains."
    )
    if _has_chain_matrix:
        _intro += (
            "\n\nFor assemblies with more than two chains, the **pairwise ipTM matrix** "
            "alongside the PAE plot shows the predicted interface accuracy for every chain "
            "pair — a quick way to spot which interfaces in the complex are confidently "
            "modeled and which are not."
        )

    mo.vstack([mo.md(_intro), *_figs])
    return


@app.cell(hide_code=True)
def _ipsae_intro(mo, prediction_models):
    _has_full_data = any(_m["full_data"] is not None for _m in prediction_models)
    mo.stop(
        not _has_full_data,
        mo.callout(mo.md("Enable **Compute atom-level confidence** and re-run to compute ipSAE."), kind="info"),
    )

    ipsae_cutoff = mo.ui.slider(
        5, 30, value=10, step=1, label="PAE cutoff for ipSAE (Å)", show_value=True
    )

    mo.vstack([
        mo.md(r"""
        ### ipSAE — interface confidence between chains

        **ipSAE** ([Dunbrack et al.](https://www.biorxiv.org/content/10.1101/2025.02.10.637595))
        is an alternative to ipTM for scoring inter-chain interfaces. It only counts
        residue pairs whose predicted aligned error is below the cutoff below, and
        normalises each residue's contribution by how many partner-chain residues are
        confidently placed near it. This makes it markedly more sensitive than ipTM to
        confidently docked sub-interfaces within larger or partly flexible assemblies —
        where a single weak/disordered chain can otherwise drag the global ipTM down.

        Like ipTM, scores above ~0.5 generally indicate a confidently predicted interface.
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
                "Sample": f"Sample {_m['index']} (rank {_m['rank']})",
                "Chain 1": _c1,
                "Chain 2": _c2,
                "ipSAE": round(_score, 3),
            })

    mo.stop(not _rows, mo.callout(mo.md("No chain pairs to score."), kind="info"))

    _df = _pd.DataFrame(_rows)
    mo.ui.table(_df, selection=None)
    return


@app.cell(hide_code=True)
def _structures(atoms_to_pdb_str, get_b_range, mo, prediction_models, superimpose_all):
    import pandas as _pd

    _atoms_list = [m["atoms"] for m in prediction_models]
    _labels = [f"Sample {m['index']} (rank {m['rank']})" for m in prediction_models]

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
            "Large inter-model RMSDs can reflect genuine conformational ambiguity rather "
            "than poor predictions — cross-check against the pTM/ipTM/PAE values above. "
            "Switch **Color mode → B-factor** in the controls below to display each "
            "residue's **pLDDT** confidence score (blue/green = confident, "
            "red = disordered or uncertain)."
        ),
        mo.ui.table(_pd.DataFrame(_rows), selection=None),
    ])
    return b_ranges, structure_labels, structure_pdbs


@app.cell(hide_code=True)
def _structure_controls(mo, structure_labels, BASIC_COLORS, default_color):
    _n = len(structure_labels)
    _color_names = list(BASIC_COLORS.keys())

    # ── Global controls ────────────────────────────────────────────────────────
    global_visible = mo.ui.dropdown(
        ["Show all", "Hide all", "Custom"],
        value="Show all",
        label="Visibility",
    )
    global_color_mode = mo.ui.dropdown(
        ["Individual", "Solid color", "B-factor"],
        value="Individual",
        label="Color mode",
    )
    global_color = mo.ui.dropdown(
        ["Individual"] + _color_names,
        value="Individual",
        label="Color",
    )

    # ── Per-structure controls ─────────────────────────────────────────────────
    visible_controls = mo.ui.array(
        [mo.ui.checkbox(True) for _ in range(_n)]
    )
    color_mode_controls = mo.ui.array(
        [mo.ui.dropdown(["Solid color", "B-factor"], value="Solid color") for _ in range(_n)]
    )
    color_controls = mo.ui.array(
        [mo.ui.dropdown(_color_names, value=default_color(i)) for i in range(_n)]
    )

    # ── Layout ─────────────────────────────────────────────────────────────────
    _global_row = mo.hstack(
        [
            mo.md("**All structures:**"),
            global_visible,
            global_color_mode,
            global_color,
        ],
        gap=2,
        align="center",
    )

    _header = mo.hstack(
        [
            mo.Html('<div style="min-width:220px;max-width:220px;font-weight:600;font-size:13px;">Sample</div>'),
            mo.Html('<div style="min-width:36px;font-weight:600;font-size:13px;">Vis.</div>'),
            mo.Html('<div style="min-width:155px;font-weight:600;font-size:13px;">Color mode</div>'),
            mo.Html('<div style="font-weight:600;font-size:13px;">Solid color</div>'),
        ],
        gap=1,
        align="center",
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
            gap=1,
            align="center",
        )
        for i in range(_n)
    ]

    mo.vstack([
        mo.md("### Structure controls"),
        _global_row,
        mo.Html("<hr style='margin:6px 0;border:none;border-top:1px solid #e0e0e0;'>"),
        _header,
        *_rows,
    ])
    return (visible_controls, color_mode_controls, color_controls, global_visible, global_color_mode, global_color)


@app.cell(hide_code=True)
def _viewer_3d(
    mo,
    structure_labels,
    structure_pdbs,
    b_ranges,
    BASIC_COLORS,
    visible_controls,
    color_mode_controls,
    color_controls,
    global_visible,
    global_color_mode,
    global_color,
    build_structure_html,
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


if __name__ == "__main__":
    app.run()
