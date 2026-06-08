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

    # Make the local Protenix package (and this repo's `widget` helpers) importable
    # even without `pip install`
    _repo_root = str(Path(__file__).parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Reuse the same confidence-figure / ipSAE / structure-viewer building blocks as
    # the AlphaFold3 prediction viewer widget — Protenix's outputs are normalized to
    # the same schema (see `normalize_summary` / `normalize_full_data`) so the exact
    # same visualizations apply to both.
    from widget.af3_helpers import (
        normalize_summary,
        normalize_full_data,
        build_confidence_figure,
        compute_ipsae,
    )
    from widget.structure_helpers import (
        BASIC_COLORS,
        default_color,
        parse_structure,
        atoms_to_pdb_str,
        superimpose_all,
        get_b_range,
        build_structure_html,
    )
    return (
        BASIC_COLORS,
        Path,
        atoms_to_pdb_str,
        build_confidence_figure,
        build_structure_html,
        compute_ipsae,
        default_color,
        get_b_range,
        glob,
        json,
        mo,
        normalize_full_data,
        normalize_summary,
        os,
        parse_structure,
        shutil,
        subprocess,
        superimpose_all,
        sys,
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
    ]

    _env = dict(os.environ, PROTENIX_ROOT_DIR=weights_dir_ui.value)

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
