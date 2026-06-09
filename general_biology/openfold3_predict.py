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
def _bootstrap_widget():
    import urllib.request
    from pathlib import Path

    # Bootstrap the widget folder if it's missing (e.g., in a sandboxed MoLab environment)
    widget_dir = Path("widget")
    required_files = ["__init__.py", "af3_helpers.py", "structure_helpers.py", "msa_helpers.py"]
    
    if not widget_dir.exists() or not all((widget_dir / f).exists() for f in required_files):
        widget_dir.mkdir(exist_ok=True)
        base_url = "https://raw.githubusercontent.com/baptiste-roelens/marimo_notebooks/master/widget/"
        for f in required_files:
            try:
                urllib.request.urlretrieve(base_url + f, widget_dir / f)
            except Exception as e:
                # Fallback to main branch
                try:
                    alt_url = "https://raw.githubusercontent.com/baptiste-roelens/marimo_notebooks/main/widget/"
                    urllib.request.urlretrieve(alt_url + f, widget_dir / f)
                except Exception as e2:
                    print(f"Warning: failed to bootstrap widget file {f}: {e2}")
    widget_bootstrapped = True
    return (widget_bootstrapped,)


@app.cell(hide_code=True)
def _imports(widget_bootstrapped):
    import glob
    import json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    from widget.af3_helpers import (
        detect_prediction_source,
        group_prediction_files,
        parse_prediction_json,
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
        Path,
        glob,
        json,
        mo,
        os,
        shutil,
        subprocess,
        sys,
        detect_prediction_source,
        group_prediction_files,
        parse_prediction_json,
        normalize_summary,
        normalize_full_data,
        build_confidence_figure,
        compute_ipsae,
        BASIC_COLORS,
        default_color,
        parse_structure,
        atoms_to_pdb_str,
        superimpose_all,
        get_b_range,
        build_structure_html,
    )


@app.cell(hide_code=True)
def _header(mo):
    mo.md("""
    # OpenFold3 Structure Prediction

    Run macromolecular complex structure prediction locally using
    [OpenFold3](https://github.com/aqlaboratory/openfold-3).
    Model weights, cache directories, and alignment databases should be set up on your machine.

    Results are explored with the same confidence-metrics, PAE/ipSAE and
    multi-model 3D structure viewer as the AlphaFold3 prediction viewer widget —
    so OpenFold3, AlphaFold Server, and Protenix outputs can be compared side by side.

    This notebook declares its own dependencies (PEP 723 inline metadata) —
    launch it in an isolated, notebook-specific environment managed by
    [`uv`](https://docs.astral.sh/uv/) with:

    ```
    uvx marimo edit --sandbox general_biology/openfold3_predict.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _settings_ui(Path, mo):
    openfold_cmd_ui = mo.ui.text(
        value="run_openfold",
        label="OpenFold3 executable (or path to run_openfold)",
        full_width=True,
    )
    output_dir_ui = mo.ui.text(
        value=str(Path(__file__).parent / "output"),
        label="Prediction output directory",
        full_width=True,
    )
    mo.vstack([
        mo.md("## Settings"),
        openfold_cmd_ui,
        output_dir_ui,
    ])
    return openfold_cmd_ui, output_dir_ui


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

    # Proteins
    for _f in protein_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "protein",
                    "sequence": _seq,
                    "chain_ids": [_get_chain_letter(_chain_idx)]
                })
                _chain_idx += 1

    # DNA
    for _f in dna_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "dna",
                    "sequence": _seq,
                    "chain_ids": [_get_chain_letter(_chain_idx)]
                })
                _chain_idx += 1

    # RNA
    for _f in rna_forms_ui.value:
        _seq = "".join(_f["sequence"].split()).upper()
        if _seq:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "rna",
                    "sequence": _seq,
                    "chain_ids": [_get_chain_letter(_chain_idx)]
                })
                _chain_idx += 1

    # Ligands
    for _f in ligand_forms_ui.value:
        _lig = "".join(_f["ligand"].split())
        if _lig:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "ligand",
                    "smiles": _lig,
                    "chain_ids": [_get_chain_letter(_chain_idx)]
                })
                _chain_idx += 1

    # Ions
    for _f in ion_forms_ui.value:
        _ion = "".join(_f["ion"].split()).upper()
        if _ion:
            for _ in range(int(_f["count"])):
                _chains.append({
                    "molecule_type": "ligand",
                    "ccd_codes": [_ion],
                    "chain_ids": [_get_chain_letter(_chain_idx)]
                })
                _chain_idx += 1

    _job_name = job_name_ui.value.strip() or "prediction"
    input_data = {
        "queries": {
            _job_name: {
                "chains": _chains
            }
        }
    }

    _json_text = json.dumps(input_data, indent=2)
    json_preview_ui = mo.ui.code_editor(
        value=_json_text,
        language="json",
        label="Input query JSON preview (updated from fields above)",
    )
    mo.vstack([mo.md("## JSON Preview"), json_preview_ui])
    return input_data, _job_name


@app.cell(hide_code=True)
def _config_ui(mo, _job_name, output_dir_ui):
    import yaml as _yaml

    # Build default runner.yml content
    default_config = {
        "experiment_settings": {
            "mode": "predict",
            "output_dir": output_dir_ui.value,
            "seeds": [101],
            "use_msa_server": True,
            "use_templates": True
        },
        "model_update": {
            "presets": ["predict", "low_mem"]
        },
        "output_writer_settings": {
            "structure_format": "cif",
            "full_confidence_output_format": "json",
            "write_full_confidence_scores": True
        }
    }

    _yaml_text = _yaml.dump(default_config, default_flow_style=False)
    config_editor_ui = mo.ui.code_editor(
        value=_yaml_text,
        language="yaml",
        label="runner.yml Configuration (adjust parameters directly below)",
    )
    mo.vstack([
        mo.md("## Configuration Override"),
        mo.md(
            "Customize the OpenFold3 runner settings below. The notebook will write this "
            "to a temporary YAML file and pass it as the runner configuration."
        ),
        config_editor_ui,
    ])
    return (config_editor_ui,)


@app.cell(hide_code=True)
def _run_section(mo):
    run_btn = mo.ui.run_button(label="▶  Run Prediction")
    mo.vstack([
        mo.md("## Run Inference"),
        mo.callout(
            mo.md(
                "Prediction requires configured checkpoints and parameters. "
                "Ensure your GPU has enough memory to run the model."
            ),
            kind="warn",
        ),
        run_btn,
    ])
    return (run_btn,)


@app.cell(hide_code=True)
def _run_inference(
    mo,
    run_btn,
    openfold_cmd_ui,
    output_dir_ui,
    input_data,
    config_editor_ui,
    _job_name,
    json,
    os,
    subprocess,
    shutil,
):
    mo.stop(not run_btn.value, mo.md("*Click **▶ Run Prediction** to start.*"))

    pred_out_root = output_dir_ui.value
    os.makedirs(pred_out_root, exist_ok=True)

    # Write input.json
    _json_path = os.path.join(pred_out_root, "input_query.json")
    with open(_json_path, "w") as _f:
        json.dump(input_data, _f, indent=2)

    # Write runner.yml
    _yaml_path = os.path.join(pred_out_root, "runner.yml")
    with open(_yaml_path, "w") as _f:
        _f.write(config_editor_ui.value)

    # Find openfold3 command
    _exe = shutil.which(openfold_cmd_ui.value)
    if not _exe:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"Executable `{openfold_cmd_ui.value}` not found. "
                    f"Please verify the path to the executable."
                ),
                kind="danger"
            )
        )

    _cmd = [
        _exe,
        "predict",
        "--query_json", _json_path,
        "--runner_yaml", _yaml_path,
    ]

    with mo.status.spinner("Running OpenFold3 prediction…"):
        _result = subprocess.run(_cmd, text=True, capture_output=True)

    if _result.returncode != 0:
        inference_ok = False
        inference_error = (
            f"Inference failed with exit code {_result.returncode}.\n\n"
            f"**Stdout:**\n```\n{_result.stdout}\n```\n\n"
            f"**Stderr:**\n```\n{_result.stderr}\n```"
        )
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
            
            # Find all predicted structures (mmCIF or PDB)
            _cifs = sorted(glob.glob(os.path.join(_sd, "*_model.cif")) + glob.glob(os.path.join(_sd, "*_model.pdb")))
            for _cif in _cifs:
                _stem = os.path.splitext(os.path.basename(_cif))[0]
                
                # Match index
                import re
                _m = re.search(r"sample_(\d+)", _stem)
                if not _m:
                    continue
                _sample_idx = int(_m.group(1))

                _prefix = _cif.replace("_model.cif", "").replace("_model.pdb", "")
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

                with open(_cif, "rb") as _cf:
                    _atoms = parse_structure(_cf.read(), os.path.basename(_cif))

                # Populate token_chain_ids using the _atoms structure
                if _full_data is not None:
                    _full_data = normalize_full_data(_full_data, "openfold3", _atoms)

                _index = f"Seed {_seed_str} Sample {_sample_idx}"

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
            "matches the true structure — values range from 0 to 1, higher is better.\n\n"
            "**ipTM** (interface pTM) applies the same idea to *inter-chain interfaces* — "
            "above ~0.8 indicates a confident interface. Models are ranked by the combined "
            "**ranking score**."
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
            mo.md("Confidence files containing PAE matrices are missing or full confidence scores were not written."),
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
        "indicate well-resolved chains/domains."
    )
    if _has_chain_matrix:
        _intro += (
            "\n\nFor assemblies with more than two chains, the **pairwise ipTM matrix** "
            "alongside the PAE plot shows the predicted interface accuracy for every chain "
            "pair."
        )

    mo.vstack([mo.md(_intro), *_figs])
    return


@app.cell(hide_code=True)
def _ipsae_intro(mo, prediction_models):
    _has_full_data = any(_m["full_data"] is not None for _m in prediction_models)
    mo.stop(
        not _has_full_data,
        mo.callout(mo.md("Full data (PAE matrices) are needed to compute ipSAE."), kind="info"),
    )

    ipsae_cutoff = mo.ui.slider(
        5, 30, value=10, step=1, label="PAE cutoff for ipSAE (Å)", show_value=True
    )

    mo.vstack([
        mo.md(r"""
        ### ipSAE — interface confidence between chains

        **ipSAE** ([Dunbrack et al.](https://www.biorxiv.org/content/10.1101/2025.02.10.637595))
        is an alternative to ipTM for scoring inter-chain interfaces. It only counts
        residue pairs whose predicted aligned error is below the cutoff, and
        normalises each residue's contribution by how many partner-chain residues are
        confidently placed near it.

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
