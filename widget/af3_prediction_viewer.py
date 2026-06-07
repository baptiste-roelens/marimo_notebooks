import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="AlphaFold3 Prediction Viewer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from widget.af3_helpers import group_af3_files, parse_af3_json, build_confidence_figure
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
        atoms_to_pdb_str,
        build_confidence_figure,
        build_structure_html,
        default_color,
        get_b_range,
        group_af3_files,
        parse_af3_json,
        parse_structure,
        superimpose_all,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
    # AlphaFold3 Prediction Viewer

    Explore the output of an **AlphaFold Server / AlphaFold3** folding job: confidence
    metrics (pTM, ipTM), per-model predicted aligned error (PAE), and the predicted
    3D structures themselves.

    Upload the job's output files below — model structures (`..._model_N.cif`),
    confidence summaries (`..._summary_confidences_N.json`) and, optionally, the
    detailed `..._full_data_N.json` files (needed for the PAE plots).
    """))
    return


@app.cell(hide_code=True)
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".cif", ".mmcif", ".json"],
        multiple=True,
        label="Upload AlphaFold3 output files",
    )
    mo.vstack([
        mo.md(
            "Select all `model_*.cif`, `summary_confidences_*.json` and "
            "`full_data_*.json` files for a single folding job."
        ),
        file_upload,
    ])
    return (file_upload,)


@app.cell(hide_code=True)
def _(file_upload, group_af3_files, mo, parse_af3_json, parse_structure):
    mo.stop(
        not file_upload.value,
        mo.callout(mo.md("Upload AlphaFold3 prediction output files above."), kind="info"),
    )

    _files = file_upload.value
    _groups = group_af3_files(_files)
    mo.stop(
        not _groups,
        mo.callout(
            mo.md(
                "No recognizable AlphaFold3 output files found. Expected names following "
                "the AlphaFold Server convention, e.g. `fold_x_model_0.cif`, "
                "`fold_x_summary_confidences_0.json`, `fold_x_full_data_0.json`."
            ),
            kind="warn",
        ),
    )

    with mo.status.spinner("Parsing predicted models…"):
        _parsed = []
        for _idx, _grp in sorted(_groups.items()):
            try:
                _summary = parse_af3_json(_grp["summary"].contents)
                _atoms = parse_structure(_grp["model"].contents, _grp["model"].name)
                _full_data = (
                    parse_af3_json(_grp["full_data"].contents)
                    if "full_data" in _grp else None
                )
            except Exception as _e:
                mo.output.append(
                    mo.callout(mo.md(f"Could not parse model {_idx}: `{_e}`"), kind="warn")
                )
                continue
            _parsed.append({
                "index": _idx,
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

    af3_models = _parsed
    mo.callout(
        mo.md(f"Parsed **{len(af3_models)}** predicted model(s)."),
        kind="success",
    )
    return (af3_models,)


@app.cell(hide_code=True)
def _(af3_models, mo):
    import pandas as _pd

    _rows = []
    for _m in af3_models:
        _s = _m["summary"]
        _rows.append({
            "Rank": _m["rank"],
            "Model": _m["index"],
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
            "Models are ranked by AlphaFold3's combined **ranking score**."
        ),
        mo.ui.table(_df, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(af3_models, build_confidence_figure, mo):
    _figs = [
        mo.as_html(build_confidence_figure(_m["index"], _m["summary"], _m["full_data"], rank=_m["rank"]))
        for _m in af3_models
        if _m["full_data"] is not None
    ]

    mo.stop(
        not _figs,
        mo.callout(
            mo.md("Upload the `..._full_data_N.json` files to see PAE matrices."),
            kind="info",
        ),
    )

    _has_chain_matrix = any(
        _m["full_data"] is not None
        and _m["summary"].get("chain_pair_iptm") is not None
        and len(_m["summary"]["chain_pair_iptm"]) > 2
        for _m in af3_models
    )

    _intro = (
        "### Predicted aligned error (PAE)\n\n"
        "Cell *(i, j)* shows AlphaFold3's expected position error (in Å) for residue *j* "
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
def _(af3_models, atoms_to_pdb_str, get_b_range, mo, superimpose_all):
    import pandas as _pd

    _atoms_list = [m["atoms"] for m in af3_models]
    _labels = [f"Model {m['index']} (rank {m['rank']})" for m in af3_models]

    with mo.status.spinner("Superimposing predicted models on Cα atoms…"):
        _aligned, _rmsds = superimpose_all(_atoms_list)

    structure_labels = _labels
    structure_pdbs = [atoms_to_pdb_str(a) for a in _aligned]
    b_ranges = [get_b_range(a) for a in _aligned]

    _rows = [{"Model": _labels[0], "RMSD vs top-ranked (Å)": "0.00 (reference)"}]
    for _label, _rmsd in zip(_labels[1:], _rmsds):
        _rows.append({
            "Model": _label,
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
def _(mo, structure_labels, BASIC_COLORS, default_color):
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
            mo.Html('<div style="min-width:220px;max-width:220px;font-weight:600;font-size:13px;">Model</div>'),
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
def _(
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
