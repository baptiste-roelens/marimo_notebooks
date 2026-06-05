import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="Protein Structure Viewer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
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
        build_structure_html,
        default_color,
        get_b_range,
        parse_structure,
        superimpose_all,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md("# Protein Structure Viewer"))
    return


@app.cell(hide_code=True)
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".pdb", ".cif", ".mmcif"],
        multiple=True,
        label="Upload structure files (PDB / CIF)",
    )
    mo.vstack([
        mo.md(
            "Upload two or more structures. The **first file** is used as the "
            "superimposition reference."
        ),
        file_upload,
    ])
    return (file_upload,)


@app.cell(hide_code=True)
def _(mo, file_upload, parse_structure, superimpose_all, atoms_to_pdb_str, get_b_range):
    import pandas as _pd

    mo.stop(
        not file_upload.value,
        mo.callout(mo.md("Upload at least one structure file above."), kind="info"),
    )

    _files = file_upload.value
    _names = [f.name for f in _files]

    with mo.status.spinner("Parsing structures…"):
        try:
            _atoms_list = [parse_structure(f.contents, f.name) for f in _files]
        except Exception as _e:
            mo.stop(True, mo.callout(mo.md(f"Failed to parse structure: `{_e}`"), kind="danger"))

    with mo.status.spinner("Superimposing onto reference…"):
        _aligned, _rmsds = superimpose_all(_atoms_list)

    structure_names = _names
    structure_pdbs = [atoms_to_pdb_str(a) for a in _aligned]
    b_ranges = [get_b_range(a) for a in _aligned]

    # Build RMSD summary table
    _rmsd_rows = [{"Structure": _names[0], "Role": "Reference (RMSD = 0.00 Å)", "Cα atoms": len(_aligned[0][((_aligned[0].atom_name == "CA") & ~_aligned[0].hetero)])}]
    for _i, (_name, _rmsd) in enumerate(zip(_names[1:], _rmsds), start=1):
        _ca = _aligned[_i][(_aligned[_i].atom_name == "CA") & ~_aligned[_i].hetero]
        _rmsd_rows.append({
            "Structure": _name,
            "Role": f"RMSD = {_rmsd:.2f} Å" if _rmsd == _rmsd else "RMSD = N/A",
            "Cα atoms": len(_ca),
        })
    _rmsd_df = _pd.DataFrame(_rmsd_rows)

    mo.vstack([
        mo.callout(
            mo.md(
                f"**{len(structure_names)}** structure(s) loaded. "
                f"Reference: **{structure_names[0]}**"
            ),
            kind="success",
        ),
        mo.ui.table(_rmsd_df, selection=None),
    ])
    return (structure_names, structure_pdbs, b_ranges)


@app.cell(hide_code=True)
def _(mo, structure_names, BASIC_COLORS, default_color):
    _n = len(structure_names)
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
            mo.Html('<div style="min-width:220px;max-width:220px;font-weight:600;font-size:13px;">Structure</div>'),
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
                    f'font-family:monospace;font-size:13px;" title="{structure_names[i]}">'
                    f'{structure_names[i]}</div>'
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
    structure_names,
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
    _n = len(structure_names)

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
            labels=structure_names,
            visibilities=_visibilities,
            color_modes=_color_modes,
            colors=_hex_colors,
            b_ranges=b_ranges,
        )
    )
    return


if __name__ == "__main__":
    app.run()
