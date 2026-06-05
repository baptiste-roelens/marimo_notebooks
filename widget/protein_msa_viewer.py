import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="Protein MSA Viewer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import matplotlib
    matplotlib.use("Agg")
    from widget.msa_helpers import (
        PYMSAVIZ_SCHEMES,
        detect_format,
        parse_a3m,
        build_msaviz_figure,
    )
    return (PYMSAVIZ_SCHEMES, build_msaviz_figure, detect_format, parse_a3m)


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md("# Protein MSA Viewer"))
    return


@app.cell(hide_code=True)
def _(mo, PYMSAVIZ_SCHEMES):
    file_upload = mo.ui.file(
        filetypes=[".fasta", ".fa", ".a3m"],
        label="Upload MSA file",
    )
    text_input = mo.ui.text_area(
        placeholder="…or paste FASTA / A3M alignment text here…",
        rows=10,
        full_width=True,
    )
    format_selector = mo.ui.dropdown(
        ["Auto-detect", "FASTA", "A3M"],
        value="Auto-detect",
        label="Format",
    )
    color_scheme = mo.ui.dropdown(
        PYMSAVIZ_SCHEMES,
        value="Clustal",
        label="Color scheme",
    )
    wrap_length = mo.ui.slider(
        40, 200,
        value=80,
        step=10,
        label="Wrap length",
        show_value=True,
    )
    show_consensus = mo.ui.checkbox(True, label="Show consensus")
    show_counts = mo.ui.checkbox(True, label="Show position counts")

    mo.vstack([
        mo.md("### Input"),
        mo.hstack([file_upload, format_selector], align="end", gap=1),
        text_input,
        mo.md("### Display options"),
        mo.hstack([color_scheme, wrap_length, show_consensus, show_counts], gap=2),
    ])
    return (
        file_upload,
        text_input,
        format_selector,
        color_scheme,
        wrap_length,
        show_consensus,
        show_counts,
    )


@app.cell(hide_code=True)
def _(mo, file_upload, text_input, format_selector, detect_format, parse_a3m):
    from io import StringIO
    from Bio import AlignIO

    if file_upload.value:
        _raw = file_upload.value[0].contents.decode("utf-8", errors="replace")
    else:
        _raw = text_input.value.strip()

    mo.stop(
        not _raw,
        mo.callout(mo.md("Upload a file or paste an MSA above to begin."), kind="info"),
    )

    _fmt_choice = format_selector.value
    _fmt = detect_format(_raw) if _fmt_choice == "Auto-detect" else _fmt_choice

    try:
        if _fmt == "A3M":
            alignment = parse_a3m(_raw)
        else:
            alignment = AlignIO.read(StringIO(_raw), "fasta")
    except Exception as _e:
        mo.stop(
            True,
            mo.callout(mo.md(f"Could not parse alignment: `{_e}`"), kind="danger"),
        )

    _n_seqs = len(alignment)
    _aln_len = alignment.get_alignment_length()
    mo.callout(
        mo.md(
            f"**{_n_seqs}** sequences · **{_aln_len}** alignment columns · "
            f"Format: **{_fmt}**"
        ),
        kind="success",
    )
    return (alignment,)


@app.cell(hide_code=True)
def _(mo, alignment, color_scheme, wrap_length, show_consensus, show_counts, build_msaviz_figure):
    with mo.status.spinner("Rendering alignment…"):
        _fig = build_msaviz_figure(
            alignment,
            color_scheme=color_scheme.value,
            wrap_length=wrap_length.value,
            show_consensus=show_consensus.value,
            show_counts=show_counts.value,
        )
    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
