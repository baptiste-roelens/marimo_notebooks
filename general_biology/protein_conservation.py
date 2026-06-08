# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9",
#     "biopython>=1.83",
#     "requests>=2.31",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "pymsaviz>=0.5",
#     "biotite>=0.39",
#     "py3dmol>=2.0",
#     "marimo-bio-widget-helpers",
# ]
#
# [tool.uv.sources]
# marimo-bio-widget-helpers = { git = "https://github.com/baptiste-roelens/marimo_notebooks", subdirectory = "widget" }
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="Protein Conservation Viewer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import time
    import functools

    import requests
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")

    from widget.msa_helpers import (
        PYMSAVIZ_SCHEMES,
        build_msaviz_figure,
    )
    from widget.structure_helpers import (
        BASIC_COLORS,
        default_color,
        parse_pdb_str,
        atoms_to_pdb_str,
        superimpose_all,
        compute_tm_score,
        get_b_range,
        build_structure_html,
    )

    # ── UniProt ───────────────────────────────────────────────────────────────

    UNIPROT_BASE  = "https://rest.uniprot.org/uniprotkb"
    AFDB_BASE     = "https://alphafold.ebi.ac.uk/api/prediction"
    CLUSTALO_BASE = "https://www.ebi.ac.uk/Tools/services/rest/clustalo"

    HEADERS = {"User-Agent": "ProteinConservationViewer/1.0 (contact: user@example.com)"}

    def _parse_entry(entry: dict) -> dict:
        gene = ""
        genes = entry.get("genes", [])
        if genes:
            gn = genes[0].get("geneName", {})
            gene = gn.get("value", "")

        organism = entry.get("organism", {})
        sci_name  = organism.get("scientificName", "")
        common    = organism.get("commonName", "")
        taxon_id  = organism.get("taxonId", "")
        org_label = f"{sci_name} ({common})" if common else sci_name

        desc = entry.get("proteinDescription", {})
        rec  = desc.get("recommendedName", {})
        full = rec.get("fullName", {}).get("value", "")
        if not full:
            sub = desc.get("submissionNames", [])
            full = sub[0].get("fullName", {}).get("value", "") if sub else ""

        seq    = entry.get("sequence", {})
        length = seq.get("length", 0)

        return {
            "accession":   entry.get("primaryAccession", ""),
            "gene":        gene,
            "organism":    org_label,
            "taxon_id":    str(taxon_id),
            "length":      length,
            "description": full,
        }

    def search_uniprot(query: str, size: int = 10) -> pd.DataFrame:
        params = {
            "query":  f"({query}) AND reviewed:true",
            "format": "json",
            "size":   size,
            "fields": "accession,gene_names,organism_name,organism_id,length,protein_name",
        }
        r = requests.get(f"{UNIPROT_BASE}/search", params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return pd.DataFrame(columns=["accession","gene","organism","taxon_id","length","description"])
        return pd.DataFrame([_parse_entry(e) for e in results])

    def get_orthologs_by_gene(gene_name: str, size: int = 100) -> pd.DataFrame:
        params = {
            "query":  f"gene_exact:{gene_name} AND reviewed:true",
            "format": "json",
            "size":   size,
            "fields": "accession,gene_names,organism_name,organism_id,length,protein_name",
        }
        r = requests.get(f"{UNIPROT_BASE}/search", params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        rows = [_parse_entry(e) for e in r.json().get("results", [])]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["accession","gene","organism","taxon_id","length","description"]
        )
        if not df.empty:
            df["has_afdb"] = df["accession"].apply(check_afdb)
        return df

    def get_fasta(accession: str) -> str:
        r = requests.get(f"{UNIPROT_BASE}/{accession}.fasta", headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text.strip()

    # ── AFDB ──────────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=512)
    def check_afdb(accession: str) -> bool:
        try:
            r = requests.get(f"{AFDB_BASE}/{accession}", headers=HEADERS, timeout=10)
            return r.status_code == 200 and len(r.json()) > 0
        except Exception:
            return False

    def fetch_pdb(accession: str) -> str | None:
        try:
            r = requests.get(f"{AFDB_BASE}/{accession}", headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            pdb_url = data[0].get("pdbUrl", "")
            if not pdb_url:
                return None
            rp = requests.get(pdb_url, headers=HEADERS, timeout=30)
            rp.raise_for_status()
            return rp.text
        except Exception:
            return None

    # ── EBI Clustal Omega ─────────────────────────────────────────────────────

    def run_clustalo(fasta_block: str, email: str = "user@example.com") -> str:
        payload = {
            "email":    email,
            "sequence": fasta_block,
            "outfmt":   "fa",
            "stype":    "protein",
        }
        r = requests.post(f"{CLUSTALO_BASE}/run", data=payload, headers=HEADERS, timeout=30)
        r.raise_for_status()
        job_id = r.text.strip()

        for _ in range(120):
            time.sleep(3)
            sr = requests.get(f"{CLUSTALO_BASE}/status/{job_id}", headers=HEADERS, timeout=10)
            sr.raise_for_status()
            status = sr.text.strip()
            if status == "FINISHED":
                break
            if status in ("FAILED", "ERROR", "DELETED"):
                raise RuntimeError(f"Clustal Omega job {job_id} ended with status: {status}")
        else:
            raise TimeoutError("Clustal Omega job did not finish within 6 minutes")

        rr = requests.get(
            f"{CLUSTALO_BASE}/result/{job_id}/aln-fasta",
            headers=HEADERS,
            timeout=15,
        )
        rr.raise_for_status()
        return rr.text.strip()

    return (
        BASIC_COLORS,
        PYMSAVIZ_SCHEMES,
        atoms_to_pdb_str,
        build_msaviz_figure,
        build_structure_html,
        compute_tm_score,
        default_color,
        fetch_pdb,
        get_b_range,
        get_fasta,
        get_orthologs_by_gene,
        parse_pdb_str,
        run_clustalo,
        search_uniprot,
        superimpose_all,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Protein Conservation Explorer

    Proteins evolve under constant mutational pressure, yet many retain their function for
    hundreds of millions of years. Natural selection preserves what matters most: the ability
    to fold and to carry out a specific biochemical role.

    **The core insight this notebook illustrates:** *structure is more conserved than sequence.*
    At the sequence level, orthologs can diverge to below 20% identity over long evolutionary
    timescales while still performing identical functions. At the structural level, the 3D fold
    is far more constrained — homologous proteins from distantly related organisms often
    superimpose with Cα RMSD values well below 2 Å even when sequence identity alone would
    not predict any relationship.

    This notebook lets you explore both levels of conservation side by side for any protein
    of your choice, using data from **UniProt**, **Clustal Omega** (EBI), and the
    **AlphaFold Database**.

    ---

    **Workflow**

    1. **Search** for a protein in UniProt Swiss-Prot and pick a reference entry
    2. **Select orthologs** — the same gene across different organisms
    3. **Align sequences** with Clustal Omega and explore residue-level conservation
    4. **Fetch and superimpose** AlphaFold predicted structures; compare RMSD values
    5. **Visualize** a neighbour-joining phylogenetic tree built from sequence identity

    > **Good proteins to try:** *actin*, *histone H3*, *PCNA* (near-universal conservation);
    > *cytochrome c*, *ubiquitin* (extreme sequence conservation across eukaryotes);
    > *globins* (moderate sequence divergence, highly conserved fold — a textbook example
    > of sequence vs. structure conservation).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Step 1 — Search for a protein")
    return


@app.cell(hide_code=True)
def _(mo):
    protein_name = mo.ui.text(
        placeholder="e.g. Insulin, hemoglobin, Actin …",
        label="Protein name",
        full_width=True,
    )
    search_btn = mo.ui.run_button(label="Search UniProt", kind="success")
    mo.output.replace(mo.hstack([protein_name, search_btn], align="end", gap=1))
    return protein_name, search_btn


@app.cell(hide_code=True)
def _(mo, protein_name, search_btn, search_uniprot):
    mo.stop(
        not search_btn.value,
        mo.callout(mo.md("Enter a protein name and click **Search UniProt**."), kind="info"),
    )
    mo.stop(
        not protein_name.value.strip(),
        mo.callout(mo.md("Please enter a protein name first."), kind="warn"),
    )
    with mo.status.spinner("Searching UniProt…"):
        _search_df = search_uniprot(protein_name.value.strip())
    mo.stop(
        _search_df.empty,
        mo.callout(mo.md(f"No Swiss-Prot results for **{protein_name.value}**."), kind="warn"),
    )
    search_table = mo.ui.table(
        _search_df,
        selection="single",
        label="Select the reference protein (one row)",
        pagination=True,
    )
    search_table
    return (search_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Select orthologs to compare

    The table below lists all **reviewed Swiss-Prot entries** carrying the same gene name,
    each from a different organism. These are the orthologs — proteins that descend from a
    common ancestral gene and perform the same biological role.

    Select **≥ 2 entries** (checkboxes) to include in the comparison. The **AF structure**
    column (✓/✗) indicates whether an AlphaFold prediction is available; include entries
    with ✓ if you want to compare 3D structures downstream.
    """)
    return


@app.cell(hide_code=True)
def _(get_orthologs_by_gene, mo, search_table):
    mo.stop(
        search_table.value is None or len(search_table.value) == 0,
        mo.callout(mo.md("Select a reference protein in the table above."), kind="info"),
    )
    ref_entry = search_table.value.iloc[0]
    ref_gene  = ref_entry["gene"]
    ref_acc   = ref_entry["accession"]
    mo.stop(
        not ref_gene,
        mo.callout(mo.md("The selected entry has no gene name — cannot search for orthologs."), kind="warn"),
    )
    with mo.status.spinner(f"Searching orthologs of **{ref_gene}** across species…"):
        orthologs_df = get_orthologs_by_gene(ref_gene)
    mo.stop(
        orthologs_df.empty,
        mo.callout(mo.md(f"No reviewed orthologs found for gene **{ref_gene}**."), kind="warn"),
    )
    display_df = orthologs_df.copy()
    display_df["has_afdb"] = display_df["has_afdb"].map({True: "✓", False: "✗"})
    display_df = display_df.rename(columns={
        "accession":   "Accession",
        "gene":        "Gene",
        "organism":    "Organism",
        "taxon_id":    "Taxon ID",
        "length":      "Length (aa)",
        "description": "Protein name",
        "has_afdb":    "AF structure",
    })
    return display_df, orthologs_df, ref_acc, ref_gene


@app.cell(hide_code=True)
def _(display_df, mo, orthologs_df, ref_acc, ref_gene):
    _n   = len(orthologs_df)
    _acc = ref_acc
    _gen = ref_gene
    ortholog_table = mo.ui.table(
        display_df,
        selection="multi",
        label=(
            f"Found **{_n}** reviewed entries for *{_gen}* "
            f"(ref: **{_acc}**) — select ≥ 2 to align:"
        ),
        pagination=True,
        page_size=20,
    )
    ortholog_table
    return (ortholog_table,)


@app.cell(hide_code=True)
def _(mo, ortholog_table):
    _n = len(ortholog_table.value)
    _status = (
        mo.callout(mo.md("Select at least **2 orthologs** in the table above."), kind="info")
        if _n == 0 else
        mo.callout(mo.md("Select **one more** ortholog to enable the alignment."), kind="warn")
        if _n == 1 else
        mo.callout(mo.md(f"**{_n} orthologs** selected. Scroll down to compute the alignment."), kind="success")
    )
    _status
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3 — Sequence conservation

    Sequences are aligned using **Clustal Omega** (EBI REST service), a fast progressive
    multiple aligner. Each column in the resulting alignment corresponds to an equivalent
    position across all orthologs, and conservation at each column reflects evolutionary
    constraint.

    After the alignment loads, try the **% Conservation** color scheme — columns shaded in
    deep blue are invariant (or nearly so) across all species, while pale columns vary freely.
    Highly conserved columns often correspond to the protein's structural core, active site,
    or key binding interface.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    align_btn = mo.ui.run_button(label="Compute alignment (Clustal Omega)", kind="success")
    align_btn
    return (align_btn,)


@app.cell(hide_code=True)
def _(align_btn, get_fasta, mo, ortholog_table, run_clustalo):
    mo.stop(
        not align_btn.value,
        mo.callout(mo.md("Click **Compute alignment** to run Clustal Omega."), kind="info"),
    )
    _sel = ortholog_table.value
    mo.stop(
        _sel is None or len(_sel) < 2,
        mo.callout(mo.md("Select ≥ 2 orthologs first."), kind="warn"),
    )
    _sel_accs = _sel["Accession"].tolist()

    with mo.status.spinner("Fetching sequences from UniProt…"):
        _fastas = []
        for _acc in _sel_accs:
            try:
                _fastas.append(get_fasta(_acc))
            except Exception as _e:
                mo.output.append(
                    mo.callout(mo.md(f"Could not fetch sequence for {_acc}: {_e}"), kind="warn")
                )

    mo.stop(
        len(_fastas) < 2,
        mo.callout(mo.md("Fewer than 2 sequences retrieved — cannot align."), kind="warn"),
    )

    with mo.status.spinner("Running Clustal Omega (EBI) — this may take 10–60 s…"):
        try:
            aligned_fasta = run_clustalo("\n".join(_fastas))
        except Exception as _err:
            mo.stop(True, mo.callout(mo.md(f"Clustal Omega failed: {_err}"), kind="danger"))

    mo.callout(
        mo.md(f"Alignment complete — **{len(_sel_accs)} sequences**."),
        kind="success",
    )
    return (aligned_fasta,)


@app.cell(hide_code=True)
def _(mo, aligned_fasta, PYMSAVIZ_SCHEMES):
    mo.stop(
        not aligned_fasta,
        mo.callout(mo.md("Compute alignment first."), kind="info"),
    )
    msa_color_scheme = mo.ui.dropdown(
        PYMSAVIZ_SCHEMES, value="Clustal", label="Color scheme"
    )
    msa_wrap_length = mo.ui.slider(
        40, 200, value=80, step=10, label="Wrap length", show_value=True
    )
    msa_show_consensus = mo.ui.checkbox(True, label="Show consensus")
    msa_show_counts = mo.ui.checkbox(True, label="Show counts")
    mo.hstack([msa_color_scheme, msa_wrap_length, msa_show_consensus, msa_show_counts], gap=2)
    return msa_color_scheme, msa_wrap_length, msa_show_consensus, msa_show_counts


@app.cell(hide_code=True)
def _(
    mo,
    aligned_fasta,
    msa_color_scheme,
    msa_wrap_length,
    msa_show_consensus,
    msa_show_counts,
    build_msaviz_figure,
):
    from Bio import AlignIO as _AlignIO
    from io import StringIO as _StringIO

    mo.stop(not aligned_fasta, mo.callout(mo.md("Compute alignment first."), kind="info"))
    _alignment = _AlignIO.read(_StringIO(aligned_fasta), "fasta")

    with mo.status.spinner("Rendering alignment…"):
        _fig = build_msaviz_figure(
            _alignment,
            color_scheme=msa_color_scheme.value,
            wrap_length=msa_wrap_length.value,
            show_consensus=msa_show_consensus.value,
            show_counts=msa_show_counts.value,
        )
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4 — Structural conservation

    Sequences diverge far faster than folds. Below, **AlphaFold Database (AFDB)** predicted
    structures are fetched for the selected orthologs and superimposed onto the first entry
    (reference) using **Cα atoms only**. The table reports two complementary similarity
    measures:

    - **RMSD** (root-mean-square deviation of Cα positions after superimposition) is the
      classic, intuitive measure — but it is dominated by the worst-aligned residues (e.g. a
      single flexible loop can inflate it) and is not directly comparable across proteins of
      different lengths. Treat it as illustrative.
    - **TM-score** is normalised by chain length and saturates with distance, so a few poorly
      aligned residues barely move it — it tracks *overall fold* similarity much more robustly
      and is comparable across different proteins. It ranges from 0 to 1.

    | RMSD | Interpretation |
    |---|---|
    | < 2 Å | Highly similar fold |
    | 2–4 Å | Similar fold, local variations (loops, termini, flexible regions) |
    | > 4 Å | Substantial structural differences |

    | TM-score | Interpretation |
    |---|---|
    | > 0.5 | (Roughly) the same fold |
    | 0.2–0.5 | Possibly related fold |
    | < 0.2 | Likely unrelated structures |

    Compare these values with the sequence identities visible in the alignment above.
    Even orthologs with strikingly low sequence identity often superimpose with RMSD a few Å
    and TM-score well above 0.5 — a concrete demonstration that **structure is more conserved
    than sequence**.

    In the viewer, switching **Color mode → B-factor** displays AlphaFold's per-residue
    confidence score (pLDDT): blue/green = high confidence, red = disordered or uncertain.
    """)
    return


@app.cell(hide_code=True)
def _(
    atoms_to_pdb_str,
    compute_tm_score,
    fetch_pdb,
    get_b_range,
    mo,
    ortholog_table,
    orthologs_df,
    parse_pdb_str,
    superimpose_all,
):
    import pandas as _pd

    _sel = ortholog_table.value
    mo.stop(
        _sel is None or len(_sel) == 0,
        mo.callout(mo.md("Select orthologs in the table above first."), kind="info"),
    )
    _sel_accs = _sel["Accession"].tolist()
    _af_rows  = orthologs_df[
        orthologs_df["accession"].isin(_sel_accs) & orthologs_df["has_afdb"]
    ].reset_index(drop=True)

    mo.stop(
        _af_rows.empty,
        mo.callout(
            mo.md("None of the selected orthologs have AFDB structures. "
                  "Try selecting entries with **✓** in the *AF structure* column."),
            kind="warn",
        ),
    )

    with mo.status.spinner(f"Fetching {len(_af_rows)} structures from AFDB…"):
        _pdbs_raw  = []
        struct_labels = []
        for _, _row in _af_rows.iterrows():
            _pdb = fetch_pdb(_row["accession"])
            if _pdb:
                _pdbs_raw.append(_pdb)
                _org = _row["organism"].split("(")[0].strip()
                struct_labels.append(f"{_row['gene']} | {_org}")

    mo.stop(
        len(_pdbs_raw) == 0,
        mo.callout(mo.md("Could not download any PDB files from AFDB."), kind="danger"),
    )

    with mo.status.spinner("Superimposing structures on Cα atoms…"):
        _atoms_list    = [parse_pdb_str(p) for p in _pdbs_raw]
        _aligned_atoms, _rmsds = superimpose_all(_atoms_list)
        struct_pdbs    = [atoms_to_pdb_str(a) for a in _aligned_atoms]
        b_ranges       = [get_b_range(a) for a in _aligned_atoms]
        _tm_scores     = [
            compute_tm_score(_aligned_atoms[0], _mobile)
            for _mobile in _aligned_atoms[1:]
        ]

    _rmsd_df = _pd.DataFrame({
        "Gene | Organism":  struct_labels,
        "RMSD vs ref (Å)": [
            round(v, 2) if v == v else "N/A"
            for v in [0.0] + _rmsds
        ],
        "TM-score vs ref": [
            round(v, 2) if v == v else "N/A"
            for v in [1.0] + _tm_scores
        ],
    })
    mo.vstack([
        mo.callout(
            mo.md(
                f"**{len(struct_pdbs)}** structure(s) superimposed. "
                f"Reference: **{struct_labels[0]}**"
            ),
            kind="success",
        ),
        mo.ui.table(_rmsd_df, selection=None),
    ])
    return b_ranges, struct_labels, struct_pdbs


@app.cell(hide_code=True)
def _(mo, struct_labels, BASIC_COLORS, default_color):
    _n = len(struct_labels)
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
        [mo.ui.dropdown(["Solid color", "B-factor"], value="Solid color")
         for _ in range(_n)]
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
                    f'font-family:monospace;font-size:13px;" title="{struct_labels[i]}">'
                    f'{struct_labels[i]}</div>'
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
    return color_controls, color_mode_controls, visible_controls, global_visible, global_color_mode, global_color


@app.cell(hide_code=True)
def _(
    BASIC_COLORS,
    b_ranges,
    build_structure_html,
    color_controls,
    color_mode_controls,
    global_color,
    global_color_mode,
    global_visible,
    mo,
    struct_labels,
    struct_pdbs,
    visible_controls,
):
    _n = len(struct_labels)

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
            pdb_contents=struct_pdbs,
            labels=struct_labels,
            visibilities=_visibilities,
            color_modes=_color_modes,
            colors=_hex_colors,
            b_ranges=b_ranges,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 5 — Phylogenetic context

    The tree below is reconstructed from the aligned sequences using the **neighbour-joining
    algorithm** with pairwise identity distances. It provides a rough picture of the
    evolutionary relationships among the selected orthologs based on sequence alone.

    Branch lengths reflect sequence divergence — long branches indicate rapidly evolving
    lineages, short branches indicate more conserved ones. Compare the tree topology with
    known species phylogenies: concordance suggests vertical inheritance, while unexpected
    groupings can hint at lineage-specific evolutionary pressures or, rarely, horizontal
    gene transfer.
    """)
    return


@app.cell(hide_code=True)
def _(aligned_fasta, mo):
    from io import StringIO
    from Bio import AlignIO as _AlignIO
    from Bio import Phylo as _Phylo
    from Bio.Phylo.TreeConstruction import DistanceCalculator as _DistCalc
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor as _DistCons
    import matplotlib.pyplot as _plt

    _msa = _AlignIO.read(StringIO(aligned_fasta), "fasta")
    _n   = len(_msa)

    with mo.status.spinner("Building neighbour-joining tree…"):
        _calc        = _DistCalc("identity")
        _dm          = _calc.get_distance(_msa)
        _constructor = _DistCons(_calc, "nj")
        _tree        = _constructor.build_tree(_msa)
        _tree.root_at_midpoint()

    _fig, _ax = _plt.subplots(figsize=(12, max(4, _n * 0.45 + 1.5)))
    _Phylo.draw(_tree, axes=_ax, do_show=False)
    _ax.set_xlabel("Branch length (identity distance)")
    _ax.set_title("Neighbour-joining tree from pairwise identity distances", fontsize=11)
    _fig.tight_layout()

    mo.vstack([
        mo.callout(mo.md(f"Phylogenetic tree built from **{_n} sequences**."), kind="success"),
        mo.as_html(_fig),
    ])
    return


if __name__ == "__main__":
    app.run()
