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
    import textwrap
    import functools
    from io import StringIO

    import requests
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import biotite.structure as struc
    import biotite.structure.io.pdb as bpdb

    UNIPROT_BASE  = "https://rest.uniprot.org/uniprotkb"
    AFDB_BASE     = "https://alphafold.ebi.ac.uk/api/prediction"
    CLUSTALO_BASE = "https://www.ebi.ac.uk/Tools/services/rest/clustalo"

    HEADERS = {"User-Agent": "ProteinConservationViewer/1.0 (contact: user@example.com)"}

    # ── UniProt ───────────────────────────────────────────────────────────────

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
        """Free-text search on Swiss-Prot; returns top results."""
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
        """Return reviewed UniProt entries sharing the same gene name, across all species."""
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
        """Download FASTA for a single UniProt accession."""
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
        """Download PDB-format structure from AFDB, or None if unavailable."""
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
        """
        Submit sequences to EBI Clustal Omega, poll until done, return aligned FASTA.
        Blocking — call inside mo.status.spinner().
        """
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

    # ── Structure superimposition ─────────────────────────────────────────────

    def _pdb_str_to_atoms(pdb_text: str):
        pf = bpdb.PDBFile.read(StringIO(pdb_text))
        return bpdb.get_structure(pf, model=1)

    def _atoms_to_pdb_str(atoms) -> str:
        pf = bpdb.PDBFile()
        bpdb.set_structure(pf, atoms)
        buf = StringIO()
        pf.write(buf)
        return buf.getvalue()

    def superimpose_all(pdb_contents: list[str]) -> tuple[list[str], list[float]]:
        """
        Superimpose all structures onto the first using Cα atoms.
        Returns (transformed_pdb_strings, rmsds_vs_reference).
        """
        structures = [_pdb_str_to_atoms(p) for p in pdb_contents]
        reference  = structures[0]
        ca_ref     = reference[(reference.atom_name == "CA") & ~reference.hetero]

        out_pdbs = [pdb_contents[0]]
        rmsds    = []

        for mobile in structures[1:]:
            ca_mob = mobile[(mobile.atom_name == "CA") & ~mobile.hetero]
            n = min(len(ca_ref), len(ca_mob))
            try:
                _, transform   = struc.superimpose(ca_ref[:n], ca_mob[:n])
                mobile_t       = transform.apply(mobile)
                ca_mob_t       = mobile_t[(mobile_t.atom_name == "CA") & ~mobile_t.hetero]
                rmsd_val       = struc.rmsd(ca_ref[:n], ca_mob_t[:n])
            except Exception:
                mobile_t = mobile
                rmsd_val = float("nan")

            out_pdbs.append(_atoms_to_pdb_str(mobile_t))
            rmsds.append(float(rmsd_val))

        return out_pdbs, rmsds

    # ── 3Dmol HTML builder ────────────────────────────────────────────────────

    _CHAIN_COLOURS = [
        "#4e9af1", "#f1a94e", "#e45c3a", "#4eba6f",
        "#9b59b6", "#1abc9c", "#e74c3c", "#f39c12",
        "#2980b9", "#27ae60", "#8e44ad", "#c0392b",
    ]

    def build_3dmol_html(pdb_contents: list[str], labels: list[str]) -> str:
        import html as _html

        def _js_esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

        add_models = []
        for i, (pdb, label) in enumerate(zip(pdb_contents, labels)):
            colour = _CHAIN_COLOURS[i % len(_CHAIN_COLOURS)]
            add_models.append(
                f"viewer.addModel(`{_js_esc(pdb)}`, 'pdb');\n"
                f"  viewer.setStyle({{model:{i}}}, {{cartoon:{{color:'{colour}'}}}});"
            )
        models_js = "\n  ".join(add_models)

        legend_items = " ".join(
            f'<span style="display:inline-flex;align-items:center;margin-right:10px;">'
            f'<span style="width:12px;height:12px;background:{_CHAIN_COLOURS[i % len(_CHAIN_COLOURS)]};'
            f'border-radius:2px;display:inline-block;margin-right:4px;"></span>'
            f'<span style="font-size:11px;">{_html.escape(label)}</span></span>'
            for i, label in enumerate(labels)
        )

        # Build a complete self-contained HTML document.
        # window.onload fires after the 3Dmol CDN script has loaded,
        # so $3Dmol is guaranteed to exist by the time createViewer() is called.
        inner = f"""<!DOCTYPE html>
    <html><head>
    <script src='https://3Dmol.csb.pitt.edu/build/3Dmol-min.js'></script>
    <style>
      body{{margin:0;padding:0;background:#f8f8f8;font-family:sans-serif;}}
      #legend{{padding:4px 6px;font-size:11px;}}
      #v{{position:absolute;top:28px;left:0;right:0;bottom:0;}}
    </style>
    </head><body>
    <div id='legend'>{legend_items}</div>
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

        # HTML-escape the whole document for the srcdoc attribute (double-quoted).
        # The browser will unescape it before parsing, so the JS and PDB content
        # arrive intact.
        return (
            f'<iframe srcdoc="{_html.escape(inner, quote=True)}" '
            f'style="width:100%;height:540px;border:1px solid #ddd;'
            f'border-radius:4px;" frameborder="0"></iframe>'
        )

    return (
        StringIO,
        build_3dmol_html,
        fetch_pdb,
        get_fasta,
        get_orthologs_by_gene,
        run_clustalo,
        search_uniprot,
        superimpose_all,
    )


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
    search_table  # bare expression → last_expr → displayed
    return (search_table,)


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
    # Direct variable references so marimo's AST tracker sees them as deps
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
    ortholog_table  # bare expression → last_expr → displayed
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
    align_btn = mo.ui.run_button(label="Compute alignment (Clustal Omega)", kind="success")
    align_btn  # bare expression → displayed
    return (align_btn,)


@app.cell(hide_code=True)
def _(StringIO, align_btn, get_fasta, mo, ortholog_table, run_clustalo):
    from Bio import AlignIO
    import pymsaviz

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

    import math as _math
    _msa_obj  = AlignIO.read(StringIO(aligned_fasta), "fasta")
    _n_seqs   = len(_msa_obj)
    _n_blocks = _math.ceil(_msa_obj.get_alignment_length() / 80)
    _msaviz   = pymsaviz.MsaViz(
        _msa_obj,
        color_scheme="Clustal",
        show_consensus=True,
        show_count=True,
        wrap_length=80,
    )
    _fig = _msaviz.plotfig()
    _fig.set_size_inches(14, max(4, _n_blocks * (_n_seqs * 0.4 + 2.5)))
    mo.vstack([
        mo.callout(mo.md(f"Alignment complete — **{len(_sel_accs)} sequences**."), kind="success"),
        mo.as_html(_fig),
    ])  # bare expression → displayed
    return (aligned_fasta,)


@app.cell(hide_code=True)
def _(fetch_pdb, mo, ortholog_table, orthologs_df, superimpose_all):
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
        struct_pdbs, _rmsds = superimpose_all(_pdbs_raw)

    _rmsd_df = _pd.DataFrame({
        "Gene | Organism":  struct_labels,
        "RMSD vs ref (Å)": [round(v, 2) for v in [0.0] + _rmsds],
    })
    struct_table = mo.ui.table(
        _rmsd_df,
        selection="multi",
        label=f"**{len(struct_pdbs)}** structure(s) superimposed — select rows to display in the viewer:",
    )
    struct_table  # bare expression → displayed
    return struct_labels, struct_pdbs, struct_table


@app.cell(hide_code=True)
def _(build_3dmol_html, mo, struct_labels, struct_pdbs, struct_table):
    _sel = struct_table.value
    mo.stop(
        len(_sel) == 0,
        mo.callout(mo.md("Select one or more structures in the table above to display them."), kind="info"),
    )
    # Map selected rows back to PDB list by label name
    _sel_labels = _sel["Gene | Organism"].tolist()
    _sel_pdbs   = [struct_pdbs[struct_labels.index(lbl)] for lbl in _sel_labels]
    mo.Html(build_3dmol_html(_sel_pdbs, _sel_labels))
    return


@app.cell(hide_code=True)
def _(StringIO, aligned_fasta, mo):
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
