# Marimo Bioinformatics Notebooks

A collection of interactive [marimo](https://marimo.io) notebooks for common bioinformatics tasks.

## Notebooks

### General Biology

| Notebook | Description | Open |
|----------|-------------|------|
| [Protein Conservation Viewer](general_biology/protein_conservation.py) | Fetches orthologs from UniProt, aligns sequences with ClustalOmega, and compares them on both axes of conservation: an interactive MSA (with selectable color schemes, including % conservation) and superimposed AlphaFold structures (RMSD and TM-score, per-structure show/hide and B-factor coloring), plus a neighbour-joining phylogenetic tree. | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_EmobwA4vrbbjSJTfhXeLLt) |
| [Protein Conservation Viewer (no widget)](general_biology/protein_conservation_no_widget.py) | Self-contained variant of the above that doesn't depend on the local `widget/` helpers — useful as a single-file reference or when running somewhere the `widget` package isn't available. | |
| [Protenix Structure Prediction](general_biology/protenix_predict.py) | Runs [Protenix](https://github.com/bytedance/protenix), ByteDance's open-source AlphaFold3-like model, to jointly predict structures of proteins, DNA, RNA, ligands and ions, with optional structural constraints, and reports ranked models with confidence metrics (pTM, ipTM, ipSAE, PAE). | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_ekqc1wdwLGCSMz6MrTbytw) |

### Widgets

Standalone interactive widgets that can also be imported as Python modules (see [`widget/msa_helpers.py`](widget/msa_helpers.py), [`widget/structure_helpers.py`](widget/structure_helpers.py) and [`widget/af3_helpers.py`](widget/af3_helpers.py)).

| Notebook | Description |
|----------|-------------|
| [Protein MSA Viewer](widget/protein_msa_viewer.py) | Upload or paste FASTA/A3M alignments. Color schemes: Clustal, Zappo, Hydrophobicity, % Conservation, and more. Configurable wrap length, consensus row, and position counts. |
| [Protein Structure Viewer](widget/protein_structure_viewer.py) | Upload PDB/CIF files. Structures are superimposed onto the first (reference) using Cα atoms. Per-structure show/hide, solid color, and B-factor coloring controls. Global override controls for all structures at once. |
| [AlphaFold3 Prediction Viewer](widget/af3_prediction_viewer.py) | Upload an AlphaFold Server / AlphaFold3 job's output files (`model_*.cif`, `summary_confidences_*.json`, `full_data_*.json`). Shows pTM/ipTM confidence metrics (with pairwise ipTM matrices for &gt; 2 chains), per-model PAE heatmaps, and the superimposed predicted structures with pLDDT (B-factor) coloring. Bring your own AlphaFold3 job output — point it at the files downloaded from the AlphaFold Server. |

## Project layout

```
general_biology/          # End-to-end workflow notebooks
widget/
  msa_helpers.py          # Shared MSA visualization helpers
  structure_helpers.py    # Shared structure parsing and visualization helpers
  af3_helpers.py          # Shared AlphaFold3 output parsing and confidence-plot helpers
  protein_msa_viewer.py   # Standalone MSA viewer widget
  protein_structure_viewer.py  # Standalone structure viewer widget
  af3_prediction_viewer.py     # Standalone AlphaFold3 prediction viewer widget
```

## Usage

Run any notebook with:

```bash
uv run marimo run <notebook.py>
```

Or open it for editing with:

```bash
uv run marimo edit <notebook.py>
```

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/). Install and run in one step:

```bash
uv run marimo run general_biology/protein_conservation.py
```

Key dependencies: `marimo`, `biopython`, `pymsaviz`, `biotite`, `requests`, `pandas`, `matplotlib`.
