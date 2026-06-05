# Marimo Bioinformatics Notebooks

A collection of interactive [marimo](https://marimo.io) notebooks for common bioinformatics tasks.

## Notebooks

### General Biology

| Notebook | Description | Open |
|----------|-------------|------|
| [Protein Conservation Viewer](general_biology/protein_conservation.py) | Fetches orthologs from UniProt, aligns sequences with ClustalOmega, visualizes the MSA and superimposed AlphaFold structures, and builds a neighbour-joining phylogenetic tree. | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_KcmBSS9c8C6fFMjVrYf48K) |
| [Protein Conservation Viewer v2](general_biology/protein_conservation_v2.py) | Variant of the above with interactive MSA color schemes (including % conservation) and per-structure show/hide and B-factor coloring controls. | |

### Widgets

Standalone interactive widgets that can also be imported as Python modules (see [`widget/msa_helpers.py`](widget/msa_helpers.py) and [`widget/structure_helpers.py`](widget/structure_helpers.py)).

| Notebook | Description |
|----------|-------------|
| [Protein MSA Viewer](widget/protein_msa_viewer.py) | Upload or paste FASTA/A3M alignments. Color schemes: Clustal, Zappo, Hydrophobicity, % Conservation, and more. Configurable wrap length, consensus row, and position counts. |
| [Protein Structure Viewer](widget/protein_structure_viewer.py) | Upload PDB/CIF files. Structures are superimposed onto the first (reference) using Cα atoms. Per-structure show/hide, solid color, and B-factor coloring controls. Global override controls for all structures at once. |

## Project layout

```
general_biology/          # End-to-end workflow notebooks
widget/
  msa_helpers.py          # Shared MSA visualization helpers
  structure_helpers.py    # Shared structure parsing and visualization helpers
  protein_msa_viewer.py   # Standalone MSA viewer widget
  protein_structure_viewer.py  # Standalone structure viewer widget
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
uv run marimo run general_biology/protein_conservation_v2.py
```

Key dependencies: `marimo`, `biopython`, `pymsaviz`, `biotite`, `requests`, `pandas`, `matplotlib`.
