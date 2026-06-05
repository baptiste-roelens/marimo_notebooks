import re
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import pymsaviz

PYMSAVIZ_SCHEMES = [
    "Clustal", "Zappo", "Taylor", "Flower", "Blossom",
    "BuriedIndex", "HelixPropensity", "Hydrophobicity",
    "Identity", "Ocean", "Sunset", "% Conservation",
]


def detect_format(text: str) -> str:
    for line in text.splitlines():
        if line.startswith(">"):
            continue
        if re.search(r"[a-z]", line):
            return "A3M"
    return "FASTA"


def parse_a3m(text: str) -> MultipleSeqAlignment:
    records = []
    current_id = None
    current_desc = ""
    current_seq: list[str] = []

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if current_id is not None:
                raw = "".join(current_seq)
                seq = re.sub(r"[a-z]", "", raw).replace(".", "-")
                records.append(
                    SeqRecord(Seq(seq), id=current_id, description=current_desc)
                )
            parts = line[1:].split(None, 1)
            current_id = parts[0] if parts else "seq"
            current_desc = parts[1] if len(parts) > 1 else ""
            current_seq = []
        else:
            current_seq.append(line)

    if current_id is not None:
        raw = "".join(current_seq)
        seq = re.sub(r"[a-z]", "", raw).replace(".", "-")
        records.append(
            SeqRecord(Seq(seq), id=current_id, description=current_desc)
        )

    return MultipleSeqAlignment(records)


def _conservation_scores(alignment: MultipleSeqAlignment) -> list[float]:
    n_seqs = len(alignment)
    scores = []
    for col in range(alignment.get_alignment_length()):
        chars = [str(alignment[i].seq[col]) for i in range(n_seqs)]
        non_gap = [c for c in chars if c not in ("-", ".", "X", "x")]
        if non_gap:
            counts = Counter(non_gap)
            scores.append(max(counts.values()) / len(non_gap))
        else:
            scores.append(0.0)
    return scores


def make_conservation_color_func(alignment: MultipleSeqAlignment):
    scores = _conservation_scores(alignment)
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def color_func(row: int, col: int, char: str, msa) -> str | None:
        if char in ("-", "."):
            return None
        r, g, b, _ = cmap(norm(scores[col]))
        return mcolors.to_hex((r, g, b))

    return color_func


def build_msaviz_figure(
    alignment: MultipleSeqAlignment,
    color_scheme: str,
    wrap_length: int,
    show_consensus: bool,
    show_counts: bool,
):
    use_conservation = color_scheme == "% Conservation"
    mv = pymsaviz.MsaViz(
        alignment,
        color_scheme="None" if use_conservation else color_scheme,
        wrap_length=wrap_length,
        show_consensus=show_consensus,
        show_count=show_counts,
    )
    if use_conservation:
        mv.set_custom_color_func(make_conservation_color_func(alignment))

    n_seqs = len(alignment)
    n_blocks = math.ceil(alignment.get_alignment_length() / wrap_length)
    fig = mv.plotfig()
    fig.set_size_inches(14, max(4, n_blocks * (n_seqs * 0.4 + 2.5)))
    return fig
