import html as _html
from io import StringIO

import biotite.structure as struc
import biotite.structure.io.pdb as bpdb
from biotite.structure.io.pdbx import CIFFile, get_structure as _get_structure_cif

BASIC_COLORS = {
    "Blue":   "#4e9af1",
    "Orange": "#f1a94e",
    "Red":    "#e45c3a",
    "Green":  "#4eba6f",
    "Purple": "#9b59b6",
    "Teal":   "#1abc9c",
    "Pink":   "#e91e8c",
    "Yellow": "#f1c40f",
    "Gray":   "#95a5a6",
    "Cyan":   "#00bcd4",
}

_COLOR_CYCLE = list(BASIC_COLORS.keys())


def default_color(index: int) -> str:
    return _COLOR_CYCLE[index % len(_COLOR_CYCLE)]


def _is_cif(filename: str) -> bool:
    return filename.rsplit(".", 1)[-1].lower() in ("cif", "mmcif")


def parse_structure(contents: bytes, filename: str):
    """Parse PDB or CIF bytes → AtomArray (with b_factor annotation)."""
    text = contents.decode("utf-8", errors="replace")
    if _is_cif(filename):
        cf = CIFFile.read(StringIO(text))
        atoms = _get_structure_cif(cf, model=1, extra_fields=["b_factor"])
    else:
        pf = bpdb.PDBFile.read(StringIO(text))
        atoms = bpdb.get_structure(pf, model=1, extra_fields=["b_factor"])
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]
    return atoms


def parse_pdb_str(pdb_text: str):
    """Parse PDB string → AtomArray (with b_factor annotation)."""
    pf = bpdb.PDBFile.read(StringIO(pdb_text))
    atoms = bpdb.get_structure(pf, model=1, extra_fields=["b_factor"])
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]
    return atoms


def atoms_to_pdb_str(atoms) -> str:
    pf = bpdb.PDBFile()
    bpdb.set_structure(pf, atoms)
    buf = StringIO()
    pf.write(buf)
    return buf.getvalue()


def superimpose_all(structures: list) -> tuple[list, list[float]]:
    """Superimpose all structures onto the first using Cα atoms."""
    ref = structures[0]
    ca_ref = ref[(ref.atom_name == "CA") & ~ref.hetero]

    out = [ref]
    rmsds: list[float] = []

    for mobile in structures[1:]:
        ca_mob = mobile[(mobile.atom_name == "CA") & ~mobile.hetero]
        n = min(len(ca_ref), len(ca_mob))
        try:
            _, tf = struc.superimpose(ca_ref[:n], ca_mob[:n])
            mobile_t = tf.apply(mobile)
            ca_mob_t = mobile_t[(mobile_t.atom_name == "CA") & ~mobile_t.hetero]
            rmsd = float(struc.rmsd(ca_ref[:n], ca_mob_t[:n]))
        except Exception:
            mobile_t = mobile
            rmsd = float("nan")
        out.append(mobile_t)
        rmsds.append(rmsd)

    return out, rmsds


def get_b_range(atoms) -> tuple[float, float]:
    cats = atoms.get_annotation_categories()
    if "b_factor" not in cats:
        return (0.0, 100.0)
    ca_mask = (atoms.atom_name == "CA") & ~atoms.hetero
    b = atoms.b_factor[ca_mask]
    if len(b) == 0:
        return (0.0, 100.0)
    return (float(b.min()), float(b.max()))


def _js_esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def build_structure_html(
    pdb_contents: list[str],
    labels: list[str],
    visibilities: list[bool],
    color_modes: list[str],
    colors: list[str],
    b_ranges: list[tuple[float, float]],
    height: int = 600,
) -> str:
    add_models_js: list[str] = []
    for i, (pdb, visible, mode, color, (bmin, bmax)) in enumerate(
        zip(pdb_contents, visibilities, color_modes, colors, b_ranges)
    ):
        add_models_js.append(f"viewer.addModel(`{_js_esc(pdb)}`, 'pdb');")
        if not visible:
            add_models_js.append(f"  viewer.setStyle({{model:{i}}}, {{}});")
        elif mode == "B-factor":
            add_models_js.append(
                "  viewer.setStyle({model:%d}, {cartoon:{colorscheme:{prop:'b',gradient:'roygb',min:%.2f,max:%.2f}}});"
                % (i, bmin, bmax)
            )
        else:
            add_models_js.append(
                f"  viewer.setStyle({{model:{i}}}, {{cartoon:{{color:'{color}'}}}});"
            )

    models_js = "\n  ".join(add_models_js)

    legend_parts: list[str] = []
    for label, visible, mode, color in zip(labels, visibilities, color_modes, colors):
        if not visible:
            continue
        if mode == "B-factor":
            swatch = (
                '<span style="background:linear-gradient(to right,'
                '#FF0000,#FFFF00,#00FF00,#0000FF);'
                'width:28px;height:12px;display:inline-block;'
                'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
            )
        else:
            swatch = (
                f'<span style="background:{_html.escape(color)};'
                'width:12px;height:12px;border-radius:2px;'
                'display:inline-block;margin-right:4px;vertical-align:middle;"></span>'
            )
        legend_parts.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:12px;">'
            f'{swatch}<span style="font-size:11px;">{_html.escape(label)}</span></span>'
        )

    legend_html = (
        " ".join(legend_parts) if legend_parts else "<em style='color:#999'>No structures visible</em>"
    )

    inner = f"""<!DOCTYPE html>
<html><head>
<script src='https://3Dmol.csb.pitt.edu/build/3Dmol-min.js'></script>
<style>
  body{{margin:0;padding:0;background:#f8f8f8;font-family:sans-serif;}}
  #legend{{padding:5px 8px;font-size:11px;border-bottom:1px solid #e0e0e0;min-height:26px;}}
  #v{{position:absolute;top:30px;left:0;right:0;bottom:0;}}
</style>
</head><body>
<div id='legend'>{legend_html}</div>
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

    return (
        f'<iframe srcdoc="{_html.escape(inner, quote=True)}" '
        f'style="width:100%;height:{height}px;border:1px solid #ddd;'
        f'border-radius:4px;" frameborder="0"></iframe>'
    )
