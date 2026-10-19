"""Every R export at SpaNorm c12e5e5 is mapped in docs/r-to-python.md, and the mapping is real."""

import importlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R_REPO = ROOT.parent / "SpaNorm"
DOC = ROOT / "docs" / "r-to-python.md"


def _r_exports() -> list[str]:
    ns = subprocess.run(
        ["git", "-C", str(R_REPO), "show", "c12e5e5:NAMESPACE"],
        check=True, capture_output=True, text=True,
    ).stdout
    return re.findall(r"^export\(([^)]+)\)$", ns, flags=re.M)


def _rows() -> dict[str, str]:
    """Map the first-column R name to the second-column Python cell of every table row."""
    rows = {}
    for line in DOC.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 2 and cells[0].startswith("`"):
            rows[cells[0].strip("`")] = cells[1]
    return rows


def _resolve(dotted: str):
    """Import the longest module prefix, then getattr the rest (e.g. ``torch.Tensor.numpy``)."""
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


@pytest.mark.skipif(not R_REPO.exists(), reason="../SpaNorm not available")
def test_every_r_export_mapped():
    rows = _rows()
    missing = [name for name in _r_exports() if name not in rows]
    assert not missing, f"R exports missing from docs/r-to-python.md: {missing}"


def test_python_names_in_mapping_resolve():
    rows = _rows()
    assert rows, "no mapping rows parsed"
    for r_name, cell in rows.items():
        refs = re.findall(r"`([A-Za-z_][\w.]*)`", cell)
        assert refs, f"{r_name}: no Python name given"
        # `torch.*` and `spanormpy.*` names are resolved; `spanormpy._x.y` rows are internal
        # (leading underscore modules), public rows must live directly in spanormpy or spanormpy.pl
        if not cell.startswith("internal:"):
            assert all(re.fullmatch(r"spanormpy(\.pl)?\.\w+", ref) for ref in refs), (r_name, refs)
        for ref in refs:
            if ref.startswith(("spanormpy.", "torch.")):
                assert _resolve(ref) is not None, f"{r_name}: {ref} not importable"
