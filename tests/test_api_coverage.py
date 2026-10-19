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
    mod, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


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
        if cell.startswith("internal:"):
            for ref in refs:
                if ref.startswith("spanormpy."):
                    importlib.import_module(ref.rsplit(".", 1)[0])
                    if not ref.endswith("."):
                        assert _resolve(ref) is not None, f"{r_name}: {ref}"
        else:
            for ref in refs:
                assert ref.startswith(("spanormpy.", "spanormpy.pl.")), (r_name, ref)
                assert _resolve(ref) is not None, f"{r_name}: {ref} not importable"
