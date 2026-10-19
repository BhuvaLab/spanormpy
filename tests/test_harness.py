import numpy as np

from _ref import load_ref


def test_roundtrip():
    ref = load_ref("harness")
    assert ref["m"].shape == (2, 3)
    assert ref["m"][1, 2] == 6.5
    assert np.isnan(ref["v"][1])
    assert list(ref["s"]) == ["a", "b"]
    assert ref["x"] == 3.25
    assert list(ref["f"]) == ["u", "v", "u"]


def test_dlpfc_small(dlpfc_small):
    a = dlpfc_small
    assert a.n_vars == 300
    assert a.obsm["spatial"].shape == (a.n_obs, 2)
    assert "AnnotatedCluster" in a.obs
