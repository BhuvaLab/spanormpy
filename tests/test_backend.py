import math

import numpy as np
import pytest
import torch

from _ref import load_ref
from spanormpy import _backend as B
from spanormpy import _nbdist as nb

INT_MIN = -2147483648


@pytest.fixture(scope="module")
def ref():
    return load_ref("backend")


@pytest.fixture(autouse=True)
def _clean_cache():
    B.reset_gpu_cache()
    yield
    B.reset_gpu_cache()


def _pin_device(name):
    B._cache["device"] = name


def _grid(ref):
    g = ref["count"]["grid"]
    return list(zip(g["n_genes"], g["n_cells"], g["n_cov"], g["budget"]))


@pytest.mark.parametrize(
    "key,device,backend",
    [
        ("cuda_gpu", "cuda", "gpu"),
        ("cuda_auto", "cuda", "auto"),
        ("cuda_cpu", "cuda", "cpu"),
        ("mps_gpu", "mps", "gpu"),
        ("cpu_gpu", "cpu", "gpu"),
    ],
)
def test_gene_block_count_matches_r(ref, key, device, backend):
    _pin_device(device)
    expected = np.atleast_1d(ref["count"][key])
    for (g, c, k, b), e in zip(_grid(ref), expected):
        got = B.gene_block_count(int(g), int(c), int(k), backend=backend, budget_bytes=float(b))
        if e == INT_MIN:
            # R overflows as.integer() to NA when needed/budget > 2^31 (r-discrepancies.md);
            # the port clamps to n_genes instead.
            assert got == int(g)
        else:
            assert got == e, (g, c, k, b)


def test_gene_block_count_edge_cases():
    _pin_device("cuda")
    assert B.gene_block_count(100, 100, 10, budget_bytes=math.inf) == 1
    assert B.gene_block_count(100, 100, 10, budget_bytes=1e12) == 1
    assert B.gene_block_count(1e5, 1e5, 20, backend="cpu", budget_bytes=1) == 1
    nblocks = B.gene_block_count(1000, 1000, 10, backend="gpu", budget_bytes=1000)
    assert 1 < nblocks <= 1000
    assert B.gene_block_count(5, 10**6, 10, backend="gpu", budget_bytes=1) == 5
    with pytest.raises(ValueError):
        B.gene_block_count(5, 5, 5, backend="tpu")


def test_gene_block_indices_matches_r(ref):
    g = ref["idx_grid"]
    for i, (n, nblocks) in enumerate(zip(g["n_genes"], g["nblocks"])):
        r = ref["idx"][f"case{i + 1}"]
        blocks = B.gene_block_indices(int(n), int(nblocks))
        np.testing.assert_array_equal(np.concatenate(blocks), np.atleast_1d(r["flat"]) - 1)
        np.testing.assert_array_equal([len(b) for b in blocks], np.atleast_1d(r["lens"]))


def test_gene_block_indices_contiguous():
    blocks = B.gene_block_indices(10, 3)
    assert len(blocks) == 3
    np.testing.assert_array_equal(np.concatenate(blocks), np.arange(10))
    np.testing.assert_array_equal(np.concatenate(B.gene_block_indices(5, 1)), np.arange(5))


def _nb_inputs(ref):
    d = ref["nb"]
    return d["x"], d["mu"], d["size"]


def test_dnbinom_matches_r(ref, device):
    x, mu, size = _nb_inputs(ref)
    d = ref["nb"]
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    got = nb.dnbinom(t(x), t(mu), t(size)).cpu().numpy()
    np.testing.assert_allclose(got, d["d"], rtol=1e-10, atol=0)
    got = nb.dnbinom(t(x), t(mu), t(size), log=True).cpu().numpy()
    np.testing.assert_allclose(got, d["dlog"], rtol=1e-10, atol=0)


def test_dnbinom_matrix_with_per_gene_size(ref, device):
    d = ref["nb"]
    got = nb.dnbinom(
        torch.as_tensor(d["xm"], device=device, dtype=torch.float64),
        torch.as_tensor(d["mum"], device=device, dtype=torch.float64),
        torch.as_tensor(d["szv"], device=device, dtype=torch.float64),
    )
    np.testing.assert_allclose(got.cpu().numpy(), d["dm"], rtol=1e-10, atol=0)


def test_pnbinom_qnbinom_match_r(ref, device):
    d = ref["nb"]
    x, mu, size = _nb_inputs(ref)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    got = nb.pnbinom(t(x), t(mu), t(size)).cpu().numpy()
    np.testing.assert_allclose(got, d["p"], rtol=1e-10, atol=1e-300)
    gotlog = nb.pnbinom(t(x), t(mu), t(size), log=True).cpu().numpy()
    np.testing.assert_allclose(gotlog, d["plog"], rtol=1e-9, atol=1e-12)

    pm = nb.pnbinom(
        t(d["xm"]), t(d["mum"]), t(d["szv"])
    ).cpu().numpy()
    np.testing.assert_allclose(pm, d["pm"], rtol=1e-10, atol=1e-300)

    q = nb.qnbinom(t(d["qp"]), t(d["qmu"]), t(d["qsize"])).cpu().numpy()
    np.testing.assert_array_equal(q, d["q"])
    # p exactly on a CDF step: the fuzz must agree with R
    rt = nb.qnbinom(t(d["rt_p"]), t(d["rt_mu"]), t(d["rt_size"])).cpu().numpy()
    np.testing.assert_array_equal(rt, d["rt_q"])


def test_accelerator_loop_matches_r(ref):
    """The cumulative-sum path used on GPUs, forced here on CPU tensors."""
    d = ref["nb"]
    x, mu, size = _nb_inputs(ref)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    got = nb._pnbinom_loop(t(x), t(mu), t(size)).numpy()
    np.testing.assert_allclose(got, d["p"], rtol=1e-10, atol=1e-300)
    q = nb._qnbinom_loop(t(d["qp"]), t(d["qmu"]), t(d["qsize"])).numpy()
    np.testing.assert_array_equal(q, d["q"])
    # Documented deviation (widened fuzz): at most one step off R, never below it.
    rt = nb._qnbinom_loop(t(d["rt_p"]), t(d["rt_mu"]), t(d["rt_size"])).numpy()
    np.testing.assert_array_equal(np.isinf(rt), np.isinf(d["rt_q"]))
    fin = np.isfinite(d["rt_q"])
    assert np.all(rt[fin] >= d["rt_q"][fin]) and np.all(rt[fin] - d["rt_q"][fin] <= 1)
    assert (rt[fin] == d["rt_q"][fin]).mean() > 0.97


def test_qnbinom_cap_warns_nan(monkeypatch):
    monkeypatch.setattr(nb, "_QNBINOM_CAP", 3)
    p = torch.tensor([0.5, 0.999999], dtype=torch.float64)
    with pytest.warns(UserWarning, match="cap"):
        q = nb._qnbinom_loop(p, torch.tensor([100.0, 100.0], dtype=torch.float64), 10.0)
    assert torch.isnan(q).all()


def test_nbdist_empty_tensors():
    e = torch.zeros(0, dtype=torch.float64)
    for fn in (nb.pnbinom, nb._pnbinom_loop):
        assert fn(e, e, 1.0).numel() == 0
    for fn in (nb.qnbinom, nb._qnbinom_loop):
        assert fn(e, e, 1.0).numel() == 0


def test_nbdist_edge_semantics():
    mu0 = torch.zeros(3, dtype=torch.float64)
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    np.testing.assert_array_equal(nb.dnbinom(x, mu0, 2.0).numpy(), [1, 0, 0])
    # non-integer / negative support has zero mass
    assert nb.dnbinom(torch.tensor([-1.0, 0.5], dtype=torch.float64), 1.0, 1.0).sum() == 0
    assert math.isnan(nb.qnbinom(torch.tensor([1.5], dtype=torch.float64), 1.0, 1.0).item())


def test_winsorise_matches_r(ref):
    w = ref["win"]
    np.testing.assert_allclose(B.winsorise_cols(w["alpha"], 4), w["cols4"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(B.winsorise_cols(w["alpha"], 1), w["cols1"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nb.winsorise_psi(w["psi"]), w["psi4"], rtol=1e-12)
    np.testing.assert_allclose(nb.winsorise_psi(w["psi"], winsor=1), w["psi1"], rtol=1e-12)
    # torch input follows the same path
    got = B.winsorise_cols(torch.as_tensor(w["alpha"]), 4)
    assert isinstance(got, torch.Tensor)
    np.testing.assert_allclose(got.numpy(), w["cols4"], rtol=1e-12, atol=1e-12)


def test_winsorise_validation_and_inf():
    m = np.column_stack([np.full(6, 5.0), np.arange(1.0, 7.0)])
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="positive"):
            B.winsorise_cols(m, k=bad)
        with pytest.raises(ValueError, match="positive"):
            nb.winsorise_psi([1.0, 2.0, 3.0], winsor=bad)
    res = B.winsorise_cols(m, k=np.inf)  # zero-MAD column must not produce NaN
    assert not np.isnan(res).any()
    np.testing.assert_array_equal(res, m)
    np.testing.assert_array_equal(nb.winsorise_psi([1.0, 2.0, 3.0], winsor=np.inf), [1, 2, 3])
    np.testing.assert_allclose(nb.winsorise_psi([1.0, 2.0, 100.0], psi_max=3.0), [1, 2, 3])


def test_has_bad_values():
    m = np.arange(1.0, 7.0).reshape(2, 3)
    assert not B.has_bad_values(m)
    for bad in (np.nan, np.inf, -np.inf):
        m2 = m.copy()
        m2[0, 0] = bad
        assert B.has_bad_values(m2)
        assert B.has_bad_values(torch.as_tensor(m2))
    assert not B.has_bad_values(torch.as_tensor(m))


def test_auto_backend_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    B.reset_gpu_cache()
    assert B.get_backend_device() == "cpu"
    assert B.resolve_device("gpu").type == "cpu"
    assert B.resolve_device("auto").type == "cpu"
    assert B.resolve_device("cpu").type == "cpu"
    assert B.get_backend_dtype() == torch.float64 and B.gpu_dtype_bytes() == 8
    assert not B.check_gpu()
    assert B.get_gpu_memory_budget() == math.inf
    with pytest.raises(ValueError):
        B.resolve_device("tpu")


def test_resolve_device_accelerator():
    _pin_device("cuda")
    assert B.resolve_device("gpu").type == "cuda" and B.resolve_device("auto").type == "cuda"
    assert B.resolve_device("cpu").type == "cpu"
    _pin_device("mps")
    assert B.get_backend_dtype() == torch.float32 and B.gpu_dtype_bytes() == 4


def test_set_get_budget_roundtrip():
    B.set_gpu_memory_budget(12345)
    assert B.get_gpu_memory_budget() == 12345
    B.set_gpu_memory_budget(math.inf)
    assert B.get_gpu_memory_budget() == math.inf
    B.reset_gpu_cache()
    B.set_gpu_memory_budget(12345)
    B.set_gpu_memory_budget(None)
    assert "gpu_mem_budget" not in B._cache
    assert B.get_gpu_memory_budget(gpu_mem_budget=777) == 777
    assert B.get_gpu_memory_budget(gpu_mem_budget=math.inf) == math.inf


@pytest.mark.parametrize("bad", [-1, 0, float("nan"), [1, 2], "a", True])
def test_invalid_budgets_raise(bad):
    with pytest.raises(ValueError, match="positive"):
        B.set_gpu_memory_budget(bad)
    with pytest.raises(ValueError, match="positive"):
        B.get_gpu_memory_budget(gpu_mem_budget=bad)


def test_budget_detection(monkeypatch):
    _pin_device("cuda")
    monkeypatch.setattr(B, "_cuda_mem_free_bytes", lambda: 8192 * 1024**2)
    assert B.get_gpu_memory_budget() == 8192 * 1024**2
    monkeypatch.setattr(B, "_cuda_mem_free_bytes", lambda: 1.0)
    assert B.get_gpu_memory_budget() == 8192 * 1024**2  # cached for the session
    B.reset_gpu_cache()
    _pin_device("mps")
    monkeypatch.setattr(B, "_mps_mem_budget_bytes", lambda: 17179869184 * 0.5)
    assert B.get_gpu_memory_budget() == 17179869184 * 0.5
    B.reset_gpu_cache()
    _pin_device("cuda")
    monkeypatch.setattr(B, "_cuda_mem_free_bytes", lambda: math.nan)
    with pytest.warns(UserWarning, match="Could not auto-detect"):
        assert B.get_gpu_memory_budget() == B.GPU_MEM_BUDGET_FALLBACK_BYTES


def test_as_tensor(device):
    t = B.as_tensor([[1, 2], [3, 4]], device)
    assert t.dtype == torch.float64 and t.device.type == device
    t32 = B.as_tensor(np.ones(3), "cpu", dtype=torch.float32)
    assert t32.dtype == torch.float32
    assert B.as_tensor(t, "cpu").device.type == "cpu"
