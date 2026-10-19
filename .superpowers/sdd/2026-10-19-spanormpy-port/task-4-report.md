# Task 4 report: Newton solvers and block design

## Implemented
- `src/spanormpy/_solver.py`: `nb_newton_solver` (dense, diagonal-nested, blocked per-sample, compact via block design), `nb_newton_solver_batch`, `nb_gram_batch`, `nb_absorb_gram_batch`; private `_absorb_blocks`, `_newton_solver_blocked`, `_gram_batch`, `_segment_sum`, `_absorb_batch`, `_chol_batch`, `_chol_solve_batch`, `_subset_state`, `_newton_solver_batch`. Closures are returned as a frozen `NewtonSolver(factor, solve, xcov)` (also indexable by name); states are `NFac` (attribute access = R `$`). R `NULL` returns are `None`. Contract/shapes documented in the module docstring.
- `src/spanormpy/_block_design.py`: `NBBlockDesign` (`.shape`, `.colnames`, `.to_dense()`, `__repr__`), `nb_block_design`, `_block_expand`, `_block_dense_absorb`, `_block_col_cells`, `_block_eta_vec/_batch`, `_block_score_vec/_batch`, `_design_eta/_cross/_col_nonzero` (dense tensor or block design), `_block_half_inverse`, `_newton_solver_compact`. The sparse `Zsp` is replaced by gather/index_add (no scipy). Column indices are 0-based; R `NA` block ids are `-1` internally.
- `tests/ref/make_solver.R` -> `tests/ref/data/solver.h5` (committed, like tps.h5); `tests/test_solver.py`, `tests/test_block_design.py`.
- `docs/r-discrepancies.md`: three entries (empty nested group, singularity test, batched xcov on a failed slice).

## Tests
`pytest -q`: 84 passed (36 new: 22 solver, 14 block design). R comparisons at rtol=1e-10, atol=1e-12 for dense / nested / blocked / batch / gram / absorb (parts, tile, empty group) / compact (main, rank-2 aliased group via the eigen route, empty group) / helper products. Ported testthat assertions from test-nbSolver.R (all non-torch-gated), test-nbBlockDesign.R (validation, expand, repr, dense-only errors, must-be-NULL, generalised-inverse and empty-group behaviour vs the dense oracle) and test-nbArrayBatch.R (`.segmentSum`; the other `.setRows`/`.mulRows`... helpers belong to the polish engine and are not ported here).
Not ported: the `polishNB(...)`-based tests of test-nbBlockDesign.R (need Tasks 7/13), `.muBatch` product check, and the two torch-vs-base agreement tests (single torch path here, which is compared to R's CPU output directly). CUDA not testable on this node; tests take the `device` fixture where a tensor is built.

TDD note: tests were written after the implementation; the RED run was not captured.

## Concerns / decisions
- `nbGramBatch`/`.absorbBatch` use the sqrt-weighted form (R's torch branch); R's CPU branch uses `Wi * w` - equal to ~1e-16, verified against R's CPU output.
- R's dense `solve()` singularity (dgesv + rcond < eps) is mimicked with LU and an exact 1-norm rcond, which costs an extra inverse per call (O(p^3), same order as the LU). Fine for the dense (non-absorbed) path but worth knowing for large p.
- Blocked-state `fac` holds lower Cholesky factors (R keeps upper); internal only.
- Dense nested path: empty group no longer misaligns (see r-discrepancies).
- `nb_block_design` takes column names from DataFrame inputs (R takes `colnames()`); otherwise `X1..`, `Z1..`.
- Ruff was run on the new files only; pre-existing files already fail `ruff check`.

## Fix round 1
Changes:
- `make_solver.R` now also writes `blk_alias3` (5 groups, q=3, group 4 has 2 cells, unpenalised: rank [3,3,3,2,3]) with `alias3_extra` (reduced-design xcov, dense grouped factor/solve/xcov NULL flags), `blk_near` (noise 2e-5: every pivot^2 above rank_tol at kappa > 1e8, rank full, dense grouped step/xcov) and `blk_near_trunc` (noise 1e-6: rank [3,2,3,3,3], eigen route), and `blk_partial` (a column zero and unpenalised in all groups but one: rank [3,2,2,2], the partial-keep branch) with `partial_extra` (dense grouped path returns NULL).
- New tests: aliased group (rank vector, compact xcov equals reduced-design xcov, R flags dense factor NULL); `test_dense_grouped_solver_flags_exactly_singular_block` drives `_newton_solver_blocked.parts` to None through `factor`, `solve(w)`, `xcov(w)`, `solve(None)`, `xcov(None)`; near-aliased (compact step/xcov equal R's dense grouped path and `rank == q`, which pins the pivot threshold); truncated eigen route vs R; non-finite weights give None from the compact solver (`factor`/`solve`/`xcov`, plus `None` states); the partial-keep case is in the R-compared CASES loop.
- `_block_expand` is now device-agnostic: a torch tensor stays a tensor on its device (numpy in, numpy out). `nb_block_design` gained an optional `device=` argument. All block-design tests take the `device` fixture; a new test checks `_block_expand` on a device tensor.
- Tolerance note: for `blk_near` (kappa ~1e9) S/step differ from R by ~1e-8 relative, so those comparisons use rtol 1e-6 (kappa*eps), commented in the test. All other cases stay at 1e-10.
- Finding: with the 2-cell unpenalised group the dense grouped Cholesky pivot is pure roundoff, so R returns NULL while torch's Gram happened to give a tiny positive pivot (factor not None). That outcome is rounding-dependent in both languages; the deterministic None path is therefore tested on the exactly zero-diagonal `blk_partial` design instead, and only R's NULL flag is asserted for the 2-cell case.

Command: `~/miniconda3/envs/spanormpy/bin/python -m pytest tests/test_solver.py tests/test_block_design.py -v`
Output: `51 passed in 0.49s` (ruff check and format clean on the touched files).
