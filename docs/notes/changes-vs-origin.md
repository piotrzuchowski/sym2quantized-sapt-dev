# Changes with respect to origin (updated 2026-09-12)

Work done in this clone on top of `origin/main` (`675c0aa`) and
`origin/feature/density_fitting` (`df29562`), by Claude Code sessions
driven by a downstream user of the package.  Each stream sits on its
own branch for review, published on the fork
`piotrzuchowski/sym2quantized-sapt-dev`.  The downstream
application (a SAPT dispersion code) is referred to only as the test
bed — nothing in these changes assumes or requires its theory.

## `feature/term-graphs` (off `main`)

**`term_graphs.py` — terms as tensor-network graphs.**  Each fully
contracted term becomes nodes (tensor factors), edges (contracted
dummies, labelled by space and monomer) and half-edges (free indices).
An index on three or more slots is carried as a *hyperedge* — needed
for resolvent denominators, which share every index with the
amplitudes they divide.  On top of the encoding:

- `canonical_key` — exact canonical labelling (dummy names never
  enter; declared tensor *symmetries* are deliberately not folded in,
  see the module docstring);
- `invariants` / `classify` — filterable facts and grouping; loop
  counts call `spin_integrator._count_loops` itself, so `2**loops`
  is exactly the spin-integration factor;
- `contraction_cost` — exhaustive pairwise-order search with symbolic
  dimensions, per-index bookkeeping (hyperedges and traces cost
  correctly);
- `common_pairs` — recurring two-tensor contractions across a term
  list, the shared-intermediate candidates;
- `to_dict` / `to_dot` — JSON and Graphviz export.

**`derivation_registry.py` — persistent, diffable derivation records.**
One JSON record per derived object; each term stored with coefficient,
graph, generated einsum line and canonical key, plus provenance and
explicit numerical-validation stamps (topology can never pin a
normalisation, so the stamp field is first-class).  `diff_records` /
`verify_expr` give term-level diffs keyed by topology — a rerun that
only reorders or renames diffs silent; `lint_record` checks what a
correct derivation cannot violate (uniform external pattern, connected
terms, merged topologies, no coefficient far above `2**loops` — the
mis-declared-direction signature).  `density_fitting=True` einsum
output needs a `code_generator` that knows the option (the branch
below); requesting it elsewhere raises cleanly.

New: `examples/derivation_registry_demo.py` (auto-tested via the slow
example runner), `tests/test_term_graphs.py`,
`tests/test_derivation_registry.py`.  No new dependencies.  The one
existing file modified is `tests/test_examples.py`: the example runner
now puts the checkout's own `src` first on the subprocess
`PYTHONPATH`, so examples run against the branch under test rather
than whatever checkout the editable install points at (see the last
section — this same artifact breaks the density-fitting branch's
examples, and merging this fix resolves it there too).

## `feature/density_fitting_w` (off `origin/feature/density_fitting`)

`generate_einsum(..., density_fitting=True)` also factorizes the
*intramonomer* two-electron integral `w^{p p'}_{q q'} = (q p | q' p')`
— same positional pairing as `v`, `sum_Q B_qp B_q'p'`.  Changes:
`_is_eri` matches `w`; `_check_eri_indices` additionally rejects a `w`
straddling monomers, which the per-pair check cannot see (each pair of
a v-style split sits on one monomer).  Four new tests.  Validated
downstream: the factorized output of a 20-term derivation equals its
dense evaluation to ~1e-15 relative.

## `feature/uhf-spin-summation` (off `main`)

Open-shell support, validated against psi4 at every step (numeric
records live in the downstream application's `docs/RESULTS_UHF_*.md`).

- **`open_shell.py`** holds every open-shell addition, so the existing
  modules keep small, reviewable diffs: `spin_integrator.py` gains only
  the `_loop_partition` refactor (no behaviour change -- it exposes the
  loop membership `_count_loops` was already computing), and
  `double_fermi_vac.py` only an import plus the two-line tag guard.
  `array_table` deliberately stays in `code_generator.py`: it describes
  the arrays the generator emits for *any* expression, blocked or not,
  and downstream code imports it from there.
- **Spin tags** (`is_alpha` / `is_beta`): the monomer-tag mechanism
  applied to spin -- `contraction_double_vac` vanishes across
  opposite tags, and the Wick fallback's fresh summation dummies
  inherit the shared tag (a bubble otherwise silently sums both
  spins).  With per-sector resolvent normalisation (`1/(n!)^2` per
  same-spin pair group, distinguishable pairs unpermuted), UMP2 and
  UMP3 match psi4's conventional UHF-MP2/MP3 to ~1e-15 / ~5e-13 on
  doublet, triplet and quartet references; SAPT's disp20 matches
  open-shell SAPT0 to machine zero; exch-disp20 (S^2) matches psi4's
  S^2 modules to 6e-20 at closed shell; the full blocked CCPP2
  equations reduce to the validated closed-shell result to 6e-13.
- **`spin_integration_uhf` / `rhf_collapse`** in
  `spin_integrator.py`: per-Goldstone-loop spin summation
  (`_loop_partition` now exposes the membership `_count_loops`
  always computed; RHF path unchanged).  Valid for
  single-pair-per-space projections; carries a prominent warning that
  multi-pair resolvents need the tags instead -- a benchmark-caught
  limitation, pinned by the `ump2_ump3_uhf.py` example, which also
  demonstrates the tagged route.
- **`code_generator.array_table`**: defines every emitted array axis
  by axis (base, upper/lower role, space, monomer, spin from tag or
  block label) -- the missing half of code generation, letting numeric
  code build spin-blocked arrays mechanically.
- New tests throughout; the module suites pass.

## Pre-existing issue observed (fixed on `feature/term-graphs`)

On any checkout of `feature/density_fitting`, the four slow
example-runner tests fail: the runner executed examples with the
*installed* (editable, main-branch) package, whose operator aliases
are still lowercase, while the branch's examples import the renamed
`A, Ad, B, Bd`.  An artifact of the editable install pointing at the
main checkout.  The `tests/test_examples.py` fix on
`feature/term-graphs` (checkout `src` first on the subprocess path)
resolves the class; the density-fitting branch inherits it on merge.
