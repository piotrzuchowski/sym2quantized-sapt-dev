# Changes with respect to origin (2026-09-11)

Work done in this clone on top of `origin/main` (`675c0aa`) and
`origin/feature/density_fitting` (`df29562`), by Claude Code sessions
driven by a downstream user of the package.  Nothing has been pushed;
each stream sits on its own local branch for review.  The downstream
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

## Pre-existing issue observed (fixed on `feature/term-graphs`)

On any checkout of `feature/density_fitting`, the four slow
example-runner tests fail: the runner executed examples with the
*installed* (editable, main-branch) package, whose operator aliases
are still lowercase, while the branch's examples import the renamed
`A, Ad, B, Bd`.  An artifact of the editable install pointing at the
main checkout.  The `tests/test_examples.py` fix on
`feature/term-graphs` (checkout `src` first on the subprocess path)
resolves the class; the density-fitting branch inherits it on merge.
