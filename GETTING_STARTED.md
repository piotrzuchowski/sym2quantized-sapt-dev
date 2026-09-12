# Getting started

How to begin a new derivation project with `sym2quantized-sapt`: the
environment, which branch to start from, the shape of a derivation
script, and — most importantly — the conventions that cost real
debugging time when they are violated.

`README.md` is the short introduction; `CLAUDE.md` is the module map
and API detail.  This file is the practical on-ramp.

## 1. Environment

```shell
python3 -m venv venv && source venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pytest ./tests/            # add --slow for the example runners
```

Two constraints worth knowing before they surprise you:

* **sympy is pinned `>=1.8,<1.13`.**  The package subclasses sympy's
  `secondquant` internals, and 1.13 changed `Float`/`int` equality,
  which breaks coefficient formatting in `code_generator`.  Treat a
  sympy upgrade as breaking until proven otherwise.
* Many tests assert on exact `latex()` output, so they double as
  regression guards against sympy behaviour changes.  A test failing
  after an upgrade is usually the guard doing its job.

Python 3.8 is the reference interpreter (CI); 3.9–3.12 also work.

## 2. Which branch

The features live on separate branches.  Pick by what you need:

| branch | gives you |
|---|---|
| `main` | the core: double-vacuum Wick, RHF spin integration, einsum generation |
| `feature/uhf-spin-summation` | **open shell**: spin tags, `spin_integration_uhf`, `rhf_collapse`, `code_generator.array_table` |
| `feature/term-graphs` | terms as tensor-network graphs; the derivation registry |
| `feature/density_fitting` | density-fitted emission for the intermolecular `v` |
| `feature/density_fitting_w` | the above, extended to the intramonomer `w` |

Merge status, measured rather than assumed:

* `feature/term-graphs` + `feature/uhf-spin-summation` merge with
  **zero conflicts** — they touch nearly disjoint files.  That
  combination is a safe base for open-shell work today.
* Adding a density-fitting branch shows no *textual* conflicts either,
  but do not trust that: the DF lineage carries two deliberate API
  breaks — operator aliases renamed `a, ad, b, bd` → `A, Ad, B, Bd`,
  and `DoubleVacuumTensorSymbol.upper/lower/symbol` changed from
  methods to properties.  A merged tree imports fine and then fails in
  every derivation script.  Agree on one convention before merging
  rather than shimming it per project.

`docs/notes/changes-vs-origin.md` describes what each non-upstream
branch changed and why.

## 3. The shape of a derivation

Every script follows the same pipeline:

```python
expr = projection * operator_product            # build
expr = wicks_double_vac(expr.expand(),          # contract
                        keep_only_fully_contracted=True,
                        substitute_dummies=False)
expr = evaluate_deltas_double_vac(expr)         # resolve deltas
expr = substitute_dummies_double_vac(expr)      # canonicalise
expr = get_only_linked(expr)                    # drop disconnected
expr = spin_integration(expr)                   # RHF only; see §5
print(generate_einsum(expr))                    # emit
```

Operators come from `DoubleVacuumTensorSymbol` (the tensor) multiplied
by fermion operators; `sapt_utils` has ready-made builders
(`get_V_operator`, `get_P2_operator`, `get_R_nm`, …).
`examples/sapt_pol20.py` is the canonical end-to-end demonstration.

## 4. Conventions that bite

Each of these has cost real debugging time — several of them more than
once, in more than one project.  They are cheap to respect and
expensive to rediscover.

1. **Free indices are plain `Symbol`s; summation indices are
   `Dummy`s.**  The package keys its behaviour off this.  A `Dummy`
   used as an external gets renamed by canonicalisation, which
   destroys any output ordering that depends on its identity.  The
   reverse error — `Dummy` where a free index belongs — makes
   `get_a_operator`/`get_b_operator` identically zero for `n >= 2`
   (an antisymmetric product summed over a symmetric range) and
   silently collapses everything built on it.

2. **Direction convention: declare a tensor with `upper` = the indices
   its operator string *annihilates*.**  A de-excitation amplitude
   declared with upper = occupied while the operators annihilate
   virtuals leaves the loop-counting graph edgeless and inflates every
   coefficient by `2**loops`.  The symptom is a suspiciously
   power-of-two coefficient set.

3. **Filter with `get_only_linked` whenever you project.**  An
   unfiltered projection also returns disconnected pieces — a
   self-traced vertex multiplying the rest, with no line joining them.
   These are not small: in one case they inflated an energy by ~10³,
   and by *different* factors in two halves that should have been
   related, which is what made it obvious the error was structural.

4. **Give every dummy a unique name before emitting.**  The generator's
   renamer is name-keyed, and the Wick fallback creates its summation
   dummies with bare names (`i`, `a`).  Two identically named dummies
   in one term collapse into a single einsum subscript, contracting
   indices that should stay separate — with no error raised.

5. **Fix the output ordering yourself when it matters.**  With more
   than about four free indices, or when two of them are related by an
   antisymmetry, letting each emitted term choose its own output order
   makes the axes unidentifiable afterwards — and a misidentified axis
   in an antisymmetric pair flips a sign rather than merely permuting.
   Emit with an explicit external order instead.

6. **`pretty_indices` output is for `latex()` only.**  Feeding renamed
   indices to `generate_einsum` mangles them, and names already in the
   psi4numpy alphabet collapse onto each other.  Pinned as an `xfail`
   in `tests/test_code_generator.py`.

## 5. Open shell

Two routes, and choosing wrongly is a documented trap.

Both routes live in `sym2quantized_sapt/open_shell.py`; the restricted
path in `spin_integrator.py` is untouched by either.

**Spin tags** (`is_alpha` / `is_beta`) are the general route: they are
the monomer-tag mechanism applied to spin, so contractions vanish
across opposite tags and Wick's theorem does the bookkeeping exactly.
Write each operator as its spin sectors and give each resolvent sector
its own normalisation — `1/(n!)**2` per group of same-spin
(indistinguishable) pairs, `1` for distinguishable pairs.

**Per-loop summation** (`spin_integration_uhf`) is the cheap route:
spin is constant along a Goldstone loop, so each term becomes
`2**loops` spin-blocked copies.  It is valid **only** where no
projector carries two or more index pairs in one space.  Applied to a
doubles resolvent it halves the opposite-spin MP2 energy *while
passing its own RHF-collapse gate* — see the warning on the function
and `docs/notes/uhf-spin-summation.md`.

`rhf_collapse` strips the block labels and must reproduce
`spin_integration` exactly.  Use it — but see §7 on what it can and
cannot prove.

## 6. From symbols to numbers

`generate_einsum` emits code that references arrays *by name only*.
`code_generator.array_table(expr)` closes that gap: it returns, for
every array the emitted code mentions, what each of its axes **is** —
role (upper/lower), space (occupied/virtual/general), monomer, and
spin.  Numeric code can then build each array mechanically, which is
what makes a derivation reproducible without hand-transcription.

That table is also the key to density fitting **without** the DF
branches: because both members of a fitted index pair always share
monomer and spin, an ERI operand in an emitted term can be split into
its two three-index factors by the consuming driver, indexed by
`(monomer, spin, space_row, space_col)`.

The `derivation_registry` module (on `feature/term-graphs`) stores a
finished derivation as JSON — every term with its coefficient, its
tensor-network graph, its generated code and a canonical topology key
— so later runs diff against it term by term instead of against a
transcript.

## 7. Validation discipline

The single most important lesson from using this package in anger:

> **A self-consistency gate cannot certify a formula.**

Two real examples, both of which passed their own internal check while
being wrong: an energy-shaped test of the form `<sigma|X sigma>` cannot
pin the normalisation of `X`, because a scalar factor cancels; and the
per-loop spin summation passes `rhf_collapse` on an expression whose
opposite-spin channel is a factor of two too small.

So: gate against an **external** reference (another program, a
published number, an independent reimplementation).  Where you can,
construct a gate in which the approximation is set to zero so
agreement must be at machine precision — for example, validating a
density-fitting code path by feeding it an *exact* factorisation of
the integrals, which tests the factorisation logic with the fitting
error removed.  Record what was checked against what, and treat a
derivation without a numerical stamp as a hypothesis.

## 8. Worked examples

In this repository: `examples/sapt_pol20.py` (canonical),
`examples/sapt_exch10.py` and `examples/sapt_exch-ind200.py`
(exchange machinery), `examples/ump2_ump3_uhf.py` (spin tags, and a
pinned demonstration of the per-loop limitation),
`examples/derivation_registry_demo.py` (registry round trip).

A downstream application drove most of the above and keeps a larger
set of derivation/validation script pairs — a dispersion code with
open-shell MP2/MP3, SAPT dispersion and exchange-dispersion, and a
coupled-pair dispersion method, each with its psi4 gate.  Nothing in
this package assumes that theory; the scripts are useful mainly as
worked examples of the pipeline, the conventions above, and the
symbol-to-number bridge of §6.
