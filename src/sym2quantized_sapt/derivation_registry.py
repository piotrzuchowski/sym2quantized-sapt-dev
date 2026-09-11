"""A persistent, diffable record of a derivation's terms.

A derivation script prints its result and exits; the expression is
gone, and whoever implements, reviews, or later *re-derives* it works
against a transcript.  This module stores the result instead: one JSON
record per derived object, each term carried in three synchronized
forms -- its coefficient, its tensor-network graph
(:func:`term_graphs.to_dict`), and its generated ``np.einsum`` line --
together with a canonical key that names the term stably across runs,
dummy renamings, and factor reorderings.

What that buys, in order of practical weight:

* **Term-level diffs.**  Re-deriving after a package change and calling
  :func:`diff_records` (or :func:`verify_expr` against the live
  expression) reports which terms appeared, vanished, or changed
  coefficient -- keyed by topology, so a mere reordering is silence,
  and a real change is named.
* **Structural lint.**  :func:`lint_record` checks the facts a correct
  derivation cannot violate: every term of one projection shares one
  external-index pattern, terms are connected, no two terms share a
  topology (upstream canonicalization should have merged them), and no
  coefficient is suspiciously large relative to ``2**loops`` -- the
  signature of a mis-declared operator direction inflating
  coefficients by powers of two.
* **Provenance and validation stamps.**  A record without a numerical
  validation stamp is a hypothesis; :func:`add_validation` appends
  what was checked, against what, and how well it agreed.  Topology
  can never pin a normalisation -- scalar factors are invisible to a
  graph -- so the stamp field is first-class, not an afterthought.

The format is plain JSON with string coefficients; nothing here
assumes any particular theory, only the package's own conventions
(terms as ``Mul`` of a number and ``TensorSymbol`` factors, canonical
dummies).  See ``examples/derivation_registry_demo.py`` for the
end-to-end flow on the dispersion energy.
"""

import datetime
import json
from collections import Counter

from sympy import Add, nsimplify

from sym2quantized_sapt.code_generator import generate_einsum
from sym2quantized_sapt.term_graphs import (
    canonical_key,
    expr_to_graphs,
    to_dict,
)

__all__ = [
    "add_validation",
    "build_record",
    "diff_records",
    "lint_record",
    "load_record",
    "save_record",
    "verify_expr",
]

FORMAT = "sym2quantized-sapt derivation record v1"


def _wick_fraction(graph):
    """The coefficient with the RHF spin factor divided out."""
    return nsimplify(graph.coefficient) / 2**graph.loops


def build_record(
    name,
    expr,
    description="",
    generator=None,
    code=True,
    density_fitting=False,
):
    """Encode every term of ``expr`` into a registry record (a dict).

    ``expr`` is a finished derivation: an ``Add`` of fully processed
    terms (or a single term) with canonical dummy names, as produced by
    the usual pipeline ending in ``substitute_dummies_double_vac`` and
    ``spin_integration``.  ``generator`` is a free-form dict recording
    where the expression came from (script, package commit, options);
    the build date is added for you.  ``code=False`` skips the einsum
    lines (e.g. for expressions whose index names the code generator
    would mangle); ``density_fitting`` is passed through to
    :func:`code_generator.generate_einsum` per term.
    """
    terms = list(expr.args) if isinstance(expr, Add) else [expr]
    graphs = expr_to_graphs(expr)

    entries = []
    for term, graph in zip(terms, graphs):
        entry = {
            "key": canonical_key(graph),
            "coefficient": str(nsimplify(graph.coefficient)),
            "loops": graph.loops,
            "wick_fraction": str(_wick_fraction(graph)),
            "graph": to_dict(graph),
        }
        if code and density_fitting:
            # only builds of code_generator that know the option
            # (the density_fitting feature branch) accept the kwarg
            try:
                entry["einsum"] = generate_einsum(
                    term, density_fitting=True
                )
            except TypeError as exc:
                raise ValueError(
                    "this build of code_generator has no "
                    "density_fitting support"
                ) from exc
        elif code:
            entry["einsum"] = generate_einsum(term)
        entries.append(entry)

    return {
        "format": FORMAT,
        "name": name,
        "description": description,
        "created": datetime.date.today().isoformat(),
        "generator": dict(generator or {}),
        "conventions": {
            "canonical_key": (
                "term_graphs v1: exact node relabelling; tensor "
                "permutation symmetries NOT folded in"
            ),
            "coefficient": "includes the RHF spin factor 2**loops",
        },
        "n_terms": len(entries),
        "terms": entries,
        "validation": [],
    }


def save_record(record, path):
    """Write ``record`` to ``path`` as indented, diff-friendly JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=False)
        handle.write("\n")


def load_record(path):
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a derivation record "
            f"(format {record.get('format')!r})."
        )
    return record


def add_validation(record, check, result, **extra):
    """Append a validation stamp: *what* was checked, against *what*
    reference, with what outcome -- e.g.
    ``add_validation(rec, "vs dense reference implementation",
    "rel diff 3e-16", system="He2")``."""
    stamp = {
        "check": check,
        "result": result,
        "date": datetime.date.today().isoformat(),
    }
    stamp.update(extra)
    record.setdefault("validation", []).append(stamp)
    return record


# --- comparison ---------------------------------------------------------


def _by_key(record):
    """key -> summed coefficient (records may legitimately carry a
    topology once; summing makes the diff robust if they do not)."""
    totals = {}
    for entry in record["terms"]:
        key = entry["key"]
        totals[key] = totals.get(key, 0) + nsimplify(entry["coefficient"])
    return totals


def diff_records(old, new):
    """Term-level difference, keyed by canonical topology.

    Returns a dict with ``added`` / ``removed`` (lists of keys),
    ``changed`` (list of ``(key, old_coefficient, new_coefficient)``)
    and ``unchanged`` (count).  Two records that merely reordered or
    renamed their dummies diff clean.
    """
    old_terms, new_terms = _by_key(old), _by_key(new)
    added = sorted(set(new_terms) - set(old_terms))
    removed = sorted(set(old_terms) - set(new_terms))
    changed, unchanged = [], 0
    for key in sorted(set(old_terms) & set(new_terms)):
        if old_terms[key] != new_terms[key]:
            changed.append((key, str(old_terms[key]), str(new_terms[key])))
        else:
            unchanged += 1
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


def verify_expr(record, expr):
    """Diff a stored record against a live expression.

    The everyday guard: the implementation carries the stored record,
    a rerun of the derivation carries ``expr``, and a clean result
    (``added == removed == changed == []``) says the code still
    matches the algebra.
    """
    fresh = build_record(record["name"], expr, code=False)
    return diff_records(record, fresh)


# --- lint ----------------------------------------------------------------


def lint_record(record, max_wick_numerator=4):
    """Structural checks a correct derivation cannot fail.

    Returns a list of human-readable findings (empty = clean):

    * mixed external patterns -- terms of one projection must share
      their open indices;
    * a disconnected term -- linked-diagram output should have none;
    * duplicate topologies -- upstream canonicalization should have
      merged them into one coefficient;
    * a Wick fraction (coefficient / 2**loops) whose numerator exceeds
      ``max_wick_numerator`` -- honest Wick coefficients are small
      rationals, and powers of two beyond the loop count are the
      classic signature of a mis-declared operator direction.
    """
    findings = []

    patterns = Counter(
        entry["graph"]["invariants"]["external_pattern"]
        for entry in record["terms"]
    )
    if len(patterns) > 1:
        findings.append(
            f"mixed external patterns: {dict(sorted(patterns.items()))}"
        )

    for position, entry in enumerate(record["terms"]):
        if not entry["graph"]["invariants"]["connected"]:
            findings.append(f"term {position} ({entry['key'][:40]}...) "
                            "is disconnected")

    keys = Counter(entry["key"] for entry in record["terms"])
    for key, count in sorted(keys.items()):
        if count > 1:
            findings.append(
                f"topology appears {count} times unmerged: {key[:60]}..."
            )

    for position, entry in enumerate(record["terms"]):
        fraction = nsimplify(entry["wick_fraction"])
        numerator = abs(fraction.p if fraction.is_Rational else fraction)
        if numerator > max_wick_numerator:
            findings.append(
                f"term {position}: wick fraction {entry['wick_fraction']} "
                f"(coefficient {entry['coefficient']}, "
                f"loops {entry['loops']}) -- suspiciously large"
            )

    return findings
