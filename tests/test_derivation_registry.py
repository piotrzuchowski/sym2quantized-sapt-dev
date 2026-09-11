import json

import pytest
from sympy import symbols

from sym2quantized_sapt.derivation_registry import (
    add_validation,
    build_record,
    diff_records,
    lint_record,
    load_record,
    save_record,
    verify_expr,
)
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def _indices(suffix=""):
    a = symbols("a" + suffix, is_molA=True, above_fermi=True)
    i = symbols("i" + suffix, is_molA=True, below_fermi=True)
    b = symbols("b" + suffix, is_molB=True, above_fermi=True)
    j = symbols("j" + suffix, is_molB=True, below_fermi=True)
    return a, i, b, j


def _disp_term(suffix=""):
    a, i, b, j = _indices(suffix)
    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    return t * v


def test_build_and_roundtrip(tmp_path):
    record = build_record(
        "demo", 4 * _disp_term(), description="a demo",
        generator={"script": "here"},
    )

    assert record["n_terms"] == 1
    (entry,) = record["terms"]
    assert entry["coefficient"] == "4"
    assert entry["einsum"].lstrip("+").startswith("4 * np.einsum")
    assert entry["graph"]["invariants"]["tensors"] == ["t", "v"]

    path = tmp_path / "demo.json"
    save_record(record, path)
    assert load_record(path) == json.loads(path.read_text())


def test_load_rejects_a_foreign_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"anything": 1}')

    with pytest.raises(ValueError, match="not a derivation record"):
        load_record(path)


def test_identical_rederivation_diffs_clean():
    record = build_record("demo", 4 * _disp_term())

    # renamed dummies, reordered factors: still silence
    report = verify_expr(record, 4 * _disp_term("_7"))

    assert report == {
        "added": [], "removed": [], "changed": [], "unchanged": 1
    }


def test_changed_coefficient_is_reported():
    old = build_record("demo", 4 * _disp_term())
    new = build_record("demo", 2 * _disp_term())

    report = diff_records(old, new)

    assert report["added"] == [] and report["removed"] == []
    ((_, before, after),) = report["changed"]
    assert (before, after) == ("4", "2")


def test_vanished_term_is_reported():
    a, i, b, j = _indices()
    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v_direct = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    v_crossed = DoubleVacuumTensorSymbol("v", (a, b), (j, i))

    old = build_record("demo", 4 * t * v_direct - 2 * t * v_crossed)
    new = build_record("demo", 4 * t * v_direct)

    report = diff_records(old, new)

    assert len(report["removed"]) == 1
    assert report["added"] == [] and report["changed"] == []


def test_lint_is_clean_on_an_honest_record():
    assert lint_record(build_record("demo", 4 * _disp_term())) == []


def test_lint_flags_an_inflated_coefficient():
    # 16 on a two-loop term leaves a wick fraction of 4 ... times 4
    # more is the direction-convention signature
    findings = lint_record(build_record("demo", 64 * _disp_term()))

    assert len(findings) == 1
    assert "suspiciously large" in findings[0]


def test_lint_flags_unmerged_duplicate_topologies():
    expr = 4 * _disp_term() - 2 * _disp_term("_9")

    findings = lint_record(build_record("demo", expr))

    assert len(findings) == 1
    assert "unmerged" in findings[0]


def test_validation_stamps_accumulate():
    record = build_record("demo", 4 * _disp_term())

    add_validation(record, "vs reference", "exact", system="model")

    (stamp,) = record["validation"]
    assert stamp["check"] == "vs reference"
    assert stamp["system"] == "model"
    assert "date" in stamp
