"""
The derivation registry, end to end on E_disp(20).

Derives the second-order dispersion energy, stores it as a registry
record (terms with coefficients, tensor-network graphs, generated
einsum code, canonical keys), lints it, and shows the two everyday
uses: a clean re-derivation diffs silent, and a corrupted one is
reported term by term.
"""

import json
import tempfile
from pathlib import Path

from sym2quantized_sapt.derivation_registry import (
    add_validation,
    build_record,
    diff_records,
    lint_record,
    load_record,
    save_record,
    verify_expr,
)
from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.sapt_utils import get_R_nm, get_V_operator
from sym2quantized_sapt.spin_integrator import spin_integration

# ---- derive, as usual -------------------------------------------------
E_disp_20 = wicks_double_vac(
    get_V_operator() * get_R_nm(1, 1, get_V_operator()),
    keep_only_fully_contracted=True,
)
E_disp_20 = spin_integration(E_disp_20)

# ---- store ------------------------------------------------------------
record = build_record(
    "e_disp20",
    E_disp_20,
    description="Second-order SAPT dispersion energy, RHF.",
    generator={"script": Path(__file__).name},
)
add_validation(
    record,
    check="reproduces the pinned code_generator reference string",
    result="exact",
)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "e_disp20.json"
    save_record(record, path)
    stored = load_record(path)
    print(f"stored {stored['n_terms']} term(s) in {path.name}:")
    for entry in stored["terms"]:
        print(f"  coefficient {entry['coefficient']}"
              f"  loops {entry['loops']}"
              f"  wick fraction {entry['wick_fraction']}")
        print(f"  {entry['einsum']}")
        print(f"  key {entry['key'][:64]}...")

findings = lint_record(stored)
print(f"\nlint: {findings if findings else 'clean'}")

# ---- a clean re-derivation diffs silent -------------------------------
print("\nre-derivation vs stored record:", verify_expr(stored, E_disp_20))

# ---- a corrupted one is reported, term by term ------------------------
corrupted = build_record("e_disp20", 2 * E_disp_20, code=False)
report = diff_records(stored, corrupted)
print("doubled-coefficient rederivation:",
      json.dumps(report["changed"], default=str)[:100], "...")
assert report["changed"] and not report["added"] and not report["removed"]
assert not findings
assert verify_expr(stored, E_disp_20)["changed"] == []
