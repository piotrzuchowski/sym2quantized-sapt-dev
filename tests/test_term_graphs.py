import json

import pytest
from sympy import Rational, symbols

from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.term_graphs import (
    canonical_key,
    classify,
    common_pairs,
    contraction_cost,
    expr_to_graphs,
    invariants,
    term_to_graph,
    to_dict,
    to_dot,
)


def _disp20_indices(suffix=""):
    a = symbols("a" + suffix, is_molA=True, above_fermi=True)
    i = symbols("i" + suffix, is_molA=True, below_fermi=True)
    b = symbols("b" + suffix, is_molB=True, above_fermi=True)
    j = symbols("j" + suffix, is_molB=True, below_fermi=True)
    return a, i, b, j


def _disp20_term(suffix=""):
    """the E_disp(20)-shaped scalar `t^{i j}_{a b} v^{a b}_{i j}`"""
    a, i, b, j = _disp20_indices(suffix)
    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    return t * v


def test_encodes_a_fully_contracted_term():
    graph = term_to_graph(4 * _disp20_term())

    assert graph.coefficient == 4
    assert tuple(n.name for n in graph.nodes) == ("t", "v")
    assert len(graph.edges) == 4
    assert graph.externals == ()


def test_free_indices_become_externals():
    a, i, b, j = _disp20_indices()
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    s = DoubleVacuumTensorSymbol("s", (b,), (i,))

    graph = term_to_graph(v * s)

    # a and j touch only v; b and i are shared
    assert len(graph.edges) == 2
    assert sorted(x.name for x in graph.externals) == ["a", "j"]


def test_canonical_key_ignores_dummy_names():
    key1 = canonical_key(term_to_graph(_disp20_term()))
    key2 = canonical_key(term_to_graph(_disp20_term("_9")))

    assert key1 == key2


def test_canonical_key_separates_different_wirings():
    a, i, b, j = _disp20_indices()
    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v_direct = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    # same tensors, different contraction: v's monomer-A slots swapped
    v_crossed = DoubleVacuumTensorSymbol("v", (a, b), (j, i))

    key_direct = canonical_key(term_to_graph(t * v_direct))
    key_crossed = canonical_key(term_to_graph(t * v_crossed))

    assert key_direct != key_crossed


def test_classify_groups_equal_topologies():
    expr = 4 * _disp20_term() - 2 * _disp20_term("_9")

    buckets = classify(expr)

    assert len(buckets) == 1
    (graphs,) = buckets.values()
    assert sorted(str(g.coefficient) for g in graphs) == ["-2", "4"]


def test_loops_agree_with_spin_integration():
    term = _disp20_term()

    integrated = spin_integration(term)
    graph = expr_to_graphs(integrated)[0]

    # the factor spin_integration multiplied in is exactly 2**loops
    assert graph.coefficient == 2**graph.loops


def test_invariants_of_a_disp20_term():
    facts = invariants(term_to_graph(_disp20_term()))

    assert facts["tensors"] == ("t", "v")
    assert facts["n_contracted"] == 4
    assert facts["n_external"] == 0
    assert facts["connected"] is True
    assert facts["external_pattern"] == ""


def test_contraction_cost_of_disp20_is_ov_squared():
    cost = contraction_cost(term_to_graph(_disp20_term()))

    # one GEMM over all four dimensions; nothing survives
    assert cost["flops"] == {"Ao": 1, "Av": 1, "Bo": 1, "Bv": 1}
    assert cost["memory"] == {}


def test_common_pairs_counts_recurring_contractions():
    expr = 4 * _disp20_term() - 2 * _disp20_term("_9")

    pairs = common_pairs(expr_to_graphs(expr))

    assert len(pairs) == 1
    count, _, example = pairs[0]
    assert count == 2
    assert tuple(n.name for n in example.nodes) == ("t", "v")


def test_an_index_in_three_slots_is_a_hyperedge():
    # the resolvent denominator shares every index with the amplitudes
    # it divides - E_disp(20) is `e * v * v` with each index on three
    # slots, and the encoding must carry that as one hyperedge
    a, i, b, j = _disp20_indices()
    e = DoubleVacuumTensorSymbol("e", (i, j), (a, b))
    v_ket = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    v_bra = DoubleVacuumTensorSymbol("v", (i, j), (a, b))

    graph = term_to_graph(e * v_ket * v_bra)

    assert len(graph.edges) == 4
    assert all(len(edge.ends) == 3 for edge in graph.edges)
    assert graph.externals == ()
    assert invariants(graph)["connected"] is True

    # one Hadamard evaluation: cost is the four dimensions, once
    cost = contraction_cost(graph)
    assert cost["flops"] == {"Ao": 1, "Av": 1, "Bo": 1, "Bv": 1}


def test_exports_are_serializable_and_render():
    graph = term_to_graph(Rational(1, 2) * _disp20_term())

    payload = to_dict(graph)
    json.dumps(payload)
    assert payload["coefficient"] == "1/2"
    assert payload["invariants"]["tensors"] == ["t", "v"]

    dot = to_dot(graph, name="disp20")
    assert dot.startswith('graph "disp20"')
    assert 't0 -- t1' in dot
