import pytest
import string

from sympy import symbols, Dummy, Rational
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.double_fermi_vac import (
    wicks_double_vac,
    substitute_dummies_double_vac,
    CreateFermion_A,
    CreateFermion_B,
    AnnihilateFermion_A,
    AnnihilateFermion_B,
)
from sym2quantized_sapt.sapt_utils import get_V_operator, get_R_nm
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.code_generator import generate_einsum


def test_single_self_contraction():
    reference = """+np.einsum("kk", A_kk)"""

    k = symbols("k", below_fermi=True)
    A = DoubleVacuumTensorSymbol("A", (k,), (k,))

    tested_expr = generate_einsum(A)

    assert reference == tested_expr


def test_doucle_self_contraction():
    reference = """+np.einsum("klkl", A_klkl)"""

    k, l = symbols("k l", below_fermi=True)
    A = DoubleVacuumTensorSymbol(
        "A",
        (
            k,
            l,
        ),
        (
            k,
            l,
        ),
    )

    tested_expr = generate_einsum(A)

    assert reference == tested_expr


def test_partial_self_contraction():
    reference = """+np.einsum("kmkl->ml", A_kmkl)"""

    k, l, m = symbols("k l m", below_fermi=True)
    A = DoubleVacuumTensorSymbol(
        "A",
        (
            k,
            l,
        ),
        (
            k,
            m,
        ),
    )

    tested_expr = generate_einsum(A)

    assert reference == tested_expr


def test_two_eris():
    reference = """-np.einsum("abrs,rsab", v_abrs, v_rsab)"""

    a = symbols("a", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    v_ijab = DoubleVacuumTensorSymbol(
        "v",
        (
            a,
            b,
        ),
        (
            i,
            j,
        ),
    )
    v_abij = DoubleVacuumTensorSymbol(
        "v",
        (
            i,
            j,
        ),
        (
            a,
            b,
        ),
    )

    tested_expr = generate_einsum((-1.0) * v_ijab * v_abij)

    assert reference == tested_expr


def test_matrix_multiplication():
    reference = """+np.einsum("kl,lm->km", A_kl, B_lm)"""

    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str


def test_index_name_collision():
    # `p_1` must not be renamed to `c`, which is already taken by another index
    reference = """+np.einsum("cd,dc", A_cp, B_pc)"""

    c, p_1 = symbols("c p_1")

    A = DoubleVacuumTensorSymbol("A", (p_1,), (c,))
    B = DoubleVacuumTensorSymbol("B", (c,), (p_1,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str


@pytest.mark.xfail(
    strict=True,
    reason="`pretty_indices` already names the dummies in the psi4numpy "
    "alphabet, and `generate_einsum` renames them a second time: the "
    "particle `r` stays `r` while the hole `a` also becomes `r`, so the "
    "four indices collapse to two and the einsum contracts the wrong pairs",
)
def test_psi4_indices_name_collision():
    # NOTE: pretty_indices in dummies substitution is not compatible with
    # code generation - see the docstring of `substitute_dummies_double_vac`
    reference = """+np.einsum("rsab,abrs", t_rsab, v_abrs)"""

    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    b = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

    T11 = (
        DoubleVacuumTensorSymbol("t", (i, j), (a, b))
        * CreateFermion_A(a)
        * AnnihilateFermion_A(i)
        * CreateFermion_B(b)
        * AnnihilateFermion_B(j)
    )
    V = get_V_operator()

    expr = V * T11
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    expr = substitute_dummies_double_vac(
        expr,
        pretty_indices={
            "above_molA": "r",
            "above_molB": "s",
            "below_molA": "a",
            "below_molB": "b",
        },
    )

    tested_expr = generate_einsum(expr)

    assert reference == tested_expr


def test_multidigit_index_names():
    # `p_1` must not be substituted inside `p_10`
    reference = """+np.einsum("cd,dc", A_pp, B_pp)"""

    p_1, p_10 = symbols("p_1 p_10")

    A = DoubleVacuumTensorSymbol("A", (p_1,), (p_10,))
    B = DoubleVacuumTensorSymbol("B", (p_10,), (p_1,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str


def test_too_many_indices():
    expr = 1.0
    for char in string.ascii_lowercase:
        idx_next = symbols(char)
        for i in range(1, 10):
            idx_previous = idx_next
            idx_next = symbols(f"{char}_{i}")
            expr *= DoubleVacuumTensorSymbol("X", (idx_previous,), (idx_next,))

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(expr)

    assert exec_info.type == IndexError
    assert (
        exec_info.value.args[0]
        == "Too many indices!!! Not enough names for them."
    )


def test_pretty_indices():
    reference = """+np.einsum("Pp,qP,Qq,Rr,sR,Ss,aS,Aa,bA,Bb,pB->Qr", X_pp, X_qp, X_qq, X_rr, X_sr, X_ss, X_as, X_aa, X_ba, X_bb, X_pb)"""

    a, a_1 = symbols("a a_1", is_molA=True, above_fermi=True)
    i, i_1 = symbols("i i_1", is_molA=True, below_fermi=True)

    b, b_1 = symbols("b b_1", is_molB=True, above_fermi=True)
    j, j_1 = symbols("j j_1", is_molB=True, below_fermi=True)

    p, p_1 = symbols("p p_1", is_molA=True)
    q, q_1 = symbols("q q_1", is_molB=True)

    expr = (
        DoubleVacuumTensorSymbol("X", (a,), (a_1,))
        * DoubleVacuumTensorSymbol("X", (a_1,), (b,))
        * DoubleVacuumTensorSymbol("X", (b,), (b_1,))
        * DoubleVacuumTensorSymbol("X", (b_1,), (i,))
        * DoubleVacuumTensorSymbol("X", (i,), (i_1,))
        * DoubleVacuumTensorSymbol("X", (i_1,), (j,))
        * DoubleVacuumTensorSymbol("X", (j,), (j_1,))
        * DoubleVacuumTensorSymbol("X", (j_1,), (p,))
        * DoubleVacuumTensorSymbol("X", (p,), (p_1,))
        * DoubleVacuumTensorSymbol("X", (p_1,), (q,))
        * DoubleVacuumTensorSymbol("X", (q,), (q_1,))
    )
    tested_str = generate_einsum(expr, pretty_indices=True)

    assert reference == tested_str


def _matrix_product(coeff):
    """`coeff * A^{k}_{l} B^{l}_{m}`, the expression the coefficient
    formatting tests share"""
    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    return coeff * A * B


def test_integral_float_coefficient():
    # An integral Float is printed as an int, which relies on
    # `Float(2.0) == 2`. That holds up to sympy 1.12 and is False from
    # sympy 1.13 on, where the coefficient comes out as
    # "2.00000000000000" instead. This assertion is the tripwire for the
    # `sympy<1.13` cap in setup.py - see the version sensitivity note in
    # CLAUDE.md.
    reference = """+2 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(2.0))

    assert reference == tested_str


def test_negative_integral_float_coefficient():
    # a negative coefficient already carries its sign, no plus is prepended
    reference = """-2 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(-2.0))

    assert reference == tested_str


def test_non_integral_float_coefficient():
    reference = """+0.500000000000000 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(0.5))

    assert reference == tested_str


def test_rational_coefficient():
    reference = """+1/2 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(Rational(1, 2)))

    assert reference == tested_str


def test_sum_of_terms():
    # every term of an Add gets a line of its own
    reference = (
        '+np.einsum("kl,lm->km", A_kl, B_lm)\n' '+np.einsum("km->km", C_km)'
    )

    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))
    C = DoubleVacuumTensorSymbol("C", (m,), (k,))

    tested_str = generate_einsum(A * B + C)

    assert reference == tested_str


def test_pretty_indices_for_single_tensor():
    # the `pretty_indices` route for a bare tensor: `a_1` is renamed to
    # `r_1` by the psi4numpy mapping and then to `R` by the pretty table.
    # The variable name drops the subscript, exactly as the `Mul` route
    # does - see test_tensor_and_mul_routes_agree_on_variable_names.
    reference = """+np.einsum("Rsab->Rsab", t_rsab)"""

    a_1 = symbols("a_1", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    t = DoubleVacuumTensorSymbol("t", (i, j), (a_1, b))

    tested_str = generate_einsum(t, pretty_indices=True)

    assert reference == tested_str


def test_pretty_indices_rejects_index_outside_the_table():
    # `k` and `l` survive the psi4numpy mapping unchanged and are not in
    # `PRETTY_INDICES`, so the table cannot be applied
    k, l = symbols("k l")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (k,), (l,))

    with pytest.raises(ValueError) as exec_info:
        generate_einsum(A * B, pretty_indices=True)

    assert (
        exec_info.value.args[0]
        == "Code generator: `pretty_indices` cannot be applied!"
    )


@pytest.mark.xfail(
    strict=True,
    reason="a factor `generate_einsum` cannot translate is dropped "
    "silently: the fallback returns an empty string, so the term "
    "disappears from the generated code (leaving a blank line) instead of "
    "raising. The exception type asserted below is a proposal - whoever "
    "fixes this picks it",
)
def test_unsupported_term_is_not_dropped_silently():
    # `x` carries no tensor, so `generate_einsum` returns "" for it and the
    # term vanishes from the sum - the generated code is then quietly wrong
    k, l, m, x = symbols("k l m x")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    with pytest.raises((TypeError, ValueError)):
        generate_einsum(A * B + x)


def test_tensor_and_mul_routes_agree_on_variable_names():
    """A lone tensor and the same tensor inside a product have to name
    their array identically. The two routes build the name separately, so
    the subscript stripping has to be kept in step in both of them."""
    a_1 = symbols("a_1", is_molA=True, above_fermi=True)
    i_1 = symbols("i_1", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    t = DoubleVacuumTensorSymbol("t", (i_1, j), (a_1, b))

    bare = generate_einsum(t)
    in_a_product = generate_einsum(2.0 * t)

    assert '+np.einsum("csdb->csdb", t_rsab)' == bare
    assert '+2 * np.einsum("csdb->csdb", t_rsab)' == in_a_product


# --------------------------------------------------------------------------
# density fitting
#
# `generate_einsum(..., density_fitting=True)` replaces every intermolecular
# two-electron integral `v^{p r}_{q s} = (p q | r s)` with its factorization
# `sum_Q B^{Q}_{q p} B^{Q}_{s r}`, emitting the three-index arrays `Qqp` and
# `Qsr` in place of `v_qspr`. See `docs/notes/density-fitting.md`.
# --------------------------------------------------------------------------


def _sapt_indices():
    """the four canonical dummies of a doubly excited SAPT term"""
    a = symbols("a", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    return a, i, b, j


def test_density_fitting_splits_an_eri_into_two_three_index_arrays():
    # the A indices land in one array and the B indices in the other, with
    # the psi4numpy letters: A occupied `a` / virtual `r`, B occupied `b` /
    # virtual `s`
    reference = """+np.einsum("rsab,Qar,Qbs", t_rsab, Qar, Qbs)"""

    a, i, b, j = _sapt_indices()

    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    tested_str = generate_einsum(t * v, density_fitting=True)

    assert reference == tested_str


def test_density_fitting_gives_each_eri_its_own_auxiliary_index():
    # Two ERIs in one term are two independent auxiliary sums,
    # `(sum_Q B B)(sum_P B B)`. Sharing one label would contract all four
    # three-index arrays against the same auxiliary index - a different, and
    # wrong, quantity. The array names keep the `Q` prefix either way: it
    # spells out `B^{Q}`, it is not the subscript.
    reference = (
        """+4 * np.einsum("rsab,Qar,Qbs,Pra,Psb", """
        """e_rsab, Qar, Qbs, Qra, Qsb)"""
    )

    a, i, b, j = _sapt_indices()

    e = DoubleVacuumTensorSymbol("e", (i, j), (a, b))
    v_abij = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    v_ijab = DoubleVacuumTensorSymbol("v", (i, j), (a, b))

    tested_str = generate_einsum(
        4.0 * e * v_abij * v_ijab, density_fitting=True
    )

    assert reference == tested_str


def test_density_fitting_on_derived_e_disp_20():
    # the same term as above, this time derived rather than hand-built: the
    # end-to-end check that the pipeline hands `generate_einsum` an
    # expression it factorizes correctly
    reference = (
        """+4 * np.einsum("rsab,Qar,Qbs,Pra,Psb", """
        """e_rsab, Qar, Qbs, Qra, Qsb)"""
    )

    E_disp_20 = wicks_double_vac(
        get_V_operator() * get_R_nm(1, 1, get_V_operator()),
        keep_only_fully_contracted=True,
    )
    E_disp_20 = spin_integration(E_disp_20)

    assert (
        '+4 * np.einsum("rsab,abrs,rsab", e_rsab, v_abrs, v_rsab)'
        == generate_einsum(E_disp_20)
    )
    assert reference == generate_einsum(E_disp_20, density_fitting=True)


def test_density_fitting_leaves_non_eri_tensors_alone():
    # only `v` is factorized; the overlap `s`, the monomer potential
    # `(v_A)` and the amplitudes come out exactly as they do without the
    # option
    a, i, b, j = _sapt_indices()

    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    s = DoubleVacuumTensorSymbol("s", (b,), (i,))
    vA = DoubleVacuumTensorSymbol("(v_A)", (a,), (j,))

    expr = t * s * vA

    assert generate_einsum(expr) == generate_einsum(expr, density_fitting=True)


def test_density_fitting_auxiliary_index_is_summed_not_returned():
    # the auxiliary index is contracted away inside the ERI, so it must not
    # reach the einsum output - otherwise the term gains a free index over
    # the auxiliary basis
    reference = """+np.einsum("as,Qar,Qbs->asabrs", s_as, Qar, Qbs)"""

    a, i, b, j = _sapt_indices()

    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    s = DoubleVacuumTensorSymbol("s", (b,), (i,))

    tested_str = generate_einsum(v * s, density_fitting=True)

    assert reference == tested_str

    subscript = tested_str.split('"')[1]
    assert "Q" not in subscript.split("->")[1]


def test_density_fitting_for_a_bare_tensor():
    # a lone tensor takes the `_get_einsum_for_Tensor` route; density
    # fitting has to work there too. A tensor that is not an ERI is
    # untouched by the option.
    a, i, b, j = _sapt_indices()

    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    assert generate_einsum(t) == generate_einsum(t, density_fitting=True)
    assert '+np.einsum("Qar,Qbs->abrs", Qar, Qbs)' == generate_einsum(
        v, density_fitting=True
    )


def test_density_fitting_in_a_sum():
    # an `Add` mixing a product with a bare tensor has to generate both
    # lines - the bare-tensor route must not take the whole expression down
    reference = (
        '+np.einsum("rsab,Qar,Qbs", t_rsab, Qar, Qbs)\n'
        '+np.einsum("Qar,Qbs->abrs", Qar, Qbs)'
    )

    a, i, b, j = _sapt_indices()

    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    tested_str = generate_einsum(t * v + v, density_fitting=True)

    assert reference == tested_str


def test_density_fitting_rejects_an_eri_without_four_indices():
    # the factorization needs a `(p q | r s)`; anything else is not an ERI
    # this code knows how to split
    a, i, b, j = _sapt_indices()

    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a,), (i,))

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(t * v, density_fitting=True)

    assert exec_info.value.args[0] == (
        "Code generator: density fitting expected 4 indices "
        "in tensor v((a,),(i,)), got 2."
    )

    # and from the bare-tensor route as well
    with pytest.raises(IndexError):
        generate_einsum(v, density_fitting=True)


def test_density_fitting_rejects_an_eri_that_is_not_two_by_two():
    # four indices are not enough on their own: the split takes one upper
    # and one lower index per monomer, so they have to be spread 2 and 2
    a, i, b, j = _sapt_indices()

    v = DoubleVacuumTensorSymbol("v", (a, b, i), (j,))

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(v, density_fitting=True)

    assert exec_info.value.args[0] == (
        "Code generator: density fitting expected 2 upper and 2 lower "
        "indices in tensor v((a, b, i),(j,)), got 3 and 1."
    )


def test_density_fitting_rejects_an_eri_that_is_not_monomer_ordered():
    # the split pairs index slots positionally, so it is only the right
    # factorization when each (lower, upper) slot pair sits on one monomer.
    # `v^{a b}_{j i}` would silently give `Qbr`, `Qas` - three-index arrays
    # straddling both monomers.
    a, i, b, j = _sapt_indices()

    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (j, i))

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(t * v, density_fitting=True)

    assert exec_info.value.args[0] == (
        "Code generator: density fitting expected each pair of indices "
        "of tensor v((a, b),(j, i)) on one monomer, got BA and AB."
    )


def test_density_fitting_keeps_positional_pairing_without_monomer_tags():
    # a plain ERI carrying no monomer assumption is not second-guessed
    k, l, m, n = symbols("k l m n")

    v = DoubleVacuumTensorSymbol("v", (k, m), (l, n))

    assert '+np.einsum("Qlk,Qnm->lnkm", Qlk, Qnm)' == generate_einsum(
        v, density_fitting=True
    )


def test_density_fitting_with_pretty_indices():
    # `pretty_indices` renames `p_1` to `P` and `q_1` to `Q`, both of which
    # the auxiliary index would otherwise want. It is named last, out of the
    # letters the subscript has not already spent, so it steps aside to `R`.
    reference = """+np.einsum("pQPq,RPp,RqQ", t_pqpq, Qpp, Qqq)"""

    p, p_1 = symbols("p p_1", is_molA=True)
    q, q_1 = symbols("q q_1", is_molB=True)

    t = DoubleVacuumTensorSymbol("t", (p_1, q), (p, q_1))
    v = DoubleVacuumTensorSymbol("v", (p, q_1), (p_1, q))

    tested_str = generate_einsum(
        t * v, pretty_indices=True, density_fitting=True
    )

    assert reference == tested_str


def test_density_fitting_routes_agree_on_variable_names():
    """The bare-tensor and the product route name the density-fitted arrays
    separately; a `v` has to give the same pair of arrays either way."""
    a, i, b, j = _sapt_indices()

    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    bare = generate_einsum(v, density_fitting=True)
    in_a_product = generate_einsum(2.0 * v, density_fitting=True)

    assert '+np.einsum("Qar,Qbs->abrs", Qar, Qbs)' == bare
    assert '+2 * np.einsum("Qar,Qbs->abrs", Qar, Qbs)' == in_a_product


def _many_eris(count, padding=0):
    """A product of `count` ERIs and `padding` amplitudes, none of them
    sharing an index. Every index is subscripted, so each amplitude costs
    the renamer four more letters."""
    expr = 1.0
    for n in range(count):
        a = symbols(f"a_{n}", is_molA=True, above_fermi=True)
        i = symbols(f"i_{n}", is_molA=True, below_fermi=True)
        b = symbols(f"b_{n}", is_molB=True, above_fermi=True)
        j = symbols(f"j_{n}", is_molB=True, below_fermi=True)
        expr *= DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    for n in range(count, count + padding):
        a = symbols(f"a_{n}", is_molA=True, above_fermi=True)
        i = symbols(f"i_{n}", is_molA=True, below_fermi=True)
        b = symbols(f"b_{n}", is_molB=True, above_fermi=True)
        j = symbols(f"j_{n}", is_molB=True, below_fermi=True)
        expr *= DoubleVacuumTensorSymbol("t", (i, j), (a, b))

    return expr


def test_density_fitting_runs_out_of_auxiliary_indices():
    # one auxiliary index per ERI, so a term eventually exhausts the pool.
    # It has to say so rather than reuse a label and fuse two ERIs into one
    # auxiliary sum.
    assert generate_einsum(_many_eris(8), density_fitting=True)

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(_many_eris(9), density_fitting=True)

    assert exec_info.value.args[0] == (
        "Too many ERIs!!! Not enough auxiliary indices for them."
    )


def test_density_fitting_auxiliary_indices_can_be_starved_by_ordinary_ones():
    # the auxiliary indices are named last, out of the letters the ordinary
    # indices have not taken, so a term can run out of them without holding
    # more ERIs than there are sentinels. It has to say so rather than reuse
    # a label.
    assert generate_einsum(_many_eris(8, padding=2), density_fitting=True)

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(_many_eris(8, padding=3), density_fitting=True)

    assert exec_info.value.args[0] == (
        "Too many ERIs!!! Not enough auxiliary indices for them."
    )


@pytest.mark.xfail(
    strict=True,
    reason="a squared ERI is a `Pow`, not a `Mul` of two `TensorSymbol`s, "
    "so no argument of the term is recognised and the factor is dropped: "
    'the line comes out as `+2 * np.einsum("", )`. Same silent-drop '
    "defect as test_unsupported_term_is_not_dropped_silently, and not "
    "specific to density fitting - the non-fitted route drops it too",
)
def test_density_fitting_does_not_drop_a_squared_eri():
    a, i, b, j = _sapt_indices()

    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    assert '+2 * np.einsum("", )' != generate_einsum(
        2.0 * v * v, density_fitting=True
    )


# --------------------------------------------------------------------------
# density fitting of the intramonomer `w`
#
# `w^{p p'}_{q q'} = (q p | q' p')` carries all four indices on one monomer
# and factorizes by the same positional pairing as `v`:
# `sum_Q B^{Q}_{q p} B^{Q}_{q' p'}`.
# --------------------------------------------------------------------------


def _intra_indices():
    """four distinct monomer-A indices, two occupied and two virtual"""
    a = symbols("a", is_molA=True, above_fermi=True)
    c = symbols("c", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)
    k = symbols("k", is_molA=True, below_fermi=True)

    return a, c, i, k


def test_density_fitting_splits_w_into_two_three_index_arrays():
    reference = """+np.einsum("rcak,Qar,Qkc", t_rcak, Qar, Qkc)"""

    a, c, i, k = _intra_indices()

    t = DoubleVacuumTensorSymbol("t", (i, k), (a, c))
    w = DoubleVacuumTensorSymbol("w", (a, c), (i, k))

    assert reference == generate_einsum(t * w, density_fitting=True)


def test_density_fitting_v_and_w_get_separate_auxiliary_indices():
    # one term holding both integrals is two independent auxiliary sums
    reference = (
        """+np.einsum("rsab,rcak,Qar,Qbs,Par,Pkc", """
        """t_rsab, t_rcak, Qar, Qbs, Qar, Qkc)"""
    )

    a, c, i, k = _intra_indices()
    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    t4 = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    t2 = DoubleVacuumTensorSymbol("t", (i, k), (a, c))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    w = DoubleVacuumTensorSymbol("w", (a, c), (i, k))

    assert reference == generate_einsum(
        t4 * w * v * t2, density_fitting=True
    )


def test_density_fitting_rejects_a_w_straddling_monomers():
    # each pair sitting on one monomer is the *v* pattern; an intramonomer
    # `w` split across A and B denotes a different integral and must raise
    a, c, i, k = _intra_indices()
    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    t4 = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    w_bad = DoubleVacuumTensorSymbol("w", (a, b), (i, j))

    with pytest.raises(IndexError, match="single monomer"):
        generate_einsum(t4 * w_bad, density_fitting=True)


def test_density_fitting_w_unchanged_without_the_option():
    a, c, i, k = _intra_indices()

    t = DoubleVacuumTensorSymbol("t", (i, k), (a, c))
    w = DoubleVacuumTensorSymbol("w", (a, c), (i, k))

    assert '+np.einsum("rcak,akrc", t_rcak, w_akrc)' == generate_einsum(
        t * w
    )
