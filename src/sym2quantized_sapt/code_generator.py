import string
import re
from sympy import Add, Mul, Expr, expand
from sympy.physics.secondquant import TensorSymbol


def _psi4numpy_indices(index: str) -> str:
    """
    Renames a canonical dummy to the psi4numpy convention:
    a -> r, b -> s, i -> a, j -> b (p and q are left alone).

    NOTE: this is only injective on the canonical names produced by
    `substitute_dummies_double_vac`. Every occurrence of the first
    matching letter is replaced, so a name that already lives in the
    target alphabet is mapped a second time: `r` stays `r` while a hole
    named `a` also becomes `r`, and the two indices collapse into one.
    """
    if "a" in index:
        index = index.replace("a", "r")

    elif "b" in index:
        index = index.replace("b", "s")

    elif "i" in index:
        index = index.replace("i", "a")

    elif "j" in index:
        index = index.replace("j", "b")

    return index


def _pretty_indices_names(indices: str) -> str:
    PRETTY_INDICES = {
        "a": "a",
        "b": "b",
        "r": "r",
        "s": "s",
        "p": "p",
        "q": "q",
        "a_1": "A",
        "b_1": "B",
        "r_1": "R",
        "s_1": "S",
        "p_1": "P",
        "q_1": "Q",
    }
    REPLACABLE_INDICES = set(PRETTY_INDICES.keys())
    merged = "".join(indices.replace("->", ",").split(","))
    unique_indices = set(re.findall(r"[a-z](?:_\d+)?", merged))

    if unique_indices.issubset(REPLACABLE_INDICES):
        pattern = re.compile(
            "|".join(
                re.escape(index_key)
                for index_key in sorted(PRETTY_INDICES, key=len, reverse=True)
            )
        )
        return pattern.sub(lambda m: PRETTY_INDICES[m.group(0)], indices)

    raise ValueError("Code generator: `pretty_indices` cannot be applied!")


def _replace_indices_names(indices: str) -> str:
    """
    Renames every subscripted index (`a_1`, `p_10`, ...) to a single unused
    letter, so that an einsum subscript stays one character per index.

    Three rules keep the renaming faithful:
    - the pool of new names excludes every name already present in
      `indices`, so a dummy is never aliased onto an index in use,
    - the longest names are substituted first, so `a_1` is not substituted
      inside `a_10`,
    - the candidates keep their first-appearance order, so the generated
      code does not depend on the hash seed.
    """
    new_names = list(string.ascii_lowercase) + list(string.ascii_uppercase)

    used_names = {"a", "b", "r", "s"}
    used_names |= {i for i in indices if i in new_names}
    for elem in used_names:
        new_names.remove(elem)

    merged = "".join(indices.replace("->", ",").split(","))
    unique_indices = list(dict.fromkeys(re.findall(r"[a-z](?:_\d+)?", merged)))
    to_substitute = sorted(
        (e for e in unique_indices if "_" in e), key=len, reverse=True
    )

    new_indices = indices
    for elem in to_substitute:
        try:
            old_name = elem
            new_indices = new_indices.replace(old_name, new_names.pop(0))
        except IndexError as exc:
            raise IndexError(
                "Too many indices!!! Not enough names for them."
            ) from exc

    return new_indices


# Placeholders standing in for a density-fitting auxiliary index while the
# ordinary indices are renamed. None of them is matched by the
# `[a-z](?:_\d+)?` pattern the two renamers key off, nor drawn from the ASCII
# letter pool `_replace_indices_names` hands out, so they travel through both
# untouched and turn into letters only in `_assign_auxiliary_names`.
_AUX_SENTINELS = "#$%&@!~?"

# Candidate letters for the auxiliary index, `Q` first after the psi4numpy
# convention. Letters already spent by the finished subscript are skipped.
_AUX_NAMES = "QPRSTUVWXYZ"

_TOO_MANY_ERIS = "Too many ERIs!!! Not enough auxiliary indices for them."


def _assign_auxiliary_names(indices: str) -> str:
    """
    Replaces every density-fitting sentinel in a finished einsum subscript
    with a letter of its own.

    Each ERI carries its own sentinel and so ends up with its own auxiliary
    sum. Sharing one letter would fuse them into a single sum over the
    auxiliary basis - `sum_Q B B B B` instead of `(sum_Q B B)(sum_P B B)` -
    which is a different, and wrong, quantity.

    Runs last, after `_replace_indices_names` / `_pretty_indices_names`, so
    that the letters already spent on ordinary indices are known: the pretty
    table renames `q_1` to `Q`, and the auxiliary index has to step aside
    when it does.
    """
    used_sentinels = [s for s in _AUX_SENTINELS if s in indices]
    if not used_sentinels:
        return indices

    free_names = [name for name in _AUX_NAMES if name not in indices]

    for sentinel in used_sentinels:
        try:
            indices = indices.replace(sentinel, free_names.pop(0))
        except IndexError as exc:
            raise IndexError(_TOO_MANY_ERIS) from exc

    return indices


def _get_code_str(
    coeff, cont_ind, uncont_ind, variables, pretty_indices=False
) -> str:
    indices = ",".join(cont_ind)
    if uncont_ind:
        indices += "->" + "".join(uncont_ind)

    if pretty_indices:
        indices = _pretty_indices_names(indices)
    else:
        indices = _replace_indices_names(indices)

    indices = _assign_auxiliary_names(indices)

    variables = ", ".join(variables)
    variables = variables.replace("v_A", "vA")
    variables = variables.replace("v_B", "vB")
    variables = variables.replace("o_A", "omegaA")
    variables = variables.replace("o_B", "omegaB")

    code_str = 'np.einsum("{0}", {1})'.format(indices, variables)

    if coeff == 1:
        code_str = "".join(("+", code_str))
    elif coeff == -1:
        code_str = "".join(("-", code_str))
    else:
        if coeff == int(coeff):
            coeff_str = str(int(coeff))
        else:
            coeff_str = str(coeff)
        code_str = " * ".join((coeff_str, code_str))

        # extra plus in front
        if coeff > 0:
            code_str = "".join(("+", code_str))

    return code_str


def _is_eri(tensor: TensorSymbol) -> bool:
    """
    The two-electron integrals the density-fitting split applies to: `v`,
    the intermolecular interaction integral, and `w`, the intramonomer one
    (`w^{p p'}_{q q'} = (q p | q' p')`, all four indices on one monomer).
    Both factorize by the same positional pairing - lower slot k with upper
    slot k - which `_check_eri_indices` guards: for `v` each pair must sit
    on one monomer (A first by `get_V_operator`'s order), and a `w` pair
    straddling monomers is rejected the same way. The monomer potentials
    print as `v_A` and `v_B` and are left alone, as are `s`, `e` and every
    amplitude.
    """
    return str(tensor.symbol) in ("v", "w")


def _monomer_of(index) -> str:
    """`"A"`, `"B"`, or `""` when the index carries no monomer assumption."""
    assumptions = index.assumptions0

    if assumptions.get("is_molA"):
        return "A"

    if assumptions.get("is_molB"):
        return "B"

    return ""


def _check_eri_indices(tensor: TensorSymbol) -> None:
    """
    Guards the two assumptions the density-fitting split makes about `v`.

    `v^{p r}_{q s} = (p q | r s)` is factorized as
    `sum_Q B^{Q}_{q p} B^{Q}_{s r}`, so the split pairs slot 0 of the lower
    indices with slot 0 of the upper ones, and likewise for slot 1. That is
    the right factorization only for a `v` carrying two upper and two lower
    indices ordered monomer A first - the order `get_V_operator` builds. A
    `v` ordered any other way denotes a different integral, and pairing it
    positionally would silently hand back three-index arrays straddling both
    monomers.

    Indices without a monomer assumption are not second-guessed: a plain
    ERI keeps the positional pairing.
    """
    upper, lower = tensor.upper, tensor.lower

    if len(upper) + len(lower) != 4:
        raise IndexError(
            f"Code generator: density fitting expected 4 indices "
            f"in tensor {str(tensor)}, got {len(upper) + len(lower)}."
        )

    if len(upper) != 2 or len(lower) != 2:
        raise IndexError(
            f"Code generator: density fitting expected 2 upper and 2 lower "
            f"indices in tensor {str(tensor)}, got {len(upper)} and "
            f"{len(lower)}."
        )

    pairs = [(_monomer_of(lower[i]), _monomer_of(upper[i])) for i in range(2)]

    if all(all(pair) for pair in pairs) and any(
        pair[0] != pair[1] for pair in pairs
    ):
        raise IndexError(
            f"Code generator: density fitting expected each pair of indices "
            f"of tensor {str(tensor)} on one monomer, got "
            f"{pairs[0][0]}{pairs[0][1]} and {pairs[1][0]}{pairs[1][1]}."
        )

    # `w` is the *intramonomer* integral: all four indices on one monomer.
    # A per-pair check cannot catch a `w` split v-style across A and B, so
    # guard it separately.
    monomers = {m for pair in pairs for m in pair if m}
    if str(tensor.symbol) == "w" and len(monomers) > 1:
        raise IndexError(
            f"Code generator: density fitting expected the intramonomer "
            f"tensor {str(tensor)} on a single monomer, got "
            f"{''.join(sorted(monomers))}."
        )


def _variable_name(tensor: TensorSymbol, density_fitting: bool = False):
    """
    Names of the numpy arrays holding `tensor`: the tensor symbol followed by
    one letter per index, lower indices first.

    Only the leading letter of every index survives, so `t^{i_1 j}_{a_1 b}`
    and `t^{i j}_{a b}` share the array `t_rsab`. Both code paths
    (`_get_einsum_for_Tensor` and `_get_einsum_for_Mul`) name their arrays
    here, so a tensor keeps the same name whether it stands alone or sits
    in a product.

    Under `density_fitting` an ERI is never stored as a four-index array:
    `v^{a b}_{i j}` becomes the pair of three-index arrays `Qar`, `Qbs`, one
    per monomer. Hence the tuple return - every other tensor yields a
    one-element tuple.
    """
    if density_fitting and _is_eri(tensor):
        _check_eri_indices(tensor)

    var_indices = [
        _psi4numpy_indices(idx.name[0])
        for idx in (*tensor.lower, *tensor.upper)
    ]

    # v_abrs -> Qar, Qbs
    if density_fitting and _is_eri(tensor):
        return (
            f"Q{var_indices[0]}{var_indices[2]}",
            f"Q{var_indices[1]}{var_indices[3]}",
        )

    return ("_".join((str(tensor.symbol), "".join(var_indices))),)


def _expand_tensor(
    tensor: TensorSymbol, aux_sentinel=None, density_fitting=False
):
    """
    One tensor as it enters an einsum call: the subscript group(s) it
    contributes, the array name(s) holding it, and its indices in
    `lower + upper` order.

    Under density fitting an ERI contributes two three-index groups sharing
    `aux_sentinel`; every other tensor contributes a single group either
    way. `indices_raw` never carries the auxiliary index, so the caller's
    uncontracted-index scan leaves it out of the einsum output and it stays
    summed.

    Both code paths go through here, so a tensor is named and sliced
    identically whether it stands alone or sits in a product.
    """
    arg_indices = [
        _psi4numpy_indices(idx.name) for idx in (*tensor.lower, *tensor.upper)
    ]

    variables = _variable_name(tensor, density_fitting=density_fitting)

    if len(variables) == 2:
        indices = [
            f"{aux_sentinel}{arg_indices[0]}{arg_indices[2]}",
            f"{aux_sentinel}{arg_indices[1]}{arg_indices[3]}",
        ]
    else:
        indices = ["".join(arg_indices)]

    return indices, list(variables), arg_indices


def _next_aux_sentinel(sentinels) -> str:
    try:
        return next(sentinels)
    except StopIteration as exc:
        raise IndexError(_TOO_MANY_ERIS) from exc


def _get_einsum_for_Tensor(
    tensor: TensorSymbol, pretty_indices=True, density_fitting=False
) -> str:
    upper = [_psi4numpy_indices(idx.name) for idx in tensor.upper]
    lower = [_psi4numpy_indices(idx.name) for idx in tensor.lower]

    indices, variables, indices_raw = _expand_tensor(
        tensor,
        aux_sentinel=_AUX_SENTINELS[0],
        density_fitting=density_fitting,
    )

    # check for uncontracted indicies
    uncont_ind = []
    for elem in indices_raw:
        if (elem not in lower) or (elem not in upper):
            uncont_ind.append(elem)

    return _get_code_str(
        1, indices, uncont_ind, variables, pretty_indices=pretty_indices
    )


def _get_einsum_for_Mul(
    term: Mul, pretty_indices=False, density_fitting=False
) -> str:
    if isinstance(term.args[0], TensorSymbol):
        coeff = 1
    else:
        coeff = term.args[0]

    indices = []
    indices_raw = []
    upper = []
    lower = []
    variables = []
    sentinels = iter(_AUX_SENTINELS)

    for arg in term.args:
        if isinstance(arg, TensorSymbol):
            upper += [_psi4numpy_indices(idx.name) for idx in arg.upper]
            lower += [_psi4numpy_indices(idx.name) for idx in arg.lower]

            # every ERI gets an auxiliary index of its own
            if density_fitting and _is_eri(arg):
                aux_sentinel = _next_aux_sentinel(sentinels)
            else:
                aux_sentinel = None

            arg_ind, arg_var, arg_indices = _expand_tensor(
                arg,
                aux_sentinel=aux_sentinel,
                density_fitting=density_fitting,
            )

            indices += arg_ind
            variables += arg_var
            indices_raw += arg_indices

    # check for uncontracted indicies
    uncont_ind = []
    for elem in indices_raw:
        if (elem not in lower) or (elem not in upper):
            uncont_ind.append(elem)

    return _get_code_str(
        coeff, indices, uncont_ind, variables, pretty_indices=pretty_indices
    )


def generate_einsum(
    expr: Expr, pretty_indices=False, density_fitting=False
) -> str:
    """
    Generates string containing numpy einsum code of given expression.

    The expression has to carry the canonical dummy names (a, b, i, j, p, q
    with the `_1, _2, ...` suffixes), because the indices are renamed to the
    psi4numpy convention here. Do NOT rename them beforehand with
    `substitute_dummies_double_vac(expr, pretty_indices=...)` - see the NOTE
    in that function.

    Args:
        expr (Expr): SymPy expression for code generation
        pretty_indices (bool): name the indices after the fixed table in
            `_pretty_indices_names` instead of renaming every subscripted
            index to an unused letter
        density_fitting (bool): factorize every intermolecular two-electron
            integral `v^{p r}_{q s} = (p q | r s)` into
            `sum_Q B^{Q}_{q p} B^{Q}_{s r}`, emitting the two three-index
            arrays `Qqp`, `Qsr` instead of the four-index `v_qspr`. Each ERI
            of a term is given an auxiliary index of its own. No other
            tensor is touched - `(v_A)`, `(v_B)`, `s`, `e` and the
            amplitudes are emitted as usual.

    Returns:
        str: string with numpy code

    Raises:
        IndexError: the expression has more indices than there are names,
            more ERIs than there are auxiliary indices, or a `v` that
            `density_fitting` cannot factorize
        ValueError: `pretty_indices` is set and an index is outside the table
    """

    expr = expand(expr)

    if isinstance(expr, TensorSymbol):
        return _get_einsum_for_Tensor(
            expr,
            pretty_indices=pretty_indices,
            density_fitting=density_fitting,
        )

    if isinstance(expr, Add):
        return "\n".join(
            [
                generate_einsum(
                    arg,
                    pretty_indices=pretty_indices,
                    density_fitting=density_fitting,
                )
                for arg in expr.args
            ]
        )

    if isinstance(expr, Mul):
        return _get_einsum_for_Mul(
            expr,
            pretty_indices=pretty_indices,
            density_fitting=density_fitting,
        )

    # expr is neither Mul, Add nor TensorSymbol:
    return ""
