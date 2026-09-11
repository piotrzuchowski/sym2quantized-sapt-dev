"""Terms as graphs: encode, canonicalize, classify, estimate, export.

A fully contracted term is a tensor network: every ``TensorSymbol``
factor is a node, every dummy index shared by two tensor slots is an
edge, and every free (external) index is a half-edge.  This module
makes that graph explicit, so terms can be *classified* (group by
canonical topology), *selected* (filter on invariants: which tensors,
loop count, external pattern), *optimized* (leading contraction-cost
scaling per term; recurring two-tensor subgraphs that are candidates
for shared intermediates), and *exported* for tools and agents (plain
dicts and Graphviz DOT).

The encoding reads the same structure the rest of the package already
walks implicitly - ``diagrams.get_only_linked`` for connectivity,
``spin_integrator._count_loops`` for Goldstone loops - and agrees with
their conventions: an edge is an index appearing in one tensor's
``upper`` and another's ``lower`` (or twice within one tensor).

Limitations, on purpose: tensor permutation *symmetries* declared at
construction are not folded into the canonical key, so two terms equal
only under a symmetry of an amplitude get distinct keys (upstream,
``substitute_dummies_double_vac`` already canonicalizes those where it
can).  Graphs here are small (a handful of tensors), so the canonical
labeling is an exact search, not a heuristic.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations

from sympy import Add, Mul
from sympy.physics.secondquant import TensorSymbol

from sym2quantized_sapt.spin_integrator import _count_loops

__all__ = [
    "Edge",
    "External",
    "Node",
    "TermGraph",
    "canonical_key",
    "classify",
    "common_pairs",
    "contraction_cost",
    "expr_to_graphs",
    "invariants",
    "term_to_graph",
    "to_dict",
    "to_dot",
]


def _space_of(index) -> str:
    """``"o"`` hole, ``"v"`` particle, ``"g"`` general."""
    assumptions = index.assumptions0
    if assumptions.get("below_fermi"):
        return "o"
    if assumptions.get("above_fermi"):
        return "v"
    return "g"


def _monomer_of(index) -> str:
    assumptions = index.assumptions0
    if assumptions.get("is_molA"):
        return "A"
    if assumptions.get("is_molB"):
        return "B"
    return ""


@dataclass(frozen=True)
class Node:
    """One tensor factor: its symbol name and slot structure."""

    name: str
    n_upper: int
    n_lower: int

    @property
    def n_ports(self) -> int:
        return self.n_upper + self.n_lower

    def port_role(self, port: int) -> str:
        """``"u0"``, ``"u1"``, ... then ``"l0"``, ``"l1"``, ..."""
        if port < self.n_upper:
            return f"u{port}"
        return f"l{port - self.n_upper}"


@dataclass(frozen=True)
class Edge:
    """A dummy index joining two or more ports (``(node, port)`` pairs)
    plus the index character.  Two ports is an ordinary Einstein
    contraction; more is a Hadamard-style hyperedge -- a resolvent
    denominator sharing the amplitudes' indices is the standing
    example.  Self-loops (several ports on one tensor) are legal."""

    ends: tuple  # ((node_index, port), ...), sorted, len >= 2
    space: str  # "o" / "v" / "g"
    monomer: str  # "A" / "B" / ""


@dataclass(frozen=True)
class External:
    """A free index: a half-edge with a name that must survive."""

    node: int
    port: int
    name: str
    space: str
    monomer: str


@dataclass(frozen=True)
class TermGraph:
    coefficient: object  # the sympy numeric prefactor
    nodes: tuple = field(default_factory=tuple)  # of Node
    edges: tuple = field(default_factory=tuple)  # of Edge
    externals: tuple = field(default_factory=tuple)  # of External
    #: Goldstone loops, by ``spin_integrator._count_loops`` itself, so
    #: the number is the one whose power of two ``spin_integration``
    #: multiplied in.
    loops: int = 0


def term_to_graph(term) -> TermGraph:
    """Encode one term (a ``Mul`` of a number and tensors, or a bare
    tensor) as a :class:`TermGraph`.

    An index in one slot is a free external; in two, an Einstein
    contraction; in three or more, a Hadamard-style hyperedge (the
    package's resolvent denominators share every index with the
    amplitudes they divide).
    """
    factors = term.args if isinstance(term, Mul) else [term]
    coefficient = 1
    tensors = []
    for factor in factors:
        if isinstance(factor, TensorSymbol):
            tensors.append(factor)
        elif factor.is_number:
            coefficient = coefficient * factor
        else:
            raise ValueError(
                f"term_to_graph expected numbers and tensors, got {factor!r}"
            )

    nodes = tuple(
        Node(str(t.symbol()), len(t.upper()), len(t.lower()))
        for t in tensors
    )

    # index -> list of (node, port) slots holding it
    slots = {}
    for node_index, tensor in enumerate(tensors):
        indices = list(tensor.upper()) + list(tensor.lower())
        for port, index in enumerate(indices):
            slots.setdefault(index, []).append((node_index, port))

    edges, externals = [], []
    for index, ends in sorted(
        slots.items(), key=lambda item: str(item[0])
    ):
        if len(ends) == 1:
            (node_index, port) = ends[0]
            externals.append(
                External(
                    node=node_index,
                    port=port,
                    name=str(index),
                    space=_space_of(index),
                    monomer=_monomer_of(index),
                )
            )
        else:
            edges.append(
                Edge(
                    ends=tuple(sorted(ends)),
                    space=_space_of(index),
                    monomer=_monomer_of(index),
                )
            )

    upper, lower = [], []
    for tensor in tensors:
        upper += list(tensor.upper())
        lower += list(tensor.lower())

    return TermGraph(
        coefficient=coefficient,
        nodes=nodes,
        edges=tuple(edges),
        externals=tuple(externals),
        loops=_count_loops(upper, lower),
    )


def expr_to_graphs(expr) -> list:
    """Every term of ``expr`` (an ``Add``, or a single term) encoded."""
    terms = expr.args if isinstance(expr, Add) else [expr]
    return [term_to_graph(term) for term in terms]


# --- canonical labeling -----------------------------------------------


def _relabelled_signature(graph: TermGraph, order) -> tuple:
    """The graph's edge/external structure under a node relabelling."""
    position = {old: new for new, old in enumerate(order)}
    edges = sorted(
        (
            tuple(sorted((position[n], p) for n, p in edge.ends)),
            edge.space,
            edge.monomer,
        )
        for edge in graph.edges
    )
    externals = sorted(
        (position[x.node], x.port, x.name, x.space, x.monomer)
        for x in graph.externals
    )
    return tuple(edges), tuple(externals)


def canonical_key(graph: TermGraph, with_coefficient: bool = False) -> str:
    """A string equal for two graphs iff they are the same tensor
    network up to the order the factors happened to be written in.

    Node order is canonicalized by exact search: nodes are grouped by
    ``(name, n_upper, n_lower)`` and only permutations within a group
    are tried, so the search is exhaustive yet cheap for the handful of
    factors a term carries.  Dummy names never enter - only topology,
    spaces and monomers - while external index *names* do, since they
    are the term's open slots.
    """
    groups = {}
    for node_index, node in enumerate(graph.nodes):
        groups.setdefault((node.name, node.n_upper, node.n_lower), []).append(
            node_index
        )
    group_items = sorted(groups.items())

    def orders(remaining):
        if not remaining:
            yield []
            return
        (_, members), *rest = remaining
        for perm in permutations(members):
            for tail in orders(rest):
                yield list(perm) + tail

    best = None
    for order in orders(group_items):
        signature = _relabelled_signature(graph, order)
        if best is None or signature < best:
            best = signature

    names = tuple(key for key, _ in group_items)
    key = repr((names, best))
    if with_coefficient:
        key = f"{graph.coefficient}*{key}"
    return key


# --- invariants and classification ------------------------------------


def invariants(graph: TermGraph) -> dict:
    """Cheap, name-stable facts about one term, for filtering."""
    edge_spaces = Counter(
        f"{edge.monomer or '-'}{edge.space}" for edge in graph.edges
    )
    external_pattern = "".join(
        f"{x.monomer or '-'}{x.space}"
        for x in sorted(graph.externals, key=lambda x: x.name)
    )
    adjacency = {i: set() for i in range(len(graph.nodes))}
    for edge in graph.edges:
        members = {n for n, _ in edge.ends}
        for n1 in members:
            adjacency[n1].update(members - {n1})
    seen, stack = set(), [0] if graph.nodes else []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node])
    return {
        "tensors": tuple(sorted(node.name for node in graph.nodes)),
        "n_tensors": len(graph.nodes),
        "n_contracted": len(graph.edges),
        "n_external": len(graph.externals),
        "edge_spaces": dict(sorted(edge_spaces.items())),
        "external_pattern": external_pattern,
        "loops": graph.loops,
        "connected": len(seen) == len(graph.nodes),
    }


def classify(expr, key=canonical_key) -> dict:
    """Group the terms of ``expr`` by ``key(graph)``.

    Returns ``{key: [TermGraph, ...]}``.  With the default key, terms
    landing in one bucket are the same contraction pattern and differ
    at most by their coefficient - the groups a selection or a merge
    should reason about.  Pass e.g.
    ``lambda g: invariants(g)["external_pattern"]`` for coarser
    families.
    """
    buckets = {}
    for graph in expr_to_graphs(expr):
        buckets.setdefault(key(graph), []).append(graph)
    return buckets


# --- contraction cost ---------------------------------------------------


def _dims_of(edge_or_external) -> str:
    label = f"{edge_or_external.monomer or 'g'}{edge_or_external.space}"
    return label


def contraction_cost(graph: TermGraph) -> dict:
    """Leading flop and memory scaling of the best pairwise order.

    Dimensions stay symbolic - one per ``(monomer, space)`` label, e.g.
    ``Ao``, ``Av``, ``Bo`` - and the search over contraction orders is
    exhaustive, which a term-sized network affords.  Bookkeeping is per
    *index* (edge), so hyperedges and self-loops cost correctly: an
    index dimension enters a cluster once, and is summed away only when
    every slot holding it sits inside one cluster.  Returned as
    ``{"flops": {label: exponent, ...}, "memory": {...}, "order": [...]}``
    with costs compared by total degree, then lexicographically.
    """
    if not graph.nodes:
        return {"flops": {}, "memory": {}, "order": []}

    edge_nodes = [
        frozenset(node for node, _ in edge.ends) for edge in graph.edges
    ]
    external_nodes = [
        (external.node, _dims_of(external)) for external in graph.externals
    ]

    def cluster_dims(members):
        """dimension labels of the tensor holding ``members`` merged"""
        dims = Counter()
        for edge, holders in zip(graph.edges, edge_nodes):
            if holders & members and not holders <= members:
                dims[_dims_of(edge)] += 1
        for node, label in external_nodes:
            if node in members:
                dims[label] += 1
        return dims

    def merge_cost(members):
        """flops of forming ``members`` from any two parts: every index
        touching the merged cluster, counted once"""
        dims = Counter()
        for edge, holders in zip(graph.edges, edge_nodes):
            if holders & members:
                dims[_dims_of(edge)] += 1
        for node, label in external_nodes:
            if node in members:
                dims[label] += 1
        return dims

    def rank(counter):
        return (sum(counter.values()), tuple(sorted(counter.items())))

    best = {"flops": None, "memory": None, "order": None}

    def search(clusters, flops, memory, order):
        if len(clusters) == 1:
            candidate = (rank(flops), rank(memory))
            if best["flops"] is None or candidate < (
                rank(best["flops"]),
                rank(best["memory"]),
            ):
                best.update(flops=flops, memory=memory, order=list(order))
            return
        keys = sorted(clusters)
        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1:]:
                members = clusters[key1] | clusters[key2]
                cost = merge_cost(members)
                result = cluster_dims(members)
                new_flops = flops.copy()
                for label, count in cost.items():
                    new_flops[label] = max(new_flops[label], count)
                new_memory = memory.copy()
                for label, count in result.items():
                    new_memory[label] = max(new_memory[label], count)
                new_clusters = {
                    key: value
                    for key, value in clusters.items()
                    if key not in (key1, key2)
                }
                new_clusters[f"({key1}.{key2})"] = members
                search(
                    new_clusters,
                    new_flops,
                    new_memory,
                    order + [f"({key1}.{key2})"],
                )

    clusters = {
        str(i): frozenset([i]) for i in range(len(graph.nodes))
    }
    if len(clusters) == 1:
        only = clusters["0"]
        return {
            "flops": dict(sorted(merge_cost(only).items())),
            "memory": dict(sorted(cluster_dims(only).items())),
            "order": [],
        }
    search(clusters, Counter(), Counter(), [])
    return {
        "flops": dict(sorted((+best["flops"]).items())),
        "memory": dict(sorted((+best["memory"]).items())),
        "order": best["order"],
    }


# --- shared intermediates ----------------------------------------------


def _pair_subgraph(graph: TermGraph, n1: int, n2: int) -> TermGraph:
    """The two-node subnetwork: an index whose every slot sits on the
    pair stays an edge; an index also held outside cannot be summed by
    the pair, so each of its inside slots becomes an anonymous external
    ("" name - dangling position and character matter, the outside
    index name does not)."""
    keep = {n1: 0, n2: 1}
    edges, externals = [], []
    covered = set()
    for edge in graph.edges:
        inside = [(n, p) for n, p in edge.ends if n in keep]
        covered.update(inside)
        if len(inside) == len(edge.ends):
            edges.append(
                Edge(
                    ends=tuple(sorted((keep[n], p) for n, p in inside)),
                    space=edge.space,
                    monomer=edge.monomer,
                )
            )
        else:
            for n, p in inside:
                externals.append(
                    External(keep[n], p, "", edge.space, edge.monomer)
                )
    for external in graph.externals:
        if external.node in keep:
            externals.append(
                External(
                    keep[external.node],
                    external.port,
                    "",
                    external.space,
                    external.monomer,
                )
            )
            covered.add((external.node, external.port))
    assert covered == {
        (node, port)
        for node in keep
        for port in range(graph.nodes[node].n_ports)
    }
    return TermGraph(
        coefficient=1,
        nodes=(graph.nodes[n1], graph.nodes[n2]),
        edges=tuple(edges),
        externals=tuple(externals),
    )


def common_pairs(graphs, min_count: int = 2) -> list:
    """Two-tensor contractions recurring across ``graphs`` - the
    candidates for shared intermediates.

    Returns ``[(count, key, example)]`` sorted most-frequent first,
    where ``example`` is one :class:`TermGraph` of the pair.  Only
    pairs actually joined by some index (an edge with slots on both)
    count; a mere co-occurrence is not an intermediate.
    """
    seen = {}
    for graph in graphs:
        contracted_pairs = set()
        for edge in graph.edges:
            members = sorted({n for n, _ in edge.ends})
            for i, na in enumerate(members):
                for nb in members[i + 1:]:
                    contracted_pairs.add((na, nb))
        for n1, n2 in contracted_pairs:
            sub = _pair_subgraph(graph, n1, n2)
            key = canonical_key(sub)
            entry = seen.setdefault(key, [0, sub])
            entry[0] += 1
    result = [
        (count, key, example)
        for key, (count, example) in seen.items()
        if count >= min_count
    ]
    return sorted(result, key=lambda item: (-item[0], item[1]))


# --- export -------------------------------------------------------------


def to_dict(graph: TermGraph) -> dict:
    """A plain, JSON-ready description of one term."""
    return {
        "coefficient": str(graph.coefficient),
        "nodes": [
            {"name": n.name, "upper": n.n_upper, "lower": n.n_lower}
            for n in graph.nodes
        ],
        "edges": [
            {
                "ends": [
                    [n, graph.nodes[n].port_role(p)] for n, p in e.ends
                ],
                "space": e.space,
                "monomer": e.monomer,
            }
            for e in graph.edges
        ],
        "externals": [
            {
                "node": x.node,
                "port": graph.nodes[x.node].port_role(x.port),
                "name": x.name,
                "space": x.space,
                "monomer": x.monomer,
            }
            for x in graph.externals
        ],
        "invariants": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in invariants(graph).items()
        },
    }


def to_dot(graph: TermGraph, name: str = "term") -> str:
    """Graphviz DOT: tensors as boxes, contractions as labelled edges,
    externals as open circles.  A hyperedge (an index on three or more
    slots) renders through a small junction point."""
    lines = [f'graph "{name}" {{', "  node [shape=box];"]
    for i, node in enumerate(graph.nodes):
        lines.append(f'  t{i} [label="{node.name}"];')
    for j, x in enumerate(graph.externals):
        lines.append(
            f'  x{j} [shape=circle, label="{x.name}", style=dashed];'
        )
        role = graph.nodes[x.node].port_role(x.port)
        lines.append(
            f'  t{x.node} -- x{j} [label="{role}:{x.monomer}{x.space}"];'
        )
    for k, e in enumerate(graph.edges):
        label = f"{e.monomer}{e.space}"
        if len(e.ends) == 2:
            (n1, p1), (n2, p2) = e.ends
            role1 = graph.nodes[n1].port_role(p1)
            role2 = graph.nodes[n2].port_role(p2)
            lines.append(
                f'  t{n1} -- t{n2} '
                f'[label="{role1}-{role2}:{label}"];'
            )
        else:
            lines.append(f'  h{k} [shape=point];')
            for n, p in e.ends:
                role = graph.nodes[n].port_role(p)
                lines.append(
                    f'  t{n} -- h{k} [label="{role}:{label}"];'
                )
    lines.append("}")
    return "\n".join(lines)
