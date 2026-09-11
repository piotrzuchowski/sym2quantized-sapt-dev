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
    """A dummy index joining two ports: ``(node, port)`` pairs plus the
    index character.  Self-loops (both ports on one tensor) are legal."""

    ends: tuple  # ((node_index, port), (node_index, port)), sorted
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

    Raises ``ValueError`` on a dummy index shared by more than two
    slots - that is not an Einstein contraction and the graph encoding
    would be ambiguous.
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
        elif len(ends) == 2:
            edges.append(
                Edge(
                    ends=tuple(sorted(ends)),
                    space=_space_of(index),
                    monomer=_monomer_of(index),
                )
            )
        else:
            raise ValueError(
                f"index {index} appears in {len(ends)} slots; "
                "an Einstein contraction pairs at most two."
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
            tuple(sorted(((position[n1], p1), (position[n2], p2)))),
            edge.space,
            edge.monomer,
        )
        for edge, ((n1, p1), (n2, p2)) in (
            (e, e.ends) for e in graph.edges
        )
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
        (n1, _), (n2, _) = edge.ends
        adjacency[n1].add(n2)
        adjacency[n2].add(n1)
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


# --- contraction cost --------------------------------------------------


def _dims_of(edge_or_external) -> str:
    label = f"{edge_or_external.monomer or 'g'}{edge_or_external.space}"
    return label


def contraction_cost(graph: TermGraph) -> dict:
    """Leading flop and memory scaling of the best pairwise order.

    Dimensions stay symbolic - one per ``(monomer, space)`` label, e.g.
    ``Ao``, ``Av``, ``Bo`` - and the search over contraction orders is
    exhaustive, which a term-sized network affords.  Returned as
    ``{"flops": {label: exponent, ...}, "memory": {...}, "order": [...]}``
    with costs compared by total degree, then lexicographically.
    """
    # Each node's indices as multisets of dimension labels; edges
    # between a pair contract away, everything else survives.
    node_dims = []
    for node_index in range(len(graph.nodes)):
        dims = []
        for edge in graph.edges:
            hits = sum(1 for end_node, _ in edge.ends if end_node == node_index)
            if hits:
                # a self-edge (trace) is summable immediately: one power
                dims.append(_dims_of(edge))
        for external in graph.externals:
            if external.node == node_index:
                dims.append(_dims_of(external))
        node_dims.append(Counter(dims))

    shared = {}
    for edge in graph.edges:
        (n1, _), (n2, _) = edge.ends
        if n1 != n2:
            pair = (min(n1, n2), max(n1, n2))
            shared.setdefault(pair, Counter())[_dims_of(edge)] += 1

    def degree(counter):
        return sum(counter.values())

    def rank(counter):
        return (degree(counter), tuple(sorted(counter.items())))

    best = {"flops": None, "memory": None, "order": None}

    def search(clusters, dims, flops, memory, order):
        if len(dims) == 1:
            candidate = (rank(flops), rank(memory))
            if best["flops"] is None or candidate < (
                rank(best["flops"]),
                rank(best["memory"]),
            ):
                best.update(
                    flops=flops, memory=memory, order=list(order)
                )
            return
        keys = sorted(dims)
        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1:]:
                contracted = Counter()
                for cluster1 in clusters[key1]:
                    for cluster2 in clusters[key2]:
                        pair = (min(cluster1, cluster2), max(cluster1, cluster2))
                        contracted.update(shared.get(pair, {}))
                cost = dims[key1] + dims[key2]
                for label, count in contracted.items():
                    cost[label] -= count  # shared dims counted once
                result = cost.copy()
                for label, count in contracted.items():
                    result[label] -= count  # and contracted away
                result = +result
                cost = +cost
                new_key = f"({key1}.{key2})"
                new_clusters = dict(clusters)
                new_clusters[new_key] = (
                    clusters[key1] + clusters[key2]
                )
                del new_clusters[key1], new_clusters[key2]
                new_dims = dict(dims)
                new_dims[new_key] = result
                del new_dims[key1], new_dims[key2]
                new_flops = flops.copy()
                for label, count in cost.items():
                    new_flops[label] = max(new_flops[label], count)
                new_memory = memory.copy()
                for label, count in result.items():
                    new_memory[label] = max(new_memory[label], count)
                search(
                    new_clusters,
                    new_dims,
                    new_flops,
                    new_memory,
                    order + [new_key],
                )

    if not graph.nodes:
        return {"flops": {}, "memory": {}, "order": []}
    clusters = {str(i): (i,) for i in range(len(graph.nodes))}
    dims = {str(i): node_dims[i] for i in range(len(graph.nodes))}
    search(clusters, dims, Counter(), Counter(), [])
    return {
        "flops": dict(sorted((+best["flops"]).items())),
        "memory": dict(sorted((+best["memory"]).items())),
        "order": best["order"],
    }


# --- shared intermediates ----------------------------------------------


def _pair_subgraph(graph: TermGraph, n1: int, n2: int) -> TermGraph:
    """The two-node subnetwork: mutual edges stay edges, every other
    index of the pair becomes an anonymous external ("" name - dangling
    position and character matter, the outside index name does not)."""
    keep = {n1: 0, n2: 1}
    edges, externals = [], []
    dangling = set(
        (node, port)
        for node in keep
        for port in range(graph.nodes[node].n_ports)
    )
    for edge in graph.edges:
        (a, pa), (b, pb) = edge.ends
        if a in keep and b in keep:
            edges.append(
                Edge(
                    ends=tuple(sorted(((keep[a], pa), (keep[b], pb)))),
                    space=edge.space,
                    monomer=edge.monomer,
                )
            )
            dangling.discard((a, pa))
            dangling.discard((b, pb))
        elif a in keep:
            externals.append(
                External(keep[a], pa, "", edge.space, edge.monomer)
            )
            dangling.discard((a, pa))
        elif b in keep:
            externals.append(
                External(keep[b], pb, "", edge.space, edge.monomer)
            )
            dangling.discard((b, pb))
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
            dangling.discard((external.node, external.port))
    assert not dangling
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
    pairs actually contracted together (sharing at least one edge)
    count; a mere co-occurrence is not an intermediate.
    """
    seen = {}
    for graph in graphs:
        contracted_pairs = set()
        for edge in graph.edges:
            (a, _), (b, _) = edge.ends
            if a != b:
                contracted_pairs.add((min(a, b), max(a, b)))
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
                "from": [e.ends[0][0], graph.nodes[e.ends[0][0]].port_role(e.ends[0][1])],
                "to": [e.ends[1][0], graph.nodes[e.ends[1][0]].port_role(e.ends[1][1])],
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
    externals as open circles."""
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
    for e in graph.edges:
        (n1, p1), (n2, p2) = e.ends
        role1 = graph.nodes[n1].port_role(p1)
        role2 = graph.nodes[n2].port_role(p2)
        lines.append(
            f'  t{n1} -- t{n2} '
            f'[label="{role1}-{role2}:{e.monomer}{e.space}"];'
        )
    lines.append("}")
    return "\n".join(lines)
