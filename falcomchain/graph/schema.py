"""
Attribute schema for FalcomChain graphs.

Defines exactly what node, edge, and graph-level attributes the
hierarchical capacitated facility location algorithms read and write.

This is the single source of truth — every constructor, validator,
and documentation page references this schema.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class AttributeSpec:
    """Specification for a single attribute."""
    name: str
    required: bool
    type: type
    default: Any
    purpose: str
    validator: Optional[Callable[[Any], bool]] = None


# ---------------------------------------------------------------------------
# Node attributes
# ---------------------------------------------------------------------------

NODE_ATTRIBUTES: Dict[str, AttributeSpec] = {
    "demand": AttributeSpec(
        name="demand",
        required=True,
        type=float,
        default=None,
        purpose=(
            "Demand of the geographic unit (e.g., population, workload). "
            "Used for demand-balance constraints in district formation."
        ),
        validator=lambda v: v is not None and v >= 0,
    ),
    "candidate": AttributeSpec(
        name="candidate",
        required=True,
        type=int,
        default=0,
        purpose=(
            "Whether this node is a facility candidate (0 or 1). "
            "Each district must contain at least one candidate."
        ),
        validator=lambda v: v in (0, 1, True, False),
    ),
    "C_X": AttributeSpec(
        name="C_X",
        required=False,
        type=float,
        default=0.0,
        purpose="X coordinate of the node centroid (for visualization).",
    ),
    "C_Y": AttributeSpec(
        name="C_Y",
        required=False,
        type=float,
        default=0.0,
        purpose="Y coordinate of the node centroid (for visualization).",
    ),
    "area": AttributeSpec(
        name="area",
        required=False,
        type=float,
        default=1.0,
        purpose="Geographic area of the node (used for compactness metrics).",
    ),
    # Set by Partition.write_to_graph()
    "district": AttributeSpec(
        name="district",
        required=False,
        type=int,
        default=None,
        purpose=(
            "District ID this node belongs to. Set by "
            "``Partition.write_to_graph()``. Used by ``Partition.from_graph()`` "
            "to reconstruct a partition from a saved graph."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

EDGE_ATTRIBUTES: Dict[str, AttributeSpec] = {
    "shared_perim": AttributeSpec(
        name="shared_perim",
        required=False,
        type=float,
        default=1.0,
        purpose=(
            "Length of the shared boundary between two adjacent nodes. "
            "Used by compactness computations and rook adjacency."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Graph-level attributes (graph.graph[...])
# ---------------------------------------------------------------------------

GRAPH_ATTRIBUTES: Dict[str, AttributeSpec] = {
    "crs": AttributeSpec(
        name="crs",
        required=False,
        type=str,
        default=None,
        purpose="Coordinate reference system (set by from_geodataframe).",
    ),
    "teams_per_district": AttributeSpec(
        name="teams_per_district",
        required=False,
        type=dict,
        default=None,
        purpose=(
            "Dict mapping district ID -> team count. Set by "
            "``Partition.write_to_graph()`` so a saved graph fully describes a partition."
        ),
    ),
    "capacity_level": AttributeSpec(
        name="capacity_level",
        required=False,
        type=int,
        default=None,
        purpose=(
            "Maximum number of teams per district. Set by "
            "``Partition.write_to_graph()``."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when a graph's attributes don't match the FalcomChain schema."""


def required_node_attributes() -> list:
    """List of node attribute names that MUST be present."""
    return [name for name, spec in NODE_ATTRIBUTES.items() if spec.required]


def validate_graph(graph, strict: bool = True) -> list:
    """
    Validate that a graph satisfies the FalcomChain attribute schema.

    :param graph: A networkx-like Graph.
    :param strict: If True, raise SchemaValidationError on failure.
        If False, return a list of error strings.

    :returns: List of error messages (empty if valid).
    :raises SchemaValidationError: If strict=True and validation fails.
    """
    errors = []

    g = graph.graph if hasattr(graph, "graph") and hasattr(graph.graph, "nodes") else graph

    if g.number_of_nodes() == 0:
        errors.append("Graph has no nodes.")
        if strict:
            raise SchemaValidationError("\n".join(errors))
        return errors

    # Check required node attributes
    for attr_name in required_node_attributes():
        spec = NODE_ATTRIBUTES[attr_name]
        missing = [n for n in g.nodes if attr_name not in g.nodes[n]]
        if missing:
            sample = missing[:5]
            more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            errors.append(
                f"Missing node attribute '{attr_name}' on nodes: {sample}{more}. "
                f"{spec.purpose}"
            )
            continue

        # Run validator on each node value
        if spec.validator:
            invalid = [
                n for n in g.nodes
                if not spec.validator(g.nodes[n][attr_name])
            ]
            if invalid:
                sample = invalid[:5]
                more = f" (and {len(invalid) - 5} more)" if len(invalid) > 5 else ""
                errors.append(
                    f"Invalid value for node attribute '{attr_name}' on nodes: "
                    f"{sample}{more}."
                )

    # Each district needs at least one candidate (warning, not error)
    candidate_count = sum(
        1 for n in g.nodes if g.nodes[n].get("candidate", 0)
    )
    if candidate_count == 0:
        errors.append(
            "No nodes are marked as facility candidates (all 'candidate' values "
            "are 0 or False). At least one candidate is required for districting."
        )

    if strict and errors:
        raise SchemaValidationError("\n".join(errors))
    return errors


def describe_schema() -> str:
    """Return a human-readable description of the schema."""
    lines = ["FalcomChain Graph Attribute Schema", "=" * 40, ""]

    lines.append("NODE ATTRIBUTES:")
    for name, spec in NODE_ATTRIBUTES.items():
        req = "REQUIRED" if spec.required else "optional"
        default = "" if spec.default is None else f", default={spec.default}"
        lines.append(f"  {name} ({spec.type.__name__}, {req}{default})")
        lines.append(f"    {spec.purpose}")
        lines.append("")

    lines.append("EDGE ATTRIBUTES:")
    for name, spec in EDGE_ATTRIBUTES.items():
        req = "REQUIRED" if spec.required else "optional"
        default = "" if spec.default is None else f", default={spec.default}"
        lines.append(f"  {name} ({spec.type.__name__}, {req}{default})")
        lines.append(f"    {spec.purpose}")
        lines.append("")

    lines.append("GRAPH-LEVEL ATTRIBUTES (graph.graph[...]):")
    for name, spec in GRAPH_ATTRIBUTES.items():
        req = "REQUIRED" if spec.required else "optional"
        lines.append(f"  {name} ({spec.type.__name__}, {req})")
        lines.append(f"    {spec.purpose}")
        lines.append("")

    return "\n".join(lines)
