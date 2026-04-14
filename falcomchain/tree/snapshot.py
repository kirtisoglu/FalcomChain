"""
Chain recording in compact binary format with delta encoding.

Stores the initial assignment once, then only the diffs per step.
Optional substep recording captures spanning tree data for animation.

File layout::

    chain.fcrec            -- main chain record (header + initial + steps)
    chain.fcrec.sub        -- substep data (tree edges, cuts) if recorded

Usage::

    from falcomchain.tree.snapshot import Recorder

    recorder = Recorder("output/run_001")

    # During chain setup
    recorder.write_header(graph, initial_partition, params)

    # During chain iteration (called by MarkovChain automatically)
    recorder.record_step(state, accepted)

    # Optional: substeps called from bipartition_tree
    recorder.record_substep(tree_edges, cut_node, psi)

    recorder.close()

    # Later: export to JSON for FalcomPlot
    Recorder.export_to_json("output/run_001", "output/json_for_plot/")
"""

import gzip
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Binary format constants
MAGIC = b"FCREC"
VERSION = 2

# Step flags
FLAG_ACCEPTED = 0x01
FLAG_HAS_SUBSTEPS = 0x02


def _to_py(v):
    """Convert numpy/pandas scalars to JSON-safe builtins."""
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, (set, frozenset)):
        return sorted(str(x) for x in v)
    return v


class Recorder:
    """
    Records chain iterations in a compact binary format.

    The format uses delta encoding: only changed nodes are stored per step.
    Substep data (spanning trees, cuts) is stored in a separate file.

    :param output_dir: Directory for output files.
    :param record_substeps: If True, capture tree data from bipartition calls.
    """

    def __init__(self, output_dir: str, record_substeps: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_substeps = record_substeps

        self._main_path = self.output_dir / "chain.fcrec"
        self._sub_path = self.output_dir / "chain.fcrec.sub"
        self._main_file = None
        self._sub_file = None

        # Node ID mapping: arbitrary node IDs -> int32 indices
        self._node_to_idx = {}
        self._idx_to_node = {}

        # Current assignment (for computing deltas)
        self._current_assignment = {}

        self._step_count = 0
        self._phase_buffer = []      # phases within current step
        self._tree_cut_buffer = []   # tree cuts accumulating for phase 2+3
        self._substep_offsets = []   # byte offset of each step's phase data

        # Metadata for export
        self._params = {}
        self._node_coords = {}
        self._node_candidates = {}  # node_id_str -> bool

    def write_header(self, graph, initial_partition, params: Dict[str, Any]):
        """
        Write the file header, node ID table, and initial assignment.
        Must be called before any record_step calls.

        :param graph: The base graph.
        :param initial_partition: The initial Partition object.
        :param params: Chain parameters dict.
        """
        self._params = {k: _to_py(v) for k, v in params.items()}

        # Build node ID mapping
        g = graph.graph if hasattr(graph, "graph") and hasattr(graph.graph, "nodes") else graph
        nodes = list(g.nodes)
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        self._idx_to_node = {i: n for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        # Extract coordinates and candidate flags for later export
        for node in nodes:
            data = g.nodes[node]
            x = float(data.get("C_X", data.get("x", 0.0)))
            y = float(data.get("C_Y", data.get("y", 0.0)))
            self._node_coords[str(node)] = [x, y]
            self._node_candidates[str(node)] = bool(data.get("candidate", 0))

        # Store initial assignment
        self._current_assignment = dict(initial_partition.assignment.mapping)

        # Open main file
        self._main_file = open(self._main_path, "wb")

        # Write header: magic + version + n_nodes + reserved(n_steps placeholder)
        self._main_file.write(MAGIC)
        self._main_file.write(struct.pack("<HII", VERSION, n_nodes, 0))  # n_steps=0, updated on close

        # Write node ID table as JSON (one-time cost, needed for non-integer node IDs)
        node_id_json = json.dumps([str(n) for n in nodes]).encode("utf-8")
        self._main_file.write(struct.pack("<I", len(node_id_json)))
        self._main_file.write(node_id_json)

        # Write initial assignment as packed int32 array
        # assignment[i] = district index for node i
        district_set = sorted(set(initial_partition.assignment.mapping.values()))
        self._district_to_idx = {d: i for i, d in enumerate(district_set)}
        self._idx_to_district = {i: d for i, d in enumerate(district_set)}

        # Write district ID table
        district_json = json.dumps([str(d) for d in district_set]).encode("utf-8")
        self._main_file.write(struct.pack("<I", len(district_json)))
        self._main_file.write(district_json)

        # Write initial assignment as int16 array (node_idx -> district_idx)
        for node in nodes:
            d = initial_partition.assignment.mapping[node]
            self._main_file.write(struct.pack("<H", self._district_to_idx[d]))

        # Write params as JSON
        params_json = json.dumps(self._params).encode("utf-8")
        self._main_file.write(struct.pack("<I", len(params_json)))
        self._main_file.write(params_json)

        # Write coordinates as JSON
        coords_json = json.dumps(self._node_coords).encode("utf-8")
        self._main_file.write(struct.pack("<I", len(coords_json)))
        self._main_file.write(coords_json)

        # Write candidate flags as JSON
        cand_json = json.dumps(self._node_candidates).encode("utf-8")
        self._main_file.write(struct.pack("<I", len(cand_json)))
        self._main_file.write(cand_json)

        # Open substep file if needed
        if self.record_substeps:
            self._sub_file = open(self._sub_path, "wb")
            self._sub_file.write(MAGIC)
            self._sub_file.write(struct.pack("<H", VERSION))

    def record_step(self, state, accepted: bool, parent_energy: Optional[float] = None):
        """
        Record one chain iteration as a delta from the previous assignment.

        :param state: Current ChainState after this step.
        :param accepted: Whether the proposal was accepted.
        :param parent_energy: Energy of the previous state.
        """
        if self._main_file is None:
            return

        # Nothing to flush — phases are structured by begin_level/end_level

        self._step_count += 1
        new_assignment = state.assignment.mapping

        # Compute delta: nodes whose assignment changed
        diffs = []
        for node, new_dist in new_assignment.items():
            old_dist = self._current_assignment.get(node)
            if old_dist != new_dist:
                node_idx = self._node_to_idx[node]
                # District might be new — add to mapping
                if new_dist not in self._district_to_idx:
                    new_idx = len(self._district_to_idx)
                    self._district_to_idx[new_dist] = new_idx
                    self._idx_to_district[new_idx] = new_dist
                diffs.append((node_idx, self._district_to_idx[new_dist]))

        # Update current assignment
        self._current_assignment = dict(new_assignment)

        # Write step record
        flags = 0
        if accepted:
            flags |= FLAG_ACCEPTED
        if self.record_substeps and hasattr(self, '_step_data') and self._step_data:
            flags |= FLAG_HAS_SUBSTEPS

        energy = float(state.energy) if state.energy is not None else 0.0
        log_pr = float(state.log_proposal_ratio) if state.log_proposal_ratio is not None else 0.0

        self._main_file.write(struct.pack("<B", flags))
        self._main_file.write(struct.pack("<d", energy))
        self._main_file.write(struct.pack("<d", log_pr))
        self._main_file.write(struct.pack("<I", len(diffs)))
        for node_idx, dist_idx in diffs:
            self._main_file.write(struct.pack("<IH", node_idx, dist_idx))

        # Write nested phase data to separate file (one JSON record per step)
        if self._sub_file and hasattr(self, '_step_data') and self._step_data:
            offset = self._sub_file.tell()
            self._substep_offsets.append(offset)
            # Write as single JSON record
            self._sub_file.write(struct.pack("<H", 1))
            self._write_phase(self._step_data)
            self._step_data = None
        elif self._sub_file:
            self._substep_offsets.append(-1)

    # ------------------------------------------------------------------
    # Phase recording — nested structure matching Algorithm 1
    #
    # One FalCom iteration:
    #   upper_level: Phase 2 (recursive) → Phase 3 (tree cuts) → Phase 4 (facilities)
    #   select: choose D², highlight merged region
    #   lower_level: Phase 2 (recursive) → Phase 3 (tree cuts) → Phase 4 (facilities)
    #   accept_reject: energy comparison, accept/reject
    # ------------------------------------------------------------------

    def begin_step(self):
        """Begin recording a new chain step."""
        self._step_data = {
            "label": "Phase 1: Hierarchical Proposal",
            "upper_level": None,
            "select": None,
            "lower_level": None,
            "accept_reject": None,
        }
        self._current_tree_cuts = []
        self._current_level = None  # "supergraph" or "base"

    def begin_level(self, level: str, partition=None):
        """
        Begin recording tree cuts for a level.
        :param level: "supergraph" or "base"
        :param partition: The current partition (used to compute supergraph node coordinates).
        """
        if not self.record_substeps:
            return
        self._current_level = level
        self._current_tree_cuts = []

        # Compute supergraph node coordinates for upper level
        if level == "supergraph" and partition is not None:
            self._supergraph_coords = {}
            g = partition.graph.graph if hasattr(partition.graph, "graph") else partition.graph
            for part_id, nodes in partition.parts.items():
                xs, ys, cnt = 0.0, 0.0, 0
                for node in nodes:
                    nd = g.nodes[node]
                    xs += float(nd.get("C_X", nd.get("x", 0.0)))
                    ys += float(nd.get("C_Y", nd.get("y", 0.0)))
                    cnt += 1
                if cnt > 0:
                    self._supergraph_coords[str(part_id)] = [xs / cnt, ys / cnt]
        else:
            self._supergraph_coords = {}

    def record_tree_cut(
        self, tree_edges, root, cut_node, psi_chosen, psi_total, n_cuts,
        spanning_tree_obj=None, extracted_nodes=None,
    ):
        """Record one Phase 3: Capacitated Tree Cut."""
        if not self.record_substeps:
            return

        node_feasibility = {}
        if spanning_tree_obj is not None:
            h = spanning_tree_obj
            for node in h.graph.nodes:
                pop = h.graph.nodes[node]["demand"]
                has_fac = h.has_facility(node) if not h.supertree else True
                compl_fac = h.complement_has_facility(node) if not h.supertree else True

                demand_ok = False
                for c in range(1, min(h.capacity_level + 1, h.n_teams + 1)):
                    if h.has_ideal_demand(c, pop):
                        demand_ok = True
                        break

                compl_demand_ok = False
                for c in range(1, min(h.capacity_level + 1, h.n_teams + 1)):
                    if h.complement_has_the_ideal_demand(c, pop):
                        compl_demand_ok = True
                        break

                node_feasibility[str(node)] = {
                    "demand_ok": demand_ok,
                    "compl_demand_ok": compl_demand_ok,
                    "has_facility": has_fac,
                    "compl_facility": compl_fac,
                    "demand": int(pop),
                }

        idx = len(self._current_tree_cuts) + 1
        level = self._current_level or "unknown"
        self._current_tree_cuts.append({
            "label": f"Phase 3: Tree Cut {idx}",
            "level": level,
            "edges": [[str(u), str(v)] for u, v in tree_edges],
            "root": str(root),
            "cut_node": str(cut_node),
            "psi_chosen": float(psi_chosen),
            "psi_total": float(psi_total),
            "n_cuts": int(n_cuts),
            "node_feasibility": node_feasibility,
            "extracted_nodes": [str(n) for n in extracted_nodes] if extracted_nodes else [],
        })

    def end_level(self, centers=None):
        """
        Finish recording a level. Packs tree cuts into Phase 2 + Phase 4.
        :param centers: Dict of facility centers for this level.
        """
        if not self.record_substeps:
            return

        level = self._current_level or "unknown"
        is_upper = level == "supergraph"

        level_label = "Upper Level \u2014 Supergraph G\u00b2" if is_upper else "Lower Level \u2014 Base Graph H"

        level_data = {
            "label": level_label,
            "phase2": {
                "label": "Phase 2: Recursive Partitioning",
                "tree_cuts": list(self._current_tree_cuts),
            },
            "phase4": {
                "label": f"Phase 4: {'Level-2' if is_upper else 'Level-1'} Facility Assignment",
                "centers": {str(k): str(v) for k, v in (centers or {}).items() if v is not None},
            },
        }
        if is_upper and self._supergraph_coords:
            level_data["supergraph_coords"] = self._supergraph_coords

        if is_upper:
            self._step_data["upper_level"] = level_data
        else:
            self._step_data["lower_level"] = level_data

        self._current_tree_cuts = []
        self._current_level = None

    def record_select(self, supergraph, selected_superdistricts, merged_base_nodes,
                       partition=None):
        """Record the superdistrict selection between upper and lower levels."""
        if not self.record_substeps:
            return

        sg_nodes = {}
        sg_coords = {}
        for n, data in supergraph.nodes(data=True):
            sg_nodes[str(n)] = {
                "demand": int(data.get("demand", 0)),
                "teams": int(data.get("n_teams", 0)),
            }
            # Compute centroid of this superdistrict's base-graph nodes
            if partition is not None and n in partition.parts:
                xs, ys, cnt = 0.0, 0.0, 0
                g = partition.graph.graph if hasattr(partition.graph, "graph") else partition.graph
                for node in partition.parts[n]:
                    nd = g.nodes[node]
                    xs += float(nd.get("C_X", nd.get("x", 0.0)))
                    ys += float(nd.get("C_Y", nd.get("y", 0.0)))
                    cnt += 1
                if cnt > 0:
                    sg_coords[str(n)] = [xs / cnt, ys / cnt]

        self._step_data["select"] = {
            "label": "Select Superdistrict D\u00b2",
            "selected_superdistricts": [str(s) for s in selected_superdistricts],
            "merged_base_nodes": [str(n) for n in merged_base_nodes],
            "supergraph_nodes": sg_nodes,
            "supergraph_coords": sg_coords,
            "supergraph_edges": [[str(u), str(v)] for u, v in supergraph.edges()],
        }

    def record_accept_reject(self, proposed_state, current_state, accepted):
        """Record the accept/reject decision."""
        if not self.record_substeps:
            return

        self._step_data["accept_reject"] = {
            "label": "Accept/Reject",
            "energy_proposed": float(proposed_state.energy) if proposed_state.energy else 0.0,
            "energy_current": float(current_state.energy) if current_state.energy else 0.0,
            "accepted": accepted,
        }

    # Legacy interface — called from bipartition_tree via CutParams.recorder
    def record_substep(
        self, tree_edges, root, cut_node, psi_chosen, psi_total, n_cuts,
        spanning_tree_obj=None,
    ):
        """Delegates to record_tree_cut."""
        self.record_tree_cut(
            tree_edges=tree_edges, root=root, cut_node=cut_node,
            psi_chosen=psi_chosen, psi_total=psi_total, n_cuts=n_cuts,
            spanning_tree_obj=spanning_tree_obj,
        )

    def _write_phase(self, phase):
        """Write a single phase record to the substep file as length-prefixed JSON."""
        line = json.dumps(phase, separators=(",", ":")).encode("utf-8")
        self._sub_file.write(struct.pack("<I", len(line)))
        self._sub_file.write(line)

    def close(self):
        """Finalize and close all files. Updates the step count in the header."""
        if self._main_file:
            # Update district ID table at end (might have grown during chain)
            district_json = json.dumps(
                {str(k): str(v) for k, v in self._idx_to_district.items()}
            ).encode("utf-8")
            self._main_file.write(struct.pack("<I", len(district_json)))
            self._main_file.write(district_json)

            # Seek back and update n_steps in header
            self._main_file.seek(len(MAGIC) + 2 + 4)  # after magic + version + n_nodes
            self._main_file.write(struct.pack("<I", self._step_count))
            self._main_file.close()
            self._main_file = None

        if self._sub_file:
            self._sub_file.close()
            self._sub_file = None

    # ------------------------------------------------------------------
    # Export to JSON for FalcomPlot
    # ------------------------------------------------------------------

    @staticmethod
    def export_to_json(input_dir: str, output_dir: str):
        """
        Read a binary chain record and export step JSON files + manifest
        for FalcomPlot visualization.

        :param input_dir: Directory containing chain.fcrec (and optionally .sub).
        :param output_dir: Directory to write manifest.json, blocks.json, step_NNNN.json.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        main_path = input_path / "chain.fcrec"
        sub_path = input_path / "chain.fcrec.sub"

        with open(main_path, "rb") as f:
            # Read header
            magic = f.read(5)
            assert magic == MAGIC, f"Bad magic: {magic}"
            version, n_nodes, n_steps = struct.unpack("<HII", f.read(10))

            # Read node ID table
            node_json_len = struct.unpack("<I", f.read(4))[0]
            node_ids = json.loads(f.read(node_json_len).decode("utf-8"))

            # Read district ID table
            dist_json_len = struct.unpack("<I", f.read(4))[0]
            district_ids = json.loads(f.read(dist_json_len).decode("utf-8"))

            # Read initial assignment
            assignment = {}
            for i in range(n_nodes):
                dist_idx = struct.unpack("<H", f.read(2))[0]
                assignment[node_ids[i]] = district_ids[dist_idx]

            # Read params
            params_json_len = struct.unpack("<I", f.read(4))[0]
            params = json.loads(f.read(params_json_len).decode("utf-8"))

            # Read coordinates
            coords_json_len = struct.unpack("<I", f.read(4))[0]
            node_coords = json.loads(f.read(coords_json_len).decode("utf-8"))

            # Read candidate flags
            cand_json_len = struct.unpack("<I", f.read(4))[0]
            node_candidates = json.loads(f.read(cand_json_len).decode("utf-8"))

            # Write manifest
            manifest = {
                "total_steps": n_steps,
                "graph_nodes": n_nodes,
                "parameters": params,
                "node_coordinates": node_coords,
                "node_candidates": node_candidates,
            }
            with open(output_path / "manifest.json", "w") as mf:
                json.dump(manifest, mf, separators=(",", ":"))

            # Replay steps and write JSON files
            # Build a full district_ids lookup that may grow
            all_district_ids = list(district_ids)

            for step_num in range(1, n_steps + 1):
                flags = struct.unpack("<B", f.read(1))[0]
                energy = struct.unpack("<d", f.read(8))[0]
                log_pr = struct.unpack("<d", f.read(8))[0]
                n_diffs = struct.unpack("<I", f.read(4))[0]

                accepted = bool(flags & FLAG_ACCEPTED)
                changed = {}
                for _ in range(n_diffs):
                    node_idx, dist_idx = struct.unpack("<IH", f.read(6))
                    node_id = node_ids[node_idx]
                    # District index might exceed original table
                    while dist_idx >= len(all_district_ids):
                        all_district_ids.append(str(dist_idx))
                    dist_id = all_district_ids[dist_idx]
                    assignment[node_id] = dist_id
                    changed[node_id] = dist_id

                # Build district summary
                districts = {}
                for node_id, dist_id in assignment.items():
                    if dist_id not in districts:
                        districts[dist_id] = {"nodes": 0, "demand": 0}
                    districts[dist_id]["nodes"] += 1

                frame = {
                    "step": step_num,
                    "accepted": accepted,
                    "energy": energy,
                    "log_proposal_ratio": log_pr,
                    "assignment": dict(assignment),
                    "changed_nodes": changed,
                    "districts": districts,
                }

                step_path = output_path / f"step_{step_num:04d}.json"
                with open(step_path, "w") as sf:
                    json.dump(frame, sf, separators=(",", ":"))

            # Read final district table (appended at end)
            try:
                final_dist_len = struct.unpack("<I", f.read(4))[0]
                f.read(final_dist_len)  # skip — already handled dynamically
            except struct.error:
                pass  # end of file

        # Read and export substeps if available
        if sub_path.exists():
            _export_substeps(sub_path, node_ids, output_path)


def _export_substeps(sub_path, node_ids, output_path):
    """Export phase data as a separate JSON file per step."""
    phases_dir = output_path / "phases"
    phases_dir.mkdir(exist_ok=True)

    with open(sub_path, "rb") as f:
        magic = f.read(5)
        if magic != MAGIC:
            return
        version = struct.unpack("<H", f.read(2))[0]

        step_num = 0
        while True:
            step_num += 1
            try:
                n_phases = struct.unpack("<H", f.read(2))[0]
            except struct.error:
                break

            phases = []
            for _ in range(n_phases):
                line_len = struct.unpack("<I", f.read(4))[0]
                line_data = f.read(line_len)
                record = json.loads(line_data.decode("utf-8"))
                phases.append(record)

            path = phases_dir / f"phases_{step_num:04d}.json"
            with open(path, "w") as sf:
                json.dump(phases, sf, separators=(",", ":"))
