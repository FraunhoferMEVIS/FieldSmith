# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file verifies magnet configurations for geometric intersections.
#
#  magnet_geometry_verifier.py
#  Kostiantyn Lavronenko
#  20.08.2026
# -----------------------------------------------------------------------------


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.configurations.setup_npy import load_halbach_setup_npy
from fieldsmith.geometry.collision import find_box_intersections
from fieldsmith.geometry.cube_axes import cube_axes_from_directions, cube_axes_min_z_extent


MM_TO_M = 1e-3
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check a saved magnet configuration's geometry.")
    parser.add_argument("config", type=Path, nargs="?", help="Configuration checkpoint (.pt or .npy).")
    parser.add_argument(
        "--dimensions", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Override every magnet's full X/Y/Z body dimensions in metres.",
    )
    parser.add_argument(
        "--cube-size-mm", type=float,
        help="Override every magnet with this cube edge length in millimetres.",
    )
    parser.add_argument(
        "--frame-rule", choices=("argmin", "gs"), default="argmin",
        help="Rule used to derive complete body rotations from magnetization directions.",
    )
    parser.add_argument(
        "--touching-counts-as-intersection", action="store_true",
        help="Treat face, edge, or corner contact as an intersection.",
    )
    parser.add_argument(
        "--max-allowed-intersection-mm", type=float, default=1.0,
        help="Maximum penetration before the check fails (default: 1.0 mm).",
    )
    parser.add_argument("--log-file", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--visualization-directory", type=Path,
        help="Directory for intersection PNGs; defaults to a folder beside --log-file.",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = Path(input("Please enter the path to the configuration file: "))
    if args.dimensions is not None and args.cube_size_mm is not None:
        parser.error("--dimensions and --cube-size-mm cannot be used together")
    if args.max_allowed_intersection_mm < 0.0:
        parser.error("--max-allowed-intersection-mm must be non-negative")
    if args.cube_size_mm is not None and args.cube_size_mm <= 0.0:
        parser.error("--cube-size-mm must be positive")
    return args


def _load_configuration(path: Path) -> MagnetConfiguration:
    if path.suffix.lower() == ".npy":
        return load_halbach_setup_npy(path, dtype=torch.float64)
    return MagnetConfiguration.from_file(path, dtype=torch.float64)


def _dimensions_from_configuration(
    configuration: MagnetConfiguration, override: list[float] | None
) -> torch.Tensor:
    if override is not None:
        dimensions = configuration.positions.new_tensor(override).repeat(configuration.num_magnets, 1)
    elif "magnet_dimensions_m" in configuration.metadata:
        dimensions = torch.as_tensor(
            configuration.metadata["magnet_dimensions_m"],
            device=configuration.positions.device,
            dtype=configuration.positions.dtype,
        )
    elif "magnet_size_m" in configuration.metadata:
        size = float(configuration.metadata["magnet_size_m"])
        dimensions = configuration.positions.new_full((configuration.num_magnets, 3), size)
    else:
        raise ValueError(
            'configuration metadata needs "magnet_dimensions_m" or "magnet_size_m"; '
            "alternatively pass --dimensions or --cube-size-mm"
        )
    if dimensions.shape != (configuration.num_magnets, 3):
        raise ValueError('"magnet_dimensions_m" must have shape (M, 3).')
    if bool((dimensions <= 0).any()):
        raise ValueError("magnet dimensions must be positive.")
    return dimensions


def load_config(path: Path, dimensions_override: list[float] | None = None) -> dict[str, Any]:
    """Load a configuration into the verifier's stable dictionary representation."""
    configuration = _load_configuration(path)
    dimensions = _dimensions_from_configuration(configuration, dimensions_override)
    return {
        "configuration": configuration,
        "magnet_positions_m": configuration.positions,
        "magnet_orientations": configuration.orientations,
        "magnet_dimensions_m": dimensions,
    }


def center_distance_stats(positions_m: np.ndarray) -> dict[str, Any]:
    """Compute closest Euclidean and coordinate-axis center distances."""
    if positions_m.ndim != 2 or positions_m.shape[1] != 3:
        raise ValueError("positions_m must have shape (M, 3).")
    result = {}
    if positions_m.shape[0] < 2:
        for axis in "xyz":
            result[f"min_abs_d{axis}_m"] = None
            result[f"min_abs_d{axis}_mm"] = None
            result[f"pair_for_min_abs_d{axis}"] = None
            result[f"min_nonzero_abs_d{axis}_m"] = None
            result[f"min_nonzero_abs_d{axis}_mm"] = None
            result[f"pair_for_min_nonzero_abs_d{axis}"] = None
            result[f"zero_d{axis}_pair_count"] = 0
            result[f"example_zero_d{axis}_pair"] = None
        result.update({"min_center_distance_m": None, "min_center_distance_mm": None,
                       "pair_for_min_center_distance": None})
        return result

    index_i, index_j = np.triu_indices(positions_m.shape[0], k=1)
    deltas = np.abs(positions_m[index_j] - positions_m[index_i])
    distances = np.linalg.norm(deltas, axis=1)
    closest = int(np.argmin(distances))
    result.update({
        "min_center_distance_m": float(distances[closest]),
        "min_center_distance_mm": float(distances[closest] / MM_TO_M),
        "pair_for_min_center_distance": (int(index_i[closest]), int(index_j[closest])),
    })
    for axis_index, axis in enumerate("xyz"):
        values = deltas[:, axis_index]
        minimum = int(np.argmin(values))
        positive = np.flatnonzero(values > EPS)
        positive_minimum = int(positive[np.argmin(values[positive])]) if positive.size else None
        zero = np.flatnonzero(values <= EPS)
        result.update({
            f"min_abs_d{axis}_m": float(values[minimum]),
            f"min_abs_d{axis}_mm": float(values[minimum] / MM_TO_M),
            f"pair_for_min_abs_d{axis}": (int(index_i[minimum]), int(index_j[minimum])),
            f"min_nonzero_abs_d{axis}_m": None if positive_minimum is None else float(values[positive_minimum]),
            f"min_nonzero_abs_d{axis}_mm": None if positive_minimum is None else float(values[positive_minimum] / MM_TO_M),
            f"pair_for_min_nonzero_abs_d{axis}": None if positive_minimum is None else (
                int(index_i[positive_minimum]), int(index_j[positive_minimum])
            ),
            f"zero_d{axis}_pair_count": int(zero.size),
            f"example_zero_d{axis}_pair": None if not zero.size else (
                int(index_i[zero[0]]), int(index_j[zero[0]])
            ),
        })
    return result


def min_center_axis_distances(positions_m: np.ndarray) -> dict[str, Any]:
    """Backward-compatible name used by the original verifier."""
    return center_distance_stats(positions_m)


def find_intersections(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    dimensions: torch.Tensor,
    frame_rule: str,
    touching_counts: bool,
) -> dict[str, Any]:
    """Return intersection details for oriented cuboid magnets."""
    build_axes = cube_axes_min_z_extent if frame_rule == "gs" else cube_axes_from_directions
    axes = build_axes(orientations)
    pairs, depths = find_box_intersections(
        positions, axes, dimensions, touching_counts=touching_counts
    )
    intersections = []
    for pair, depth in zip(pairs.tolist(), depths.tolist()):
        delta = positions[pair[1]] - positions[pair[0]]
        delta_m = delta.tolist()
        intersections.append({
            "pair": pair,
            "center_delta_m": delta_m,
            "center_delta_mm": [value / MM_TO_M for value in delta_m],
            "abs_center_delta_m": [abs(value) for value in delta_m],
            "abs_center_delta_mm": [abs(value) / MM_TO_M for value in delta_m],
            "center_distance_m": float(torch.linalg.vector_norm(delta)),
            "center_distance_mm": float(torch.linalg.vector_norm(delta) / MM_TO_M),
            "penetration_depth_m": max(0.0, depth),
            "penetration_depth_mm": max(0.0, depth / MM_TO_M),
        })
    max_index = int(torch.argmax(depths)) if depths.numel() else None
    max_depth = max(0.0, float(depths[max_index])) if max_index is not None else 0.0
    return {
        "intersection_count": len(intersections),
        "has_intersections": bool(intersections),
        "intersections": intersections,
        "max_intersection_depth_m": max_depth,
        "max_intersection_depth_mm": max_depth / MM_TO_M,
        "max_intersection_pair": None if max_index is None else pairs[max_index].tolist(),
    }


def classify_intersection_severity(report: dict[str, Any], max_allowed_mm: float) -> dict[str, Any]:
    """Classify a report as ok, warning, or error."""
    max_depth = float(report.get("max_intersection_depth_mm", 0.0))
    if not report.get("has_intersections", False):
        return {"geometry_status": "ok", "should_fail": False, "warning_only": False,
                "message": "No intersections detected."}
    if max_depth <= max_allowed_mm:
        message = (f"Minor intersections detected. Maximum penetration depth is {max_depth:.6f} mm, "
                   f"which is within the allowed threshold of {max_allowed_mm:.6f} mm.")
        return {"geometry_status": "warning", "should_fail": False, "warning_only": True,
                "message": message}
    message = (f"Intersection too large. Maximum penetration depth is {max_depth:.6f} mm, "
               f"exceeding the allowed threshold of {max_allowed_mm:.6f} mm.")
    return {"geometry_status": "error", "should_fail": True, "warning_only": False,
            "message": message}


def build_report(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Build a complete serializable geometry report."""
    positions = config["magnet_positions_m"]
    dimensions = config["magnet_dimensions_m"]
    if args.cube_size_mm is not None:
        dimensions = positions.new_full(positions.shape, args.cube_size_mm * MM_TO_M)
    report = {
        "config_file": str(args.config),
        "magnet_count": int(positions.shape[0]),
        "position_unit": "m",
        "magnet_dimensions_m": dimensions.detach().cpu().numpy(),
        "geometry_mode": f"oriented_cuboids_from_magnetization_{args.frame_rule}",
        "touching_counts_as_intersection": bool(args.touching_counts_as_intersection),
    }
    report.update(center_distance_stats(positions.detach().cpu().numpy()))
    report.update(find_intersections(
        positions, config["magnet_orientations"], dimensions, args.frame_rule,
        args.touching_counts_as_intersection,
    ))
    report["max_allowed_intersection_mm"] = float(args.max_allowed_intersection_mm)
    report.update(classify_intersection_severity(report, args.max_allowed_intersection_mm))
    return report


def _format_distance(label: str, value: float | None, pair: tuple[int, int] | None) -> str:
    if value is None or pair is None:
        return f"  {label}: not available"
    return f"  {label} = {value:.6f} mm, pair = {tuple(pair)}"


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable geometry report."""
    print("=== Magnet Geometry Check ===")
    print(f"Config file: {report['config_file']}")
    print(f"Magnets: {report['magnet_count']}")
    print(f"Geometry mode: {report['geometry_mode']}")
    print(f"Touching counts as intersection: {report['touching_counts_as_intersection']}")
    print("\nNearest center-to-center Euclidean distance:")
    print(_format_distance("min |delta r|", report["min_center_distance_mm"],
                           report["pair_for_min_center_distance"]))
    print("\nSmallest non-zero center-to-center axis distances:")
    for axis in "xyz":
        print(_format_distance(
            f"min non-zero |d{axis}|", report[f"min_nonzero_abs_d{axis}_mm"],
            report[f"pair_for_min_nonzero_abs_d{axis}"],
        ))
    print("\nAxis-coordinate coincidences:")
    for axis in "xyz":
        print(f"  |d{axis}| = 0 for {report[f'zero_d{axis}_pair_count']} pair(s)")
    print(f"\nIntersections found: {report['intersection_count']}")
    for item in report["intersections"]:
        print(f"  pair {tuple(item['pair'])}: center distance {item['center_distance_mm']:.6f} mm, "
              f"penetration {item['penetration_depth_mm']:.6f} mm")
    print(f"\nGeometry status: {report['geometry_status'].upper()}")
    print(report["message"])
    if report["max_intersection_pair"] is not None:
        print(f"Maximum penetration: {report['max_intersection_depth_mm']:.6f} mm, "
              f"pair = {tuple(report['max_intersection_pair'])}")


def _box_vertices(
    centre: np.ndarray, axes: np.ndarray, dimensions: np.ndarray
) -> np.ndarray:
    """Return the eight world-space vertices of an oriented box."""
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=float)
    return centre + (signs * dimensions / 2.0) @ axes


def _box_faces(vertices: np.ndarray) -> list[np.ndarray]:
    """Return the six quadrilateral faces for eight box vertices."""
    indices = (
        (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
    )
    return [vertices[list(face)] for face in indices]


def _set_equal_3d_limits(axis: Any, vertices: np.ndarray, padding: float = 0.05) -> None:
    """Set equal-scale limits around a collection of 3D vertices."""
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    centre = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, EPS) * (1.0 + padding)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def save_intersection_visualizations(
    report: dict[str, Any],
    config: dict[str, Any],
    output_directory: Path,
    frame_rule: str,
) -> list[str]:
    """Save one whole-setup and close-up visualization per intersecting pair."""
    import matplotlib.pyplot as plt

    if not report["has_intersections"]:
        return []
    output_directory.mkdir(parents=True, exist_ok=True)
    positions = config["magnet_positions_m"].detach().cpu().numpy()
    dimensions = config["magnet_dimensions_m"].detach().cpu().numpy()
    orientations = config["magnet_orientations"]
    build_axes = cube_axes_min_z_extent if frame_rule == "gs" else cube_axes_from_directions
    axes = build_axes(orientations).detach().cpu().numpy()
    box_vertices = [
        _box_vertices(position, body_axes, size)
        for position, body_axes, size in zip(positions, axes, dimensions)
    ]
    all_vertices = np.concatenate(box_vertices)
    background_faces = [face for vertices in box_vertices for face in _box_faces(vertices)]
    saved_paths = []

    for item in report["intersections"]:
        index_i, index_j = item["pair"]
        figure = plt.figure(figsize=(14, 7), constrained_layout=True)
        overview = figure.add_subplot(1, 2, 1, projection="3d")
        closeup = figure.add_subplot(1, 2, 2, projection="3d")

        overview.add_collection3d(Poly3DCollection(
            background_faces, facecolors="lightgray", edgecolors="gray",
            linewidths=0.25, alpha=0.08,
        ))
        colors = ("tab:red", "tab:blue")
        pair_vertices = np.concatenate((box_vertices[index_i], box_vertices[index_j]))
        for magnet_index, color in zip((index_i, index_j), colors):
            faces = _box_faces(box_vertices[magnet_index])
            overview.add_collection3d(Poly3DCollection(
                faces, facecolors=color, edgecolors="black", linewidths=1.0, alpha=0.8,
            ))
            closeup.add_collection3d(Poly3DCollection(
                faces, facecolors=color, edgecolors="black", linewidths=1.5, alpha=0.5,
            ))
            closeup.text(*positions[magnet_index], str(magnet_index), color=color, fontsize=11)

        _set_equal_3d_limits(overview, all_vertices)
        _set_equal_3d_limits(closeup, pair_vertices, padding=0.25)
        for axis in (overview, closeup):
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_zlabel("z [m]")
        overview.set_title(f"Pair ({index_i}, {index_j}) in complete setup")
        closeup.set_title(
            f"Intersection close-up\npenetration = {item['penetration_depth_mm']:.6f} mm"
        )
        figure.suptitle(f"Intersecting magnets {index_i} and {index_j}")
        output_path = output_directory / f"intersection_{index_i}_{index_j}.png"
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        saved_paths.append(str(output_path.resolve()))
    return saved_paths


class NumpyJSONEncoder(json.JSONEncoder):
    """Encode NumPy values in JSON reports."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
