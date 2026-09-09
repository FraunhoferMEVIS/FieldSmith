# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains overlap geometry for oriented magnet bodies.
#
#  collision.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch


def candidate_pairs(positions: torch.Tensor, reach: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the index pairs of magnets close enough to possibly overlap.

    Testing every pair is quadratic and pointless: two bodies farther apart than
    the sum of their circumscribed radii cannot touch. A generous ``reach`` is
    the right way to be safe, since the pair list is built once for a geometry
    that may then move.

    Args:
        positions: Magnet centres in meters, shape (M, 3).
        reach: Centre distance beyond which a pair is ignored, in meters.

    Returns:
        Two index tensors (i, j) with i < j, shape (P,) each.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (M, 3).")
    distance = torch.cdist(positions, positions)
    near = (distance <= reach).triu(diagonal=1).nonzero(as_tuple=False)
    return near[:, 0], near[:, 1]


def box_penetration(
    centres_i: torch.Tensor,
    centres_j: torch.Tensor,
    axes_i: torch.Tensor,
    axes_j: torch.Tensor,
    half_extent: float,
) -> torch.Tensor:
    """
    Penetration depth of pairs of oriented cubes, by the separating axis theorem.

    Two convex boxes are disjoint exactly when some axis separates their
    projections. It suffices to test fifteen: the three face normals of each box
    and the nine pairwise cross products of their edges. The depth returned is
    the least overlap over those axes, so it is positive only when every axis
    overlaps, and it is the distance the pair must be moved apart to just touch.

    Differentiable in both the centres and the frames, so it can be used as a
    penalty during optimisation as well as for a hard check afterwards.

    Args:
        centres_i, centres_j: Paired magnet centres in meters, shape (P, 3).
        axes_i, axes_j: Paired body frames, shape (P, 3, 3); row k is local axis k.
        half_extent: Half the cube edge in meters, plus any clearance to enforce.

    Returns:
        Penetration depth in meters, shape (P,); non-positive means disjoint.
    """
    axes = [axes_i[:, k, :] for k in range(3)] + [axes_j[:, k, :] for k in range(3)]
    for a in range(3):
        for b in range(3):
            axes.append(torch.linalg.cross(axes_i[:, a, :], axes_j[:, b, :]))

    offset = centres_j - centres_i
    widest = torch.full((centres_i.shape[0],), -torch.inf,
                        dtype=centres_i.dtype, device=centres_i.device)
    for axis in axes:
        length = axis.norm(dim=-1, keepdim=True)
        unit = axis / length.clamp_min(1e-12)
        gap = (offset * unit).sum(-1).abs() - half_extent * sum(
            (axes_i[:, k, :] * unit).sum(-1).abs() + (axes_j[:, k, :] * unit).sum(-1).abs()
            for k in range(3)
        )
        # A cross product of near-parallel edges carries no information; drop it
        # rather than let its normalised direction become arbitrary.
        widest = torch.maximum(widest, torch.where(length.squeeze(-1) < 1e-9,
                                                   torch.full_like(gap, -torch.inf), gap))
    return -widest


def oriented_box_penetration(
    centres_i: torch.Tensor,
    centres_j: torch.Tensor,
    axes_i: torch.Tensor,
    axes_j: torch.Tensor,
    dimensions_i: torch.Tensor,
    dimensions_j: torch.Tensor,
) -> torch.Tensor:
    """Return penetration depths for pairs of arbitrarily sized oriented boxes.

    ``dimensions_i`` and ``dimensions_j`` contain the full side lengths along
    the corresponding rows of ``axes_i`` and ``axes_j``. A positive result is
    an intersection, zero is contact, and a negative result is separation.
    """
    pair_count = centres_i.shape[0]
    expected_centres = (pair_count, 3)
    expected_axes = (pair_count, 3, 3)
    if centres_i.shape != expected_centres or centres_j.shape != expected_centres:
        raise ValueError("centres_i and centres_j must have shape (P, 3).")
    if axes_i.shape != expected_axes or axes_j.shape != expected_axes:
        raise ValueError("axes_i and axes_j must have shape (P, 3, 3).")
    if dimensions_i.shape != expected_centres or dimensions_j.shape != expected_centres:
        raise ValueError("dimensions_i and dimensions_j must have shape (P, 3).")
    if bool((dimensions_i <= 0).any() or (dimensions_j <= 0).any()):
        raise ValueError("box dimensions must be positive.")

    half_i = dimensions_i / 2.0
    half_j = dimensions_j / 2.0
    separating_axes = [axes_i[:, k, :] for k in range(3)] + [axes_j[:, k, :] for k in range(3)]
    for index_i in range(3):
        for index_j in range(3):
            separating_axes.append(torch.linalg.cross(axes_i[:, index_i, :], axes_j[:, index_j, :]))

    offset = centres_j - centres_i
    widest = torch.full(
        (pair_count,), -torch.inf, dtype=centres_i.dtype, device=centres_i.device
    )
    for axis in separating_axes:
        length = axis.norm(dim=-1, keepdim=True)
        unit = axis / length.clamp_min(1e-12)
        radius_i = sum(
            half_i[:, k] * (axes_i[:, k, :] * unit).sum(-1).abs() for k in range(3)
        )
        radius_j = sum(
            half_j[:, k] * (axes_j[:, k, :] * unit).sum(-1).abs() for k in range(3)
        )
        gap = (offset * unit).sum(-1).abs() - radius_i - radius_j
        valid_gap = torch.where(
            length.squeeze(-1) < 1e-9, torch.full_like(gap, -torch.inf), gap
        )
        widest = torch.maximum(widest, valid_gap)
    return -widest


def find_box_intersections(
    positions: torch.Tensor,
    axes: torch.Tensor,
    dimensions: torch.Tensor,
    tolerance: float = 0.0,
    touching_counts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return intersecting index pairs and their penetration depths.

    ``tolerance`` is the penetration that is allowed before a pair is reported.
    Merely touching boxes therefore do not intersect at the default value.
    """
    magnet_count = positions.shape[0]
    if positions.shape != (magnet_count, 3):
        raise ValueError("positions must have shape (M, 3).")
    if axes.shape != (magnet_count, 3, 3):
        raise ValueError("axes must have shape (M, 3, 3).")
    if dimensions.shape != (magnet_count, 3):
        raise ValueError("dimensions must have shape (M, 3).")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    if bool((dimensions <= 0).any()):
        raise ValueError("magnet dimensions must be positive.")

    radii = torch.linalg.vector_norm(dimensions, dim=1) / 2.0
    index_i, index_j = candidate_pairs(positions, 2.0 * float(radii.max()) if magnet_count else 0.0)
    if index_i.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device), positions.new_empty(0)
    depth = oriented_box_penetration(
        positions[index_i], positions[index_j], axes[index_i], axes[index_j],
        dimensions[index_i], dimensions[index_j],
    )
    intersects = depth >= -1e-12 if touching_counts else depth > tolerance
    return torch.stack((index_i[intersects], index_j[intersects]), dim=1), depth[intersects]


def count_overlaps(
    positions: torch.Tensor,
    axes: torch.Tensor,
    magnet_size: float,
    tolerance: float = 4e-4,
    reach: float | None = None,
) -> int:
    """
    Count magnet pairs that overlap by more than a tolerance.

    This is the manufacturability gate: it runs on the true body size with no
    clearance added, so a design passing it can be built. The default tolerance
    of 0.4 mm matches the acceptance used for the reference designs.
    """
    if reach is None:
        reach = 2.0 * magnet_size * 3**0.5
    index_i, index_j = candidate_pairs(positions, reach)
    depth = box_penetration(positions[index_i], positions[index_j],
                            axes[index_i], axes[index_j], magnet_size / 2.0)
    return int((depth > tolerance).sum())
