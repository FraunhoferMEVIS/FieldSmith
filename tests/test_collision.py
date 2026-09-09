import pytest
import torch

from fieldsmith.geometry.collision import (
    box_penetration,
    candidate_pairs,
    count_overlaps,
    find_box_intersections,
    oriented_box_penetration,
)


def test_candidate_pairs_returns_only_near_unique_pairs() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    index_i, index_j = candidate_pairs(positions, reach=1.0)

    assert index_i.tolist() == [0]
    assert index_j.tolist() == [1]


@pytest.mark.parametrize(
    "positions",
    [
        torch.zeros(3),
        torch.zeros((2, 2)),
    ],
)
def test_candidate_pairs_rejects_invalid_position_shapes(
    positions: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"positions must have shape \(M, 3\)"):
        candidate_pairs(positions, reach=1.0)


def test_box_penetration_returns_positive_overlap_and_negative_separation() -> None:
    centres_i = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    centres_j = torch.tensor(
        [[0.8, 0.0, 0.0], [1.2, 0.0, 0.0]],
        dtype=torch.float64,
    )
    axes = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)

    depth = box_penetration(
        centres_i,
        centres_j,
        axes,
        axes,
        half_extent=0.5,
    )

    assert torch.allclose(depth, torch.tensor([0.2, -0.2], dtype=torch.float64))


def test_oriented_box_penetration_supports_different_dimensions() -> None:
    centres_i = torch.zeros((1, 3), dtype=torch.float64)
    centres_j = torch.tensor([[1.15, 0.0, 0.0]], dtype=torch.float64)
    axes_i = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    axes_j = torch.tensor(
        [[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    dimensions_i = torch.tensor([[2.0, 0.5, 0.5]], dtype=torch.float64)
    dimensions_j = torch.tensor([[0.5, 2.0, 0.5]], dtype=torch.float64)

    depth = oriented_box_penetration(
        centres_i,
        centres_j,
        axes_i,
        axes_j,
        dimensions_i,
        dimensions_j,
    )

    assert torch.allclose(depth, torch.tensor([0.5], dtype=torch.float64))


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (
            "centres_i",
            torch.zeros((1, 2)),
            r"centres_i and centres_j must have shape \(P, 3\)",
        ),
        (
            "centres_j",
            torch.zeros((1, 2)),
            r"centres_i and centres_j must have shape \(P, 3\)",
        ),
        (
            "axes_i",
            torch.zeros((1, 3, 2)),
            r"axes_i and axes_j must have shape \(P, 3, 3\)",
        ),
        (
            "axes_j",
            torch.zeros((1, 3, 2)),
            r"axes_i and axes_j must have shape \(P, 3, 3\)",
        ),
        (
            "dimensions_i",
            torch.ones((1, 2)),
            r"dimensions_i and dimensions_j must have shape \(P, 3\)",
        ),
        (
            "dimensions_j",
            torch.ones((1, 2)),
            r"dimensions_i and dimensions_j must have shape \(P, 3\)",
        ),
        (
            "dimensions_i",
            torch.tensor([[1.0, 0.0, 1.0]]),
            "box dimensions must be positive",
        ),
        (
            "dimensions_j",
            torch.tensor([[1.0, -1.0, 1.0]]),
            "box dimensions must be positive",
        ),
    ],
)
def test_oriented_box_penetration_rejects_invalid_inputs(
    argument: str,
    value: torch.Tensor,
    message: str,
) -> None:
    arguments = {
        "centres_i": torch.zeros((1, 3)),
        "centres_j": torch.ones((1, 3)),
        "axes_i": torch.eye(3).unsqueeze(0),
        "axes_j": torch.eye(3).unsqueeze(0),
        "dimensions_i": torch.ones((1, 3)),
        "dimensions_j": torch.ones((1, 3)),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        oriented_box_penetration(**arguments)


def test_find_box_intersections_uses_dimensions_and_rotation() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]], dtype=torch.float64)
    dimensions = torch.tensor([[2.0, 0.5, 0.5], [2.0, 0.5, 0.5]], dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    rotation = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    axes = torch.stack((identity, rotation))

    pairs, depths = find_box_intersections(positions, axes, dimensions)

    assert pairs.tolist() == [[0, 1]]
    assert torch.allclose(depths, torch.tensor([0.1], dtype=torch.float64))


def test_find_box_intersections_excludes_contact_and_respects_tolerance() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float64)
    dimensions = torch.ones((2, 3), dtype=torch.float64)
    axes = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)

    pairs, _ = find_box_intersections(positions, axes, dimensions, tolerance=0.1)

    assert pairs.numel() == 0


def test_find_box_intersections_can_include_touching_boxes() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    dimensions = torch.ones((2, 3), dtype=torch.float64)
    axes = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)

    excluded_pairs, _ = find_box_intersections(
        positions,
        axes,
        dimensions,
        touching_counts=False,
    )
    included_pairs, depths = find_box_intersections(
        positions,
        axes,
        dimensions,
        touching_counts=True,
    )

    assert excluded_pairs.shape == (0, 2)
    assert included_pairs.tolist() == [[0, 1]]
    assert torch.allclose(depths, torch.zeros(1, dtype=torch.float64), atol=1e-12)


def test_find_box_intersections_handles_empty_configuration() -> None:
    positions = torch.empty((0, 3), dtype=torch.float64)
    axes = torch.empty((0, 3, 3), dtype=torch.float64)
    dimensions = torch.empty((0, 3), dtype=torch.float64)

    pairs, depths = find_box_intersections(positions, axes, dimensions)

    assert pairs.shape == (0, 2)
    assert pairs.dtype == torch.long
    assert depths.shape == (0,)
    assert depths.dtype == positions.dtype


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("positions", torch.zeros(3), r"positions must have shape \(M, 3\)"),
        ("positions", torch.zeros((2, 2)), r"positions must have shape \(M, 3\)"),
        ("axes", torch.zeros((2, 3, 2)), r"axes must have shape \(M, 3, 3\)"),
        ("dimensions", torch.ones((2, 2)), r"dimensions must have shape \(M, 3\)"),
        ("tolerance", -0.1, "tolerance must be non-negative"),
        (
            "dimensions",
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
            "magnet dimensions must be positive",
        ),
    ],
)
def test_find_box_intersections_rejects_invalid_inputs(
    argument: str,
    value: torch.Tensor | float,
    message: str,
) -> None:
    arguments = {
        "positions": torch.zeros((2, 3)),
        "axes": torch.eye(3).repeat(2, 1, 1),
        "dimensions": torch.ones((2, 3)),
        "tolerance": 0.0,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        find_box_intersections(**arguments)


def test_count_overlaps_uses_default_tolerance_and_optional_reach() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    axes = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)

    assert count_overlaps(
        positions,
        axes,
        magnet_size=1.0,
        tolerance=0.1,
    ) == 1
    assert count_overlaps(
        positions,
        axes,
        magnet_size=1.0,
        tolerance=0.1,
        reach=0.5,
    ) == 0
