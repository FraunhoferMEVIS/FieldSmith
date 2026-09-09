import pytest
import torch

from fieldsmith.objectives import (
    CollisionPenalty,
    FieldFloor,
    SmoothedMinimaxObjective,
    field_band_loss,
    peak_to_peak_ppm,
    relative_deviation,
    rms_deviation_ppm,
    soft_peak_to_peak_ppm,
)


def test_relative_deviation_uses_field_magnitude_and_mean() -> None:
    field = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -3.0]],
        dtype=torch.float64,
    )

    deviation = relative_deviation(field)

    assert torch.allclose(
        deviation,
        torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "field",
    [
        torch.zeros(3),
        torch.zeros((2, 2)),
        torch.zeros((2, 3, 1)),
    ],
)
def test_relative_deviation_rejects_invalid_field_shapes(
    field: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"field must have shape \(P, 3\)"):
        relative_deviation(field)


def test_exact_peak_to_peak_and_rms_match_known_values() -> None:
    field = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
    )

    assert peak_to_peak_ppm(field).item() == pytest.approx(1_000_000.0)
    assert rms_deviation_ppm(field).item() == pytest.approx(
        (1.0 / 6.0) ** 0.5 * 1_000_000.0,
    )


def test_soft_peak_to_peak_is_zero_for_uniform_field() -> None:
    field = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    assert soft_peak_to_peak_ppm(field, alpha=100.0).item() == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_soft_peak_to_peak_approaches_exact_value_and_is_differentiable() -> None:
    field = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    low_alpha = soft_peak_to_peak_ppm(field, alpha=2.0)
    high_alpha = soft_peak_to_peak_ppm(field, alpha=200.0)
    exact = peak_to_peak_ppm(field)
    high_alpha.backward()

    assert 0.0 < low_alpha.item() < high_alpha.item() < exact.item()
    assert exact.item() - high_alpha.item() < exact.item() - low_alpha.item()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize(
    ("magnitude", "expected"),
    [
        (0.15, 0.0),
        (0.05, 0.25),
        (0.30, 0.25),
    ],
)
def test_field_band_loss_penalizes_only_values_outside_band(
    magnitude: float,
    expected: float,
) -> None:
    field = torch.tensor([[magnitude, 0.0, 0.0]], dtype=torch.float64)

    assert field_band_loss(field, lower_t=0.1, upper_t=0.2).item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("lower_t", "upper_t"),
    [
        (0.0, 0.2),
        (-0.1, 0.2),
        (0.2, 0.1),
    ],
)
def test_field_band_loss_rejects_invalid_bounds(
    lower_t: float,
    upper_t: float,
) -> None:
    with pytest.raises(ValueError, match="0 < lower_t <= upper_t"):
        field_band_loss(torch.ones((1, 3)), lower_t, upper_t)


def test_smoothed_minimax_objective_combines_weighted_terms() -> None:
    field = torch.tensor(
        [[0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=torch.float64,
    )
    objective = SmoothedMinimaxObjective(
        alpha=50.0,
        rms_weight=0.25,
        band_weight=4.0,
        band_t=(0.1, 0.2),
    )

    loss = objective({"B": field})
    expected = (
        soft_peak_to_peak_ppm(field, alpha=50.0)
        + 0.25 * rms_deviation_ppm(field)
        + 4.0 * field_band_loss(field, 0.1, 0.2)
    )

    assert torch.allclose(loss, expected)


def test_smoothed_minimax_alpha_growth_stops_at_maximum() -> None:
    objective = SmoothedMinimaxObjective(
        alpha=10.0,
        alpha_max=25.0,
        alpha_growth=2.0,
    )

    assert objective.step() == pytest.approx(20.0)
    assert objective.step() == pytest.approx(25.0)
    assert objective.step() == pytest.approx(25.0)


def test_field_floor_calculates_penalty_and_updates_multiplier() -> None:
    field = torch.tensor([[0.08, 0.0, 0.0]], dtype=torch.float64)
    floor = FieldFloor(floor_t=0.1, rho=100.0, multiplier=2.0)

    assert floor.violation(field).item() == pytest.approx(0.02)
    assert floor(field).item() == pytest.approx(0.06)
    assert floor.update_multiplier(field) == pytest.approx(4.0)

    sufficient_field = torch.tensor([[0.12, 0.0, 0.0]], dtype=torch.float64)
    assert floor.violation(sufficient_field).item() == pytest.approx(0.0)
    assert floor(sufficient_field).item() == pytest.approx(0.0)
    assert floor.update_multiplier(sufficient_field) == pytest.approx(4.0)


def test_field_floor_rejects_nonpositive_floor() -> None:
    with pytest.raises(ValueError, match="floor_t must be positive"):
        FieldFloor(0.0)


def test_collision_penalty_applies_margin_weight_and_gradients() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    axes = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    penalty_fn = CollisionPenalty(
        half_extent_m=0.5,
        margin_m=0.05,
        weight=2.0,
    )

    penalty = penalty_fn(
        positions,
        axes,
        torch.tensor([0]),
        torch.tensor([1]),
    )
    penalty.backward()

    assert penalty.item() == pytest.approx(2.0 * 0.15**2)
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_collision_penalty_is_zero_within_allowed_margin() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        dtype=torch.float64,
    )
    axes = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    penalty_fn = CollisionPenalty(half_extent_m=0.5, margin_m=0.05)

    penalty = penalty_fn(
        positions,
        axes,
        torch.tensor([0]),
        torch.tensor([1]),
    )

    assert penalty.item() == pytest.approx(0.0)
