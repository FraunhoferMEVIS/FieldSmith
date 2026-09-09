import pytest
import torch

from fieldsmith.metrics import (
    component_field_strength,
    component_peak_to_peak_ppm,
    relative_field_strength_shortfall_loss,
)


def test_component_metrics_use_absolute_signed_mean() -> None:
    field = torch.tensor([-2.0, -1.0], dtype=torch.float64)

    assert component_field_strength(field).item() == pytest.approx(1.5)
    assert component_peak_to_peak_ppm(field).item() == pytest.approx(1e6 / 1.5)


def test_relative_field_strength_shortfall_uses_reference_without_penalizing_gain() -> None:
    weaker = torch.tensor([1.0, 1.0], dtype=torch.float64)
    stronger = torch.tensor([3.0, 3.0], dtype=torch.float64)

    assert relative_field_strength_shortfall_loss(weaker, 2.0).item() == pytest.approx(1.0)
    assert relative_field_strength_shortfall_loss(stronger, 2.0).item() == pytest.approx(0.0)


def test_relative_field_strength_shortfall_rejects_nonpositive_reference() -> None:
    with pytest.raises(ValueError, match="reference_strength must be positive"):
        relative_field_strength_shortfall_loss(torch.ones(2), 0.0)
