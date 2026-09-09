import torch

from fieldsmith.models.small_halbach import (
    SmallHalbachAngleModel,
    SmallHalbachDistanceModel,
    SmallHalbachRadiusModel,
    build_small_halbach_configuration,
    distance_stage_loss,
    radius_stage_loss,
)


def test_small_halbach_stage_shapes_and_handoff() -> None:
    configuration = build_small_halbach_configuration(
        num_rings=3,
        num_magnets_per_ring=(4, 8),
        ring_radius_inner=0.08,
        ring_radius_outer=0.10,
        distance_between_rings=0.04,
        magnet_size=0.012,
        remanence=1.31,
        device="cpu",
    )
    fov = (0.02, 0.02, 0.02)
    resolution = (0.01, 0.01, 0.01)

    distance_model = SmallHalbachDistanceModel(configuration, fov, resolution, 0.01)
    distance_output = distance_model()
    distance_stage_loss(distance_output).backward()
    assert configuration.num_magnets == 36
    assert distance_model.z_levels_trainable.grad is not None

    radius_model = SmallHalbachRadiusModel(
        distance_model.get_configuration(), fov, resolution, 0.01,
        num_rings=3, num_magnets_per_ring=(4, 8),
        ring_radius_inner=0.08, ring_radius_outer=0.10,
    )
    radius_output = radius_model()
    radius_stage_loss(radius_output, expected_field=0.04).backward()
    assert radius_model.inner_factor.grad is not None
    assert radius_model.outer_factor.grad is not None

    angle_model = SmallHalbachAngleModel(radius_model.get_configuration(), fov, resolution, 0.01)
    angle_output = angle_model()
    angle_output["B"].square().sum().backward()
    assert angle_model.magnet_angles_phi.grad is not None
    assert torch.isfinite(angle_model.magnet_angles_phi.grad).all()
