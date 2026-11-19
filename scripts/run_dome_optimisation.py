# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file runs an open cap (dome) magnet angle optimization.
#
#  run_dome_optimisation.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.configurations.dome_config import DomeConfig
from fieldsmith.geometry.cap_lattice import CapLatticeGeometry, head_envelope_mm
from fieldsmith.geometry.collision import candidate_pairs
from fieldsmith.geometry.cube_axes import cube_axes_min_z_extent
from fieldsmith.geometry.dome import DomeGeometry
from fieldsmith.geometry.halbach_ring import make_spherical_mask
from fieldsmith.geometry.lsq_seed import lsq_seed_orientations
from fieldsmith.logger import WandbLogger
from fieldsmith.callbacks import BestStateTracker, GeometryRefresh, PeriodicTruthEvaluation
from fieldsmith.geometry.roi import (load_evaluation_points, roi_boundary_points,
                                     roi_mask_fn)
from fieldsmith.models.dome import DomeAngleModel, DomePositionModel
from fieldsmith.objectives import CollisionPenalty, SmoothedMinimaxObjective, peak_to_peak_ppm
from fieldsmith.point_sampler import (CartesianGridSampler, MaskedPointSampler,
                                      PredefinedPointSampler, UnionPointSampler)
from fieldsmith.train_loop import TrainLoop

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dome_geometry.cfg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize an open cap magnet array with smoothed worst-case homogeneity.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--geometry", choices=("dome", "head"), default="dome",
                        help="Rule to build the starting arrangement from.")
    parser.add_argument("--ring-pitch-mm", type=float, default=16.5,
                        help="Radial and circumferential pitch of the head lattice.")
    parser.add_argument("--head-outer-mm", type=float, default=144.6,
                        help="Outer radius of the head envelope.")
    parser.add_argument("--head-zmax-mm", type=float, default=130.0,
                        help="Highest plate of the head envelope. Plates sit every 20 mm from z = 10 mm.")
    parser.add_argument("--init-design", type=str, default="",
                        help="Start from a stored .pt design instead of building and seeding.")
    parser.add_argument("--seed-max-points", type=int, default=4000,
                        help="Most region points the least-squares seed solves on.")
    parser.add_argument("--fov-resolution-mm", type=float, default=0.0,
                        help="Training grid spacing. 0 keeps the configured one.")
    parser.add_argument("--train-boundary-points", type=int, default=0,
                        help="Points on the region boundary added to the training set.")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lock-elongated-polar", action="store_true",
                        help="Keep every body that is not a cube in its plane.")
    parser.add_argument("--polar-lr", type=float, default=-1.0,
                        help="Step for the out-of-plane angle. -1 follows --lr, 0 freezes it.")
    parser.add_argument("--alpha", type=float, default=2.0e3, help="Initial log-sum-exp sharpness.")
    parser.add_argument("--alpha-max", type=float, default=2.0e4)
    parser.add_argument("--alpha-growth", type=float, default=1.005, help="Geometric alpha ramp per epoch.")
    parser.add_argument("--alpha-auto", type=float, default=0.0,
                        help="Keep alpha times the relative spread at this value. 0 keeps the geometric ramp.")
    parser.add_argument("--rms-weight", type=float, default=0.0)
    parser.add_argument("--band-weight", type=float, default=0.0, help="Weight of the mean-field band penalty, in ppm.")
    parser.add_argument("--band-lower-t", type=float, default=0.045)
    parser.add_argument("--band-upper-t", type=float, default=0.500)
    parser.add_argument("--double-precision", action="store_true", help="Run in float64; recommended below 1000 ppm.")
    parser.add_argument("--target", type=str, default="",
                        help='Seed target field direction as "x,y,z"; empty uses the geometry default.')
    parser.add_argument("--roi-shape", choices=("half_sphere", "sphere"),
                        default="half_sphere", help="Shape of the region the field is optimised over.")
    parser.add_argument("--roi-radius-mm", type=float, default=0.0,
                        help="Region radius. 0 keeps the configured one.")
    parser.add_argument("--roi-zmin-mm", type=float, default=-1000.0,
                        help="Region floor height. -1000 keeps the configured one.")
    parser.add_argument("--positions", action="store_true",
                        help="Optimise magnet centres as well as angles.")
    parser.add_argument("--position-bound-mm", type=float, default=2.0,
                        help="How far a centre may move in x and y.")
    parser.add_argument("--position-bound-axial-mm", type=float, default=None,
                        help="How far a centre may move in z. Defaults to --position-bound-mm.")
    parser.add_argument("--position-base-design", type=str, default="",
                        help="Design whose centres the bound is measured from, usually the original seats.")
    parser.add_argument("--position-lr", type=float, default=1.0e-5,
                        help="Adam step for the centre offsets, in meters.")
    parser.add_argument("--in-plane", action="store_true",
                        help="Keep centre displacement in the plate plane.")
    parser.add_argument("--best-allow-overlaps", action="store_true",
                        help="Let the best-design tracker keep designs with overlaps.")
    parser.add_argument("--collision-weight", type=float, default=1.0e10,
                        help="Weight of the overlap penalty.")
    parser.add_argument("--collision-margin-mm", type=float, default=-0.2,
                        help="Positive tolerates penetration, negative demands a gap.")
    parser.add_argument("--refresh-every", type=int, default=1200,
                        help="Bake drift and rebuild the coupling on this cadence.")
    parser.add_argument("--evaluate-every", type=int, default=1200,
                        help="Measure the design on the full region on this cadence.")
    parser.add_argument("--evaluation-points", type=str, default="",
                        help="Stored region for the truth evaluation; empty uses the training points.")
    parser.add_argument("--best-design", type=str, default="best_design.pt")
    parser.add_argument("--grad-clip-norm", type=float, default=25.0)
    parser.add_argument("--frame-rule", type=str, default="argmin",
                        choices=("argmin", "gs", "world", "seated"),
                        help="Body frame. 'world' keeps bodies as built, 'seated' turns cubes with "
                             "their moments and bars only about the bore axis.")
    parser.add_argument("--chunk-points", type=int, default=32768)
    parser.add_argument("--log-each-n-epochs", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="fieldsmith")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Under the default argmin frame, the gate, the truth reading and the clearance
    # penalty use the pocket frame ("gs") that the plates are cut in.
    gate_frame = "gs" if args.frame_rule == "argmin" else args.frame_rule
    timestamp = datetime.now().strftime("%Y_%m_%d/%Hh%Mm%Ss_dome")
    log_dir = Path("logs") / timestamp
    dome_config = DomeConfig.from_file(args.config)
    config = vars(args)
    config["config"] = str(args.config)
    config.update(dome_config.as_log_config())
    dtype = torch.float64 if args.double_precision else torch.float32
    if args.fov_resolution_mm > 0.0:
        step = args.fov_resolution_mm * 1e-3
        dome_config = replace(dome_config, resolution=(step, step, step))
        config.update(dome_config.as_log_config())
    if args.roi_radius_mm > 0.0 or args.roi_zmin_mm > -1000.0:
        # The training grid has to cover whatever the region reaches.
        radius = args.roi_radius_mm * 1e-3 if args.roi_radius_mm > 0.0 else dome_config.fov_radius
        z_min = args.roi_zmin_mm * 1e-3 if args.roi_zmin_mm > -1000.0 else dome_config.fov_z_min
        reach = max(radius, abs(z_min))
        extent = 2.0 * (reach + 0.005)
        dome_config = replace(dome_config, fov_radius=radius, fov_z_min=z_min,
                              fov=(extent, extent, extent))
        config.update(dome_config.as_log_config())
    roi_params = {"shape": args.roi_shape, "a_m": dome_config.fov_radius,
                  "z_min_m": dome_config.fov_z_min}

    if args.geometry == "head":
        # Plates of concentric rings inside the head envelope.
        z_levels = [float(z) for z in range(10, int(args.head_zmax_mm) + 1, 20)]
        envelope = head_envelope_mm(r_out_mm=args.head_outer_mm, z_levels_mm=tuple(sorted(z_levels)))
        geometry = CapLatticeGeometry(
            envelope_mm=envelope,
            ring_pitch_mm=args.ring_pitch_mm,
            magnet_size=dome_config.magnet_size,
            remanence=dome_config.remanence,
            device=args.device,
            dtype=dtype,
        )
        geometry.target_direction = (0.0, 0.0, 1.0)
    else:
        geometry = DomeGeometry(
            aperture_radius=dome_config.aperture_radius,
            dome_radius=dome_config.dome_radius,
            dome_height=dome_config.dome_height,
            n_layers=dome_config.n_layers,
            magnet_size=dome_config.magnet_size,
            remanence=dome_config.remanence,
            min_clearance=dome_config.min_clearance,
            device=args.device,
            dtype=dtype,
        )
    if args.init_design:
        # A restart takes positions and orientations from the file, without a new seed.
        initial_configuration = MagnetConfiguration.from_file(args.init_design, device=args.device, dtype=dtype)
        initial_configuration.metadata.setdefault("geometry", args.geometry)
    else:
        initial_configuration = geometry.build()
    grid_sampler = CartesianGridSampler(
        fov=dome_config.fov,
        resolution=dome_config.resolution,
        device=args.device,
        dtype=dtype,
    )
    point_sampler = MaskedPointSampler(base_sampler=grid_sampler,
                                       mask_fn=roi_mask_fn(**roi_params))
    if args.train_boundary_points > 0:
        boundary = roi_boundary_points(
            **roi_params, n_points=args.train_boundary_points,
            device=args.device, dtype=dtype,
        )
        point_sampler = UnionPointSampler((point_sampler, PredefinedPointSampler(boundary)))

    def build_model(configuration):
        configuration.metadata["frame_rule"] = args.frame_rule
        configuration.metadata["lock_elongated_polar"] = args.lock_elongated_polar
        model_class = DomePositionModel if args.positions else DomeAngleModel
        axial_mm = args.position_bound_mm if args.position_bound_axial_mm is None else args.position_bound_axial_mm
        bound = (args.position_bound_mm * 1e-3 if args.position_bound_axial_mm is None
                 else (args.position_bound_mm * 1e-3, args.position_bound_mm * 1e-3, axial_mm * 1e-3))
        extra = {"position_bound_m": bound, "in_plane": args.in_plane} if args.positions else {}
        if args.positions and args.position_base_design:
            extra["base_positions"] = MagnetConfiguration.from_file(args.position_base_design, device=args.device,
                                                                    dtype=dtype).positions
        return model_class(
            configuration=configuration,
            fov=dome_config.fov,
            resolution=dome_config.resolution,
            fov_radius=dome_config.fov_radius,
            device=args.device,
            dtype=dtype,
            chunk_points=args.chunk_points,
            point_sampler=point_sampler,
            field_model=dome_config.field_model,
            chi_par=dome_config.chi_par,
            chi_perp=dome_config.chi_perp,
            **extra,
        )

    # The seed is solved on the model's own region points, then the model is rebuilt.
    model = build_model(initial_configuration)
    if not args.init_design:
        stride = max(1, -(-model.points.shape[0] // args.seed_max_points))
        initial_configuration.orientations = lsq_seed_orientations(
            positions=initial_configuration.positions,
            roi_points=model.points[::stride],
            target_direction=torch.tensor(
                [float(v) for v in args.target.split(",")] if args.target
                else geometry.target_direction, dtype=dtype, device=args.device),
            volumes=initial_configuration.volumes,
            reg=dome_config.seed_reg,
            iterations=dome_config.seed_iterations,
            mirror=None if dome_config.seed_mirror == "none" else dome_config.seed_mirror,
        )
        model = build_model(initial_configuration)
    config["n_magnets"] = initial_configuration.num_magnets
    config["n_fov_points"] = int(model.points.shape[0])
    print(f"Magnets: {config['n_magnets']}, FOV evaluation points: {config['n_fov_points']}")
    with torch.no_grad():
        print(f"Seeded peak-to-peak: {float(peak_to_peak_ppm(model()['B'])):.0f} ppm")

    param_groups = [
        {"params": model.magnet_angles, "lr": args.lr, "name": "magnet_angles"},
        {"params": model.magnet_polar_angles,
         "lr": args.lr if args.polar_lr < 0.0 else args.polar_lr,
         "name": "magnet_polar_angles"},
    ]
    if args.positions:
        # Offsets are in meters and need their own, smaller step.
        param_groups.append({"params": model.position_offsets, "lr": args.position_lr,
                             "name": "position_offsets"})
    optimizer = torch.optim.Adam(param_groups)
    objective = SmoothedMinimaxObjective(
        alpha=args.alpha,
        alpha_max=args.alpha_max,
        alpha_growth=args.alpha_growth,
        rms_weight=args.rms_weight,
        band_weight=args.band_weight,
        band_t=(args.band_lower_t, args.band_upper_t),
        auto_sharpness=args.alpha_auto if args.alpha_auto > 0.0 else None,
    )

    collision, pair_state = None, {}
    if args.positions:
        # The pair list is rebuilt whenever GeometryRefresh bakes the drift.
        collision = CollisionPenalty(half_extent_m=0.5 * dome_config.magnet_size,
                                     margin_m=args.collision_margin_mm * 1e-3,
                                     weight=args.collision_weight)

        def rebuild_pairs(m) -> None:
            widest = max(args.position_bound_mm, args.position_bound_axial_mm or 0.0)
            reach = 2.0 * (dome_config.magnet_size * 1.7321 + widest * 1e-3)
            pair_state["pairs"] = candidate_pairs(m.effective_positions().detach(), reach)

        rebuild_pairs(model)

    def loss_fn(output: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = objective(output)
        if collision is not None:
            index_i, index_j = pair_state["pairs"]
            # Frames are held fixed per step, so only the centres carry gradient.
            with torch.no_grad():
                if gate_frame == "gs":
                    axes = cube_axes_min_z_extent(model.create_magnet_orientations())
                else:
                    axes = model.body_axes()
                if axes is None:
                    axes = torch.eye(3, dtype=model.magnet_positions.dtype, device=model.magnet_positions.device
                                     ).expand(model.magnet_positions.shape[0], 3, 3)
            loss = loss + collision(model.effective_positions(), axes, index_i, index_j)
        objective.step()
        return loss

    logger = WandbLogger(
        project=args.wandb_project,
        run_name=timestamp.replace("\\", "/"),
        log_dir=log_dir,
        config=config,
        mode=args.wandb_mode,
    )
    evaluation_points = (load_evaluation_points(args.evaluation_points, args.device, dtype)
                         if args.evaluation_points else model.points)
    evaluation = PeriodicTruthEvaluation(evaluation_points, every=args.evaluate_every,
                                         frame_rule=gate_frame,
                                         chi_par=dome_config.chi_par, chi_perp=dome_config.chi_perp)
    tracker = BestStateTracker(evaluation, path=args.best_design,
                               require_buildable=not args.best_allow_overlaps)
    callbacks = [evaluation, tracker]
    if args.positions:
        callbacks.append(GeometryRefresh(every=args.refresh_every, pair_builder=rebuild_pairs))

    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        logger=logger,
        log_each_n_epochs=args.log_each_n_epochs,
        log_dir=log_dir,
        callbacks=callbacks,
        grad_clip_norm=args.grad_clip_norm,
    )
    loop.run()
    if tracker.best is not None:
        print(f"Best: {tracker.best['ppm']:.0f} ppm at {tracker.best['field_mT']:.2f} mT "
              f"(epoch {tracker.best['epoch']}, {tracker.best['overlaps']} overlaps) -> {args.best_design}")
    with torch.no_grad():
        print(f"Final peak-to-peak: {float(peak_to_peak_ppm(model()['B'])):.0f} ppm at alpha {objective.alpha:.0f}")


if __name__ == "__main__":
    main()
