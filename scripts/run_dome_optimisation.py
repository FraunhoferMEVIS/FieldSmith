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
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations.dome_config import DomeConfig
from fieldsmith.geometry.dome import DomeGeometry
from fieldsmith.geometry.halbach_ring import make_spherical_mask
from fieldsmith.geometry.lsq_seed import lsq_seed_orientations
from fieldsmith.logger import WandbLogger
from fieldsmith.models.dome import DomeAngleModel
from fieldsmith.objectives import SmoothedMinimaxObjective, peak_to_peak_ppm
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler
from fieldsmith.train_loop import TrainLoop

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dome_geometry.cfg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize an open cap magnet array with smoothed worst-case homogeneity.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=2.0e3, help="Initial log-sum-exp sharpness.")
    parser.add_argument("--alpha-max", type=float, default=2.0e4)
    parser.add_argument("--alpha-growth", type=float, default=1.005, help="Geometric alpha ramp per epoch.")
    parser.add_argument("--rms-weight", type=float, default=0.0)
    parser.add_argument("--band-weight", type=float, default=0.0, help="Weight of the mean-field band penalty, in ppm.")
    parser.add_argument("--band-lower-t", type=float, default=0.045)
    parser.add_argument("--band-upper-t", type=float, default=0.500)
    parser.add_argument("--double-precision", action="store_true", help="Run in float64; recommended below 1000 ppm.")
    parser.add_argument("--chunk-points", type=int, default=32768)
    parser.add_argument("--log-each-n-epochs", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="fieldsmith")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y_%m_%d/%Hh%Mm%Ss_dome")
    log_dir = Path("logs") / timestamp
    dome_config = DomeConfig.from_file(args.config)
    config = vars(args)
    config["config"] = str(args.config)
    config.update(dome_config.as_log_config())
    dtype = torch.float64 if args.double_precision else torch.float32

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
    initial_configuration = geometry.build()
    grid_sampler = CartesianGridSampler(
        fov=dome_config.fov,
        resolution=dome_config.resolution,
        device=args.device,
        dtype=dtype,
    )
    # The cap is open: only the half space above the aperture plane is imaged.
    def roi_mask(points: torch.Tensor) -> torch.Tensor:
        return make_spherical_mask(points, dome_config.fov_radius) & (points[:, 2] >= dome_config.fov_z_min)

    point_sampler = MaskedPointSampler(base_sampler=grid_sampler, mask_fn=roi_mask)

    def build_model(configuration):
        return DomeAngleModel(
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
        )

    # The seed needs the actual ROI points, which only the model's sampler knows;
    # solve on them, re-seed the configuration and rebuild. The grid ROI is
    # symmetric, so the seed inherits the geometry's symmetry.
    model = build_model(initial_configuration)
    initial_configuration.orientations = lsq_seed_orientations(
        positions=initial_configuration.positions,
        roi_points=model.points,
        target_direction=torch.tensor(geometry.target_direction, dtype=dtype, device=args.device),
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

    optimizer = torch.optim.Adam(
        [
            {"params": model.magnet_angles, "lr": args.lr, "name": "magnet_angles"},
            {"params": model.magnet_polar_angles, "lr": args.lr, "name": "magnet_polar_angles"},
        ]
    )
    objective = SmoothedMinimaxObjective(
        alpha=args.alpha,
        alpha_max=args.alpha_max,
        alpha_growth=args.alpha_growth,
        rms_weight=args.rms_weight,
        band_weight=args.band_weight,
        band_t=(args.band_lower_t, args.band_upper_t),
    )

    def loss_fn(output: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = objective(output)
        objective.step()
        return loss

    logger = WandbLogger(
        project=args.wandb_project,
        run_name=timestamp.replace("\\", "/"),
        log_dir=log_dir,
        config=config,
        mode=args.wandb_mode,
    )
    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        logger=logger,
        log_each_n_epochs=args.log_each_n_epochs,
        log_dir=log_dir,
    )
    loop.run()
    with torch.no_grad():
        print(f"Final peak-to-peak: {float(peak_to_peak_ppm(model()['B'])):.0f} ppm at alpha {objective.alpha:.0f}")


if __name__ == "__main__":
    main()
