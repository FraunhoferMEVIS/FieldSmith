"""Run the matching small-Halbach distance, radius, and angle optimization."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, TextIO

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.logger import WandbLogger
from fieldsmith.metrics import component_homogeneity_loss, field_metrics
from fieldsmith.models.small_halbach import (
    SmallHalbachAngleModel,
    SmallHalbachDistanceModel,
    SmallHalbachRadiusModel,
    build_small_halbach_configuration,
    distance_stage_loss,
    radius_stage_loss,
)
from fieldsmith.train_loop import TrainLoop


class CsvHistoryLogger(WandbLogger):
    """Persist every scalar locally while retaining optional wandb logging."""

    def __init__(
        self,
        project: str,
        run_name: str | None,
        log_dir: str | Path,
        config: dict[str, Any] | None = None,
        mode: str = "online",
    ) -> None:
        super().__init__(project, run_name, log_dir, config, mode)
        history_path = self.log_dir / "history.csv"
        self._history_file: TextIO = history_path.open("w", newline="", encoding="utf-8")
        self._history_writer: csv.DictWriter = csv.DictWriter(
            self._history_file,
            fieldnames=("epoch", "metric", "value"),
        )
        self._history_writer.writeheader()

    def log_scalars(self, scalars: dict[str, Any], epoch: int) -> None:
        super().log_scalars(scalars, epoch)
        for name, value in scalars.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self._history_writer.writerow(
                {"epoch": epoch, "metric": name, "value": float(value)},
            )
        self._history_file.flush()

    def close(self) -> None:
        self._history_file.close()
        super().close()


def mse_angle_loss(
      output: dict[str, torch.Tensor],
      expected_field: float,
  ) -> torch.Tensor:
      field = output["B"][..., 0].abs()
      normalized = field / field.mean().clamp_min(1.0e-12)

      homogeneity = torch.linalg.vector_norm(
          normalized - torch.ones_like(normalized),
      )
      field_target = torch.nn.functional.mse_loss(
          field,
          torch.full_like(field, expected_field),
      ) / expected_field**2

      return homogeneity + 50.0 * field_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the small Halbach system in three stages: axial ring "
            "distance, inner/outer radii, and in-plane magnet angles."
        )
    )
    parser.add_argument("--distance-epochs", type=int, default=1000)
    parser.add_argument("--radius-epochs", type=int, default=1500)
    parser.add_argument("--angle-epochs", type=int, default=5001)
    parser.add_argument("--distance-lr", type=float, default=1.0e-4)
    parser.add_argument("--inner-radius-lr", type=float, default=0.01)
    parser.add_argument("--outer-radius-lr", type=float, default=0.001)
    parser.add_argument("--angle-lr", type=float, default=0.01)
    parser.add_argument("--num-rings", type=int, default=5)
    parser.add_argument("--inner-magnets", type=int, default=10)
    parser.add_argument("--outer-magnets", type=int, default=30)
    parser.add_argument("--inner-radius", type=float, default=0.08)
    parser.add_argument("--outer-radius", type=float, default=0.10)
    parser.add_argument("--ring-distance", type=float, default=0.04)
    parser.add_argument("--magnet-size", type=float, default=0.012)
    parser.add_argument("--remanence", type=float, default=1.31)
    parser.add_argument("--expected-field", type=float, default=0.04)
    parser.add_argument("--fov-size", type=float, default=0.06)
    parser.add_argument("--resolution", type=float, default=0.001)
    parser.add_argument("--fov-radius", type=float, default=0.025)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-each-n-epochs", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="smh-magnetopts")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--detect-anomaly", action=argparse.BooleanOptionalAction, default=True,
    )
    return parser.parse_args()


def _run_stage(
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    epochs: int,
    scheduler_patience: int,
    log_each_n_epochs: int,
    output_directory: Path,
    wandb_project: str,
    wandb_mode: str,
    config: dict[str, object],
) -> tuple[MagnetConfiguration, dict[str, float]]:
    if epochs <= 0:
        raise ValueError(f"{name} epochs must be positive.")
    stage_directory = output_directory / name
    logger = CsvHistoryLogger(
        project=wandb_project,
        run_name=f"{output_directory.name}/{name}",
        log_dir=stage_directory,
        config=config,
        mode=wandb_mode,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience,
    )
    logger.save_configuration(model.get_configuration(), 0)

    def stage_callback(
        epoch: int,
        model: nn.Module,
        loss: torch.Tensor,
        loop: TrainLoop,
    ) -> None:
        scheduler.step(loss)
        logger.log_scalars(
            {group["name"]: group["lr"] for group in optimizer.param_groups},
            epoch,
        )
        if epoch == epochs - 1:
            logger.save_configuration(model.get_configuration(), epoch)

    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=epochs,
        logger=logger,
        log_each_n_epochs=log_each_n_epochs,
        log_dir=stage_directory,
        callbacks=[stage_callback],
    )
    loop.run()

    with torch.no_grad():
        final_output = model()
        final_loss = loss_fn(final_output)
    configuration = model.get_configuration().detached_cpu()
    final_metrics = field_metrics(final_output["B"])
    summary = {
        "loss": float(final_loss.cpu()),
        "fieldstrength_mT": float(final_metrics["fieldstrength_mT"].detach().cpu()),
        "homogeneity_ppm": float(final_metrics["homogeneity_ppm"].detach().cpu()),
    }
    return configuration, summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    device = torch.device(args.device)
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_small_halbach"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    fov = (args.fov_size,) * 3
    resolution = (args.resolution,) * 3
    magnet_counts = (args.inner_magnets, args.outer_magnets)
    config = vars(args).copy()
    config["output_directory"] = str(output_directory)
    print(f"Device: {device}; output: {output_directory.resolve()}")

    initial = build_small_halbach_configuration(
        num_rings=args.num_rings,
        num_magnets_per_ring=magnet_counts,
        ring_radius_inner=args.inner_radius,
        ring_radius_outer=args.outer_radius,
        distance_between_rings=args.ring_distance,
        magnet_size=args.magnet_size,
        remanence=args.remanence,
        device=device,
    )
    print(f"Total magnets: {initial.num_magnets}")

    distance_model = SmallHalbachDistanceModel(
        initial, fov=fov, resolution=resolution, fov_radius=args.fov_radius,
    )
    distance_optimizer = torch.optim.Adam([
        {"params": distance_model.z_levels_trainable, "lr": args.distance_lr, "name": "lr_distances"},
    ])
    distance_config, distance_summary = _run_stage(
        "distance", distance_model, distance_optimizer, distance_stage_loss,
        args.distance_epochs, 50, args.log_each_n_epochs, output_directory,
        args.wandb_project, args.wandb_mode, config,
    )

    radius_model = SmallHalbachRadiusModel(
        distance_config.to(device),
        fov=fov,
        resolution=resolution,
        fov_radius=args.fov_radius,
        num_rings=args.num_rings,
        num_magnets_per_ring=magnet_counts,
        ring_radius_inner=args.inner_radius,
        ring_radius_outer=args.outer_radius,
    )
    radius_optimizer = torch.optim.Adam([
        {"params": radius_model.inner_factor, "lr": args.inner_radius_lr, "name": "lr_inner_factor"},
        {"params": radius_model.outer_factor, "lr": args.outer_radius_lr, "name": "lr_outer_factor"},
    ])
    radius_loss = lambda output: radius_stage_loss(output, args.expected_field, args.magnet_size)
    radius_config, radius_summary = _run_stage(
        "radius", radius_model, radius_optimizer, radius_loss,
        args.radius_epochs, 150, args.log_each_n_epochs, output_directory,
        args.wandb_project, args.wandb_mode, config,
    )

    angle_model = SmallHalbachAngleModel(
        radius_config.to(device), fov=fov, resolution=resolution, fov_radius=args.fov_radius,
    )
    angle_optimizer = torch.optim.Adam([
        {"params": angle_model.magnet_angles_phi, "lr": args.angle_lr, "name": "lr_phi"},
    ])
    angle_loss = lambda output: mse_angle_loss(output, args.expected_field)
    angle_config, angle_summary = _run_stage(
        "angle", angle_model, angle_optimizer, angle_loss,
        args.angle_epochs, 150, args.log_each_n_epochs, output_directory,
        args.wandb_project, args.wandb_mode, config,
    )

    final_path = output_directory / "small_halbach_final.pt"
    torch.save(angle_config.to_dict(), final_path)
    summary = {
        "distance": distance_summary,
        "radius": radius_summary,
        "angle": angle_summary,
        "final_configuration": str(final_path.resolve()),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
