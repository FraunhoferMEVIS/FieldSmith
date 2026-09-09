# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains logging utilities for scalar, image, and configuration data.
#
#  logger.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


class WandbLogger:
    """Small wandb logger with an offline/disabled fallback mode."""

    def __init__(
        self,
        project: str,
        run_name: str | None,
        log_dir: str | Path,
        config: dict[str, Any] | None = None,
        mode: str = "online",
    ) -> None:
        self.project: str = project
        self.run_name: str | None = run_name
        self.log_dir: Path = Path(log_dir)
        self.config: dict[str, Any] | None = config
        self.mode: str = mode
        self.run: Any | None = None
        self.wandb: Any | None = None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "configurations").mkdir(exist_ok=True)

        if mode == "disabled":
            return

        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                dir=str(self.log_dir),
                mode=mode,
                reinit=True,
            )
        except Exception as exc:
            self.mode = "disabled"
            print(f"wandb disabled: {exc}")

    def log_scalars(self, scalars: dict[str, Any], epoch: int) -> None:
        if self.run is None or self.wandb is None:
            return

        log_dict = {}
        for key, value in scalars.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            log_dict[f"Scalar/{key}"] = value
        self.wandb.log(log_dict, step=epoch)

    def log_images(self, images: dict[str, plt.Figure], epoch: int) -> None:
        if self.run is None or self.wandb is None:
            for figure in images.values():
                if not figure.__class__.__module__.startswith("plotly"):
                    plt.close(figure)
            return

        log_dict = {}
        for key, figure in images.items():
            if figure.__class__.__module__.startswith("plotly"):
                log_dict[f"Images/{key}"] = self.wandb.Plotly(figure)
            else:
                log_dict[f"Images/{key}"] = self.wandb.Image(figure)
                plt.close(figure)
        self.wandb.log(log_dict, step=epoch)

    def save_configuration(self, configuration: Any, epoch: int) -> None:
        save_path = self.log_dir / "configurations" / f"epoch_{epoch}.pt"
        if hasattr(configuration, "to_dict"):
            configuration = configuration.to_dict()
        torch.save(configuration, save_path)
        if self.run is None or self.wandb is None:
            return
        artifact = self.wandb.Artifact(name=f"configuration_epoch_{epoch}", type="configuration")
        artifact.add_file(str(save_path))
        self.run.log_artifact(artifact)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
