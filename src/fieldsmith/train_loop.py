# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the training loop for magnet optimization models.
#
#  train_loop.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from fieldsmith.logger import WandbLogger


class TrainLoop:
    """Compact optimization loop for magnet models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        num_epochs: int,
        logger: WandbLogger,
        log_each_n_epochs: int,
        log_dir: str | Path,
        time_limit_seconds: float | None = None,
        callbacks: list[object] | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor] = loss_fn
        self.num_epochs: int = num_epochs
        self.logger: WandbLogger = logger
        self.log_each_n_epochs: int = log_each_n_epochs
        self.log_dir: Path = Path(log_dir)
        self.time_limit_seconds: float | None = time_limit_seconds

        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive when provided.")
        # Both default to inert, so an existing run behaves exactly as before.
        self.callbacks: list[object] = list(callbacks or [])
        self.grad_clip_norm: float | None = grad_clip_norm

    def run(self) -> torch.Tensor:
        """Run optimization and return the final loss."""
        final_loss = torch.zeros((), device=next(self.model.parameters()).device)
        time_start = time.monotonic()
        time_last = time_start

        try:
            self.model.train()
            epoch = 0
            while self.time_limit_seconds is not None or epoch < self.num_epochs:
                if self.time_limit_seconds is not None and time.monotonic() - time_start >= self.time_limit_seconds:
                    print(f"Time limit reached after {epoch} epochs.")
                    break

                output = self.model()
                loss = self.loss_fn(output)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                final_loss = loss.detach()

                # A model may need its parameters kept in range after a step --
                # a polar angle outside [0, pi] is a different orientation, not
                # an invalid one, so the optimiser will happily wander there.
                constrain = getattr(self.model, "constrain_parameters", None)
                if constrain is not None:
                    with torch.no_grad():
                        constrain()
                for callback in self.callbacks:
                    callback(epoch=epoch, model=self.model, loss=final_loss, loop=self)

                with torch.no_grad():
                    self.logger.log_scalars(self.model.get_scalars_to_log(output, loss), epoch)

                    if epoch == 0 or (epoch + 1) % self.log_each_n_epochs == 0:
                        metrics = self.model.get_scalars_to_log(output, loss)
                        print(
                            f"Epoch {epoch}: loss={loss.item():.6f}, "
                            f"field={metrics['fieldstrength_mT'].item():.3f} mT, "
                            f"homogeneity={metrics['homogeneity_ppm'].item():.2f} ppm, "
                            f"dt={time.monotonic() - time_last:.2f}s"
                        )
                        time_last = time.monotonic()
                        self.logger.log_images(self.model.get_images_to_log(output), epoch)
                        self.logger.save_configuration(self.model.get_configuration(), epoch)
                epoch += 1
        finally:
            self.logger.close()

        return final_loss
