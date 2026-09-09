# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains training callbacks for magnet optimization.
#
#  callbacks.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch

from fieldsmith.configurations.setup_npy import save_halbach_setup_npy
from fieldsmith.evaluation import truth_eval_report


class PeriodicTruthEvaluation:
    """
    Measure the design on the full region, on a cadence.

    Training runs on a subsample for cost, so the training loss is a biased view
    of the design: it is optimised against those points and reports what it
    optimised. The number a result is quoted as has to come from the whole
    region under the acceptance physics, which is far too expensive to do every
    step and essential to do sometimes.

    Results are placed in ``self.history`` and handed to any callback listed
    after this one, so the ordering in the callback list is meaningful.
    """

    def __init__(self, points: torch.Tensor, every: int, frame_rule: str = "gs",
                 chi_par: float = 0.05, chi_perp: float = 0.10, chunk_points: int = 2048) -> None:
        self.points = points
        self.every = every
        self.frame_rule = frame_rule
        self.chi = (chi_par, chi_perp)
        self.chunk_points = chunk_points
        self.history: list[dict[str, float]] = []
        self.latest: dict[str, float] | None = None

    def __call__(self, epoch: int, model, **_) -> None:
        if (epoch + 1) % self.every:
            return
        with torch.no_grad():
            configuration = model.get_configuration().to(self.points.device, self.points.dtype)
            configuration.metadata.setdefault("magnet_size_m", getattr(model, "magnet_size", 0.0))
            report = truth_eval_report(configuration, self.points, *self.chi,
                                       self.chunk_points, self.frame_rule)
        self.latest = {"epoch": epoch + 1, "ppm": report["demag"]["peak_to_peak_ppm"],
                       "field_mT": report["demag"]["fieldstrength_mT"],
                       "overlaps": report["overlaps"]}
        self.history.append(self.latest)


class BestStateTracker:
    """
    Keep the best design seen, not the last one.

    A worst-case descent is a noisy walk, not a monotone descent: successive
    measurements of the same run vary by a hundred ppm or more in either
    direction. A loop that reports its final state therefore reports wherever
    the walk happened to be when the clock ran out. In the campaign that
    produced the reference designs, the best state ever reached was 448 ppm and
    it was overwritten by the following segment's 589; the run ended at 555 and
    the 448 no longer exists anywhere.

    Nothing here changes the trajectory. It only decides what is kept, which is
    why it is close to free and worth having on every run. Designs failing the
    overlap gate are not eligible, so an unbuildable design cannot win.
    """

    def __init__(self, evaluation: PeriodicTruthEvaluation, path: str | Path | None = None,
                 require_buildable: bool = True) -> None:
        self.evaluation = evaluation
        self.path = Path(path) if path else None
        self.require_buildable = require_buildable
        self.best: dict[str, float] | None = None
        self.best_configuration = None

    def __call__(self, epoch: int, model, **_) -> None:
        latest = self.evaluation.latest
        if latest is None or latest.get("epoch") != epoch + 1:
            return
        if self.require_buildable and latest["overlaps"] > 0:
            return
        if self.best is not None and latest["ppm"] >= self.best["ppm"]:
            return
        self.best = dict(latest)
        with torch.no_grad():
            self.best_configuration = model.get_configuration().detached_cpu()
            self.best_configuration.metadata.setdefault(
                "magnet_size_m", getattr(model, "magnet_size", 0.0))
        if self.path is not None:
            save_halbach_setup_npy(self.path, self.best_configuration)


class GeometryRefresh:
    """
    Rebuild what the moving geometry invalidates.

    Two things are held fixed for cost and are only exact for the arrangement
    they were built from: the mutual demagnetisation coupling and the collision
    pair list. While positions are frozen neither goes stale. Once positions are
    free, both do, and the optimiser is then steered by the physics of an array
    that no longer exists -- silently, because the loss keeps falling.

    This also bakes the achieved displacement into the base arrangement, which
    re-centres the bounded offset and lets a long run travel further than one
    bound in bounded steps.
    """

    def __init__(self, every: int, pair_builder=None) -> None:
        self.every = every
        self.pair_builder = pair_builder
        self.baked = 0

    def __call__(self, epoch: int, model, **_) -> None:
        if (epoch + 1) % self.every:
            return
        if hasattr(model, "bake_positions"):
            model.bake_positions()
            self.baked += 1
        if hasattr(model, "refresh_demagnetisation"):
            model.refresh_demagnetisation()
        if self.pair_builder is not None:
            self.pair_builder(model)
