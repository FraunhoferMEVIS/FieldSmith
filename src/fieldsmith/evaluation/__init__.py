"""Configuration evaluation, including manufacturing-tolerance studies."""

from fieldsmith.evaluation.core import evaluate_configuration, field_at_points, truth_eval_report
from fieldsmith.evaluation.monte_carlo import (
    MonteCarloResult,
    plot_monte_carlo_distributions,
    run_monte_carlo,
)

__all__ = [
    "MonteCarloResult",
    "evaluate_configuration",
    "field_at_points",
    "plot_monte_carlo_distributions",
    "run_monte_carlo",
    "truth_eval_report",
]
