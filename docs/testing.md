# Testing

FieldSmith uses pytest for unit and numerical-validation tests. Run commands from
the repository root after installing the locked development environment:

```bash
uv sync --locked --group dev --extra cpu
```

## Running tests

Run the complete suite:

```bash
uv run --locked --group dev --extra cpu pytest tests
```

Run only fast unit tests:

```bash
uv run --locked --group dev --extra cpu pytest tests -m unit
```

Run only numerical validation tests:

```bash
uv run --locked --group dev --extra cpu pytest tests -m validation
```

Run one file or test while developing:

```bash
uv run --locked --group dev --extra cpu pytest tests/test_dipole.py
uv run --locked --group dev --extra cpu pytest tests/test_dipole.py::test_single_dipole_matches_axial_and_equatorial_fields
```

Test coverage can be generated with:

```bash
uv run --locked --group dev --extra cpu pytest tests -m unit --cov=fieldsmith --cov-report=term --cov-report=html
```

The HTML report is written to `htmlcov/`.

## Test suites

Tests below `tests/validation/` are automatically marked `validation`; all other
tests below `tests/` are marked `unit` by `tests/conftest.py`. Tests therefore do
not need explicit marker decorators.

Unit tests cover isolated behavior, input validation, configuration loading,
model contracts, logging, plotting, and optimization-loop behavior. They should
be deterministic and should not require a GPU or network access.

Validation tests compare independent physical formulations or external reference
implementations. The current suite checks:

- the finite cuboid field against Magpylib;
- the dipole approximation against the cuboid far field;
- the rectangular-prism potential against the direct cuboid kernel.

## Writing tests

- Add focused regression tests for fixed defects and behavior-oriented tests for
  new features.
- Keep reusable setup small and place broadly shared fixtures in
  `tests/conftest.py`.
- Use type hints for test function inputs and return values.
- Use `pytest.mark.parametrize` when several inputs exercise the same contract.
- Use `tmp_path` for files created by a test and close Matplotlib figures after
  assertions.
- Mock logging and external integrations in unit tests; do not contact W&B or
  other services.
- Prefer observable results over assertions about private implementation details.

## Numerical tests

Use `torch.float64` when testing analytical reference values, gradients, rotation
invariance, or agreement between field formulations. Select `rtol` and `atol`
from the expected numerical error rather than using exact equality for computed
floating-point results. Exact equality remains appropriate for shapes, metadata,
integer values, and operations that are expected to preserve tensor values.

Validation points must remain outside magnet volumes unless a test explicitly
targets interior-field behavior. Use fixed inputs and seeds so failures are
reproducible on CPU and GPU systems.
