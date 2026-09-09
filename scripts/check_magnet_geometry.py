"""Check a saved FieldSmith configuration's physical magnet geometry."""

from __future__ import annotations

import json

from fieldsmith import magnet_geometry_verifier as mgv


def main() -> int:
    """Build, print, and optionally save a magnet geometry report."""
    args = mgv.parse_args()
    dimensions = args.dimensions
    if args.cube_size_mm is not None:
        dimensions = [args.cube_size_mm * mgv.MM_TO_M] * 3
    config = mgv.load_config(args.config, dimensions)
    report = mgv.build_report(args, config)

    mgv.print_report(report)
    visualization_directory = args.visualization_directory
    if visualization_directory is None and args.log_file is not None:
        visualization_directory = args.log_file.parent / f"{args.log_file.stem}_visualizations"
    if visualization_directory is not None and report["has_intersections"]:
        report["intersection_visualizations"] = mgv.save_intersection_visualizations(
            report, config, visualization_directory, args.frame_rule
        )
        print(
            f"Saved {len(report['intersection_visualizations'])} intersection visualization(s) "
            f"to {visualization_directory.resolve()}"
        )
    else:
        report["intersection_visualizations"] = []
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with args.log_file.open("w", encoding="utf-8") as log_file:
            json.dump(report, log_file, indent=2, cls=mgv.NumpyJSONEncoder)

    if report["geometry_status"] == "warning":
        print(f"WARNING: {report['message']}")
        return 0
    if report["geometry_status"] == "error":
        print(f"ERROR: {report['message']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
