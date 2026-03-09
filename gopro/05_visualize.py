"""
Step 5: Generate GP-BO Visualization Report.

Produces a self-contained HTML report with interactive Plotly charts
showing the current optimization state.

Inputs:
  - All outputs from steps 02-04 in data/
Output:
  - data/report_round{N}.html
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gopro.config import DATA_DIR


if __name__ == "__main__":
    import argparse

    from gopro.visualize_report import generate_report

    parser = argparse.ArgumentParser(
        description="Generate GP-BO visualization report"
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help=f"Data directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML path (default: data_dir/report_round{N}.html)"
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    report_path = generate_report(Path(args.data_dir), output_path)
    print(f"Report generated: {report_path}")
