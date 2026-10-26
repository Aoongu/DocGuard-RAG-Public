"""Create the one-time S12 preflight seal without executing Test methods."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from docguard.evaluation.preflight import run_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    payload = run_preflight(
        root,
        args.asset_root,
        root / "reports/s12_final/preflight.json",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
