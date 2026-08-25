#!/usr/bin/env python
"""CLI offline de treino (não roda no request do Streamlit)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prob_ml.config import load_config
from prob_ml.data import LocalFileDataSource, build_datasource, make_synthetic_matches
from prob_ml.pipeline import train_pipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Treino offline do pipeline probabilístico")
    p.add_argument("--budget", choices=["fast", "standard", "full"], default="fast")
    p.add_argument("--synthetic", action="store_true", help="Usa dados sintéticos")
    p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    cfg["compute_budget"] = args.budget

    if args.synthetic:
        matches = make_synthetic_matches(n_teams=10, n_rounds=10, seed=1)
        report = {"fingerprint": "synthetic", "source": "synthetic"}
    else:
        try:
            ds = build_datasource(cfg, ROOT)
            matches, report = ds.load_canonical()
        except FileNotFoundError as e:
            logging.error("%s — use --synthetic ou configure a base FPT/Drive", e)
            return 1

    bundle = train_pipeline(matches, cfg, run_backtest=not args.no_backtest)
    logging.info("status=%s champion=%s fingerprint=%s",
                 bundle.status.status, bundle.status.champion, report.get("fingerprint"))
    logging.info("metrics=%s", bundle.status.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
