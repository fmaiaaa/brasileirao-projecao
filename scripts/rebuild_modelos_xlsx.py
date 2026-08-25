"""Rebuild brasileirao_modelos.xlsx from current artifact CSVs."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from entrega_xlsx import MODELOS_XLSX_NAME, SHEETS_MODELO, build_entrega_xlsx  # noqa: E402


def main() -> int:
    art = ROOT / "artifacts"
    proj_reg = pd.read_csv(art / "regressao" / "calendar_projecoes.csv")
    coefs = pd.read_csv(art / "regressao" / "coefs.csv")
    classif_reg = pd.read_csv(art / "regressao" / "classificacao.csv")
    proj_prob = pd.read_csv(art / "prob_ml" / "calendar_projecoes.csv")
    fc = pd.read_csv(art / "prob_ml" / "match_forecasts.csv")
    stand = pd.read_csv(art / "prob_ml" / "standings_sim.csv")
    ctx_path = art / "entrega" / "base_contexto.csv"
    ctx = pd.read_csv(ctx_path) if ctx_path.exists() else None

    metrics = None
    met_path = art / "prob_ml" / "model_metrics.json"
    if met_path.exists():
        metrics = json.loads(met_path.read_text(encoding="utf-8"))

    meta: dict = {}
    st_path = art / "prob_ml" / "status.json"
    if st_path.exists():
        st = json.loads(st_path.read_text(encoding="utf-8"))
        meta = {
            "champion": st.get("champion"),
            "status": st.get("status"),
            "fingerprint": st.get("dataset_fingerprint"),
            "runtime_sec": st.get("runtime_sec"),
        }

    dados = ROOT / "dados"
    dados.mkdir(exist_ok=True)
    primary = dados / MODELOS_XLSX_NAME
    build_entrega_xlsx(
        primary,
        classif_reg=classif_reg,
        classif_prob=stand,
        proj_reg=proj_reg,
        proj_prob=proj_prob,
        metrics_prob=metrics,
        overlay_report={"info": "rebuild_from_csv"},
        coefs_reg=coefs,
        base_contexto=ctx,
        match_forecasts=fc,
        meta=meta,
    )

    stamp = datetime.now().strftime("%Y%m%d")
    dests = [
        art / "entrega" / MODELOS_XLSX_NAME,
        art / "entrega" / "brasileirao_modelos_latest.xlsx",
        Path.home() / "Downloads" / MODELOS_XLSX_NAME,
        Path.home() / "Downloads" / f"brasileirao_modelos_{stamp}.xlsx",
    ]
    for d in dests:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary, d)
        print(f"OK {d} ({d.stat().st_size} bytes)")

    xl = pd.ExcelFile(primary)
    print("Abas:", xl.sheet_names)
    print("Esperado:", list(SHEETS_MODELO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
