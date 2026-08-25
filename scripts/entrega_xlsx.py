"""
XLSX canônico de modelos (job semanal) — abas fixas para o app.

Arquivo padrão: dados/brasileirao_modelos.xlsx
(colocar junto à base de resultados / planilha Jogos)

Abas:
  - Leia-me
  - Projecoes_Regressao
  - Coefs_Regressao
  - Classif_Regressao
  - Projecoes_Prob
  - Match_Forecasts
  - Classif_Prob_MC
  - Metricas_Prob
  - Base_Contexto
  - Overlay_Calendario
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Nomes canônicos das abas (não renomear no Drive/Sheets)
SHEET_LEIA_ME = "Leia-me"
SHEET_PROJ_REG = "Projecoes_Regressao"
SHEET_COEFS_REG = "Coefs_Regressao"
SHEET_CLASSIF_REG = "Classif_Regressao"
SHEET_PROJ_PROB = "Projecoes_Prob"
SHEET_FORECASTS = "Match_Forecasts"
SHEET_CLASSIF_PROB = "Classif_Prob_MC"
SHEET_METRICAS = "Metricas_Prob"
SHEET_CONTEXTO = "Base_Contexto"
SHEET_OVERLAY = "Overlay_Calendario"

SHEETS_MODELO = (
    SHEET_LEIA_ME,
    SHEET_PROJ_REG,
    SHEET_COEFS_REG,
    SHEET_CLASSIF_REG,
    SHEET_PROJ_PROB,
    SHEET_FORECASTS,
    SHEET_CLASSIF_PROB,
    SHEET_METRICAS,
    SHEET_CONTEXTO,
    SHEET_OVERLAY,
)

MODELOS_XLSX_NAME = "brasileirao_modelos.xlsx"


def build_entrega_xlsx(
    out_path: Path,
    *,
    classif_reg: pd.DataFrame,
    classif_prob: pd.DataFrame | None,
    proj_reg: pd.DataFrame,
    proj_prob: pd.DataFrame,
    metrics_prob: dict[str, Any] | None,
    overlay_report: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
    coefs_reg: pd.DataFrame | None = None,
    base_contexto: pd.DataFrame | None = None,
    match_forecasts: pd.DataFrame | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    leia = pd.DataFrame(
        [
            {
                "campo": "gerado_em",
                "valor": datetime.now().isoformat(timespec="seconds"),
            },
            {
                "campo": "arquivo",
                "valor": MODELOS_XLSX_NAME,
            },
            {
                "campo": "uso",
                "valor": (
                    "Coloque este arquivo junto à base de resultados (aba Jogos). "
                    "O app lê só essas abas + a planilha de placares. "
                    "Não renomeie as abas listadas abaixo."
                ),
            },
            {
                "campo": "abas",
                "valor": " | ".join(SHEETS_MODELO),
            },
            {
                "campo": "contexto",
                "valor": (
                    "Descanso = dias desde o último jogo (liga+copas). "
                    "Importante = Classificatórias/Oitavas/Quartas/Semi/Final "
                    "nos próximos 10 dias."
                ),
            },
            *[{"campo": k, "valor": str(v)} for k, v in meta.items()],
        ]
    )

    metric_rows = []
    if metrics_prob and isinstance(metrics_prob, dict):
        if "models" in metrics_prob:
            for name, m in (metrics_prob.get("models") or {}).items():
                metric_rows.append({"Modelo": name, **(m or {})})
            ens = metrics_prob.get("ensemble")
            if ens:
                metric_rows.append({"Modelo": "ensemble", **ens})
        else:
            metric_rows.append({"Modelo": "raw", "payload": str(metrics_prob)[:500]})
    met_df = (
        pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame({"info": ["sem métricas"]})
    )

    ov_df = (
        pd.DataFrame([overlay_report])
        if overlay_report
        else pd.DataFrame({"info": ["sem overlay"]})
    )

    fc = match_forecasts if match_forecasts is not None else pd.DataFrame()
    coefs = coefs_reg if coefs_reg is not None else pd.DataFrame()
    ctx = base_contexto if base_contexto is not None else pd.DataFrame()
    classif_p = classif_prob if classif_prob is not None else pd.DataFrame()

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        leia.to_excel(xw, sheet_name=SHEET_LEIA_ME, index=False)
        proj_reg.to_excel(xw, sheet_name=SHEET_PROJ_REG, index=False)
        coefs.to_excel(xw, sheet_name=SHEET_COEFS_REG, index=False)
        classif_reg.to_excel(xw, sheet_name=SHEET_CLASSIF_REG, index=False)
        proj_prob.to_excel(xw, sheet_name=SHEET_PROJ_PROB, index=False)
        fc.to_excel(xw, sheet_name=SHEET_FORECASTS, index=False)
        classif_p.to_excel(xw, sheet_name=SHEET_CLASSIF_PROB, index=False)
        met_df.to_excel(xw, sheet_name=SHEET_METRICAS, index=False)
        ctx.to_excel(xw, sheet_name=SHEET_CONTEXTO, index=False)
        ov_df.to_excel(xw, sheet_name=SHEET_OVERLAY, index=False)

    return out_path
