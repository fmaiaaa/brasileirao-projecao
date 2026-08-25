"""
Reformata backtest R19+ (sem leakage) e publica abas por ano na planilha.

Colunas:
  Ranking Final | Time | Pts Finais | Rodada 19 … Rodada 38 | Erro Abs R19

Ranking Final = classificação real ao fim.
Rodada N = posição final média estimada só com dados até a rodada N
           (treino sem ver anos futuros nem o 2º turno da temporada-alvo).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

load_dotenv(ROOT / ".env", override=True)

from brasileirao_gsheets import (  # noqa: E402
    load_service_account_info,
    publicar_modelos_na_planilha,
    spreadsheet_id_brasileirao,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("publish_ranking")

CANDIDATES = [
    ROOT / "artifacts" / "prob_ml" / "backtest_ranking_R19plus.xlsx",
    Path.home() / "Downloads" / "backtest_ranking_R19plus_brasileirao.xlsx",
    ROOT / "artifacts" / "prob_ml" / "backtest_ranking.xlsx",
]


def _rename_round_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    for c in out.columns:
        m = re.fullmatch(r"R(\d{1,2})", str(c))
        if m:
            mapping[c] = f"Rodada {int(m.group(1))}"
    return out.rename(columns=mapping)


def format_year_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking Final primeiro; colunas Rodada N; erro em R19."""
    out = _rename_round_cols(df)
    # ordem canônica
    front = [c for c in ("Ranking Final", "Time", "Pts Finais") if c in out.columns]
    rods = sorted(
        [c for c in out.columns if str(c).startswith("Rodada ")],
        key=lambda x: int(str(x).split()[-1]),
    )
    rest = [c for c in out.columns if c not in front and c not in rods]
    out = out[front + rods + rest]
    if "Rodada 19" in out.columns and "Ranking Final" in out.columns:
        out["Erro Abs R19"] = (
            out["Rodada 19"].astype(float) - out["Ranking Final"].astype(float)
        ).abs().round(2)
        # ranking projetado na R19 vs real final (explicito)
        out["Proj Final (c/ dados até R19)"] = out["Rodada 19"].round(2)
        out["Real Final (R38)"] = out["Ranking Final"]
    return out


def load_year_sheets(path: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    out: dict[str, pd.DataFrame] = {}
    for name in xl.sheet_names:
        if re.fullmatch(r"20\d{2}", str(name)):
            raw = pd.read_excel(path, sheet_name=name)
            out[str(name)] = format_year_sheet(raw)
            logger.info("Ano %s → %s linhas, %s cols", name, len(out[str(name)]), len(out[str(name)].columns))
    # resumo se existir
    if "Resumo_Ranking" in xl.sheet_names:
        out["Resumo_Ranking"] = pd.read_excel(path, sheet_name="Resumo_Ranking")
    return out


def main() -> int:
    path = next((p for p in CANDIDATES if p.exists()), None)
    if path is None:
        logger.error("Nenhum backtest_ranking*.xlsx encontrado. Rode scripts/backtest_ranking_xlsx.py")
        return 1
    info = load_service_account_info()
    if not info:
        logger.error("Sem credenciais Google")
        return 2
    logger.info("Fonte: %s", path)
    logger.info("SA: %s · planilha %s", info.get("client_email"), spreadsheet_id_brasileirao())
    sheets = load_year_sheets(path)
    if not sheets:
        logger.error("Nenhuma aba de ano encontrada")
        return 3
    # salva cópia local formatada
    out_local = ROOT / "artifacts" / "prob_ml" / "ranking_anos_formatado.xlsx"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_local, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    report = publicar_modelos_na_planilha(sheets, service_account_info=info)
    logger.info("Publicadas: %s", report.get("sheets"))
    logger.info("Protegidas ignoradas: %s", report.get("skipped_protected"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
