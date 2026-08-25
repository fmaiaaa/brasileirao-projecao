"""Publica brasileirao_modelos.xlsx na planilha Sheets (sem tocar Jogos/Placares)."""
from __future__ import annotations

import logging
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
from entrega_xlsx import (  # noqa: E402
    MODELOS_XLSX_NAME,
    SHEETS_MODELO,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("publish_modelos")


def main() -> int:
    candidates = [
        ROOT / "dados" / MODELOS_XLSX_NAME,
        ROOT / "artifacts" / "entrega" / MODELOS_XLSX_NAME,
        ROOT / "artifacts" / "entrega" / "brasileirao_modelos_latest.xlsx",
        Path.home() / "Downloads" / MODELOS_XLSX_NAME,
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        logger.error("brasileirao_modelos.xlsx não encontrado")
        return 1

    info = load_service_account_info()
    if not info:
        logger.error("Sem credenciais Google (.env GOOGLE_SERVICE_ACCOUNT_FILE)")
        return 2

    logger.info("Arquivo: %s", path)
    logger.info("SA: %s", info.get("client_email"))
    logger.info("Planilha: %s", spreadsheet_id_brasileirao())

    xl = pd.ExcelFile(path)
    sheets: dict[str, pd.DataFrame] = {}
    for name in SHEETS_MODELO:
        if name in xl.sheet_names:
            sheets[name] = pd.read_excel(path, sheet_name=name)
        else:
            logger.warning("Aba ausente no XLSX: %s", name)

    report = publicar_modelos_na_planilha(sheets, service_account_info=info)
    logger.info("Publicadas: %s", report.get("sheets"))
    logger.info("Protegidas (ignoradas): %s", report.get("skipped_protected"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
