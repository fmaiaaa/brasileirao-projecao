"""Leitura do calendário via Google Sheets (mesmas secrets do velocímetro)."""
from __future__ import annotations

from typing import Any

import pandas as pd

# Planilha Brasileirão (editável no Google Sheets)
SPREADSHEET_ID_PADRAO = "1QkOIvRa9YinnOveOK4BkX4h_ZtGYRg1ZLzIfokbQh5I"
URL_PLANILHA = (
    "https://docs.google.com/spreadsheets/d/"
    f"{SPREADSHEET_ID_PADRAO}/edit"
)
ABAS_PREFERIDAS = ("Jogos", "Página1", "Pagina1", "Planilha1", "Sheet1")


def _secrets_connections_gsheets() -> dict[str, Any]:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and st.secrets.get("connections"):
            g = st.secrets["connections"].get("gsheets")
            if g is not None:
                return dict(g)
    except Exception:
        pass
    return {}


def _normalizar_private_key(pk: str) -> str:
    s = (pk or "").strip()
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\n", "\n")
    return s


def montar_service_account_info(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not raw:
        return None
    chaves = (
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    )
    out: dict[str, Any] = {}
    for k in chaves:
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
        if v == "":
            continue
        out[k] = v
    if "private_key" in out:
        out["private_key"] = _normalizar_private_key(str(out["private_key"]))
    if "private_key" not in out or "client_email" not in out:
        return None
    if not str(out.get("type") or "").strip():
        out["type"] = "service_account"
    out.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    out.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    return out


def spreadsheet_id_brasileirao() -> str:
    try:
        import streamlit as st

        sec = st.secrets.get("brasileirao")
        if isinstance(sec, dict):
            for k in ("spreadsheet_id", "SPREADSHEET_ID", "planilha_id"):
                v = str(sec.get(k) or "").strip()
                if v:
                    return v
    except Exception:
        pass
    return SPREADSHEET_ID_PADRAO


def _valores_para_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    header = [str(c).strip() for c in rows[0]]
    w = len(header)
    if w == 0:
        return pd.DataFrame()
    body = rows[1:]
    if not body:
        return pd.DataFrame(columns=header)
    norm: list[list[str]] = []
    for r in body:
        cells = [str(c) for c in r]
        if len(cells) < w:
            cells = cells + [""] * (w - len(cells))
        else:
            cells = cells[:w]
        norm.append(cells)
    return pd.DataFrame(norm, columns=header)


def _abrir_worksheet(sh: Any, nome_preferido: str | None = None):
    import gspread

    if nome_preferido:
        try:
            return sh.worksheet(nome_preferido)
        except gspread.WorksheetNotFound:
            for w in sh.worksheets():
                if w.title.strip().lower() == nome_preferido.strip().lower():
                    return w

    for titulo in ABAS_PREFERIDAS:
        try:
            return sh.worksheet(titulo)
        except gspread.WorksheetNotFound:
            continue

    worksheets = sh.worksheets()
    if not worksheets:
        raise ValueError("Planilha sem abas.")
    return worksheets[0]


def ler_planilha_gsheets(
    service_account_info: dict[str, Any],
    spreadsheet_id: str,
    worksheet: str | None = None,
) -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id.strip())
    ws = _abrir_worksheet(sh, worksheet)
    return _valores_para_dataframe(ws.get_all_values())


def credenciais_disponiveis() -> bool:
    return montar_service_account_info(_secrets_connections_gsheets()) is not None


def fingerprint_credenciais(info: dict[str, Any]) -> str:
    pk = str(info.get("private_key") or "")
    return str(hash(pk))[-12:] if pk else "0"
