"""Google Sheets / Drive — leitura e escrita da base do Brasileirão."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Planilha Brasileirão (editável no Google Sheets)
SPREADSHEET_ID_PADRAO = "1QkOIvRa9YinnOveOK4BkX4h_ZtGYRg1ZLzIfokbQh5I"
URL_PLANILHA = (
    "https://docs.google.com/spreadsheets/d/"
    f"{SPREADSHEET_ID_PADRAO}/edit"
)
ABAS_PREFERIDAS = ("Jogos", "Página1", "Pagina1", "Planilha1", "Sheet1")

# Nunca sobrescrever estas abas (resultados / placares)
ABAS_PROTEGIDAS = frozenset(
    {
        "jogos",
        "placares",
        "placar",
        "resultados",
        "página1",
        "pagina1",
    }
)

SCOPES_READONLY = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
SCOPES_READWRITE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


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


def load_service_account_info() -> dict[str, Any] | None:
    """
    Ordem:
      1) GOOGLE_SERVICE_ACCOUNT_JSON (string JSON no .env)
      2) GOOGLE_SERVICE_ACCOUNT_FILE (caminho .json explícito do projeto)
      3) GOOGLE_APPLICATION_CREDENTIALS (padrão GCP)
      4) Streamlit secrets [connections.gsheets]
    """
    raw_json = (os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
    if raw_json:
        try:
            return montar_service_account_info(json.loads(raw_json))
        except json.JSONDecodeError as e:
            logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON inválido: %s", e)

    for env_key in ("GOOGLE_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = (os.environ.get(env_key) or "").strip()
        if path and Path(path).is_file():
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                info = montar_service_account_info(data)
                if info:
                    return info
            except Exception as e:
                logger.warning("%s: %s", env_key, e)

    return montar_service_account_info(_secrets_connections_gsheets())


def credenciais_disponiveis() -> bool:
    return load_service_account_info() is not None


def spreadsheet_id_brasileirao() -> str:
    env_id = (os.environ.get("BRASILEIRAO_SPREADSHEET_ID") or "").strip()
    if env_id:
        return env_id
    try:
        import streamlit as st

        sec = st.secrets.get("brasileirao")
        if isinstance(sec, dict):
            for k in ("spreadsheet_id", "SPREADSHEET_ID", "planilha_id"):
                v = str(sec.get(k) or "").strip()
                if v:
                    return v
        g = _secrets_connections_gsheets()
        v = str(g.get("spreadsheet_id") or "").strip()
        # só usa se parecer a planilha do brasileirão (não a do velocímetro genérica)
        if v and v == SPREADSHEET_ID_PADRAO:
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


def _authorize(info: dict[str, Any], *, write: bool = False):
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = SCOPES_READWRITE if write else SCOPES_READONLY
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def ler_planilha_gsheets(
    service_account_info: dict[str, Any],
    spreadsheet_id: str,
    worksheet: str | None = None,
) -> pd.DataFrame:
    gc = _authorize(service_account_info, write=False)
    sh = gc.open_by_key(spreadsheet_id.strip())
    ws = _abrir_worksheet(sh, worksheet)
    return _valores_para_dataframe(ws.get_all_values())


def ler_aba_exata_gsheets(
    service_account_info: dict[str, Any],
    spreadsheet_id: str,
    worksheet: str,
) -> pd.DataFrame | None:
    """Lê uma aba pelo nome exato; None se não existir (sem fallback para Jogos)."""
    import gspread

    gc = _authorize(service_account_info, write=False)
    sh = gc.open_by_key(spreadsheet_id.strip())
    try:
        ws = sh.worksheet(worksheet)
    except gspread.WorksheetNotFound:
        for w in sh.worksheets():
            if w.title.strip().lower() == worksheet.strip().lower():
                ws = w
                break
        else:
            return None
    return _valores_para_dataframe(ws.get_all_values())


def _df_to_values(df: pd.DataFrame) -> list[list[Any]]:
    if df is None or df.empty:
        return [[]]
    out = df.copy()

    def _cell(x: Any) -> Any:
        if x is None:
            return ""
        try:
            if isinstance(x, float) and pd.isna(x):
                return ""
        except Exception:
            pass
        if isinstance(x, (int, float, bool)):
            return x
        return str(x)

    for c in list(out.columns):
        cl = str(c).strip().lower().replace("²", "2").replace(" ", "")
        if cl == "r2":
            def _fmt_r2(x: Any) -> str:
                # USER_ENTERED + locale BR: "0.8888" vira milhar 8888 ("8.888").
                # Prefixo ' força texto; vírgula = decimal BR se a Sheets converter.
                if x is None:
                    return ""
                try:
                    if isinstance(x, float) and pd.isna(x):
                        return ""
                except Exception:
                    pass
                s = str(x).strip().lstrip("'").replace(",", ".")
                try:
                    v = float(s)
                except (TypeError, ValueError):
                    return str(x)
                if v > 1.0:
                    digits = "".join(ch for ch in s if ch.isdigit())
                    if digits:
                        v = float("0." + digits[:4].ljust(4, "0")[:4])
                return f"'{v:.4f}".replace(".", ",")

            out[c] = out[c].map(_fmt_r2)
        else:
            out[c] = out[c].map(_cell)

    header = [str(c) for c in out.columns]
    rows = out.astype(object).where(pd.notnull(out), "").values.tolist()
    return [header] + rows


def _aba_protegida(nome: str) -> bool:
    n = str(nome or "").strip().lower()
    if n in ABAS_PROTEGIDAS:
        return True
    if "placar" in n or n.startswith("jogo"):
        return True
    return False


def escrever_aba_gsheets(
    service_account_info: dict[str, Any],
    spreadsheet_id: str,
    worksheet: str,
    df: pd.DataFrame,
) -> None:
    """Cria/atualiza aba de modelo. Nunca toca em Jogos/Placares/resultados."""
    import gspread

    if _aba_protegida(worksheet):
        raise ValueError(f"Recusa sobrescrever aba de resultados/placares: {worksheet}")

    gc = _authorize(service_account_info, write=True)
    sh = gc.open_by_key(spreadsheet_id.strip())
    values = _df_to_values(df if df is not None else pd.DataFrame())
    try:
        ws = sh.worksheet(worksheet)
    except gspread.WorksheetNotFound:
        rows = max(len(values) + 10, 100)
        cols = max(len(values[0]) if values else 1, 10)
        ws = sh.add_worksheet(title=worksheet, rows=rows, cols=cols)

    ws.clear()
    if values and values != [[]]:
        ws.update(values, value_input_option="USER_ENTERED")
    logger.info("Aba Sheets '%s' atualizada (%s linhas)", worksheet, max(0, len(values) - 1))


def publicar_modelos_na_planilha(
    sheets: dict[str, pd.DataFrame],
    *,
    spreadsheet_id: str | None = None,
    service_account_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Publica abas de modelo na mesma planilha dos resultados.
    Nunca altera Jogos / Placares / resultados.
    """
    info = service_account_info or load_service_account_info()
    if not info:
        raise RuntimeError(
            "Sem credenciais Google. Defina GOOGLE_SERVICE_ACCOUNT_FILE "
            "ou GOOGLE_SERVICE_ACCOUNT_JSON no .env / ambiente do agendador."
        )
    sid = (spreadsheet_id or spreadsheet_id_brasileirao()).strip()
    report: dict[str, Any] = {
        "spreadsheet_id": sid,
        "client_email": info.get("client_email"),
        "sheets": [],
        "skipped_protected": [],
    }
    for name, df in sheets.items():
        if not name or _aba_protegida(name):
            report["skipped_protected"].append(name)
            continue
        escrever_aba_gsheets(info, sid, name, df if df is not None else pd.DataFrame())
        report["sheets"].append(name)
    return report


def upload_xlsx_drive(
    local_path: Path | str,
    *,
    file_id: str | None = None,
    service_account_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Substitui o conteúdo de um arquivo já existente no Drive (mesmo file_id)."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    info = service_account_info or load_service_account_info()
    if not info:
        raise RuntimeError("Sem credenciais Google para upload Drive.")
    fid = (file_id or os.environ.get("MODELOS_DRIVE_FILE_ID") or "").strip()
    if not fid:
        raise RuntimeError("MODELOS_DRIVE_FILE_ID ausente.")
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    creds = Credentials.from_service_account_info(info, scopes=SCOPES_READWRITE)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    media = MediaFileUpload(
        str(path),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        resumable=True,
    )
    updated = (
        service.files()
        .update(fileId=fid, media_body=media, supportsAllDrives=True)
        .execute()
    )
    logger.info("Drive file %s atualizado (%s)", fid, path.name)
    return {"file_id": fid, "name": updated.get("name"), "path": str(path)}


def fingerprint_credenciais(info: dict[str, Any]) -> str:
    pk = str(info.get("private_key") or "")
    return str(hash(pk))[-12:] if pk else "0"
