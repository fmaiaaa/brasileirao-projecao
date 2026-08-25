"""Camada de dados: schema, fontes, fingerprint, validação."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prob_ml.config import DEFAULT_SCHEMA_MAP_PATH, load_schema_map

logger = logging.getLogger(__name__)

CANONICAL_REQUIRED = (
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

_DRIVE_ID_RE = re.compile(
    r"(?:/d/|id=|open\?id=)([a-zA-Z0-9_-]{10,})",
    re.I,
)


def parse_drive_file_id(url_or_id: str) -> str:
    """Extrai file ID de URL do Google Drive ou devolve o ID se já for limpo."""
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("URL/ID do Google Drive vazio")
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", s) and "http" not in s:
        return s
    m = _DRIVE_ID_RE.search(s)
    if m:
        return m.group(1)
    raise ValueError(f"Não foi possível extrair file ID de: {s[:80]}")


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash estável do conteúdo tabular (ordem de colunas normalizada)."""
    cols = sorted(df.columns.astype(str))
    payload = df.reindex(columns=cols).astype(str).fillna("").to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def infer_and_map_schema(
    df: pd.DataFrame,
    schema_map: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Mapeia colunas reais → canônicas; retorna DF canônico + mapping usado."""
    schema_map = schema_map or load_schema_map()
    canonical = schema_map.get("canonical", {})
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    used: dict[str, str] = {}
    out = pd.DataFrame(index=df.index.copy())

    for canon, aliases in canonical.items():
        aliases_list = aliases if isinstance(aliases, list) else [aliases]
        found = None
        for alias in aliases_list:
            key = str(alias).strip().lower()
            if key in lower_map:
                found = lower_map[key]
                break
            if alias in df.columns:
                found = alias
                break
        if found is not None:
            out[canon] = df[found]
            used[canon] = str(found)

    missing = [c for c in CANONICAL_REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(
            "Colunas canônicas obrigatórias ausentes após mapping: "
            + ", ".join(missing)
            + f" | Colunas disponíveis: {list(df.columns)[:40]}"
        )
    return out, used


def validate_matches(df: pd.DataFrame) -> dict[str, Any]:
    """Relatório de validação; levanta se crítico."""
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_required": [],
        "duplicate_rows": 0,
        "invalid_scores": 0,
        "ok": True,
        "warnings": [],
    }
    for col in CANONICAL_REQUIRED:
        if col not in df.columns:
            report["missing_required"].append(col)
            report["ok"] = False

    if not report["ok"]:
        raise ValueError(f"Schema inválido: {report}")

    subset = [c for c in ("date", "home_team", "away_team") if c in df.columns]
    report["duplicate_rows"] = int(df.duplicated(subset=subset).sum())

    hg = pd.to_numeric(df["home_goals"], errors="coerce")
    ag = pd.to_numeric(df["away_goals"], errors="coerce")
    played = hg.notna() & ag.notna()
    bad = played & ((hg < 0) | (ag < 0) | (hg > 20) | (ag > 20))
    report["invalid_scores"] = int(bad.sum())
    if report["invalid_scores"]:
        report["warnings"].append(f"{report['invalid_scores']} placares inválidos")
    return report


def normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    if "round" in out.columns:
        out["round"] = pd.to_numeric(out["round"], errors="coerce")
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out = out.sort_values(["date", "home_team", "away_team"], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


class DataSource(ABC):
    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        ...

    def load_canonical(
        self, schema_map: dict[str, Any] | None = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        raw = self.load_raw()
        mapped, used = infer_and_map_schema(raw, schema_map)
        canon = normalize_matches(mapped)
        report = validate_matches(canon)
        report["fingerprint"] = dataset_fingerprint(canon)
        report["column_mapping"] = used
        report["source"] = self.__class__.__name__
        return canon, report


class LocalFileDataSource(DataSource):
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load_raw(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Base local não encontrada: {self.path}")
        suf = self.path.suffix.lower()
        if suf in {".csv", ".txt"}:
            return pd.read_csv(self.path)
        if suf in {".xlsx", ".xls"}:
            return pd.read_excel(self.path)
        if suf == ".parquet":
            return pd.read_parquet(self.path)
        raise ValueError(f"Formato não suportado: {suf}")


class GoogleDriveDataSource(DataSource):
    """Download via service account / API Drive. Cache local por file ID + hash."""

    def __init__(
        self,
        file_id: str | None = None,
        file_url: str | None = None,
        cache_dir: Path | str = "artifacts/prob_ml/cache",
        credentials_json: str | None = None,
        credentials_path: str | Path | None = None,
    ):
        raw_id = file_id or os.environ.get("GOOGLE_DRIVE_FILE_ID", "")
        raw_url = file_url or os.environ.get("GOOGLE_DRIVE_FILE_URL", "")
        if not raw_id and raw_url:
            raw_id = parse_drive_file_id(raw_url)
        if not raw_id:
            raise ValueError(
                "Configure GOOGLE_DRIVE_FILE_ID ou GOOGLE_DRIVE_FILE_URL"
            )
        self.file_id = parse_drive_file_id(raw_id)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_json = credentials_json or os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_JSON"
        )
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_FILE"
        )

    def _credentials(self):
        from google.oauth2 import service_account

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        if self.credentials_json:
            info = json.loads(self.credentials_json)
            return service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
        if self.credentials_path and Path(self.credentials_path).exists():
            return service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=scopes
            )
        # Mesmas secrets do Streamlit / planilha (streamlit-bot@bot-promocional...)
        try:
            from brasileirao_gsheets import (
                montar_service_account_info,
                _secrets_connections_gsheets,
            )

            info = montar_service_account_info(_secrets_connections_gsheets())
            if info:
                return service_account.Credentials.from_service_account_info(
                    info, scopes=scopes
                )
        except Exception:
            pass
        raise RuntimeError(
            "Defina GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_SERVICE_ACCOUNT_FILE "
            "ou [connections.gsheets] no secrets.toml "
            "(ex.: streamlit-bot@bot-promocional.iam.gserviceaccount.com)"
        )

    def load_raw(self) -> pd.DataFrame:
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as e:
            raise ImportError(
                "Instale google-api-python-client para Google Drive"
            ) from e

        meta_path = self.cache_dir / f"{self.file_id}.meta.json"
        data_path = self.cache_dir / f"{self.file_id}.bin"

        creds = self._credentials()
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        meta = (
            service.files()
            .get(fileId=self.file_id, fields="id,name,md5Checksum,modifiedTime,mimeType")
            .execute()
        )
        md5 = meta.get("md5Checksum") or meta.get("modifiedTime") or self.file_id
        if meta_path.exists() and data_path.exists():
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
            if prev.get("fingerprint") == md5:
                logger.info("Cache Drive hit file_id=%s", self.file_id)
                return self._read_bytes(data_path, meta.get("name", ""))

        logger.info("Baixando Drive file_id=%s", self.file_id)
        request = service.files().get_media(fileId=self.file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        data_path.write_bytes(buf.getvalue())
        meta_path.write_text(
            json.dumps(
                {"fingerprint": md5, "name": meta.get("name"), "id": self.file_id},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._read_bytes(data_path, meta.get("name", ""))

    @staticmethod
    def _read_bytes(path: Path, name: str) -> pd.DataFrame:
        raw = path.read_bytes()
        lower = name.lower()
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw))
        if lower.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(raw))
        # CSV default
        return pd.read_csv(io.BytesIO(raw))


def build_datasource(cfg: dict[str, Any], root: Path | None = None) -> DataSource:
    root = root or Path(__file__).resolve().parent.parent
    data_cfg = cfg.get("data", {})
    source = str(data_cfg.get("source", "local")).lower()
    if source == "google_drive":
        return GoogleDriveDataSource(
            file_id=data_cfg.get("google_drive_file_id") or None,
            file_url=data_cfg.get("google_drive_file_url") or None,
            cache_dir=root / data_cfg.get("cache_dir", "artifacts/prob_ml/cache"),
        )
    path = root / data_cfg.get("local_path", "dados/fpt_matches.parquet")
    return LocalFileDataSource(path)


def matches_from_calendar_jogos(jogos: list[Any]) -> pd.DataFrame:
    """Converte lista de Jogo do app atual em DF canônico (fallback sem FPT)."""
    rows = []
    for j in jogos:
        placar = getattr(j, "placar", "-")
        from brasileirao_projecao_core import parse_placar

        sc = parse_placar(str(placar))
        rows.append(
            {
                "match_id": f"{getattr(j, 'r', '')}-{getattr(j, 'mand', '')}-{getattr(j, 'vis', '')}",
                "season": 2026,
                "date": getattr(j, "data", None),
                "kickoff_time": getattr(j, "hora", ""),
                "round": getattr(j, "r", None),
                "home_team": getattr(j, "mand", ""),
                "away_team": getattr(j, "vis", ""),
                "home_goals": sc[0] if sc else np.nan,
                "away_goals": sc[1] if sc else np.nan,
            }
        )
    return normalize_matches(pd.DataFrame(rows))


def make_synthetic_matches(
    n_teams: int = 8,
    n_rounds: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Dataset sintético leakage-safe para testes."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    day = pd.Timestamp("2024-01-01")
    mid = n_teams // 2
    for r in range(1, n_rounds + 1):
        home = teams[:mid]
        away = teams[mid:]
        if r % 2 == 0:
            home, away = away, home
        for h, a in zip(home, away):
            hg = int(rng.poisson(1.3))
            ag = int(rng.poisson(1.1))
            rows.append(
                {
                    "match_id": f"{r}-{h}-{a}",
                    "season": 2024,
                    "date": day,
                    "round": r,
                    "home_team": h,
                    "away_team": a,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_xg": max(0.1, hg + rng.normal(0, 0.3)),
                    "away_xg": max(0.1, ag + rng.normal(0, 0.3)),
                    "odd_home_ft": float(rng.uniform(1.5, 3.5)),
                    "odd_draw_ft": float(rng.uniform(2.8, 4.0)),
                    "odd_away_ft": float(rng.uniform(1.8, 4.5)),
                }
            )
            day += pd.Timedelta(days=1)
    return normalize_matches(pd.DataFrame(rows))
