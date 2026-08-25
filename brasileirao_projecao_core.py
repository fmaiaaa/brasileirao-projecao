"""Lógica de projeção - Brasileirão 2026."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ModoProjecao = Literal[
    "media_simples",
    "media_casa_fora",
    "repetir_turno",
    "regressao_momento_aceleracao",
    "regressao_momento_historico",
    "regressao_completa",
    "prob_ml",
]
TipoRegressao = Literal["simples", "mandante_visitante"]
VarianteRegressao = Literal[
    "interacao",
    "casa_sem_interacao",
    "interacao_adv_turno",
]
VarianteRegressaoAcumulada = Literal[
    "momento_aceleracao",
    "momento_historico",
    "completa",
    "completa_limites",
]

NOME_REGRESSAO_ACUMULADA: dict[VarianteRegressaoAcumulada, str] = {
    "momento_aceleracao": "Regressão de Momento e Aceleração (efeitos fixos)",
    "momento_historico": "Regressão de Momento e Histórico (efeitos fixos)",
    "completa": "Regressão",
    "completa_limites": "Regressão (Centrada)",
}

MODO_PARA_VARIANTE: dict[str, VarianteRegressaoAcumulada] = {
    "regressao_momento_aceleracao": "momento_aceleracao",
    "regressao_momento_historico": "momento_historico",
    "regressao_completa": "completa",
}


def modo_e_regressao_acumulada(modo: ModoProjecao) -> bool:
    return modo in MODO_PARA_VARIANTE

FORMA_RECENTE_JOGOS = 5
RODADA_FIM_PRIMEIRO_TURNO = 19
RODADA_CENTRO = 19  # centro do campeonato para Rodada Centrada
DELTA_PTS_MAX_POR_RODADA = 3.0
PESO_FORMA_RECENTE_H1 = 0.80  # próxima rodada
PESO_FORMA_RECENTE_H5 = 0.50  # daqui a 5 rodadas
PESO_FORMA_RECENTE_PISO = 0.20
ALTURA_GRAFICO = 520
TICKS_RODADA = [1, 9.5, 19, 28.5, 38]

_DIR_APP = Path(__file__).resolve().parent
ARQUIVO_CALENDARIO = _DIR_APP / "dados" / "calendario_brasileirao_2026.xlsx"


def parse_placar(placar: str) -> tuple[int, int] | None:
    s = str(placar).strip()
    if s in ("-", "nan", ""):
        return None
    m = re.match(r"(\d+)\s*x\s*(\d+)", s, re.I)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pontos_placar(gm: int, gv: int) -> tuple[int, int]:
    if gm > gv:
        return 3, 0
    if gm < gv:
        return 0, 3
    return 1, 1


@dataclass
class Jogo:
    r: int
    data: str
    hora: str
    mand: str
    vis: str
    placar: str
    est: str = ""
    proj_pm: float | None = field(default=None, repr=False)
    proj_pv: float | None = field(default=None, repr=False)
    proj_gm: int | None = field(default=None, repr=False)
    proj_gv: int | None = field(default=None, repr=False)
    origem: str = field(default="", repr=False)

    @property
    def jogado(self) -> bool:
        return parse_placar(self.placar) is not None

    @property
    def par(self) -> frozenset[str]:
        return frozenset({self.mand, self.vis})

    def pts_reais(self) -> tuple[int, int] | None:
        p = parse_placar(self.placar)
        if p is None:
            return None
        return pontos_placar(*p)

    def pts_efetivos(self) -> tuple[float, float]:
        if self.jogado:
            pm, pv = self.pts_reais()  # type: ignore[misc]
            return float(pm), float(pv)
        if self.proj_pm is not None and self.proj_pv is not None:
            return float(self.proj_pm), float(self.proj_pv)
        return 0.0, 0.0


def jogos_from_records(records: list[dict]) -> list[Jogo]:
    return [Jogo(**{k: rec[k] for k in rec}) for rec in records]


def _dataframe_para_jogos(df: pd.DataFrame) -> list[Jogo]:
    norm = {str(c).strip().lower(): c for c in df.columns}
    records: list[dict] = []
    for _, row in df.iterrows():
        rodada_raw = row[norm.get("rodada", "Rodada")]
        m = re.search(r"(\d+)", str(rodada_raw))
        if not m:
            continue
        r = int(m.group(1))
        data = str(row[norm.get("data", "Data")])[:10]
        hora_val = row.get(norm.get("hora", "Hora"), "")
        hora = str(hora_val)[:8] if pd.notna(hora_val) else ""
        mand = str(row[norm.get("mandante", "Mandante")]).strip()
        vis = str(row[norm.get("visitante", "Visitante")]).strip()
        if not mand or not vis or mand.lower() == "nan":
            continue
        placar = str(row[norm.get("placar", "Placar")]).strip()
        est_col = norm.get("estadio") or norm.get("estádio") or "Estadio"
        est = (
            str(row[est_col]).strip()
            if est_col in df.columns and pd.notna(row[est_col])
            else ""
        )
        records.append(
            {
                "r": r,
                "data": data,
                "hora": hora,
                "mand": mand,
                "placar": placar,
                "vis": vis,
                "est": est,
            }
        )
    return jogos_from_records(records)


def carregar_jogos_xlsx(caminho: Path | str | None = None) -> list[Jogo]:
    """Lê calendário/placares do XLSX local (fallback)."""
    path = Path(caminho) if caminho else ARQUIVO_CALENDARIO
    if not path.is_file():
        raise FileNotFoundError(f"Planilha não encontrada: {path}")
    df = pd.read_excel(path, sheet_name=0)
    return _dataframe_para_jogos(df)


def carregar_jogos_gsheets() -> tuple[list[Jogo], pd.DataFrame, str]:
    """Lê calendário da planilha Google (secrets / .env service account)."""
    from brasileirao_gsheets import (
        load_service_account_info,
        ler_planilha_gsheets,
        spreadsheet_id_brasileirao,
    )

    info = load_service_account_info()
    if not info:
        raise ValueError(
            "Credenciais Google ausentes. Defina GOOGLE_SERVICE_ACCOUNT_FILE no .env "
            "ou [connections.gsheets] no Streamlit secrets."
        )
    sid = spreadsheet_id_brasileirao()
    df = ler_planilha_gsheets(info, sid)
    if df.empty:
        raise ValueError(f"Planilha {sid} retornou vazia.")
    return _dataframe_para_jogos(df), df, sid


def carregar_jogos(
    *, preferir_gsheets: bool = True
) -> tuple[list[Jogo], str, pd.DataFrame | None]:
    """Google Sheets (padrão) ou XLSX local."""
    if preferir_gsheets:
        try:
            jogos, df, sid = carregar_jogos_gsheets()
            return jogos, f"Google Sheets ({sid})", df
        except Exception:
            pass
    jogos = carregar_jogos_xlsx()
    df = pd.read_excel(ARQUIVO_CALENDARIO)
    return jogos, f"Arquivo local ({ARQUIVO_CALENDARIO.name})", df


def times_do_calendario(jogos: list[Jogo]) -> list[str]:
    ts = set()
    for j in jogos:
        ts.add(j.mand)
        ts.add(j.vis)
    return sorted(ts)


def _ols_pts_rodada(rodadas: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    """pts ~ intercept + beta_rodada * rodada (por jogo)."""
    if len(pts) == 0:
        return 1.0, 0.0
    if len(pts) == 1:
        return float(pts[0]), 0.0
    r = rodadas.astype(float)
    y = pts.astype(float)
    X = np.column_stack([np.ones(len(y)), r])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _pvalor_t_bilateral(t_stat: float, df: int) -> float:
    """p-valor bilateral da estatística t de Student."""
    if df <= 0 or not np.isfinite(t_stat):
        return float("nan")
    from scipy.stats import t as student_t

    return float(2.0 * student_t.sf(abs(t_stat), df))


def _ols_pvalues(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> list[float]:
    n, k = X.shape
    if n <= k:
        return [float("nan")] * k
    resid = y.astype(float) - X @ coef
    rss = float(np.sum(resid ** 2))
    sigma2 = rss / (n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    df = n - k
    return [_pvalor_t_bilateral(float(c / s) if s > 0 else float("nan"), df) for c, s in zip(coef, se)]


def _ols_r2(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    y = y.astype(float)
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pvalor_estrela(p: float | None) -> str:
    """Converte p-valor em marcador de significância."""
    if p is None or not np.isfinite(p):
        return "-"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "-"


def _ordem_variaveis_regressao(variante: VarianteRegressao) -> list[str]:
    cols = ["Intercepto", "Rodada", "Indicador casa", "Rodada × casa"]
    if variante == "interacao_adv_turno":
        return [
            "Intercepto",
            "Rodada",
            "Rodada²",
            "Indicador casa",
            "Rodada × casa",
            "Força adversário",
            "Forma recente",
        ]
    return cols


def _nomes_termos_regressao(
    interacao: bool,
    forca: bool,
    turno_flag: bool,
    forma_flag: bool,
    rodada2_flag: bool,
    casa_flag: bool,
) -> list[str]:
    nomes = ["Intercepto", "Rodada"]
    if rodada2_flag:
        nomes.append("Rodada²")
    if casa_flag:
        nomes.append("Indicador casa")
    if interacao:
        nomes.append("Rodada × casa")
    if forca:
        nomes.append("Força adversário")
    if turno_flag:
        nomes.append("Turno")
    if forma_flag:
        nomes.append("Forma recente")
    return nomes


def mapa_forca_adversario(
    jogos: list[Jogo], r_ini: int, r_fim: int
) -> dict[str, float]:
    """Média de pts/jogo no intervalo (força do adversário)."""
    return {
        t: media_pts_jogo(jogos, t, r_ini, r_fim, "simples")["geral"]
        for t in times_do_calendario(jogos)
    }


def _historico_pts_time(
    jogos: list[Jogo], time: str, *, incluir_proj: bool = False
) -> list[tuple[int, str, float]]:
    """Jogos do time com pts efetivos, ordenados por (rodada, hora)."""
    hist: list[tuple[int, str, float]] = []
    for j in jogos:
        if time not in (j.mand, j.vis):
            continue
        pts_val: float | None = None
        if j.jogado:
            pts_pair = j.pts_reais()
            if pts_pair is None:
                continue
            pm, pv = pts_pair
            pts_val = float(pm if j.mand == time else pv)
        elif incluir_proj and j.proj_pm is not None:
            pts_val = float(j.proj_pm if j.mand == time else j.proj_pv)
        if pts_val is None:
            continue
        hist.append((j.r, j.hora, pts_val))
    hist.sort(key=lambda x: (x[0], x[1]))
    return hist


def forma_recente_ate(
    jogos: list[Jogo],
    time: str,
    rodada: int,
    hora: str = "",
    *,
    n: int = FORMA_RECENTE_JOGOS,
    padrao: float = 1.0,
    incluir_proj: bool = False,
) -> float:
    """Média de pts nos últimos n jogos antes de (rodada, hora)."""
    anteriores: list[float] = []
    for r, h, pts in _historico_pts_time(jogos, time, incluir_proj=incluir_proj):
        if r < rodada or (r == rodada and h < hora):
            anteriores.append(pts)
    if not anteriores:
        return padrao
    return sum(anteriores[-n:]) / len(anteriores[-n:])


def forma_recente_atual(
    jogos: list[Jogo],
    time: str,
    *,
    n: int = FORMA_RECENTE_JOGOS,
    padrao: float = 1.0,
) -> float:
    """Média de pts nos últimos n jogos realizados (para projeção)."""
    hist = _historico_pts_time(jogos, time)
    if not hist:
        return padrao
    ultimos = [pts for _, _, pts in hist[-n:]]
    return sum(ultimos) / len(ultimos)


def _coletar_obs_regressao(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    rodadas: list[float] = []
    casa: list[float] = []
    turno: list[float] = []
    y_vals: list[float] = []
    adversarios: list[str] = []
    formas: list[float] = []

    jogos_time = [
        j
        for j in jogos
        if j.jogado
        and r_ini <= j.r <= r_fim
        and (j.mand == time or j.vis == time)
    ]
    jogos_time.sort(key=lambda x: (x.r, x.hora))

    for j in jogos_time:
        parsed = parse_placar(j.placar)
        if parsed is None:
            continue
        gm, gv = parsed
        formas.append(forma_recente_ate(jogos, time, j.r, j.hora))
        if j.mand == time:
            rodadas.append(float(j.r))
            casa.append(1.0)
            turno.append(1.0 if j.r >= 20 else 0.0)
            adversarios.append(j.vis)
            y_vals.append(float(pontos_placar(gm, gv)[0]))
        else:
            rodadas.append(float(j.r))
            casa.append(0.0)
            turno.append(1.0 if j.r >= 20 else 0.0)
            adversarios.append(j.mand)
            y_vals.append(float(pontos_placar(gm, gv)[1]))

    return (
        np.array(rodadas, dtype=float),
        np.array(casa, dtype=float),
        np.array(turno, dtype=float),
        np.array(y_vals, dtype=float),
        adversarios,
        np.array(formas, dtype=float),
    )


def _ajustar_regressao(
    rodadas: np.ndarray,
    casa: np.ndarray,
    turno: np.ndarray,
    y: np.ndarray,
    adversarios: list[str],
    forca_map: dict[str, float],
    variante: VarianteRegressao,
    forma: np.ndarray | None = None,
) -> dict[str, float]:
    """Ajusta OLS conforme a variante selecionada."""
    usa_interacao = variante != "casa_sem_interacao"
    usa_forca = variante == "interacao_adv_turno"
    usa_turno = False
    usa_forma = variante == "interacao_adv_turno"
    usa_rodada2 = variante == "interacao_adv_turno"

    n = len(y)
    if n == 0:
        return _coeficientes_vazios(variante, 1.0)

    r = rodadas.astype(float)
    forma_arr = forma.astype(float) if forma is not None and len(forma) == n else None
    b0_g, b1_g = _ols_pts_rodada(r, y)

    def _montar(
        interacao: bool,
        forca: bool,
        turno_flag: bool,
        forma_flag: bool,
        rodada2_flag: bool,
        casa_flag: bool,
    ):
        cols = [np.ones(n), r]
        if rodada2_flag:
            cols.append(r ** 2)
        if casa_flag:
            cols.append(casa.astype(float))
        if interacao:
            cols.append(r * casa.astype(float))
        if forca:
            cols.append(
                np.array([forca_map.get(a, 1.0) for a in adversarios], dtype=float)
            )
        if turno_flag:
            cols.append(turno.astype(float))
        if forma_flag and forma_arr is not None:
            cols.append(forma_arr)
        return cols

    def _extrair(
        coef: np.ndarray,
        interacao: bool,
        forca: bool,
        turno_flag: bool,
        forma_flag: bool,
        rodada2_flag: bool,
        casa_flag: bool,
    ):
        idx = 0
        b0 = float(coef[idx]); idx += 1
        b1 = float(coef[idx]); idx += 1
        b_r2 = b2 = b3 = b4 = b5 = b6 = 0.0
        if rodada2_flag:
            b_r2 = float(coef[idx]); idx += 1
        if casa_flag:
            b2 = float(coef[idx]); idx += 1
        if interacao:
            b3 = float(coef[idx]); idx += 1
        if forca:
            b4 = float(coef[idx]); idx += 1
        if turno_flag:
            b5 = float(coef[idx]); idx += 1
        if forma_flag:
            b6 = float(coef[idx])
        return b0, b1, b_r2, b2, b3, b4, b5, b6

    tentativas = [
        (usa_interacao, usa_forca, usa_turno, usa_forma, usa_rodada2, True),
        (usa_interacao, usa_forca, usa_turno, usa_forma, False, True),
        (usa_interacao, usa_forca, usa_turno, False, False, True),
        (usa_interacao, False, usa_turno, False, False, True),
        (usa_interacao, False, False, False, False, True),
        (False, False, False, False, False, True),
        (False, False, False, False, False, False),
    ]
    seen: set[tuple[bool, bool, bool, bool, bool, bool]] = set()
    b0, b1, b_r2, b2, b3, b4, b5, b6 = b0_g, b1_g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fit_interacao = fit_forca = fit_turno = fit_forma = fit_rodada2 = fit_casa = usa_interacao
    termos: list[dict[str, float | str]] = []
    r2 = float("nan")
    for interacao, forca, turno_flag, forma_flag, rodada2_flag, casa_flag in tentativas:
        key = (interacao, forca, turno_flag, forma_flag, rodada2_flag, casa_flag)
        if key in seen:
            continue
        seen.add(key)
        cols = _montar(interacao, forca, turno_flag, forma_flag, rodada2_flag, casa_flag)
        if n < len(cols):
            continue
        X = np.column_stack(cols)
        coef, _, _, _ = np.linalg.lstsq(X, y.astype(float), rcond=None)
        pvals = _ols_pvalues(X, y.astype(float), coef)
        nomes = _nomes_termos_regressao(
            interacao, forca, turno_flag, forma_flag, rodada2_flag, casa_flag
        )
        termos = [
            {
                "Variável": nomes[i],
                "Beta": round(float(coef[i]), 4),
                "p-valor": round(float(pvals[i]), 4) if np.isfinite(pvals[i]) else None,
            }
            for i in range(len(nomes))
        ]
        r2 = _ols_r2(X, y.astype(float), coef)
        b0, b1, b_r2, b2, b3, b4, b5, b6 = _extrair(
            coef, interacao, forca, turno_flag, forma_flag, rodada2_flag, casa_flag
        )
        fit_interacao = interacao
        fit_forca = forca
        fit_turno = turno_flag
        fit_forma = forma_flag
        fit_rodada2 = rodada2_flag
        fit_casa = casa_flag
        break

    return {
        "geral": max(b1_g, 0.0),
        "casa": max(b1 + (b3 if fit_interacao else 0.0), 0.0),
        "fora": max(b1, 0.0),
        "intercept": b0,
        "beta_rodada": b1,
        "beta_rodada2": b_r2 if fit_rodada2 else 0.0,
        "beta_casa_ind": b2 if fit_casa else 0.0,
        "beta_interacao": b3 if fit_interacao else 0.0,
        "beta_forca": b4 if fit_forca else 0.0,
        "beta_turno": b5 if fit_turno else 0.0,
        "beta_forma": b6 if fit_forma else 0.0,
        "variante": variante,
        "usa_casa": fit_casa,
        "usa_interacao": fit_interacao,
        "usa_forca": fit_forca,
        "usa_turno": fit_turno,
        "usa_forma": fit_forma,
        "usa_rodada2": fit_rodada2,
        "termos": termos,
        "n_obs": n,
        "r2": round(r2, 4) if termos and np.isfinite(r2) else None,
    }


def _coeficientes_vazios(
    variante: VarianteRegressao,
    padrao: float,
) -> dict[str, float]:
    return {
        "geral": padrao,
        "casa": padrao,
        "fora": padrao,
        "intercept": padrao,
        "beta_rodada": 0.0,
        "beta_rodada2": 0.0,
        "beta_casa_ind": 0.0,
        "beta_interacao": 0.0,
        "beta_forca": 0.0,
        "beta_turno": 0.0,
        "beta_forma": 0.0,
        "variante": variante,
        "usa_casa": variante != "casa_sem_interacao",
        "usa_interacao": variante != "casa_sem_interacao",
        "usa_forca": variante == "interacao_adv_turno",
        "usa_turno": False,
        "usa_forma": variante == "interacao_adv_turno",
        "usa_rodada2": variante == "interacao_adv_turno",
        "termos": [],
        "n_obs": 0,
        "r2": None,
    }


def _prever_regressao(
    b: dict[str, float],
    *,
    rodada: int,
    em_casa: bool,
    adversario: str,
    forca_map: dict[str, float],
) -> float:
    r = float(rodada)
    casa = 1.0 if em_casa else 0.0
    val = b["intercept"] + b["beta_rodada"] * r
    if b.get("usa_rodada2"):
        val += b.get("beta_rodada2", 0.0) * r * r
    if b.get("usa_casa"):
        val += b.get("beta_casa_ind", 0.0) * casa
    if b.get("usa_interacao"):
        val += b.get("beta_interacao", 0.0) * r * casa
    if b.get("usa_forca"):
        val += b.get("beta_forca", 0.0) * forca_map.get(adversario, 1.0)
    if b.get("usa_turno"):
        val += b.get("beta_turno", 0.0) * (1.0 if rodada >= 20 else 0.0)
    if b.get("usa_forma"):
        val += b.get("beta_forma", 0.0) * b.get("forma_recente", 1.0)
    return max(val, 0.0)


def regressao_beta(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressao,
    forca_map: dict[str, float],
) -> dict[str, float]:
    """Coeficientes de regressão por jogo conforme variante."""
    rodadas, casa, turno, y, adversarios, formas = _coletar_obs_regressao(
        jogos, time, r_ini, r_fim
    )
    coefs = _ajustar_regressao(
        rodadas, casa, turno, y, adversarios, forca_map, variante, formas
    )
    if variante == "interacao_adv_turno":
        coefs["forma_recente"] = forma_recente_atual(jogos, time)
    return coefs


def _ordem_variaveis_regressao_acumulada(variante: VarianteRegressaoAcumulada) -> list[str]:
    if variante == "momento_aceleracao":
        return [
            "Intercepto",
            "Rodada",
            "Rodada ao Quadrado",
            "Interação Rodada × Time",
            "Interação Rodada ao Quadrado × Time",
            "Forma Recente",
        ]
    if variante == "momento_historico":
        return [
            "Intercepto",
            "Rodada",
            "Interação Rodada × Time",
            "Proporção Casa",
            "Força dos Adversários Passados",
        ]
    if variante == "completa_limites":
        return [
            "Intercepto",
            "Rodada Centrada",
            "Rodada Centrada ao Quadrado",
            "Interação Rodada Centrada × Time",
            "Interação Rodada Centrada ao Quadrado × Time",
            "Forma Recente",
            "Força dos Adversários Passados",
            "Proporção Casa",
            "Dias de Descanso",
            "Classificatórias",
            "Oitavas",
            "Quartas",
            "Semi",
            "Final",
        ]
    return [
        "Intercepto",
        "Rodada",
        "Rodada ao Quadrado",
        "Interação Rodada × Time",
        "Interação Rodada ao Quadrado × Time",
        "Forma Recente",
        "Força dos Adversários Passados",
        "Proporção Casa",
        "Dias de Descanso",
        "Classificatórias",
        "Oitavas",
        "Quartas",
        "Semi",
        "Final",
    ]


def _flags_regressao_acumulada(variante: VarianteRegressaoAcumulada) -> dict[str, bool]:
    if variante == "momento_aceleracao":
        return {
            "usa_rodada": True,
            "usa_rodada2": True,
            "usa_interacao_rodada": True,
            "usa_interacao_rodada2": True,
            "usa_forma": True,
            "usa_prop_casa": False,
            "usa_forca": False,
            "usa_descanso": True,
            "usa_importante": True,
            "usa_rodada_centrada": False,
            "limita_delta_rodada": True,
            "forma_decaindo": True,
        }
    if variante == "momento_historico":
        return {
            "usa_rodada": True,
            "usa_rodada2": False,
            "usa_interacao_rodada": True,
            "usa_interacao_rodada2": False,
            "usa_forma": False,
            "usa_prop_casa": True,
            "usa_forca": True,
            "usa_descanso": True,
            "usa_importante": True,
            "usa_rodada_centrada": False,
            "limita_delta_rodada": True,
            "forma_decaindo": True,
        }
    if variante == "completa_limites":
        return {
            "usa_rodada": True,
            "usa_rodada2": True,
            "usa_interacao_rodada": True,
            "usa_interacao_rodada2": True,
            "usa_forma": True,
            "usa_prop_casa": True,
            "usa_forca": True,
            "usa_descanso": True,
            "usa_importante": True,
            "usa_rodada_centrada": True,
            "limita_delta_rodada": True,
            "forma_decaindo": True,
        }
    return {
        "usa_rodada": True,
        "usa_rodada2": True,
        "usa_interacao_rodada": True,
        "usa_interacao_rodada2": True,
        "usa_forma": True,
        "usa_prop_casa": True,
        "usa_forca": True,
        "usa_descanso": True,
        "usa_importante": True,
        "usa_rodada_centrada": False,
        "limita_delta_rodada": True,
        "forma_decaindo": True,
    }


def peso_forma_recente_horizonte(horizonte: int) -> float:
    """
    Peso da forma recente na mistura com a forma geral.
    Horizonte 1 (próxima rodada) → 80%; horizonte 5 → 50%; piso 20%.
    """
    h = max(1, int(horizonte))
    # Interpolação linear entre h=1 e h=5
    passo = (PESO_FORMA_RECENTE_H1 - PESO_FORMA_RECENTE_H5) / 4.0
    w = PESO_FORMA_RECENTE_H1 - passo * (h - 1)
    return float(np.clip(w, PESO_FORMA_RECENTE_PISO, PESO_FORMA_RECENTE_H1))



def _contagem_casa_fora_ate(
    jogos: list[Jogo],
    time: str,
    ate_rodada: int,
    *,
    incluir_proj: bool = False,
) -> tuple[int, int]:
    """Jogos em casa e fora disputados (ou projetados) até a rodada."""
    n_casa = n_fora = 0
    for j in jogos:
        if j.r > ate_rodada or time not in (j.mand, j.vis):
            continue
        if j.jogado:
            if j.mand == time:
                n_casa += 1
            else:
                n_fora += 1
        elif incluir_proj and j.proj_pm is not None:
            if j.mand == time:
                n_casa += 1
            else:
                n_fora += 1
    return n_casa, n_fora


def _proporcao_casa(n_casa: int, n_fora: int) -> float:
    """Proporção jogos em casa sobre jogos fora."""
    if n_fora == 0:
        return float(n_casa) if n_casa > 0 else 1.0
    return n_casa / n_fora


def _proporcao_casa_ate(
    jogos: list[Jogo],
    time: str,
    rodada: int,
    *,
    em_casa_jogo: bool,
    incluir_proj: bool = False,
) -> float:
    """Proporção casa/fora ao fim da rodada, incluindo o jogo corrente."""
    n_casa, n_fora = _contagem_casa_fora_ate(
        jogos, time, rodada - 1, incluir_proj=incluir_proj
    )
    if em_casa_jogo:
        n_casa += 1
    else:
        n_fora += 1
    return _proporcao_casa(n_casa, n_fora)


def _forca_oponentes_passados(
    jogos: list[Jogo],
    time: str,
    antes_rodada: int,
    forca_map: dict[str, float],
    *,
    incluir_proj: bool = False,
) -> float:
    """Média da pontuação/jogo dos adversários já enfrentados."""
    vals: list[float] = []
    for j in jogos:
        if j.r >= antes_rodada or time not in (j.mand, j.vis):
            continue
        if j.jogado:
            adv = j.vis if j.mand == time else j.mand
            vals.append(forca_map.get(adv, 1.0))
        elif incluir_proj and j.proj_pm is not None:
            adv = j.vis if j.mand == time else j.mand
            vals.append(forca_map.get(adv, 1.0))
    return sum(vals) / len(vals) if vals else 1.0


def _forca_pts_jogo_time(
    jogos: list[Jogo],
    time: str,
    ate_rodada: int,
) -> float:
    """Média de pts/jogo do time até a rodada (real + projetado)."""
    pts, n = 0.0, 0
    for j in jogos:
        if j.r > ate_rodada or time not in (j.mand, j.vis):
            continue
        if j.jogado:
            pm, pv = j.pts_reais()
            pts += float(pm if j.mand == time else pv)
            n += 1
        elif j.proj_pm is not None:
            pts += float(j.proj_pm if j.mand == time else j.proj_pv)
            n += 1
    return pts / n if n else 1.0


def _coletar_obs_regressao_acumulada(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
    forca_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pts acumulados ao fim de cada rodada disputada no intervalo."""
    rodadas: list[float] = []
    y_vals: list[float] = []
    props: list[float] = []
    forcas: list[float] = []
    formas: list[float] = []

    for r in range(r_ini, r_fim + 1):
        jogo_r = jogo_do_time_na_rodada(jogos, time, r)
        if jogo_r is None or not jogo_r.jogado:
            continue
        acum = stats_acumuladas_ate(jogos, time, r, so_realizados=True)
        n_casa, n_fora = _contagem_casa_fora_ate(jogos, time, r)
        rodadas.append(float(r))
        y_vals.append(float(acum.pts))
        props.append(_proporcao_casa(n_casa, n_fora))
        forcas.append(_forca_oponentes_passados(jogos, time, r + 1, forca_map))
        formas.append(forma_recente_ate(jogos, time, r, jogo_r.hora))

    return (
        np.array(rodadas, dtype=float),
        np.array(y_vals, dtype=float),
        np.array(props, dtype=float),
        np.array(forcas, dtype=float),
        np.array(formas, dtype=float),
    )


def _ajustar_regressao_acumulada(
    rodadas: np.ndarray,
    y: np.ndarray,
    prop_casa: np.ndarray,
    forca: np.ndarray,
    forma: np.ndarray,
    variante: VarianteRegressaoAcumulada,
) -> dict:
    """Ajusta pts acumulados conforme variante de regressão."""
    n = len(y)
    if n == 0:
        return _coeficientes_vazios_acumulada(variante)

    r = rodadas.astype(float)
    y_f = y.astype(float)
    prop = prop_casa.astype(float)
    forca_arr = forca.astype(float)
    forma_arr = forma.astype(float)
    flags_alvo = _flags_regressao_acumulada(variante)

    def _montar(
        rodada_flag: bool,
        rodada2_flag: bool,
        prop_flag: bool,
        forca_flag: bool,
        forma_flag: bool,
    ):
        cols = [np.ones(n)]
        if rodada_flag:
            cols.append(r)
        if rodada2_flag:
            cols.append(r ** 2)
        if forma_flag:
            cols.append(forma_arr)
        if prop_flag:
            cols.append(prop)
        if forca_flag:
            cols.append(forca_arr)
        return cols

    def _nomes(
        rodada_flag: bool,
        rodada2_flag: bool,
        prop_flag: bool,
        forca_flag: bool,
        forma_flag: bool,
    ) -> list[str]:
        nomes = ["Intercepto"]
        if rodada_flag:
            nomes.append("Rodada")
        if rodada2_flag:
            nomes.append("Rodada ao Quadrado")
        if forma_flag:
            nomes.append("Forma Recente")
        if prop_flag:
            nomes.append("Proporção Casa")
        if forca_flag:
            nomes.append("Força dos Adversários Passados")
        return nomes

    def _extrair_coefs(
        coef: np.ndarray,
        rodada_flag: bool,
        rodada2_flag: bool,
        prop_flag: bool,
        forca_flag: bool,
        forma_flag: bool,
    ) -> dict[str, float]:
        idx = 0
        out = {
            "intercept": float(coef[idx]),
            "beta_rodada": 0.0,
            "beta_rodada2": 0.0,
            "beta_forma": 0.0,
            "beta_forca": 0.0,
            "beta_prop_casa": 0.0,
        }
        idx += 1
        if rodada_flag:
            out["beta_rodada"] = float(coef[idx]); idx += 1
        if rodada2_flag:
            out["beta_rodada2"] = float(coef[idx]); idx += 1
        if forma_flag:
            out["beta_forma"] = float(coef[idx]); idx += 1
        if prop_flag:
            out["beta_prop_casa"] = float(coef[idx]); idx += 1
        if forca_flag:
            out["beta_forca"] = float(coef[idx])
        return out

    if variante == "completa":
        tentativas = [
            (True, True, True, True, True),
            (True, True, True, True, False),
            (True, True, False, True, False),
            (True, True, False, False, False),
            (True, False, False, False, False),
            (False, False, False, False, False),
        ]
    else:
        f = flags_alvo
        tentativas = [
            (
                f["usa_rodada"],
                f["usa_rodada2"],
                f["usa_prop_casa"],
                f["usa_forca"],
                f["usa_forma"],
            )
        ]

    termos: list[dict] = []
    r2 = float("nan")
    fit_flags = flags_alvo.copy()
    coef_vals = _coeficientes_vazios_acumulada(variante)

    for rodada_flag, rodada2_flag, prop_flag, forca_flag, forma_flag in tentativas:
        cols = _montar(rodada_flag, rodada2_flag, prop_flag, forca_flag, forma_flag)
        if n < len(cols):
            continue
        X = np.column_stack(cols)
        coef, _, _, _ = np.linalg.lstsq(X, y_f, rcond=None)
        pvals = _ols_pvalues(X, y_f, coef)
        nomes = _nomes(rodada_flag, rodada2_flag, prop_flag, forca_flag, forma_flag)
        termos = [
            {
                "Variável": nomes[i],
                "Beta": round(float(coef[i]), 4),
                "p-valor": round(float(pvals[i]), 4) if np.isfinite(pvals[i]) else None,
            }
            for i in range(len(nomes))
        ]
        r2 = _ols_r2(X, y_f, coef)
        coef_vals.update(
            _extrair_coefs(
                coef, rodada_flag, rodada2_flag, prop_flag, forca_flag, forma_flag
            )
        )
        fit_flags = {
            "usa_rodada": rodada_flag,
            "usa_rodada2": rodada2_flag,
            "usa_prop_casa": prop_flag,
            "usa_forca": forca_flag,
            "usa_forma": forma_flag,
        }
        break

    return {
        **coef_vals,
        **fit_flags,
        "variante_acum": variante,
        "termos": termos,
        "n_obs": n,
        "r2": round(r2, 4) if termos and np.isfinite(r2) else None,
    }


def _coeficientes_vazios_acumulada(variante: VarianteRegressaoAcumulada) -> dict:
    flags = _flags_regressao_acumulada(variante)
    return {
        "intercept": 1.0,
        "beta_rodada": 0.0,
        "beta_rodada2": 0.0,
        "beta_prop_casa": 0.0,
        "beta_forca": 0.0,
        "beta_forma": 0.0,
        "gamma_rodada": 0.0,
        "gamma_rodada2": 0.0,
        **flags,
        "usa_rodada": True,
        "usa_rodada2": bool(
            flags.get("usa_rodada2") or flags.get("usa_interacao_rodada2")
        ),
        "variante_acum": variante,
        "termos": [],
        "n_obs": 0,
        "r2": None,
    }


def _coletar_painel_efeitos_fixos(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    forca_map: dict[str, float],
    *,
    metrica: str | None = None,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Painel time×rodada: Y=PA + R, PC, FAP, FR, descanso, jogos importantes."""
    try:
        from prob_ml.context_calendar import context_for_team
    except Exception:
        context_for_team = None  # type: ignore

    times_obs: list[str] = []
    rodadas: list[float] = []
    y_vals: list[float] = []
    props: list[float] = []
    forcas: list[float] = []
    formas: list[float] = []
    descansos: list[float] = []
    importantes: list[str] = []

    for time in times_do_calendario(jogos):
        for r in range(r_ini, r_fim + 1):
            jogo_r = jogo_do_time_na_rodada(jogos, time, r)
            if jogo_r is None or not jogo_r.jogado:
                continue
            if metrica is None:
                y_val = float(
                    stats_acumuladas_ate(jogos, time, r, so_realizados=True).pts
                )
            else:
                agg = _agregar_stats_time(jogos, time, r_ini, r)
                raw = _metricas_estatisticas_de_agregado(agg).get(metrica, 0.0)
                try:
                    y_val = float(raw)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(y_val):
                    continue
            n_casa, n_fora = _contagem_casa_fora_ate(jogos, time, r)
            times_obs.append(time)
            rodadas.append(float(r))
            y_vals.append(y_val)
            props.append(_proporcao_casa(n_casa, n_fora))
            forcas.append(_forca_oponentes_passados(jogos, time, r + 1, forca_map))
            formas.append(forma_recente_ate(jogos, time, r, jogo_r.hora))
            if context_for_team is not None:
                d, imp = context_for_team(time, jogo_r.data)
            else:
                d, imp = 7.0, "Não tem"
            descansos.append(float(d))
            importantes.append(str(imp))

    return (
        times_obs,
        np.array(rodadas, dtype=float),
        np.array(y_vals, dtype=float),
        np.array(props, dtype=float),
        np.array(forcas, dtype=float),
        np.array(formas, dtype=float),
        np.array(descansos, dtype=float),
        importantes,
    )


def ajustar_painel_efeitos_fixos(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressaoAcumulada,
    forca_map: dict[str, float] | None = None,
    *,
    metrica: str | None = None,
    blocos_extra: list[list[Jogo]] | None = None,
) -> dict:
    """
    Painel FE por variante:
    Y (pontos acumulados ou métrica) = Efeito Fixo do Time + controles
    + Interação Rodada × Time (+ Interação Rodada ao Quadrado × Time, se aplicável).

    Controles comuns seguem a variante.
    Identificação: time de referência com interações nulas.

    ``blocos_extra``: outros campeonatos/temporadas (Série B/C/D, feminino, etc.).
    Cada bloco tem força/forma calculadas internamente. Só entra no treino da
    regressão — modos média e repetir 1º turno não usam isso.
    """
    flags = _flags_regressao_acumulada(variante)
    fm = forca_map or mapa_forca_adversario(jogos, r_ini, r_fim)
    times_alvo = times_do_calendario(jogos)
    cal_set = set(times_alvo)

    times_obs, r, y, prop, forca, forma, descanso, importante = _coletar_painel_efeitos_fixos(
        jogos, r_ini, r_fim, fm, metrica=metrica
    )
    times_obs_l = list(times_obs)
    r_l = list(r.tolist()) if hasattr(r, "tolist") else list(r)
    y_l = list(y.tolist()) if hasattr(y, "tolist") else list(y)
    prop_l = list(prop.tolist()) if hasattr(prop, "tolist") else list(prop)
    forca_l = list(forca.tolist()) if hasattr(forca, "tolist") else list(forca)
    forma_l = list(forma.tolist()) if hasattr(forma, "tolist") else list(forma)
    descanso_l = list(descanso.tolist()) if hasattr(descanso, "tolist") else list(descanso)
    importante_l = list(importante)

    if blocos_extra:
        try:
            from brasileirao_multi_liga import remap_block_teams
        except Exception:
            remap_block_teams = None  # type: ignore
        for bloco in blocos_extra:
            if not bloco:
                continue
            bloco_use = (
                remap_block_teams(bloco, cal_set) if remap_block_teams else bloco
            )
            r0 = min(j.r for j in bloco_use)
            r1 = max(j.r for j in bloco_use)
            fm_b = mapa_forca_adversario(bloco_use, r0, r1)
            t_b, r_b, y_b, p_b, f_b, fr_b, d_b, i_b = _coletar_painel_efeitos_fixos(
                bloco_use, r0, r1, fm_b, metrica=metrica
            )
            times_obs_l.extend(t_b)
            r_l.extend(r_b.tolist())
            y_l.extend(y_b.tolist())
            prop_l.extend(p_b.tolist())
            forca_l.extend(f_b.tolist())
            forma_l.extend(fr_b.tolist())
            descanso_l.extend(d_b.tolist())
            importante_l.extend(i_b)

    times_obs = times_obs_l
    r = np.array(r_l, dtype=float)
    y = np.array(y_l, dtype=float)
    prop = np.array(prop_l, dtype=float)
    forca = np.array(forca_l, dtype=float)
    forma = np.array(forma_l, dtype=float)
    descanso = np.array(descanso_l, dtype=float)
    importante = importante_l

    # FE: times do calendário + quaisquer times extras que apareceram no painel
    times = list(times_alvo)
    for t in sorted(set(times_obs)):
        if t not in cal_set:
            times.append(t)
    n = len(y)
    vazio = {
        "times": times,
        "time_ref": times[0] if times else "",
        "variante": variante,
        "r2": None,
        "n_obs": 0,
        "comum": {
            "beta_rodada": 0.0,
            "beta_rodada2": 0.0,
            "beta_forma": 0.0,
            "beta_forca": 0.0,
            "beta_prop_casa": 0.0,
            "beta_descanso": 0.0,
            "beta_classificatorias": 0.0,
            "beta_oitavas": 0.0,
            "beta_quartas": 0.0,
            "beta_semi": 0.0,
            "beta_final": 0.0,
            "p_rodada": None,
            "p_rodada2": None,
            "p_forma": None,
            "p_forca": None,
            "p_prop_casa": None,
            "p_descanso": None,
            "p_classificatorias": None,
            "p_oitavas": None,
            "p_quartas": None,
            "p_semi": None,
            "p_final": None,
            **flags,
        },
        "por_time": {
            t: {
                "intercept": 1.0,
                "gamma_rodada": 0.0,
                "gamma_rodada2": 0.0,
                "p_intercept": None,
                "p_gamma_rodada": None,
                "p_gamma_rodada2": None,
            }
            for t in times
        },
    }
    if n == 0 or not times:
        return vazio

    time_ref = times_alvo[0] if times_alvo else times[0]
    # Interações Rodada×Time só para o calendário-alvo; demais ligas
    # contribuem via FE de nível + controles comuns (forma, força, prop. casa).
    non_ref_inter = [t for t in times_alvo if t != time_ref]
    non_ref = [t for t in times if t != time_ref]
    usa_centrada = flags.get("usa_rodada_centrada", False)
    r_c = r - float(RODADA_CENTRO)
    r_lin = r_c if usa_centrada else r
    r2_arr = (r_c ** 2) if usa_centrada else (r ** 2)
    nome_r = "Rodada Centrada" if usa_centrada else "Rodada"
    nome_r2 = "Rodada Centrada ao Quadrado" if usa_centrada else "Rodada ao Quadrado"
    nome_int_r_base = (
        "Interação Rodada Centrada" if usa_centrada else "Interação Rodada"
    )
    nome_int_r2_base = (
        "Interação Rodada Centrada ao Quadrado"
        if usa_centrada
        else "Interação Rodada ao Quadrado"
    )
    usa_int_r = flags.get("usa_interacao_rodada", True)
    usa_int_r2 = flags.get("usa_interacao_rodada2", True)

    cols: list[np.ndarray] = []
    nomes: list[str] = []

    for t in times:
        cols.append(np.array([1.0 if x == t else 0.0 for x in times_obs], dtype=float))
        nomes.append(f"Efeito Fixo [{t}]")

    if flags["usa_rodada"]:
        cols.append(r_lin)
        nomes.append(nome_r)
    if flags["usa_rodada2"]:
        cols.append(r2_arr)
        nomes.append(nome_r2)

    if usa_int_r:
        for t in non_ref_inter:
            d = np.array([1.0 if x == t else 0.0 for x in times_obs], dtype=float)
            cols.append(r_lin * d)
            nomes.append(f"{nome_int_r_base} × [{t}]")
    if usa_int_r2:
        for t in non_ref_inter:
            d = np.array([1.0 if x == t else 0.0 for x in times_obs], dtype=float)
            cols.append(r2_arr * d)
            nomes.append(f"{nome_int_r2_base} × [{t}]")

    if flags["usa_forma"]:
        cols.append(forma)
        nomes.append("Forma Recente")
    if flags["usa_forca"]:
        cols.append(forca)
        nomes.append("Força dos Adversários Passados")
    if flags["usa_prop_casa"]:
        cols.append(prop)
        nomes.append("Proporção Casa")
    if flags.get("usa_descanso", True):
        cols.append(descanso)
        nomes.append("Dias de Descanso")
    if flags.get("usa_importante", True):
        for lab in ("Classificatórias", "Oitavas", "Quartas", "Semi", "Final"):
            cols.append(
                np.array([1.0 if x == lab else 0.0 for x in importante], dtype=float)
            )
            nomes.append(lab)

    X = np.column_stack(cols)
    # Se poucas obs, remove interações e tenta de novo
    if n < X.shape[1]:
        cols = []
        nomes = []
        for t in times:
            cols.append(
                np.array([1.0 if x == t else 0.0 for x in times_obs], dtype=float)
            )
            nomes.append(f"Efeito Fixo [{t}]")
        if flags["usa_rodada"]:
            cols.append(r_lin)
            nomes.append(nome_r)
        if flags["usa_rodada2"]:
            cols.append(r2_arr)
            nomes.append(nome_r2)
        if flags["usa_forma"]:
            cols.append(forma)
            nomes.append("Forma Recente")
        if flags["usa_forca"]:
            cols.append(forca)
            nomes.append("Força dos Adversários Passados")
        if flags["usa_prop_casa"]:
            cols.append(prop)
            nomes.append("Proporção Casa")
        if flags.get("usa_descanso", True):
            cols.append(descanso)
            nomes.append("Dias de Descanso")
        if flags.get("usa_importante", True):
            for lab in ("Classificatórias", "Oitavas", "Quartas", "Semi", "Final"):
                cols.append(
                    np.array(
                        [1.0 if x == lab else 0.0 for x in importante], dtype=float
                    )
                )
                nomes.append(lab)
        X = np.column_stack(cols)
        non_ref = []
        non_ref_inter = []
        usa_int_r = False
        usa_int_r2 = False

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pvals = _ols_pvalues(X, y, coef)
    r2_fit = _ols_r2(X, y, coef)

    idx = 0
    por_time: dict[str, dict] = {}
    for t in times:
        por_time[t] = {
            "intercept": float(coef[idx]),
            "gamma_rodada": 0.0,
            "gamma_rodada2": 0.0,
            "p_intercept": (
                round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
            ),
            "p_gamma_rodada": None,
            "p_gamma_rodada2": None,
        }
        idx += 1

    b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    p_b1 = p_b2 = p_b3 = p_b4 = p_b5 = p_b6 = None
    b_imp = {k: 0.0 for k in ("classificatorias", "oitavas", "quartas", "semi", "final")}
    p_imp = {k: None for k in b_imp}

    if flags["usa_rodada"]:
        b1 = float(coef[idx])
        p_b1 = round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
        idx += 1
    if flags["usa_rodada2"]:
        b2 = float(coef[idx])
        p_b2 = round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
        idx += 1

    if usa_int_r:
        for t in non_ref_inter:
            por_time[t]["gamma_rodada"] = float(coef[idx])
            por_time[t]["p_gamma_rodada"] = (
                round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
            )
            idx += 1
    if usa_int_r2:
        for t in non_ref_inter:
            por_time[t]["gamma_rodada2"] = float(coef[idx])
            por_time[t]["p_gamma_rodada2"] = (
                round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
            )
            idx += 1

    if flags["usa_forma"]:
        b3 = float(coef[idx])
        p_b3 = round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
        idx += 1
    if flags["usa_forca"]:
        b4 = float(coef[idx])
        p_b4 = round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
        idx += 1
    if flags["usa_prop_casa"]:
        b5 = float(coef[idx])
        p_b5 = round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
        idx += 1
    if flags.get("usa_descanso", True):
        b6 = float(coef[idx])
        p_b6 = round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
        idx += 1
    if flags.get("usa_importante", True):
        for key in ("classificatorias", "oitavas", "quartas", "semi", "final"):
            b_imp[key] = float(coef[idx])
            p_imp[key] = (
                round(float(pvals[idx]), 4) if np.isfinite(pvals[idx]) else None
            )
            idx += 1

    return {
        "times": times,
        "time_ref": time_ref,
        "variante": variante,
        "r2": round(r2_fit, 4) if np.isfinite(r2_fit) else None,
        "n_obs": n,
        "comum": {
            "beta_rodada": b1,
            "beta_rodada2": b2,
            "beta_forma": b3,
            "beta_forca": b4,
            "beta_prop_casa": b5,
            "beta_descanso": b6,
            "beta_classificatorias": b_imp["classificatorias"],
            "beta_oitavas": b_imp["oitavas"],
            "beta_quartas": b_imp["quartas"],
            "beta_semi": b_imp["semi"],
            "beta_final": b_imp["final"],
            "p_rodada": p_b1,
            "p_rodada2": p_b2,
            "p_forma": p_b3,
            "p_forca": p_b4,
            "p_prop_casa": p_b5,
            "p_descanso": p_b6,
            "p_classificatorias": p_imp["classificatorias"],
            "p_oitavas": p_imp["oitavas"],
            "p_quartas": p_imp["quartas"],
            "p_semi": p_imp["semi"],
            "p_final": p_imp["final"],
            **flags,
        },
        "por_time": por_time,
    }


def coeficientes_efeitos_fixos_por_time(painel: dict) -> dict[str, dict]:
    """Converte painel FE em coeficientes por time para projeção/tabela."""
    comum = painel["comum"]
    variante = painel.get("variante", "completa")
    flags = _flags_regressao_acumulada(variante)
    out: dict[str, dict] = {}

    for t, fe in painel["por_time"].items():
        g1 = float(fe["gamma_rodada"])
        g2 = float(fe["gamma_rodada2"])
        b1 = float(comum["beta_rodada"])
        b2 = float(comum["beta_rodada2"])

        termos: list[dict] = [
            {
                "Variável": "Intercepto",
                "Beta": round(float(fe["intercept"]), 4),
                "p-valor": fe.get("p_intercept"),
            }
        ]
        if flags["usa_rodada"]:
            termos.append(
                {
                    "Variável": (
                        "Rodada Centrada"
                        if flags.get("usa_rodada_centrada")
                        else "Rodada"
                    ),
                    "Beta": round(b1, 4),
                    "p-valor": comum.get("p_rodada"),
                }
            )
        if flags["usa_rodada2"]:
            termos.append(
                {
                    "Variável": (
                        "Rodada Centrada ao Quadrado"
                        if flags.get("usa_rodada_centrada")
                        else "Rodada ao Quadrado"
                    ),
                    "Beta": round(b2, 4),
                    "p-valor": comum.get("p_rodada2"),
                }
            )
        if flags.get("usa_interacao_rodada", True):
            termos.append(
                {
                    "Variável": (
                        "Interação Rodada Centrada × Time"
                        if flags.get("usa_rodada_centrada")
                        else "Interação Rodada × Time"
                    ),
                    "Beta": round(g1, 4),
                    "p-valor": fe.get("p_gamma_rodada"),
                }
            )
        if flags.get("usa_interacao_rodada2", True):
            termos.append(
                {
                    "Variável": (
                        "Interação Rodada Centrada ao Quadrado × Time"
                        if flags.get("usa_rodada_centrada")
                        else "Interação Rodada ao Quadrado × Time"
                    ),
                    "Beta": round(g2, 4),
                    "p-valor": fe.get("p_gamma_rodada2"),
                }
            )
        if flags["usa_forma"]:
            termos.append(
                {
                    "Variável": "Forma Recente",
                    "Beta": round(float(comum["beta_forma"]), 4),
                    "p-valor": comum.get("p_forma"),
                }
            )
        if flags["usa_forca"]:
            termos.append(
                {
                    "Variável": "Força dos Adversários Passados",
                    "Beta": round(float(comum["beta_forca"]), 4),
                    "p-valor": comum.get("p_forca"),
                }
            )
        if flags["usa_prop_casa"]:
            termos.append(
                {
                    "Variável": "Proporção Casa",
                    "Beta": round(float(comum["beta_prop_casa"]), 4),
                    "p-valor": comum.get("p_prop_casa"),
                }
            )
        if flags.get("usa_descanso", True):
            termos.append(
                {
                    "Variável": "Dias de Descanso",
                    "Beta": round(float(comum.get("beta_descanso", 0.0)), 4),
                    "p-valor": comum.get("p_descanso"),
                }
            )
        if flags.get("usa_importante", True):
            for lab, key in (
                ("Classificatórias", "classificatorias"),
                ("Oitavas", "oitavas"),
                ("Quartas", "quartas"),
                ("Semi", "semi"),
                ("Final", "final"),
            ):
                termos.append(
                    {
                        "Variável": lab,
                        "Beta": round(float(comum.get(f"beta_{key}", 0.0)), 4),
                        "p-valor": comum.get(f"p_{key}"),
                    }
                )

        usa_r2_proj = bool(
            flags.get("usa_rodada2") or flags.get("usa_interacao_rodada2")
        )
        out[t] = {
            "intercept": float(fe["intercept"]),
            # inclinações efetivas (comum + interação)
            "beta_rodada": b1 + (g1 if flags.get("usa_interacao_rodada", True) else 0.0),
            "beta_rodada2": b2 + (g2 if flags.get("usa_interacao_rodada2", True) else 0.0),
            "beta_forma": float(comum["beta_forma"]),
            "beta_forca": float(comum["beta_forca"]),
            "beta_prop_casa": float(comum["beta_prop_casa"]),
            "beta_descanso": float(comum.get("beta_descanso", 0.0)),
            "beta_classificatorias": float(comum.get("beta_classificatorias", 0.0)),
            "beta_oitavas": float(comum.get("beta_oitavas", 0.0)),
            "beta_quartas": float(comum.get("beta_quartas", 0.0)),
            "beta_semi": float(comum.get("beta_semi", 0.0)),
            "beta_final": float(comum.get("beta_final", 0.0)),
            "gamma_rodada": g1 if flags.get("usa_interacao_rodada", True) else 0.0,
            "gamma_rodada2": g2 if flags.get("usa_interacao_rodada2", True) else 0.0,
            "usa_rodada": True,
            "usa_rodada2": usa_r2_proj,
            "usa_forma": flags["usa_forma"],
            "usa_forca": flags["usa_forca"],
            "usa_prop_casa": flags["usa_prop_casa"],
            "usa_descanso": bool(flags.get("usa_descanso", True)),
            "usa_importante": bool(flags.get("usa_importante", True)),
            "usa_rodada_centrada": bool(flags.get("usa_rodada_centrada")),
            "limita_delta_rodada": bool(flags.get("limita_delta_rodada")),
            "forma_decaindo": bool(flags.get("forma_decaindo")),
            "variante_acum": variante,
            "termos": termos,
            "n_obs": painel.get("n_obs", 0),
            "r2": painel.get("r2"),
            "time_ref": painel.get("time_ref"),
        }
    return out


def _prever_acumulado(
    b: dict,
    rodada: int,
    *,
    prop_casa: float = 1.0,
    forca_oponentes: float = 1.0,
    forma_recente: float = 1.0,
    dias_descanso: float = 7.0,
    importante: str = "Não tem",
) -> float:
    r = float(rodada)
    if b.get("usa_rodada_centrada"):
        r_lin = r - float(RODADA_CENTRO)
        r_quad = r_lin * r_lin
    else:
        r_lin = r
        r_quad = r * r
    val = b["intercept"]
    if b.get("usa_rodada"):
        val += b.get("beta_rodada", 0.0) * r_lin
    if b.get("usa_rodada2"):
        val += b.get("beta_rodada2", 0.0) * r_quad
    if b.get("usa_forma"):
        val += b.get("beta_forma", 0.0) * forma_recente
    if b.get("usa_forca"):
        val += b.get("beta_forca", 0.0) * forca_oponentes
    if b.get("usa_prop_casa"):
        val += b.get("beta_prop_casa", 0.0) * prop_casa
    if b.get("usa_descanso"):
        val += b.get("beta_descanso", 0.0) * float(dias_descanso)
    if b.get("usa_importante"):
        mapa = {
            "Classificatórias": "beta_classificatorias",
            "Oitavas": "beta_oitavas",
            "Quartas": "beta_quartas",
            "Semi": "beta_semi",
            "Final": "beta_final",
        }
        key = mapa.get(str(importante))
        if key:
            val += b.get(key, 0.0)
    return max(val, 0.0)


def _forma_misturada_projecao(
    jogos: list[Jogo],
    time: str,
    jogo: Jogo,
    *,
    r_ini: int,
    r_fim_obs: int,
    horizonte: int,
) -> float:
    """Mistura forma recente e forma geral com peso decrescente no horizonte."""
    fr = forma_recente_ate(
        jogos, time, jogo.r, jogo.hora, incluir_proj=True
    )
    fg = media_pts_jogo(jogos, time, r_ini, r_fim_obs, "simples")["geral"]
    if fg <= 0:
        fg = 1.0
    w = peso_forma_recente_horizonte(horizonte)
    return w * fr + (1.0 - w) * fg


def _contexto_projecao_acumulada(
    jogos: list[Jogo],
    time: str,
    jogo: Jogo,
    forca_map: dict[str, float],
) -> tuple[float, float, float, float, str]:
    em_casa = jogo.mand == time
    prop = _proporcao_casa_ate(
        jogos, time, jogo.r, em_casa_jogo=em_casa, incluir_proj=True
    )
    forca = _forca_oponentes_passados(
        jogos, time, jogo.r, forca_map, incluir_proj=True
    )
    forma = forma_recente_ate(
        jogos, time, jogo.r, jogo.hora, incluir_proj=True
    )
    try:
        from prob_ml.context_calendar import context_for_team

        descanso, importante = context_for_team(time, jogo.data)
    except Exception:
        descanso, importante = 7.0, "Não tem"
    return prop, forca, forma, float(descanso), str(importante)


def regressao_acumulada_beta(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressaoAcumulada,
    forca_map: dict[str, float] | None = None,
    *,
    blocos_extra: list[list[Jogo]] | None = None,
) -> dict:
    fm = forca_map or mapa_forca_adversario(jogos, r_ini, r_fim)
    if blocos_extra is None:
        try:
            from brasileirao_multi_liga import carregar_blocos_treino_regressao

            blocos_extra = carregar_blocos_treino_regressao()
        except Exception:
            blocos_extra = []
    painel = ajustar_painel_efeitos_fixos(
        jogos, r_ini, r_fim, variante, fm, blocos_extra=blocos_extra
    )
    por_time = coeficientes_efeitos_fixos_por_time(painel)
    if time in por_time:
        return por_time[time]
    return _coeficientes_vazios_acumulada(variante)


def _discretizar_pts_esperados(em: float, ev: float) -> tuple[int, int]:
    """Converte pts esperados mand/vis em placar discreto (3-0, 1-1, 0-3)."""
    total = em + ev
    if total <= 0:
        return 1, 1
    pm_f = 3.0 * em / total
    pv_f = 3.0 * ev / total
    pm, pv = round(pm_f), round(pv_f)
    if pm + pv > 3:
        scale = 3 / (pm + pv)
        pm, pv = int(round(pm * scale)), int(round(pv * scale))
    if pm > 3:
        pm = 3
    if pv > 3:
        pv = 3
    if pm > pv:
        return 3, 0
    if pv > pm:
        return 0, 3
    return 1, 1


def fator_forma_recente(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
) -> float:
    """Razão média últimos 5 jogos / média do campeonato no intervalo."""
    media_camp = media_pts_jogo(jogos, time, r_ini, r_fim, "simples")["geral"]
    if media_camp <= 0:
        return 1.0
    return forma_recente_atual(jogos, time) / media_camp


def projetar_jogo_media(
    jogo: Jogo,
    medias: dict[str, dict[str, float]],
    fatores: dict[str, float],
) -> tuple[float, float]:
    """Pts/jogo decimais: média casa/fora × fator forma recente."""
    mm = medias[jogo.mand]
    mv = medias[jogo.vis]
    pm = max(mm.get("casa", mm["geral"]) * fatores[jogo.mand], 0.0)
    pv = max(mv.get("fora", mv["geral"]) * fatores[jogo.vis], 0.0)
    return pm, pv


def media_pts_jogo(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
    tipo: TipoRegressao,
) -> dict[str, float]:
    """
    Média de pontos por jogo (≈ por rodada) no intervalo [r_ini, r_fim].
    mandante_visitante: médias separadas em casa e fora.
    """
    pts_total = pts_casa = pts_fora = 0
    n_total = n_casa = n_fora = 0

    for j in jogos:
        if not j.jogado or j.r < r_ini or j.r > r_fim:
            continue
        pm, pv = j.pts_reais()
        if j.mand == time:
            pts_total += pm
            pts_casa += pm
            n_total += 1
            n_casa += 1
        elif j.vis == time:
            pts_total += pv
            pts_fora += pv
            n_total += 1
            n_fora += 1

    media_geral = pts_total / n_total if n_total else 1.0
    if tipo == "simples":
        return {"geral": max(media_geral, 0.0)}

    media_casa = pts_casa / n_casa if n_casa else media_geral
    media_fora = pts_fora / n_fora if n_fora else media_geral
    return {
        "geral": max(media_geral, 0.0),
        "casa": max(media_casa, 0.0),
        "fora": max(media_fora, 0.0),
    }


def metricas_time(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
    modo: ModoProjecao,
    tipo: TipoRegressao,
    *,
    variante: VarianteRegressao = "interacao",
    forca_map: dict[str, float] | None = None,
) -> dict[str, float]:
    """Médias de pts/jogo conforme o modo."""
    return media_pts_jogo(jogos, time, r_ini, r_fim, tipo)


def expected_pts_jogo(
    time: str,
    mand: str,
    betas: dict[str, dict[str, float]],
    tipo: TipoRegressao,
    *,
    rodada: int | None = None,
    adversario: str | None = None,
    forca_map: dict[str, float] | None = None,
) -> float:
    b = betas.get(time, {"geral": 1.0})
    if tipo == "simples":
        return max(b.get("geral", 1.0), 0.0)

    if rodada is not None and "intercept" in b:
        adv = adversario or (mand if time != mand else "")
        return _prever_regressao(
            b,
            rodada=rodada,
            em_casa=(time == mand),
            adversario=adv,
            forca_map=forca_map or {},
        )

    if time == mand:
        return max(b.get("casa", b.get("geral", 1.0)), 0.0)
    return max(b.get("fora", b.get("geral", 1.0)), 0.0)


def projetar_jogo_regressao(
    jogo: Jogo,
    betas: dict[str, dict[str, float]],
    tipo: TipoRegressao,
    *,
    variante: VarianteRegressao = "interacao",
    forca_map: dict[str, float] | None = None,
) -> tuple[int, int]:
    fm = forca_map or {}
    em = expected_pts_jogo(
        jogo.mand, jogo.mand, betas, tipo,
        rodada=jogo.r, adversario=jogo.vis, forca_map=fm,
    )
    ev = expected_pts_jogo(
        jogo.vis, jogo.mand, betas, tipo,
        rodada=jogo.r, adversario=jogo.mand, forca_map=fm,
    )
    return _discretizar_pts_esperados(em, ev)


def mapa_contrapartidas(jogos: list[Jogo]) -> dict[tuple[frozenset[str], str, str], Jogo]:
    """
    Chave: (par, mand, vis) identifica confronto orientado.
    Valor: jogo correspondente no calendário.
    """
    m: dict[tuple[frozenset[str], str, str], Jogo] = {}
    for j in jogos:
        m[(j.par, j.mand, j.vis)] = j
    return m


def espelhar_contrapartida(jogo_alvo: Jogo, ref: Jogo) -> tuple[int, int]:
    """Espelha resultado da ida na volta (ou vice-versa)."""
    parsed = parse_placar(ref.placar)
    if parsed is None:
        raise ValueError("referência sem placar")
    gm_ref, gv_ref = parsed
    pm_ref, pv_ref = pontos_placar(gm_ref, gv_ref)

    # ref: A mand x B vis -> pontos A=pm_ref, B=pv_ref
    # alvo: se for B mand x A vis, pontos B=pv_ref, A=pm_ref
    if jogo_alvo.mand == ref.mand:
        return pm_ref, pv_ref
    return pv_ref, pm_ref


def media_ppg(jogos: list[Jogo], time: str) -> float:
    pts, n = 0, 0
    for j in jogos:
        if not j.jogado:
            continue
        pm, pv = j.pts_reais()
        if j.mand == time:
            pts += pm
            n += 1
        elif j.vis == time:
            pts += pv
            n += 1
    return pts / n if n else 1.0


def _acum_proj_ate(
    acum_cache: dict[str, dict[int, float]],
    time: str,
    rodada: int,
    jogos: list[Jogo],
    r_fim: int,
) -> float:
    if rodada <= 0:
        return 0.0
    if rodada in acum_cache.get(time, {}):
        return acum_cache[time][rodada]
    if rodada <= r_fim:
        return float(stats_acumuladas_ate(jogos, time, rodada, so_realizados=True).pts)
    return 0.0


def aplicar_projecoes_acumulada(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressaoAcumulada,
    *,
    blocos_extra: list[list[Jogo]] | None = None,
) -> tuple[list[Jogo], pd.DataFrame]:
    """Projeta rodada a rodada; força dos oponentes atualizada após cada rodada."""
    jogos = [Jogo(**j.__dict__) for j in jogos]
    times = times_do_calendario(jogos)
    forca_map = mapa_forca_adversario(jogos, r_ini, r_fim)
    if blocos_extra is None:
        try:
            from brasileirao_multi_liga import carregar_blocos_treino_regressao

            blocos_extra = carregar_blocos_treino_regressao()
        except Exception:
            blocos_extra = []
    painel = ajustar_painel_efeitos_fixos(
        jogos, r_ini, r_fim, variante, forca_map, blocos_extra=blocos_extra
    )
    betas = coeficientes_efeitos_fixos_por_time(painel)
    acum_cache: dict[str, dict[int, float]] = {t: {0: 0.0} for t in times}
    for t in times:
        for r in range(1, r_fim + 1):
            acum_cache[t][r] = float(
                stats_acumuladas_ate(jogos, t, r, so_realizados=True).pts
            )

    label = NOME_REGRESSAO_ACUMULADA[variante]
    log_rows: list[dict] = []
    rodadas_pendentes = sorted({j.r for j in jogos if not j.jogado})
    ult_r_real = max((j.r for j in jogos if j.jogado), default=r_fim)

    for r in rodadas_pendentes:
        jogos_r = [
            j for j in jogos if j.r == r and not j.jogado
        ]
        horizonte = max(1, r - ult_r_real)
        for j in sorted(jogos_r, key=lambda x: (x.hora, x.mand)):
            m, v = j.mand, j.vis
            prev_m = _acum_proj_ate(acum_cache, m, r - 1, jogos, r_fim)
            prev_v = _acum_proj_ate(acum_cache, v, r - 1, jogos, r_fim)

            prop_m, forca_m, forma_m, desc_m, imp_m = _contexto_projecao_acumulada(
                jogos, m, j, forca_map
            )
            prop_v, forca_v, forma_v, desc_v, imp_v = _contexto_projecao_acumulada(
                jogos, v, j, forca_map
            )
            if betas[m].get("forma_decaindo"):
                forma_m = _forma_misturada_projecao(
                    jogos, m, j, r_ini=r_ini, r_fim_obs=r_fim, horizonte=horizonte
                )
            if betas[v].get("forma_decaindo"):
                forma_v = _forma_misturada_projecao(
                    jogos, v, j, r_ini=r_ini, r_fim_obs=r_fim, horizonte=horizonte
                )
            target_m = _prever_acumulado(
                betas[m],
                r,
                prop_casa=prop_m,
                forca_oponentes=forca_m,
                forma_recente=forma_m,
                dias_descanso=desc_m,
                importante=imp_m,
            )
            target_v = _prever_acumulado(
                betas[v],
                r,
                prop_casa=prop_v,
                forca_oponentes=forca_v,
                forma_recente=forma_v,
                dias_descanso=desc_v,
                importante=imp_v,
            )

            delta_m = max(0.0, target_m - prev_m)
            delta_v = max(0.0, target_v - prev_v)
            if betas[m].get("limita_delta_rodada"):
                delta_m = min(DELTA_PTS_MAX_POR_RODADA, delta_m)
            if betas[v].get("limita_delta_rodada"):
                delta_v = min(DELTA_PTS_MAX_POR_RODADA, delta_v)
            j.proj_pm, j.proj_pv = delta_m, delta_v
            j.origem = label
            acum_cache[m][r] = prev_m + delta_m
            acum_cache[v][r] = prev_v + delta_v

            log_rows.append(
                {
                    "Rodada": j.r,
                    "Mandante": j.mand,
                    "Visitante": j.vis,
                    "Proj": f"{delta_m:.2f} / {delta_v:.2f}",
                    "Descanso M/V": f"{desc_m:.0f} / {desc_v:.0f}",
                    "Importante M/V": f"{imp_m} / {imp_v}",
                }
            )

        for t in times:
            forca_map[t] = _forca_pts_jogo_time(jogos, t, r)

    return jogos, pd.DataFrame(log_rows)


def aplicar_projecoes_media(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    *,
    usar_forma: bool = True,
    forma_decaindo: bool = True,
) -> tuple[list[Jogo], pd.DataFrame]:
    """Média casa/fora; opcionalmente × fator forma recente (pts decimais)."""
    jogos = [Jogo(**j.__dict__) for j in jogos]
    times = times_do_calendario(jogos)
    medias = {
        t: media_pts_jogo(jogos, t, r_ini, r_fim, "mandante_visitante") for t in times
    }
    fatores = (
        {t: fator_forma_recente(jogos, t, r_ini, r_fim) for t in times}
        if usar_forma
        else {t: 1.0 for t in times}
    )
    origem = (
        "média casa/fora × forma recente"
        if usar_forma
        else "média casa/fora"
    )
    if usar_forma and forma_decaindo:
        origem += " (peso decaindo)"
    log_rows: list[dict] = []
    ult_r_real = max((j.r for j in jogos if j.jogado), default=r_fim)

    for j in sorted(jogos, key=lambda x: (x.r, x.hora, x.mand)):
        if j.jogado:
            continue
        fat_m = fatores[j.mand]
        fat_v = fatores[j.vis]
        if usar_forma and forma_decaindo:
            w = peso_forma_recente_horizonte(max(1, j.r - ult_r_real))
            fat_m = w * fat_m + (1.0 - w)
            fat_v = w * fat_v + (1.0 - w)
        mm = medias[j.mand]
        mv = medias[j.vis]
        pm = max(mm.get("casa", mm["geral"]) * fat_m, 0.0)
        pv = max(mv.get("fora", mv["geral"]) * fat_v, 0.0)
        j.proj_pm, j.proj_pv = pm, pv
        j.origem = origem
        log_rows.append(
            {
                "Rodada": j.r,
                "Mandante": j.mand,
                "Visitante": j.vis,
                "Proj": f"{pm:.2f} / {pv:.2f}",
            }
        )

    return jogos, pd.DataFrame(log_rows)


def aplicar_projecoes(
    jogos: list[Jogo],
    modo: ModoProjecao,
    r_ini: int,
    r_fim: int,
    tipo_reg: TipoRegressao,
    *,
    variante_reg: VarianteRegressao = "interacao",
) -> tuple[list[Jogo], pd.DataFrame]:
    if modo_e_regressao_acumulada(modo):
        return aplicar_projecoes_acumulada(
            jogos, r_ini, r_fim, MODO_PARA_VARIANTE[modo]
        )
    if modo == "media_simples":
        return aplicar_projecoes_media(jogos, r_ini, r_fim, usar_forma=True)
    if modo == "media_casa_fora":
        return aplicar_projecoes_media(
            jogos, r_ini, r_fim, usar_forma=False, forma_decaindo=False
        )

    jogos = [Jogo(**j.__dict__) for j in jogos]
    mapa = mapa_contrapartidas(jogos)
    medias = {
        t: media_pts_jogo(jogos, t, r_ini, r_fim, "mandante_visitante")
        for t in times_do_calendario(jogos)
    }
    fatores = {
        t: fator_forma_recente(jogos, t, r_ini, r_fim)
        for t in times_do_calendario(jogos)
    }
    label_fallback = "sem espelho - média casa/fora × forma recente (peso decaindo)"
    log_rows: list[dict] = []
    ult_r_real = max((j.r for j in jogos if j.jogado), default=r_fim)

    for j in sorted(jogos, key=lambda x: (x.r, x.hora, x.mand)):
        if j.jogado:
            continue

        chave = (j.par, j.vis, j.mand)
        ref = mapa.get(chave)
        if ref and ref.jogado:
            pm, pv = espelhar_contrapartida(j, ref)
            j.proj_pm, j.proj_pv = float(pm), float(pv)
            j.origem = f"espelho R{ref.r} ({ref.mand} {ref.placar} {ref.vis})"
            proj_txt = f"{pm} / {pv}"
        else:
            w = peso_forma_recente_horizonte(max(1, j.r - ult_r_real))
            fat_m = w * fatores[j.mand] + (1.0 - w)
            fat_v = w * fatores[j.vis] + (1.0 - w)
            mm = medias[j.mand]
            mv = medias[j.vis]
            pm = max(mm.get("casa", mm["geral"]) * fat_m, 0.0)
            pv = max(mv.get("fora", mv["geral"]) * fat_v, 0.0)
            j.proj_pm, j.proj_pv = pm, pv
            j.origem = label_fallback
            proj_txt = f"{pm:.2f} / {pv:.2f}"

        log_rows.append(
            {
                "Rodada": j.r,
                "Mandante": j.mand,
                "Visitante": j.vis,
                "Proj": proj_txt,
            }
        )

    return jogos, pd.DataFrame(log_rows)


def classificacao(jogos: list[Jogo], incluir_proj: bool = True) -> pd.DataFrame:
    jog_real, jog_proj = 0, 0
    for j in jogos:
        if j.jogado:
            jog_real += 1
        elif incluir_proj and j.proj_pm is not None:
            jog_proj += 1

    linhas = ordenar_times_desempate(jogos, incluir_proj=incluir_proj)
    df = pd.DataFrame(
        {
            "Time": [t for t, _ in linhas],
            "Pontos": [round(s.pts, 1) for _, s in linhas],
        }
    )
    df.insert(0, "Pos", range(1, len(df) + 1))
    df.attrs["jog_real"] = jog_real
    df.attrs["jog_proj"] = jog_proj
    return df


def tabela_betas(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    tipo: TipoRegressao,
    modo: ModoProjecao = "media_simples",
) -> pd.DataFrame:
    rows = []
    col_main = "Media_pts/jogo"
    col_casa = "Media_casa"
    col_fora = "Media_fora"

    for t in times_do_calendario(jogos):
        b = metricas_time(jogos, t, r_ini, r_fim, modo, tipo)
        row = {"Time": t, col_main: round(b["geral"], 3)}
        if tipo == "mandante_visitante":
            row[col_casa] = round(b.get("casa", b["geral"]), 3)
            row[col_fora] = round(b.get("fora", b["geral"]), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(col_main, ascending=False)


def tabela_regressao_resumo(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressao,
) -> pd.DataFrame:
    """R² e significância (estrelas) de cada termo, por time."""
    forca_map = mapa_forca_adversario(jogos, r_ini, r_fim)
    ordem = _ordem_variaveis_regressao(variante)
    rows: list[dict] = []
    for t in times_do_calendario(jogos):
        b = regressao_beta(jogos, t, r_ini, r_fim, variante, forca_map)
        row: dict = {"Time": t, "R²": b.get("r2")}
        sig = {termo["Variável"]: pvalor_estrela(termo.get("p-valor")) for termo in b.get("termos", [])}
        for var in ordem:
            row[var] = sig.get(var, "-")
        rows.append(row)
    cols = ["Time", "R²", *ordem]
    return pd.DataFrame(rows, columns=cols)


def tabela_regressao_acumulada_resumo(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressaoAcumulada,
    *,
    blocos_extra: list[list[Jogo]] | None = None,
) -> pd.DataFrame:
    """R² e significância dos termos da regressão FE de pts acumulados."""
    forca_map = mapa_forca_adversario(jogos, r_ini, r_fim)
    ordem = _ordem_variaveis_regressao_acumulada(variante)
    if blocos_extra is None:
        try:
            from brasileirao_multi_liga import carregar_blocos_treino_regressao

            blocos_extra = carregar_blocos_treino_regressao()
        except Exception:
            blocos_extra = []
    painel = ajustar_painel_efeitos_fixos(
        jogos, r_ini, r_fim, variante, forca_map, blocos_extra=blocos_extra
    )
    por_time = coeficientes_efeitos_fixos_por_time(painel)
    rows: list[dict] = []
    for t in times_do_calendario(jogos):
        b = por_time.get(t) or _coeficientes_vazios_acumulada(variante)
        row: dict = {"Time": t, "R²": b.get("r2")}
        sig = {
            termo["Variável"]: pvalor_estrela(termo.get("p-valor"))
            for termo in b.get("termos", [])
        }
        for var in ordem:
            row[var] = sig.get(var, "-")
        rows.append(row)
    return pd.DataFrame(rows, columns=["Time", "R²", *ordem])


def tabela_medias_simples_times(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    *,
    usar_forma: bool = True,
) -> pd.DataFrame:
    """Médias de pts/jogo; opcionalmente com fator forma e projeção ajustada."""
    rows: list[dict] = []
    for t in times_do_calendario(jogos):
        m = media_pts_jogo(jogos, t, r_ini, r_fim, "mandante_visitante")
        media_casa = m.get("casa", m["geral"])
        media_fora = m.get("fora", m["geral"])
        row: dict = {
            "Time": t,
            "Média pts/jogo (geral)": round(m["geral"], 3),
            "Média pts/jogo (casa)": round(media_casa, 3),
            "Média pts/jogo (fora)": round(media_fora, 3),
        }
        if usar_forma:
            fator = fator_forma_recente(jogos, t, r_ini, r_fim)
            row["Fator forma (últ. 5 / camp.)"] = round(fator, 3)
            row["Proj. pts/jogo (casa)"] = round(media_casa * fator, 3)
            row["Proj. pts/jogo (fora)"] = round(media_fora * fator, 3)
        else:
            row["Proj. pts/jogo (casa)"] = round(media_casa, 3)
            row["Proj. pts/jogo (fora)"] = round(media_fora, 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Time")


def tabela_jogos_primeiro_turno(jogos: list[Jogo]) -> pd.DataFrame:
    """Todos os jogos do 1º turno (rodadas 1–19)."""
    rows: list[dict] = []
    for j in sorted(jogos, key=lambda x: (x.r, x.hora, x.mand)):
        if j.r > RODADA_FIM_PRIMEIRO_TURNO:
            continue
        rows.append(
            {
                "Rodada": j.r,
                "Data": j.data,
                "Hora": j.hora,
                "Mandante": j.mand,
                "Placar": j.placar if j.jogado else "-",
                "Visitante": j.vis,
            }
        )
    return pd.DataFrame(rows)


def tabela_coeficientes_regressao(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    variante: VarianteRegressao,
) -> pd.DataFrame:
    """Beta e p-valor de cada termo da regressão, por time."""
    forca_map = mapa_forca_adversario(jogos, r_ini, r_fim)
    rows: list[dict] = []
    for t in times_do_calendario(jogos):
        b = regressao_beta(jogos, t, r_ini, r_fim, variante, forca_map)
        n_obs = int(b.get("n_obs", 0))
        for termo in b.get("termos", []):
            rows.append(
                {
                    "Time": t,
                    "Variável": termo["Variável"],
                    "Beta": termo["Beta"],
                    "p-valor": termo["p-valor"],
                    "N obs.": n_obs,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["Time", "Variável", "Beta", "p-valor", "N obs."]
        )
    return pd.DataFrame(rows)


def kpis_globais(jogos: list[Jogo]) -> dict[str, float]:
    """Médias de gols e pontos (jogos realizados)."""
    n_jog = gols_partida = gols_mand = gols_vis = 0
    pts_mand = pts_vis = 0.0

    for j in jogos:
        if not j.jogado:
            continue
        p = parse_placar(j.placar)
        if p is None:
            continue
        gm, gv = p
        pm, pv = pontos_placar(gm, gv)
        n_jog += 1
        gols_partida += gm + gv
        gols_mand += gm
        gols_vis += gv
        pts_mand += pm
        pts_vis += pv

    if n_jog == 0:
        return {
            "media_gols_jogo": 0.0,
            "media_gols_mandante": 0.0,
            "media_gols_visitante": 0.0,
            "media_pts_mandante": 0.0,
            "media_pts_visitante": 0.0,
        }
    return {
        "media_gols_jogo": gols_partida / n_jog,
        "media_gols_mandante": gols_mand / n_jog,
        "media_gols_visitante": gols_vis / n_jog,
        "media_pts_mandante": pts_mand / n_jog,
        "media_pts_visitante": pts_vis / n_jog,
    }


COLUNAS_ESTATISTICAS_GRAFICO = [
    "Total gols marcados",
    "Total gols sofridos",
    "Média gols marcados",
    "Média gols sofridos",
    "Média gols marcados/Média gols sofridos",
    "Média gols marcados casa",
    "Média gols sofridos casa",
    "Média gols marcados casa/Média gols sofridos casa",
    "Média gols marcados fora",
    "Média gols sofridos fora",
    "Média gols marcados fora/Média gols sofridos fora",
    "Total pontos",
    "Total pontos mandante",
    "Total pontos visitante",
    "Média pontos mandante",
    "Média pontos visitante",
    "Média dias de descanso",
    "% jogos c/ importante à frente",
    "Média xG marcados",
    "Média xG sofridos",
    "Média xG marcados/Média xG sofridos",
]


def _ratio_metrica(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return round(num / den, 3)


def _agregar_stats_time(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
) -> dict[str, int]:
    """Contagens de jogos realizados e gols/pontos no intervalo [r_ini, r_fim]."""
    jr = jp = 0
    gf = gc = 0
    gf_c = gc_c = gf_f = gc_f = 0
    n_c = n_f = 0
    pts = pts_c = pts_f = 0

    for j in jogos:
        if j.r < r_ini or j.r > r_fim or time not in (j.mand, j.vis):
            continue
        if j.jogado:
            jr += 1
            stt = _stats_jogo_para_time(j, time)
            if not stt:
                continue
            gf += stt.gf
            gc += stt.gc
            pts += stt.pts
            if j.mand == time:
                gf_c += stt.gf
                gc_c += stt.gc
                pts_c += stt.pts
                n_c += 1
            else:
                gf_f += stt.gf
                gc_f += stt.gc
                pts_f += stt.pts
                n_f += 1
        else:
            jp += 1

    return {
        "jr": jr,
        "jp": jp,
        "gf": gf,
        "gc": gc,
        "gf_c": gf_c,
        "gc_c": gc_c,
        "gf_f": gf_f,
        "gc_f": gc_f,
        "n_c": n_c,
        "n_f": n_f,
        "pts": pts,
        "pts_c": pts_c,
        "pts_f": pts_f,
    }


def _metricas_estatisticas_de_agregado(agg: dict[str, int]) -> dict[str, float | int]:
    jr = agg["jr"]
    n_c = agg["n_c"]
    n_f = agg["n_f"]
    mg = agg["gf"] / jr if jr else 0.0
    ms = agg["gc"] / jr if jr else 0.0
    mg_c = agg["gf_c"] / n_c if n_c else 0.0
    ms_c = agg["gc_c"] / n_c if n_c else 0.0
    mg_f = agg["gf_f"] / n_f if n_f else 0.0
    ms_f = agg["gc_f"] / n_f if n_f else 0.0
    mp_c = agg["pts_c"] / n_c if n_c else 0.0
    mp_f = agg["pts_f"] / n_f if n_f else 0.0

    return {
        "Total gols marcados": agg["gf"],
        "Total gols sofridos": agg["gc"],
        "Média gols marcados": round(mg, 3),
        "Média gols sofridos": round(ms, 3),
        "Média gols marcados/Média gols sofridos": _ratio_metrica(mg, ms),
        "Média gols marcados casa": round(mg_c, 3),
        "Média gols sofridos casa": round(ms_c, 3),
        "Média gols marcados casa/Média gols sofridos casa": _ratio_metrica(mg_c, ms_c),
        "Média gols marcados fora": round(mg_f, 3),
        "Média gols sofridos fora": round(ms_f, 3),
        "Média gols marcados fora/Média gols sofridos fora": _ratio_metrica(mg_f, ms_f),
        "Total pontos": agg["pts"],
        "Total pontos mandante": agg["pts_c"],
        "Total pontos visitante": agg["pts_f"],
        "Média pontos mandante": round(mp_c, 3),
        "Média pontos visitante": round(mp_f, 3),
    }


def tabela_estatisticas_times(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
) -> pd.DataFrame:
    rows = []
    for time in times_do_calendario(jogos):
        agg = _agregar_stats_time(jogos, time, r_ini, r_fim)
        row = {
            "Time": time,
            "Jogos realizados": agg["jr"],
            "Jogos pendentes": agg["jp"],
        }
        row.update(_metricas_estatisticas_de_agregado(agg))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Total pontos", ascending=False)


def tabela_comparativa_posicoes(
    jogos_base: list[Jogo],
    jogos_proj: list[Jogo],
) -> pd.DataFrame:
    atual = classificacao(jogos_base, incluir_proj=False).rename(
        columns={"Pos": "Posição Atual", "Pontos": "Pts Atual"}
    )
    proj = classificacao(jogos_proj, incluir_proj=True).rename(
        columns={"Pos": "Posição Projetada", "Pontos": "Pts Projetados"}
    )
    df = atual.merge(proj[["Time", "Posição Projetada", "Pts Projetados"]], on="Time")
    df["Delta"] = df["Posição Atual"] - df["Posição Projetada"]
    probs = probabilidades_cenarios_finais(jogos_proj)
    df["Prob. Campeão"] = df["Time"].map(
        lambda t: round(100.0 * float(probs.get(t, {}).get("campeao", 0.0)), 1)
    )
    df["Prob. G4"] = df["Time"].map(
        lambda t: round(100.0 * float(probs.get(t, {}).get("g4", 0.0)), 1)
    )
    df["Prob. G6"] = df["Time"].map(
        lambda t: round(100.0 * float(probs.get(t, {}).get("g6", 0.0)), 1)
    )
    df["Prob. Z4"] = df["Time"].map(
        lambda t: round(100.0 * float(probs.get(t, {}).get("z4", 0.0)), 1)
    )
    df = df.sort_values("Posição Projetada")
    return df[
        [
            "Posição Projetada",
            "Time",
            "Posição Atual",
            "Delta",
            "Pts Projetados",
            "Prob. Campeão",
            "Prob. G4",
            "Prob. G6",
            "Prob. Z4",
        ]
    ]


def mapa_posicao_pontos(
    jogos: list[Jogo], *, incluir_proj: bool
) -> dict[str, tuple[int, float]]:
    df = classificacao(jogos, incluir_proj=incluir_proj)
    return {row.Time: (int(row.Pos), float(row.Pontos)) for row in df.itertuples()}


def _probs_vitoria_empate_derrota(em: float, ev: float) -> tuple[float, float, float]:
    """
    Converte pts esperados mand/vis em P(vitória mandante, empate, vitória visitante)
    preservando a razão de força e garantindo E[pts] coerente (soma ≤ 3).
    """
    em = max(0.0, float(em))
    ev = max(0.0, float(ev))
    s = em + ev
    if s <= 1e-9:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    if s > 3.0:
        em, ev = 3.0 * em / s, 3.0 * ev / s
        s = 3.0
    pd = max(0.0, 3.0 - s)
    ph = max(0.0, (em - pd) / 3.0)
    pa = max(0.0, (ev - pd) / 3.0)
    z = ph + pd + pa
    if z <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return ph / z, pd / z, pa / z


def probabilidades_cenarios_finais(
    jogos: list[Jogo],
    *,
    n_sims: int = 5000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Monte Carlo jogo a jogo, alinhado à ordem da projeção, com incerteza realista.

    1) Sorteia W/D/L de cada jogo pendente com probs calibradas aos pts esperados.
    2) Recentra na projeção (com gaps comprimidos) para o líder projetado seguir
       como favorito, sem odds extremas tipo 95%×4% no meio do campeonato.
    3) Soma ruído extra ∝ √(jogos restantes) - na prática, ~14 jogos ainda
       abrem bastante o leque de cenários.

    Retorna, por time: campeao, g4, g6, z4 (frações 0–1).
    G4 = 1º–4º | G6 = 1º–6º | Z4 = 17º–20º.
    """
    times = times_do_calendario(jogos)
    n_times = len(times)
    vazio = {
        t: {"campeao": 0.0, "g4": 0.0, "g6": 0.0, "z4": 0.0} for t in times
    }
    if n_times == 0:
        return vazio

    idx = {t: i for i, t in enumerate(times)}
    mapa_proj = mapa_posicao_pontos(jogos, incluir_proj=True)
    pts_proj = np.array(
        [float(mapa_proj.get(t, (0, 0.0))[1]) for t in times], dtype=float
    )

    # Comprime gaps só no cálculo de probabilidade (preserva a ordem da tabela).
    # Evita campeão “matemático” quando o modelo FE estica demais a pontuação.
    pts_shrink = 0.62
    pts_centro = float(pts_proj.mean())
    pts_target = pts_centro + pts_shrink * (pts_proj - pts_centro)

    base = [
        stats_acumuladas_ate(jogos, t, 38, so_realizados=True) for t in times
    ]
    pts0 = np.array([float(s.pts) for s in base], dtype=float)
    vit0 = np.array([float(s.vit) for s in base], dtype=float)
    sg0 = np.array([float(s.sg) for s in base], dtype=float)
    gf0 = np.array([float(s.gf) for s in base], dtype=float)

    n_rest = np.zeros(n_times, dtype=float)
    pendentes: list[tuple[int, int, float, float, float]] = []
    for j in jogos:
        if j.jogado or j.proj_pm is None or j.proj_pv is None:
            continue
        if j.mand not in idx or j.vis not in idx:
            continue
        n_rest[idx[j.mand]] += 1.0
        n_rest[idx[j.vis]] += 1.0
        pendentes.append(
            (
                idx[j.mand],
                idx[j.vis],
                *_probs_vitoria_empate_derrota(float(j.proj_pm), float(j.proj_pv)),
            )
        )

    camp = np.zeros(n_times, dtype=float)
    g4 = np.zeros(n_times, dtype=float)
    g6 = np.zeros(n_times, dtype=float)
    z4 = np.zeros(n_times, dtype=float)

    if not pendentes:
        for t, (pos, _) in mapa_proj.items():
            i = idx[t]
            if pos == 1:
                camp[i] = 1.0
            if pos <= 4:
                g4[i] = 1.0
            if pos <= 6:
                g6[i] = 1.0
            if pos >= n_times - 3:
                z4[i] = 1.0
        return {
            t: {
                "campeao": float(camp[i]),
                "g4": float(g4[i]),
                "g6": float(g6[i]),
                "z4": float(z4[i]),
            }
            for t, i in idx.items()
        }

    n_g = len(pendentes)
    p_mat = np.array(
        [[ph, pd_, pa] for _, _, ph, pd_, pa in pendentes], dtype=float
    )
    im = np.array([p[0] for p in pendentes], dtype=int)
    iv = np.array([p[1] for p in pendentes], dtype=int)

    rng = np.random.default_rng(seed)
    u = rng.random((n_sims, n_g))
    cdf = np.cumsum(p_mat, axis=1)
    outcomes = (u[..., None] > cdf[None, :, :]).sum(axis=2).astype(np.int8)

    pts = np.tile(pts0, (n_sims, 1))
    vit = np.tile(vit0, (n_sims, 1))
    sg = np.tile(sg0, (n_sims, 1))
    gf = np.tile(gf0, (n_sims, 1))

    for g in range(n_g):
        a = im[g]
        b = iv[g]
        o = outcomes[:, g]
        mh = o == 0
        md = o == 1
        ma = o == 2
        pts[mh, a] += 3.0
        vit[mh, a] += 1.0
        sg[mh, a] += 1.0
        gf[mh, a] += 1.0
        sg[mh, b] -= 1.0

        pts[md, a] += 1.0
        pts[md, b] += 1.0
        gf[md, a] += 1.0
        gf[md, b] += 1.0

        pts[ma, b] += 3.0
        vit[ma, b] += 1.0
        sg[ma, b] += 1.0
        gf[ma, b] += 1.0
        sg[ma, a] -= 1.0

    # Recentra no alvo suavizado (mesma ordem da projeção, gaps menores)
    media = pts.mean(axis=0)
    pts = pts - media + pts_target

    # Incerteza residual dos jogos que faltam (ainda ~14 rodadas)
    sigma_jogo_extra = 2.35
    sigma_extra = sigma_jogo_extra * np.sqrt(np.maximum(n_rest, 1.0))
    pts = pts + rng.normal(0.0, 1.0, size=pts.shape) * sigma_extra[None, :]
    pts = np.maximum(pts, 0.0)

    for s in range(n_sims):
        ordem = np.lexsort((-gf[s], -sg[s], -vit[s], -pts[s]))
        pos = np.empty(n_times, dtype=int)
        pos[ordem] = np.arange(1, n_times + 1)
        camp[pos == 1] += 1.0
        g4[pos <= 4] += 1.0
        g6[pos <= 6] += 1.0
        z4[pos >= n_times - 3] += 1.0

    inv = float(n_sims)
    return {
        t: {
            "campeao": float(camp[i] / inv),
            "g4": float(g4[i] / inv),
            "g6": float(g6[i] / inv),
            "z4": float(z4[i] / inv),
        }
        for t, i in idx.items()
    }


def mapa_vitorias_saldo_proj(
    jogos: list[Jogo],
) -> dict[str, tuple[int, int]]:
    """Vitórias e saldo de gols totais (realizados + projetados) ao fim do campeonato."""
    out: dict[str, tuple[int, int]] = {}
    for time in times_do_calendario(jogos):
        st = stats_acumuladas_ate(jogos, time, 38, so_realizados=False)
        out[time] = (st.vit, st.sg)
    return out


@dataclass
class StatsTime:
    pts: float = 0.0
    vit: int = 0
    emp: int = 0
    der: int = 0
    gf: int = 0
    gc: int = 0

    @property
    def sg(self) -> int:
        return self.gf - self.gc

    def chave_classificacao(self) -> tuple:
        """Critérios gerais: pts, vitórias, saldo, gols marcados (sem confronto)."""
        return (-self.pts, -self.vit, -self.sg, -self.gf)

    def copy(self) -> "StatsTime":
        return StatsTime(self.pts, self.vit, self.emp, self.der, self.gf, self.gc)

    def add(self, pts: float, gf: int, gc: int) -> None:
        self.pts += pts
        self.gf += gf
        self.gc += gc
        if pts == 3:
            self.vit += 1
        elif pts == 1:
            self.emp += 1
        elif pts == 0:
            self.der += 1


def _stats_de_placar(gm: int, gv: int, time: str, mand: str) -> StatsTime:
    pm, pv = pontos_placar(gm, gv)
    s = StatsTime()
    if time == mand:
        s.add(pm, gm, gv)
    else:
        s.add(pv, gv, gm)
    return s


def _stats_extremos_jogo(time: str, jogo: Jogo) -> tuple[StatsTime, StatsTime]:
    """Melhor e pior cenário (3-0 / 0-3) para o time no jogo pendente."""
    if time == jogo.mand:
        return (
            _stats_de_placar(3, 0, time, jogo.mand),
            _stats_de_placar(0, 3, time, jogo.mand),
        )
    return (
        _stats_de_placar(0, 3, time, jogo.mand),
        _stats_de_placar(3, 0, time, jogo.mand),
    )


def _stats_jogo_para_time(jogo: Jogo, time: str) -> StatsTime | None:
    if time not in (jogo.mand, jogo.vis):
        return None
    if jogo.jogado:
        p = parse_placar(jogo.placar)
        if p is None:
            return None
        return _stats_de_placar(p[0], p[1], time, jogo.mand)
    if jogo.proj_gm is not None and jogo.proj_gv is not None:
        return _stats_de_placar(jogo.proj_gm, jogo.proj_gv, time, jogo.mand)
    if jogo.proj_pm is not None and jogo.proj_pv is not None:
        pts = float(jogo.proj_pm if time == jogo.mand else jogo.proj_pv)
        s = StatsTime()
        s.pts = pts
        return s
    return None


def _stats_rodada_time(jogos: list[Jogo], time: str, rodada: int) -> StatsTime:
    s = StatsTime()
    for j in jogos:
        if j.r != rodada:
            continue
        if time not in (j.mand, j.vis):
            continue
        st = _stats_jogo_para_time(j, time)
        if st:
            s.add(st.pts, st.gf, st.gc)
    return s


def stats_acumuladas_ate(
    jogos: list[Jogo],
    time: str,
    rodada: int,
    *,
    so_realizados: bool = True,
) -> StatsTime:
    total = StatsTime()
    for r in range(1, rodada + 1):
        for j in jogos:
            if j.r != r or time not in (j.mand, j.vis):
                continue
            if so_realizados:
                if not j.jogado:
                    continue
                st = _stats_jogo_para_time(j, time)
            else:
                if j.jogado:
                    st = _stats_jogo_para_time(j, time)
                elif j.proj_pm is not None:
                    st = _stats_jogo_para_time(j, time)
                else:
                    st = None
            if st:
                total.add(st.pts, st.gf, st.gc)
    return total


def _chave_criterios_basicos(s: StatsTime) -> tuple[float, int, int, int]:
    return (s.pts, s.vit, s.sg, s.gf)


def stats_mapa_times(
    jogos: list[Jogo],
    *,
    incluir_proj: bool,
    ate_rodada: int = 38,
) -> dict[str, StatsTime]:
    so_realizados = not incluir_proj
    return {
        t: stats_acumuladas_ate(
            jogos, t, ate_rodada, so_realizados=so_realizados
        )
        for t in times_do_calendario(jogos)
    }


def stats_confronto_grupo(
    jogos: list[Jogo],
    time: str,
    grupo: frozenset[str],
    *,
    incluir_proj: bool,
    ate_rodada: int = 38,
) -> StatsTime:
    """Estatísticas só nos jogos entre times do grupo empatado."""
    so_realizados = not incluir_proj
    total = StatsTime()
    for j in jogos:
        if j.r > ate_rodada:
            continue
        if j.mand not in grupo or j.vis not in grupo:
            continue
        if time not in (j.mand, j.vis):
            continue
        if so_realizados:
            if not j.jogado:
                continue
            st = _stats_jogo_para_time(j, time)
        elif j.jogado:
            st = _stats_jogo_para_time(j, time)
        elif j.proj_pm is not None:
            st = _stats_jogo_para_time(j, time)
        else:
            st = None
        if st:
            total.add(st.pts, st.gf, st.gc)
    return total


def ordenar_stats_desempate(
    jogos: list[Jogo],
    stats: dict[str, StatsTime],
    *,
    incluir_proj: bool,
    ate_rodada: int = 38,
) -> list[tuple[str, StatsTime]]:
    """
    Desempate: pontos → vitórias → saldo → gols marcados → confronto direto.
    """
    times = list(stats.keys())
    grupos: dict[tuple[int, int, int, int], list[str]] = {}
    for t in times:
        grupos.setdefault(_chave_criterios_basicos(stats[t]), []).append(t)

    ordenado: list[tuple[str, StatsTime]] = []
    for chave in sorted(
        grupos.keys(), key=lambda k: (-k[0], -k[1], -k[2], -k[3])
    ):
        grupo = grupos[chave]
        if len(grupo) == 1:
            t = grupo[0]
            ordenado.append((t, stats[t]))
            continue
        gset = frozenset(grupo)
        confronto = {
            t: stats_confronto_grupo(
                jogos, t, gset, incluir_proj=incluir_proj, ate_rodada=ate_rodada
            )
            for t in grupo
        }
        sub = sorted(
            grupo,
            key=lambda t: (
                -confronto[t].pts,
                -confronto[t].vit,
                -confronto[t].sg,
                -confronto[t].gf,
                t,
            ),
        )
        for t in sub:
            ordenado.append((t, stats[t]))
    return ordenado


def ordenar_times_desempate(
    jogos: list[Jogo],
    *,
    incluir_proj: bool,
    ate_rodada: int = 38,
) -> list[tuple[str, StatsTime]]:
    stats = stats_mapa_times(
        jogos, incluir_proj=incluir_proj, ate_rodada=ate_rodada
    )
    return ordenar_stats_desempate(
        jogos, stats, incluir_proj=incluir_proj, ate_rodada=ate_rodada
    )


def posicao_time_na_rodada(
    jogos: list[Jogo],
    time: str,
    rodada: int,
    extra: dict[str, StatsTime] | None = None,
) -> int:
    """Posição na tabela ao fim da rodada (só jogos realizados + extras opcionais)."""
    times = times_do_calendario(jogos)
    extra = extra or {}
    stats: dict[str, StatsTime] = {}
    for t in times:
        s = stats_acumuladas_ate(jogos, t, rodada, so_realizados=True)
        if t in extra:
            e = extra[t]
            s.add(e.pts, e.gf, e.gc)
        stats[t] = s

    linhas = ordenar_stats_desempate(
        jogos, stats, incluir_proj=False, ate_rodada=rodada
    )
    for i, (t, _) in enumerate(linhas, 1):
        if t == time:
            return i
    return len(times)


def jogo_faltante_pode_afetar_posicao(
    jogos: list[Jogo],
    time: str,
    jogo: Jogo,
) -> bool:
    """
    True se o pior e o melhor desfecho do jogo pendente alteram posição/critério
    do time ao fim da rodada do jogo (pts, vitórias, saldo, gols pró).
    """
    if jogo.jogado or time not in (jogo.mand, jogo.vis):
        return False

    melhor, pior = _stats_extremos_jogo(time, jogo)
    pos_melhor = posicao_time_na_rodada(jogos, time, jogo.r, extra={time: melhor})
    pos_pior = posicao_time_na_rodada(jogos, time, jogo.r, extra={time: pior})

    if pos_melhor != pos_pior:
        return True

    def _posicao_com_extra(cen: StatsTime) -> tuple[int, tuple]:
        st_extra = {time: cen}
        stats_c: dict[str, StatsTime] = {}
        for t in times_do_calendario(jogos):
            s = stats_acumuladas_ate(jogos, t, jogo.r, so_realizados=True)
            if t in st_extra:
                s.add(st_extra[t].pts, st_extra[t].gf, st_extra[t].gc)
            stats_c[t] = s
        linhas = ordenar_stats_desempate(
            jogos, stats_c, incluir_proj=False, ate_rodada=jogo.r
        )
        pos = next(i for i, (t, _) in enumerate(linhas, 1) if t == time)
        chave = next(s for t, s in linhas if t == time)
        return pos, _chave_criterios_basicos(chave)

    return _posicao_com_extra(melhor) != _posicao_com_extra(pior)


def ultima_rodada_com_resultado(jogos: list[Jogo]) -> int:
    jogados = [j.r for j in jogos if j.jogado]
    return max(jogados) if jogados else 1


def jogo_do_time_na_rodada(
    jogos: list[Jogo], time: str, rodada: int
) -> Jogo | None:
    for j in jogos:
        if j.r == rodada and time in (j.mand, j.vis):
            return j
    return None


@dataclass
class SegmentoEvolucao:
    rodadas: list[int]
    pontos: list[float]
    tracejado: bool


@dataclass
class EvolucaoTime:
    time: str
    rodadas: list[int]
    pts_confirmado: list[float]
    pts_total: list[float]
    segmentos: list[SegmentoEvolucao]


def evolucao_pontos_time(
    jogos_base: list[Jogo],
    jogos_proj: list[Jogo],
    time: str,
    ult_r: int | None = None,
) -> EvolucaoTime:
    del jogos_base, ult_r
    rodadas = list(range(1, 39))
    pts_conf: list[float] = []
    pts_tot: list[float] = []
    ac_conf = 0.0
    ac_tot = 0.0
    estilos: list[bool] = []

    for r in rodadas:
        j = jogo_do_time_na_rodada(jogos_proj, time, r)
        d_real = 0
        d_proj = 0
        tracejado = False

        if j:
            if j.jogado:
                st = _stats_jogo_para_time(j, time)
                d_real = st.pts if st else 0
            elif j.proj_pm is not None:
                st = _stats_jogo_para_time(j, time)
                d_proj = st.pts if st else 0
                tracejado = True

        ac_conf += d_real
        ac_tot += d_real + d_proj
        pts_conf.append(ac_conf)
        pts_tot.append(ac_tot)
        estilos.append(tracejado)

    segmentos_linha: list[SegmentoEvolucao] = []
    for idx, r in enumerate(rodadas):
        y = pts_tot[idx]
        dashed = estilos[idx]
        if not segmentos_linha or segmentos_linha[-1].tracejado != dashed:
            if segmentos_linha and idx > 0:
                prev_r, prev_y = rodadas[idx - 1], pts_tot[idx - 1]
                segmentos_linha.append(
                    SegmentoEvolucao([prev_r, r], [prev_y, y], dashed)
                )
            else:
                segmentos_linha.append(SegmentoEvolucao([r], [y], dashed))
        else:
            segmentos_linha[-1].rodadas.append(r)
            segmentos_linha[-1].pontos.append(y)

    return EvolucaoTime(
        time=time,
        rodadas=rodadas,
        pts_confirmado=pts_conf,
        pts_total=pts_tot,
        segmentos=segmentos_linha,
    )


def _posicao_time_classificacao_ate(
    jogos: list[Jogo],
    time: str,
    rodada: int,
    *,
    incluir_proj: bool,
) -> int:
    linhas = ordenar_times_desempate(
        jogos, incluir_proj=incluir_proj, ate_rodada=rodada
    )
    for i, (t, _) in enumerate(linhas, 1):
        if t == time:
            return i
    return len(linhas)


def evolucao_posicao_time(
    jogos_base: list[Jogo],
    jogos_proj: list[Jogo],
    time: str,
    ult_r: int | None = None,
) -> EvolucaoTime:
    """Posição na tabela ao fim de cada rodada (real e com projeção)."""
    del jogos_base, ult_r
    rodadas = list(range(1, 39))
    pos_conf: list[float] = []
    pos_tot: list[float] = []
    estilos: list[bool] = []

    for r in rodadas:
        j = jogo_do_time_na_rodada(jogos_proj, time, r)
        tracejado = bool(
            j and not j.jogado and j.proj_pm is not None
        )
        pos_conf.append(
            float(_posicao_time_classificacao_ate(jogos_proj, time, r, incluir_proj=False))
        )
        pos_tot.append(
            float(_posicao_time_classificacao_ate(jogos_proj, time, r, incluir_proj=True))
        )
        estilos.append(tracejado)

    segmentos_linha: list[SegmentoEvolucao] = []
    for idx, r in enumerate(rodadas):
        y = pos_tot[idx]
        dashed = estilos[idx]
        if not segmentos_linha or segmentos_linha[-1].tracejado != dashed:
            if segmentos_linha and idx > 0:
                prev_r, prev_y = rodadas[idx - 1], pos_tot[idx - 1]
                segmentos_linha.append(
                    SegmentoEvolucao([prev_r, r], [prev_y, y], dashed)
                )
            else:
                segmentos_linha.append(SegmentoEvolucao([r], [y], dashed))
        else:
            segmentos_linha[-1].rodadas.append(r)
            segmentos_linha[-1].pontos.append(y)

    return EvolucaoTime(
        time=time,
        rodadas=rodadas,
        pts_confirmado=pos_conf,
        pts_total=pos_tot,
        segmentos=segmentos_linha,
    )


def fig_evolucao_times(
    evolucoes: list[EvolucaoTime],
    cores: dict[str, str] | None = None,
):
    import plotly.graph_objects as go

    palette = [
        "#14532d",
        "#15803d",
        "#ca8a04",
        "#0f766e",
        "#b45309",
        "#166534",
        "#047857",
        "#854d0e",
        "#115e59",
        "#365314",
    ]
    fig = go.Figure()

    r_recente = 1.0
    for ev in evolucoes:
        for seg in ev.segmentos:
            if not seg.tracejado and seg.rodadas:
                r_recente = max(r_recente, float(max(seg.rodadas)))

    for i, ev in enumerate(evolucoes):
        cor = (cores or {}).get(ev.time, palette[i % len(palette)])
        hover_tpl = (
            f"{ev.time} - Pontuação atual: %{{customdata[0]}}<br>"
            f"{ev.time} - Pontuação Projetada da Rodada: %{{customdata[1]}}"
            "<extra></extra>"
        )
        for j, seg in enumerate(ev.segmentos):
            dash = "dash" if seg.tracejado else "solid"
            cd = [
                [
                    int(ev.pts_confirmado[r - 1]),
                    int(ev.pts_total[r - 1]),
                ]
                for r in seg.rodadas
            ]
            fig.add_trace(
                go.Scatter(
                    x=seg.rodadas,
                    y=seg.pontos,
                    mode="lines+markers+text",
                    name=ev.time,
                    line=dict(color=cor, width=2.5, dash=dash),
                    marker=dict(size=5, color=cor),
                    text=_rotulos_em_ticks(
                        seg.rodadas, seg.pontos, r_recente=r_recente
                    ),
                    textposition="top center",
                    textfont=dict(size=10, color=cor),
                    legendgroup=ev.time,
                    showlegend=(j == 0),
                    customdata=cd,
                    hovertemplate=hover_tpl,
                )
            )

    fig.update_layout(
        xaxis_title="Rodada",
        yaxis_title="Pontos acumulados",
        hovermode="x unified",
        height=ALTURA_GRAFICO,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        **_layout_grafico("Pontuação acumulada por rodada"),
    )
    _config_eixo_x_rodada(fig)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
    )
    return fig


def fig_evolucao_posicao_times(
    evolucoes: list[EvolucaoTime],
    cores: dict[str, str] | None = None,
):
    import plotly.graph_objects as go

    palette = [
        "#14532d",
        "#15803d",
        "#ca8a04",
        "#0f766e",
        "#b45309",
        "#166534",
        "#047857",
        "#854d0e",
        "#115e59",
        "#365314",
    ]
    fig = go.Figure()

    r_recente = 1.0
    for ev in evolucoes:
        for seg in ev.segmentos:
            if not seg.tracejado and seg.rodadas:
                r_recente = max(r_recente, float(max(seg.rodadas)))

    for i, ev in enumerate(evolucoes):
        cor = (cores or {}).get(ev.time, palette[i % len(palette)])
        hover_tpl = (
            f"{ev.time} - Posição confirmada: %{{customdata[0]:.0f}}º<br>"
            f"{ev.time} - Posição projetada: %{{customdata[1]:.0f}}º"
            "<extra></extra>"
        )
        for j, seg in enumerate(ev.segmentos):
            dash = "dash" if seg.tracejado else "solid"
            cd = [
                [ev.pts_confirmado[r - 1], ev.pts_total[r - 1]]
                for r in seg.rodadas
            ]
            fig.add_trace(
                go.Scatter(
                    x=seg.rodadas,
                    y=seg.pontos,
                    mode="lines+markers+text",
                    name=ev.time,
                    line=dict(color=cor, width=2.5, dash=dash),
                    marker=dict(size=5, color=cor),
                    text=_rotulos_em_ticks(
                        seg.rodadas, seg.pontos, "Posição", r_recente=r_recente
                    ),
                    textposition="top center",
                    textfont=dict(size=10, color=cor),
                    legendgroup=ev.time,
                    showlegend=(j == 0),
                    customdata=cd,
                    hovertemplate=hover_tpl,
                )
            )

    fig.update_layout(
        xaxis_title="Rodada",
        yaxis_title="Posição",
        hovermode="x unified",
        height=ALTURA_GRAFICO,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        **_layout_grafico("Posição por rodada"),
    )
    _config_eixo_x_rodada(fig)
    _config_eixo_y_posicao(fig, "Posição")
    return fig


def colunas_estatisticas_grafico(df: pd.DataFrame) -> list[str]:
    """Métricas disponíveis nos gráficos de estatísticas."""
    return [c for c in COLUNAS_ESTATISTICAS_GRAFICO if c in df.columns]


_PALETA_SERIES = [
    "#14532d",
    "#ca8a04",
    "#0f766e",
    "#b45309",
    "#15803d",
    "#7c3aed",
    "#0369a1",
    "#be123c",
    "#166534",
    "#854d0e",
    "#047857",
    "#4338ca",
    "#65a30d",
    "#c2410c",
    "#0e7490",
    "#a21caf",
    "#1d4ed8",
    "#b45309",
    "#115e59",
    "#9f1239",
]


def fig_estatisticas_times(
    df: pd.DataFrame,
    times: list[str],
    colunas: list[str],
    *,
    ordenacao: str = "alfabetica",
):
    """Barras agrupadas: times no eixo X; cor por série (estatística)."""
    import plotly.graph_objects as go

    sub = df[df["Time"].isin(times)].copy()
    colunas = [c for c in colunas if c in sub.columns]
    if sub.empty or not colunas:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    disponiveis = [t for t in times if t in sub["Time"].values]
    chave = colunas[0]
    if ordenacao == "maior_menor":
        ordem = sorted(
            disponiveis,
            key=lambda t: float(sub.loc[sub["Time"] == t, chave].iloc[0]),
            reverse=True,
        )
    elif ordenacao == "menor_maior":
        ordem = sorted(
            disponiveis,
            key=lambda t: float(sub.loc[sub["Time"] == t, chave].iloc[0]),
        )
    else:
        ordem = sorted(disponiveis)
    sub = sub.set_index("Time").loc[ordem]

    fig = go.Figure()
    for i, col in enumerate(colunas):
        vals = sub[col].tolist()
        fig.add_trace(
            go.Bar(
                name=col,
                x=sub.index.tolist(),
                y=vals,
                marker_color=_PALETA_SERIES[i % len(_PALETA_SERIES)],
                text=[_fmt_rotulo(v, col) for v in vals],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=10),
                hovertemplate="%{x}<br>" + col + ": %{y}<extra></extra>",
            )
        )

    titulo = (
        f"Comparativo - {colunas[0]}"
        if len(colunas) == 1
        else "Comparativo de estatísticas"
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Valor" if len(colunas) > 1 else colunas[0],
        barmode="group",
        hovermode="x unified",
        height=ALTURA_GRAFICO,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        **_layout_grafico(titulo),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
        automargin=True,
    )
    return fig


def estatisticas_por_rodada(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
) -> pd.DataFrame:
    """Métricas acumuladas no intervalo [r_ini, r] para cada rodada r."""
    times = times_do_calendario(jogos)
    rows: list[dict] = []
    for time in times:
        for r in range(r_ini, r_fim + 1):
            agg = _agregar_stats_time(jogos, time, r_ini, r)
            row = {"Time": time, "Rodada": r, "Projetado": False}
            row.update(_metricas_estatisticas_de_agregado(agg))
            rows.append(row)
    return pd.DataFrame(rows)


def _eh_metrica_total(metrica: str) -> bool:
    return metrica.startswith("Total ")


def projetar_estatisticas_por_rodada(
    jogos: list[Jogo],
    df_real: pd.DataFrame,
    r_ini: int,
    r_fim_obs: int,
    colunas: list[str],
    *,
    variante: VarianteRegressaoAcumulada = "completa",
    r_proj_fim: int = 38,
) -> pd.DataFrame:
    """
    Estende o DF real até r_proj_fim com o mesmo painel FE da pontuação,
    trocando apenas a variável explicada (Y = métrica).
    """
    if df_real.empty:
        return df_real.copy()

    out = df_real.copy()
    if "Projetado" not in out.columns:
        out["Projetado"] = False
    if r_fim_obs >= r_proj_fim:
        return out

    colunas = [c for c in colunas if c in COLUNAS_ESTATISTICAS_GRAFICO]
    if not colunas:
        return out

    times = times_do_calendario(jogos)
    forca_map = mapa_forca_adversario(jogos, r_ini, r_fim_obs)
    casa_fora: dict[str, tuple[int, int]] = {
        t: _contagem_casa_fora_ate(jogos, t, r_fim_obs) for t in times
    }
    forma_base: dict[str, float] = {}
    forma_geral: dict[str, float] = {}
    for t in times:
        jr = jogo_do_time_na_rodada(jogos, t, r_fim_obs)
        if jr is not None and jr.jogado:
            forma_base[t] = float(forma_recente_ate(jogos, t, r_fim_obs, jr.hora))
        else:
            forma_base[t] = 1.0
        fg = media_pts_jogo(jogos, t, r_ini, r_fim_obs, "simples").get("geral", 1.0)
        forma_geral[t] = float(fg) if fg else 1.0

    prev: dict[tuple[str, str], float] = {}
    for t in times:
        row = out[(out["Time"] == t) & (out["Rodada"] == r_fim_obs)]
        for col in colunas:
            if not row.empty and col in row.columns:
                try:
                    prev[(t, col)] = float(row.iloc[0][col])
                except (TypeError, ValueError):
                    prev[(t, col)] = 0.0
            else:
                prev[(t, col)] = 0.0

    # App runtime: só calendário/resultados (FPT multi-liga fica no job semanal).
    betas_por_col: dict[str, dict[str, dict]] = {}
    for col in colunas:
        painel = ajustar_painel_efeitos_fixos(
            jogos,
            r_ini,
            r_fim_obs,
            variante,
            forca_map,
            metrica=col,
            blocos_extra=[],
        )
        betas_por_col[col] = coeficientes_efeitos_fixos_por_time(painel)

    proj_rows: list[dict] = []
    n_casa_prog = {t: casa_fora[t][0] for t in times}
    n_fora_prog = {t: casa_fora[t][1] for t in times}

    for r in range(r_fim_obs + 1, r_proj_fim + 1):
        horizonte = max(1, r - r_fim_obs)
        w = peso_forma_recente_horizonte(horizonte)
        for t in times:
            jogo_r = jogo_do_time_na_rodada(jogos, t, r)
            if jogo_r is not None:
                if jogo_r.mand == t:
                    n_casa_prog[t] += 1
                else:
                    n_fora_prog[t] += 1
            prop = _proporcao_casa(n_casa_prog[t], n_fora_prog[t])
            forca = _forca_oponentes_passados(
                jogos, t, r_fim_obs + 1, forca_map, incluir_proj=False
            )
            forma = w * forma_base[t] + (1.0 - w) * forma_geral[t]
            row: dict = {"Time": t, "Rodada": r, "Projetado": True}
            for col in colunas:
                b = betas_por_col[col].get(t)
                if not b:
                    row[col] = prev[(t, col)]
                    continue
                target = _prever_acumulado(
                    b,
                    r,
                    prop_casa=prop,
                    forca_oponentes=forca,
                    forma_recente=forma,
                )
                ant = prev[(t, col)]
                if _eh_metrica_total(col):
                    delta = max(0.0, target - ant)
                    if col == "Total pontos" and b.get("limita_delta_rodada"):
                        delta = min(DELTA_PTS_MAX_POR_RODADA, delta)
                    val = ant + delta
                else:
                    val = float(target)
                prev[(t, col)] = val
                row[col] = round(val, 3) if isinstance(val, float) else val
            proj_rows.append(row)

    if not proj_rows:
        return out
    return pd.concat([out, pd.DataFrame(proj_rows)], ignore_index=True)


def colunas_estatisticas_rodada_grafico(df: pd.DataFrame) -> list[str]:
    return [c for c in COLUNAS_ESTATISTICAS_GRAFICO if c in df.columns]


_TICKS_EIXO_POSICAO = [1, 5, 10, 15, 20]


def _layout_legenda_desktop() -> dict:
    """Legenda horizontal centralizada acima do gráfico (PC)."""
    return {
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.5,
            xanchor="center",
        ),
        "margin": dict(t=60),
    }


def _layout_grafico(titulo: str) -> dict:
    """Layout padrão: sem título no Plotly (título fica fora, estilo seção)."""
    layout = _layout_legenda_desktop()
    # text="" evita o "undefined" que o Plotly/JS mostra com title=None
    layout["title"] = dict(text="")
    layout["meta"] = {"titulo": titulo}
    return layout


def titulo_fig(fig) -> str | None:
    meta = fig.layout.meta
    if isinstance(meta, dict):
        t = meta.get("titulo")
        return str(t) if t else None
    return None


def _fmt_rotulo(val, col: str | None = None) -> str:
    if val is None:
        return ""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if col == "Posição":
        return f"{int(round(f))}º"
    if col and "média" in col.lower():
        return f"{f:.2f}"
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.2f}"


def _rotulos_em_ticks(
    xs,
    ys,
    col: str | None = None,
    *,
    r_recente: float | None = None,
) -> list[str]:
    """Rótulos na rodada 19, na rodada atual e na 38."""
    if not xs:
        return []
    alvos = {19.0, 38.0}
    if r_recente is not None:
        alvos.add(float(r_recente))
    return [
        _fmt_rotulo(y, col)
        if any(abs(float(x) - a) < 1e-9 for a in alvos)
        else ""
        for x, y in zip(xs, ys, strict=False)
    ]


def extrair_itens_legenda(fig) -> list[dict[str, str]]:
    """Nome e cor das séries visíveis na legenda (para expander no mobile)."""
    itens: list[dict[str, str]] = []
    vistos: set[str] = set()
    for trace in fig.data:
        if getattr(trace, "showlegend", True) is False:
            continue
        nome = getattr(trace, "name", None)
        if not nome or nome in vistos:
            continue
        vistos.add(str(nome))
        cor = "#14532d"
        linha = getattr(trace, "line", None)
        marcador = getattr(trace, "marker", None)
        if linha is not None and getattr(linha, "color", None):
            cor = str(linha.color)
        elif marcador is not None and getattr(marcador, "color", None):
            c = marcador.color
            if isinstance(c, str):
                cor = c
            elif isinstance(c, (list, tuple)) and c:
                cor = str(c[0])
        itens.append({"nome": str(nome), "cor": cor})
    return itens


def _config_eixo_x_rodada(fig, r_max: float = 38) -> None:
    """Eixo X de rodada: 1, 9.5, 19, 28.5 e 38 (até r_max)."""
    r_max = max(1.0, float(r_max))
    ticks = [float(t) for t in TICKS_RODADA if float(t) <= r_max + 1e-9]
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[
            str(int(t)) if float(t).is_integer() else str(t) for t in ticks
        ],
        range=[0.5, min(38.0, r_max) + 0.5],
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
    )


def _kwargs_eixo_y(col: str) -> dict:
    kwargs: dict = {
        "showgrid": True,
        "gridcolor": "rgba(15, 23, 42, 0.08)",
        "zeroline": False,
    }
    if col == "Posição":
        kwargs.update(
            autorange="reversed",
            tickmode="array",
            tickvals=_TICKS_EIXO_POSICAO,
            ticktext=[str(t) for t in _TICKS_EIXO_POSICAO],
            range=[0.5, 20.5],
        )
    return kwargs


def _config_eixo_y_posicao(
    fig,
    col: str,
    *,
    row: int | None = None,
    col_num: int = 1,
    title_text: str | None = None,
):
    """Posição 1 no topo; demais métricas mantêm escala crescente."""
    kwargs = _kwargs_eixo_y(col)
    if title_text:
        kwargs["title_text"] = title_text
    if row is not None:
        fig.update_yaxes(**kwargs, row=row, col=col_num)
    else:
        fig.update_yaxes(**kwargs)


def fig_estatisticas_por_rodada(
    df: pd.DataFrame,
    times: list[str],
    colunas: list[str],
    *,
    r_atual: int | None = None,
):
    """Linhas por rodada; sólido = real, tracejado = projetado até 38."""
    import plotly.graph_objects as go

    sub = df[df["Time"].isin(times)].copy()
    colunas = [c for c in colunas if c in sub.columns]
    if sub.empty or not colunas:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    sub["Rodada"] = pd.to_numeric(sub["Rodada"], errors="coerce")
    sub = sub.dropna(subset=["Rodada"])
    sub["Rodada"] = sub["Rodada"].astype(int)
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    if "Projetado" not in sub.columns:
        sub["Projetado"] = False
    else:
        sub["Projetado"] = sub["Projetado"].fillna(False).astype(bool)

    r_eixo = float(sub["Rodada"].max())
    if r_atual is None:
        reais = sub.loc[~sub["Projetado"], "Rodada"]
        r_atual = int(reais.max()) if not reais.empty else int(r_eixo)

    ordem_times = [t for t in times if t in sub["Time"].unique()]
    um_time = len(ordem_times) == 1
    uma_serie = len(colunas) == 1
    fig = go.Figure()
    k = 0
    for time in ordem_times:
        s = sub[sub["Time"] == time].sort_values("Rodada")
        for col in colunas:
            if um_time and not uma_serie:
                nome = col
            elif uma_serie and not um_time:
                nome = time
            elif um_time and uma_serie:
                nome = col
            else:
                nome = f"{time} - {col}"
            cor = _PALETA_SERIES[k % len(_PALETA_SERIES)]
            s_real = s[~s["Projetado"]]
            s_proj = s[s["Projetado"]]

            xs_real = s_real["Rodada"].tolist()
            ys_real = s_real[col].tolist()
            hover_nome = f"{time} - {col}"
            if xs_real:
                fig.add_trace(
                    go.Scatter(
                        x=xs_real,
                        y=ys_real,
                        mode="lines+markers+text",
                        name=nome,
                        line=dict(color=cor, width=2.5, dash="solid"),
                        marker=dict(size=5),
                        text=_rotulos_em_ticks(
                            xs_real, ys_real, col, r_recente=float(r_atual)
                        ),
                        textposition="top center",
                        textfont=dict(size=9, color=cor),
                        legendgroup=f"{time}||{col}",
                        showlegend=True,
                        hovertemplate=(
                            f"{hover_nome}<br>Rodada %{{x}}<br>{col}: %{{y}}"
                            "<extra></extra>"
                        ),
                    )
                )

            if not s_proj.empty:
                # Conecta último ponto real ao início da projeção
                xs_p = s_proj["Rodada"].tolist()
                ys_p = s_proj[col].tolist()
                if xs_real:
                    xs_p = [xs_real[-1]] + xs_p
                    ys_p = [ys_real[-1]] + ys_p
                fig.add_trace(
                    go.Scatter(
                        x=xs_p,
                        y=ys_p,
                        mode="lines+markers+text",
                        name=nome,
                        line=dict(color=cor, width=2.5, dash="dash"),
                        marker=dict(size=5),
                        text=_rotulos_em_ticks(xs_p, ys_p, col),
                        textposition="top center",
                        textfont=dict(size=9, color=cor),
                        legendgroup=f"{time}||{col}",
                        showlegend=False,
                        hovertemplate=(
                            f"{hover_nome} (proj.)<br>Rodada %{{x}}<br>{col}: %{{y}}"
                            "<extra></extra>"
                        ),
                    )
                )
            k += 1

    titulo = (
        f"Evolução rodada a rodada - {colunas[0]}"
        if len(colunas) == 1
        else "Evolução rodada a rodada"
    )
    fig.update_layout(
        xaxis_title="Rodada",
        yaxis_title="Valor" if len(colunas) > 1 else colunas[0],
        hovermode="x unified",
        height=ALTURA_GRAFICO,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        **_layout_grafico(titulo),
    )
    _config_eixo_x_rodada(
        fig,
        max(r_eixo, 38.0) if sub["Projetado"].any() else r_eixo,
    )
    if len(colunas) == 1:
        _config_eixo_y_posicao(fig, colunas[0])
    else:
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
        )
    return fig

