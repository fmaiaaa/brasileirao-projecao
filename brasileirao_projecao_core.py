"""Lógica de projeção — Brasileirão 2026."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ModoProjecao = Literal["regressao", "media_simples", "repetir_turno"]
TipoRegressao = Literal["simples", "mandante_visitante"]

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
    proj_pm: int | None = field(default=None, repr=False)
    proj_pv: int | None = field(default=None, repr=False)
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

    def pts_efetivos(self) -> tuple[int, int]:
        if self.jogado:
            return self.pts_reais()  # type: ignore[return-value]
        if self.proj_pm is not None and self.proj_pv is not None:
            return self.proj_pm, self.proj_pv
        return 0, 0


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
    """Lê calendário da planilha Google (secrets [connections.gsheets])."""
    from brasileirao_gsheets import (
        _secrets_connections_gsheets,
        montar_service_account_info,
        ler_planilha_gsheets,
        spreadsheet_id_brasileirao,
    )

    raw = _secrets_connections_gsheets()
    info = montar_service_account_info(raw)
    if not info:
        raise ValueError(
            "Credenciais [connections.gsheets] ausentes ou incompletas. "
            "Use as mesmas secrets do velocímetro (private_key + client_email)."
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


def _regressao_interacao_mando(
    rodadas: np.ndarray,
    indicador_casa: np.ndarray,
    pts: np.ndarray,
) -> dict[str, float]:
    """
    pts ~ intercept + beta_rodada*rodada + beta_casa*I_casa + beta_inter*rodada*I_casa.
    Fora: pts = b0 + b1*r  |  Casa: pts = (b0+b2) + (b1+b3)*r
    """
    n = len(pts)
    n_casa = int(indicador_casa.sum())
    n_fora = n - n_casa

    if n == 0:
        return {
            "geral": 1.0,
            "casa": 1.0,
            "fora": 1.0,
            "intercept": 1.0,
            "beta_rodada": 0.0,
            "beta_casa_ind": 0.0,
            "beta_interacao": 0.0,
        }

    b0_g, b1_g = _ols_pts_rodada(rodadas, pts)

    if n_fora == 0 or n_casa == 0 or n < 4:
        b0, b1 = b0_g, b1_g
        return {
            "geral": max(b1_g, 0.0),
            "casa": max(b1_g, 0.0),
            "fora": max(b1_g, 0.0),
            "intercept": b0,
            "beta_rodada": b1,
            "beta_casa_ind": 0.0,
            "beta_interacao": 0.0,
        }

    r = rodadas.astype(float)
    casa = indicador_casa.astype(float)
    y = pts.astype(float)
    X = np.column_stack([np.ones(n), r, casa, r * casa])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1, b2, b3 = (float(c) for c in coef)

    return {
        "geral": max(b1_g, 0.0),
        "casa": max(b1 + b3, 0.0),
        "fora": max(b1, 0.0),
        "intercept": b0,
        "beta_rodada": b1,
        "beta_casa_ind": b2,
        "beta_interacao": b3,
    }


def regressao_beta(
    jogos: list[Jogo],
    time: str,
    r_ini: int,
    r_fim: int,
    tipo: TipoRegressao,
) -> dict[str, float]:
    """
    Retorna coeficientes de projeção no intervalo [r_ini, r_fim].
    simples: inclinação do acumulado por rodada (pts/rodada).
    mandante_visitante: OLS por jogo
        pts ~ rodada + indicador_casa + rodada × indicador_casa.
    """
    pts_por_r: dict[int, int] = {}
    rodadas_jogo: list[int] = []
    indicador_casa: list[float] = []
    pts_jogo: list[float] = []

    for j in jogos:
        if not j.jogado or j.r < r_ini or j.r > r_fim:
            continue
        pm, pv = j.pts_reais()
        if j.mand == time:
            pts_por_r[j.r] = pts_por_r.get(j.r, 0) + pm
            rodadas_jogo.append(j.r)
            indicador_casa.append(1.0)
            pts_jogo.append(float(pm))
        elif j.vis == time:
            pts_por_r[j.r] = pts_por_r.get(j.r, 0) + pv
            rodadas_jogo.append(j.r)
            indicador_casa.append(0.0)
            pts_jogo.append(float(pv))

    def slope_from_round_dict(d: dict[int, int]) -> float:
        if not d:
            return 1.0
        rounds = sorted(d.keys())
        acum = np.cumsum([d[r] for r in rounds]).astype(float)
        xs = np.array(rounds, dtype=float)
        if len(xs) < 2:
            return float(acum[-1]) / max(len(xs), 1)
        coef = np.polyfit(xs, acum, 1)
        return float(coef[0])

    beta_geral = slope_from_round_dict(pts_por_r)
    if tipo == "simples":
        return {"geral": max(beta_geral, 0.0)}

    return _regressao_interacao_mando(
        np.array(rodadas_jogo, dtype=float),
        np.array(indicador_casa, dtype=float),
        np.array(pts_jogo, dtype=float),
    )


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
) -> dict[str, float]:
    """Betas (regressão) ou médias (pts/jogo) conforme o modo."""
    if modo == "media_simples":
        return media_pts_jogo(jogos, time, r_ini, r_fim, tipo)
    return regressao_beta(jogos, time, r_ini, r_fim, tipo)


def expected_pts_jogo(
    time: str,
    mand: str,
    betas: dict[str, dict[str, float]],
    tipo: TipoRegressao,
    *,
    rodada: int | None = None,
) -> float:
    b = betas.get(time, {"geral": 1.0})
    if tipo == "simples":
        return max(b.get("geral", 1.0), 0.0)

    if rodada is not None and "intercept" in b:
        r = float(rodada)
        b0 = b["intercept"]
        b1 = b["beta_rodada"]
        b2 = b.get("beta_casa_ind", 0.0)
        b3 = b.get("beta_interacao", 0.0)
        if time == mand:
            val = b0 + b2 + (b1 + b3) * r
        else:
            val = b0 + b1 * r
        return max(val, 0.0)

    if time == mand:
        return max(b.get("casa", b.get("geral", 1.0)), 0.0)
    return max(b.get("fora", b.get("geral", 1.0)), 0.0)


def projetar_jogo_regressao(
    jogo: Jogo,
    betas: dict[str, dict[str, float]],
    tipo: TipoRegressao,
) -> tuple[int, int]:
    em = expected_pts_jogo(
        jogo.mand, jogo.mand, betas, tipo, rodada=jogo.r
    )
    ev = expected_pts_jogo(
        jogo.vis, jogo.mand, betas, tipo, rodada=jogo.r
    )
    total = em + ev
    if total <= 0:
        return 1, 1
    # converte expectativa contínua em placar 3/1/0 coerente
    pm_f = 3.0 * em / total
    pv_f = 3.0 * ev / total
    pm, pv = round(pm_f), round(pv_f)
    # ajuste para soma típica (0-3 cada)
    if pm + pv > 3:
        scale = 3 / (pm + pv)
        pm, pv = int(round(pm * scale)), int(round(pv * scale))
    if pm > 3:
        pm = 3
    if pv > 3:
        pv = 3
    # desempate por comparação de força
    if pm > pv:
        return 3, 0
    if pv > pm:
        return 0, 3
    return 1, 1


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


def aplicar_projecoes(
    jogos: list[Jogo],
    modo: ModoProjecao,
    r_ini: int,
    r_fim: int,
    tipo_reg: TipoRegressao,
) -> tuple[list[Jogo], pd.DataFrame]:
    jogos = [Jogo(**j.__dict__) for j in jogos]  # cópia
    times = times_do_calendario(jogos)
    mapa = mapa_contrapartidas(jogos)

    if modo == "repetir_turno":
        modo_metrica: ModoProjecao = "regressao"
        tipo_metrica: TipoRegressao = "mandante_visitante"
    else:
        modo_metrica = modo
        tipo_metrica = tipo_reg

    betas = {
        t: metricas_time(jogos, t, r_ini, r_fim, modo_metrica, tipo_metrica)
        for t in times
    }

    label_modelo = {
        "regressao": "regressão",
        "media_simples": "média simples",
    }.get(modo_metrica, "regressão")
    label_fallback = f"sem espelho — {label_modelo}"

    log_rows = []

    for j in sorted(jogos, key=lambda x: (x.r, x.hora, x.mand)):
        if j.jogado:
            continue

        if modo == "repetir_turno":
            chave = (j.par, j.vis, j.mand)
            ref = mapa.get(chave)
            if ref and ref.jogado:
                pm, pv = espelhar_contrapartida(j, ref)
                j.proj_pm, j.proj_pv = pm, pv
                j.origem = f"espelho R{ref.r} ({ref.mand} {ref.placar} {ref.vis})"
            else:
                pm, pv = projetar_jogo_regressao(j, betas, tipo_metrica)
                j.proj_pm, j.proj_pv = pm, pv
                j.origem = label_fallback
        else:
            pm, pv = projetar_jogo_regressao(j, betas, tipo_reg)
            j.proj_pm, j.proj_pv = pm, pv
            j.origem = label_modelo

        log_rows.append(
            {
                "Rodada": j.r,
                "Mandante": j.mand,
                "Visitante": j.vis,
                "Proj": f"{j.proj_pm} / {j.proj_pv}",
            }
        )

    return jogos, pd.DataFrame(log_rows)


def classificacao(jogos: list[Jogo], incluir_proj: bool = True) -> pd.DataFrame:
    pts: dict[str, int] = {t: 0 for t in times_do_calendario(jogos)}
    jog_real, jog_proj = 0, 0

    for j in jogos:
        if j.jogado:
            pm, pv = j.pts_reais()  # type: ignore[misc]
            jog_real += 1
        elif incluir_proj and j.proj_pm is not None:
            pm, pv = j.proj_pm, j.proj_pv
            jog_proj += 1
        else:
            continue
        pts[j.mand] += pm
        pts[j.vis] += pv

    df = pd.DataFrame(
        {"Time": list(pts.keys()), "Pontos": list(pts.values())}
    ).sort_values(["Pontos", "Time"], ascending=[False, True])
    df.insert(0, "Pos", range(1, len(df) + 1))
    df.attrs["jog_real"] = jog_real
    df.attrs["jog_proj"] = jog_proj
    return df


def tabela_betas(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    tipo: TipoRegressao,
    modo: ModoProjecao = "regressao",
) -> pd.DataFrame:
    rows = []
    col_main = (
        "Media_pts/jogo" if modo == "media_simples" else "Beta_pts/rodada"
    )
    col_casa = "Media_casa" if modo == "media_simples" else "Beta_casa"
    col_fora = "Media_fora" if modo == "media_simples" else "Beta_fora"

    for t in times_do_calendario(jogos):
        b = metricas_time(jogos, t, r_ini, r_fim, modo, tipo)
        row = {"Time": t, col_main: round(b["geral"], 3)}
        if tipo == "mandante_visitante":
            row[col_casa] = round(b.get("casa", b["geral"]), 3)
            row[col_fora] = round(b.get("fora", b["geral"]), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(col_main, ascending=False)


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


def tabela_estatisticas_times(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
) -> pd.DataFrame:
    rows = []
    for time in times_do_calendario(jogos):
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

        betas = regressao_beta(jogos, time, r_ini, r_fim, "mandante_visitante")
        rows.append(
            {
                "Time": time,
                "Jogos realizados": jr,
                "Jogos pendentes": jp,
                "Total gols marcados": gf,
                "Total gols sofridos": gc,
                "Média gols marcados/jogo": round(gf / jr, 3) if jr else 0.0,
                "Média gols sofridos/jogo": round(gc / jr, 3) if jr else 0.0,
                "Média gols marcados casa": round(gf_c / n_c, 3) if n_c else 0.0,
                "Média gols sofridos casa": round(gc_c / n_c, 3) if n_c else 0.0,
                "Média gols marcados fora": round(gf_f / n_f, 3) if n_f else 0.0,
                "Média gols sofridos fora": round(gc_f / n_f, 3) if n_f else 0.0,
                "Total pontos": pts,
                "Total pts mandante": pts_c,
                "Total pts visitante": pts_f,
                "Média pts mandante": round(pts_c / n_c, 3) if n_c else 0.0,
                "Média pts visitante": round(pts_f / n_f, 3) if n_f else 0.0,
                "Beta pts/rodada": round(betas["geral"], 3),
                "Beta pts mandante/rodada": round(betas.get("casa", betas["geral"]), 3),
                "Beta pts visitante/rodada": round(betas.get("fora", betas["geral"]), 3),
            }
        )
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
    df = df.sort_values("Posição Projetada")
    df.insert(0, "Posição", range(1, len(df) + 1))
    return df[
        ["Posição", "Time", "Posição Atual", "Posição Projetada", "Delta"]
    ]


def mapa_posicao_pontos(
    jogos: list[Jogo], *, incluir_proj: bool
) -> dict[str, tuple[int, int]]:
    df = classificacao(jogos, incluir_proj=incluir_proj)
    return {row.Time: (int(row.Pos), int(row.Pontos)) for row in df.itertuples()}


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
    pts: int = 0
    vit: int = 0
    emp: int = 0
    der: int = 0
    gf: int = 0
    gc: int = 0

    @property
    def sg(self) -> int:
        return self.gf - self.gc

    def chave_classificacao(self) -> tuple:
        return (-self.pts, -self.vit, -self.sg, -self.gf)

    def copy(self) -> "StatsTime":
        return StatsTime(self.pts, self.vit, self.emp, self.der, self.gf, self.gc)

    def add(self, pts: int, gf: int, gc: int) -> None:
        self.pts += pts
        self.gf += gf
        self.gc += gc
        if pts == 3:
            self.vit += 1
        elif pts == 1:
            self.emp += 1
        else:
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
    if jogo.proj_pm is not None and jogo.proj_pv is not None:
        pts = jogo.proj_pm if time == jogo.mand else jogo.proj_pv
        s = StatsTime()
        if pts == 3:
            s.add(3, 1, 0)
        elif pts == 0:
            s.add(0, 0, 1)
        else:
            s.add(1, 1, 1)
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


def posicao_time_na_rodada(
    jogos: list[Jogo],
    time: str,
    rodada: int,
    extra: dict[str, StatsTime] | None = None,
) -> int:
    """Posição na tabela ao fim da rodada (só jogos realizados + extras opcionais)."""
    times = times_do_calendario(jogos)
    extra = extra or {}
    linhas = []
    for t in times:
        s = stats_acumuladas_ate(jogos, t, rodada, so_realizados=True)
        if t in extra:
            e = extra[t]
            s.add(e.pts, e.gf, e.gc)
        linhas.append((t, s.chave_classificacao()))
    linhas.sort(key=lambda x: x[1])
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

    # empate na posição numérica: compara chave completa com vizinhos
    times = times_do_calendario(jogos)
    chaves = {}
    for cen, label in ((melhor, "m"), (pior, "p")):
        extra = {time: cen}
        linhas = []
        for t in times:
            s = stats_acumuladas_ate(jogos, t, jogo.r, so_realizados=True)
            if t in extra:
                s.add(extra[t].pts, extra[t].gf, extra[t].gc)
            linhas.append((t, s.chave_classificacao()))
        linhas.sort(key=lambda x: x[1])
        chaves[label] = next(k for t, k in linhas if t == time)

    return chaves["m"] != chaves["p"]


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
                    mode="lines+markers",
                    name=ev.time,
                    line=dict(color=cor, width=2.5, dash=dash),
                    marker=dict(size=5, color=cor),
                    legendgroup=ev.time,
                    showlegend=(j == 0),
                    customdata=cd,
                    hovertemplate=hover_tpl,
                )
            )

    fig.update_layout(
        title="Pontuação acumulada por rodada (1–38)",
        xaxis_title="Rodada",
        yaxis_title="Pontos acumulados",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=520,
        margin=dict(t=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
    )
    fig.update_xaxes(
        dtick=1,
        range=[0.5, 38.5],
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
    )
    return fig

