"""
Modelos alternativos de pontos acumulados (mesmas variáveis da regressão completa).

Kalman (coeficientes evoluindo no tempo), XGBoost e GAM compartilham o painel
time×rodada com Rodada, Rodada², Forma, FAP, Indicador casa, Descanso e fases de copa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from recency import (
    JanelaTreino,
    load_recency_settings,
    parse_match_date,
    weights_for_panel_rounds,
)
from brasileirao_secoes import (
    LABEL_FE,
    LABEL_KALMAN,
    LABEL_XGB,
    LABEL_GAM,
    MODO_PARA_LABEL,
    usa_features_serie_ab,
)

TipoModeloAcumulado = Literal["kalman", "xgboost", "gam"]

COPA_FASES = ("Classificatórias", "Oitavas", "Quartas", "Semi", "Final")
CONTINUOUS_FEATURES = (
    "rodada",
    "rodada2",
    "forma",
    "forca",
    "casa",
    "descanso",
)
COPA_COLS = [f"imp_{f.lower().replace('á', 'a').replace('ó', 'o')}" for f in COPA_FASES]
COPA_COL_MAP = {
    "Classificatórias": "imp_classificatorias",
    "Oitavas": "imp_oitavas",
    "Quartas": "imp_quartas",
    "Semi": "imp_semi",
    "Final": "imp_final",
}

NOME_MODELO_ACUMULADO: dict[TipoModeloAcumulado, str] = {
    "kalman": LABEL_KALMAN,
    "xgboost": LABEL_XGB,
    "gam": LABEL_GAM,
}


@dataclass
class PainelTreino:
    times: list[str]
    times_obs: list[str]
    rodadas: np.ndarray
    y: np.ndarray
    X: np.ndarray
    feature_names: list[str]
    weights: np.ndarray
    n_obs: int = 0


@dataclass
class ModeloAcumuladoAjustado:
    tipo: TipoModeloAcumulado
    janela: JanelaTreino
    painel: PainelTreino
    state: Any = None
    r2: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _importante_para_colunas(importante: str) -> dict[str, float]:
    out = {c: 0.0 for c in COPA_COLS}
    col = COPA_COL_MAP.get(str(importante))
    if col:
        out[col] = 1.0
    return out


def _encode_team_dummies(times_obs: list[str], times_ref: list[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(times_ref)}
    n = len(times_obs)
    k = len(times_ref)
    mat = np.zeros((n, k), dtype=float)
    for i, t in enumerate(times_obs):
        j = idx.get(t)
        if j is not None:
            mat[i, j] = 1.0
    return mat


def montar_matriz_features(
    times_obs: list[str],
    rodadas: np.ndarray,
    prop: np.ndarray,
    forca: np.ndarray,
    forma: np.ndarray,
    descanso: np.ndarray,
    importantes: list[str],
    times_ref: list[str],
    *,
    comp_serie_a: np.ndarray | None = None,
    usar_interacao_serie: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Design matrix: time FE + variáveis + opcional Série A/B e interação time×série."""
    r = np.asarray(rodadas, dtype=float)
    team = _encode_team_dummies(times_obs, times_ref)
    base = np.column_stack(
        [
            r,
            r ** 2,
            np.asarray(forma, dtype=float),
            np.asarray(forca, dtype=float),
            np.asarray(prop, dtype=float),
            np.asarray(descanso, dtype=float),
        ]
    )
    copa = np.zeros((len(times_obs), len(COPA_COLS)), dtype=float)
    for i, imp in enumerate(importantes):
        for col, val in _importante_para_colunas(imp).items():
            j = COPA_COLS.index(col)
            copa[i, j] = val
    parts = [team]
    names = [f"FE[{t}]" for t in times_ref]
    if usar_interacao_serie and comp_serie_a is not None:
        comp = np.asarray(comp_serie_a, dtype=float).reshape(-1, 1)
        team_x = team * comp
        parts.extend([comp, team_x])
        names.extend(["comp_serie_a"] + [f"FE[{t}]*serie_a" for t in times_ref])
    parts.extend([base, copa])
    names.extend(list(CONTINUOUS_FEATURES) + COPA_COLS)
    X = np.column_stack(parts)
    return X, names


def coletar_painel_treino(
    jogos_calendario: list,
    janela: JanelaTreino,
    *,
    blocos_extra: list[list] | None = None,
    ano_calendario: int = 2026,
) -> PainelTreino:
    """Monta painel unificado respeitando a janela de treino."""
    from brasileirao_projecao_core import (
        _coletar_painel_efeitos_fixos,
        mapa_forca_adversario,
        preparar_blocos_treino_janela,
        times_do_calendario,
    )

    cal_teams = times_do_calendario(jogos_calendario)
    cal_set = set(cal_teams)
    rcfg = load_recency_settings()
    blocos = preparar_blocos_treino_janela(
        jogos_calendario,
        janela,
        blocos_extra=blocos_extra,
        ano_calendario=ano_calendario,
    )

    times_obs_l: list[str] = []
    r_l: list[float] = []
    y_l: list[float] = []
    prop_l: list[float] = []
    forca_l: list[float] = []
    forma_l: list[float] = []
    descanso_l: list[float] = []
    importante_l: list[str] = []
    datas_l: list[str] = []

    comp_l: list[float] = []
    blocos_flat: list = []
    for bloco in blocos:
        blocos_flat.extend(bloco or [])

    for bloco in blocos:
        if not bloco:
            continue
        r0 = min(j.r for j in bloco)
        r1 = max(j.r for j in bloco)
        fm = mapa_forca_adversario(bloco, r0, r1)
        t_b, r_b, y_b, p_b, f_b, fr_b, d_b, i_b, dt_b = _coletar_painel_efeitos_fixos(
            bloco, r0, r1, fm
        )
        times_obs_l.extend(t_b)
        r_l.extend(r_b.tolist())
        y_l.extend(y_b.tolist())
        prop_l.extend(p_b.tolist())
        forca_l.extend(f_b.tolist())
        forma_l.extend(fr_b.tolist())
        descanso_l.extend(d_b.tolist())
        importante_l.extend(i_b)
        datas_l.extend(dt_b)

    from brasileirao_projecao_core import _comp_serie_a_jogo, jogo_do_time_na_rodada

    for t, rv, d in zip(times_obs_l, r_l, datas_l):
        comp = 1.0
        jr = jogo_do_time_na_rodada(blocos_flat, t, int(rv))
        if jr is not None:
            comp = _comp_serie_a_jogo(jr)
        comp_l.append(comp)

    times_ref = list(cal_teams)
    for t in sorted(set(times_obs_l)):
        if t not in cal_set:
            times_ref.append(t)

    rodadas = np.array(r_l, dtype=float)
    y = np.array(y_l, dtype=float)
    if len(y) == 0:
        return PainelTreino(
            times=times_ref,
            times_obs=[],
            rodadas=rodadas,
            y=y,
            X=np.zeros((0, 0)),
            feature_names=[],
            weights=np.array([]),
            n_obs=0,
        )

    X, names = montar_matriz_features(
        times_obs_l,
        rodadas,
        np.array(prop_l),
        np.array(forca_l),
        np.array(forma_l),
        np.array(descanso_l),
        importante_l,
        times_ref,
        comp_serie_a=np.array(comp_l, dtype=float),
        usar_interacao_serie=usa_features_serie_ab(janela),
    )
    w = weights_for_panel_rounds(
        r_l,
        seasons=[
            int(parse_match_date(d).year) if parse_match_date(d) else int(ano_calendario)
            for d in datas_l
        ],
        current_season=int(ano_calendario),
        r_latest=int(max(r_l)) if r_l else 38,
        janela=janela,
        ano_calendario=int(ano_calendario),
    )
    return PainelTreino(
        times=times_ref,
        times_obs=times_obs_l,
        rodadas=rodadas,
        y=y,
        X=X,
        feature_names=names,
        weights=w,
        n_obs=len(y),
    )


def _r2_weighted(y: np.ndarray, yhat: np.ndarray, w: np.ndarray | None) -> float | None:
    if len(y) < 2:
        return None
    if w is None:
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
    else:
        w = np.clip(np.asarray(w, dtype=float), 1e-12, None)
        sw = w / w.sum()
        ym = float(np.dot(y, sw))
        ss_res = float(np.dot(w, (y - yhat) ** 2))
        ss_tot = float(np.dot(w, (y - ym) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


class KalmanRegressor:
    """Filtro de Kalman: coeficientes evoluem como passeio aleatório."""

    def __init__(
        self,
        n_features: int,
        *,
        process_noise: float = 0.05,
        obs_noise: float = 4.0,
    ):
        self.n = n_features
        self.x = np.zeros(n_features, dtype=float)
        self.P = np.eye(n_features, dtype=float) * 100.0
        self.Q = np.eye(n_features, dtype=float) * process_noise
        self.R = float(obs_noise)

    def update(self, x_row: np.ndarray, y: float, weight: float = 1.0) -> None:
        x_row = np.asarray(x_row, dtype=float)
        x_pred = self.x
        P_pred = self.P + self.Q
        y_pred = float(x_row @ x_pred)
        R_eff = self.R / max(weight, 1e-6)
        S = float(x_row @ P_pred @ x_row + R_eff)
        if S <= 0:
            return
        K = (P_pred @ x_row) / S
        self.x = x_pred + K * (y - y_pred)
        self.P = (np.eye(self.n) - np.outer(K, x_row)) @ P_pred

    def predict(self, x_row: np.ndarray) -> float:
        return float(np.asarray(x_row, dtype=float) @ self.x)

    def fit_panel(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, rodadas: np.ndarray | None = None) -> None:
        order = np.argsort(rodadas if rodadas is not None else np.arange(len(y)))
        for i in order:
            self.update(X[i], float(y[i]), float(weights[i]))


def ajustar_kalman(painel: PainelTreino) -> ModeloAcumuladoAjustado:
    kf = KalmanRegressor(painel.X.shape[1])
    kf.fit_panel(painel.X, painel.y, painel.weights, painel.rodadas)
    yhat = np.array([kf.predict(painel.X[i]) for i in range(len(painel.y))])
    r2 = _r2_weighted(painel.y, yhat, painel.weights)
    return ModeloAcumuladoAjustado(
        tipo="kalman",
        janela="ultimas_38_rodadas",
        painel=painel,
        state=kf,
        r2=round(r2, 4) if r2 is not None else None,
    )


def ajustar_xgboost(painel: PainelTreino) -> ModeloAcumuladoAjustado:
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError("Instale xgboost: pip install xgboost") from e

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(painel.X, painel.y, sample_weight=painel.weights)
    yhat = model.predict(painel.X)
    r2 = _r2_weighted(painel.y, yhat, painel.weights)
    return ModeloAcumuladoAjustado(
        tipo="xgboost",
        janela="ultimas_38_rodadas",
        painel=painel,
        state=model,
        r2=round(r2, 4) if r2 is not None else None,
    )


def ajustar_gam(painel: PainelTreino) -> ModeloAcumuladoAjustado:
    """GAM via splines (Rodada, Rodada²) + termos lineares (demais variáveis + FE time)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer

    n_team = len(painel.times)
    n_cont = len(CONTINUOUS_FEATURES)
    n = painel.n_obs
    if n == 0:
        return ModeloAcumuladoAjustado(
            tipo="gam", janela="ultimas_38_rodadas", painel=painel, state=None
        )

    cont = painel.X[:, n_team : n_team + n_cont]
    rest = painel.X[:, n_team + n_cont :]
    spl = SplineTransformer(
        n_knots=5, degree=3, include_bias=False, knots="quantile"
    )
    X_spline = spl.fit_transform(cont[:, :2])
    X_full = np.column_stack([X_spline, cont[:, 2:], rest])
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_full, painel.y, sample_weight=painel.weights)
    yhat = ridge.predict(X_full)
    r2 = _r2_weighted(painel.y, yhat, painel.weights)
    return ModeloAcumuladoAjustado(
        tipo="gam",
        janela="ultimas_38_rodadas",
        painel=painel,
        state={"ridge": ridge, "splines": spl, "n_team": n_team, "n_cont": n_cont},
        r2=round(r2, 4) if r2 is not None else None,
    )


def ajustar_modelo_acumulado(
    painel: PainelTreino,
    tipo: TipoModeloAcumulado,
    janela: JanelaTreino,
) -> ModeloAcumuladoAjustado:
    if tipo == "kalman":
        m = ajustar_kalman(painel)
    elif tipo == "xgboost":
        m = ajustar_xgboost(painel)
    else:
        m = ajustar_gam(painel)
    m.janela = janela
    return m


def _vetor_features_projecao(
    time: str,
    rodada: int,
    *,
    prop_casa: float,
    forca: float,
    forma: float,
    descanso: float,
    importante: str,
    times_ref: list[str],
) -> np.ndarray:
    r = float(rodada)
    team = _encode_team_dummies([time], times_ref)[0]
    base = np.array([r, r * r, forma, forca, prop_casa, descanso], dtype=float)
    copa = np.array(
        [_importante_para_colunas(importante)[c] for c in COPA_COLS], dtype=float
    )
    return np.concatenate([team, base, copa])


def prever_acumulado_modelo(
    modelo: ModeloAcumuladoAjustado,
    time: str,
    rodada: int,
    *,
    prop_casa: float,
    forca: float,
    forma: float,
    descanso: float,
    importante: str,
) -> float:
    x = _vetor_features_projecao(
        time,
        rodada,
        prop_casa=prop_casa,
        forca=forca,
        forma=forma,
        descanso=descanso,
        importante=importante,
        times_ref=modelo.painel.times,
    )
    if modelo.tipo == "kalman":
        assert isinstance(modelo.state, KalmanRegressor)
        return max(modelo.state.predict(x), 0.0)
    if modelo.tipo == "xgboost":
        return max(float(modelo.state.predict(x.reshape(1, -1))[0]), 0.0)
    # GAM (splines + ridge)
    st = modelo.state
    n_team = st["n_team"]
    n_cont = st["n_cont"]
    cont = x[n_team : n_team + n_cont]
    rest = x[n_team + n_cont :]
    X_spline = st["splines"].transform(cont[:2].reshape(1, -1))
    X_full = np.column_stack([X_spline, cont[2:].reshape(1, -1), rest.reshape(1, -1)])
    return max(float(st["ridge"].predict(X_full)[0]), 0.0)


def coeficientes_para_base(
    modelo: ModeloAcumuladoAjustado,
    nome_modelo: str,
    janela_lbl: str,
) -> pd.DataFrame:
    """Serializa coeficientes para a base semanal (aplicação = produto matricial)."""
    rows: list[dict] = []
    if modelo.tipo == "kalman" and isinstance(modelo.state, KalmanRegressor):
        for name, val in zip(modelo.painel.feature_names, modelo.state.x):
            rows.append(
                {
                    "Modelo": nome_modelo,
                    "Janela": janela_lbl,
                    "Time": name[3:-1] if name.startswith("FE[") else "",
                    "Variável": name,
                    "Beta": round(float(val), 6),
                }
            )
    elif modelo.tipo == "gam" and isinstance(modelo.state, dict):
        ridge = modelo.state.get("ridge")
        if ridge is not None and hasattr(ridge, "coef_"):
            names = modelo.painel.feature_names
            n_spl = modelo.state["splines"].n_features_out_
            for i, val in enumerate(ridge.coef_):
                label = names[i] if i < len(names) else f"spline_{i}"
                rows.append(
                    {
                        "Modelo": nome_modelo,
                        "Janela": janela_lbl,
                        "Time": "",
                        "Variável": label,
                        "Beta": round(float(val), 6),
                    }
                )
    elif modelo.tipo == "xgboost" and modelo.state is not None:
        for name, val in zip(
            modelo.painel.feature_names, modelo.state.feature_importances_
        ):
            rows.append(
                {
                    "Modelo": nome_modelo,
                    "Janela": janela_lbl,
                    "Time": name[3:-1] if name.startswith("FE[") else "",
                    "Variável": name,
                    "Beta": round(float(val), 6),
                    "Tipo": "importancia",
                }
            )
    return pd.DataFrame(rows)


def exportar_coefs_regressao_fe(
    jogos: list,
    r_ini: int,
    r_fim: int,
    janela: JanelaTreino,
    *,
    ano_calendario: int,
    janela_lbl: str,
    nome_modelo: str = "Regressão FE",
) -> pd.DataFrame:
    from brasileirao_sheet_names import COL_SECAO
    from brasileirao_secoes import LABEL_FE
    from brasileirao_projecao_core import (
        ajustar_painel_efeitos_fixos,
        coeficientes_efeitos_fixos_por_time,
        mapa_forca_adversario,
    )

    fm = mapa_forca_adversario(jogos, r_ini, r_fim)
    painel = ajustar_painel_efeitos_fixos(
        jogos,
        r_ini,
        r_fim,
        "completa",
        fm,
        janela=janela,
        ano_calendario=ano_calendario,
    )
    por_time = coeficientes_efeitos_fixos_por_time(painel)
    rows: list[dict] = []
    for time, b in por_time.items():
        for term in b.get("termos", []):
            rows.append(
                {
                    "Modelo": nome_modelo or LABEL_FE,
                    COL_SECAO: janela_lbl,
                    "Janela": janela_lbl,
                    "Time": time,
                    "Variável": term.get("Variável"),
                    "Beta": term.get("Beta"),
                    "p-valor": term.get("p-valor"),
                    "R²": b.get("r2"),
                }
            )
    return pd.DataFrame(rows)


def tabela_resumo_modelo(modelo: ModeloAcumuladoAjustado) -> pd.DataFrame:
    """Resumo compacto para o app."""
    rows = [
        {"Campo": "Modelo", "Valor": NOME_MODELO_ACUMULADO[modelo.tipo]},
        {"Campo": "Janela", "Valor": modelo.janela},
        {"Campo": "N observações", "Valor": modelo.painel.n_obs},
        {"Campo": "R²", "Valor": modelo.r2},
    ]
    if modelo.meta.get("fallback"):
        rows.append({"Campo": "Nota", "Valor": modelo.meta["fallback"]})
    if modelo.tipo == "kalman" and isinstance(modelo.state, KalmanRegressor):
        coefs = modelo.state.x
        for name, val in zip(modelo.painel.feature_names, coefs):
            if name.startswith("FE["):
                continue
            rows.append({"Campo": name, "Valor": round(float(val), 4)})
    return pd.DataFrame(rows)


def aplicar_projecoes_modelo_acumulado(
    jogos: list,
    tipo: TipoModeloAcumulado,
    janela: JanelaTreino,
    r_ini: int,
    r_fim: int,
    *,
    blocos_extra: list[list] | None = None,
    ano_calendario: int = 2026,
) -> tuple[list, pd.DataFrame]:
    """Projeta rodada a rodada com Kalman, XGBoost ou GAM."""
    from brasileirao_projecao_core import (
        DELTA_PTS_MAX_POR_RODADA,
        Jogo,
        _acum_proj_ate,
        _contexto_projecao_acumulada,
        _forma_misturada_projecao,
        mapa_forca_adversario,
        stats_acumuladas_ate,
        times_do_calendario,
    )

    jogos = [Jogo(**j.__dict__) for j in jogos]
    times = times_do_calendario(jogos)
    if blocos_extra is None:
        try:
            from brasileirao_multi_liga import carregar_blocos_treino_regressao

            blocos_extra = carregar_blocos_treino_regressao(calendar_teams=set(times))
        except Exception:
            blocos_extra = []

    painel = coletar_painel_treino(
        jogos,
        janela,
        blocos_extra=blocos_extra,
        ano_calendario=ano_calendario,
    )
    modelo = ajustar_modelo_acumulado(painel, tipo, janela)
    forca_map = mapa_forca_adversario(jogos, r_ini, r_fim)

    acum_cache: dict[str, dict[int, float]] = {t: {0: 0.0} for t in times}
    for t in times:
        for r in range(1, r_fim + 1):
            acum_cache[t][r] = float(
                stats_acumuladas_ate(jogos, t, r, so_realizados=True).pts
            )

    from recency import JANELA_TREINO_LABELS

    label = NOME_MODELO_ACUMULADO[tipo]
    janela_lbl = JANELA_TREINO_LABELS.get(janela, str(janela))
    origem = f"{label} ({janela_lbl})"
    log_rows: list[dict] = []
    rodadas_pendentes = sorted({j.r for j in jogos if not j.jogado})
    ult_r_real = max((j.r for j in jogos if j.jogado), default=r_fim)

    for r in rodadas_pendentes:
        jogos_r = [j for j in jogos if j.r == r and not j.jogado]
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
            forma_m = _forma_misturada_projecao(
                jogos, m, j, r_ini=r_ini, r_fim_obs=r_fim, horizonte=horizonte
            )
            forma_v = _forma_misturada_projecao(
                jogos, v, j, r_ini=r_ini, r_fim_obs=r_fim, horizonte=horizonte
            )

            target_m = prever_acumulado_modelo(
                modelo, m, r,
                prop_casa=prop_m, forca=forca_m, forma=forma_m,
                descanso=desc_m, importante=imp_m,
            )
            target_v = prever_acumulado_modelo(
                modelo, v, r,
                prop_casa=prop_v, forca=forca_v, forma=forma_v,
                descanso=desc_v, importante=imp_v,
            )

            delta_m = min(DELTA_PTS_MAX_POR_RODADA, max(0.0, target_m - prev_m))
            delta_v = min(DELTA_PTS_MAX_POR_RODADA, max(0.0, target_v - prev_v))
            j.proj_pm, j.proj_pv = delta_m, delta_v
            j.origem = origem
            acum_cache[m][r] = prev_m + delta_m
            acum_cache[v][r] = prev_v + delta_v
            forca_map = mapa_forca_adversario(jogos, r_ini, r_fim)

            log_rows.append(
                {
                    "Rodada": r,
                    "Mandante": m,
                    "Visitante": v,
                    "Proj": f"{delta_m:.2f} / {delta_v:.2f}",
                    "Modelo": origem,
                }
            )

    return jogos, pd.DataFrame(log_rows)
