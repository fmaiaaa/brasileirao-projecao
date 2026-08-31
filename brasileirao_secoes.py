"""Seções de projeção, rótulos de modelos e mapeamento modo ↔ base."""
from __future__ import annotations

from recency import JANELA_TREINO_LABELS, JanelaTreino

# Rótulos na base (colunas Modelo / Seção)
LABEL_TURNO = "Repetir primeiro turno"
LABEL_MEDIA = "Modelo de médias"
LABEL_FE = "Regressão FE"
LABEL_KALMAN = "Modelo de Kalman (Espaço de Estados)"
LABEL_XGB = "XGBoost"
LABEL_GAM = "GAM"
LABEL_PROB = "Probabilístico"

SECOES_ORDEM: list[JanelaTreino] = (
    "2026",
    "ultimas_38_rodadas",
    "ultimos_3_anos",
    "base_completa",
)

SECAO_LABEL: dict[JanelaTreino, str] = {
    k: JANELA_TREINO_LABELS[k] for k in SECOES_ORDEM  # type: ignore[misc]
}

MODO_PARA_LABEL: dict[str, str] = {
    "repetir_turno": LABEL_TURNO,
    "media_simples": LABEL_MEDIA,
    "regressao_completa": LABEL_FE,
    "kalman_acumulada": LABEL_KALMAN,
    "xgboost_acumulada": LABEL_XGB,
    "gam_acumulada": LABEL_GAM,
    "prob_ml": LABEL_PROB,
}

LABEL_PARA_MODO: dict[str, str] = {v: k for k, v in MODO_PARA_LABEL.items()}

MODELOS_SECAO_PADRAO: list[tuple[str, str]] = [
    (LABEL_TURNO, "repetir_turno"),
    (LABEL_MEDIA, "media_simples"),
    (LABEL_FE, "regressao_completa"),
    (LABEL_KALMAN, "kalman_acumulada"),
    (LABEL_XGB, "xgboost_acumulada"),
    (LABEL_GAM, "gam_acumulada"),
    (LABEL_PROB, "prob_ml"),
]

MODELOS_SEM_TURNO: list[tuple[str, str]] = [
    m for m in MODELOS_SECAO_PADRAO if m[1] != "repetir_turno"
]

# Janelas com dummy Série A × Série B + interação time×série
JANELAS_COM_SERIE_AB: frozenset[JanelaTreino] = frozenset(
    {"ultimas_38_rodadas", "ultimos_3_anos", "base_completa"}  # type: ignore[arg-type]
)


def modelos_da_secao(janela: JanelaTreino) -> list[tuple[str, str]]:
    if janela == "2026":
        return list(MODELOS_SECAO_PADRAO)
    return list(MODELOS_SEM_TURNO)


def usa_features_serie_ab(janela: JanelaTreino | None) -> bool:
    return janela is not None and janela in JANELAS_COM_SERIE_AB
