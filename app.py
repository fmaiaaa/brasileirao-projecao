"""App Streamlit - Projeção Brasileirão 2026."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from brasileirao_estilo import (
    aplicar_estilo,
    bloquear_graficos_mobile,
    bloco_classificacao_time,
    cabecalho_pagina,
    kpi_row,
    rodape_desenvolvedor,
    titulo_grafico,
    titulo_secao,
)
from brasileirao_projecao_core import (
    Jogo,
    ModoProjecao,
    TipoRegressao,
    modo_e_regressao_acumulada,
    aplicar_projecoes,
    carregar_jogos,
    colunas_estatisticas_grafico,
    estatisticas_por_rodada,
    projetar_estatisticas_por_rodada,
    colunas_estatisticas_rodada_grafico,
    evolucao_pontos_time,
    evolucao_posicao_time,
    fig_estatisticas_por_rodada,
    fig_estatisticas_times,
    fig_evolucao_times,
    fig_evolucao_posicao_times,
    extrair_itens_legenda,
    titulo_fig,
    kpis_globais,
    mapa_posicao_pontos,
    probabilidades_cenarios_finais,
    tabela_jogos_primeiro_turno,
    tabela_comparativa_posicoes,
    tabela_estatisticas_times,
    times_do_calendario,
)
from brasileirao_weekly_base import (
    anos_disponiveis_estatisticas,
    aplicar_projecoes_csv_com_gap,
    enriquecer_stats_com_contexto,
    jogos_serie_a_ano,
    load_classif_modelos_acum,
    load_coefs_modelos_acum,
    load_forecasts_modelos_acum,
    load_prob_metricas,
    load_projecoes_modelos_acum,
    load_resumo_modelos_acum,
)
from recency import JANELA_TREINO_LABELS, JanelaTreino
from brasileirao_secoes import (
    LABEL_MEDIA,
    LABEL_PROB,
    LABEL_TURNO,
    MODELOS_SECAO_PADRAO,
    PRAZO_CURTO,
    PRAZO_PIPE_LEGENDA,
    SECAO_LABEL,
    SECOES_ORDEM,
)

_PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": False,
    "displayModeBar": False,
    "editable": False,
    "showTips": False,
}

_DEFAULT_TIMES_GRAF = ("Palmeiras", "Flamengo", "Athletico-PR", "Cruzeiro")
_DEFAULT_METRICA_GRAF = "Média gols marcados/Média gols sofridos"


def _bloquear_interacao_fig(fig) -> None:
    """Desativa zoom/pan/seleção no Plotly (reforço; mobile também via CSS/JS)."""
    fig.update_layout(
        dragmode=False,
        clickmode="none",
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)


def _tabela(df, *, column_config=None, key=None) -> None:
    """Tabela com sort e scroll, sem rearranjar colunas (arrastar)."""
    kwargs = dict(
        use_container_width=True,
        hide_index=True,
        # No Streamlit, seleção de coluna desliga o drag-and-drop de colunas
        selection_mode="multi-column",
        on_select="ignore",
    )
    if column_config is not None:
        kwargs["column_config"] = column_config
    if key is not None:
        kwargs["key"] = key
    st.dataframe(df, **kwargs)


def _html_legenda_mobile(itens: list[dict[str, str]]) -> str:
    linhas = []
    for item in itens:
        nome = html.escape(item["nome"])
        cor = html.escape(item["cor"])
        linhas.append(
            f'<div class="grafico-legenda-item">'
            f'<span class="grafico-legenda-swatch" style="background:{cor}"></span>'
            f"<span>{nome}</span></div>"
        )
    return '<div class="grafico-legenda-list">' + "".join(linhas) + "</div>"


def _grafico(fig) -> None:
    titulo = titulo_fig(fig)
    if titulo:
        titulo_grafico(titulo)
    _bloquear_interacao_fig(fig)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)
    bloquear_graficos_mobile()
    itens = extrair_itens_legenda(fig)
    if not itens:
        return
    with st.expander("Legendas", expanded=False):
        st.markdown(_html_legenda_mobile(itens), unsafe_allow_html=True)


@st.cache_data(ttl=120, show_spinner="Carregando planilha…")
def _carregar_dados_cached() -> tuple[list, object | None]:
    jogos, _, df = carregar_jogos(preferir_gsheets=True)
    return jogos, df


@st.cache_data(ttl=600, show_spinner="Carregando temporada…")
def _jogos_historicos_cached(ano: int) -> list:
    return jogos_serie_a_ano(int(ano))


_ANO_CALENDARIO = 2026

st.set_page_config(
    page_title="Projeção Brasileirão 2026",
    layout="wide",
    initial_sidebar_state="collapsed",
)
aplicar_estilo()
bloquear_graficos_mobile()
cabecalho_pagina("Projeção Brasileirão 2026")

try:
    _jogos_base, _df_fonte = _carregar_dados_cached()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

_anos_stats = anos_disponiveis_estatisticas(ano_atual=_ANO_CALENDARIO)
ano_stats = st.selectbox(
    "Temporada (estatísticas)",
    options=_anos_stats,
    index=0,
    help=(
        "2026 usa a planilha de resultados. "
        "Anos anteriores usam a Série A da aba Base_Contexto (base semanal)."
    ),
    key="stats_ano",
)

if int(ano_stats) == _ANO_CALENDARIO:
    _jogos_stats = _jogos_base
    _fonte_stats = "calendário / planilha de resultados"
else:
    _jogos_stats = _jogos_historicos_cached(int(ano_stats))
    _fonte_stats = f"Base_Contexto · Série A {ano_stats}"
    if not _jogos_stats:
        st.warning(
            f"Sem jogos da Série A {ano_stats} na Base_Contexto. "
            "Atualize a base semanal (segunda 03:00)."
        )

_times = times_do_calendario(_jogos_stats)
_jogados = sum(1 for j in _jogos_stats if j.jogado)
_pendentes = len(_jogos_stats) - _jogados
_ult_r = max((j.r for j in _jogos_stats if j.jogado), default=1)
_r_max_stats = max((j.r for j in _jogos_stats), default=38)

_kpi = kpis_globais(_jogos_stats)

st.caption(f"Estatísticas: temporada **{ano_stats}** ({_fonte_stats})")

kpi_row([
    ("Jogos realizados", str(_jogados), False),
    ("Jogos pendentes", str(_pendentes), True),
    ("Rodada atual", str(_ult_r), False),
])
kpi_row([
    ("Média gols / jogo", f"{_kpi['media_gols_jogo']:.2f}", False),
    ("Média gols mandantes", f"{_kpi['media_gols_mandante']:.2f}", False),
    ("Média gols visitantes", f"{_kpi['media_gols_visitante']:.2f}", False),
])
kpi_row([
    ("Média pts mandantes", f"{_kpi['media_pts_mandante']:.2f}", False),
    ("Média pts visitantes", f"{_kpi['media_pts_visitante']:.2f}", False),
])

titulo_secao("Estatísticas por time (intervalo selecionado)")

c_stats1, c_stats2 = st.columns(2)
with c_stats1:
    r_ini_stats = st.number_input(
        "Rodada início (estatísticas)",
        min_value=1,
        max_value=int(max(1, _r_max_stats)),
        value=1,
        step=1,
        key="stats_r_ini",
    )
with c_stats2:
    r_fim_stats = st.number_input(
        "Rodada fim (estatísticas)",
        min_value=1,
        max_value=int(max(1, _r_max_stats)),
        value=int(min(_ult_r, _r_max_stats)),
        step=1,
        key="stats_r_fim",
    )
if r_fim_stats < r_ini_stats:
    st.warning("Rodada fim (estatísticas) menor que início - usando fim = início.")
    r_fim_stats = r_ini_stats

_df_stats = tabela_estatisticas_times(_jogos_stats, int(r_ini_stats), int(r_fim_stats))
_df_stats = enriquecer_stats_com_contexto(_df_stats, int(ano_stats))

_tabela(
    _df_stats,
    column_config={
        "Time": st.column_config.TextColumn("Time", pinned="left", width="medium"),
    },
    key="tbl_stats",
)

titulo_secao("Gráfico de estatísticas")
_cols_stats = colunas_estatisticas_grafico(_df_stats)
_default_times = [t for t in _DEFAULT_TIMES_GRAF if t in _times]
_default_stats = [c for c in (_DEFAULT_METRICA_GRAF,) if c in _cols_stats]

c_graf1, c_graf2 = st.columns(2)
with c_graf1:
    times_stats_graf = st.multiselect(
        "Times no gráfico",
        options=_times,
        default=list(_times),
        key="stats_graf_times",
    )
with c_graf2:
    metricas_graf = st.multiselect(
        "Estatísticas no gráfico",
        options=_cols_stats,
        default=_default_stats,
        key="stats_graf_metricas",
    )

_ORD_MAIOR = "Ordenar do maior para o menor"
_ORD_MENOR = "Ordenar do menor para o maior"
_ORD_ALPHA = "Ordem alfabética"
ordem_stats_label = st.radio(
    "Ordenação do gráfico",
    options=[_ORD_MAIOR, _ORD_MENOR, _ORD_ALPHA],
    index=2,
    horizontal=True,
    key="stats_graf_ordem",
)
if ordem_stats_label == _ORD_MAIOR:
    ordem_stats = "maior_menor"
elif ordem_stats_label == _ORD_MENOR:
    ordem_stats = "menor_maior"
else:
    ordem_stats = "alfabetica"

if times_stats_graf and metricas_graf:
    _grafico(
        fig_estatisticas_times(
            _df_stats,
            times_stats_graf,
            metricas_graf,
            ordenacao=ordem_stats,
        )
    )
else:
    st.info("Selecione ao menos um time e uma estatística para exibir o gráfico.")

_r_fim_rodada = int(min(int(r_fim_stats), int(_ult_r)))
_df_stats_rodada = estatisticas_por_rodada(
    _jogos_stats, int(r_ini_stats), _r_fim_rodada
)

titulo_secao("Gráfico de estatísticas por rodada")
_cols_stats_rodada = colunas_estatisticas_rodada_grafico(_df_stats_rodada)
_default_stats_rodada = [
    c for c in (_DEFAULT_METRICA_GRAF,) if c in _cols_stats_rodada
]

c_graf_r1, c_graf_r2 = st.columns(2)
with c_graf_r1:
    times_rodada_graf = st.multiselect(
        "Times no gráfico",
        options=_times,
        default=_default_times if _default_times else list(_times)[:4],
        key="stats_rodada_times",
    )
with c_graf_r2:
    metricas_rodada_graf = st.multiselect(
        "Estatísticas no gráfico",
        options=_cols_stats_rodada,
        default=_default_stats_rodada,
        key="stats_rodada_metricas",
    )

_ano_atual_stats = int(ano_stats) == _ANO_CALENDARIO
incluir_projecao_rodada = False
if _ano_atual_stats:
    incluir_projecao_rodada = st.toggle(
        "Incluir projeções no gráfico",
        value=False,
        key="stats_rodada_projecao",
    )
else:
    st.caption("Projeções no gráfico só na temporada atual (calendário).")

if times_rodada_graf and metricas_rodada_graf:
    if incluir_projecao_rodada:
        _df_graf_rodada = projetar_estatisticas_por_rodada(
            _jogos_stats,
            _df_stats_rodada,
            int(r_ini_stats),
            _r_fim_rodada,
            metricas_rodada_graf,
        )
    else:
        _df_graf_rodada = _df_stats_rodada
    _grafico(
        fig_estatisticas_por_rodada(
            _df_graf_rodada,
            times_rodada_graf,
            metricas_rodada_graf,
            r_atual=_r_fim_rodada,
        )
    )
else:
    st.info("Selecione ao menos um time e uma estatística para exibir o gráfico.")

with st.expander("Dados carregados"):
    if int(ano_stats) == _ANO_CALENDARIO and _df_fonte is not None:
        _tabela(_df_fonte, key="tbl_fonte")
    elif _jogos_stats:
        _tabela(
            pd.DataFrame(
                [
                    {
                        "Rodada": j.r,
                        "Data": j.data,
                        "Mandante": j.mand,
                        "Placar": j.placar,
                        "Visitante": j.vis,
                    }
                    for j in _jogos_stats
                ]
            ),
            key="tbl_fonte_hist",
        )
    else:
        st.info("Sem preview tabular.")

# Ranking projetado ano a ano (pré-computado, sem leakage) — só leitura
_rk_ano = None
try:
    from brasileirao_weekly_base import load_sheet as _load_sheet_rk

    _rk_ano = _load_sheet_rk(str(int(ano_stats)))
except Exception:
    _rk_ano = None
if _rk_ano is not None and not _rk_ano.empty:
    with st.expander(
        f"Ranking projetado {ano_stats} (Rodada 19→38, sem leakage)",
        expanded=False,
    ):
        st.caption(
            "Ranking Final = real ao fim. "
            "Rodada N = posição final média estimada só com dados até a rodada N "
            "(treino sem anos futuros nem 2º turno da temporada-alvo)."
        )
        _tabela(_rk_ano, key="tbl_ranking_ano")

# Projeção sempre no calendário atual (ano do app)
_times_proj = times_do_calendario(_jogos_base)
_ult_r_proj = (
    max(j.r for j in _jogos_base if j.jogado)
    if any(j.jogado for j in _jogos_base)
    else 1
)

titulo_secao("Projeção")
st.caption(
    f"Calendário {_ANO_CALENDARIO}. Escolha o modelo e compare os quatro prazos de treino "
    f"na mesma tabela ({PRAZO_PIPE_LEGENDA})."
)

r_ini_proj = 1
r_fim_proj = int(min(_ult_r_proj, 38))


def _fmt_r2_ui(v):
    if v is None or str(v).strip() == "":
        return ""
    s = str(v).strip().lstrip("'")
    try:
        x = float(s.replace(",", "."))
    except (TypeError, ValueError):
        return s
    if x > 1.0:
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            x = float("0." + digits[:4].ljust(4, "0")[:4])
    return f"{x:.4f}"


def _pipe_valores(valores: list[str]) -> str:
    return " | ".join(valores)


def _aplicar_modelo_prazo(
    modo: str,
    modelo_lbl: str,
    secao_key: JanelaTreino,
    secao_lbl: str,
) -> tuple[list[Jogo], pd.DataFrame]:
    """Carrega projeções da base para um modelo × prazo."""
    if modo == "prob_ml":
        cal = load_projecoes_modelos_acum(LABEL_PROB, secao=secao_lbl)
        if cal is None or cal.empty:
            return [Jogo(**j.__dict__) for j in _jogos_base], pd.DataFrame()
        return aplicar_projecoes_csv_com_gap(
            _jogos_base,
            cal,
            r_ini=r_ini_proj,
            r_fim=r_fim_proj,
            origem=f"prob_ml/{secao_lbl}",
        )
    if modo == "media_simples":
        cal = load_projecoes_modelos_acum(LABEL_MEDIA, secao=secao_lbl)
        if cal is None or cal.empty:
            return [Jogo(**j.__dict__) for j in _jogos_base], pd.DataFrame()
        return aplicar_projecoes_csv_com_gap(
            _jogos_base,
            cal,
            r_ini=r_ini_proj,
            r_fim=r_fim_proj,
            origem=f"media/{secao_lbl}",
        )
    if modo_e_regressao_acumulada(modo):  # type: ignore[arg-type]
        cal = load_projecoes_modelos_acum(modelo_lbl, secao=secao_lbl)
        if cal is None or cal.empty:
            return [Jogo(**j.__dict__) for j in _jogos_base], pd.DataFrame()
        return aplicar_projecoes_csv_com_gap(
            _jogos_base,
            cal,
            r_ini=r_ini_proj,
            r_fim=r_fim_proj,
            origem=f"modelos_acum/{modelo_lbl}",
        )
    if modo == "repetir_turno":
        if secao_key != "2026":
            return [Jogo(**j.__dict__) for j in _jogos_base], pd.DataFrame()
        return aplicar_projecoes(
            _jogos_base,
            "repetir_turno",
            r_ini_proj,
            r_fim_proj,
            "mandante_visitante",
            janela="2026",
            ano_calendario=_ANO_CALENDARIO,
        )
    return [Jogo(**j.__dict__) for j in _jogos_base], pd.DataFrame()


def _tabela_classif_multi_prazo(
    jogos_base: list[Jogo],
    por_prazo: dict[JanelaTreino, list[Jogo]],
    *,
    prob_por_prazo: dict[JanelaTreino, pd.DataFrame | None] | None = None,
) -> pd.DataFrame:
    """Classificação com posição/pts/probs de todos os prazos na mesma linha (pipe)."""
    from prob_ml.integration import _safe_float as _sf

    mapa_atual = mapa_posicao_pontos(jogos_base, incluir_proj=False)
    rows: list[dict] = []
    for time in times_do_calendario(jogos_base):
        pa, pta = mapa_atual.get(time, (0, 0))
        pos_parts: list[str] = []
        pts_parts: list[str] = []
        prob_c: list[str] = []
        prob_g4: list[str] = []
        prob_g6: list[str] = []
        prob_z4: list[str] = []
        for sk in SECOES_ORDEM:
            jogos_p = por_prazo.get(sk)
            if jogos_p is None:
                pos_parts.append("—")
                pts_parts.append("—")
                prob_c.append("—")
                prob_g4.append("—")
                prob_g6.append("—")
                prob_z4.append("—")
                continue
            mp = mapa_posicao_pontos(jogos_p, incluir_proj=True)
            pf, ptf = mp.get(time, (None, None))
            if pf is None:
                pos_parts.append("—")
                pts_parts.append("—")
            else:
                pos_parts.append(f"{pf}º")
                pts_parts.append(
                    f"{ptf:.1f}" if abs(ptf - round(ptf)) > 0.05 else str(int(round(ptf)))
                )
            mc = None
            if prob_por_prazo and prob_por_prazo.get(sk) is not None:
                mc_df = prob_por_prazo[sk]
                if mc_df is not None and not mc_df.empty and "Time" in mc_df.columns:
                    mc = mc_df.set_index("Time") if "Time" in mc_df.columns else mc_df
            if mc is not None and time in mc.index:
                prob_c.append(f"{_sf(mc.loc[time].get('Prob. Campeão'), 0):.1f}%")
                prob_g4.append(f"{_sf(mc.loc[time].get('Prob. G4'), 0):.1f}%")
                prob_g6.append(f"{_sf(mc.loc[time].get('Prob. G6'), 0):.1f}%")
                prob_z4.append(f"{_sf(mc.loc[time].get('Prob. Z4'), 0):.1f}%")
            else:
                pr = probabilidades_cenarios_finais(jogos_p).get(time, {})
                prob_c.append(f"{100 * pr.get('campeao', 0):.1f}%")
                prob_g4.append(f"{100 * pr.get('g4', 0):.1f}%")
                prob_g6.append(f"{100 * pr.get('g6', 0):.1f}%")
                prob_z4.append(f"{100 * pr.get('z4', 0):.1f}%")
        rows.append(
            {
                "Time": time,
                "Posição Atual": pa,
                f"Posição ({PRAZO_PIPE_LEGENDA})": _pipe_valores(pos_parts),
                f"Pts ({PRAZO_PIPE_LEGENDA})": _pipe_valores(pts_parts),
                f"Prob. Campeão ({PRAZO_PIPE_LEGENDA})": _pipe_valores(prob_c),
                f"Prob. G4 ({PRAZO_PIPE_LEGENDA})": _pipe_valores(prob_g4),
                f"Prob. G6 ({PRAZO_PIPE_LEGENDA})": _pipe_valores(prob_g6),
                f"Prob. Z4 ({PRAZO_PIPE_LEGENDA})": _pipe_valores(prob_z4),
            }
        )
    ref_key = next((k for k in SECOES_ORDEM if k in por_prazo), SECOES_ORDEM[0])
    ref_pos = {
        t: mapa_posicao_pontos(por_prazo[ref_key], incluir_proj=True).get(t, (999, 0))[0]
        for t in times_do_calendario(jogos_base)
    }
    df = pd.DataFrame(rows)
    df["_ord"] = df["Time"].map(lambda t: ref_pos.get(t, 999))
    return df.sort_values("_ord").drop(columns="_ord")


def _tabela_jogos_multi_prazo(logs: dict[JanelaTreino, pd.DataFrame]) -> pd.DataFrame:
    """Junta projeções de jogos por prazo (coluna Proj com pipe)."""
    if not logs:
        return pd.DataFrame()
    keys = ["Rodada", "Mandante", "Visitante"]
    merged: pd.DataFrame | None = None
    for sk in SECOES_ORDEM:
        df = logs.get(sk)
        if df is None or df.empty:
            continue
        part = df[keys + ["Proj"]].copy()
        part = part.rename(columns={"Proj": PRAZO_CURTO[sk]})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=keys, how="outer")
    if merged is None or merged.empty:
        return pd.DataFrame()
    cols_prazo = [PRAZO_CURTO[sk] for sk in SECOES_ORDEM if PRAZO_CURTO[sk] in merged.columns]
    merged["Proj"] = merged[cols_prazo].apply(
        lambda r: _pipe_valores([str(v) if pd.notna(v) else "—" for v in r]),
        axis=1,
    )
    return merged[keys + ["Proj"]].sort_values(["Rodada", "Mandante"])


modelo_lbl = st.radio(
    "Modelo",
    options=[m[0] for m in MODELOS_SECAO_PADRAO],
    horizontal=True,
    key="proj_modelo",
)
modo = next(m[1] for m in MODELOS_SECAO_PADRAO if m[0] == modelo_lbl)

with st.expander("Detalhes do modelo"):
    for sk in SECOES_ORDEM:
        sec_lbl = SECAO_LABEL[sk]
        st.markdown(f"**{sec_lbl}**")
        if modo_e_regressao_acumulada(modo):  # type: ignore[arg-type]
            _resumo = load_resumo_modelos_acum(modelo_lbl, secao=sec_lbl)
            if _resumo is not None and not _resumo.empty:
                st.caption(
                    f"R² = {_resumo.iloc[0].get('R²', '—')} · "
                    f"N obs = {_resumo.iloc[0].get('N observações', '—')}"
                )
            _coefs = load_coefs_modelos_acum(modelo_lbl, secao=sec_lbl)
            if _coefs is not None and not _coefs.empty:
                _coefs_show = _coefs.copy()
                for _c in list(_coefs_show.columns):
                    if str(_c).strip().lower().replace("²", "2") in {"r2", "r²"}:
                        _coefs_show[_c] = _coefs_show[_c].map(_fmt_r2_ui)
                        break
                _tabela(_coefs_show.head(40), key=f"coef_{sk}")
        elif modo == "media_simples":
            _resumo = load_resumo_modelos_acum(LABEL_MEDIA, secao=sec_lbl)
            if _resumo is not None and not _resumo.empty:
                st.caption(f"N obs = {_resumo.iloc[0].get('N observações', '—')}")
        elif modo == "prob_ml":
            _resumo = load_resumo_modelos_acum(LABEL_PROB, secao=sec_lbl)
            if _resumo is not None and not _resumo.empty:
                st.caption(
                    f"N obs = {_resumo.iloc[0].get('N observações', '—')} · "
                    f"R² = {_resumo.iloc[0].get('R²', '—')}"
                )
        elif modo == "repetir_turno" and sk == "2026":
            _tabela(tabela_jogos_primeiro_turno(_jogos_base), key="tbl_turno_ref")

por_prazo: dict[JanelaTreino, list[Jogo]] = {}
logs_prazo: dict[JanelaTreino, pd.DataFrame] = {}
prob_stands: dict[JanelaTreino, pd.DataFrame | None] = {}
_prob_fc: dict[JanelaTreino, pd.DataFrame | None] = {}
_faltando = False
for sk in SECOES_ORDEM:
    if modo == "repetir_turno" and sk != "2026":
        continue
    sec_lbl = SECAO_LABEL[sk]
    jogos_p, df_log = _aplicar_modelo_prazo(modo, modelo_lbl, sk, sec_lbl)
    por_prazo[sk] = jogos_p
    if not df_log.empty:
        logs_prazo[sk] = df_log
    if modo == "prob_ml":
        stand = load_classif_modelos_acum(LABEL_PROB, secao=sec_lbl)
        prob_stands[sk] = (
            stand.drop(columns=["Modelo", "Seção", "Janela"], errors="ignore")
            if stand is not None and not stand.empty
            else None
        )
        _prob_fc[sk] = load_forecasts_modelos_acum(LABEL_PROB, secao=sec_lbl)
        if load_projecoes_modelos_acum(LABEL_PROB, secao=sec_lbl) is None:
            _faltando = True
    elif modo == "media_simples":
        if load_projecoes_modelos_acum(LABEL_MEDIA, secao=sec_lbl) is None:
            _faltando = True
    elif modo_e_regressao_acumulada(modo):  # type: ignore[arg-type]
        if load_projecoes_modelos_acum(modelo_lbl, secao=sec_lbl) is None:
            _faltando = True

if _faltando:
    st.warning(
        "Algumas projeções ainda não estão na base para todos os prazos. "
        "Execute scripts/weekly_retrain.py."
    )

st.markdown("#### Classificação")
_df_classif = _tabela_classif_multi_prazo(
    _jogos_base,
    por_prazo,
    prob_por_prazo=prob_stands if modo == "prob_ml" else None,
)
_tabela(
    _df_classif,
    column_config={
        "Time": st.column_config.TextColumn("Time", pinned="left"),
        "Posição Atual": st.column_config.NumberColumn("Posição Atual"),
    },
    key="tbl_classif_multi",
)

with st.expander("Jogos projetados"):
    _df_jogos = _tabela_jogos_multi_prazo(logs_prazo)
    if _df_jogos.empty:
        st.success("Todos os jogos já têm placar — nada a projetar.")
    else:
        st.caption(f"Projeção por jogo ({PRAZO_PIPE_LEGENDA})")
        _tabela(_df_jogos, key="tbl_jogos_multi")

if modo == "prob_ml":
    with st.expander("Model Lab — previsões de placar (38 rodadas)", expanded=False):
        fc38 = _prob_fc.get("ultimas_38_rodadas")
        if fc38 is not None and not fc38.empty:
            pend_pairs = {(j.mand, j.vis) for j in _jogos_base if not j.jogado}
            from prob_ml.integration import _safe_float as _sf

            _lab = []
            for p in fc38.to_dict(orient="records"):
                ht = str(p.get("home_team") or "")
                at = str(p.get("away_team") or "")
                if (ht, at) not in pend_pairs:
                    continue
                tops = p.get("top_scores")
                if isinstance(tops, str):
                    try:
                        import json as _json

                        tops = _json.loads(tops)
                    except Exception:
                        tops = []
                tops = tops or []
                tops_s = ", ".join(
                    f"{i}-{j} ({pr:.0%})" for i, j, pr in tops[:3]
                )
                _lab.append(
                    {
                        "Rodada": p.get("round"),
                        "Jogo": f"{ht} x {at}",
                        "λ H": round(_sf(p.get("xg_home"), 0), 2),
                        "λ A": round(_sf(p.get("xg_away"), 0), 2),
                        "P(H)": round(100 * _sf(p.get("p_home"), 0), 1),
                        "P(D)": round(100 * _sf(p.get("p_draw"), 0), 1),
                        "P(A)": round(100 * _sf(p.get("p_away"), 0), 1),
                    }
                )
            if _lab:
                _tabela(pd.DataFrame(_lab), key="tbl_model_lab")
        else:
            st.info("Forecasts probabilísticos não disponíveis.")

st.markdown("#### Evolução por rodada")
_modo_graf = st.radio(
    "Visualização",
    options=["Um time · todos os prazos", "Vários times · um prazo"],
    horizontal=True,
    key="graf_modo",
)
mapa_atual = mapa_posicao_pontos(_jogos_base, incluir_proj=False)

if _modo_graf == "Um time · todos os prazos":
    from dataclasses import replace

    _time_graf = st.selectbox(
        "Time",
        options=_times_proj,
        index=0,
        key="graf_time_unico",
    )
    pa, pta = mapa_atual.get(_time_graf, (0, 0))
    final_parts: list[str] = []
    prob_c_parts: list[str] = []
    prob_g4_parts: list[str] = []
    prob_g6_parts: list[str] = []
    prob_z4_parts: list[str] = []
    for sk in SECOES_ORDEM:
        jogos_p = por_prazo.get(sk)
        if jogos_p is None:
            final_parts.append("—")
            prob_c_parts.append("—")
            prob_g4_parts.append("—")
            prob_g6_parts.append("—")
            prob_z4_parts.append("—")
            continue
        pf, ptf = mapa_posicao_pontos(jogos_p, incluir_proj=True).get(_time_graf, (None, None))
        if pf is None:
            final_parts.append("—")
        else:
            pts_s = f"{ptf:.1f}" if abs(ptf - round(ptf)) > 0.05 else str(int(round(ptf)))
            final_parts.append(f"{pf}º · {pts_s} pts")
        if modo == "prob_ml" and prob_stands.get(sk) is not None:
            mc = prob_stands[sk]
            if mc is not None and not mc.empty and _time_graf in mc["Time"].values:
                row = mc.loc[mc["Time"] == _time_graf].iloc[0]
                prob_c_parts.append(f"{row.get('Prob. Campeão', 0):.1f}%")
                prob_g4_parts.append(f"{row.get('Prob. G4', 0):.1f}%")
                prob_g6_parts.append(f"{row.get('Prob. G6', 0):.1f}%")
                prob_z4_parts.append(f"{row.get('Prob. Z4', 0):.1f}%")
            else:
                prob_c_parts.append("—")
                prob_g4_parts.append("—")
                prob_g6_parts.append("—")
                prob_z4_parts.append("—")
        else:
            pr = probabilidades_cenarios_finais(jogos_p).get(_time_graf, {})
            prob_c_parts.append(f"{100 * pr.get('campeao', 0):.1f}%")
            prob_g4_parts.append(f"{100 * pr.get('g4', 0):.1f}%")
            prob_g6_parts.append(f"{100 * pr.get('g6', 0):.1f}%")
            prob_z4_parts.append(f"{100 * pr.get('z4', 0):.1f}%")

    bloco_classificacao_time(
        _time_graf,
        pa,
        pta,
        0,
        0,
        classif_final=_pipe_valores(final_parts),
        prob_campeao=_pipe_valores(prob_c_parts),
        prob_g4=_pipe_valores(prob_g4_parts),
        prob_g6=_pipe_valores(prob_g6_parts),
        prob_z4=_pipe_valores(prob_z4_parts),
    )
    st.caption(f"Ordem dos prazos: {PRAZO_PIPE_LEGENDA}")

    evolucoes = []
    for sk in SECOES_ORDEM:
        if sk not in por_prazo:
            continue
        ev = evolucao_pontos_time(_jogos_base, por_prazo[sk], _time_graf, _ult_r_proj)
        evolucoes.append(replace(ev, time=f"{_time_graf} ({PRAZO_CURTO[sk]})"))
    if evolucoes:
        _grafico(fig_evolucao_times(evolucoes))
        evolucoes_pos = [
            replace(
                evolucao_posicao_time(_jogos_base, por_prazo[sk], _time_graf, _ult_r_proj),
                time=f"{_time_graf} ({PRAZO_CURTO[sk]})",
            )
            for sk in SECOES_ORDEM
            if sk in por_prazo
        ]
        _grafico(fig_evolucao_posicao_times(evolucoes_pos))
else:
    _prazo_graf = st.selectbox(
        "Prazo",
        options=[SECAO_LABEL[sk] for sk in SECOES_ORDEM],
        index=1,
        key="graf_prazo",
    )
    sk_graf = next(k for k, v in SECAO_LABEL.items() if v == _prazo_graf)
    jogos_graf = por_prazo.get(sk_graf, _jogos_base)
    times_graf = st.multiselect(
        "Times para comparar",
        options=_times_proj,
        default=[t for t in ("Palmeiras", "Flamengo", "Cruzeiro") if t in _times_proj],
        key="graf_times_multi",
    )
    if times_graf:
        mapa_final = mapa_posicao_pontos(jogos_graf, incluir_proj=True)
        if modo == "prob_ml" and prob_stands.get(sk_graf) is not None:
            mc = prob_stands[sk_graf]
            probs_finais = {
                str(r["Time"]): {
                    "campeao": float(r["Prob. Campeão"]) / 100.0,
                    "g4": float(r["Prob. G4"]) / 100.0,
                    "g6": float(r["Prob. G6"]) / 100.0,
                    "z4": float(r["Prob. Z4"]) / 100.0,
                }
                for _, r in mc.iterrows()
            } if mc is not None and not mc.empty else {}
        else:
            probs_finais = probabilidades_cenarios_finais(jogos_graf)
        times_graf_ord = sorted(
            times_graf,
            key=lambda t: mapa_final.get(t, (999, 0.0))[0],
        )
        for time in times_graf_ord:
            pa, pta = mapa_atual.get(time, (0, 0))
            pf, ptf = mapa_final.get(time, (0, 0))
            pr = probs_finais.get(time, {})
            bloco_classificacao_time(
                time,
                pa,
                pta,
                pf,
                ptf,
                prob_campeao=pr.get("campeao", 0.0),
                prob_g4=pr.get("g4", 0.0),
                prob_g6=pr.get("g6", 0.0),
                prob_z4=pr.get("z4", 0.0),
            )
        evolucoes = [
            evolucao_pontos_time(_jogos_base, jogos_graf, t, _ult_r_proj)
            for t in times_graf_ord
        ]
        _grafico(fig_evolucao_times(evolucoes))
        evolucoes_pos = [
            evolucao_posicao_time(_jogos_base, jogos_graf, t, _ult_r_proj)
            for t in times_graf_ord
        ]
        _grafico(fig_evolucao_posicao_times(evolucoes_pos))
    else:
        st.info("Selecione ao menos um time para exibir o gráfico.")

rodape_desenvolvedor()
