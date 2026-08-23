"""App Streamlit — Projeção Brasileirão 2026."""
from __future__ import annotations

import streamlit as st

from brasileirao_estilo import (
    aplicar_estilo,
    bloco_classificacao_time,
    cabecalho_pagina,
    kpi_row,
    rodape_desenvolvedor,
    titulo_secao,
)
from brasileirao_projecao_core import (
    ModoProjecao,
    TipoRegressao,
    VarianteRegressaoAcumulada,
    aplicar_projecoes,
    carregar_jogos,
    colunas_estatisticas_grafico,
    estatisticas_por_rodada,
    colunas_estatisticas_rodada_grafico,
    evolucao_pontos_time,
    evolucao_posicao_time,
    fig_estatisticas_por_rodada,
    fig_estatisticas_times,
    fig_evolucao_times,
    fig_evolucao_posicao_times,
    kpis_globais,
    mapa_posicao_pontos,
    mapa_vitorias_saldo_proj,
    tabela_regressao_acumulada_resumo,
    tabela_medias_simples_times,
    tabela_jogos_primeiro_turno,
    tabela_comparativa_posicoes,
    tabela_estatisticas_times,
    times_do_calendario,
)


@st.cache_data(ttl=120, show_spinner="Carregando planilha…")
def _carregar_dados_cached() -> tuple[list, object | None]:
    jogos, _, df = carregar_jogos(preferir_gsheets=True)
    return jogos, df


st.set_page_config(
    page_title="Projeção Brasileirão 2026",
    layout="wide",
    initial_sidebar_state="collapsed",
)
aplicar_estilo()
cabecalho_pagina("Projeção Brasileirão 2026")

try:
    _jogos_base, _df_fonte = _carregar_dados_cached()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

_times = times_do_calendario(_jogos_base)
_jogados = sum(1 for j in _jogos_base if j.jogado)
_pendentes = len(_jogos_base) - _jogados
_ult_r = max(j.r for j in _jogos_base if j.jogado) if _jogados else 1

_kpi = kpis_globais(_jogos_base)

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
        max_value=38,
        value=1,
        step=1,
        key="stats_r_ini",
    )
with c_stats2:
    r_fim_stats = st.number_input(
        "Rodada fim (estatísticas)",
        min_value=1,
        max_value=38,
        value=int(min(_ult_r, 38)),
        step=1,
        key="stats_r_fim",
    )
if r_fim_stats < r_ini_stats:
    st.warning("Rodada fim (estatísticas) menor que início — usando fim = início.")
    r_fim_stats = r_ini_stats

_df_stats = tabela_estatisticas_times(_jogos_base, int(r_ini_stats), int(r_fim_stats))

st.dataframe(
    _df_stats,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Time": st.column_config.TextColumn("Time", pinned="left", width="medium"),
    },
)

titulo_secao("Gráfico de estatísticas")
_cols_stats = colunas_estatisticas_grafico(_df_stats)
_default_times = [t for t in ("Palmeiras", "Flamengo", "Cruzeiro") if t in _times]
_default_stats = [c for c in ("Total pontos", "Média gols marcados") if c in _cols_stats]

c_graf1, c_graf2 = st.columns(2)
with c_graf1:
    times_stats_graf = st.multiselect(
        "Times no gráfico",
        options=_times,
        default=_default_times,
        key="stats_graf_times",
    )
with c_graf2:
    metricas_graf = st.multiselect(
        "Estatísticas no gráfico",
        options=_cols_stats,
        default=_default_stats,
        key="stats_graf_metricas",
    )

if times_stats_graf and metricas_graf:
    st.plotly_chart(
        fig_estatisticas_times(_df_stats, times_stats_graf, metricas_graf),
        use_container_width=True,
    )
else:
    st.info("Selecione ao menos um time e uma estatística para exibir o gráfico.")

_df_stats_rodada = estatisticas_por_rodada(
    _jogos_base, int(r_ini_stats), int(r_fim_stats)
)

titulo_secao("Gráfico de estatísticas por rodada")
_cols_stats_rodada = colunas_estatisticas_rodada_grafico(_df_stats_rodada)
_default_stats_rodada = [
    c for c in ("Total pontos", "Média gols marcados") if c in _cols_stats_rodada
]

c_graf_r1, c_graf_r2 = st.columns(2)
with c_graf_r1:
    times_rodada_graf = st.multiselect(
        "Times no gráfico",
        options=_times,
        default=_default_times,
        key="stats_rodada_times",
    )
with c_graf_r2:
    metricas_rodada_graf = st.multiselect(
        "Estatísticas no gráfico",
        options=_cols_stats_rodada,
        default=_default_stats_rodada,
        key="stats_rodada_metricas",
    )

if times_rodada_graf and metricas_rodada_graf:
    st.plotly_chart(
        fig_estatisticas_por_rodada(
            _df_stats_rodada, times_rodada_graf, metricas_rodada_graf
        ),
        use_container_width=True,
    )
else:
    st.info("Selecione ao menos um time e uma estatística para exibir o gráfico.")

with st.expander("Dados carregados"):
    if _df_fonte is not None:
        st.dataframe(_df_fonte, use_container_width=True, hide_index=True)
    else:
        st.info("Sem preview tabular.")

titulo_secao("Configuração da projeção")

r_ini_proj = 1
r_fim_proj = int(min(_ult_r, 38))

_MODO_ACUM_SIMPLES = (
    "Regressão acumulada + simples: "
    "pts_acumulados ~ rodada + indicador_casa + rodada × indicador_casa"
)
_MODO_ACUM_ROBUSTA = (
    "Regressão acumulada + robusta: "
    "pts_acumulados ~ rodada + rodada² + proporção_casa "
    "+ força oponentes passados + forma recente (últimos 5 jogos)"
)
_MODO_MEDIA = (
    "Média casa x fora × forma recente "
    "(média casa/fora × últimos 5 / média camp.; pts decimais)"
)
_MODO_TURNO = "Repetir 1 turno"

_modo_opcoes = [
    _MODO_ACUM_SIMPLES,
    _MODO_ACUM_ROBUSTA,
    _MODO_MEDIA,
    _MODO_TURNO,
]
modo_label = st.radio("Modo de projeção", options=_modo_opcoes, index=0)

modo: ModoProjecao
tipo: TipoRegressao = "mandante_visitante"
variante_acum: VarianteRegressaoAcumulada = "acum_simples"

if modo_label.startswith("Regressão acumulada + robusta"):
    modo = "regressao_acum_robusta"
    variante_acum = "acum_robusta"
elif modo_label.startswith("Regressão acumulada + simples"):
    modo = "regressao_acum_simples"
    variante_acum = "acum_simples"
elif modo_label == _MODO_MEDIA:
    modo = "media_simples"
elif modo_label == _MODO_TURNO:
    modo = "repetir_turno"
else:
    modo = "regressao_acum_simples"
    variante_acum = "acum_simples"

with st.expander("Detalhes do modelo"):
    if modo in ("regressao_acum_simples", "regressao_acum_robusta"):
        st.caption(
            "Curva de pontos acumulados por rodada; cada jogo recebe o delta "
            "decimal da curva (sem arredondar para 3/1/0). Significância: "
            "*** p<0,001 | ** p<0,01 | * p<0,05 | - não significativo"
        )
        st.dataframe(
            tabela_regressao_acumulada_resumo(
                _jogos_base, r_ini_proj, r_fim_proj, variante_acum
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time", pinned="left"),
                "R²": st.column_config.NumberColumn("R²", format="%.3f"),
            },
        )
    elif modo == "media_simples":
        st.caption(
            "Projeção = média pts/jogo em casa ou fora × "
            "(média últimos 5 jogos / média campeonato no intervalo)."
        )
        st.dataframe(
            tabela_medias_simples_times(_jogos_base, r_ini_proj, r_fim_proj),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time", pinned="left"),
            },
        )
    else:
        st.caption(
            "Jogos do 1º turno (rodadas 1–19) usados como referência para espelhar a volta; "
            "sem par já disputado, usa média casa/fora × forma recente."
        )
        st.dataframe(
            tabela_jogos_primeiro_turno(_jogos_base),
            use_container_width=True,
            hide_index=True,
        )

jogos_proj, df_log = aplicar_projecoes(
    _jogos_base, modo, r_ini_proj, r_fim_proj, tipo
)

titulo_secao("Classificação")
st.dataframe(
    tabela_comparativa_posicoes(_jogos_base, jogos_proj),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Jogos projetados"):
    if df_log.empty:
        st.success("Todos os jogos já têm placar — nada a projetar.")
    else:
        st.dataframe(df_log, use_container_width=True, hide_index=True)

titulo_secao("Evolução por rodada")
times_graf = st.multiselect(
    "Times para comparar",
    options=_times,
    default=[t for t in ("Palmeiras", "Flamengo", "Cruzeiro") if t in _times],
)

if times_graf:
    mapa_atual = mapa_posicao_pontos(_jogos_base, incluir_proj=False)
    mapa_final = mapa_posicao_pontos(jogos_proj, incluir_proj=True)
    mapa_vit_saldo = mapa_vitorias_saldo_proj(jogos_proj)

    for time in times_graf:
        pa, pta = mapa_atual.get(time, (0, 0))
        pf, ptf = mapa_final.get(time, (0, 0))
        vit, saldo = mapa_vit_saldo.get(time, (0, 0))
        bloco_classificacao_time(time, pa, pta, pf, ptf, vit, saldo)

    evolucoes = [
        evolucao_pontos_time(_jogos_base, jogos_proj, t, _ult_r) for t in times_graf
    ]
    st.plotly_chart(fig_evolucao_times(evolucoes), use_container_width=True)

    evolucoes_pos = [
        evolucao_posicao_time(_jogos_base, jogos_proj, t, _ult_r) for t in times_graf
    ]
    st.plotly_chart(fig_evolucao_posicao_times(evolucoes_pos), use_container_width=True)
else:
    st.info("Selecione ao menos um time para exibir o gráfico.")

rodape_desenvolvedor()
