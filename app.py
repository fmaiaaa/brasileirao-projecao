"""App Streamlit — Projeção Brasileirão 2026."""
from __future__ import annotations

import streamlit as st

from brasileirao_estilo import (
    aplicar_estilo,
    cabecalho_pagina,
    kpi_duo,
    kpi_row,
    rodape_desenvolvedor,
    titulo_secao,
)
from brasileirao_projecao_core import (
    ModoProjecao,
    TipoRegressao,
    aplicar_projecoes,
    carregar_jogos,
    evolucao_pontos_time,
    fig_evolucao_times,
    kpis_globais,
    mapa_posicao_pontos,
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

titulo_secao("Configuração da projeção")

c1, c2 = st.columns(2)
with c1:
    modo_label = st.radio(
        "Modo de projeção",
        options=[
            "Regressão linear (betas por rodada)",
            "Média simples (pts por jogo/rodada)",
            "Repetir 1º turno (só jogos faltantes)",
        ],
        index=0,
    )
    if "Regressão" in modo_label:
        modo: ModoProjecao = "regressao"
    elif "Média" in modo_label:
        modo = "media_simples"
    else:
        modo = "repetir_turno"
with c2:
    tipo_label = st.radio(
        "Efeito mando de campo",
        options=[
            "Simples (único para casa e fora)",
            "Mandante × Visitante (casa/fora separados)",
        ],
        index=0,
    )
    tipo: TipoRegressao = (
        "mandante_visitante" if "Mandante" in tipo_label else "simples"
    )

if modo == "repetir_turno":
    fb_label = st.radio(
        "Fallback quando não há jogo espelhado",
        options=["Regressão linear", "Média simples"],
        index=0,
        horizontal=True,
    )
    modo_fallback: ModoProjecao = (
        "media_simples" if "Média" in fb_label else "regressao"
    )
else:
    modo_fallback = "regressao"

c3, c4 = st.columns(2)
with c3:
    r_ini = st.number_input(
        "Rodada início (janela)", min_value=1, max_value=38, value=1, step=1
    )
with c4:
    r_fim = st.number_input(
        "Rodada fim (janela)",
        min_value=1,
        max_value=38,
        value=int(min(_ult_r, 38)),
        step=1,
    )
if r_fim < r_ini:
    st.warning("Rodada fim menor que início — usando fim = início.")
    r_fim = r_ini

jogos_proj, df_log = aplicar_projecoes(
    _jogos_base, modo, int(r_ini), int(r_fim), tipo, modo_fallback=modo_fallback
)

titulo_secao("Estatísticas por time (intervalo selecionado)")
st.dataframe(
    tabela_estatisticas_times(_jogos_base, int(r_ini), int(r_fim)),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Dados carregados"):
    if _df_fonte is not None:
        st.dataframe(_df_fonte, use_container_width=True, hide_index=True)
    else:
        st.info("Sem preview tabular.")

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

    for i in range(0, len(times_graf), 2):
        cols = st.columns(2)
        for col, time in zip(cols, times_graf[i : i + 2]):
            with col:
                st.markdown(
                    f'<p class="vel-kpi-time-label">{time}</p>',
                    unsafe_allow_html=True,
                )
                pa, pta = mapa_atual.get(time, (0, 0))
                pf, ptf = mapa_final.get(time, (0, 0))
                kpi_duo(
                    "Classificação Atual",
                    f"{pa}º · {pta} pts",
                    "Classificação Final",
                    f"{pf}º · {ptf} pts",
                )

    evolucoes = [
        evolucao_pontos_time(_jogos_base, jogos_proj, t, _ult_r) for t in times_graf
    ]
    st.plotly_chart(fig_evolucao_times(evolucoes), use_container_width=True)
else:
    st.info("Selecione ao menos um time para exibir o gráfico.")

rodape_desenvolvedor()
