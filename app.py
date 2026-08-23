"""App Streamlit — Projeção Brasileirão 2026."""
from __future__ import annotations

import streamlit as st

from brasileirao_estilo import aplicar_estilo, cabecalho_pagina, kpi_row
from brasileirao_projecao_core import (
    ARQUIVO_CALENDARIO,
    ModoProjecao,
    TipoRegressao,
    aplicar_projecoes,
    carregar_jogos_xlsx,
    classificacao,
    evolucao_pontos_time,
    fig_evolucao_times,
    jogo_do_time_na_rodada,
    jogo_faltante_pode_afetar_posicao,
    tabela_betas,
    times_do_calendario,
)
import pandas as pd

st.set_page_config(
    page_title="Projeção Brasileirão 2026",
    layout="wide",
    initial_sidebar_state="collapsed",
)
aplicar_estilo()
cabecalho_pagina(
    "Projeção Brasileirão 2026",
    "Classificação projetada · betas por rodada · evolução por time",
)

try:
    _jogos_base = carregar_jogos_xlsx()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

_times = times_do_calendario(_jogos_base)
_jogados = sum(1 for j in _jogos_base if j.jogado)
_pendentes = len(_jogos_base) - _jogados
_ult_r = max(j.r for j in _jogos_base if j.jogado) if _jogados else 1

kpi_row([
    ("Jogos realizados", str(_jogados), False),
    ("Jogos pendentes", str(_pendentes), True),
    ("Última rodada c/ resultado", str(_ult_r), False),
    ("Times", str(len(_times)), False),
])

with st.expander("Configuração da projeção", expanded=True):
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
            help="Vale para regressão e média simples; também no fallback do modo espelho.",
        )
        tipo: TipoRegressao = (
            "mandante_visitante" if "Mandante" in tipo_label else "simples"
        )

    if modo == "repetir_turno":
        fb_label = st.radio(
            "Fallback quando não há jogo espelhado",
            options=[
                "Regressão linear",
                "Média simples",
            ],
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
            "Rodada início (janela)",
            min_value=1,
            max_value=38,
            value=1,
            step=1,
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

modo_metrica_ui: ModoProjecao = (
    modo if modo != "repetir_turno" else modo_fallback
)
jogos_proj, df_log = aplicar_projecoes(
    _jogos_base, modo, int(r_ini), int(r_fim), tipo, modo_fallback=modo_fallback
)

with st.expander(
    "Betas / médias por time (intervalo selecionado)"
):
    df_betas = tabela_betas(
        _jogos_base, int(r_ini), int(r_fim), tipo, modo=modo_metrica_ui
    )
    st.dataframe(df_betas, use_container_width=True, hide_index=True)

with st.expander("Planilha de dados (fonte dos resultados)"):
    st.caption(
        f"Edite o arquivo **`{ARQUIVO_CALENDARIO.name}`** na pasta `dados/` e recarregue o app. "
        "Use `-` em **Placar** para jogos ainda não realizados."
    )
    df_fonte = pd.read_excel(ARQUIVO_CALENDARIO)
    st.dataframe(df_fonte, use_container_width=True, hide_index=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Classificação")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("##### Hoje (só jogos reais)")
    df_atual = classificacao(_jogos_base, incluir_proj=False)
    st.dataframe(df_atual[["Pos", "Time", "Pontos"]], use_container_width=True, hide_index=True)
with col_b:
    st.markdown("##### Projetada (rodada 38)")
    df_final = classificacao(jogos_proj, incluir_proj=True)
    df_show = df_final[["Pos", "Time", "Pontos"]].merge(
        df_atual[["Time", "Pontos"]].rename(columns={"Pontos": "Pts_hoje"}),
        on="Time",
        how="left",
    )
    df_show["Delta"] = df_show["Pontos"] - df_show["Pts_hoje"]
    st.dataframe(
        df_show[["Pos", "Time", "Pts_hoje", "Pontos", "Delta"]].rename(
            columns={"Pts_hoje": "Hoje", "Pontos": "Proj. R38"}
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Jogos projetados (somente pendentes)"):
    if df_log.empty:
        st.success("Todos os jogos já têm placar — nada a projetar.")
    else:
        st.dataframe(df_log, use_container_width=True, hide_index=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Evolução por rodada")

with st.expander("Times no gráfico e legenda", expanded=True):
    st.caption(
        "Linha **sólida** = confirmado ou projeção passada sem impacto na tabela. "
        "**Tracejada** = futuro ou pendência passada que pode mudar pontos, vitórias, saldo ou gols. "
        "**Pontilhada fina** = só jogos realizados."
    )
    times_graf = st.multiselect(
        "Times para comparar",
        options=_times,
        default=[t for t in ("Palmeiras", "Flamengo", "Cruzeiro") if t in _times],
    )

if times_graf:
    evolucoes = [
        evolucao_pontos_time(_jogos_base, jogos_proj, t, _ult_r) for t in times_graf
    ]
    fig = fig_evolucao_times(evolucoes)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detalhe por rodada (times selecionados)"):
        linhas = []
        for ev in evolucoes:
            for r, pc, pt in zip(ev.rodadas, ev.pts_confirmado, ev.pts_total):
                j = jogo_do_time_na_rodada(jogos_proj, ev.time, r)
                status = "Realizado"
                tracej = ""
                if j and not j.jogado and j.proj_pm is not None:
                    status = "Projetado"
                    if r > _ult_r or jogo_faltante_pode_afetar_posicao(
                        _jogos_base, ev.time, j
                    ):
                        tracej = "Sim"
                    else:
                        tracej = "Não (passado, sem impacto)"
                linhas.append(
                    {
                        "Time": ev.time,
                        "Rodada": r,
                        "Pts confirmados": int(pc),
                        "Pts total (c/ proj.)": int(pt),
                        "Status R": status,
                        "Tracejado?": tracej,
                    }
                )
        st.dataframe(
            pd.DataFrame(linhas),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Selecione ao menos um time no expander acima.")

with st.expander("Metodologia"):
    st.markdown(
        """
**Regressão:** reta `pts_acumulados ~ rodada` no intervalo; inclinação = **beta** (pts/rodada).

**Média simples:** média de pontos por jogo no intervalo (≈ pts/rodada); opção casa/fora separada.

**Repetir 1º turno:** espelha ida/volta já disputada; sem espelho, usa regressão ou média (fallback).

**Gráfico:** trechos tracejados = incerteza que pode alterar classificação (pontos, vitórias, saldo, gols).

Jogos já realizados **nunca** são alterados.

**Dados:** edite `dados/calendario_brasileirao_2026.xlsx` (coluna **Placar**: `-` = pendente).
        """
    )
