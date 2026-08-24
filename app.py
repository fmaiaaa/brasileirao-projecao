"""App Streamlit - Projeção Brasileirão 2026."""
from __future__ import annotations

import html

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
    ModoProjecao,
    TipoRegressao,
    VarianteRegressaoAcumulada,
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
    tabela_regressao_acumulada_resumo,
    tabela_medias_simples_times,
    tabela_jogos_primeiro_turno,
    tabela_comparativa_posicoes,
    tabela_estatisticas_times,
    times_do_calendario,
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
    st.warning("Rodada fim (estatísticas) menor que início - usando fim = início.")
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
    _jogos_base, int(r_ini_stats), _r_fim_rodada
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

incluir_projecao_rodada = st.toggle(
    "Incluir projeções no gráfico",
    value=False,
    key="stats_rodada_projecao",
)

if times_rodada_graf and metricas_rodada_graf:
    if incluir_projecao_rodada:
        _df_graf_rodada = projetar_estatisticas_por_rodada(
            _jogos_base,
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
    if _df_fonte is not None:
        st.dataframe(_df_fonte, use_container_width=True, hide_index=True)
    else:
        st.info("Sem preview tabular.")

titulo_secao("Configuração da projeção")

r_ini_proj = 1
r_fim_proj = int(min(_ult_r, 38))

_MODO_REG = (
    "Regressão - "
    "Pontos Acumulados ~ Efeito Fixo do Time + Rodada + Rodada ao Quadrado + "
    "Interação Rodada × Time + Interação Rodada ao Quadrado × Time + "
    "Forma Recente + Força dos Adversários Passados + Proporção Casa"
)
_MODO_MEDIA = "Média casa x fora × forma recente"
_MODO_TURNO = "Repetir 1º turno"

_modo_opcoes = [
    _MODO_REG,
    _MODO_MEDIA,
    _MODO_TURNO,
]
modo_label = st.radio("Modo de projeção", options=_modo_opcoes, index=0)

modo: ModoProjecao
tipo: TipoRegressao = "mandante_visitante"
variante_acum: VarianteRegressaoAcumulada = "completa"

if modo_label.startswith("Regressão"):
    modo = "regressao_completa"
    variante_acum = "completa"
elif modo_label == _MODO_MEDIA:
    modo = "media_simples"
elif modo_label == _MODO_TURNO:
    modo = "repetir_turno"
else:
    modo = "regressao_completa"
    variante_acum = "completa"

with st.expander("Detalhes do modelo"):
    if modo_e_regressao_acumulada(modo):
        st.caption(
            "Modelo de efeitos fixos (Efeito Fixo do Time) com Interação Rodada × Time "
            "e Interação Rodada ao Quadrado × Time "
            "(time de referência com interações nulas). "
            "Curva de Pontos Acumulados por rodada; cada jogo recebe o delta decimal "
            "(máximo 3 pontos por rodada). "
            "O peso da Forma Recente cai de 80% (próxima) para 50% (daqui a 5) "
            "até o piso de 20%, misturando com a forma geral. "
            "Significância: *** p<0,001 | ** p<0,01 | * p<0,05 | - não significativo"
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
            "Projeção = média pts/jogo em casa ou fora × fator de forma. "
            "O fator mistura forma recente e forma geral: peso da recente cai de "
            "80% (próxima rodada) para 50% (daqui a 5) até o piso de 20%."
        )
        st.dataframe(
            tabela_medias_simples_times(
                _jogos_base, r_ini_proj, r_fim_proj, usar_forma=True
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time", pinned="left"),
            },
        )
    else:
        st.caption(
            "Jogos do 1º turno usados como referência para espelhar a volta; "
            "sem par já disputado, usa média casa/fora × forma recente "
            "(com o mesmo decaimento de peso da forma)."
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
_df_classif = tabela_comparativa_posicoes(_jogos_base, jogos_proj)
st.dataframe(
    _df_classif,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Posição Projetada": st.column_config.NumberColumn("Posição Projetada"),
        "Time": st.column_config.TextColumn("Time", pinned="left"),
        "Posição Atual": st.column_config.NumberColumn("Posição Atual"),
        "Delta": st.column_config.NumberColumn("Delta"),
        "Pts Projetados": st.column_config.NumberColumn(
            "Pontuação Projetada", format="%.1f"
        ),
        "Prob. Campeão": st.column_config.NumberColumn(
            "Probabilidade de ser campeão", format="%.1f%%"
        ),
        "Prob. G4": st.column_config.NumberColumn(
            "Probabilidade de G4", format="%.1f%%"
        ),
        "Prob. G6": st.column_config.NumberColumn(
            "Probabilidade de G6", format="%.1f%%"
        ),
        "Prob. Z4": st.column_config.NumberColumn(
            "Probabilidade de Z4", format="%.1f%%"
        ),
    },
)

with st.expander("Jogos projetados"):
    if df_log.empty:
        st.success("Todos os jogos já têm placar - nada a projetar.")
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
    probs_finais = probabilidades_cenarios_finais(jogos_proj)
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
        evolucao_pontos_time(_jogos_base, jogos_proj, t, _ult_r) for t in times_graf_ord
    ]
    _grafico(fig_evolucao_times(evolucoes))

    evolucoes_pos = [
        evolucao_posicao_time(_jogos_base, jogos_proj, t, _ult_r) for t in times_graf_ord
    ]
    _grafico(fig_evolucao_posicao_times(evolucoes_pos))
else:
    st.info("Selecione ao menos um time para exibir o gráfico.")

rodape_desenvolvedor()
