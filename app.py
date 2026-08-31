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
    SECAO_LABEL,
    SECOES_ORDEM,
    modelos_da_secao,
)
from modelos_acumulados import MODO_PARA_LABEL

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
    f"Calendário {_ANO_CALENDARIO}. Treino e projeção usam somente dados de 2026 "
    "na seção «Só 2026»; demais seções incluem histórico com decaimento de peso "
    "e dummy Série A/B + interação time×série."
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


def _render_bloco_projecao(
    secao_key: JanelaTreino,
    secao_lbl: str,
    *,
    key_prefix: str,
) -> None:
    modelos = modelos_da_secao(secao_key)
    modelo_lbl = st.radio(
        "Modelo",
        options=[m[0] for m in modelos],
        horizontal=True,
        key=f"{key_prefix}_modelo",
    )
    modo = next(m[1] for m in modelos if m[0] == modelo_lbl)

    with st.expander("Detalhes do modelo"):
        if modo_e_regressao_acumulada(modo):
            st.caption(
                f"Coeficientes e projeções da base semanal · {secao_lbl}. "
                "Pesos: últimas 5 rodadas 100→90%; temporada 90→25%; anos passados 25→0%."
            )
            _resumo = load_resumo_modelos_acum(modelo_lbl, secao=secao_lbl)
            if _resumo is not None and not _resumo.empty:
                st.caption(
                    f"R² = {_resumo.iloc[0].get('R²', '—')} · "
                    f"N obs = {_resumo.iloc[0].get('N observações', '—')}"
                )
            _coefs = load_coefs_modelos_acum(modelo_lbl, secao=secao_lbl)
            if _coefs is not None and not _coefs.empty:
                _coefs_show = _coefs.copy()
                for _c in list(_coefs_show.columns):
                    if str(_c).strip().lower().replace("²", "2") in {"r2", "r²"}:
                        _coefs_show[_c] = _coefs_show[_c].map(_fmt_r2_ui)
                        break
                _tabela(
                    _coefs_show,
                    column_config={
                        "Time": st.column_config.TextColumn("Time", pinned="left"),
                        "Variável": st.column_config.TextColumn("Variável"),
                        "Beta": st.column_config.NumberColumn("Beta", format="%.4f"),
                    },
                    key=f"{key_prefix}_coef_reg",
                )
            else:
                st.info(
                    "Coeficientes ainda não disponíveis na base. "
                    "Aguarde o job semanal ou execute scripts/weekly_retrain.py."
                )
        elif modo == "media_simples":
            st.caption(
                f"Médias e projeções da base semanal · {secao_lbl}. "
                "Projeção = média pts/jogo × fator de forma."
            )
            _resumo = load_resumo_modelos_acum(LABEL_MEDIA, secao=secao_lbl)
            if _resumo is not None and not _resumo.empty:
                st.caption(
                    f"N obs treino = {_resumo.iloc[0].get('N observações', '—')}"
                )
            _coefs = load_coefs_modelos_acum(LABEL_MEDIA, secao=secao_lbl)
            if _coefs is not None and not _coefs.empty:
                _tabela(
                    _coefs,
                    column_config={
                        "Time": st.column_config.TextColumn("Time", pinned="left"),
                        "Variável": st.column_config.TextColumn("Variável"),
                        "Beta": st.column_config.NumberColumn("Valor", format="%.3f"),
                    },
                    key=f"{key_prefix}_coef_media",
                )
            else:
                st.info("Médias por time ainda não disponíveis na base para esta seção.")
        elif modo == "prob_ml":
            st.caption(f"Previsões e Monte Carlo da base semanal · {secao_lbl}.")
            _resumo = load_resumo_modelos_acum(LABEL_PROB, secao=secao_lbl)
            if _resumo is not None and not _resumo.empty:
                st.caption(
                    f"N obs treino = {_resumo.iloc[0].get('N observações', '—')} · "
                    f"R² ensemble = {_resumo.iloc[0].get('R²', '—')}"
                )
            _met = load_prob_metricas()
            if _met is not None and not _met.empty and secao_key == "ultimas_38_rodadas":
                st.caption("Métricas OOF (referência — Últimas 38 rodadas):")
                _tabela(_met, key=f"{key_prefix}_metricas_prob")
        else:
            st.caption(
                "Espelha o 1º turno deste Brasileirão (somente jogos de 2026); "
                "sem par, usa média casa/fora × forma recente."
            )
            _tabela(
                tabela_jogos_primeiro_turno(_jogos_base),
                key=f"{key_prefix}_tbl_turno",
            )

    _prob_preds: list = []
    _prob_sim_df = None

    if modo == "prob_ml":
        _cal = load_projecoes_modelos_acum(LABEL_PROB, secao=secao_lbl)
        _stand = load_classif_modelos_acum(LABEL_PROB, secao=secao_lbl)
        _fc = load_forecasts_modelos_acum(LABEL_PROB, secao=secao_lbl)
        if _cal is None or _cal.empty:
            st.warning(
                f"Projeções probabilísticas ({secao_lbl}) não encontradas na base. "
                "Execute o job semanal."
            )
            jogos_proj = [Jogo(**j.__dict__) for j in _jogos_base]
            df_log = pd.DataFrame()
        else:
            jogos_proj, df_log = aplicar_projecoes_csv_com_gap(
                _jogos_base,
                _cal,
                r_ini=r_ini_proj,
                r_fim=r_fim_proj,
                origem=f"prob_ml/{secao_lbl}",
            )
        if _stand is not None and not _stand.empty:
            _prob_sim_df = _stand.drop(
                columns=["Modelo", "Seção", "Janela"], errors="ignore"
            )
        if _fc is not None and not _fc.empty:
            pend_pairs = {(j.mand, j.vis) for j in _jogos_base if not j.jogado}
            _prob_preds = [
                p
                for p in _fc.to_dict(orient="records")
                if (
                    str(p.get("home_team") or p.get("Mandante") or ""),
                    str(p.get("away_team") or p.get("Visitante") or ""),
                )
                in pend_pairs
            ]
            from prob_ml.integration import _safe_float as _sf

            for p in _prob_preds:
                p["xg_home"] = _sf(p.get("xg_home"), 0.0) or 0.0
                p["xg_away"] = _sf(p.get("xg_away"), 0.0) or 0.0
                p["p_home"] = _sf(p.get("p_home"), 0.0) or 0.0
                p["p_draw"] = _sf(p.get("p_draw"), 0.0) or 0.0
                p["p_away"] = _sf(p.get("p_away"), 0.0) or 0.0
                p["over_25"] = _sf(p.get("over_25"), 0.0) or 0.0
                p["btts_yes"] = _sf(p.get("btts_yes"), 0.0) or 0.0
                tops = p.get("top_scores")
                if isinstance(tops, str):
                    try:
                        import json as _json

                        p["top_scores"] = _json.loads(tops)
                    except Exception:
                        p["top_scores"] = []
                elif not tops:
                    p["top_scores"] = []

    elif modo == "media_simples":
        _cal_media = load_projecoes_modelos_acum(LABEL_MEDIA, secao=secao_lbl)
        if _cal_media is None or _cal_media.empty:
            st.warning(
                f"Projeções de {LABEL_MEDIA} ({secao_lbl}) não encontradas na base. "
                "Execute o job semanal."
            )
            jogos_proj = [Jogo(**j.__dict__) for j in _jogos_base]
            df_log = pd.DataFrame()
        else:
            jogos_proj, df_log = aplicar_projecoes_csv_com_gap(
                _jogos_base,
                _cal_media,
                r_ini=r_ini_proj,
                r_fim=r_fim_proj,
                origem=f"media/{secao_lbl}",
            )

    elif modo_e_regressao_acumulada(modo):
        _cal_acum = load_projecoes_modelos_acum(modelo_lbl, secao=secao_lbl)
        if _cal_acum is None or _cal_acum.empty:
            st.warning(
                f"Projeções de {modelo_lbl} ({secao_lbl}) não encontradas na base. "
                "Execute o job semanal."
            )
            jogos_proj = [Jogo(**j.__dict__) for j in _jogos_base]
            df_log = pd.DataFrame()
        else:
            jogos_proj, df_log = aplicar_projecoes_csv_com_gap(
                _jogos_base,
                _cal_acum,
                r_ini=r_ini_proj,
                r_fim=r_fim_proj,
                origem=f"modelos_acum/{modelo_lbl}",
            )

    elif modo == "repetir_turno":
        jogos_proj, df_log = aplicar_projecoes(
            _jogos_base,
            modo,
            r_ini_proj,
            r_fim_proj,
            "mandante_visitante",
            janela="2026",
            ano_calendario=_ANO_CALENDARIO,
        )
    else:
        jogos_proj, df_log = aplicar_projecoes(
            _jogos_base, modo, r_ini_proj, r_fim_proj, "mandante_visitante"
        )

    st.markdown("#### Classificação")
    _df_classif = tabela_comparativa_posicoes(_jogos_base, jogos_proj)
    if modo == "prob_ml" and _prob_sim_df is not None and not getattr(_prob_sim_df, "empty", True):
        from prob_ml.integration import _safe_float as _sf

        _mc = _prob_sim_df.copy()
        if "Time" in _mc.columns:
            _mc = _mc.drop_duplicates(subset=["Time"], keep="first").set_index("Time")
        for col_src, col_dst in (
            ("Prob. Campeão", "Prob. Campeão"),
            ("Prob. G4", "Prob. G4"),
            ("Prob. G6", "Prob. G6"),
            ("Prob. Z4", "Prob. Z4"),
            ("Pts Esperados", "Pts Projetados"),
        ):
            if col_src in _mc.columns and col_dst in _df_classif.columns:
                _df_classif[col_dst] = _df_classif["Time"].map(
                    lambda t, c=col_src: _sf(_mc.loc[t, c]) if t in _mc.index else None
                )
    _tabela(
        _df_classif,
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
        key=f"{key_prefix}_classif",
    )

    with st.expander("Jogos projetados"):
        if df_log.empty:
            st.success("Todos os jogos já têm placar — nada a projetar.")
        else:
            _tabela(df_log, key=f"{key_prefix}_jogos_proj")

    if modo == "prob_ml" and _prob_preds:
        with st.expander("Model Lab — previsões de placar", expanded=False):
            _lab = []
            for p in _prob_preds[:40]:
                tops = ", ".join(
                    f"{i}-{j} ({pr:.0%})" for i, j, pr in p["top_scores"][:3]
                )
                _lab.append(
                    {
                        "Rodada": p.get("round"),
                        "Jogo": f"{p['home_team']} x {p['away_team']}",
                        "λ H": round(p["xg_home"], 2),
                        "λ A": round(p["xg_away"], 2),
                        "P(H)": round(100 * p["p_home"], 1),
                        "P(D)": round(100 * p["p_draw"], 1),
                        "P(A)": round(100 * p["p_away"], 1),
                        "O2.5": round(100 * p["over_25"], 1),
                        "BTTS": round(100 * p["btts_yes"], 1),
                        "Top placares": tops,
                    }
                )
            _tabela(pd.DataFrame(_lab), key=f"{key_prefix}_model_lab")
            if _prob_sim_df is not None and not getattr(_prob_sim_df, "empty", True):
                st.caption("Simulação Monte Carlo")
                _tabela(_prob_sim_df, key=f"{key_prefix}_mc_prob")

    st.markdown("#### Evolução por rodada")
    times_graf = st.multiselect(
        "Times para comparar",
        options=_times_proj,
        default=[t for t in ("Palmeiras", "Flamengo", "Cruzeiro") if t in _times_proj],
        key=f"{key_prefix}_times_graf",
    )
    if times_graf:
        mapa_atual = mapa_posicao_pontos(_jogos_base, incluir_proj=False)
        mapa_final = mapa_posicao_pontos(jogos_proj, incluir_proj=True)
        if modo == "prob_ml" and _prob_sim_df is not None and not _prob_sim_df.empty:
            probs_finais = {
                str(r["Time"]): {
                    "campeao": float(r["Prob. Campeão"]) / 100.0,
                    "g4": float(r["Prob. G4"]) / 100.0,
                    "g6": float(r["Prob. G6"]) / 100.0,
                    "z4": float(r["Prob. Z4"]) / 100.0,
                }
                for _, r in _prob_sim_df.iterrows()
            }
        else:
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
            evolucao_pontos_time(_jogos_base, jogos_proj, t, _ult_r_proj)
            for t in times_graf_ord
        ]
        _grafico(fig_evolucao_times(evolucoes))
        evolucoes_pos = [
            evolucao_posicao_time(_jogos_base, jogos_proj, t, _ult_r_proj)
            for t in times_graf_ord
        ]
        _grafico(fig_evolucao_posicao_times(evolucoes_pos))
    else:
        st.info("Selecione ao menos um time para exibir o gráfico.")


_tabs = st.tabs([SECAO_LABEL[s] for s in SECOES_ORDEM])
for _secao_key, _tab in zip(SECOES_ORDEM, _tabs):
    with _tab:
        _render_bloco_projecao(
            _secao_key,
            SECAO_LABEL[_secao_key],
            key_prefix=f"sec_{_secao_key}",
        )

rodape_desenvolvedor()
