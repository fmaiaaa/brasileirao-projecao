"""Estilo visual do app Brasileirão (inspirado no velocímetro, tema futebol)."""
from __future__ import annotations

# Paleta futebol (gramado / estádio) — sem cores Direcional
COR_VERDE_ESC = "#14532d"
COR_VERDE = "#15803d"
COR_VERDE_CLARO = "#22c55e"
COR_ACCENT = "#ca8a04"
COR_TEXTO = "#0f172a"
COR_TEXTO_MUTED = "#334155"
COR_FUNDO_CARD = "rgba(255, 255, 255, 0.82)"
COR_BORDA = "#e2e8f0"
COR_INPUT_BG = "#f0fdf4"

URL_FUNDO_FUTEBOL = (
    "https://images.unsplash.com/photo-1574629810360-7efbbe195018"
    "?auto=format&fit=crop&w=1920&q=80"
)


def _hex_rgb(hex_color: str) -> str:
    x = (hex_color or "").strip().lstrip("#")
    if len(x) != 6:
        return "0, 0, 0"
    return f"{int(x[0:2], 16)}, {int(x[2:4], 16)}, {int(x[4:6], 16)}"


RGB_VERDE = _hex_rgb(COR_VERDE_ESC)
RGB_ACCENT = _hex_rgb(COR_ACCENT)


def aplicar_estilo() -> None:
    import streamlit as st

    bg = URL_FUNDO_FUTEBOL
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes barShimmer {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 200% 50%; }}
        }}
        html, body, :root, [data-testid="stApp"] {{
            color-scheme: light !important;
        }}
        html, body {{
            font-family: 'Inter', sans-serif;
            color: {COR_TEXTO};
            background: transparent !important;
        }}
        .stApp, [data-testid="stApp"] {{
            background:
                linear-gradient(135deg,
                    rgba({RGB_VERDE}, 0.88) 0%,
                    rgba(20, 83, 45, 0.72) 42%,
                    rgba(15, 23, 42, 0.55) 100%),
                url("{bg}") center / cover no-repeat !important;
            background-attachment: scroll !important;
        }}
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        header[data-testid="stHeader"],
        [data-testid="stDecoration"],
        [data-testid="stToolbar"] {{
            background: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] {{
            display: none !important;
        }}
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] button,
        [data-testid="stHeader"] button {{
            color: rgba(255, 255, 255, 0.92) !important;
            background: transparent !important;
        }}
        [data-testid="stMain"] {{
            padding: clamp(12px, 3vh, 36px) clamp(14px, 4vw, 52px) !important;
        }}
        .block-container {{
            max-width: 1680px !important;
            padding: 1.5rem 2rem 1.65rem 2rem !important;
            background: {COR_FUNDO_CARD} !important;
            backdrop-filter: blur(16px) saturate(1.12);
            -webkit-backdrop-filter: blur(16px) saturate(1.12);
            border-radius: 22px !important;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
            box-shadow:
                0 4px 6px -1px rgba({RGB_VERDE}, 0.08),
                0 22px 44px -14px rgba(15, 23, 42, 0.28),
                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
            animation: fadeInUp 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        h1, h2, h3, h4,
        h1 *, h2 *, h3 *, h4 *,
        [data-testid="stHeadingWithAction"],
        [data-testid="stHeadingWithAction"] * {{
            font-family: 'Montserrat', sans-serif !important;
            color: {COR_VERDE_ESC} !important;
            font-weight: 800 !important;
            text-align: center !important;
        }}
        [data-testid="stCaption"],
        [data-testid="stCaptionContainer"] p,
        .block-container p,
        label, [data-testid="stWidgetLabel"] {{
            color: {COR_TEXTO_MUTED} !important;
        }}
        .ficha-hero-stack {{ width: 100%; margin-bottom: 0.5rem; }}
        .ficha-hero {{ text-align: center; max-width: 720px; margin: 0 auto; }}
        .ficha-hero .ficha-title {{
            font-family: 'Montserrat', sans-serif;
            font-size: clamp(1.4rem, 3.6vw, 1.85rem);
            font-weight: 900;
            color: {COR_VERDE_ESC};
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .ficha-hero .ficha-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            color: {COR_TEXTO_MUTED};
            margin: 0.35rem 0 0 0;
        }}
        .ficha-hero-bar {{
            height: 4px;
            width: 100%;
            margin: 1rem 0 0.25rem 0;
            border-radius: 999px;
            background: linear-gradient(90deg, {COR_VERDE_ESC}, {COR_VERDE_CLARO}, {COR_ACCENT}, {COR_VERDE_ESC});
            background-size: 200% 100%;
            animation: barShimmer 5s ease-in-out infinite alternate;
        }}
        .vel-kpi-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 1rem 0 1.25rem 0;
        }}
        .vel-kpi {{
            flex: 1 1 18%;
            min-width: 140px;
            background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(240,253,244,0.92) 100%);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 14px;
            padding: 14px 16px;
            text-align: center;
            box-shadow: 0 2px 8px rgba({RGB_VERDE}, 0.07);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .vel-kpi:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 22px -6px rgba({RGB_VERDE}, 0.18);
        }}
        .vel-kpi .lbl {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {COR_TEXTO_MUTED};
        }}
        .vel-kpi .val {{
            font-family: 'Montserrat', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            color: {COR_VERDE_ESC} !important;
            margin-top: 6px;
        }}
        .vel-kpi .val--accent {{ color: {COR_ACCENT} !important; }}
        [data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.55) !important;
            border: 1px solid {COR_BORDA} !important;
            border-radius: 14px !important;
        }}
        [data-testid="stExpander"] summary {{
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 700 !important;
            color: {COR_VERDE_ESC} !important;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.65);
            padding: 12px;
            border-radius: 12px;
            border: 1px solid {COR_BORDA};
        }}
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div {{
            border-radius: 10px !important;
            background-color: {COR_INPUT_BG} !important;
        }}
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            background-color: #ffffff !important;
            border-radius: 12px;
        }}
        hr {{ border: none; border-top: 1px solid {COR_BORDA}; margin: 1.25rem 0; }}
        .site-footer {{
            text-align: center;
            font-size: 0.72rem;
            color: {COR_TEXTO_MUTED};
            opacity: 0.75;
            margin: 2rem 0 0.5rem 0;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.04em;
        }}
        .secao-titulo {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 800;
            color: {COR_VERDE_ESC};
            text-align: center;
            margin: 1.25rem 0 0.85rem 0;
            font-size: 1.15rem;
        }}
        .vel-kpi-row--duo {{
            display: flex;
            flex-wrap: nowrap;
            gap: 12px;
            margin: 0.35rem 0 0.85rem 0;
        }}
        .vel-kpi-row--duo .vel-kpi {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .vel-kpi-row--quad {{
            display: flex;
            flex-wrap: nowrap;
            gap: 10px;
            margin: 0.35rem 0 0.85rem 0;
        }}
        .vel-kpi-row--quad .vel-kpi {{
            flex: 1 1 0;
            min-width: 0;
            padding: 12px 10px;
        }}
        .vel-kpi-row--quad .vel-kpi .val {{
            font-size: 1.1rem;
        }}
        .vel-time-evolucao-block {{
            margin: 0.5rem 0 1rem 0;
        }}
        .vel-kpi-time-label {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 800;
            color: {COR_VERDE_ESC};
            text-align: center;
            margin: 0.75rem 0 0.25rem 0;
            font-size: 0.95rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def cabecalho_pagina(titulo: str, subtitulo: str | None = None) -> None:
    import streamlit as st

    partes = [
        '<div class="ficha-hero-stack">',
        '<div class="ficha-hero">',
        f'<p class="ficha-title">{titulo}</p>',
    ]
    if subtitulo:
        partes.append(f'<p class="ficha-sub">{subtitulo}</p>')
    partes.append("</div>")
    partes.append('<div class="ficha-hero-bar" aria-hidden="true"></div>')
    partes.append("</div>")
    st.markdown("".join(partes), unsafe_allow_html=True)


def rodape_desenvolvedor() -> None:
    import streamlit as st

    st.markdown(
        '<p class="site-footer">developed by Lucas Maia</p>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, bool]]) -> None:
    """items: (label, valor, accent?)"""
    import streamlit as st

    parts = ['<div class="vel-kpi-row">']
    for lbl, val, accent in items:
        cls = "val val--accent" if accent else "val"
        parts.append(
            f'<div class="vel-kpi"><div class="lbl">{lbl}</div>'
            f'<div class="{cls}">{val}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def kpi_duo(label_esq: str, val_esq: str, label_dir: str, val_dir: str) -> None:
    import streamlit as st

    st.markdown(
        '<div class="vel-kpi-row vel-kpi-row--duo">'
        f'<div class="vel-kpi"><div class="lbl">{label_esq}</div>'
        f'<div class="val">{val_esq}</div></div>'
        f'<div class="vel-kpi"><div class="lbl">{label_dir}</div>'
        f'<div class="val">{val_dir}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def bloco_classificacao_time(
    time: str,
    pos_atual: int,
    pts_atual: int,
    pos_final: int,
    pts_final: int,
    vit_proj: int,
    saldo_proj: int,
) -> None:
    """Nome do time; abaixo, quatro boxes na mesma linha."""
    import streamlit as st

    saldo_txt = f"+{saldo_proj}" if saldo_proj > 0 else str(saldo_proj)
    st.markdown(
        '<div class="vel-time-evolucao-block">'
        f'<p class="vel-kpi-time-label">{time}</p>'
        '<div class="vel-kpi-row vel-kpi-row--quad">'
        '<div class="vel-kpi"><div class="lbl">Classificação Atual</div>'
        f'<div class="val">{pos_atual}º · {pts_atual} pts</div></div>'
        '<div class="vel-kpi"><div class="lbl">Classificação Final</div>'
        f'<div class="val">{pos_final}º · {pts_final} pts</div></div>'
        '<div class="vel-kpi"><div class="lbl">Vitórias Projetadas</div>'
        f'<div class="val">{vit_proj}</div></div>'
        '<div class="vel-kpi"><div class="lbl">Saldo de Gols Projetado</div>'
        f'<div class="val">{saldo_txt}</div></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def titulo_secao(texto: str) -> None:
    import streamlit as st

    st.markdown(f'<p class="secao-titulo">{texto}</p>', unsafe_allow_html=True)
