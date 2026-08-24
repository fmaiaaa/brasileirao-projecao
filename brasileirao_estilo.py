"""Estilo visual do app Brasileirão (inspirado no velocímetro, tema futebol)."""
from __future__ import annotations

# Paleta futebol (gramado / estádio) - sem cores Direcional
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
            color-scheme: only light !important;
        }}
        /* Força tema claro mesmo com SO/navegador em dark mode */
        @media (prefers-color-scheme: dark) {{
            html, body, :root, [data-testid="stApp"], .stApp {{
                color-scheme: only light !important;
            }}
        }}
        html[data-theme="dark"],
        body[data-theme="dark"],
        .stApp[data-theme="dark"],
        [data-testid="stApp"][data-theme="dark"] {{
            color-scheme: only light !important;
        }}
        :root {{
            --background-color: #ffffff !important;
            --secondary-background-color: #f0fdf4 !important;
            --text-color: {COR_TEXTO} !important;
            --primary-color: {COR_VERDE} !important;
        }}
        html, body {{
            font-family: 'Inter', sans-serif;
            color: {COR_TEXTO} !important;
            background: transparent !important;
        }}
        /* Widgets / tabelas sempre claros */
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stRadio"] label,
        [data-testid="stMultiSelect"] label,
        .stSelectbox label {{
            color: {COR_TEXTO_MUTED} !important;
        }}
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {{
            color-scheme: only light !important;
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        [data-baseweb="popover"] ul,
        [data-baseweb="menu"] {{
            background-color: {COR_INPUT_BG} !important;
            color: {COR_TEXTO} !important;
        }}
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {{
            background-color: rgba(255, 255, 255, 0.85) !important;
            color: {COR_TEXTO} !important;
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
        .ficha-hero-logo-wrap {{
            text-align: center;
            margin: 0 auto 0 auto;
            max-width: 240px;
            line-height: 0;
        }}
        .ficha-hero-logo-wrap img {{
            width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            filter: drop-shadow(0 2px 6px rgba(15, 23, 42, 0.12));
        }}
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
        /* Coluna Time fixa na tabela de estatísticas (fallback CSS) */
        [data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
            overflow: auto !important;
        }}
        [data-testid="stDataFrame"] .gdg-sticky-left {{
            z-index: 2 !important;
            background: #ffffff !important;
            box-shadow: 2px 0 6px rgba(15, 23, 42, 0.08);
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
        .secao-titulo--grafico {{
            margin: 0.75rem 0 0.55rem 0;
        }}
        @media (max-width: 768px) {{
            .secao-titulo--grafico {{
                margin: 0.35rem 0 0.1rem 0 !important;
            }}
            [data-testid="stPlotlyChart"] {{
                margin-top: 0 !important;
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}
            [data-testid="stExpander"]:has(.grafico-legenda-list) {{
                margin-top: 0.15rem !important;
                margin-bottom: 0.35rem !important;
            }}
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
        .vel-kpi-row--quad .vel-kpi .lbl {{
            font-size: 0.68rem;
            line-height: 1.2;
            min-height: 2.1em;
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

        /* Legendas dos gráficos: PC no Plotly; mobile em expander abaixo */
        .grafico-legenda-list {{
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            padding: 0.15rem 0 0.25rem 0;
        }}
        .grafico-legenda-item {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.88rem;
            color: {COR_TEXTO};
            line-height: 1.35;
        }}
        .grafico-legenda-swatch {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            flex-shrink: 0;
            border: 1px solid rgba(15, 23, 42, 0.12);
        }}
        /* Expanders de legenda: só no celular (PC usa legenda do Plotly) */
        @media (min-width: 769px) {{
            [data-testid="stExpander"]:has(.grafico-legenda-list) {{
                display: none !important;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
            }}
        }}
        @media (max-width: 768px) {{
            [data-testid="stExpander"]:has(.grafico-legenda-list) {{
                display: block !important;
                margin-top: 0.15rem !important;
            }}
            [data-testid="stPlotlyChart"] .bartext,
            [data-testid="stPlotlyChart"] .textpoint,
            [data-testid="stPlotlyChart"] g.textpoint {{
                display: none !important;
                visibility: hidden !important;
            }}
        }}

        /* --- Mobile: responsivo sem alterar desktop --- */
        @media (max-width: 768px) {{
            [data-testid="stMain"] {{
                padding: 10px 8px !important;
            }}
            .block-container {{
                max-width: 100% !important;
                padding: 1rem 0.85rem 1.15rem 0.85rem !important;
                border-radius: 16px !important;
            }}
            div[data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
                flex-wrap: wrap !important;
                gap: 0.35rem !important;
            }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
            .vel-kpi-row {{
                gap: 8px;
            }}
            .vel-kpi {{
                flex: 1 1 calc(50% - 4px) !important;
                min-width: calc(50% - 4px) !important;
                padding: 12px 10px;
            }}
            .vel-kpi-row--duo,
            .vel-kpi-row--quad {{
                flex-wrap: wrap !important;
            }}
            .vel-kpi-row--duo .vel-kpi,
            .vel-kpi-row--quad .vel-kpi {{
                flex: 1 1 calc(50% - 5px) !important;
                min-width: calc(50% - 5px) !important;
            }}
            .vel-kpi-row--quad .vel-kpi .lbl {{
                min-height: auto;
            }}
            .ficha-hero-logo-wrap {{
                max-width: 180px;
            }}
            .secao-titulo {{
                font-size: 1.05rem;
                margin: 1rem 0 0.65rem 0;
            }}
            [data-testid="stDataFrame"],
            [data-testid="stTable"] {{
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch;
            }}
            [data-testid="stPlotlyChart"] {{
                width: 100% !important;
                overflow-x: hidden !important;
                margin-top: 0 !important;
                margin-bottom: 0 !important;
            }}
            [data-testid="stPlotlyChart"] .legend,
            [data-testid="stPlotlyChart"] g.legend,
            [data-testid="stPlotlyChart"] .inlegend {{
                display: none !important;
                visibility: hidden !important;
            }}
            [data-testid="stPlotlyChart"] > div,
            [data-testid="stPlotlyChart"] .js-plotly-plot,
            [data-testid="stPlotlyChart"] .plot-container {{
                width: 100% !important;
                max-width: 100% !important;
            }}
            /* Sem interação nos gráficos no celular: só deslizar a página */
            [data-testid="stPlotlyChart"] {{
                pointer-events: none !important;
                touch-action: pan-y !important;
                user-select: none !important;
                -webkit-user-select: none !important;
                -webkit-touch-callout: none !important;
            }}
            [data-testid="stPlotlyChart"] * {{
                pointer-events: none !important;
                touch-action: pan-y !important;
            }}
            .js-plotly-plot .modebar {{
                display: none !important;
            }}
            [data-testid="stRadio"] > div {{
                flex-direction: column !important;
                gap: 0.35rem !important;
            }}
            [data-testid="stRadio"] label {{
                width: 100% !important;
            }}
        }}

        @media (max-width: 420px) {{
            .vel-kpi,
            .vel-kpi-row--duo .vel-kpi,
            .vel-kpi-row--quad .vel-kpi {{
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def bloquear_graficos_mobile() -> None:
    """Injeta JS (via components) que desliga interação Plotly no celular."""
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        (function () {
          const root = window.parent.document;
          function isMobile() {
            try {
              return window.parent.matchMedia("(max-width: 768px)").matches
                || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
            } catch (e) {
              return /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
            }
          }
          function lockOnce(gd) {
            if (!gd || gd.__brLocked) return;
            gd.__brLocked = true;
            try {
              if (window.parent.Plotly) {
                window.parent.Plotly.relayout(gd, {
                  dragmode: false,
                  hovermode: false
                });
              }
            } catch (e) {}
            try {
              // Remove handlers de drag/zoom
              if (gd._context) {
                gd._context.staticPlot = true;
              }
            } catch (e) {}
          }
          function lockAll() {
            if (!isMobile()) return;
            root.querySelectorAll("[data-testid='stPlotlyChart']").forEach(function (wrap) {
              wrap.style.setProperty("pointer-events", "none", "important");
              wrap.style.setProperty("touch-action", "pan-y", "important");
              wrap.querySelectorAll("*").forEach(function (el) {
                el.style.setProperty("pointer-events", "none", "important");
                el.style.setProperty("touch-action", "pan-y", "important");
              });
            });
            root.querySelectorAll(".js-plotly-plot").forEach(lockOnce);
          }
          lockAll();
          if (!window.parent.__brPlotlyLockObs) {
            window.parent.__brPlotlyLockObs = new MutationObserver(function () {
              lockAll();
            });
            window.parent.__brPlotlyLockObs.observe(root.body, {
              childList: true,
              subtree: true
            });
            window.parent.addEventListener("resize", lockAll);
            window.parent.addEventListener("orientationchange", lockAll);
          }
          setTimeout(lockAll, 200);
          setTimeout(lockAll, 800);
          setTimeout(lockAll, 1600);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _logo_brasileirao_base64() -> str | None:
    """Remove fundo claro/neutro de R.png e retorna PNG em base64."""
    import base64
    import io
    from functools import lru_cache
    from pathlib import Path

    import numpy as np
    from PIL import Image

    @lru_cache(maxsize=1)
    def _processar() -> str | None:
        logo = Path(__file__).resolve().parent / "R.png"
        if not logo.is_file():
            return None

        img = Image.open(logo).convert("RGBA")
        data = np.array(img, dtype=np.float32)
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        lum = (r + g + b) / 3.0
        # Fundo cinza/branco (inclui padrão xadrez embutido na imagem)
        fundo = (lum > 210) & (chroma < 25)
        data[:, :, 3] = np.where(fundo, 0.0, 255.0)

        buf = io.BytesIO()
        Image.fromarray(data.astype(np.uint8), mode="RGBA").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return _processar()


def cabecalho_pagina(titulo: str, subtitulo: str | None = None) -> None:
    import streamlit as st

    partes = ['<div class="ficha-hero-stack">']
    b64 = _logo_brasileirao_base64()
    if b64:
        partes.append(
            '<div class="ficha-hero-logo-wrap">'
            f'<img src="data:image/png;base64,{b64}" alt="Brasileirão" />'
            "</div>"
        )
    partes.extend([
        '<div class="ficha-hero">',
        f'<p class="ficha-title">{titulo}</p>',
    ])
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
    pts_atual: float,
    pos_final: int,
    pts_final: float,
    *,
    prob_campeao: float | None = None,
    prob_g4: float | None = None,
    prob_g6: float | None = None,
    prob_z4: float | None = None,
) -> None:
    """Nome do time; classificação atual/final; opcionalmente probs de cenário."""
    import streamlit as st

    def _fmt_pts(v: float) -> str:
        return f"{v:.1f}" if abs(v - round(v)) > 0.05 else str(int(round(v)))

    def _fmt_pct(v: float) -> str:
        return f"{100.0 * v:.1f}%"

    html = (
        '<div class="vel-time-evolucao-block">'
        f'<p class="vel-kpi-time-label">{time}</p>'
        '<div class="vel-kpi-row vel-kpi-row--duo">'
        '<div class="vel-kpi"><div class="lbl">Classificação Final</div>'
        f'<div class="val">{pos_final}º · {_fmt_pts(pts_final)} pts</div></div>'
        '<div class="vel-kpi"><div class="lbl">Classificação Atual</div>'
        f'<div class="val">{pos_atual}º · {_fmt_pts(pts_atual)} pts</div></div>'
        "</div>"
    )
    if None not in (prob_campeao, prob_g4, prob_g6, prob_z4):
        html += (
            '<div class="vel-kpi-row vel-kpi-row--quad">'
            '<div class="vel-kpi"><div class="lbl">Probabilidade de ser campeão</div>'
            f'<div class="val val--accent">{_fmt_pct(prob_campeao)}</div></div>'
            '<div class="vel-kpi"><div class="lbl">Probabilidade de G4</div>'
            f'<div class="val">{_fmt_pct(prob_g4)}</div></div>'
            '<div class="vel-kpi"><div class="lbl">Probabilidade de G6</div>'
            f'<div class="val">{_fmt_pct(prob_g6)}</div></div>'
            '<div class="vel-kpi"><div class="lbl">Probabilidade de Z4</div>'
            f'<div class="val">{_fmt_pct(prob_z4)}</div></div>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def titulo_secao(texto: str) -> None:
    import streamlit as st

    st.markdown(f'<p class="secao-titulo">{texto}</p>', unsafe_allow_html=True)


def titulo_grafico(texto: str) -> None:
    """Título acima do gráfico, no mesmo estilo das seções."""
    import streamlit as st

    st.markdown(
        f'<p class="secao-titulo secao-titulo--grafico">{texto}</p>',
        unsafe_allow_html=True,
    )
