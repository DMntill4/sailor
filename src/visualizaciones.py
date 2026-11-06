"""Componentes visuales, diseño UI y graficas declarativas para Tribuna Global.

Proporciona estilos modernos con soporte para temas oscuros y claros,
iconos vectoriales SVG en linea sin emojis unicode y graficos interactivos con Altair.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# Mapeo de codigos de pais ISO de 3 letras (sin emojis de banderas que fallan en Windows)
CODIGOS_PAISES: dict[str, str] = {
    "Argentina": "ARG",
    "Portugal": "POR",
    "Brasil": "BRA",
    "Francia": "FRA",
    "Noruega": "NOR",
    "Egipto": "EGY",
    "Polonia": "POL",
    "Croacia": "CRO",
    "Bélgica": "BEL",
    "Uruguay": "URU",
    "Colombia": "COL",
    "Corea del Sur": "KOR",
    "Senegal": "SEN",
}

COLORES_SENTIMIENTOS = {
    "positivo": "#10B981",  # Esmeralda
    "negativo": "#EF4444",  # Carmesi
    "neutro": "#3B82F6",    # Azul cielo
    "mixto": "#F59E0B",     # Ambar
}

# Iconos SVG vectoriales limpios (estilo Lucide / Feather)
SVG_ICONS = {
    "positivo": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="18 15 12 9 6 15"></polyline>'
        '</svg>'
    ),
    "negativo": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="6 9 12 15 18 9"></polyline>'
        '</svg>'
    ),
    "neutro": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="5" y1="12" x2="19" y2="12"></line>'
        '</svg>'
    ),
    "mixto": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="16 3 21 3 21 8"></polyline>'
        '<line x1="4" y1="20" x2="21" y2="3"></line>'
        '<polyline points="21 16 21 21 16 21"></polyline>'
        '<line x1="15" y1="15" x2="21" y2="21"></line>'
        '<line x1="4" y1="4" x2="9" y2="9"></line>'
        '</svg>'
    ),
    "football": (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"></circle>'
        '<polygon points="12 6 15.5 8.5 14 12.5 10 12.5 8.5 8.5 12 6"></polygon>'
        '<line x1="12" y1="2" x2="12" y2="6"></line>'
        '<line x1="12" y1="22" x2="12" y2="17.5"></line>'
        '<line x1="2" y1="12" x2="6.5" y2="12"></line>'
        '<line x1="22" y1="12" x2="17.5" y2="12"></line>'
        '</svg>'
    ),
    "shield_alert": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
        '<line x1="12" y1="8" x2="12" y2="12"></line>'
        '<line x1="12" y1="16" x2="12.01" y2="16"></line>'
        '</svg>'
    ),
    "check": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="20 6 9 17 4 12"></polyline>'
        '</svg>'
    ),
    "user": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>'
        '<circle cx="12" cy="7" r="4"></circle>'
        '</svg>'
    ),
    "database": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>'
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>'
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>'
        '</svg>'
    ),
}


def obtener_codigo_pais(pais: str) -> str:
    """Retorna el codigo ISO de 3 letras del pais o los primeros 3 caracteres."""
    return CODIGOS_PAISES.get(pais.strip(), pais.strip()[:3].upper())


# Alias de retrocompatibilidad
obtener_bandera = obtener_codigo_pais


def aplicar_estilos_personalizados():
    """Inyecta estilos CSS tecnicos de alta fidelidad compatibles con Dark Mode y Light Mode."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Variables base y tipografia */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Cabecera de la aplicacion */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 0 0.75rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
            margin-bottom: 1.25rem;
        }

        .app-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        }

        .brand-title {
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0;
            line-height: 1.2;
            color: var(--text-color, #F8FAFC);
        }

        .brand-subtitle {
            font-size: 0.85rem;
            color: #94A3B8;
            margin: 0;
            font-weight: 500;
        }

        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(37, 99, 235, 0.12);
            color: #60A5FA;
            border: 1px solid rgba(96, 165, 250, 0.25);
        }

        /* Banner de seleccion de jugador */
        .player-hero-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.25rem;
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
            margin-bottom: 1.25rem;
        }

        .player-hero-info {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .player-hero-name {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin: 0;
        }

        .player-iso-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(148, 163, 184, 0.15);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.25);
            letter-spacing: 0.05em;
        }

        /* Insignias tecnicas de sentimiento con icono SVG */
        .sentiment-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-positivo {
            background-color: rgba(16, 185, 129, 0.12);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-negativo {
            background-color: rgba(239, 68, 68, 0.12);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .badge-neutro {
            background-color: rgba(59, 130, 246, 0.12);
            color: #3B82F6;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .badge-mixto {
            background-color: rgba(245, 158, 11, 0.12);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* Tarjeta de opinion */
        .opinion-card-tech {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 1.15rem;
            margin-bottom: 0.85rem;
            transition: border-color 0.15s ease;
        }

        .opinion-card-tech:hover {
            border-color: rgba(96, 165, 250, 0.4);
        }

        .opinion-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.65rem;
        }

        .opinion-card-player {
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-color, #F1F5F9);
        }

        .opinion-card-text {
            font-size: 0.96rem;
            line-height: 1.5;
            color: var(--text-color, #E2E8F0);
            margin-bottom: 0.65rem;
        }

        .opinion-card-reason {
            font-size: 0.86rem;
            color: #94A3B8;
            background: rgba(15, 23, 42, 0.45);
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            border-left: 3px solid #3B82F6;
            margin-bottom: 0.5rem;
        }

        .opinion-card-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.72rem;
            color: #64748B;
            font-family: 'JetBrains Mono', monospace;
            padding-top: 0.35rem;
            border-top: 1px solid rgba(148, 163, 184, 0.08);
        }

        /* Alertas limpias */
        .tech-alert-warn {
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: #FBBF24;
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_insignia_sentimiento(sentimiento: str) -> str:
    """Genera HTML semantico con icono SVG y estilos CSS para el sentimiento."""
    s = sentimiento.lower().strip()
    icono = SVG_ICONS.get(s, "")
    clase = f"sentiment-badge badge-{s}"
    return f'<span class="{clase}">{icono}<span>{s}</span></span>'


def renderizar_cabecera():
    """Genera la barra de cabecera principal con branding e identificadores tecnicos."""
    icon_svg = SVG_ICONS["football"]
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-brand">
                <div class="brand-icon">
                    {icon_svg}
                </div>
                <div>
                    <h1 class="brand-title">Tribuna Global</h1>
                    <p class="brand-subtitle">Pipeline de Clasificacion Semantica y Analisis de Percepcion</p>
                </div>
            </div>
            <div>
                <span class="brand-badge">LCEL &middot; Pydantic v2 &middot; Gemini / OpenAI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderizar_tarjeta_jugador(jugador: dict[str, str]):
    """Renderiza la tarjeta de estado del futbolista activo."""
    iso = obtener_codigo_pais(jugador.get("pais", ""))
    st.markdown(
        f"""
        <div class="player-hero-card">
            <div class="player-hero-info">
                <div>
                    <span class="player-iso-badge">{iso}</span>
                </div>
                <div>
                    <p class="player-hero-name">{jugador.get('nombre')}</p>
                    <span style="font-size: 0.78rem; color: #94A3B8;">Pais: {jugador.get('pais')} &middot; ID Canonico: <code>{jugador.get('id')}</code></span>
                </div>
            </div>
            <div>
                <span style="font-size: 0.75rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">ESTADO: ACTIVO</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def crear_grafico_distribucion(opiniones: list[dict]):
    """Genera un grafico de dona interactivo en Altair con la distribucion de sentimientos."""
    if not opiniones:
        return None

    df = pd.DataFrame(opiniones)
    conteo = df["sentimiento"].value_counts().reset_index()
    conteo.columns = ["Sentimiento", "Cantidad"]

    colores = [COLORES_SENTIMIENTOS.get(s, "#94A3B8") for s in conteo["Sentimiento"]]

    chart = (
        alt.Chart(conteo)
        .mark_arc(innerRadius=50, stroke="#0F172A", strokeWidth=2)
        .encode(
            theta=alt.Theta(field="Cantidad", type="quantitative"),
            color=alt.Color(
                field="Sentimiento",
                type="nominal",
                scale=alt.Scale(
                    domain=list(conteo["Sentimiento"]),
                    range=colores,
                ),
                legend=alt.Legend(title="Sentimiento", orient="bottom"),
            ),
            tooltip=["Sentimiento", "Cantidad"],
        )
        .properties(height=260, title="Distribucion Global de Polaridades")
        .configure_view(strokeWidth=0)
    )
    return chart


def crear_grafico_por_jugador(opiniones: list[dict]):
    """Genera un grafico de barras apiladas comparando sentimientos por futbolista."""
    if not opiniones:
        return None

    df = pd.DataFrame(opiniones)
    conteo = (
        df.groupby(["jugador_nombre", "sentimiento"])
        .size()
        .reset_index(name="Cantidad")
    )

    chart = (
        alt.Chart(conteo)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("jugador_nombre:N", title="Futbolista", axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("Cantidad:Q", title="Total de Opiniones"),
            color=alt.Color(
                "sentimiento:N",
                scale=alt.Scale(
                    domain=["positivo", "negativo", "neutro", "mixto"],
                    range=["#10B981", "#EF4444", "#3B82F6", "#F59E0B"],
                ),
                legend=alt.Legend(title="Sentimiento"),
            ),
            tooltip=["jugador_nombre", "sentimiento", "Cantidad"],
        )
        .properties(height=320, title="Agrupacion por Futbolista")
        .configure_view(strokeWidth=0)
    )
    return chart
