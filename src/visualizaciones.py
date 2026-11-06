"""Módulo de componentes visuales, gráficas y estilos para Tribuna Global."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# Mapeo de banderas por país para el selector y tarjetas
BANDERAS_PAISES: dict[str, str] = {
    "Argentina": "🇦🇷",
    "Portugal": "🇵🇹",
    "Brasil": "🇧🇷",
    "Francia": "🇫🇷",
    "Noruega": "🇳🇴",
    "Egipto": "🇪🇬",
    "Polonia": "🇵🇱",
    "Croacia": "🇭🇷",
    "Bélgica": "🇧🇪",
    "Uruguay": "🇺🇾",
    "Colombia": "🇨🇴",
    "Corea del Sur": "🇰🇷",
    "Senegal": "🇸🇳",
}

COLORES_SENTIMIENTOS = {
    "positivo": "#10B981",  # Verde esmeralda
    "negativo": "#EF4444",  # Rojo carmesí
    "neutro": "#3B82F6",    # Azul cielo
    "mixto": "#F59E0B",     # Ámbar dorado
}

ICONOS_SENTIMIENTOS = {
    "positivo": "🟢 Positivo",
    "negativo": "🔴 Negativo",
    "neutro": "🔵 Neutro",
    "mixto": "🟡 Mixto",
}


def obtener_bandera(pais: str) -> str:
    """Retorna el emoji de la bandera correspondiente al país."""
    return BANDERAS_PAISES.get(pais.strip(), "🌐")


def aplicar_estilos_personalizados():
    """Inyecta estilos CSS modernos y limpios para una experiencia premium."""
    st.markdown(
        """
        <style>
        /* Estilos generales */
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        /* Tarjetas de métricas */
        .metric-card {
            background-color: #F8FAFC;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }
        /* Insignia de sentimiento */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.8rem;
        }
        .badge-positivo {
            background-color: #DCFCE7;
            color: #166534;
            border: 1px solid #86EFAC;
        }
        .badge-negativo {
            background-color: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FCA5A5;
        }
        .badge-neutro {
            background-color: #DBEAFE;
            color: #1E40AF;
            border: 1px solid #93C5FD;
        }
        .badge-mixto {
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #FCD34D;
        }
        /* Contenedor de opinión */
        .opinion-card {
            background: white;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            padding: 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .opinion-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_insignia_sentimiento(sentimiento: str) -> str:
    """Genera HTML para una insignia coloreada del sentimiento."""
    s = sentimiento.lower().strip()
    clase = f"badge badge-{s}"
    texto = ICONOS_SENTIMIENTOS.get(s, s.capitalize())
    return f'<span class="{clase}">{texto}</span>'


def crear_grafico_distribucion(opiniones: list[dict]):
    """Genera un gráfico de dona interactivo con la distribución de sentimientos."""
    if not opiniones:
        return None

    df = pd.DataFrame(opiniones)
    conteo = df["sentimiento"].value_counts().reset_index()
    conteo.columns = ["Sentimiento", "Cantidad"]

    colores = [COLORES_SENTIMIENTOS.get(s, "#94A3B8") for s in conteo["Sentimiento"]]

    chart = (
        alt.Chart(conteo)
        .mark_arc(innerRadius=45, stroke="#fff", strokeWidth=2)
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
        .properties(height=260, title="Distribución de Sentimientos")
    )
    return chart


def crear_grafico_por_jugador(opiniones: list[dict]):
    """Genera un gráfico de barras apiladas comparando sentimientos por jugador."""
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
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
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
        .properties(height=320, title="Opiniones por Futbolista y Sentimiento")
    )
    return chart

