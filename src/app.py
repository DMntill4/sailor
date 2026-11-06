"""Aplicación principal Streamlit para Tribuna Global.

Gestión de la interfaz interactiva para el registro de opiniones de aficionados
sobre futbolistas internacionales, análisis semántico con LangChain y
consulta estructurada con filtros, métricas y descarga de datos.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Asegurar que el directorio raíz y src estén en sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.almacenamiento import (  # noqa: E402
    DATOS_DIR,
    RUTA_JUGADORES_DEFAULT,
    RUTA_OPINIONES_DEFAULT,
    calcular_metricas,
    cargar_jugadores,
    cargar_opiniones,
    consultar_opiniones,
    crear_registro,
    guardar_opinion,
)
from src.analisis import (  # noqa: E402
    MODELO_DEFAULT,
    VERSION_PROMPT,
    analizar_opinion,
    analizar_opinion_mock,
    obtener_modelo_llm,
    validar_longitud_opinion,
)
from src.visualizaciones import (  # noqa: E402
    aplicar_estilos_personalizados,
    crear_grafico_distribucion,
    crear_grafico_por_jugador,
    obtener_bandera,
    renderizar_insignia_sentimiento,
)


def inicializar_estado():
    """Inicializa las variables necesarias en st.session_state."""
    if "mensajes_chat" not in st.session_state:
        st.session_state.mensajes_chat = []
    if "ultima_opinion_guardada" not in st.session_state:
        st.session_state.ultima_opinion_guardada = None
    if "ids_procesados" not in st.session_state:
        st.session_state.ids_procesados = set()


def formatear_fecha(iso_fecha: str) -> str:
    """Convierte una fecha ISO 8601 a formato legible local."""
    try:
        dt = datetime.fromisoformat(iso_fecha)
        return dt.strftime("%d/%m/%Y %H:%M UTC")
    except Exception:
        return iso_fecha


def main():
    st.set_page_config(
        page_title="Tribuna Global | Percepción de Futbolistas",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    aplicar_estilos_personalizados()
    inicializar_estado()

    # Cargar catálogo de jugadores con manejo de errores
    try:
        jugadores = cargar_jugadores(RUTA_JUGADORES_DEFAULT)
    except Exception as exc:
        st.error(f"❌ Error crítico al cargar el catálogo de futbolistas: {exc}")
        st.stop()

    # -------------------------------------------------------------------------
    # Barra Lateral: Configuración y Selección
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🏟️ Tribuna Global")
        st.caption("Percepción de aficionados con IA y LangChain")
        st.divider()

        st.subheader("1. Selección de Futbolista")
        # Selector de futbolista mostrando nombre, bandera y país
        opciones_jugadores = {
            f"{obtener_bandera(j['pais'])} {j['nombre']} ({j['pais']})": j
            for j in jugadores
        }
        etiqueta_seleccionada = st.selectbox(
            "Selecciona un jugador:",
            options=list(opciones_jugadores.keys()),
            index=0,
            help="El identificador interno del futbolista quedará capturado al enviar la opinión.",
        )
        jugador_activo = opciones_jugadores[etiqueta_seleccionada]

        st.info(
            f"**ID Interno:** `{jugador_activo['id']}`\n\n"
            f"**País:** {jugador_activo['pais']} {obtener_bandera(jugador_activo['pais'])}"
        )

        st.divider()
        st.subheader("2. Configuración de IA")

        api_key_env = os.getenv("OPENAI_API_KEY", "")
        tiene_key_env = bool(api_key_env and api_key_env != "clave_configurada_localmente")

        api_key_input = st.text_input(
            "OpenAI API Key:",
            value=api_key_env if tiene_key_env else "",
            type="password",
            help="Si no tienes API Key activa, puedes activar el 'Modo Demostración / Offline' abajo.",
        )

        modelo_seleccionado = st.selectbox(
            "Modelo LLM:",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Se utiliza temperatura 0.0 para mayor estabilidad y consistencia.",
        )

        modo_offline = st.checkbox(
            "🧪 Modo Demostración (Motor Offline)",
            value=not tiene_key_env and not api_key_input,
            help="Permite probar y evaluar todos los casos del taller de manera determinista sin consumir créditos de API.",
        )

        st.divider()
        # Botón de nueva conversación: limpia historial en memoria sin borrar archivo
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.mensajes_chat = []
            st.success("Conversación de la sesión reiniciada. Las opiniones en disco se conservan intactas.")
            st.rerun()

    # -------------------------------------------------------------------------
    # Encabezado Principal y Descargo de Responsabilidad
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-header">⚽ Tribuna Global</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Plataforma de registro y análisis de percepción sobre futbolistas de renombre internacional.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Acerca de la clasificación y reglas del sistema", expanded=False):
        st.markdown(
            """
            - **Descargo:** La clasificación describe únicamente lo expresado por el aficionado; no establece la calidad deportiva objetiva del futbolista.
            - **Alcance:** Las cifras representan las opiniones registradas en esta aplicación, no la opinión mundial.
            - **Independencia:** Cada opinión se analiza de forma aislada sin contaminar el contexto con comentarios anteriores.
            - **Categorías:** `positivo`, `negativo`, `neutro` y `mixto` (cuando coexisten elogios y críticas relevantes).
            - **Pertinencia:** Comentarios ajenos al jugador son descartados y no se registran.
            """
        )

    # -------------------------------------------------------------------------
    # Pestañas de la Aplicación
    # -------------------------------------------------------------------------
    tab_emitir, tab_consultar, tab_resumen = st.tabs([
        "💬 Emitir Opinión",
        "📊 Consultar Opiniones",
        "🏆 Explorador de Figuras",
    ])

    # =========================================================================
    # PESTAÑA 1: Emitir Opinión
    # =========================================================================
    with tab_emitir:
        st.markdown(
            f"### Opinando sobre: **{jugador_activo['nombre']}** {obtener_bandera(jugador_activo['pais'])}"
        )
        st.caption(
            "Escribe tu opinión honesta (entre 5 y 1000 caracteres). El sistema analizará el sentimiento y guardará el registro persistente."
        )

        # Contenedor para renderizar el historial de la sesión
        for msg in st.session_state.mensajes_chat:
            rol = msg.get("rol", "user")
            with st.chat_message(rol):
                if rol == "user":
                    st.markdown(f"**Para {msg.get('jugador_nombre')}:** {msg.get('contenido')}")
                else:
                    if msg.get("es_pertinente", True):
                        insignia = renderizar_insignia_sentimiento(msg.get("sentimiento", "neutro"))
                        st.markdown(
                            f"{insignia} &nbsp; **Análisis:** {msg.get('razon')}",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Registro ID: `{msg.get('id', 'N/A')}` | Modelo: `{msg.get('modelo', 'N/A')}`"
                        )
                    else:
                        st.warning(
                            f"⚠️ **Mensaje no pertinente:** {msg.get('razon')}\n\n*(Este mensaje no fue almacenado en los registros)*"
                        )

        # Entrada de chat del usuario
        prompt_placeholder = f"Escribe tu opinión sobre {jugador_activo['nombre']}..."
        opinion_usuario = st.chat_input(prompt_placeholder)

        if opinion_usuario is not None:
            # 1. Validación de longitud de entrada (5 a 1000 caracteres)
            try:
                opinion_limpia = validar_longitud_opinion(opinion_usuario)
            except ValueError as ve:
                st.warning(f"⚠️ {ve}")
                opinion_limpia = None

            if opinion_limpia:
                # Mostrar el mensaje del usuario inmediatamente en la sesión
                with st.chat_message("user"):
                    st.markdown(f"**Para {jugador_activo['nombre']}:** {opinion_limpia}")

                # Capturar los datos del jugador en el momento exacto del envío
                jugador_al_enviar = dict(jugador_activo)

                # Realizar el análisis de sentimiento
                with st.chat_message("assistant"):
                    with st.spinner("Analizando semántica y sentimiento de la opinión..."):
                        error_analisis = False
                        analisis = None

                        if modo_offline:
                            analisis = analizar_opinion_mock(jugador_al_enviar["nombre"], opinion_limpia)
                            modelo_usado = "mock-evaluator-v1"
                        else:
                            try:
                                chat_llm = obtener_modelo_llm(
                                    api_key=api_key_input or None,
                                    modelo_nombre=modelo_seleccionado,
                                    temperatura=0.0,
                                )
                                analisis = analizar_opinion(
                                    jugador=jugador_al_enviar,
                                    texto=opinion_limpia,
                                    chat_model=chat_llm,
                                )
                                modelo_usado = modelo_seleccionado
                            except Exception as exc:
                                error_analisis = True
                                st.error(
                                    f"❌ **Error en la clasificación del modelo:** {exc}\n\n"
                                    "No se generó un sentimiento inventado y el registro NO fue guardado."
                                )

                    if not error_analisis and analisis is not None:
                        # 2. Evaluación de pertinencia
                        if not analisis.es_pertinente:
                            st.warning(
                                f"⚠️ **Opinión no pertinente:** {analisis.razon}\n\n"
                                "Por las reglas del sistema, no se registrará como opinión válida."
                            )
                            st.session_state.mensajes_chat.append({
                                "rol": "user",
                                "jugador_nombre": jugador_al_enviar["nombre"],
                                "contenido": opinion_limpia,
                            })
                            st.session_state.mensajes_chat.append({
                                "rol": "assistant",
                                "es_pertinente": False,
                                "razon": analisis.razon,
                            })
                        else:
                            # 3. Creación y guardado seguro del registro persistente
                            try:
                                nuevo_registro = crear_registro(
                                    jugador=jugador_al_enviar,
                                    texto=opinion_limpia,
                                    analisis=analisis,
                                    modelo=modelo_usado,
                                    version_prompt=VERSION_PROMPT,
                                )
                                guardado_ok = guardar_opinion(
                                    RUTA_OPINIONES_DEFAULT, nuevo_registro
                                )

                                if guardado_ok:
                                    st.toast("✅ ¡Opinión guardada exitosamente en el archivo persistente!", icon="💾")
                                    insignia = renderizar_insignia_sentimiento(analisis.sentimiento)
                                    st.markdown(
                                        f"{insignia} &nbsp; **Análisis:** {analisis.razon}",
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(
                                        f"ID: `{nuevo_registro['id']}` | Fecha: `{formatear_fecha(nuevo_registro['fecha_utc'])}`"
                                    )

                                    # Agregar a la memoria de la sesión
                                    st.session_state.mensajes_chat.append({
                                        "rol": "user",
                                        "jugador_nombre": jugador_al_enviar["nombre"],
                                        "contenido": opinion_limpia,
                                    })
                                    st.session_state.mensajes_chat.append({
                                        "rol": "assistant",
                                        "es_pertinente": True,
                                        "sentimiento": analisis.sentimiento,
                                        "razon": analisis.razon,
                                        "id": nuevo_registro["id"],
                                        "modelo": modelo_usado,
                                    })
                            except Exception as err_guardado:
                                st.error(
                                    f"❌ **Error al persistir el registro:** {err_guardado}\n\n"
                                    "No se emitió confirmación de guardado. Tu opinión original se conserva en pantalla para evitar pérdida."
                                )

    # =========================================================================
    # PESTAÑA 2: Consultar Opiniones
    # =========================================================================
    with tab_consultar:
        st.markdown("### 📊 Consultas y Auditoría de Opiniones")
        st.caption(
            "Filtra y explora las opiniones persistidas en disco. Esta sección opera completamente en memoria sin invocar al modelo LLM."
        )

        # Cargar opiniones persistidas
        try:
            datos_persistidos = cargar_opiniones(RUTA_OPINIONES_DEFAULT)
            todas_opiniones = datos_persistidos.get("opiniones", [])
        except Exception as exc:
            st.error(f"❌ Error al leer el archivo de opiniones persistido: {exc}")
            todas_opiniones = []

        # Controles de filtrado
        col_f1, col_f2, col_f3 = st.columns([1.5, 1.2, 1.3])

        with col_f1:
            nombres_jugadores_filtro = ["Todos los futbolistas"] + sorted(
                list({j["nombre"] for j in jugadores})
            )
            filtro_jugador_nombre = st.selectbox(
                "Filtrar por futbolista:",
                options=nombres_jugadores_filtro,
                index=0,
            )

        with col_f2:
            filtro_sentimiento = st.selectbox(
                "Filtrar por sentimiento:",
                options=["Todos", "positivo", "negativo", "neutro", "mixto"],
                index=0,
            )

        with col_f3:
            filtro_texto = st.text_input(
                "Buscar en texto o razón:",
                value="",
                placeholder="Ej. visión, pases, velocidad...",
            )

        # Determinar ID del jugador si se seleccionó uno específico
        id_jugador_filtro = None
        if filtro_jugador_nombre != "Todos los futbolistas":
            for j in jugadores:
                if j["nombre"] == filtro_jugador_nombre:
                    id_jugador_filtro = j["id"]
                    break

        sentimiento_param = None if filtro_sentimiento == "Todos" else filtro_sentimiento

        # Ejecutar consulta en memoria sin llamadas al LLM
        opiniones_filtradas = consultar_opiniones(
            todas_opiniones,
            jugador_id=id_jugador_filtro,
            sentimiento=sentimiento_param,
            busqueda_texto=filtro_texto,
        )

        # Métricas resumidas
        metricas = calcular_metricas(opiniones_filtradas)
        total_f = metricas["total"]
        conteos = metricas["conteos"]
        pcts = metricas["porcentajes"]

        st.markdown("#### Métricas del subconjunto consultado")
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.metric("Total Opiniones", total_f)
        with m_c2:
            st.metric("🟢 Positivas", f"{conteos['positivo']} ({pcts['positivo']}%)")
        with m_c3:
            st.metric("🔴 Negativas", f"{conteos['negativo']} ({pcts['negativo']}%)")
        with m_c4:
            st.metric("🔵 Neutras", f"{conteos['neutro']} ({pcts['neutro']}%)")
        with m_c5:
            st.metric("🟡 Mixtas", f"{conteos['mixto']} ({pcts['mixto']}%)")

        st.divider()

        # Gráficas interactivas
        if opiniones_filtradas:
            g_col1, g_col2 = st.columns([1, 1.4])
            with g_col1:
                chart_dist = crear_grafico_distribucion(opiniones_filtradas)
                if chart_dist:
                    st.altair_chart(chart_dist, use_container_width=True)
            with g_col2:
                chart_bar = crear_grafico_por_jugador(opiniones_filtradas)
                if chart_bar:
                    st.altair_chart(chart_bar, use_container_width=True)

            st.divider()

            # Botones de descarga
            d_col1, d_col2 = st.columns([1, 1])
            with d_col1:
                json_data = json.dumps(
                    {"total": total_f, "opiniones": opiniones_filtradas},
                    ensure_ascii=False,
                    indent=2,
                )
                st.download_button(
                    label="📥 Descargar resultados como JSON",
                    data=json_data,
                    file_name="opiniones_filtradas.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with d_col2:
                df_export = pd.DataFrame(opiniones_filtradas)
                csv_data = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar resultados como CSV",
                    data=csv_data,
                    file_name="opiniones_filtradas.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown(f"#### Listado de opiniones ({total_f} registros)")

            # Renderizar tarjetas de opiniones
            for op in reversed(opiniones_filtradas):
                with st.container():
                    st.markdown(
                        f"""
                        <div class="opinion-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <strong>⚽ {op.get('jugador_nombre')}</strong>
                                {renderizar_insignia_sentimiento(op.get('sentimiento', 'neutro'))}
                            </div>
                            <p style="font-size: 1rem; color: #1E293B; margin-bottom: 0.5rem;">
                                <em>«{op.get('opinion_original')}»</em>
                            </p>
                            <p style="font-size: 0.88rem; color: #475569; margin-bottom: 0.3rem;">
                                <strong>Justificación:</strong> {op.get('razon')}
                            </p>
                            <div style="font-size: 0.75rem; color: #94A3B8;">
                                ID: <code>{op.get('id')}</code> | Fecha: {formatear_fecha(op.get('fecha_utc'))} | Modelo: {op.get('modelo')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("ℹ️ No se encontraron opiniones que coincidan con los filtros seleccionados.")

    # =========================================================================
    # PESTAÑA 3: Explorador de Figuras (x100 cooler)
    # =========================================================================
    with tab_resumen:
        st.markdown("### 🏆 Explorador y Análisis de Figuras")
        st.caption(
            "Visión panorámica de la percepción de cada futbolista según los registros de la aplicación."
        )

        try:
            datos_todas = cargar_opiniones(RUTA_OPINIONES_DEFAULT).get("opiniones", [])
        except Exception:
            datos_todas = []

        if not datos_todas:
            st.warning("No hay opiniones registradas para generar el análisis comparativo.")
        else:
            df_todas = pd.DataFrame(datos_todas)
            resumen_jugador = (
                df_todas.groupby("jugador_nombre")
                .agg(
                    Total=("id", "count"),
                    Positivas=("sentimiento", lambda s: (s == "positivo").sum()),
                    Negativas=("sentimiento", lambda s: (s == "negativo").sum()),
                    Neutras=("sentimiento", lambda s: (s == "neutro").sum()),
                    Mixtas=("sentimiento", lambda s: (s == "mixto").sum()),
                )
                .reset_index()
            )

            resumen_jugador["% Aprobación"] = (
                (resumen_jugador["Positivas"] / resumen_jugador["Total"]) * 100
            ).round(1)

            st.dataframe(
                resumen_jugador.sort_values(by="Total", ascending=False),
                column_config={
                    "jugador_nombre": "Futbolista",
                    "Total": st.column_config.NumberColumn("Opiniones"),
                    "Positivas": st.column_config.NumberColumn("🟢 Positivas"),
                    "Negativas": st.column_config.NumberColumn("🔴 Negativas"),
                    "Neutras": st.column_config.NumberColumn("🔵 Neutras"),
                    "Mixtas": st.column_config.NumberColumn("🟡 Mixtas"),
                    "% Aprobación": st.column_config.ProgressColumn(
                        "Tasa de Aprobación",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
