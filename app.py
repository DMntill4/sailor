"""Aplicacion principal Streamlit para Tribuna Global.

Pipeline de clasificacion semantica y analisis de percepcion de aficionados
sobre futbolistas internacionales con LangChain y modelos estructurados.
Diseno tecnico libre de emojis con soporte Dark/Light Mode e iconos vectoriales SVG.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Rutas base
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Limpiar cache de modulos en memoria de Streamlit para garantizar lectura de archivos actualizados
for _mod in [k for k in list(sys.modules.keys()) if k.startswith("src.")]:
    del sys.modules[_mod]

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


def obtener_codigo_pais(pais: str) -> str:
    """Retorna el codigo ISO de 3 letras del pais."""
    return CODIGOS_PAISES.get(pais.strip(), pais.strip()[:3].upper())


from src.almacenamiento import (  # noqa: E402
    RUTA_JUGADORES_DEFAULT,
    RUTA_OPINIONES_DEFAULT,
    actualizar_opinion,
    calcular_metricas,
    cargar_jugadores,
    cargar_opiniones,
    consultar_opiniones,
    crear_registro,
    guardar_opinion,
)
from src.analisis import (  # noqa: E402
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
    renderizar_cabecera,
    renderizar_insignia_sentimiento,
    renderizar_tarjeta_jugador,
)


def inicializar_estado():
    """Inicializa variables de sesion."""
    if "mensajes_chat" not in st.session_state:
        st.session_state.mensajes_chat = []


def formatear_fecha(iso_fecha: str) -> str:
    """Convierte fecha ISO 8601 a formato tecnico legible."""
    try:
        dt = datetime.fromisoformat(iso_fecha)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_fecha


def main():
    st.set_page_config(
        page_title="Tribuna Global | Analisis de Percepcion",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    aplicar_estilos_personalizados()
    inicializar_estado()

    # Cargar catalogo
    try:
        jugadores = cargar_jugadores(RUTA_JUGADORES_DEFAULT)
    except Exception as exc:
        st.error(f"Error critico al cargar el catalogo de futbolistas: {exc}")
        st.stop()

    # -------------------------------------------------------------------------
    # Barra Lateral
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Panel de Control")
        st.caption("Configuracion del sistema e inferencia")
        st.divider()

        st.markdown("**1. Futbolista Seleccionado**")
        opciones_jugadores = {
            f"[{obtener_codigo_pais(j['pais'])}] {j['nombre']} ({j['pais']})": j
            for j in jugadores
        }
        etiqueta_seleccionada = st.selectbox(
            "Seleccionar jugador:",
            options=list(opciones_jugadores.keys()),
            index=0,
            label_visibility="collapsed",
            help="El identificador unico queda capturado inmutablemente al enviar la opinion.",
        )
        jugador_activo = opciones_jugadores[etiqueta_seleccionada]

        st.caption(
            f"ID Canonico: `{jugador_activo['id']}` | Pais: `{jugador_activo['pais']}` ({obtener_codigo_pais(jugador_activo['pais'])})"
        )

        st.divider()
        st.markdown("**2. Motor de Inferencia**")

        proveedor_opcion = st.selectbox(
            "Proveedor LLM:",
            options=["Google Gemini (Capa Gratuita)", "OpenAI"],
            index=0,
            help="Google Gemini ofrece una capa gratuita oficial para desarrollo y pruebas.",
        )

        es_gemini = "gemini" in proveedor_opcion.lower()

        if es_gemini:
            st.markdown(
                "[Obtener clave gratuita en Google AI Studio](https://aistudio.google.com/app/apikey)"
            )
            gemini_key_env = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            tiene_key = bool(gemini_key_env and gemini_key_env != "tu_clave_de_gemini_aqui")

            api_key_input = st.text_input(
                "Gemini API Key:",
                value=gemini_key_env if tiene_key else "",
                type="password",
                placeholder="Ingresar clave o definir GEMINI_API_KEY en .env",
            )

            modelo_seleccionado = st.selectbox(
                "Modelo Gemini:",
                options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"],
                index=0,
            )
            proveedor_str = "gemini"
        else:
            openai_key_env = os.getenv("OPENAI_API_KEY", "")
            tiene_key = bool(openai_key_env and openai_key_env != "clave_configurada_localmente")

            api_key_input = st.text_input(
                "OpenAI API Key:",
                value=openai_key_env if tiene_key else "",
                type="password",
                placeholder="Ingresar clave o definir OPENAI_API_KEY en .env",
            )

            modelo_seleccionado = st.selectbox(
                "Modelo OpenAI:",
                options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
            )
            proveedor_str = "openai"

        modo_offline = st.checkbox(
            "Motor Offline (Modo Demostracion)",
            value=not tiene_key and not api_key_input,
            help="Evalua deterministicamente los casos de prueba del taller sin llamadas externas a la API.",
        )

        st.divider()
        if st.button("Nueva Conversacion", use_container_width=True):
            st.session_state.mensajes_chat = []
            st.success("Historial de sesion reiniciado. Los registros en disco se conservan intactos.")
            st.rerun()

    # -------------------------------------------------------------------------
    # Cabecera Principal y Reglas
    # -------------------------------------------------------------------------
    renderizar_cabecera()

    with st.expander("Especificaciones tecnicas y reglas de clasificacion del sistema", expanded=False):
        st.markdown(
            """
            - **Descargo Metodologico:** La clasificacion describe exclusivamente la percepcion expresada por el usuario en el texto; no establece la calidad deportiva objetiva del futbolista.
            - **Aislamiento de Inferencia:** Cada opinion se procesa de forma independiente sin inyectar el historial previo al clasificador.
            - **Categorias:** `positivo`, `negativo`, `neutro` (declaraciones puramente factuales o estadisticas) y `mixto` (coexistencia de elogios y objeciones sustanciales).
            - **Pertinencia Discursiva:** Entradas ajenas al futbolista o instrucciones de prompt injection son catalogadas con `es_pertinente=false` y excluidas del archivo persistente.
            """
        )

    # -------------------------------------------------------------------------
    # Pestanas de Navegacion (Sin Emojis)
    # -------------------------------------------------------------------------
    tab_emitir, tab_consultar, tab_resumen = st.tabs([
        "Emitir Opinion",
        "Consultar Opiniones",
        "Explorador de Figuras",
    ])

    # =========================================================================
    # PESTANA 1: Emitir Opinion
    # =========================================================================
    with tab_emitir:
        renderizar_tarjeta_jugador(jugador_activo)

        # Historial de mensajes de la sesion
        for msg in st.session_state.mensajes_chat:
            rol = msg.get("rol", "user")
            with st.chat_message(rol):
                if rol == "user":
                    st.markdown(f"**Para {msg.get('jugador_nombre')}:** {msg.get('contenido')}")
                else:
                    if msg.get("es_pertinente", True):
                        insignia = renderizar_insignia_sentimiento(msg.get("sentimiento", "neutro"))
                        st.markdown(
                            f"{insignia} &nbsp; **Analisis:** {msg.get('razon')}",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"ID: `{msg.get('id', 'N/A')}` | Modelo: `{msg.get('modelo', 'N/A')}`"
                        )
                    else:
                        st.warning(
                            f"Mensaje clasificado como no pertinente: {msg.get('razon')} (No almacenado en disco)"
                        )

        # Entrada de chat
        prompt_placeholder = f"Escribir opinion sobre {jugador_activo['nombre']} (5 a 1000 caracteres)..."
        opinion_usuario = st.chat_input(prompt_placeholder)

        if opinion_usuario is not None:
            try:
                opinion_limpia = validar_longitud_opinion(opinion_usuario)
            except ValueError as ve:
                st.warning(str(ve))
                opinion_limpia = None

            if opinion_limpia:
                with st.chat_message("user"):
                    st.markdown(f"**Para {jugador_activo['nombre']}:** {opinion_limpia}")

                jugador_al_enviar = dict(jugador_activo)

                with st.chat_message("assistant"):
                    with st.spinner("Procesando clasificacion semantica con LCEL..."):
                        error_analisis = False
                        analisis = None

                        if modo_offline:
                            analisis = analizar_opinion_mock(jugador_al_enviar["nombre"], opinion_limpia)
                            modelo_usado = "mock-evaluator-v1"
                        else:
                            try:
                                chat_llm = obtener_modelo_llm(
                                    proveedor=proveedor_str,
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
                                    f"Fallo en la inferencia del modelo estructurado: {exc}\n\n"
                                    "No se creo un registro artificial y la persistencia no fue ejecutada."
                                )

                    if not error_analisis and analisis is not None:
                        if not analisis.es_pertinente:
                            st.warning(
                                f"Opinion no pertinente: {analisis.razon}. "
                                "Por las reglas del taller, no se almacenara en opiniones.json."
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
                                    st.toast("Opinion persistida satisfactoriamente.")
                                    insignia = renderizar_insignia_sentimiento(analisis.sentimiento)
                                    st.markdown(
                                        f"{insignia} &nbsp; **Analisis:** {analisis.razon}",
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(
                                        f"UUID: `{nuevo_registro['id']}` | Timestamp: `{formatear_fecha(nuevo_registro['fecha_utc'])}`"
                                    )

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
                                    f"Fallo en la persistencia transaccional: {err_guardado}\n\n"
                                    "No se confirmo el guardado. Tu entrada se conserva en pantalla para su recuperacion."
                                )

    # =========================================================================
    # PESTANA 2: Consultar Opiniones
    # =========================================================================
    with tab_consultar:
        st.markdown("### Consultas y Auditoria de Registros")
        st.caption(
            "Filtrado y agregacion en memoria de los registros persistidos. Cero consumo de tokens durante la consulta."
        )

        try:
            datos_persistidos = cargar_opiniones(RUTA_OPINIONES_DEFAULT)
            todas_opiniones = datos_persistidos.get("opiniones", [])
        except Exception as exc:
            st.error(f"Error al leer el archivo de opiniones persistido: {exc}")
            todas_opiniones = []

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
                "Buscar por termino:",
                value="",
                placeholder="Filtrar texto original o justificacion...",
            )

        id_jugador_filtro = None
        if filtro_jugador_nombre != "Todos los futbolistas":
            for j in jugadores:
                if j["nombre"] == filtro_jugador_nombre:
                    id_jugador_filtro = j["id"]
                    break

        sentimiento_param = None if filtro_sentimiento == "Todos" else filtro_sentimiento

        opiniones_filtradas = consultar_opiniones(
            todas_opiniones,
            jugador_id=id_jugador_filtro,
            sentimiento=sentimiento_param,
            busqueda_texto=filtro_texto,
        )

        metricas = calcular_metricas(opiniones_filtradas)
        total_f = metricas["total"]
        conteos = metricas["conteos"]
        pcts = metricas["porcentajes"]

        st.markdown("#### Metricas del Subconjunto Consultado")
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.metric("Total Registros", total_f)
        with m_c2:
            st.metric("Positivas", f"{conteos['positivo']} ({pcts['positivo']}%)")
        with m_c3:
            st.metric("Negativas", f"{conteos['negativo']} ({pcts['negativo']}%)")
        with m_c4:
            st.metric("Neutras", f"{conteos['neutro']} ({pcts['neutro']}%)")
        with m_c5:
            st.metric("Mixtas", f"{conteos['mixto']} ({pcts['mixto']}%)")

        st.divider()

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

            d_col1, d_col2 = st.columns([1, 1])
            with d_col1:
                json_data = json.dumps(
                    {"total": total_f, "opiniones": opiniones_filtradas},
                    ensure_ascii=False,
                    indent=2,
                )
                st.download_button(
                    label="Exportar Dataset Filtrado (JSON)",
                    data=json_data,
                    file_name="opiniones_filtradas.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with d_col2:
                df_export = pd.DataFrame(opiniones_filtradas)
                csv_data = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Exportar Dataset Filtrado (CSV)",
                    data=csv_data,
                    file_name="opiniones_filtradas.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown(f"#### Registros Persistidos ({total_f})")

            for op in reversed(opiniones_filtradas):
                with st.container():
                    st.markdown(
                        f"""
                        <div class="opinion-card-tech">
                            <div class="opinion-card-header">
                                <span class="opinion-card-player">{op.get('jugador_nombre')} &middot; ID: <code>{op.get('jugador_id')}</code></span>
                                {renderizar_insignia_sentimiento(op.get('sentimiento', 'neutro'))}
                            </div>
                            <p class="opinion-card-text">
                                &laquo;{op.get('opinion_original')}&raquo;
                            </p>
                            <div class="opinion-card-reason">
                                <strong>Justificacion:</strong> {op.get('razon')}
                            </div>
                            <div class="opinion-card-meta">
                                <span>UUID: {op.get('id')}</span>
                                <span>Fecha: {formatear_fecha(op.get('fecha_utc'))}</span>
                                <span>Modelo: {op.get('modelo')}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if "clasificacion_original" in op:
                        orig = op["clasificacion_original"]
                        st.caption(
                            f"Registro rectificado manualmente (Original: {orig.get('sentimiento')})"
                        )

                    # Desafio de ampliacion: Correccion manual con auditoria
                    with st.expander(f"Rectificar clasificacion de {op.get('id')[:8]}"):
                        c_col1, c_col2 = st.columns([1, 2])
                        sentimientos_opciones = ["positivo", "negativo", "neutro", "mixto"]
                        idx_actual = sentimientos_opciones.index(op.get("sentimiento", "positivo")) if op.get("sentimiento") in sentimientos_opciones else 0
                        with c_col1:
                            nuevo_s = st.selectbox(
                                "Nuevo sentimiento:",
                                options=sentimientos_opciones,
                                index=idx_actual,
                                key=f"sel_{op.get('id')}",
                            )
                        with c_col2:
                            motivo = st.text_input(
                                "Justificacion tecnica:",
                                placeholder="Indicar argumento de la rectificacion manual...",
                                key=f"mot_{op.get('id')}",
                            )
                        if st.button("Guardar rectificacion", key=f"btn_{op.get('id')}"):
                            if motivo.strip():
                                actualizar_opinion(
                                    RUTA_OPINIONES_DEFAULT,
                                    op.get("id"),
                                    nuevo_s,
                                    motivo,
                                )
                                st.toast("Clasificacion actualizada y registrada en trazabilidad.")
                                st.rerun()
                            else:
                                st.warning("Se requiere ingresar una justificacion para registrar el cambio.")
        else:
            st.info("No se hallaron opiniones que satisfagan los criterios de filtro activos.")

    # =========================================================================
    # PESTANA 3: Explorador de Figuras
    # =========================================================================
    with tab_resumen:
        st.markdown("### Explorador Agregado de Figuras")
        st.caption(
            "Consolidacion de metricas de aprobacion calculadas a partir de las opiniones registradas."
        )

        try:
            datos_todas = cargar_opiniones(RUTA_OPINIONES_DEFAULT).get("opiniones", [])
        except Exception as exc:
            datos_todas = []

        if not datos_todas:
            st.warning("No se registran opiniones persistidas para computar la matriz comparativa.")
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

            resumen_jugador["Tasa_Aprobacion"] = (
                (resumen_jugador["Positivas"] / resumen_jugador["Total"]) * 100
            ).round(1)

            st.dataframe(
                resumen_jugador.sort_values(by="Total", ascending=False),
                column_config={
                    "jugador_nombre": "Futbolista",
                    "Total": st.column_config.NumberColumn("Muestra"),
                    "Positivas": st.column_config.NumberColumn("Positivas"),
                    "Negativas": st.column_config.NumberColumn("Negativas"),
                    "Neutras": st.column_config.NumberColumn("Neutras"),
                    "Mixtas": st.column_config.NumberColumn("Mixtas"),
                    "Tasa_Aprobacion": st.column_config.ProgressColumn(
                        "Tasa de Aprobacion",
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
