"""Módulo de almacenamiento y persistencia para Tribuna Global.

Gestiona la lectura y validación del catálogo de futbolistas, la persistencia
atómica en formato JSON para las opiniones y el motor de consultas en memoria.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# Rutas predeterminadas relativas a la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos"
RUTA_JUGADORES_DEFAULT = DATOS_DIR / "jugadores.json"
RUTA_OPINIONES_DEFAULT = DATOS_DIR / "opiniones.json"


def cargar_jugadores(ruta: str | Path = RUTA_JUGADORES_DEFAULT) -> list[dict[str, Any]]:
    """Lee y valida el catálogo de futbolistas desde un archivo JSON.

    Comprueba que el archivo exista, que el JSON sea válido, que contenga
    la estructura esperada y que NO existan identificadores duplicados.

    Args:
        ruta: Ruta al archivo jugadores.json.

    Returns:
        Lista de diccionarios con la información de cada jugador.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el JSON es inválido, tiene esquema incorrecto o IDs duplicados.
    """
    path_obj = Path(ruta)
    if not path_obj.is_file():
        raise FileNotFoundError(f"No se encontró el catálogo de jugadores en: {path_obj}")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"El catálogo de jugadores contiene JSON inválido: {exc}") from exc

    if not isinstance(datos, dict) or "jugadores" not in datos:
        raise ValueError("El catálogo debe ser un diccionario con la clave 'jugadores'.")

    jugadores = datos.get("jugadores")
    if not isinstance(jugadores, list):
        raise ValueError("La clave 'jugadores' debe contener una lista.")

    ids_vistos: set[str] = set()
    ids_duplicados: list[str] = []

    for idx, jugador in enumerate(jugadores):
        if not isinstance(jugador, dict):
            raise ValueError(f"El elemento en el índice {idx} no es un diccionario válido.")

        for campo in ("id", "nombre", "pais"):
            if campo not in jugador or not str(jugador[campo]).strip():
                raise ValueError(
                    f"El jugador en el índice {idx} carece del campo obligatorio '{campo}'."
                )

        j_id = str(jugador["id"]).strip()
        if j_id in ids_vistos:
            ids_duplicados.append(j_id)
        else:
            ids_vistos.add(j_id)

    if ids_duplicados:
        raise ValueError(
            f"Se detectaron identificadores de jugadores duplicados: {sorted(set(ids_duplicados))}"
        )

    return jugadores


def cargar_opiniones(ruta: str | Path = RUTA_OPINIONES_DEFAULT) -> dict[str, Any]:
    """Lee el diccionario persistente de opiniones desde el archivo JSON.

    Si el archivo no existe, inicializa la estructura vacía:
        {"version": 1, "opiniones": []}
    y la guarda en disco.

    Si el archivo existe pero contiene JSON inválido o un esquema inesperado,
    lanza un error y preserva el archivo intacto (no lo sobrescribe).

    Args:
        ruta: Ruta al archivo opiniones.json.

    Returns:
        Diccionario con versión y lista de opiniones.

    Raises:
        ValueError: Si el JSON está dañado o el esquema es incorrecto.
    """
    path_obj = Path(ruta)

    if not path_obj.exists():
        estructura_inicial = {"version": 1, "opiniones": []}
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(estructura_inicial, f, ensure_ascii=False, indent=2)
        return estructura_inicial

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
            if not contenido:
                raise ValueError("El archivo de opiniones se encuentra completamente vacío.")
            datos = json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"El archivo de opiniones contiene JSON inválido ({exc}). "
            "El archivo original ha sido conservado intacto para evitar pérdida de datos."
        ) from exc

    if not isinstance(datos, dict) or "opiniones" not in datos or not isinstance(datos["opiniones"], list):
        raise ValueError(
            "El archivo de opiniones no cumple con el esquema esperado "
            "({'version': int, 'opiniones': [...]}). El archivo ha sido conservado."
        )

    return datos


def crear_registro(
    jugador: Mapping[str, Any],
    texto: str,
    analisis: Any,
    modelo: str,
    version_prompt: str = "1.0",
) -> dict[str, Any]:
    """Crea un nuevo registro persistente de opinión con metadatos completos.

    Args:
        jugador: Diccionario con la información del futbolista (id y nombre).
        texto: Texto original de la opinión del aficionado.
        analisis: Objeto AnalisisOpinion o dict con sentimiento y razón.
        modelo: Nombre del modelo LLM utilizado.
        version_prompt: Versión del prompt empleada.

    Returns:
        Diccionario listo para ser persistido en opiniones.json.
    """
    sentimiento = getattr(analisis, "sentimiento", None)
    razon = getattr(analisis, "razon", None)

    if isinstance(analisis, dict):
        sentimiento = analisis.get("sentimiento", sentimiento)
        razon = analisis.get("razon", razon)

    if not sentimiento:
        raise ValueError("No se puede crear un registro sin un sentimiento clasificado.")

    return {
        "id": str(uuid.uuid4()),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "jugador_id": str(jugador["id"]),
        "jugador_nombre": str(jugador["nombre"]),
        "opinion_original": texto.strip(),
        "sentimiento": str(sentimiento).lower(),
        "razon": str(razon).strip() if razon else "",
        "modelo": str(modelo),
        "version_prompt": str(version_prompt),
    }


def guardar_opinion(ruta: str | Path, registro: dict[str, Any]) -> bool:
    """Añade un registro al archivo persistente mediante una escritura atómica.

    Utiliza un archivo temporal en el mismo directorio y reemplazo final atómico
    para prevenir corrupción de datos ante interrupciones. Evita duplicados
    comprobando si el ID ya existe en el registro.

    Args:
        ruta: Ruta al archivo opiniones.json.
        registro: Diccionario con el registro a persistir.

    Returns:
        True si la escritura fue exitosa.

    Raises:
        IOError / OSError: Si ocurre un error de permisos o escritura.
        ValueError: Si los datos de entrada o el archivo son inválidos.
    """
    path_obj = Path(ruta)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Cargar estado existente (o inicializar si no existe)
    estado = cargar_opiniones(path_obj)
    opiniones: list[dict[str, Any]] = estado.get("opiniones", [])

    # Validación de idempotencia: no duplicar el mismo ID
    reg_id = registro.get("id")
    if reg_id and any(op.get("id") == reg_id for op in opiniones):
        return True

    # Añadir nuevo registro
    opiniones.append(registro)
    estado["opiniones"] = opiniones

    # Escritura segura mediante archivo temporal en el mismo directorio
    directorio = path_obj.parent
    prefijo = f".{path_obj.stem}_tmp_"
    fd, temp_file_path = tempfile.mkstemp(prefix=prefijo, suffix=".tmp", dir=directorio)

    try:
        with open(fd, "w", encoding="utf-8") as temp_file:
            json.dump(estado, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Reemplazo atómico
        os.replace(temp_file_path, path_obj)
        return True
    except Exception as exc:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        raise IOError(f"Fallo durante la escritura atómica en {path_obj}: {exc}") from exc


def actualizar_opinion(
    ruta: str | Path,
    opinion_id: str,
    nuevo_sentimiento: str,
    justificacion_cambio: str,
) -> bool:
    """Corrige manualmente una etiqueta conservando la clasificacion original y la auditoria.

    Cumple con el desafio de ampliacion 3 del taller: permite la correccion
    manual de etiquetas preservando el sentimiento y razon originales.

    Args:
        ruta: Ruta al archivo opiniones.json.
        opinion_id: Identificador unico de la opinion a corregir.
        nuevo_sentimiento: Nuevo sentimiento corregido ('positivo', 'negativo', 'neutro', 'mixto').
        justificacion_cambio: Motivo o argumento de la rectificacion.

    Returns:
        True si la actualizacion y reemplazo atomico fueron exitosos.

    Raises:
        ValueError: Si el sentimiento es invalido o la opinion no se encuentra.
    """
    nuevo_sentimiento_norm = nuevo_sentimiento.lower().strip()
    if nuevo_sentimiento_norm not in ("positivo", "negativo", "neutro", "mixto"):
        raise ValueError(f"Sentimiento invalido para correccion: '{nuevo_sentimiento}'")

    path_obj = Path(ruta)
    estado = cargar_opiniones(path_obj)
    opiniones = estado.get("opiniones", [])

    encontrado = False
    for op in opiniones:
        if op.get("id") == opinion_id:
            encontrado = True
            # Preservar clasificacion original intacta en el primer cambio
            if "clasificacion_original" not in op:
                op["clasificacion_original"] = {
                    "sentimiento": op.get("sentimiento"),
                    "razon": op.get("razon"),
                    "modelo": op.get("modelo"),
                }

            # Registrar en trazabilidad / auditoria
            if "historial_ediciones" not in op:
                op["historial_ediciones"] = []

            op["historial_ediciones"].append({
                "fecha_utc": datetime.now(timezone.utc).isoformat(),
                "sentimiento_anterior": op.get("sentimiento"),
                "sentimiento_nuevo": nuevo_sentimiento_norm,
                "motivo": justificacion_cambio.strip(),
            })

            # Aplicar modificacion
            op["sentimiento"] = nuevo_sentimiento_norm
            break

    if not encontrado:
        raise ValueError(f"No se encontro ninguna opinion con el ID: '{opinion_id}'")

    # Escritura atomica de la coleccion actualizada
    directorio = path_obj.parent
    prefijo = f".{path_obj.stem}_tmp_"
    fd, temp_file_path = tempfile.mkstemp(prefix=prefijo, suffix=".tmp", dir=directorio)

    try:
        with open(fd, "w", encoding="utf-8") as temp_file:
            json.dump(estado, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_file_path, path_obj)
        return True
    except Exception as exc:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        raise IOError(f"Fallo durante la actualizacion atomica en {path_obj}: {exc}") from exc


def consultar_opiniones(
    registros: list[dict[str, Any]],
    jugador_id: str | None = None,
    sentimiento: str | None = None,
    busqueda_texto: str | None = None,
) -> list[dict[str, Any]]:
    """Filtra una lista de registros de opiniones en memoria sin consultar al LLM.

    Args:
        registros: Lista de opiniones persistidas.
        jugador_id: ID del jugador a filtrar o None/'todos' para todos.
        sentimiento: Categoría de sentimiento a filtrar o None/'todos' para todos.
        busqueda_texto: Cadena opcional para buscar dentro de la opinión o razón.

    Returns:
        Lista con los registros que cumplen con todos los criterios de filtro.
    """
    resultados = list(registros)

    if jugador_id and jugador_id.lower() not in ("todos", "all", ""):
        resultados = [r for r in resultados if r.get("jugador_id") == jugador_id]

    if sentimiento and sentimiento.lower() not in ("todos", "all", ""):
        sentimiento_norm = sentimiento.lower().strip()
        resultados = [
            r for r in resultados if str(r.get("sentimiento", "")).lower() == sentimiento_norm
        ]

    if busqueda_texto and busqueda_texto.strip():
        termino = busqueda_texto.strip().lower()
        resultados = [
            r
            for r in resultados
            if termino in str(r.get("opinion_original", "")).lower()
            or termino in str(r.get("razon", "")).lower()
            or termino in str(r.get("jugador_nombre", "")).lower()
        ]

    return resultados


def calcular_metricas(registros: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula recuentos y porcentajes por categoría de sentimiento.

    Args:
        registros: Lista de opiniones a tabular.

    Returns:
        Diccionario con conteos totales y desgloses por categoría.
    """
    total = len(registros)
    conteos = {
        "positivo": 0,
        "negativo": 0,
        "neutro": 0,
        "mixto": 0,
    }

    for r in registros:
        sent = str(r.get("sentimiento", "")).lower()
        if sent in conteos:
            conteos[sent] += 1

    porcentajes = {
        k: (round((v / total) * 100, 1) if total > 0 else 0.0)
        for k, v in conteos.items()
    }

    return {
        "total": total,
        "conteos": conteos,
        "porcentajes": porcentajes,
    }

