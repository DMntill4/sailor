"""Módulo de almacenamiento y persistencia - Reexportación desde src.almacenamiento."""

from src.almacenamiento import (  # noqa: F401
    BASE_DIR,
    DATOS_DIR,
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

__all__ = [
    "BASE_DIR",
    "DATOS_DIR",
    "RUTA_JUGADORES_DEFAULT",
    "RUTA_OPINIONES_DEFAULT",
    "actualizar_opinion",
    "calcular_metricas",
    "cargar_jugadores",
    "cargar_opiniones",
    "consultar_opiniones",
    "crear_registro",
    "guardar_opinion",
]
