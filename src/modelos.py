"""Modelos y esquemas de validación con Pydantic para Tribuna Global."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


SentimientoTipo = Literal["positivo", "negativo", "neutro", "mixto"]


class Jugador(BaseModel):
    """Representa a un futbolista del catálogo oficial."""

    id: str = Field(..., description="Identificador único y estable (ej. 'j001')")
    nombre: str = Field(..., description="Nombre deportivo o formal del jugador")
    pais: str = Field(..., description="País de origen o representación nacional")


class AnalisisOpinion(BaseModel):
    """Contrato estructurado para la clasificación semántica de una opinión.

    Cumple con el contrato exigido por el taller y valida estrictamente la
    coherencia entre la pertinencia y la asignación del sentimiento.
    """

    es_pertinente: bool = Field(
        description="Indica si el texto constituye una opinión sobre el jugador seleccionado"
    )
    sentimiento: SentimientoTipo | None = Field(
        default=None,
        description="Categoría de la opinión ('positivo', 'negativo', 'neutro', 'mixto'); null cuando no es pertinente",
    )
    razon: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Justificación breve y objetiva redactada en tercera persona (máximo 300 caracteres)",
    )

    @model_validator(mode="after")
    def validar_coherencia_pertinencia(self) -> AnalisisOpinion:
        """Verifica que una opinión pertinente tenga categoría y una no pertinente sea None."""
        if self.es_pertinente:
            if self.sentimiento is None:
                raise ValueError(
                    "Una opinión marcada como pertinente debe tener una categoría de sentimiento válida ('positivo', 'negativo', 'neutro', 'mixto')."
                )
            if self.sentimiento not in ("positivo", "negativo", "neutro", "mixto"):
                raise ValueError(f"Categoría de sentimiento inválida: {self.sentimiento}")
        else:
            # Si no es pertinente, el sentimiento debe ser forzosamente None
            self.sentimiento = None

        # Truncar suavemente la razón si excede por unos pocos caracteres o asegurar formato
        if self.razon:
            self.razon = self.razon.strip()
            if len(self.razon) > 300:
                self.razon = self.razon[:297] + "..."

        return self


class RegistroOpinion(BaseModel):
    """Esquema de un registro almacenado en opiniones.json."""

    id: str
    fecha_utc: str
    jugador_id: str
    jugador_nombre: str
    opinion_original: str
    sentimiento: SentimientoTipo
    razon: str
    modelo: str
    version_prompt: str = "1.0"
