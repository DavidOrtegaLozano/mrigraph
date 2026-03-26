"""
Paso 3 del pipeline MRI.
Archivo: atlas.py

Qué hace este archivo:
- Define el registro de atlas soportados por MRIGraph.
- Guarda información básica de cada atlas:
  - nombre,
  - número aproximado de nodos,
  - tipo,
  - descripción breve.

Este archivo no aplica todavía el atlas sobre el fMRI.
Su objetivo ahora es:
1. dejar claro qué atlas vamos a soportar,
2. centralizar esa información,
3. facilitar validaciones y decisiones posteriores.
"""

from dataclasses import dataclass
from typing import Dict, List
from .exceptions import AtlasError


@dataclass(frozen=True)
class AtlasDefinition:
    """
    Representa la definición básica de un atlas soportado.

    Campos:
    - name: nombre corto del atlas.
    - family: familia o grupo del atlas.
    - num_rois: número aproximado de regiones.
    - modality: en qué contexto lo queremos usar.
    - description: descripción breve y útil.
    """
    name: str
    family: str
    num_rois: int
    modality: str
    description: str


# Registro inicial de atlas soportados.
# Esto no impide ampliar el registro más adelante.
SUPPORTED_ATLASES: Dict[str, AtlasDefinition] = {
    "aal": AtlasDefinition(
        name="aal",
        family="AAL",
        num_rois=116,
        modality="fMRI",
        description="Atlas anatómico clásico, útil como opción de granularidad media."
    ),
    "bna": AtlasDefinition(
        name="bna",
        family="Brainnetome",
        num_rois=246,
        modality="fMRI",
        description="Atlas de granularidad media-alta, orientado a conectividad."
    ),
    "schaefer_100": AtlasDefinition(
        name="schaefer_100",
        family="Schaefer",
        num_rois=100,
        modality="fMRI",
        description="Versión funcional de Schaefer con 100 ROIs."
    ),
    "schaefer_200": AtlasDefinition(
        name="schaefer_200",
        family="Schaefer",
        num_rois=200,
        modality="fMRI",
        description="Versión funcional de Schaefer con 200 ROIs."
    ),
    "schaefer_1000": AtlasDefinition(
        name="schaefer_1000",
        family="Schaefer",
        num_rois=1000,
        modality="fMRI",
        description="Versión de alta granularidad de Schaefer."
    ),
}


def list_supported_atlases() -> List[str]:
    """
    Devuelve una lista con los nombres de los atlas soportados.
    """
    return list(SUPPORTED_ATLASES.keys())


def get_atlas_definition(atlas_name: str) -> AtlasDefinition:
    """
    Devuelve la definición de un atlas soportado.

    Lanza un error claro si el atlas no existe en el registro.
    """
    atlas_name = atlas_name.lower()

    if atlas_name not in SUPPORTED_ATLASES:
        raise AtlasError(
            f"El atlas '{atlas_name}' no está soportado. "
            f"Atlas disponibles: {', '.join(list_supported_atlases())}"
        )

    return SUPPORTED_ATLASES[atlas_name]


def is_supported_atlas(atlas_name: str) -> bool:
    """
    Indica si un atlas está soportado o no.
    """
    return atlas_name.lower() in SUPPORTED_ATLASES