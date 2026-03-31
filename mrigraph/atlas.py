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
from pathlib import Path
from typing import Optional, Dict, List


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
    "schaefer_400": AtlasDefinition(
        name="schaefer_400",
        family="Schaefer",
        num_rois=400,
        modality="fMRI",
        description="Versión de alta granularidad de Schaefer."
    ),
}

# Diccionarios opcionales de mapeo ROI ID -> nombre anatómico real.
ATLAS_ROI_NAME_MAPS: Dict[str, Dict[int, str]] = {
    "aal": {
        2001: "Precentral_L",
        2002: "Precentral_R",
        2101: "Frontal_Sup_L",
        2102: "Frontal_Sup_R",
        2111: "Frontal_Sup_Orb_L",
        2112: "Frontal_Sup_Orb_R",
        2201: "Frontal_Mid_L",
        2202: "Frontal_Mid_R",
        2211: "Frontal_Mid_Orb_L",
        2212: "Frontal_Mid_Orb_R",
        2301: "Frontal_Inf_Oper_L",
        2302: "Frontal_Inf_Oper_R",
        2311: "Frontal_Inf_Tri_L",
        2312: "Frontal_Inf_Tri_R",
        2321: "Frontal_Inf_Orb_L",
        2322: "Frontal_Inf_Orb_R",
        2331: "Rolandic_Oper_L",
        2332: "Rolandic_Oper_R",
        2401: "Supp_Motor_Area_L",
        2402: "Supp_Motor_Area_R",
        2501: "Olfactory_L",
        2502: "Olfactory_R",
        2601: "Frontal_Sup_Medial_L",
        2602: "Frontal_Sup_Medial_R",
        2611: "Frontal_Med_Orb_L",
        2612: "Frontal_Med_Orb_R",
        2701: "Rectus_L",
        2702: "Rectus_R",
        3001: "Insula_L",
        3002: "Insula_R",
        4001: "Cingulum_Ant_L",
        4002: "Cingulum_Ant_R",
        4011: "Cingulum_Mid_L",
        4012: "Cingulum_Mid_R",
        4021: "Cingulum_Post_L",
        4022: "Cingulum_Post_R",
        4101: "Hippocampus_L",
        4102: "Hippocampus_R",
        4111: "ParaHippocampal_L",
        4112: "ParaHippocampal_R",
        4201: "Amygdala_L",
        4202: "Amygdala_R",
        5001: "Calcarine_L",
        5002: "Calcarine_R",
        5011: "Cuneus_L",
        5012: "Cuneus_R",
        5021: "Lingual_L",
        5022: "Lingual_R",
        5101: "Occipital_Sup_L",
        5102: "Occipital_Sup_R",
        5201: "Occipital_Mid_L",
        5202: "Occipital_Mid_R",
        5301: "Occipital_Inf_L",
        5302: "Occipital_Inf_R",
        5401: "Fusiform_L",
        5402: "Fusiform_R",
        6001: "Postcentral_L",
        6002: "Postcentral_R",
        6101: "Parietal_Sup_L",
        6102: "Parietal_Sup_R",
        6201: "Parietal_Inf_L",
        6202: "Parietal_Inf_R",
        6211: "SupraMarginal_L",
        6212: "SupraMarginal_R",
        6221: "Angular_L",
        6222: "Angular_R",
        6301: "Precuneus_L",
        6302: "Precuneus_R",
        6401: "Paracentral_Lobule_L",
        6402: "Paracentral_Lobule_R",
        7001: "Caudate_L",
        7002: "Caudate_R",
        7011: "Putamen_L",
        7012: "Putamen_R",
        7021: "Pallidum_L",
        7022: "Pallidum_R",
        7101: "Thalamus_L",
        7102: "Thalamus_R",
        8101: "Heschl_L",
        8102: "Heschl_R",
        8111: "Temporal_Sup_L",
        8112: "Temporal_Sup_R",
        8121: "Temporal_Pole_Sup_L",
        8122: "Temporal_Pole_Sup_R",
        8201: "Temporal_Mid_L",
        8202: "Temporal_Mid_R",
        8211: "Temporal_Pole_Mid_L",
        8212: "Temporal_Pole_Mid_R",
        8301: "Temporal_Inf_L",
        8302: "Temporal_Inf_R",
        9001: "Cerebelum_Crus1_L",
        9002: "Cerebelum_Crus1_R",
        9011: "Cerebelum_Crus2_L",
        9012: "Cerebelum_Crus2_R",
        9021: "Cerebelum_3_L",
        9022: "Cerebelum_3_R",
        9031: "Cerebelum_4_5_L",
        9032: "Cerebelum_4_5_R",
        9041: "Cerebelum_6_L",
        9042: "Cerebelum_6_R",
        9051: "Cerebelum_7b_L",
        9052: "Cerebelum_7b_R",
        9061: "Cerebelum_8_L",
        9062: "Cerebelum_8_R",
        9071: "Cerebelum_9_L",
        9072: "Cerebelum_9_R",
        9081: "Cerebelum_10_L",
        9082: "Cerebelum_10_R",
        9100: "Vermis_1_2",
        9110: "Vermis_3",
        9120: "Vermis_4_5",
        9130: "Vermis_6",
        9140: "Vermis_7",
        9150: "Vermis_8",
        9160: "Vermis_9",
        9170: "Vermis_10",
    }
}

DEFAULT_ATLAS_FILENAMES: Dict[str, str] = {
    "aal": "aal.nii.gz",
    "schaefer_100": "schaefer_100.nii.gz",
    "schaefer_200": "schaefer_200.nii.gz",
    "schaefer_400": "schaefer_400.nii.gz",
}

def get_default_atlas_dir() -> Path:
    """
    Carpeta por defecto donde la librería buscará atlas empaquetados.
    """
    atlas_dir = Path(__file__).resolve().parent / "atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    return atlas_dir

def get_default_atlas_path(atlas_name: str) -> Path:
    """
    Devuelve la ruta por defecto esperada para un atlas soportado.
    """
    atlas_name = atlas_name.lower()

    if atlas_name not in DEFAULT_ATLAS_FILENAMES:
        raise AtlasError(
            f"No hay nombre de archivo por defecto definido para el atlas '{atlas_name}'."
        )

    return get_default_atlas_dir() / DEFAULT_ATLAS_FILENAMES[atlas_name]

def resolve_supported_atlas_path(
    atlas_name: str,
    atlas_path: Optional[str] = None,
) -> Path:
    """
    Resuelve la ruta real de un atlas soportado.

    Prioridad:
    1. atlas_path explícito en la configuración
    2. ruta por defecto dentro de mrigraph/atlases
    """
    atlas_name = atlas_name.lower()
    get_atlas_definition(atlas_name)

    if atlas_path is not None:
        candidate = Path(atlas_path)
        if not candidate.exists():
            raise AtlasError(
                f"Se indicó atlas_path para '{atlas_name}', pero el archivo no existe: {candidate}"
            )
        return candidate

    default_path = get_default_atlas_path(atlas_name)
    if default_path.exists():
        return default_path

    raise AtlasError(
        f"No se encontró el archivo del atlas '{atlas_name}'. "
        f"Ruta esperada por defecto: {default_path}. "
        f"Puedes añadir el archivo ahí o indicar atlas_path explícitamente."
    )
    
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

def get_atlas_roi_name_map(atlas_name: str) -> Optional[Dict[int, str]]:
    """
    Devuelve el diccionario roi_id -> nombre anatómico si existe para ese atlas.
    """
    return ATLAS_ROI_NAME_MAPS.get(atlas_name.lower())


def has_roi_name_map(atlas_name: str) -> bool:
    """
    Indica si el atlas tiene nombres anatómicos reales definidos.
    """
    return atlas_name.lower() in ATLAS_ROI_NAME_MAPS