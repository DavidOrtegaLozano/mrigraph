"""
Paso 8 del pipeline MRI.
Archivo: transform.py

Qué hace este archivo:
- Recibe la salida del denoising.
- Recibe un atlas de etiquetas 3D.
- Valida la compatibilidad espacial entre fMRI y atlas.
- Extrae series temporales por ROI promediando los voxeles de cada región.

Resultado:
- pasa de un volumen 4D (X, Y, Z, T)
- a una matriz 2D (num_rois, T)

Importante:
Este archivo sí implementa una transformación real y útil
para llegar a conectividad funcional:
- mapeo atlas -> voxeles
- agregación voxel -> ROI
- construcción de series ROI x tiempo

No implementa todavía:
- resampling automático entre espacios distintos,
- registro espacial,
- uso directo de atlas descargados por nombre,
- pipelines neuroimagen complejos.
"""

from dataclasses import dataclass, field
from importlib import metadata
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .atlas import get_atlas_definition
from .config import AtlasConfig
from .denoise import MRIDenoiseBundle
from .exceptions import AtlasError, TransformationError
import json
from pathlib import Path


@dataclass
class MRITransformBundle:
    fmri_path: Optional[str] = None
    original_metadata: Optional[Dict[str, object]] = None
    preprocess_metadata: Optional[Dict[str, object]] = None
    denoise_metadata: Optional[Dict[str, object]] = None
    atlas_name: Optional[str] = None
    atlas_source: Optional[str] = None
    atlas_labels: Optional[np.ndarray] = None
    roi_time_series: Optional[np.ndarray] = None
    roi_labels: List[str] = field(default_factory=list)

    roi_centroids_3d: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    centroid_json_path: Optional[str] = None

    auxiliary_files: Dict[str, object] = field(default_factory=dict)
    applied_steps: List[str] = field(default_factory=list)
    transform_metadata: Dict[str, object] = field(default_factory=dict)


class TransformMRIData:
    """
    Clase principal de transformación desde volumen 4D a series por ROI.

    Uso esperado:
    - recibe un MRIDenoiseBundle
    - recibe un atlas de etiquetas 3D, o lo intenta localizar en auxiliares
    - devuelve un MRITransformBundle con series ROI x tiempo
    """

    def __init__(
        self,
        denoise_bundle: MRIDenoiseBundle,
        atlas_data: Optional[object] = None,
        config: Optional[AtlasConfig] = None,
    ):
        self.denoise_bundle = denoise_bundle
        self.atlas_data = atlas_data
        self.config = config if config is not None else AtlasConfig()

    def run(self) -> MRITransformBundle:
        """
        Ejecuta la transformación a nivel de ROI.

        Devuelve:
        - MRITransformBundle con:
          - atlas 3D validado,
          - matriz ROI x tiempo,
          - metadatos de transformación.
        """
        self._validate_denoise_bundle()

        fmri_data = self.denoise_bundle.denoised_data
        spatial_shape = fmri_data.shape[:3]
        num_timepoints = fmri_data.shape[3]

        atlas_name = self.config.atlas_name.lower() if self.config.atlas_name else None
        atlas_source, atlas_labels = self._resolve_atlas(spatial_shape)
        roi_ids = self._get_valid_roi_ids(atlas_labels)
        roi_labels = self._build_roi_labels(roi_ids)

        roi_labels = self._build_roi_labels(roi_ids)

        roi_centroids_3d, centroid_json_path = self._resolve_or_build_roi_centroids(
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
            roi_labels=roi_labels,
        )

        roi_time_series, roi_sizes = self._extract_roi_time_series(
            fmri_data=fmri_data,
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
        )

        applied_steps = [
            "validate_atlas_shape",
            "extract_roi_mean_time_series",
        ]

        transform_metadata = self._build_transform_metadata(
            spatial_shape=spatial_shape,
            num_timepoints=num_timepoints,
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
            roi_sizes=roi_sizes,
            atlas_source=atlas_source,
            atlas_name=atlas_name,
        )

        bundle = MRITransformBundle(
            fmri_path=self.denoise_bundle.fmri_path,
            original_metadata=self.denoise_bundle.original_metadata,
            preprocess_metadata=self.denoise_bundle.preprocess_metadata,
            denoise_metadata=self.denoise_bundle.denoise_metadata,
            atlas_name=atlas_name,
            atlas_source=atlas_source,
            atlas_labels=atlas_labels,
            roi_time_series=roi_time_series,
            roi_labels=roi_labels,
            roi_centroids_3d=roi_centroids_3d,
            centroid_json_path=centroid_json_path,
            auxiliary_files=dict(self.denoise_bundle.auxiliary_files),
            applied_steps=applied_steps,
            transform_metadata=transform_metadata,
        )

        return bundle

    def _validate_denoise_bundle(self) -> None:
        """
        Comprueba que el bundle de denoise contiene un volumen 4D válido.
        """
        if self.denoise_bundle is None:
            raise TransformationError("No se ha proporcionado un MRIDenoiseBundle válido.")

        if self.denoise_bundle.denoised_data is None:
            raise TransformationError(
                "El bundle de denoise no contiene datos 4D denoised."
            )

        if not isinstance(self.denoise_bundle.denoised_data, np.ndarray):
            raise TransformationError(
                "Los datos denoised no son un ndarray de NumPy."
            )

        if self.denoise_bundle.denoised_data.ndim != 4:
            raise TransformationError(
                f"Se esperaba un volumen 4D denoised, pero se obtuvo "
                f"{self.denoise_bundle.denoised_data.shape}."
            )

    def _resolve_atlas(self, spatial_shape: Tuple[int, int, int]) -> Tuple[str, np.ndarray]:
        """
        Resuelve qué atlas usar.

        Prioridad:
        1. atlas_data pasado explícitamente al constructor
        2. atlas detectado en auxiliary_files
        3. error claro si no hay atlas disponible

        Devuelve:
        - atlas_source: texto indicando el origen
        - atlas_labels: ndarray 3D de enteros
        """
        if self.atlas_data is not None:
            atlas_labels = self._coerce_atlas_to_array(self.atlas_data)
            self._validate_atlas_labels(atlas_labels, spatial_shape)
            self._validate_atlas_name_if_provided()
            return "direct_input", atlas_labels

        aux_name, aux_atlas = self._find_atlas_in_auxiliary_files(
            self.denoise_bundle.auxiliary_files
        )
        if aux_atlas is not None:
            self._validate_atlas_labels(aux_atlas, spatial_shape)
            self._validate_atlas_name_if_provided()
            return f"auxiliary_file:{aux_name}", aux_atlas

        raise TransformationError(
            "No se ha proporcionado ningún atlas. "
            "Pasa un atlas 3D directamente o incluye uno en auxiliares."
        )

    def _validate_atlas_name_if_provided(self) -> None:
        """
        Valida el nombre lógico del atlas si el usuario lo ha indicado.

        Nota:
        Esto no carga el atlas por nombre. Solo comprueba que el nombre
        sea coherente con el registro de atlas soportados.
        """
        if self.config.atlas_name is None:
            return

        get_atlas_definition(self.config.atlas_name)

    def _find_atlas_in_auxiliary_files(
        self,
        auxiliary_files: Dict[str, object]
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Busca un atlas 3D entre los auxiliares.

        Estrategia:
        - busca nombres que contengan 'atlas'
        - acepta matrices ndarray 3D
        """
        for file_name, data in auxiliary_files.items():
            if "atlas" not in file_name.lower():
                continue

            if isinstance(data, np.ndarray) and data.ndim == 3:
                return file_name, self._coerce_atlas_to_array(data)

        return None, None

    def _coerce_atlas_to_array(self, atlas_data: object) -> np.ndarray:
        """
        Convierte distintas representaciones de atlas a un ndarray 3D.

        Casos soportados:
        - ndarray 3D
        - imagen tipo nibabel con get_fdata()

        Devuelve:
        - atlas 3D como ndarray de enteros
        """
        if isinstance(atlas_data, np.ndarray):
            atlas_array = atlas_data
        elif hasattr(atlas_data, "get_fdata"):
            atlas_array = atlas_data.get_fdata()
        else:
            raise AtlasError(
                "Formato de atlas no soportado. Usa un ndarray 3D o una imagen NIfTI cargada."
            )

        if not isinstance(atlas_array, np.ndarray):
            atlas_array = np.asarray(atlas_array)

        if atlas_array.ndim != 3:
            raise AtlasError(
                f"El atlas debe ser 3D, pero se recibió una matriz con forma {atlas_array.shape}."
            )

        # Convertimos a enteros porque el atlas representa etiquetas de región.
        return np.rint(atlas_array).astype(np.int32)

    def _validate_atlas_labels(
        self,
        atlas_labels: np.ndarray,
        spatial_shape: Tuple[int, int, int]
    ) -> None:
        """
        Valida que el atlas tenga forma espacial compatible con el fMRI.
        """
        if atlas_labels.shape != spatial_shape:
            raise AtlasError(
                f"El atlas tiene forma {atlas_labels.shape}, pero el fMRI tiene "
                f"forma espacial {spatial_shape}."
            )

        positive_labels = atlas_labels[atlas_labels > 0]
        if positive_labels.size == 0:
            raise AtlasError(
                "El atlas no contiene ninguna ROI válida (> 0)."
            )

    def _get_valid_roi_ids(self, atlas_labels: np.ndarray) -> np.ndarray:
        """
        Obtiene los identificadores de ROI válidos.

        Convención:
        - 0 se considera fondo
        - valores > 0 se consideran ROIs
        """
        roi_ids = np.unique(atlas_labels)
        roi_ids = roi_ids[roi_ids > 0]

        if roi_ids.size == 0:
            raise TransformationError(
                "No se encontraron ROIs válidas en el atlas."
            )

        return roi_ids

    def _build_roi_labels(self, roi_ids: np.ndarray) -> List[str]:
        """
        Construye etiquetas legibles para las ROIs.

        Si el usuario pasó roi_labels en config, se usan esas.
        Si no, se generan nombres genéricos tipo ROI_1, ROI_2, etc.
        """
        if self.config.roi_labels is not None:
            if len(self.config.roi_labels) != len(roi_ids):
                raise TransformationError(
                    "El número de roi_labels no coincide con el número de ROIs del atlas."
                )
            return list(self.config.roi_labels)

        return [f"ROI_{int(roi_id)}" for roi_id in roi_ids]

    def _extract_roi_time_series(
        self,
        fmri_data: np.ndarray,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Extrae series temporales por ROI.

        Estrategia:
        - para cada etiqueta de ROI,
        - selecciona sus voxeles,
        - promedia la señal de todos los voxeles de la región en cada timepoint.

        Devuelve:
        - roi_time_series: matriz (num_rois, T)
        - roi_sizes: diccionario con número de voxeles por ROI
        """
        num_timepoints = fmri_data.shape[3]
        roi_series_list: List[np.ndarray] = []
        roi_sizes: Dict[int, int] = {}

        for roi_id in roi_ids:
            roi_mask = atlas_labels == roi_id
            roi_voxels = fmri_data[roi_mask]

            if roi_voxels.size == 0:
                raise TransformationError(
                    f"La ROI {int(roi_id)} no contiene voxeles."
                )

            # Tras indexar con una máscara 3D sobre un volumen 4D,
            # la forma esperada es (num_voxels_roi, T)
            if roi_voxels.ndim != 2 or roi_voxels.shape[1] != num_timepoints:
                raise TransformationError(
                    f"La ROI {int(roi_id)} no produjo una matriz voxel x tiempo válida."
                )

            roi_mean_ts = np.mean(roi_voxels, axis=0, dtype=np.float32)
            roi_series_list.append(roi_mean_ts.astype(np.float32))
            roi_sizes[int(roi_id)] = int(roi_voxels.shape[0])

        roi_time_series = np.vstack(roi_series_list).astype(np.float32)
        return roi_time_series, roi_sizes

    def _build_transform_metadata(
        self,
        spatial_shape: Tuple[int, int, int],
        num_timepoints: int,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
        roi_sizes: Dict[int, int],
        atlas_source: str,
        atlas_name: Optional[str],
    ) -> Dict[str, object]:
        """
        Construye metadatos útiles de la transformación.
        """
        metadata: Dict[str, object] = {
            "fmri_spatial_shape": spatial_shape,
            "num_timepoints": int(num_timepoints),
            "atlas_shape": tuple(atlas_labels.shape),
            "atlas_source": atlas_source,
            "atlas_name": atlas_name,
            "num_rois": int(len(roi_ids)),
            "roi_ids": [int(x) for x in roi_ids.tolist()],
            "roi_sizes": roi_sizes,
            "roi_time_series_shape": (int(len(roi_ids)), int(num_timepoints)),
            "implemented_scope": [
                "validate_atlas_shape",
                "extract_mean_roi_time_series",
            ],
        }

        if atlas_name is not None:
            atlas_definition = get_atlas_definition(atlas_name)
            metadata["atlas_family"] = atlas_definition.family
            metadata["atlas_description"] = atlas_definition.description
            metadata["atlas_expected_num_rois"] = atlas_definition.num_rois
            metadata["centroid_json_path"] = str(self._get_centroid_json_path(roi_ids))
            metadata["has_saved_centroids"] = True

        return metadata

    def display_info(self, bundle: MRITransformBundle) -> None:
        """
        Muestra por pantalla un resumen del resultado de la transformación.
        """
        print("\n[MRIGraph] Transformación a ROI completada")
        print(f"fMRI original: {bundle.fmri_path}")
        print(f"Origen del atlas: {bundle.atlas_source}")
        print(f"Nombre del atlas: {bundle.atlas_name}")

        if bundle.transform_metadata:
            print(f"Shape del atlas: {bundle.transform_metadata['atlas_shape']}")
            print(f"Número de ROIs: {bundle.transform_metadata['num_rois']}")
            print(
                f"Matriz ROI x tiempo: "
                f"{bundle.transform_metadata['roi_time_series_shape']}"
            )

        if bundle.applied_steps:
            print("Pasos aplicados:")
            for step in bundle.applied_steps:
                print(f" - {step}")


    def _get_centroid_layout_dir(self) -> Path:
        """
        Carpeta donde guardamos layouts por atlas.
        Se queda dentro de mrigraph para poder versionarlos si quieres.
        """
        layouts_dir = Path(__file__).resolve().parent / "atlas_centroids"
        layouts_dir.mkdir(parents=True, exist_ok=True)
        return layouts_dir


    def _build_centroid_json_name(self, roi_ids: np.ndarray) -> str:
        """
        Genera el nombre del JSON.
        Si hay atlas_name, usamos ese nombre.
        Si no, usamos uno genérico con número de ROIs.
        """
        if self.config.atlas_name:
            return f"{self.config.atlas_name.lower()}_centroids.json"

        return f"custom_{len(roi_ids)}rois_centroids.json"


    def _get_centroid_json_path(self, roi_ids: np.ndarray) -> Path:
        return self._get_centroid_layout_dir() / self._build_centroid_json_name(roi_ids)


    def _resolve_or_build_roi_centroids(
        self,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
        roi_labels: List[str],
    ) -> Tuple[Dict[str, Tuple[float, float, float]], str]:
        """
        Intenta cargar centroides desde JSON.
        Si no existe, los calcula y los guarda.
        """
        json_path = self._get_centroid_json_path(roi_ids)

        if json_path.exists():
            print("\n\nCREADOOOS\n\n")
            centroids = self._load_roi_centroids_from_json(json_path)
            if centroids:
                return centroids, str(json_path)

        print("\n\nNO EXISTENNN\n\n")
        centroids = self._compute_roi_centroids(atlas_labels, roi_ids, roi_labels)
        self._save_roi_centroids_to_json(
            json_path=json_path,
            roi_ids=roi_ids,
            roi_labels=roi_labels,
            centroids=centroids,
            atlas_shape=atlas_labels.shape,
            atlas_name=self.config.atlas_name,
        )
        return centroids, str(json_path)


    def _compute_roi_centroids(
        self,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
        roi_labels: List[str],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Calcula el centroide de cada ROI en coordenadas de voxel.

        Más adelante, si quieres, aquí podemos aplicar affine cuando el atlas
        venga como imagen NIfTI en lugar de ndarray.
        """
        centroids: Dict[str, Tuple[float, float, float]] = {}

        for roi_id, roi_label in zip(roi_ids, roi_labels):
            coords = np.argwhere(atlas_labels == roi_id)

            if coords.size == 0:
                raise TransformationError(
                    f"No se pudieron calcular centroides para la ROI {int(roi_id)}."
                )

            centroid = coords.mean(axis=0)
            centroids[roi_label] = (
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            )

        return centroids


    def _save_roi_centroids_to_json(
        self,
        json_path: Path,
        roi_ids: np.ndarray,
        roi_labels: List[str],
        centroids: Dict[str, Tuple[float, float, float]],
        atlas_shape: Tuple[int, int, int],
        atlas_name: Optional[str],
    ) -> None:
        payload = {
            "atlas_name": atlas_name,
            "atlas_shape": list(atlas_shape),
            "coordinate_space": "voxel",
            "rois": [
                {
                    "roi_id": int(roi_id),
                    "roi_label": roi_label,
                    "centroid": list(centroids[roi_label]),
                }
                for roi_id, roi_label in zip(roi_ids, roi_labels)
            ],
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


    def _load_roi_centroids_from_json(
        self,
        json_path: Path,
    ) -> Dict[str, Tuple[float, float, float]]:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        centroids: Dict[str, Tuple[float, float, float]] = {}
        for roi in payload.get("rois", []):
            label = roi["roi_label"]
            centroid = roi["centroid"]
            centroids[label] = (
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            )

        return centroids


def build_synthetic_atlas(
    spatial_shape: Tuple[int, int, int],
    num_rois: int = 8
) -> np.ndarray:
    """
    Construye un atlas sintético sencillo para pruebas.

    Qué hace:
    - reparte todos los voxeles del volumen en num_rois grupos
    - asigna etiquetas 1..num_rois
    - no deja fondo (0), porque el objetivo aquí es solo probar el flujo

    Parámetros:
    - spatial_shape: forma 3D del espacio (X, Y, Z)
    - num_rois: número de regiones sintéticas deseadas

    Devuelve:
    - ndarray 3D de enteros con etiquetas de ROI
    """
    if len(spatial_shape) != 3:
        raise AtlasError(
            f"spatial_shape debe tener 3 dimensiones, pero recibió {spatial_shape}."
        )

    if num_rois < 1:
        raise AtlasError("num_rois debe ser al menos 1.")

    num_voxels = int(np.prod(spatial_shape))
    if num_rois > num_voxels:
        raise AtlasError(
            "num_rois no puede ser mayor que el número total de voxeles."
        )

    labels_flat = np.zeros(num_voxels, dtype=np.int32)
    voxel_indices = np.arange(num_voxels)
    voxel_groups = np.array_split(voxel_indices, num_rois)

    for roi_index, group in enumerate(voxel_groups, start=1):
        labels_flat[group] = roi_index

    atlas_labels = labels_flat.reshape(spatial_shape)
    return atlas_labels