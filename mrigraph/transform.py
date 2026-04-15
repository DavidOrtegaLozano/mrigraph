from dataclasses import dataclass, field
from importlib import metadata
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .atlas import get_atlas_definition, resolve_supported_atlas_path, get_atlas_roi_name_map
from .io.nifti import load_nifti_file
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

    centroid_coordinate_space: Optional[str] = None

    roi_centroids_3d: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    centroid_json_path: Optional[str] = None

    atlas_resampled: bool = False
    centroid_cache_used: bool = False

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

    def __init__(self, denoise_bundle: MRIDenoiseBundle, atlas_data: Optional[object] = None, config: Optional[AtlasConfig] = None):
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
        fmri_affine = self._get_fmri_affine()

        atlas_source, atlas_labels, atlas_resampled, atlas_affine = self._resolve_atlas(
            spatial_shape=spatial_shape,
            fmri_affine=fmri_affine,
        )

        centroid_coordinate_space = "world" if atlas_affine is not None else "voxel"

        roi_ids = self._get_valid_roi_ids(atlas_labels)
        roi_labels = self._build_roi_labels(roi_ids)

        # if atlas_resampled:
        #     roi_centroids_3d = self._compute_roi_centroids(
        #         atlas_labels=atlas_labels,
        #         roi_ids=roi_ids,
        #         roi_labels=roi_labels,
        #         atlas_affine=atlas_affine,
        #     )
        #     centroid_json_path = None
        #     centroid_cache_used = False
        # else:
        #     roi_centroids_3d, centroid_json_path = self._resolve_or_build_roi_centroids(
        #         atlas_labels=atlas_labels,
        #         roi_ids=roi_ids,
        #         roi_labels=roi_labels,
        #         atlas_affine=atlas_affine,
        #     )
        #     centroid_cache_used = centroid_json_path is not None

        roi_centroids_3d, centroid_json_path = self._resolve_or_build_canonical_roi_centroids(
            roi_ids=roi_ids,
            roi_labels=roi_labels,
        )

        centroid_cache_used = centroid_json_path is not None
        centroid_coordinate_space = "world"

        roi_time_series, roi_sizes = self._extract_roi_time_series(
            fmri_data=fmri_data,
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
        )

        applied_steps = [
            "validate_atlas_shape",
            "extract_roi_mean_time_series",
        ]

        if atlas_resampled:
            applied_steps.insert(0, "resample_atlas_to_fmri_space")

        transform_metadata = self._build_transform_metadata(
            spatial_shape=spatial_shape,
            num_timepoints=num_timepoints,
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
            roi_sizes=roi_sizes,
            atlas_source=atlas_source,
            atlas_name=atlas_name,
            atlas_resampled=atlas_resampled,
            centroid_json_path=centroid_json_path,
            centroid_cache_used=centroid_cache_used,
            centroid_coordinate_space=centroid_coordinate_space,
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
            atlas_resampled=atlas_resampled,
            centroid_cache_used=centroid_cache_used,
            centroid_coordinate_space=centroid_coordinate_space,
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

    def _resolve_atlas(self, spatial_shape: Tuple[int, int, int], fmri_affine: np.ndarray) -> Tuple[str, np.ndarray, bool, Optional[np.ndarray]]:
        """
        Resuelve qué atlas usar y lo adapta al espacio del fMRI si hace falta.

        Devuelve:
        - atlas_source
        - atlas_labels
        - atlas_resampled
        atlas_affine final usada para el atlas
        """
        self._validate_atlas_name_if_provided()

        # 1. atlas pasado directamente al constructor
        if self.atlas_data is not None:
            return self._prepare_atlas_for_fmri_space(
                atlas_data=self.atlas_data,
                spatial_shape=spatial_shape,
                fmri_affine=fmri_affine,
                atlas_source="direct_input",
            )

        # 2. atlas resuelto desde config (atlas_name / atlas_path)
        atlas_from_config = self._load_atlas_from_config_if_available()
        if atlas_from_config is not None:
            source_name = (
                f"config_atlas:{self.config.atlas_name.lower()}"
                if self.config.atlas_name is not None
                else f"config_path:{self.config.atlas_path}"
            )
            return self._prepare_atlas_for_fmri_space(
                atlas_data=atlas_from_config,
                spatial_shape=spatial_shape,
                fmri_affine=fmri_affine,
                atlas_source=source_name,
            )

        # 3. atlas detectado en auxiliares
        aux_name, aux_atlas = self._find_atlas_in_auxiliary_files(
            self.denoise_bundle.auxiliary_files
        )
        if aux_atlas is not None:
            self._validate_atlas_labels(aux_atlas, spatial_shape)
            return f"auxiliary_file:{aux_name}", aux_atlas, False, None

        raise TransformationError(
            "No se ha proporcionado ningún atlas. "
            "Puedes pasar atlas_data, usar atlas_name/atlas_path en AtlasConfig "
            "o incluir un atlas en auxiliares."
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

    def _find_atlas_in_auxiliary_files(self, auxiliary_files: Dict[str, object]) -> Tuple[Optional[str], Optional[np.ndarray]]:
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

    def _validate_atlas_labels(self, atlas_labels: np.ndarray, spatial_shape: Tuple[int, int, int]) -> None:
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

        Prioridad:
        1. roi_labels manuales en config
        2. nombres anatómicos reales del atlas si existen
        3. fallback genérico tipo ROI_<id>
        """
        if self.config.roi_labels is not None:
            if len(self.config.roi_labels) != len(roi_ids):
                raise TransformationError(
                    "El número de roi_labels no coincide con el número de ROIs del atlas."
                )
            return list(self.config.roi_labels)
        
        if self.config.atlas_name is not None:
            roi_name_map = get_atlas_roi_name_map(self.config.atlas_name)
            if roi_name_map is not None:
                labels = []
                for roi_id in roi_ids:
                    roi_id_int = int(roi_id)
                    labels.append(roi_name_map.get(roi_id_int, f"ROI_{roi_id_int}"))
                return labels

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

    def _build_transform_metadata(self, spatial_shape: Tuple[int, int, int], num_timepoints: int, atlas_labels: np.ndarray, roi_ids: np.ndarray, roi_sizes: Dict[int, int], atlas_source: str, atlas_name: Optional[str], atlas_resampled: bool, centroid_json_path: Optional[str], centroid_cache_used: bool, centroid_coordinate_space: str,) -> Dict[str, object]:
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
            "implemented_scope": ["validate_atlas_shape", "extract_mean_roi_time_series"],
            "atlas_resampled": bool(atlas_resampled),
            "centroid_coordinate_space": centroid_coordinate_space,
            "centroid_json_path": centroid_json_path,
            "centroid_cache_used": bool(centroid_cache_used),
            "fmri_affine_available": self.denoise_bundle.original_metadata is not None
                and "affine" in self.denoise_bundle.original_metadata,
        }

        if atlas_name is not None:
            atlas_definition = get_atlas_definition(atlas_name)
            metadata["atlas_family"] = atlas_definition.family
            metadata["atlas_description"] = atlas_definition.description
            metadata["atlas_expected_num_rois"] = atlas_definition.num_rois
            if atlas_resampled:
                metadata["centroid_json_path"] = None
                metadata["has_saved_centroids"] = False
            else:
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
        print(f"Espacio de centroides: {bundle.transform_metadata['centroid_coordinate_space']}")

        if bundle.transform_metadata:
            print(f"Shape del atlas: {bundle.transform_metadata['atlas_shape']}")
            print(f"Número de ROIs: {bundle.transform_metadata['num_rois']}")
            print(f"Atlas resampleado: {bundle.transform_metadata['atlas_resampled']}")
            print(f"Caché de centroides usada: {bundle.transform_metadata['centroid_cache_used']}")
            print(f"JSON de centroides: {bundle.transform_metadata['centroid_json_path']}")
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

    # def _resolve_or_build_roi_centroids(self, atlas_labels: np.ndarray, roi_ids: np.ndarray, roi_labels: List[str], atlas_affine: Optional[np.ndarray]) -> Tuple[Dict[str, Tuple[float, float, float]], str]:
    #     """
    #     Intenta cargar centroides desde JSON.
    #     Si no existe, o si el JSON no coincide con las labels/coordenadas esperadas, los calcula y los guarda de nuevo.
    #     """
    #     json_path = self._get_centroid_json_path(roi_ids)
    #     expected_coordinate_space = "world" if atlas_affine is not None else "voxel"

    #     if json_path.exists():
    #         centroids = self._load_roi_centroids_from_json(
    #             json_path=json_path,
    #             expected_roi_labels=roi_labels,
    #             expected_coordinate_space=expected_coordinate_space,
    #         )
    #         if centroids:
    #             return centroids, str(json_path)

    #     centroids = self._compute_roi_centroids(
    #         atlas_labels=atlas_labels,
    #         roi_ids=roi_ids,
    #         roi_labels=roi_labels,
    #         atlas_affine=atlas_affine,
    #     )

    #     self._save_roi_centroids_to_json(
    #         json_path=json_path,
    #         roi_ids=roi_ids,
    #         roi_labels=roi_labels,
    #         centroids=centroids,
    #         atlas_shape=atlas_labels.shape,
    #         atlas_name=self.config.atlas_name,
    #         coordinate_space=expected_coordinate_space,
    #     )

    #     return centroids, str(json_path)

    def _resolve_or_build_canonical_roi_centroids(self, roi_ids: np.ndarray, roi_labels: List[str]) -> Tuple[Dict[str, Tuple[float, float, float]], Optional[str]]:
        """
        Devuelve centroides canónicos del atlas, independientes del sujeto.

        Estos centroides se calculan sobre el atlas de referencia original
        asociado a atlas_name, no sobre el atlas resampleado al espacio del fMRI.
        """
        if self.config.atlas_name is None:
            raise TransformationError(
                "No se pueden construir centroides canónicos si atlas_name es None."
            )

        json_path = self._get_centroid_json_path(roi_ids)

        # 1. Intentar cargar del JSON
        if json_path.exists():
            centroids = self._load_roi_centroids_from_json(
                json_path=json_path,
                expected_roi_labels=roi_labels,
                expected_coordinate_space="world",
            )
            if centroids:
                return centroids, str(json_path)

        # 2. Cargar el atlas original de referencia
        atlas_ref = self._load_atlas_from_config_if_available()
        if atlas_ref is None:
            raise TransformationError(
                f"No se pudo cargar el atlas de referencia para {self.config.atlas_name}."
            )

        atlas_ref_labels = self._coerce_atlas_to_array(atlas_ref)
        atlas_ref_affine = (
            np.asarray(atlas_ref.affine, dtype=float)
            if hasattr(atlas_ref, "affine")
            else None
        )

        centroids = self._compute_roi_centroids(
            atlas_labels=atlas_ref_labels,
            roi_ids=roi_ids,
            roi_labels=roi_labels,
            atlas_affine=atlas_ref_affine,
        )

        self._save_roi_centroids_to_json(
            json_path=json_path,
            roi_ids=roi_ids,
            roi_labels=roi_labels,
            centroids=centroids,
            atlas_shape=atlas_ref_labels.shape,
            atlas_name=self.config.atlas_name,
            coordinate_space="world",
        )

        return centroids, str(json_path)

    def _compute_roi_centroids(self, atlas_labels: np.ndarray, roi_ids: np.ndarray, roi_labels: List[str], atlas_affine: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float, float]]:
        """
        Calcula el centroide de cada ROI.

        - Si hay affine disponible, devuelve coordenadas mundo (world).
        - Si no la hay, cae a coordenadas voxel.
        """
        centroids: Dict[str, Tuple[float, float, float]] = {}

        for roi_id, roi_label in zip(roi_ids, roi_labels):
            coords = np.argwhere(atlas_labels == roi_id)
            if coords.size == 0:
                raise TransformationError(
                    f"No se pudieron calcular centroides para la ROI {int(roi_id)}."
                )

            voxel_centroid = coords.mean(axis=0).astype(np.float64)

            if atlas_affine is not None:
                voxel_h = np.append(voxel_centroid, 1.0)
                world_centroid = np.asarray(atlas_affine, dtype=np.float64) @ voxel_h
                centroid = world_centroid[:3]
            else:
                centroid = voxel_centroid

            centroids[roi_label] = (
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            )

        return centroids

    def _save_roi_centroids_to_json(self, json_path: Path, roi_ids: np.ndarray, roi_labels: List[str], centroids: Dict[str, Tuple[float, float, float]], atlas_shape: Tuple[int, int, int], atlas_name: Optional[str], coordinate_space: str) -> None:
        payload = {
            "atlas_name": atlas_name,
            "atlas_shape": list(atlas_shape),
            "coordinate_space": coordinate_space,
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

    def _load_roi_centroids_from_json(self, json_path: Path, expected_roi_labels: Optional[List[str]] = None, expected_coordinate_space: Optional[str] = None) -> Dict[str, Tuple[float, float, float]]:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        payload_coordinate_space = payload.get("coordinate_space")
        if (
            expected_coordinate_space is not None
            and payload_coordinate_space != expected_coordinate_space
        ):
            return {}

        rois = payload.get("rois", [])
        payload_labels = [roi.get("roi_label") for roi in rois]

        if expected_roi_labels is not None and payload_labels != list(expected_roi_labels):
            return {}

        centroids: Dict[str, Tuple[float, float, float]] = {}
        for roi in rois:
            label = roi["roi_label"]
            centroid = roi["centroid"]
            centroids[label] = (
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            )

        return centroids

    def _get_fmri_affine(self) -> np.ndarray:
        """
        Recupera la affine del fMRI desde los metadatos heredados.
        """
        metadata = self.denoise_bundle.original_metadata

        if metadata is None or "affine" not in metadata:
            raise TransformationError(
                "No se encontró la affine del fMRI en los metadatos originales."
            )

        affine = np.asarray(metadata["affine"])

        if affine.shape != (4, 4):
            raise TransformationError(
                f"La affine del fMRI no tiene forma 4x4 válida: {affine.shape}"
            )

        return affine

    def _atlas_matches_fmri_space(self, atlas_img, spatial_shape: Tuple[int, int, int], fmri_affine: np.ndarray, atol: float = 1e-5) -> bool:
        """
        Comprueba si atlas y fMRI ya comparten grid espacial.
        """
        if tuple(atlas_img.shape) != tuple(spatial_shape):
            return False

        atlas_affine = np.asarray(atlas_img.affine)
        return np.allclose(atlas_affine, fmri_affine, atol=atol)

    def _resample_atlas_to_fmri_space(self, atlas_img, spatial_shape: Tuple[int, int, int], fmri_affine: np.ndarray):
        """
        Resamplea el atlas al espacio del fMRI con nearest-neighbor.
        """
        try:
            from nibabel.processing import resample_from_to
        except ImportError as exc:
            raise TransformationError(
                "No se pudo importar nibabel.processing para resamplear el atlas."
            ) from exc

        target = (spatial_shape, fmri_affine)

        try:
            resampled_img = resample_from_to(
                atlas_img,
                target,
                order=0,
            )
        except Exception as exc:
            raise TransformationError(
                "Error al resamplear el atlas al espacio del fMRI."
            ) from exc

        return resampled_img

    def _prepare_atlas_for_fmri_space(self, atlas_data: object, spatial_shape: Tuple[int, int, int], fmri_affine: np.ndarray, atlas_source: str) -> Tuple[str, np.ndarray, bool, Optional[np.ndarray]]:
        """
        Prepara el atlas para que quede en el mismo espacio/grid que el fMRI.

        Devuelve:
        - atlas_source final
        - atlas_labels
        - atlas_resampled
        - atlas_affine final usada para centroides
        """
        # Caso 1: imagen tipo nibabel -> podemos comparar affine y resamplear
        if hasattr(atlas_data, "get_fdata") and hasattr(atlas_data, "affine"):
            atlas_img = atlas_data

            if self._atlas_matches_fmri_space(atlas_img, spatial_shape, fmri_affine):
                atlas_labels = self._coerce_atlas_to_array(atlas_img)
                self._validate_atlas_labels(atlas_labels, spatial_shape)
                return atlas_source, atlas_labels, False, np.asarray(atlas_img.affine, dtype=float)

            resampled_img = self._resample_atlas_to_fmri_space(
                atlas_img=atlas_img,
                spatial_shape=spatial_shape,
                fmri_affine=fmri_affine,
            )
            atlas_labels = self._coerce_atlas_to_array(resampled_img)
            self._validate_atlas_labels(atlas_labels, spatial_shape)
            return (
                f"{atlas_source}|resampled_to_fmri",
                atlas_labels,
                True,
                np.asarray(resampled_img.affine, dtype=float),
            )

        # Caso 2: ndarray 3D -> no podemos resamplear porque no hay affine
        atlas_labels = self._coerce_atlas_to_array(atlas_data)
        self._validate_atlas_labels(atlas_labels, spatial_shape)
        return atlas_source, atlas_labels, False, None
    
    def _load_atlas_from_config_if_available(self):
        """
        Intenta cargar el atlas desde la configuración si el usuario lo ha indicado
        por nombre o por ruta.
        """
        if self.config.atlas_name is None and self.config.atlas_path is None:
            return None

        if self.config.atlas_name is None and self.config.atlas_path is not None:
            atlas_path = Path(self.config.atlas_path)
            if not atlas_path.exists():
                raise TransformationError(
                    f"El atlas indicado en atlas_path no existe: {atlas_path}"
                )
            return load_nifti_file(atlas_path)

        atlas_name = self.config.atlas_name.lower()
        resolved_path = resolve_supported_atlas_path(
            atlas_name=atlas_name,
            atlas_path=self.config.atlas_path,
        )
        return load_nifti_file(resolved_path)

def build_synthetic_atlas(spatial_shape: Tuple[int, int, int], num_rois: int = 8) -> np.ndarray:
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