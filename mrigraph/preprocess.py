"""
Paso 6 del pipeline MRI.
Archivo: preprocess.py

Qué hace este archivo:
- Recibe los datos cargados desde InputMRIData.
- Extrae el contenido numérico del NIfTI a un array de NumPy.
- Valida que la forma siga siendo compatible con fMRI 4D.
- Aplica un preprocesado básico y honesto:
  - limpieza de valores no finitos (NaN / inf),
  - normalización global opcional,
  - preparación de una matriz voxel x tiempo.

Importante:
Este archivo NO implementa todavía preprocesado neuroimagen avanzado.
Por tanto, de momento NO realiza realmente:
- motion correction,
- slice timing correction,
- outlier detection clínica,
- smoothing espacial real.

Esas opciones existen ya en la configuración para dejar preparada
la arquitectura, pero aquí solo registramos que están pendientes
de implementación si se solicitan.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .config import PreprocessConfig
from .exceptions import PreprocessingError
from .importMRIData import MRIInputBundle


@dataclass
class MRIPreprocessBundle:
    """
    Estructura que representa la salida de la fase de preprocesado.

    Campos principales:
    - fmri_path: ruta del fMRI original
    - original_metadata: metadatos básicos obtenidos en la carga
    - preprocessed_data: array 4D ya preparado
    - voxel_time_series: representación 2D con forma (num_voxels, num_timepoints)
    - auxiliary_files: archivos auxiliares heredados del bundle de entrada
    - applied_steps: pasos realmente ejecutados
    - pending_steps: pasos solicitados en config pero aún no implementados
    - preprocess_metadata: información útil para depuración y trazabilidad
    """
    fmri_path: Optional[str] = None
    original_metadata: Optional[Dict[str, object]] = None
    preprocessed_data: Optional[np.ndarray] = None
    voxel_time_series: Optional[np.ndarray] = None
    auxiliary_files: Dict[str, object] = field(default_factory=dict)
    applied_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    preprocess_metadata: Dict[str, object] = field(default_factory=dict)


class PreprocessMRIData:
    """
    Clase principal de preprocesado básico para MRIGraph.

    Uso esperado:
    - recibe un MRIInputBundle ya cargado,
    - aplica una preparación inicial de los datos,
    - devuelve un MRIPreprocessBundle listo para la siguiente fase.
    """

    def __init__(
        self,
        input_bundle: MRIInputBundle,
        config: Optional[PreprocessConfig] = None,
    ):
        self.input_bundle = input_bundle
        self.config = config if config is not None else PreprocessConfig()

    def run(self) -> MRIPreprocessBundle:
        """
        Ejecuta la fase de preprocesado básico.

        Devuelve:
        - MRIPreprocessBundle con:
          - array 4D preprocesado,
          - matriz voxel x tiempo,
          - metadatos,
          - pasos aplicados,
          - pasos pendientes.
        """
        self._validate_input_bundle()

        fmri_data = self._extract_fmri_array()
        self._validate_fmri_array(fmri_data)

        applied_steps: List[str] = []
        pending_steps: List[str] = []

        # Limpieza básica de NaN / inf.
        fmri_data, replaced_values = self._replace_non_finite_values(fmri_data)
        if replaced_values > 0:
            applied_steps.append("replace_non_finite_values")

        # Normalización básica opcional.
        if self.config.apply_normalization:
            fmri_data = self._normalize_data(fmri_data)
            applied_steps.append("global_intensity_normalization")

        # Registramos pasos que existen en arquitectura pero aún no implementamos.
        pending_steps.extend(self._collect_pending_steps())

        voxel_time_series = self._reshape_to_voxel_time_series(fmri_data)

        preprocess_metadata = self._build_preprocess_metadata(
            fmri_data=fmri_data,
            voxel_time_series=voxel_time_series,
            replaced_values=replaced_values,
        )

        bundle = MRIPreprocessBundle(
            fmri_path=self.input_bundle.fmri_path,
            original_metadata=self.input_bundle.fmri_metadata,
            preprocessed_data=fmri_data,
            voxel_time_series=voxel_time_series,
            auxiliary_files=dict(self.input_bundle.auxiliary_files),
            applied_steps=applied_steps,
            pending_steps=pending_steps,
            preprocess_metadata=preprocess_metadata,
        )

        return bundle

    def _validate_input_bundle(self) -> None:
        """
        Comprueba que el bundle de entrada tiene lo necesario para preprocesar.
        """
        if self.input_bundle is None:
            raise PreprocessingError("No se ha proporcionado un MRIInputBundle válido.")

        if self.input_bundle.fmri_image is None:
            raise PreprocessingError(
                "El bundle de entrada no contiene una imagen fMRI cargada."
            )

        if self.input_bundle.fmri_metadata is None:
            raise PreprocessingError(
                "El bundle de entrada no contiene metadatos del fMRI."
            )

    def _extract_fmri_array(self) -> np.ndarray:
        """
        Extrae el array numérico del objeto NIfTI.

        Devuelve:
        - ndarray con los datos del fMRI

        Error:
        - PreprocessingError si no se puede extraer correctamente.
        """
        try:
            fmri_data = self.input_bundle.fmri_image.get_fdata(dtype=np.float32)
        except Exception as exc:
            raise PreprocessingError(
                "No se pudo extraer el array numérico desde la imagen fMRI."
            ) from exc

        return fmri_data

    def _validate_fmri_array(self, fmri_data: np.ndarray) -> None:
        """
        Valida que el array siga teniendo estructura compatible con fMRI 4D.
        """
        if not isinstance(fmri_data, np.ndarray):
            raise PreprocessingError("Los datos extraídos no son un ndarray de NumPy.")

        if fmri_data.ndim != 4:
            raise PreprocessingError(
                f"Se esperaba un array 4D para fMRI, pero se obtuvo forma {fmri_data.shape}."
            )

        if fmri_data.shape[-1] < 2:
            raise PreprocessingError(
                "El número de timepoints es insuficiente para continuar con el pipeline."
            )

    def _replace_non_finite_values(self, fmri_data: np.ndarray):
        """
        Sustituye NaN, +inf y -inf por 0.0 si aparecen.

        Devuelve:
        - array limpio
        - número de valores reemplazados
        """
        mask_non_finite = ~np.isfinite(fmri_data)
        num_replaced = int(np.sum(mask_non_finite))

        if num_replaced == 0:
            return fmri_data, 0

        cleaned_data = np.array(fmri_data, copy=True)
        cleaned_data[mask_non_finite] = 0.0

        return cleaned_data, num_replaced

    def _normalize_data(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Aplica una normalización global sencilla sobre todo el volumen 4D.

        Estrategia:
        - media global = 0
        - desviación típica global = 1

        Nota:
        Esto es una normalización básica de ingeniería para dejar el pipeline
        consistente. No pretende sustituir un preprocesado científico completo.
        """
        mean_value = float(np.mean(fmri_data))
        std_value = float(np.std(fmri_data))

        if std_value == 0.0:
            raise PreprocessingError(
                "No se puede normalizar el fMRI porque la desviación típica es 0."
            )

        normalized_data = (fmri_data - mean_value) / std_value
        return normalized_data.astype(np.float32)

    def _reshape_to_voxel_time_series(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Convierte el array 4D (X, Y, Z, T) a una matriz 2D (num_voxels, T).

        Esta representación será útil más adelante para:
        - denoising,
        - extracción ROI,
        - cálculo de conectividad.
        """
        num_timepoints = fmri_data.shape[-1]
        voxel_time_series = fmri_data.reshape(-1, num_timepoints)
        return voxel_time_series

    def _collect_pending_steps(self) -> List[str]:
        """
        Recoge los pasos marcados en configuración pero aún no implementados.

        Esto evita dar a entender que la librería ya hace operaciones
        neuroimagen avanzadas cuando todavía no es así.
        """
        pending_steps = []

        if self.config.apply_motion_correction:
            pending_steps.append("motion_correction")

        if self.config.apply_slice_timing:
            pending_steps.append("slice_timing_correction")

        if self.config.apply_outlier_detection:
            pending_steps.append("outlier_detection")

        if self.config.apply_smoothing:
            pending_steps.append("spatial_smoothing")

        return pending_steps

    def _build_preprocess_metadata(
        self,
        fmri_data: np.ndarray,
        voxel_time_series: np.ndarray,
        replaced_values: int,
    ) -> Dict[str, object]:
        """
        Construye metadatos útiles del preprocesado.
        """
        original_shape = self.input_bundle.fmri_metadata.get("shape")
        processed_shape = tuple(fmri_data.shape)

        metadata = {
            "original_shape": original_shape,
            "processed_shape": processed_shape,
            "ndim": fmri_data.ndim,
            "dtype": str(fmri_data.dtype),
            "num_voxels": int(np.prod(fmri_data.shape[:3])),
            "num_timepoints": int(fmri_data.shape[3]),
            "voxel_time_series_shape": tuple(voxel_time_series.shape),
            "non_finite_values_replaced": replaced_values,
            "normalization_applied": self.config.apply_normalization,
            "implemented_scope": [
                "extract_array",
                "validate_4d_shape",
                "replace_non_finite_values",
                "optional_global_intensity_normalization",
                "reshape_to_voxel_time_series",
            ],
        }

        return metadata

    def display_info(self, bundle: MRIPreprocessBundle) -> None:
        """
        Muestra por pantalla un resumen legible del resultado del preprocesado.
        """
        print("\n[MRIGraph] Preprocesado completado")
        print(f"fMRI original: {bundle.fmri_path}")

        if bundle.preprocess_metadata:
            print(f"Shape original: {bundle.preprocess_metadata['original_shape']}")
            print(f"Shape procesada: {bundle.preprocess_metadata['processed_shape']}")
            print(
                f"Matriz voxel x tiempo: "
                f"{bundle.preprocess_metadata['voxel_time_series_shape']}"
            )
            print(
                f"Valores no finitos reemplazados: "
                f"{bundle.preprocess_metadata['non_finite_values_replaced']}"
            )

        if bundle.applied_steps:
            print("Pasos aplicados:")
            for step in bundle.applied_steps:
                print(f" - {step}")
        else:
            print("No se han aplicado transformaciones explícitas sobre los datos.")

        if bundle.pending_steps:
            print("Pasos pendientes de implementación:")
            for step in bundle.pending_steps:
                print(f" - {step}")