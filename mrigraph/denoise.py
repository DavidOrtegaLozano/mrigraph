from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .config import DenoiseConfig
from .exceptions import DenoisingError
from .preprocess import MRIPreprocessBundle


@dataclass
class MRIDenoiseBundle:
    """
    Estructura que representa la salida de la fase de denoising.

    Campos principales:
    - fmri_path: ruta del fMRI original
    - original_metadata: metadatos heredados de la entrada
    - preprocess_metadata: metadatos heredados del preprocesado
    - denoised_data: volumen 4D tras denoising
    - voxel_time_series: matriz 2D denoised (num_voxels, num_timepoints)
    - auxiliary_files: archivos auxiliares heredados
    - applied_steps: pasos realmente ejecutados
    - pending_steps: pasos pedidos en config pero aún no implementados
    - denoise_metadata: información útil para trazabilidad
    """
    fmri_path: Optional[str] = None
    original_metadata: Optional[Dict[str, object]] = None
    preprocess_metadata: Optional[Dict[str, object]] = None
    denoised_data: Optional[np.ndarray] = None
    voxel_time_series: Optional[np.ndarray] = None
    auxiliary_files: Dict[str, object] = field(default_factory=dict)
    applied_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    denoise_metadata: Dict[str, object] = field(default_factory=dict)


class DenoiseMRIData:
    """
    Clase principal de denoising básico para MRIGraph.

    Uso esperado:
    - recibe un MRIPreprocessBundle,
    - aplica limpieza básica de señal,
    - devuelve un MRIDenoiseBundle listo para atlas / ROI / conectividad.
    """

    def __init__(
        self,
        preprocess_bundle: MRIPreprocessBundle,
        config: Optional[DenoiseConfig] = None,
    ):
        self.preprocess_bundle = preprocess_bundle
        self.config = config if config is not None else DenoiseConfig()

    def run(self) -> MRIDenoiseBundle:
        """
        Ejecuta la fase de denoising básico.

        Devuelve:
        - MRIDenoiseBundle con:
          - señal voxel x tiempo limpia,
          - volumen 4D reconstruido,
          - metadatos del proceso.
        """
        self._validate_preprocess_bundle()

        voxel_time_series = np.array(
            self.preprocess_bundle.voxel_time_series,
            dtype=np.float32,
            copy=True
        )

        applied_steps: List[str] = []
        pending_steps: List[str] = []

        confounds_name = None
        confounds_shape = None

        # --------------------------------------------------------
        # 1) Regresión de confounds
        # --------------------------------------------------------
        if self.config.regress_confounds:
            confounds_name, confounds_matrix = self._find_confounds_matrix(
                self.preprocess_bundle.auxiliary_files,
                voxel_time_series.shape[1]
            )

            if confounds_matrix is not None:
                voxel_time_series = self._regress_confounds(
                    voxel_time_series,
                    confounds_matrix
                )
                applied_steps.append("confound_regression")
                confounds_shape = tuple(confounds_matrix.shape)
            else:
                pending_steps.append("confound_regression_requested_but_no_confounds_found")

        # --------------------------------------------------------
        # 2) Scrubbing
        # --------------------------------------------------------
        # De momento lo dejamos pendiente porque no queremos fingir
        # un scrubbing real sin métricas adecuadas (FD, DVARS, etc.).
        if self.config.apply_scrubbing:
            pending_steps.append("scrubbing")

        # --------------------------------------------------------
        # 3) Band-pass
        # --------------------------------------------------------
        # También lo dejamos pendiente porque para hacerlo bien
        # necesitamos información como el TR y una estrategia clara.
        if self.config.apply_bandpass:
            pending_steps.append("bandpass_filtering")

        denoised_data = self._reshape_to_4d(voxel_time_series)

        denoise_metadata = self._build_denoise_metadata(
            voxel_time_series=voxel_time_series,
            confounds_name=confounds_name,
            confounds_shape=confounds_shape,
        )

        bundle = MRIDenoiseBundle(
            fmri_path=self.preprocess_bundle.fmri_path,
            original_metadata=self.preprocess_bundle.original_metadata,
            preprocess_metadata=self.preprocess_bundle.preprocess_metadata,
            denoised_data=denoised_data,
            voxel_time_series=voxel_time_series,
            auxiliary_files=dict(self.preprocess_bundle.auxiliary_files),
            applied_steps=applied_steps,
            pending_steps=pending_steps,
            denoise_metadata=denoise_metadata,
        )

        return bundle

    def _validate_preprocess_bundle(self) -> None:
        """
        Comprueba que el bundle de preprocesado contiene lo necesario.
        """
        if self.preprocess_bundle is None:
            raise DenoisingError("No se ha proporcionado un MRIPreprocessBundle válido.")

        if self.preprocess_bundle.preprocessed_data is None:
            raise DenoisingError(
                "El bundle de preprocesado no contiene datos 4D preprocesados."
            )

        if self.preprocess_bundle.voxel_time_series is None:
            raise DenoisingError(
                "El bundle de preprocesado no contiene la matriz voxel x tiempo."
            )

        if not isinstance(self.preprocess_bundle.voxel_time_series, np.ndarray):
            raise DenoisingError(
                "La representación voxel x tiempo no es un ndarray de NumPy."
            )

    def _find_confounds_matrix(
        self,
        auxiliary_files: Dict[str, object],
        num_timepoints: int
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Busca una matriz de confounds entre los auxiliares cargados.

        Estrategia:
        - busca archivos cuyo nombre contenga 'confound',
        - acepta matrices 1D o 2D,
        - valida que el número de filas coincida con el número de timepoints.

        Devuelve:
        - nombre del archivo encontrado
        - matriz de confounds con forma (T, K), o None si no existe
        """
        for file_name, data in auxiliary_files.items():
            if "confound" not in file_name.lower():
                continue

            if not isinstance(data, np.ndarray):
                continue

            confounds = np.asarray(data, dtype=np.float32)

            # Caso 1: vector 1D de longitud T
            if confounds.ndim == 1 and confounds.shape[0] == num_timepoints:
                return file_name, confounds.reshape(num_timepoints, 1)

            # Caso 2: matriz 2D con forma (T, K)
            if confounds.ndim == 2 and confounds.shape[0] == num_timepoints:
                return file_name, confounds

            # Caso 3: matriz 2D con forma (K, T), la transponemos
            if confounds.ndim == 2 and confounds.shape[1] == num_timepoints:
                return file_name, confounds.T

            raise DenoisingError(
                f"El archivo de confounds '{file_name}' no tiene una forma compatible "
                f"con {num_timepoints} timepoints."
            )

        return None, None

    def _regress_confounds(
        self,
        voxel_time_series: np.ndarray,
        confounds_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Elimina el efecto lineal de los confounds sobre la señal.

        Parámetros:
        - voxel_time_series: matriz con forma (num_voxels, T)
        - confounds_matrix: matriz con forma (T, K)

        Estrategia:
        - trabajamos en formato temporal (T, num_voxels),
        - añadimos intercepto,
        - resolvemos por mínimos cuadrados,
        - devolvemos los residuos.
        """
        if voxel_time_series.ndim != 2:
            raise DenoisingError(
                "La señal voxel x tiempo debe ser una matriz 2D."
            )

        if confounds_matrix.ndim != 2:
            raise DenoisingError(
                "La matriz de confounds debe ser 2D."
            )

        num_timepoints = voxel_time_series.shape[1]

        if confounds_matrix.shape[0] != num_timepoints:
            raise DenoisingError(
                "El número de filas de los confounds no coincide con los timepoints."
            )

        # Señal reorganizada como (T, num_voxels)
        signal_t_by_v = voxel_time_series.T

        # Estandarizamos confounds columna a columna
        confounds_mean = np.mean(confounds_matrix, axis=0, keepdims=True)
        confounds_std = np.std(confounds_matrix, axis=0, keepdims=True)
        confounds_std[confounds_std == 0] = 1.0

        confounds_z = (confounds_matrix - confounds_mean) / confounds_std

        # Añadimos intercepto
        design_matrix = np.column_stack(
            [np.ones((num_timepoints, 1), dtype=np.float32), confounds_z]
        )

        try:
            betas, _, _, _ = np.linalg.lstsq(design_matrix, signal_t_by_v, rcond=None)
        except Exception as exc:
            raise DenoisingError(
                "Error al resolver la regresión de confounds."
            ) from exc

        fitted_signal = design_matrix @ betas
        residual_signal = signal_t_by_v - fitted_signal

        return residual_signal.T.astype(np.float32)

    def _reshape_to_4d(self, voxel_time_series: np.ndarray) -> np.ndarray:
        """
        Reconstruye el volumen 4D a partir de la matriz voxel x tiempo.
        """
        original_shape = self.preprocess_bundle.preprocessed_data.shape

        try:
            denoised_data = voxel_time_series.reshape(original_shape)
        except Exception as exc:
            raise DenoisingError(
                "No se pudo reconstruir el volumen 4D tras el denoising."
            ) from exc

        return denoised_data.astype(np.float32)

    def _build_denoise_metadata(
        self,
        voxel_time_series: np.ndarray,
        confounds_name: Optional[str],
        confounds_shape: Optional[Tuple[int, int]],
    ) -> Dict[str, object]:
        """
        Construye metadatos útiles de la fase de denoising.
        """
        metadata = {
            "denoised_4d_shape": tuple(self.preprocess_bundle.preprocessed_data.shape),
            "denoised_voxel_time_series_shape": tuple(voxel_time_series.shape),
            "num_voxels": int(voxel_time_series.shape[0]),
            "num_timepoints": int(voxel_time_series.shape[1]),
            "confounds_file_used": confounds_name,
            "confounds_shape": confounds_shape,
            "regress_confounds_requested": self.config.regress_confounds,
            "scrubbing_requested": self.config.apply_scrubbing,
            "bandpass_requested": self.config.apply_bandpass,
            "implemented_scope": [
                "optional_confound_regression",
                "reconstruct_4d_volume",
            ],
        }

        return metadata

    def display_info(self, bundle: MRIDenoiseBundle) -> None:
        """
        Muestra por pantalla un resumen del resultado del denoising.
        """
        print("\n[MRIGraph] Denoising completado")
        print(f"fMRI original: {bundle.fmri_path}")

        if bundle.denoise_metadata:
            print(f"Shape volumen denoised: {bundle.denoise_metadata['denoised_4d_shape']}")
            print(
                f"Matriz voxel x tiempo denoised: "
                f"{bundle.denoise_metadata['denoised_voxel_time_series_shape']}"
            )
            print(
                f"Archivo de confounds usado: "
                f"{bundle.denoise_metadata['confounds_file_used']}"
            )
            print(
                f"Shape de confounds usada: "
                f"{bundle.denoise_metadata['confounds_shape']}"
            )

        if bundle.applied_steps:
            print("Pasos aplicados:")
            for step in bundle.applied_steps:
                print(f" - {step}")
        else:
            print("No se ha aplicado ningún paso efectivo de denoising.")

        if bundle.pending_steps:
            print("Pasos pendientes de implementación:")
            for step in bundle.pending_steps:
                print(f" - {step}")