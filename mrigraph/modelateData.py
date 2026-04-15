from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .config import ConnectivityConfig
from .exceptions import ConnectivityError
from .strategy import get_connectivity_strategy
from .tools import apply_connectivity_threshold
from .transform import MRITransformBundle


@dataclass
class MRIConnectivityBundle:
    """
    Estructura que representa la salida de la fase de conectividad.

    Campos principales:
    - fmri_path: ruta del fMRI original
    - original_metadata: metadatos heredados de entrada
    - preprocess_metadata: metadatos heredados de preprocess
    - denoise_metadata: metadatos heredados de denoise
    - transform_metadata: metadatos heredados de transformación
    - atlas_name: nombre lógico del atlas usado
    - roi_labels: etiquetas de las ROIs
    - roi_time_series: matriz original ROI x tiempo
    - connectivity_matrix: matriz ROI x ROI
    - applied_steps: pasos realmente ejecutados
    - pending_steps: pasos contemplados pero aún no implementados
    - connectivity_metadata: información útil para trazabilidad
    """
    fmri_path: Optional[str] = None
    original_metadata: Optional[Dict[str, object]] = None
    preprocess_metadata: Optional[Dict[str, object]] = None
    denoise_metadata: Optional[Dict[str, object]] = None
    transform_metadata: Optional[Dict[str, object]] = None
    atlas_name: Optional[str] = None
    roi_labels: List[str] = field(default_factory=list)
    roi_time_series: Optional[np.ndarray] = None
    connectivity_matrix: Optional[np.ndarray] = None
    applied_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    connectivity_metadata: Dict[str, object] = field(default_factory=dict)


class ModelMRIConnectivityData:
    """
    Clase principal para calcular la conectividad funcional entre ROIs.

    Uso esperado:
    - recibe un MRITransformBundle,
    - selecciona la estrategia de conectividad,
    - calcula la matriz ROI x ROI,
    - aplica opcionalmente un threshold,
    - devuelve un MRIConnectivityBundle.
    """

    def __init__(
        self,
        transform_bundle: MRITransformBundle,
        config: Optional[ConnectivityConfig] = None,
    ):
        self.transform_bundle = transform_bundle
        self.config = config if config is not None else ConnectivityConfig()

    def run(self) -> MRIConnectivityBundle:
        """
        Ejecuta el cálculo de conectividad.
        """
        self._validate_transform_bundle()

        roi_time_series = np.array(
            self.transform_bundle.roi_time_series,
            dtype=np.float32,
            copy=True
        )

        # --------------------------------------------------------
        # 1) Selección de la estrategia de conectividad
        # --------------------------------------------------------
        strategy = get_connectivity_strategy(self.config.method)

        # --------------------------------------------------------
        # 2) Cálculo de la matriz ROI x ROI
        # --------------------------------------------------------
        connectivity_matrix = strategy.compute(roi_time_series)

        applied_steps = [self.config.method]
        pending_steps: List[str] = []

        # --------------------------------------------------------
        # 3) Umbral opcional sobre la conectividad
        # --------------------------------------------------------
        if self.config.threshold is not None:
            connectivity_matrix = apply_connectivity_threshold(
                connectivity_matrix,
                self.config.threshold
            )
            applied_steps.append("threshold")

        # --------------------------------------------------------
        # 4) Conectividad dinámica
        # --------------------------------------------------------
        # El parámetro window_size existe ya en la configuración,
        # pero todavía no implementamos conectividad dinámica por ventanas.
        if self.config.window_size != 1.0:
            pending_steps.append("windowed_dynamic_connectivity")

        connectivity_metadata = self._build_connectivity_metadata(
            roi_time_series=roi_time_series,
            connectivity_matrix=connectivity_matrix,
        )

        bundle = MRIConnectivityBundle(
            fmri_path=self.transform_bundle.fmri_path,
            original_metadata=self.transform_bundle.original_metadata,
            preprocess_metadata=self.transform_bundle.preprocess_metadata,
            denoise_metadata=self.transform_bundle.denoise_metadata,
            transform_metadata=self.transform_bundle.transform_metadata,
            atlas_name=self.transform_bundle.atlas_name,
            roi_labels=list(self.transform_bundle.roi_labels),
            roi_time_series=roi_time_series,
            connectivity_matrix=connectivity_matrix,
            applied_steps=applied_steps,
            pending_steps=pending_steps,
            connectivity_metadata=connectivity_metadata,
        )

        return bundle

    def _validate_transform_bundle(self) -> None:
        """
        Comprueba que el bundle de transformación contiene lo necesario.
        """
        if self.transform_bundle is None:
            raise ConnectivityError(
                "No se ha proporcionado un MRITransformBundle válido."
            )

        if self.transform_bundle.roi_time_series is None:
            raise ConnectivityError(
                "El bundle de transformación no contiene roi_time_series."
            )

        if not isinstance(self.transform_bundle.roi_time_series, np.ndarray):
            raise ConnectivityError(
                "roi_time_series no es un ndarray de NumPy."
            )

        if self.transform_bundle.roi_time_series.ndim != 2:
            raise ConnectivityError(
                f"Se esperaba una matriz ROI x tiempo 2D, pero se obtuvo "
                f"{self.transform_bundle.roi_time_series.shape}."
            )

    def _build_connectivity_metadata(
        self,
        roi_time_series: np.ndarray,
        connectivity_matrix: np.ndarray,
    ) -> Dict[str, object]:
        """
        Construye metadatos útiles del cálculo de conectividad.
        """
        off_diagonal_mask = ~np.eye(connectivity_matrix.shape[0], dtype=bool)
        off_diagonal_values = connectivity_matrix[off_diagonal_mask]

        metadata = {
            "method": self.config.method,
            "threshold": self.config.threshold,
            "window_size": self.config.window_size,
            "num_rois": int(roi_time_series.shape[0]),
            "num_timepoints": int(roi_time_series.shape[1]),
            "roi_time_series_shape": tuple(roi_time_series.shape),
            "connectivity_matrix_shape": tuple(connectivity_matrix.shape),
            "matrix_dtype": str(connectivity_matrix.dtype),
            "matrix_is_symmetric": bool(
                np.allclose(connectivity_matrix, connectivity_matrix.T, atol=1e-5)
            ),
            "diagonal_all_ones": bool(
                np.allclose(np.diag(connectivity_matrix), 1.0, atol=1e-5)
            ),
            "min_connectivity": float(np.min(off_diagonal_values)) if off_diagonal_values.size > 0 else 0.0,
            "max_connectivity": float(np.max(off_diagonal_values)) if off_diagonal_values.size > 0 else 0.0,
            "mean_connectivity": float(np.mean(off_diagonal_values)) if off_diagonal_values.size > 0 else 0.0,
            "implemented_scope": [
                "static_connectivity",
                self.config.method,
                "optional_thresholding",
            ],
        }

        return metadata

    def display_info(self, bundle: MRIConnectivityBundle) -> None:
        """
        Muestra por pantalla un resumen del resultado del cálculo de conectividad.
        """
        print("\n[MRIGraph] Conectividad calculada")
        print(f"fMRI original: {bundle.fmri_path}")
        print(f"Método: {bundle.connectivity_metadata['method']}")
        print(f"Shape ROI x tiempo: {bundle.connectivity_metadata['roi_time_series_shape']}")
        print(
            f"Shape matriz conectividad: "
            f"{bundle.connectivity_metadata['connectivity_matrix_shape']}"
        )
        print(
            f"Matriz simétrica: "
            f"{bundle.connectivity_metadata['matrix_is_symmetric']}"
        )
        print(
            f"Diagonal a 1: "
            f"{bundle.connectivity_metadata['diagonal_all_ones']}"
        )
        print(
            f"Conectividad media (sin diagonal): "
            f"{bundle.connectivity_metadata['mean_connectivity']}"
        )

        if bundle.applied_steps:
            print("Pasos aplicados:")
            for step in bundle.applied_steps:
                print(f" - {step}")

        if bundle.pending_steps:
            print("Pasos pendientes de implementación:")
            for step in bundle.pending_steps:
                print(f" - {step}")