from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InputConfig:
    """
    Configuración relacionada con los datos de entrada.

    Controla:
    - qué extensiones se aceptan,
    - si queremos exigir archivos auxiliares,
    - si permitimos entradas tabulares ya derivadas.
    """
    allowed_fmri_extensions: List[str] = field(default_factory=lambda: [".nii", ".nii.gz"])
    allowed_aux_extensions: List[str] = field(default_factory=lambda: [".json", ".tsv", ".csv", ".npy"])
    allow_tabular_inputs: bool = True


@dataclass
class PreprocessConfig:
    """
    Configuración de la fase de preprocesado.
    """
    apply_motion_correction: bool = True
    apply_slice_timing: bool = True
    apply_outlier_detection: bool = True
    apply_normalization: bool = True
    apply_smoothing: bool = True

    # Parámetro de ejemplo para smoothing.
    # Más adelante se podrá ajustar mejor según el pipeline real.
    smoothing_fwhm: float = 6.0


@dataclass
class DenoiseConfig:
    """
    Configuración de la fase de denoising.

    Aquí definimos qué limpieza de señal queremos aplicar.
    """
    regress_confounds: bool = True
    apply_scrubbing: bool = True
    apply_bandpass: bool = True

    # Banda típica para conectividad funcional.
    # De momento la dejamos configurable.
    bandpass_low: float = 0.008
    bandpass_high: float = 0.09


@dataclass
class AtlasConfig:
    """
    Configuración relacionada con el atlas.

    controla:
    - qué atlas vamos a usar,
    - si se pasa por nombre soportado o por ruta personalizada,
    - si queremos etiquetas manuales.
    """
    atlas_name: Optional[str] = None
    atlas_path: Optional[str] = None
    roi_labels: Optional[List[str]] = None


@dataclass
class ConnectivityConfig:
    """
    Configuración del cálculo de conectividad.

    De momento solo dejamos preparado Pearson, porque es el primer caso de uso y además queremos que sea coherente con EEGraph.
    """
    method: str = "pearson_correlation"
    window_size: float = 1.0
    threshold: Optional[float] = None


@dataclass
class MRIGraphConfig:
    """
    Configuración general de toda la librería.

    Esta clase agrupa todas las configuraciones parciales.
    Así podemos pasar un único objeto de configuración a los distintos componentes del pipeline.
    """
    input_config: InputConfig = field(default_factory=InputConfig)
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)
    denoise_config: DenoiseConfig = field(default_factory=DenoiseConfig)
    atlas_config: AtlasConfig = field(default_factory=AtlasConfig)
    connectivity_config: ConnectivityConfig = field(default_factory=ConnectivityConfig)