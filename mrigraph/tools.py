from typing import Dict, List
import numpy as np
from scipy import signal
from .exceptions import ConnectivityError


# ============================================================
# Registro de medidas de conectividad soportadas.
# ============================================================
CONNECTIVITY_MEASURES: Dict[str, str] = {
    "pearson_correlation": "Correlación de Pearson",
    "cross_correlation": "Correlación cruzada",
    "corr_cross_correlation": "Correlación cruzada corregida",
}

# Alias permitidos. Hacemos un diccionario para parsear los diferentes nombres que pueda meta el uusario
CONNECTIVITY_ALIASES: Dict[str, str] = {
    "pearson": "pearson_correlation",
    "pearson_correlation": "pearson_correlation",
    "pearson correlation": "pearson_correlation",

    "cross_correlation": "cross_correlation",
    "cross-correlation": "cross_correlation",
    "cross correlation": "cross_correlation",

    "corr_cross_correlation": "corr_cross_correlation",
    "corrected_cross_correlation": "corr_cross_correlation",
    "corrected cross correlation": "corr_cross_correlation",
    "corrected cross-correlation": "corr_cross_correlation",
}


def list_connectivity_measures() -> List[str]:
    """
    Devuelve la lista de nombres canónicos soportados.
    """
    return list(CONNECTIVITY_MEASURES.keys())


def validate_connectivity_method(method: str) -> str:
    """
    Valida y normaliza el método de conectividad.
    """
    if method is None:
        raise ConnectivityError("El método de conectividad no puede ser None.")

    normalized_method = method.lower().strip()
    normalized_method = CONNECTIVITY_ALIASES.get(normalized_method, normalized_method)

    if normalized_method not in CONNECTIVITY_MEASURES:
        raise ConnectivityError(
            f"El método de conectividad '{method}' no está soportado. "
            f"Métodos disponibles: {', '.join(list_connectivity_measures())}"
        )

    return normalized_method


def validate_roi_time_series(roi_time_series: np.ndarray) -> None:
    """
    Valida que la entrada tenga una forma compatible con conectividad.

    Forma esperada:
    - matriz 2D con shape (num_rois, num_timepoints)
    """
    if not isinstance(roi_time_series, np.ndarray):
        raise ConnectivityError(
            "Las series ROI deben ser un ndarray de NumPy."
        )

    if roi_time_series.ndim != 2:
        raise ConnectivityError(
            f"Se esperaba una matriz ROI x tiempo 2D, pero se recibió "
            f"una estructura con forma {roi_time_series.shape}."
        )

    num_rois, num_timepoints = roi_time_series.shape

    if num_rois < 2:
        raise ConnectivityError(
            "Se necesitan al menos 2 ROIs para calcular conectividad."
        )

    if num_timepoints < 2:
        raise ConnectivityError(
            "Se necesitan al menos 2 timepoints para calcular conectividad."
        )


def compute_pearson_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de conectividad usando correlación de Pearson.
    """
    validate_roi_time_series(roi_time_series)

    try:
        connectivity_matrix = np.corrcoef(roi_time_series, rowvar=True)
    except Exception as exc:
        raise ConnectivityError(
            "No se pudo calcular la correlación de Pearson."
        ) from exc

    connectivity_matrix = np.asarray(connectivity_matrix, dtype=np.float32)

    connectivity_matrix = np.nan_to_num(
        connectivity_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(connectivity_matrix, 1.0)

    return connectivity_matrix


def _normalized_cross_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Reproduce la normalización usada en EEGraph:

    Rxy_norm = (1 / sqrt(Rxx_0 * Ryy_0)) * Rxy
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.ndim != 1 or y.ndim != 1:
        raise ConnectivityError(
            "Las señales individuales para cross-correlation deben ser 1D."
        )

    if len(x) != len(y):
        raise ConnectivityError(
            "Las señales a comparar deben tener el mismo número de timepoints."
        )

    try:
        Rxy = signal.correlate(x, y, mode="full")
        Rxx = signal.correlate(x, x, mode="full")
        Ryy = signal.correlate(y, y, mode="full")
    except Exception as exc:
        raise ConnectivityError(
            "No se pudo calcular la correlación cruzada."
        ) from exc

    lags = np.arange(-len(x) + 1, len(x))
    lag_0 = int(np.where(lags == 0)[0][0])

    Rxx_0 = Rxx[lag_0]
    Ryy_0 = Ryy[lag_0]

    if Rxx_0 == 0 or Ryy_0 == 0:
        return np.zeros_like(Rxy, dtype=np.float32)

    Rxy_norm = (1.0 / np.sqrt(Rxx_0 * Ryy_0)) * Rxy
    Rxy_norm = np.asarray(Rxy_norm, dtype=np.float32)

    Rxy_norm = np.nan_to_num(
        Rxy_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return Rxy_norm


def _cross_correlation_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Reproduce la lógica de Cross_correlation_Estimator de EEGraph.

    En EEGraph:
    - se calcula Rxy_norm
    - se toma la media desde lag 0 hasta un desplazamiento del 10%
    """
    Rxy_norm = _normalized_cross_correlation(x, y)

    lags = np.arange(-len(x) + 1, len(x))
    lag_0 = int(np.where(lags == 0)[0][0])

    disp = max(1, round(len(x) * 0.10))
    fragment = Rxy_norm[lag_0: lag_0 + disp]

    if fragment.size == 0:
        return 0.0

    return float(np.mean(fragment))


def _corr_cross_correlation_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Reproduce la lógica de Corr_cross_correlation_Estimator de EEGraph.

    En EEGraph:
    - se calcula Rxy_norm
    - se separan negative_lag y positive_lag
    - se calcula corCC = positive_lag - negative_lag
    - se toma la media del primer 10%
    """
    Rxy_norm = _normalized_cross_correlation(x, y)

    lags = np.arange(-len(x) + 1, len(x))
    lag_0 = int(np.where(lags == 0)[0][0])

    negative_lag = Rxy_norm[:lag_0]
    positive_lag = Rxy_norm[lag_0 + 1:]

    if negative_lag.size == 0 or positive_lag.size == 0:
        return 0.0

    # Así sale en EEGraph: positive_lag - negative_lag
    corCC = positive_lag - negative_lag

    disp = max(1, round(len(x) * 0.10))
    fragment = corCC[:disp]

    if fragment.size == 0:
        return 0.0

    return float(np.mean(fragment))


def compute_cross_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz ROI x ROI usando la lógica de
    Cross_correlation_Estimator de EEGraph.

    IMPORTANTE:
    - Esta medida puede ser asimétrica.
    - Por eso NO se fuerza simetría.
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(num_rois):
            connectivity_matrix[i, j] = _cross_correlation_coef(
                roi_time_series[i],
                roi_time_series[j]
            )

    connectivity_matrix = np.nan_to_num(
        connectivity_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return connectivity_matrix.astype(np.float32)


def compute_corrected_cross_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz ROI x ROI usando la lógica de
    Corr_cross_correlation_Estimator de EEGraph.

    IMPORTANTE:
    - Esta medida puede ser asimétrica.
    - Por eso NO se fuerza simetría.
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(num_rois):
            connectivity_matrix[i, j] = _corr_cross_correlation_coef(
                roi_time_series[i],
                roi_time_series[j]
            )

    connectivity_matrix = np.nan_to_num(
        connectivity_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return connectivity_matrix.astype(np.float32)


def apply_connectivity_threshold(
    connectivity_matrix: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Aplica un umbral absoluto a una matriz de conectividad.

    Estrategia:
    - se conservan los valores cuyo valor absoluto sea >= threshold
    - los demás se ponen a 0
    - NO se fuerza la diagonal a 1, para no alterar artificialmente
      medidas que no lo hacen de forma natural.
    """
    if threshold < 0:
        raise ConnectivityError(
            "El threshold de conectividad no puede ser negativo."
        )

    thresholded_matrix = np.array(connectivity_matrix, copy=True)
    mask = np.abs(thresholded_matrix) < threshold
    thresholded_matrix[mask] = 0.0

    return thresholded_matrix.astype(np.float32)