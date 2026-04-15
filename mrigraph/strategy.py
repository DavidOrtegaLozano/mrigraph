from abc import ABC, abstractmethod
import numpy as np
from .tools import compute_pearson_correlation,compute_cross_correlation,compute_corrected_cross_correlation,validate_connectivity_method

class ConnectivityStrategy(ABC):
    """
    Clase base abstracta para estrategias de conectividad.

    Cualquier estrategia concreta debe implementar el método compute().
    """

    @abstractmethod
    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        """
        Calcula una matriz de conectividad a partir de series ROI x tiempo.
        """
        pass


class PearsonConnectivityStrategy(ConnectivityStrategy):
    """
    Estrategia concreta para conectividad basada en Pearson.
    """

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_pearson_correlation(roi_time_series)


class CrossCorrelationConnectivityStrategy(ConnectivityStrategy):
    """
    Estrategia concreta para conectividad basada en correlación cruzada.
    """

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_cross_correlation(roi_time_series)


class CorrectedCrossCorrelationConnectivityStrategy(ConnectivityStrategy):
    """
    Estrategia concreta para conectividad basada en correlación cruzada corregida.
    """

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_corrected_cross_correlation(roi_time_series)


def get_connectivity_strategy(method: str) -> ConnectivityStrategy:
    """
    Devuelve la estrategia de conectividad asociada al método solicitado.

    Soportadas ahora:
    - pearson_correlation
    - cross_correlation
    - corr_cross_correlation
    """
    normalized_method = validate_connectivity_method(method)

    if normalized_method == "pearson_correlation":
        return PearsonConnectivityStrategy()

    if normalized_method == "cross_correlation":
        return CrossCorrelationConnectivityStrategy()

    if normalized_method == "corr_cross_correlation":
        return CorrectedCrossCorrelationConnectivityStrategy()

    raise ValueError(
        f"No se pudo construir una estrategia para el método '{method}'."
    )