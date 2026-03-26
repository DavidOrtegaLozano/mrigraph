"""
MRIGraph

Submódulo de EEGraph orientado al trabajo con fMRI.
Su objetivo es seguir una filosofía similar a EEGraph, pero adaptada
al flujo de trabajo de resonancia magnética funcional:

entrada -> preprocesado -> denoising -> atlas -> ROI -> conectividad -> grafo
"""

from .config import InputConfig,PreprocessConfig,DenoiseConfig,AtlasConfig,ConnectivityConfig,MRIGraphConfig
from .importMRIData import InputMRIData, MRIInputBundle
from .preprocess import PreprocessMRIData, MRIPreprocessBundle
from .denoise import DenoiseMRIData, MRIDenoiseBundle
from .transform import TransformMRIData, MRITransformBundle, build_synthetic_atlas
from .strategy import ConnectivityStrategy, PearsonConnectivityStrategy, get_connectivity_strategy
from .modelateData import ModelMRIConnectivityData, MRIConnectivityBundle
from .tools import CONNECTIVITY_MEASURES, list_connectivity_measures, validate_connectivity_method, compute_pearson_correlation, apply_connectivity_threshold
from .strategy import ConnectivityStrategy, PearsonConnectivityStrategy, get_connectivity_strategy
from .modelateData import ModelMRIConnectivityData, MRIConnectivityBundle

__all__ = [
    "InputConfig",
    "PreprocessConfig",
    "DenoiseConfig",
    "AtlasConfig",
    "ConnectivityConfig",
    "MRIGraphConfig",
    "InputMRIData",
    "MRIInputBundle",
    "PreprocessMRIData",
    "MRIPreprocessBundle",
    "DenoiseMRIData",
    "MRIDenoiseBundle",
    "TransformMRIData",
    "MRITransformBundle",
    "build_synthetic_atlas",
    "CONNECTIVITY_MEASURES",
    "list_connectivity_measures",
    "validate_connectivity_method",
    "compute_pearson_correlation",
    "apply_connectivity_threshold",
    "ConnectivityStrategy",
    "PearsonConnectivityStrategy",
    "get_connectivity_strategy",
    "ModelMRIConnectivityData",
    "MRIConnectivityBundle",
]