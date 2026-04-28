"""
Preprocesamiento previo a la transformación a grafo para experimentos con GNN.

Este script se ejecuta ANTES de usar la librería que transforma
fMRIs en grafos. La idea es separar claramente:

1) Preprocesamiento de señal y control de calidad
2) Transformación a grafo
3) Entrenamiento / evaluación de la GNN

Qué hace este script
-----------------------
- Carga fMRI en formato NIfTI (.nii / .nii.gz)
- Busca sidecar JSON del mismo BOLD en la misma carpeta (son los JSON que acompañan los .nii.gz en BIDS)
- Busca confounds reales del mismo run en la misma carpeta (si no existen, no pasa nada, el pipeline sigue sin regresión de confounds)
- Reemplaza NaN / inf por 0 para evitar fallos numéricos
- Normalización global de intensidad 
- Spatial smoothing (esto es para eliminar ruido espacial local y reforzar patrones regionales)
- Detección de outliers para control de calidad (SOLO INFORMACIÓN, no corrige la señal)
- Regresión de confounds REALES (solo si existen)
- Scrubbing REAL (solo si existen métricas reales como FD/DVARS/outliers)
- Band-pass temporal (si existe TR y hay suficientes timepoints)
- Guarda el BOLD preprocesado y un informe QC por sujeto

Qué NO hace este script
-----------------------
- Motion correction / realignment real
- Slice timing correction real
- Normalización espacial real a MNI
- Armonización automática entre fuentes / escáneres distintos

Esas partes se dejan fuera a propósito para no fingir una estandarización que en
realidad requiere pipelines más especializados.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt
from nibabel.processing import resample_to_output


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
# En esta sección se definen los parámetros del preprocesamiento.
# Conceptualmente, aquí decides QUÉ pasos se aplican y con QUÉ intensidad.
# Esto hace el pipeline reproducible y fácil de justificar en memoria.
# =============================================================================


@dataclass
class PipelineConfig:
    # Qué archivos buscar
    bold_patterns: Tuple[str, ...] = ("*_bold.nii.gz", "*_bold.nii")
    ignore_folders: Tuple[str, ...] = (
        "derivatives",
        "sourcedata",
        "source_data",
        "graph",
        "graphs",
        "__pycache__",
        ".git",
    )
    rest_tokens: Tuple[str, ...] = ("_task-rest_", "_task-resting_")

    # Pasos de preprocesamiento
    apply_nonfinite_cleanup: bool = True
    apply_global_intensity_normalization: bool = True
    apply_spatial_smoothing: bool = True
    smoothing_fwhm: float = 6.0

    # QC de outliers (SOLO INFORMACIÓN, no corrige señal)
    apply_outlier_qc: bool = True
    outlier_z_threshold: float = 3.0
    outlier_diff_z_threshold: float = 3.0

    # Confounds reales (solo si existen)
    apply_real_confound_regression: bool = True
    preferred_confound_columns: Tuple[str, ...] = (
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "white_matter", "csf", "global_signal",
    )

    # Scrubbing real (solo si existen métricas reales)
    apply_real_scrubbing: bool = True
    scrubbing_fd_threshold: float = 0.5
    scrubbing_std_dvars_threshold: float = 3.0
    min_timepoints_after_scrubbing: int = 30

    # Band-pass temporal
    apply_bandpass: bool = True
    bandpass_low: float = 0.008
    bandpass_high: float = 0.09
    bandpass_order: int = 4

    # Estandarización geométrica entre fuentes
    apply_canonical_reorientation: bool = True
    apply_common_voxel_resampling: bool = True
    target_voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0)

    # Guardado
    output_suffix: str = "desc-preproc-bold"



@dataclass
class QCReport:
    source_name: str
    subject_like_id: str
    input_bold: str
    output_bold: str
    json_sidecar_used: Optional[str]
    real_confounds_used: Optional[str]
    tr_seconds: Optional[float]
    original_shape: Tuple[int, ...]
    final_shape: Tuple[int, ...]
    nonfinite_replaced: int
    global_norm_applied: bool
    smoothing_applied: bool
    smoothing_fwhm: float
    outlier_qc_applied: bool
    outlier_indices: List[int]
    num_outlier_volumes: int
    confound_regression_applied: bool
    selected_confound_columns: List[str]
    scrubbing_applied: bool
    scrubbed_indices: List[int]
    num_scrubbed_volumes: int
    bandpass_applied: bool
    notes: List[str]
    scanner_manufacturer: Optional[str]
    scanner_model: Optional[str]
    magnetic_field_strength: Optional[float]
    canonical_reorientation_applied: bool
    common_voxel_resampling_applied: bool
    target_voxel_size: Tuple[float, float, float]
    shape_after_geometry_standardization: Tuple[int, ...]


# =============================================================================
# UTILIDADES DE BÚSQUEDA DE ARCHIVOS
# =============================================================================
# Estas funciones sirven para localizar solo los BOLD que nos interesan y para
# asociarles sus archivos auxiliares del mismo run.
# Conceptualmente, esto evita mezclar tareas, derivados o ficheros que no
# corresponden al mismo volumen funcional.
# =============================================================================


def path_contains_ignored_folder(path: Path, ignore_folders: Sequence[str]) -> bool:
    parts_lower = {part.lower() for part in path.parts}
    return any(folder.lower() in parts_lower for folder in ignore_folders)


def is_rest_bold(path: Path, rest_tokens: Sequence[str]) -> bool:
    name = path.name.lower()
    if not (name.endswith("_bold.nii.gz") or name.endswith("_bold.nii")):
        return False
    if not any(token in name for token in rest_tokens):
        return False
    if "func" not in [part.lower() for part in path.parts]:
        return False
    return True


def remove_nii_extension(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def iter_rest_bold_files(root: Path, config: PipelineConfig) -> Iterable[Path]:
    for pattern in config.bold_patterns:
        for path in root.rglob(pattern):
            if path_contains_ignored_folder(path, config.ignore_folders):
                continue
            if is_rest_bold(path, config.rest_tokens):
                yield path


def discover_same_run_sidecars(bold_path: Path) -> Tuple[Optional[Path], List[Path]]:
    """
    Busca auxiliares en la MISMA carpeta que el BOLD.

    Reglas:
    - JSON exacto del mismo BOLD
    - TSV/CSV/NPY del mismo run que contengan 'confound', 'timeseries' u 'outlier'

    Conceptualmente:
    - el JSON sidecar nos sirve para metadatos de adquisición, sobre todo el TR.
    - los confounds reales solo se usarán si son externos y compatibles.
    """
    folder = bold_path.parent
    stem = remove_nii_extension(bold_path.name)
    run_prefix = stem[:-5] if stem.endswith("_bold") else stem

    exact_json = folder / f"{stem}.json"
    aux_files: List[Path] = []

    for candidate in folder.iterdir():
        if not candidate.is_file():
            continue
        if candidate.resolve() == bold_path.resolve():
            continue

        suffix = candidate.suffix.lower()
        if suffix not in {".json", ".tsv", ".csv", ".npy"}:
            continue

        name_lower = candidate.name.lower()
        if run_prefix.lower() not in name_lower:
            continue
        if candidate == exact_json:
            continue
        if "confound" in name_lower or "timeseries" in name_lower or "outlier" in name_lower:
            aux_files.append(candidate)

    return (exact_json if exact_json.exists() else None), sorted(aux_files)


# =============================================================================
# CARGA DE DATOS Y METADATOS
# =============================================================================
# Aquí convertimos el NIfTI en un array 4D y leemos el JSON sidecar.
# Conceptualmente:
# - el .nii.gz contiene la imagen funcional
# - el .json contiene metadatos de adquisición, especialmente el TR
# =============================================================================


def load_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Se esperaba un fMRI 4D, pero se obtuvo shape {data.shape} en {path}")
    return img, data


def load_json_sidecar(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_tr_seconds(sidecar: Dict[str, object], img: nib.Nifti1Image) -> Optional[float]:
    """
    Intenta obtener el TR de forma robusta.
    1) Primero desde el JSON (RepetitionTime)
    2) Si no existe, desde el header del NIfTI (pixdim[4])
    """
    if "RepetitionTime" in sidecar:
        try:
            tr = float(sidecar["RepetitionTime"])
            if tr > 20:
                tr = tr / 1000.0
            return tr if tr > 0 else None
        except Exception:
            pass

    try:
        zooms = img.header.get_zooms()
        if len(zooms) >= 4:
            tr = float(zooms[3])
            if tr > 20:
                tr = tr / 1000.0
            return tr if tr > 0 else None
    except Exception:
        pass

    return None

# =============================================================================
# PREPROCESAMIENTO 0: ESTANDARIZACIÓN GEOMÉTRICA ENTRE FUENTES
# =============================================================================
# Qué hacemos en código:
# - reorientamos el BOLD a una orientación canónica común
# - lo resampleamos a un tamaño de voxel común
#
# Para qué sirve conceptualmente:
# - reduce diferencias geométricas entre datasets de distintos orígenes
# - no elimina por completo efectos de escáner/protocolo, pero sí reduce
#   una parte importante de la heterogeneidad espacial
# =============================================================================


def extract_scanner_metadata(sidecar: Dict[str, object]) -> Dict[str, object]:
    return {
        "scanner_manufacturer": sidecar.get("Manufacturer"),
        "scanner_model": sidecar.get("ManufacturersModelName"),
        "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
    }


def standardize_geometry(
    img: nib.Nifti1Image,
    apply_canonical_reorientation: bool,
    apply_common_voxel_resampling: bool,
    target_voxel_size: Tuple[float, float, float],
) -> Tuple[nib.Nifti1Image, List[str]]:
    applied_steps: List[str] = []
    work_img = img

    if apply_canonical_reorientation:
        work_img = nib.as_closest_canonical(work_img)
        applied_steps.append("canonical_reorientation")

    if apply_common_voxel_resampling:
        work_img = resample_to_output(
            work_img,
            voxel_sizes=target_voxel_size,
            order=3,  # interpolación trilineal
        )
        applied_steps.append("common_voxel_resampling")

    return work_img, applied_steps


# =============================================================================
# PREPROCESAMIENTO 1: LIMPIEZA DE VALORES NO FINITOS
# =============================================================================
# Qué hacemos en código:
# - sustituimos NaN, +inf y -inf por 0.
#
# Para qué sirve conceptualmente:
# - evita fallos numéricos en pasos posteriores.
# - no "mejora" la señal, pero sí hace el pipeline estable.
# =============================================================================


def replace_nonfinite_values(data: np.ndarray) -> Tuple[np.ndarray, int]:
    mask = ~np.isfinite(data)
    count = int(mask.sum())
    if count == 0:
        return data.astype(np.float32), 0
    cleaned = np.array(data, copy=True)
    cleaned[mask] = 0.0
    return cleaned.astype(np.float32), count


# =============================================================================
# PREPROCESAMIENTO 2: NORMALIZACIÓN GLOBAL DE INTENSIDAD
# =============================================================================
# Qué hacemos en código:
# - restamos la media global y dividimos por la desviación típica global.
#
# Para qué sirve conceptualmente:
# - pone todas las muestras en una escala numérica más homogénea.
# - evita que diferencias arbitrarias de intensidad dominen el pipeline.
# =============================================================================


def global_intensity_normalization(data: np.ndarray) -> np.ndarray:
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    if std_val == 0:
        raise ValueError("No se puede normalizar: la desviación típica global es 0.")
    return ((data - mean_val) / std_val).astype(np.float32)


# =============================================================================
# PREPROCESAMIENTO 3: SPATIAL SMOOTHING
# =============================================================================
# Qué hacemos en código:
# - aplicamos un filtro gaussiano sobre X, Y, Z y NO sobre el tiempo.
#
# Para qué sirve conceptualmente:
# - reduce ruido espacial local.
# - refuerza patrones regionales más estables.
# =============================================================================


def estimate_sigma_voxels(fwhm: float, img: nib.Nifti1Image) -> Tuple[float, float, float]:
    try:
        zooms = img.header.get_zooms()
        if len(zooms) >= 3:
            vx, vy, vz = float(zooms[0]), float(zooms[1]), float(zooms[2])
            vx = vx if vx > 0 else 1.0
            vy = vy if vy > 0 else 1.0
            vz = vz if vz > 0 else 1.0
            return ((fwhm / vx) / 2.355, (fwhm / vy) / 2.355, (fwhm / vz) / 2.355)
    except Exception:
        pass
    sigma = fwhm / 2.355
    return (sigma, sigma, sigma)


def apply_spatial_smoothing(data: np.ndarray, img: nib.Nifti1Image, fwhm: float) -> np.ndarray:
    if fwhm <= 0:
        return data.astype(np.float32)
    sx, sy, sz = estimate_sigma_voxels(fwhm, img)
    smoothed = gaussian_filter(data, sigma=(sx, sy, sz, 0.0), mode="nearest")
    return smoothed.astype(np.float32)


# =============================================================================
# PREPROCESAMIENTO 4: DETECCIÓN DE OUTLIERS PARA QC
# =============================================================================
# Qué hacemos en código:
# - calculamos la señal global por volumen
# - detectamos volúmenes atípicos por z-score y por cambios bruscos entre volúmenes
#
# Para qué sirve conceptualmente:
# - NO corrige la señal por sí solo.
# - sirve para control de calidad y para saber si hay volúmenes sospechosos.
# =============================================================================


def safe_zscore(vector: np.ndarray) -> np.ndarray:
    mean_val = float(np.mean(vector))
    std_val = float(np.std(vector))
    if std_val == 0:
        return np.zeros_like(vector, dtype=np.float32)
    return ((vector - mean_val) / std_val).astype(np.float32)


def detect_outliers_qc(data: np.ndarray, z_thr: float, diff_z_thr: float) -> List[int]:
    t = data.shape[-1]
    if t < 5:
        return []
    volume_means = data.reshape(-1, t).mean(axis=0)
    z_global = safe_zscore(volume_means)
    diff_signal = np.diff(volume_means, prepend=volume_means[0])
    z_diff = safe_zscore(diff_signal)
    bad_global = np.where(np.abs(z_global) > z_thr)[0]
    bad_diff = np.where(np.abs(z_diff) > diff_z_thr)[0]
    return sorted(set(bad_global.tolist() + bad_diff.tolist()))


# =============================================================================
# PREPROCESAMIENTO 5: CARGA DE CONFOUNDS REALES
# =============================================================================
# Qué hacemos en código:
# - buscamos archivos auxiliares reales del mismo run
# - extraemos columnas numéricas útiles si existen
#
# Para qué sirve conceptualmente:
# - si el usuario dispone de confounds reales (movimiento, WM, CSF, etc.),
#   podemos usarlos para limpiar la señal de forma más justificada.
# - si NO existen, el pipeline sigue sin regresión de confounds.
# =============================================================================


def load_aux_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".json":
        return load_json_sidecar(path)
    raise ValueError(f"Extensión auxiliar no soportada: {path}")


def extract_named_columns(aux_data) -> Dict[str, np.ndarray]:
    columns: Dict[str, np.ndarray] = {}

    if isinstance(aux_data, pd.DataFrame):
        for col in aux_data.columns:
            try:
                arr = pd.to_numeric(aux_data[col], errors="coerce").to_numpy(dtype=np.float32)
                columns[str(col)] = arr.reshape(-1)
            except Exception:
                continue
        return columns

    if isinstance(aux_data, np.ndarray) and aux_data.dtype.names is not None:
        for name in aux_data.dtype.names:
            try:
                arr = np.asarray(aux_data[name], dtype=np.float32).reshape(-1)
                columns[str(name)] = arr
            except Exception:
                continue
        return columns

    if isinstance(aux_data, dict):
        for key, value in aux_data.items():
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                columns[str(key)] = arr
            except Exception:
                continue
        return columns

    return columns


def select_real_confound_matrix(
    aux_files: Sequence[Path],
    num_timepoints: int,
    preferred_columns: Sequence[str],
) -> Tuple[Optional[str], Optional[np.ndarray], List[str], Dict[str, np.ndarray]]:
    all_named_cols: Dict[str, np.ndarray] = {}
    source_file_used: Optional[str] = None

    for aux_path in aux_files:
        try:
            aux_data = load_aux_file(aux_path)
        except Exception:
            continue

        named_cols = extract_named_columns(aux_data)
        if not named_cols:
            continue

        selected_arrays: List[np.ndarray] = []
        selected_names: List[str] = []

        # 1) columnas preferidas exactas
        for col in preferred_columns:
            if col in named_cols and named_cols[col].shape[0] == num_timepoints:
                selected_arrays.append(np.nan_to_num(named_cols[col], nan=0.0))
                selected_names.append(col)

        # 2) columnas tipo CompCor
        for key, values in named_cols.items():
            low = key.lower()
            if values.shape[0] != num_timepoints:
                continue
            if low.startswith("a_comp_cor") or low.startswith("t_comp_cor"):
                selected_arrays.append(np.nan_to_num(values, nan=0.0))
                selected_names.append(key)

        all_named_cols.update(named_cols)

        if selected_arrays:
            source_file_used = aux_path.name
            matrix = np.column_stack(selected_arrays).astype(np.float32)
            return source_file_used, matrix, selected_names, all_named_cols

    return None, None, [], all_named_cols


# =============================================================================
# PREPROCESAMIENTO 6: REGRESIÓN DE CONFOUNDS REALES
# =============================================================================
# Qué hacemos en código:
# - ajustamos un modelo lineal por voxel usando los confounds como regresores
# - nos quedamos con la señal residual
#
# Para qué sirve conceptualmente:
# - intenta quitar variabilidad explicable por fuentes de ruido no deseadas
#   (movimiento, WM, CSF, etc.) si esas variables existen realmente.
# =============================================================================


def regress_confounds(voxel_time_series: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    t = voxel_time_series.shape[1]
    if confounds.shape[0] != t:
        raise ValueError("La matriz de confounds no coincide en número de timepoints.")

    signal_t_by_v = voxel_time_series.T

    conf_mean = np.mean(confounds, axis=0, keepdims=True)
    conf_std = np.std(confounds, axis=0, keepdims=True)
    conf_std[conf_std == 0] = 1.0
    conf_z = (confounds - conf_mean) / conf_std

    design = np.column_stack([np.ones((t, 1), dtype=np.float32), conf_z])
    betas, _, _, _ = np.linalg.lstsq(design, signal_t_by_v, rcond=None)
    fitted = design @ betas
    residual = signal_t_by_v - fitted
    return residual.T.astype(np.float32)


# =============================================================================
# PREPROCESAMIENTO 7: SCRUBBING REAL
# =============================================================================
# Qué hacemos en código:
# - si existen métricas reales como FD, std_dvars o motion_outlier_XX,
#   detectamos volúmenes a eliminar
#
# Para qué sirve conceptualmente:
# - elimina volúmenes particularmente problemáticos por movimiento/artefactos
# - SOLO debe usarse si las métricas vienen dadas por datos reales auxiliares
# =============================================================================


def build_real_scrubbing_mask(
    named_columns: Dict[str, np.ndarray],
    num_timepoints: int,
    fd_threshold: float,
    std_dvars_threshold: float,
) -> Optional[np.ndarray]:
    bad = np.zeros(num_timepoints, dtype=bool)
    found = False

    for key, values in named_columns.items():
        low = key.lower()
        if values.shape[0] != num_timepoints:
            continue

        vec = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if low.startswith("motion_outlier") or low.startswith("non_steady_state_outlier"):
            bad |= vec > 0
            found = True

    if "framewise_displacement" in named_columns:
        fd = np.nan_to_num(named_columns["framewise_displacement"].astype(np.float32), nan=0.0)
        if fd.shape[0] == num_timepoints:
            bad |= fd > fd_threshold
            found = True

    if "std_dvars" in named_columns:
        std_dvars = np.nan_to_num(named_columns["std_dvars"].astype(np.float32), nan=0.0)
        if std_dvars.shape[0] == num_timepoints:
            bad |= std_dvars > std_dvars_threshold
            found = True
    elif "dvars" in named_columns:
        dvars = np.nan_to_num(named_columns["dvars"].astype(np.float32), nan=0.0)
        if dvars.shape[0] == num_timepoints:
            bad |= np.abs(safe_zscore(dvars)) > std_dvars_threshold
            found = True

    return bad if found else None


# =============================================================================
# PREPROCESAMIENTO 8: BAND-PASS TEMPORAL
# =============================================================================
# Qué hacemos en código:
# - filtramos la señal temporal voxel a voxel en un rango de frecuencias útil
#
# Para qué sirve conceptualmente:
# - elimina componentes muy lentas y muy rápidas que suelen ser menos útiles
#   para conectividad funcional en reposo.
# =============================================================================


def apply_bandpass_filter(
    voxel_time_series: np.ndarray,
    tr_seconds: float,
    low: float,
    high: float,
    order: int,
) -> np.ndarray:
    if tr_seconds <= 0:
        raise ValueError("TR inválido para band-pass.")

    fs = 1.0 / tr_seconds
    nyquist = fs / 2.0

    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("Parámetros de band-pass inválidos.")
    if high >= nyquist:
        raise ValueError("La frecuencia alta supera Nyquist.")
    if voxel_time_series.shape[1] < 20:
        raise ValueError("No hay suficientes timepoints para aplicar band-pass con estabilidad.")

    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, voxel_time_series, axis=1)
    return filtered.astype(np.float32)


# =============================================================================
# GUARDADO
# =============================================================================
# Guardamos:
# - el volumen preprocesado en NIfTI
# - un informe QC en JSON
#
# Conceptualmente:
# - el NIfTI preprocesado es el que luego usarás para transformación a grafo
# - el QC te permite justificar qué se hizo y comprobar si fue razonable
# =============================================================================


def build_output_bold_path(input_bold: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    rel = input_bold.relative_to(input_root)
    stem = remove_nii_extension(rel.name)
    out_name = f"{stem}_{suffix}.nii.gz"
    return output_root / rel.parent / out_name


def build_output_qc_path(input_bold: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_bold.relative_to(input_root)
    stem = remove_nii_extension(rel.name)
    out_name = f"{stem}_qc_report.json"
    return output_root / rel.parent / out_name


def save_nifti(output_path: Path, data: np.ndarray, reference_img: nib.Nifti1Image) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(data.astype(np.float32), affine=reference_img.affine, header=reference_img.header)
    nib.save(out_img, str(output_path))


def save_qc_report(output_path: Path, report: QCReport) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# PIPELINE PRINCIPAL POR ARCHIVO
# =============================================================================
# Aquí se orquestan todos los pasos anteriores.
# =============================================================================


def infer_subject_like_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return path.stem


def preprocess_single_bold(bold_path: Path, input_root: Path, output_root: Path, source_name: str, config: PipelineConfig) -> QCReport:
    raw_img, raw_data = load_nifti(bold_path)
    sidecar_json_path, aux_files = discover_same_run_sidecars(bold_path)
    sidecar = load_json_sidecar(sidecar_json_path)
    tr_seconds = extract_tr_seconds(sidecar, raw_img)
    scanner_meta = extract_scanner_metadata(sidecar)

    notes: List[str] = []

    original_shape = tuple(int(x) for x in raw_data.shape)

    # 0) Estandarización geométrica entre fuentes
    img, geometry_steps = standardize_geometry(
        raw_img,
        apply_canonical_reorientation=config.apply_canonical_reorientation,
        apply_common_voxel_resampling=config.apply_common_voxel_resampling,
        target_voxel_size=config.target_voxel_size,
    )

    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Se esperaba un fMRI 4D tras la estandarización geométrica, pero se obtuvo {data.shape}")

    shape_after_geometry_standardization = tuple(int(x) for x in data.shape)
    
    nonfinite_replaced = 0
    outlier_indices: List[int] = []
    confound_regression_applied = False
    selected_confound_columns: List[str] = []
    real_confounds_used: Optional[str] = None
    scrubbed_indices: List[int] = []
    scrubbing_applied = False
    bandpass_applied = False

    # 1) Limpieza de no finitos
    if config.apply_nonfinite_cleanup:
        data, nonfinite_replaced = replace_nonfinite_values(data)

    # 2) Normalización global
    if config.apply_global_intensity_normalization:
        data = global_intensity_normalization(data)

    # 3) Smoothing espacial
    if config.apply_spatial_smoothing:
        data = apply_spatial_smoothing(data, img, config.smoothing_fwhm)

    # 4) QC de outliers
    if config.apply_outlier_qc:
        outlier_indices = detect_outliers_qc(
            data,
            z_thr=config.outlier_z_threshold,
            diff_z_thr=config.outlier_diff_z_threshold,
        )
        if outlier_indices:
            notes.append(
                f"Se detectaron {len(outlier_indices)} volúmenes atípicos para QC. "
                "No se usaron como confounds automáticos."
            )

    # Pasamos a representación voxel x tiempo para denoising temporal
    voxel_time_series = data.reshape(-1, data.shape[-1]).astype(np.float32)

    # 5) Confounds reales
    _, confound_matrix, selected_confound_columns, named_cols = select_real_confound_matrix(
        aux_files=aux_files,
        num_timepoints=voxel_time_series.shape[1],
        preferred_columns=config.preferred_confound_columns,
    )
    if config.apply_real_confound_regression and confound_matrix is not None:
        voxel_time_series = regress_confounds(voxel_time_series, confound_matrix)
        confound_regression_applied = True
        # Buscamos el nombre real solo para dejar trazabilidad
        confound_name, _, _, _ = select_real_confound_matrix(
            aux_files=aux_files,
            num_timepoints=voxel_time_series.shape[1],
            preferred_columns=config.preferred_confound_columns,
        )
        real_confounds_used = confound_name
    elif config.apply_real_confound_regression:
        notes.append("No se encontraron confounds reales compatibles; no se aplicó regresión de confounds.")
        named_cols = named_cols if 'named_cols' in locals() else {}
    else:
        named_cols = named_cols if 'named_cols' in locals() else {}

    # 6) Scrubbing real
    if config.apply_real_scrubbing:
        bad_mask = build_real_scrubbing_mask(
            named_columns=named_cols,
            num_timepoints=voxel_time_series.shape[1],
            fd_threshold=config.scrubbing_fd_threshold,
            std_dvars_threshold=config.scrubbing_std_dvars_threshold,
        )
        if bad_mask is not None and np.any(bad_mask):
            kept_mask = ~bad_mask
            if int(np.sum(kept_mask)) >= config.min_timepoints_after_scrubbing:
                scrubbed_indices = np.where(bad_mask)[0].tolist()
                voxel_time_series = voxel_time_series[:, kept_mask]
                scrubbing_applied = True
            else:
                notes.append("Se detectaron volúmenes para scrubbing, pero eliminarlos dejaría muy pocos timepoints.")
        elif config.apply_real_scrubbing:
            notes.append("No se encontraron métricas reales para aplicar scrubbing.")

    # 7) Band-pass
    if config.apply_bandpass:
        if tr_seconds is not None:
            try:
                voxel_time_series = apply_bandpass_filter(
                    voxel_time_series,
                    tr_seconds=tr_seconds,
                    low=config.bandpass_low,
                    high=config.bandpass_high,
                    order=config.bandpass_order,
                )
                bandpass_applied = True
            except Exception as exc:
                notes.append(f"No se pudo aplicar band-pass: {exc}")
        else:
            notes.append("No se encontró TR; no se aplicó band-pass.")

    # Reconstrucción 4D
    final_timepoints = voxel_time_series.shape[1]
    final_shape = data.shape[:3] + (final_timepoints,)
    final_data = voxel_time_series.reshape(final_shape).astype(np.float32)

    # Guardado
    out_bold = build_output_bold_path(bold_path, input_root, output_root, config.output_suffix)
    out_qc = build_output_qc_path(bold_path, input_root, output_root)
    save_nifti(out_bold, final_data, img)

    report = QCReport(
        source_name=source_name,
        subject_like_id=infer_subject_like_id(bold_path),
        input_bold=str(bold_path),
        output_bold=str(out_bold),
        json_sidecar_used=str(sidecar_json_path) if sidecar_json_path else None,
        real_confounds_used=real_confounds_used,
        tr_seconds=tr_seconds,
        original_shape=original_shape,
        final_shape=tuple(int(x) for x in final_shape),
        nonfinite_replaced=nonfinite_replaced,
        global_norm_applied=config.apply_global_intensity_normalization,
        smoothing_applied=config.apply_spatial_smoothing,
        smoothing_fwhm=config.smoothing_fwhm,
        outlier_qc_applied=config.apply_outlier_qc,
        outlier_indices=outlier_indices,
        num_outlier_volumes=len(outlier_indices),
        confound_regression_applied=confound_regression_applied,
        selected_confound_columns=selected_confound_columns,
        scrubbing_applied=scrubbing_applied,
        scrubbed_indices=scrubbed_indices,
        num_scrubbed_volumes=len(scrubbed_indices),
        bandpass_applied=bandpass_applied,
        notes=notes,
        scanner_manufacturer=scanner_meta.get("scanner_manufacturer"),
        scanner_model=scanner_meta.get("scanner_model"),
        magnetic_field_strength=scanner_meta.get("magnetic_field_strength"),
        canonical_reorientation_applied=("canonical_reorientation" in geometry_steps),
        common_voxel_resampling_applied=("common_voxel_resampling" in geometry_steps),
        target_voxel_size=config.target_voxel_size,
        shape_after_geometry_standardization=shape_after_geometry_standardization,
    )
    save_qc_report(out_qc, report)
    return report


# =============================================================================
# RESUMEN GLOBAL
# =============================================================================
# Este resumen te ayuda a comprobar rápidamente si el preprocesamiento ha ido
# razonablemente bien a nivel de lote completo.
# =============================================================================


def save_global_summary(reports: Sequence[QCReport], output_root: Path) -> Path:
    rows = []
    for rep in reports:
        rows.append(
            {
                "source_name": rep.source_name,
                "subject_like_id": rep.subject_like_id,
                "input_bold": rep.input_bold,
                "output_bold": rep.output_bold,
                "tr_seconds": rep.tr_seconds,
                "original_shape": rep.original_shape,
                "final_shape": rep.final_shape,
                "nonfinite_replaced": rep.nonfinite_replaced,
                "num_outlier_volumes": rep.num_outlier_volumes,
                "confound_regression_applied": rep.confound_regression_applied,
                "real_confounds_used": rep.real_confounds_used,
                "num_scrubbed_volumes": rep.num_scrubbed_volumes,
                "bandpass_applied": rep.bandpass_applied,
                "notes": " | ".join(rep.notes),
            }
        )
    df = pd.DataFrame(rows)
    output_path = output_root / "qc_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


# =============================================================================
# CLI
# =============================================================================
# Esto permite ejecutar el script desde terminal.
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento previo a la transformación a grafo para GNN."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Raíz con los BOLD raw.")
    parser.add_argument("--output-root", type=Path, required=True, help="Raíz donde guardar los BOLD preprocesados.")
    parser.add_argument(
        "--source-name",
        type=str,
        default="desconocida",
        help="Nombre de la fuente/dataset. Se guarda en el informe QC para trazabilidad.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    source_name: str = args.source_name

    bold_files = sorted(iter_rest_bold_files(input_root, config))
    print("=" * 90)
    print(f"BOLD rest detectados para procesar: {len(bold_files)}")
    print(f"Fuente declarada: {source_name}")
    print("=" * 90)

    reports: List[QCReport] = []
    for idx, bold_path in enumerate(bold_files, start=1):
        print("\n" + "-" * 90)
        print(f"[{idx}/{len(bold_files)}] Procesando: {bold_path}")
        try:
            report = preprocess_single_bold(
                bold_path=bold_path,
                input_root=input_root,
                output_root=output_root,
                source_name=source_name,
                config=config,
            )
            reports.append(report)
            print(f"OK -> salida: {report.output_bold}")
            print(f"   TR detectado: {report.tr_seconds}")
            print(f"   Outliers QC: {report.num_outlier_volumes}")
            print(f"   Confounds reales usados: {report.real_confounds_used}")
            print(f"   Scrubbing aplicado: {report.scrubbing_applied} ({report.num_scrubbed_volumes} volúmenes)")
            print(f"   Band-pass aplicado: {report.bandpass_applied}")
        except Exception as exc:
            print(f"ERROR procesando {bold_path}: {exc}")

    summary_path = save_global_summary(reports, output_root)
    print("\n" + "=" * 90)
    print(f"Resumen global guardado en: {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
