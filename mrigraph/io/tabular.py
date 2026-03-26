"""
Paso 4 del pipeline MRI.
Archivo: io/tabular.py

Qué hace este archivo:
- Carga archivos tabulares auxiliares o derivados.
- Soporta .tsv, .csv y .npy.
- Su objetivo NO es interpretar todavía el significado clínico
  o de pipeline de esos archivos, sino cargarlos correctamente.

Más adelante estos archivos podrán representar, por ejemplo:
- confounds,
- eventos,
- series ROI,
- matrices de conectividad,
- otros derivados del pipeline.
"""

from pathlib import Path
from typing import Union

import numpy as np

from ..exceptions import MRIInputError, MRIFormatError


def load_npy_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Carga un archivo .npy y devuelve su contenido como ndarray.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise MRIInputError(f"El archivo no existe: {file_path}")

    if file_path.suffix.lower() != ".npy":
        raise MRIInputError(f"Se esperaba un archivo .npy: {file_path.name}")

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as exc:
        raise MRIFormatError(
            f"No se pudo cargar correctamente el archivo .npy: {file_path}"
        ) from exc

    return data


def load_delimited_file(file_path: Union[str, Path], delimiter: str) -> np.ndarray:
    """
    Carga un archivo delimitado (.csv o .tsv) intentando obtener
    una matriz numérica.

    Nota:
    - Si el archivo tiene cabecera o columnas no numéricas, puede ser
      necesario interpretarlo más adelante de forma específica.
    - Aquí hacemos una carga genérica inicial.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise MRIInputError(f"El archivo no existe: {file_path}")

    try:
        data = np.genfromtxt(file_path, delimiter=delimiter, dtype=float, autostrip=True)
    except Exception as exc:
        raise MRIFormatError(
            f"No se pudo cargar correctamente el archivo tabular: {file_path}"
        ) from exc

    return data


def load_csv_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Carga un archivo .csv.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() != ".csv":
        raise MRIInputError(f"Se esperaba un archivo .csv: {file_path.name}")

    return load_delimited_file(file_path, delimiter=",")


def load_tsv_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Carga un archivo .tsv.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() != ".tsv":
        raise MRIInputError(f"Se esperaba un archivo .tsv: {file_path.name}")

    return load_delimited_file(file_path, delimiter="\t")


def load_tabular_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Carga automáticamente un archivo tabular soportado:
    - .csv
    - .tsv
    - .npy
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_csv_file(file_path)

    if suffix == ".tsv":
        return load_tsv_file(file_path)

    if suffix == ".npy":
        return load_npy_file(file_path)

    raise MRIInputError(
        f"Extensión tabular no soportada: {file_path.name}. "
        "Soportadas: .csv, .tsv, .npy"
    )