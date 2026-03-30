"""
Paso 4 del pipeline MRI.
Archivo: io/nifti.py

Qué hace este archivo:
- Carga ficheros NIfTI (.nii / .nii.gz).
- Valida que existan y que tengan una estructura coherente.
- Devuelve tanto la imagen cargada como información útil sobre sus dimensiones.

Importante:
Este archivo NO aplica preprocesado, NO aplica atlas
y NO calcula ROIs. Solo carga y valida imágenes NIfTI.
"""

from pathlib import Path
from typing import Dict, Union

from ..exceptions import MRIInputError, MRIFormatError


def load_nifti_file(file_path: Union[str, Path]):
    """
    Carga un archivo NIfTI usando nibabel.

    Parámetros:
    - file_path: ruta al archivo .nii o .nii.gz

    Devuelve:
    - objeto NIfTI cargado

    Errores:
    - MRIInputError si el archivo no existe o no tiene extensión válida
    - MRIFormatError si no se puede cargar correctamente
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise MRIInputError(
            "No se pudo importar nibabel. Instálalo para trabajar con archivos NIfTI."
        ) from exc

    file_path = Path(file_path)

    if not file_path.exists():
        raise MRIInputError(f"El archivo NIfTI no existe: {file_path}")

    suffix = "".join(file_path.suffixes).lower()
    if not (suffix.endswith(".nii") or suffix.endswith(".nii.gz")):
        raise MRIInputError(
            f"Extensión no válida para archivo NIfTI: {file_path.name}"
        )

    try:
        image = nib.load(str(file_path))
    except Exception as exc:
        raise MRIFormatError(
            f"No se pudo cargar correctamente el archivo NIfTI: {file_path}"
        ) from exc

    return image


def get_nifti_metadata(image) -> Dict[str, object]:
    """
    Extrae metadatos básicos de una imagen NIfTI ya cargada.

    Devuelve:
    - shape: dimensiones del volumen
    - ndim: número de dimensiones
    - affine: matriz affine
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise MRIInputError(
            "No se pudo importar nibabel. Instálalo para trabajar con archivos NIfTI."
        ) from exc

    shape = image.shape
    ndim = len(shape)
    affine = image.affine
    zooms = tuple(float(z) for z in image.header.get_zooms()[:ndim])
    axis_codes = tuple(nib.aff2axcodes(affine))

    return {
        "shape": shape,
        "ndim": ndim,
        "affine": affine,
        "zooms": zooms,
        "axis_codes": axis_codes,
    }


def validate_fmri_nifti(image) -> None:
    """
    Valida que una imagen NIfTI tenga forma razonable para fMRI.

    En una primera aproximación, esperamos un volumen 4D:
    (X, Y, Z, T)

    Si no lo es, lanzamos una excepción.
    """
    metadata = get_nifti_metadata(image)

    if metadata["ndim"] != 4:
        raise MRIFormatError(
            f"Se esperaba un fMRI 4D, pero se recibió una imagen con shape {metadata['shape']}"
        )