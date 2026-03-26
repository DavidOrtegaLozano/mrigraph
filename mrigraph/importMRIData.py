"""
Paso 5 del pipeline MRI.
Archivo: importMRIData.py

Qué hace este archivo:
- Es el punto de entrada de datos para MRIGraph.
- Detecta qué tipo de archivo entra.
- Valida extensiones y combinaciones.
- Carga fMRI NIfTI y también archivos auxiliares o derivados.

Importante:
Este archivo NO hace todavía:
- preprocesado,
- denoising,
- atlas,
- ROI,
- conectividad.

Su misión es únicamente dejar los datos bien cargados y organizados
para que el siguiente paso del pipeline los pueda usar.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .config import InputConfig
from .exceptions import MRIInputError
from .io.nifti import load_nifti_file, get_nifti_metadata, validate_fmri_nifti
from .io.tabular import load_tabular_file


@dataclass
class MRIInputBundle:
    """
    Estructura interna que representa los datos de entrada ya cargados.

    Campos principales:
    - fmri_path: ruta al archivo principal del fMRI
    - fmri_image: imagen NIfTI cargada
    - fmri_metadata: metadatos básicos de la imagen
    - auxiliary_files: diccionario con archivos auxiliares detectados/cargados
    """
    fmri_path: Optional[str] = None
    fmri_image: Optional[object] = None
    fmri_metadata: Optional[Dict[str, object]] = None
    auxiliary_files: Dict[str, object] = field(default_factory=dict)


class InputMRIData:
    """
    Clase principal de importación de datos MRI.

    Uso esperado:
    - recibe una ruta principal al fMRI
    - opcionalmente recibe una lista de archivos auxiliares
    - valida y carga todo en una estructura interna coherente
    """

    def __init__(
        self,
        fmri_path: Union[str, Path],
        auxiliary_paths: Optional[List[Union[str, Path]]] = None,
        config: Optional[InputConfig] = None,
    ):
        self.fmri_path = Path(fmri_path)
        self.auxiliary_paths = [Path(p) for p in auxiliary_paths] if auxiliary_paths else []
        self.config = config if config is not None else InputConfig()

    def load(self) -> MRIInputBundle:
        """
        Ejecuta la carga completa de la entrada MRI.

        Devuelve:
        - MRIInputBundle con:
          - imagen fMRI cargada
          - metadatos del fMRI
          - archivos auxiliares cargados
        """
        self._validate_main_input()

        fmri_image = load_nifti_file(self.fmri_path)
        validate_fmri_nifti(fmri_image)
        fmri_metadata = get_nifti_metadata(fmri_image)

        auxiliary_data = self._load_auxiliary_files()

        bundle = MRIInputBundle(
            fmri_path=str(self.fmri_path),
            fmri_image=fmri_image,
            fmri_metadata=fmri_metadata,
            auxiliary_files=auxiliary_data,
        )

        return bundle

    def _validate_main_input(self) -> None:
        """
        Valida la entrada principal del fMRI.
        """
        if not self.fmri_path.exists():
            raise MRIInputError(f"El archivo fMRI no existe: {self.fmri_path}")

        suffix = "".join(self.fmri_path.suffixes).lower()

        if suffix not in self.config.allowed_fmri_extensions:
            raise MRIInputError(
                f"Extensión principal no soportada para fMRI: {self.fmri_path.name}. "
                f"Soportadas: {', '.join(self.config.allowed_fmri_extensions)}"
            )

    def _load_auxiliary_files(self) -> Dict[str, object]:
        """
        Carga los archivos auxiliares o derivados.

        La clave del diccionario será el nombre del archivo.
        Más adelante podremos sofisticar esto y clasificar mejor
        qué tipo de auxiliar es cada uno.
        """
        loaded_aux = {}

        for aux_path in self.auxiliary_paths:
            if not aux_path.exists():
                raise MRIInputError(f"El archivo auxiliar no existe: {aux_path}")

            suffix = aux_path.suffix.lower()

            if suffix not in self.config.allowed_aux_extensions:
                raise MRIInputError(
                    f"Extensión auxiliar no soportada: {aux_path.name}. "
                    f"Soportadas: {', '.join(self.config.allowed_aux_extensions)}"
                )

            # JSON: por ahora lo dejamos como texto crudo cargado.
            # Más adelante podremos parsearlo si necesitamos metadatos concretos.
            if suffix == ".json":
                loaded_aux[aux_path.name] = aux_path.read_text(encoding="utf-8")
                continue

            # CSV / TSV / NPY: carga tabular genérica.
            if suffix in {".csv", ".tsv", ".npy"}:
                loaded_aux[aux_path.name] = load_tabular_file(aux_path)
                continue

        return loaded_aux

    def display_info(self, bundle: MRIInputBundle) -> None:
        """
        Muestra por pantalla información resumida sobre la entrada cargada.

        Esto es útil para depuración y para entender rápidamente
        qué ha detectado la librería.
        """
        print("\n[MRIGraph] Datos cargados correctamente")
        print(f"fMRI principal: {bundle.fmri_path}")

        if bundle.fmri_metadata is not None:
            print(f"Shape fMRI: {bundle.fmri_metadata['shape']}")
            print(f"Número de dimensiones: {bundle.fmri_metadata['ndim']}")

        if bundle.auxiliary_files:
            print("Archivos auxiliares detectados:")
            for file_name in bundle.auxiliary_files:
                print(f" - {file_name}")
        else:
            print("No se han proporcionado archivos auxiliares.")