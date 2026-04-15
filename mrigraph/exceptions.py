class MRIGraphError(Exception):
    """
    Excepción base de la librería MRIGraph.
    Todas las excepciones propias del proyecto heredarán de esta.
    """
    pass


class MRIInputError(MRIGraphError):
    """
    Error relacionado con la entrada de datos.
    Por ejemplo:
    - extensión no soportada,
    - archivo inexistente,
    - combinación de archivos inválida,
    - etc.
    """
    pass


class MRIFormatError(MRIGraphError):
    """
    Error relacionado con el formato interno de los datos.
    Por ejemplo:
    - una matriz que no es 2D,
    - una forma inesperada,
    - un NIfTI que no tiene las dimensiones esperadas,
    - etc.
    """
    pass


class AtlasError(MRIGraphError):
    """
    Error relacionado con el atlas.
    Por ejemplo:
    - atlas no soportado,
    - falta el archivo del atlas,
    - configuración incompatible,
    - etc.
    """
    pass


class PreprocessingError(MRIGraphError):
    """
    Error relacionado con la fase de preprocesado.
    Lo definimos ya aunque todavía no implementemos esa parte,
    para dejar la arquitectura lista desde el principio.
    """
    pass


class DenoisingError(MRIGraphError):
    """
    Error relacionado con la fase de denoising.
    Igual que con preprocessing, lo dejamos preparado ya.
    """
    pass


class TransformationError(MRIGraphError):
    """
    Error relacionado con la transformación de datos:
    por ejemplo, fallos al mapear el atlas o al extraer ROIs.
    """
    pass


class ConnectivityError(MRIGraphError):
    """
    Error relacionado con el cálculo de conectividad.
    Por ejemplo:
    - no hay series ROI disponibles,
    - dimensiones incorrectas,
    - método de conectividad no soportado,
    - etc.
    """
    pass