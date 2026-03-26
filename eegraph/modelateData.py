from .strategy import *

#Class that uses the Strategy Abstract class
class ModelData: 
    def __init__(self, data, ch_names, strategy: Strategy):
        self.raw_data = data.get_data()
        self.ch_names = ch_names
        self.num_channels = data.info['nchan']
        self.sample_rate = data.info['sfreq']
        self.sample_duration = data.times.max()
        self.sample_length = self.sample_rate * self.sample_duration
        self._strategy = strategy
        self.threshold = self._strategy.threshold
        
    def connectivity_workflow(self, bands, window_size, threshold):
        #If the user assigns a new threshold
        if(threshold) is not None:
            self.threshold = threshold
            
        self.connectivity_matrix = self._strategy.calculate_connectivity_workflow(self, bands, window_size)
        print('\nThreshold:', self.threshold)
        
        out = self._strategy.make_graph_workflow(self)
        if(type(out) is tuple):
            self.connectivity_graphs = out[0]
            self.connectivity_matrix = out[1]
        else:
            self.connectivity_graphs = out

        return self.connectivity_graphs, self.connectivity_matrix
        
class ModelMRIData:
    """
    Clase añadida para integrar el modelado de fMRI dentro de
    eegraph.modelateData sin modificar la clase original ModelData de EEG.

    Esta clase NO trabaja como EEG.
    En MRI, el modelado completo necesita recorrer varias fases:

    1. preprocess
    2. denoise
    3. transform a ROI
    4. conectividad

    Así conseguimos que, desde Graph, la experiencia siga siendo simple:
    - load_data(...)
    - modelate(...)
    """

    def __init__(
        self,
        data,
        ch_names=None,
        connectivity="pearson_correlation",
        atlas_data=None,
        atlas_config=None,
        preprocess_config=None,
        denoise_config=None,
        connectivity_config=None,
    ):
        """
        Parámetros:
        - data: aquí debe llegar el MRIInputBundle
        - ch_names: se acepta por compatibilidad con la API EEG, aunque en MRI
          los nombres reales aparecerán tras la transformación como roi_labels
        - connectivity: método de conectividad
        - atlas_data: atlas 3D opcional
        - atlas_config: configuración del atlas
        - preprocess_config: configuración de preprocess
        - denoise_config: configuración de denoise
        - connectivity_config: configuración de conectividad
        """
        self.data = data
        self.ch_names = ch_names if ch_names is not None else []
        self.connectivity = connectivity

        self.atlas_data = atlas_data
        self.atlas_config = atlas_config
        self.preprocess_config = preprocess_config
        self.denoise_config = denoise_config
        self.connectivity_config = connectivity_config

        # Guardamos bundles intermedios por si luego queremos depurar
        self.input_bundle = None
        self.preprocess_bundle = None
        self.denoise_bundle = None
        self.transform_bundle = None
        self.connectivity_bundle = None

    def _validate_input_bundle(self):
        """
        Comprueba que lo recibido parece un MRIInputBundle válido.
        """
        if self.data is None:
            raise ValueError(
                "ModelMRIData no ha recibido datos de entrada."
            )

        if not hasattr(self.data, "fmri_image"):
            raise ValueError(
                "ModelMRIData espera un MRIInputBundle con atributo 'fmri_image'."
            )

        if not hasattr(self.data, "fmri_metadata"):
            raise ValueError(
                "ModelMRIData espera un MRIInputBundle con atributo 'fmri_metadata'."
            )

    def _normalize_connectivity_name(self, connectivity):
        """
        Normaliza nombres frecuentes de conectividad para MRI.

        Esto ayuda a tolerar pequeñas diferencias de nombre entre
        lo que se use en EEGraph y lo que espera MRIGraph.
        """
        if connectivity is None:
            return "pearson_correlation"

        if not isinstance(connectivity, str):
            connectivity = str(connectivity)

        normalized = connectivity.strip().lower()

        aliases = {
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

        return aliases.get(normalized, normalized)

    def _build_connectivity_config(self, window_size, threshold):
        """
        Construye la configuración de conectividad MRI a partir de los
        parámetros que recibe la API de EEGraph.
        """
        from mrigraph.config import ConnectivityConfig

        if self.connectivity_config is not None:
            config = self.connectivity_config
        else:
            config = ConnectivityConfig()

        config.method = self._normalize_connectivity_name(self.connectivity)
        config.window_size = window_size
        config.threshold = threshold

        return config

    def _build_graph_from_matrix(self, connectivity_matrix, roi_labels=None):
        """
        Convierte la matriz ROI x ROI en un grafo NetworkX.

        Esto permite devolver una salida parecida a la del flujo EEG:
        - connectivity_matrix
        - G
        """
        import networkx as nx
        import numpy as np

        matrix = np.array(connectivity_matrix, copy=True)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "La matriz de conectividad debe ser cuadrada para construir el grafo."
            )
        
        es_simetrica = np.allclose(matrix, matrix.T, atol=1e-5)

        if es_simetrica:
            G = nx.from_numpy_array(matrix)
        else:
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

        if roi_labels is not None and len(roi_labels) == matrix.shape[0]:
            mapping = {indice: etiqueta for indice, etiqueta in enumerate(roi_labels)}
            G = nx.relabel_nodes(G, mapping)

        return G

    def connectivity_workflow(
        self,
        bands=[None],
        window_size=1.0,
        threshold=None,
        atlas_data=None,
        atlas_config=None,
        preprocess_config=None,
        denoise_config=None,
        connectivity_config=None,
    ):
        """
        Ejecuta el workflow completo MRI.

        Importante:
        - 'bands' se acepta por compatibilidad con EEG,
          pero en MRI no se usa en esta versión.
        - Devuelve (connectivity_matrix, G), igual que espera graph.py.
        """
        self._validate_input_bundle()

        from mrigraph.preprocess import PreprocessMRIData
        from mrigraph.denoise import DenoiseMRIData
        from mrigraph.transform import TransformMRIData
        from mrigraph.modelateData import ModelMRIConnectivityData

        self.input_bundle = self.data

        # ------------------------------------------------------
        # 1) PREPROCESS
        # ------------------------------------------------------
        resolved_preprocess_config = (
            preprocess_config if preprocess_config is not None else self.preprocess_config
        )

        preprocessor = PreprocessMRIData(
            input_bundle=self.input_bundle,
            config=resolved_preprocess_config
        )
        self.preprocess_bundle = preprocessor.run()

        # ------------------------------------------------------
        # 2) DENOISE
        # ------------------------------------------------------
        resolved_denoise_config = (
            denoise_config if denoise_config is not None else self.denoise_config
        )

        denoiser = DenoiseMRIData(
            preprocess_bundle=self.preprocess_bundle,
            config=resolved_denoise_config
        )
        self.denoise_bundle = denoiser.run()

        # ------------------------------------------------------
        # 3) TRANSFORM
        # ------------------------------------------------------
        resolved_atlas_data = atlas_data if atlas_data is not None else self.atlas_data
        resolved_atlas_config = atlas_config if atlas_config is not None else self.atlas_config

        transformer = TransformMRIData(
            denoise_bundle=self.denoise_bundle,
            atlas_data=resolved_atlas_data,
            config=resolved_atlas_config
        )
        self.transform_bundle = transformer.run()

        # ------------------------------------------------------
        # 4) CONNECTIVITY
        # ------------------------------------------------------
        resolved_connectivity_config = (
            connectivity_config
            if connectivity_config is not None
            else self._build_connectivity_config(window_size, threshold)
        )

        # Si nos pasan una config externa, aun así actualizamos los tres
        # campos clave para mantener coherencia con la API de EEGraph.
        resolved_connectivity_config.method = self._normalize_connectivity_name(self.connectivity)
        resolved_connectivity_config.window_size = window_size
        resolved_connectivity_config.threshold = threshold

        connectivity_modeler = ModelMRIConnectivityData(
            transform_bundle=self.transform_bundle,
            config=resolved_connectivity_config
        )
        self.connectivity_bundle = connectivity_modeler.run()

        connectivity_matrix = self.connectivity_bundle.connectivity_matrix
        roi_labels = self.connectivity_bundle.roi_labels

        # Actualizamos ch_names con las etiquetas ROI reales
        if roi_labels:
            self.ch_names = list(roi_labels)

        G = self._build_graph_from_matrix(
            connectivity_matrix=connectivity_matrix,
            roi_labels=roi_labels
        )

        return connectivity_matrix, G

    def display_info(self, bundle=None):
        """
        Muestra información resumida del resultado final de conectividad MRI.
        """
        if bundle is None:
            bundle = self.connectivity_bundle

        if bundle is None:
            raise ValueError(
                "No hay ningún MRIConnectivityBundle disponible. "
                "Llama antes a connectivity_workflow()."
            )

        print("\n[EEGraph] Conectividad MRI calculada")
        print(f"fMRI original: {bundle.fmri_path}")
        print(f"Método: {bundle.connectivity_metadata.get('method')}")
        print(f"Shape ROI x tiempo: {bundle.connectivity_metadata.get('roi_time_series_shape')}")
        print(
            f"Shape matriz conectividad: "
            f"{bundle.connectivity_metadata.get('connectivity_matrix_shape')}"
        )
        print(
            f"Matriz simétrica: "
            f"{bundle.connectivity_metadata.get('matrix_is_symmetric')}"
        )
        print(
            f"Diagonal a 1: "
            f"{bundle.connectivity_metadata.get('diagonal_all_ones')}"
        )
        print(
            f"Conectividad media (sin diagonal): "
            f"{bundle.connectivity_metadata.get('mean_connectivity')}"
        )

        if bundle.applied_steps:
            print("Pasos aplicados:")
            for step in bundle.applied_steps:
                print(f" - {step}")

        if bundle.pending_steps:
            print("Pasos pendientes de implementación:")
            for step in bundle.pending_steps:
                print(f" - {step}")