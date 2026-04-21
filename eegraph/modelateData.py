from .strategy import *
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    Clase añadida para integrar el modelado de fMRI dentro de eegraph.modelateData sin modificar la clase original ModelData de EEG.

    Esta clase NO trabaja como EEG.
    En MRI, el modelado completo necesita recorrer varias fases:

    1. preprocess
    2. denoise
    3. transform a ROI
    4. conectividad

    Así conseguimos que, desde Graph, la experiencia siga siendo simple para el usuario:
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
    
    def _infer_lr_partner(self, label):
        """
        Intenta deducir la pareja izquierda/derecha de una ROI a partir del nombre.

        Casos cubiertos:
        - AAL: Precentral_L <-> Precentral_R
        - variantes Left/Right
        - si no encuentra patrón, devuelve None
        """
        label = str(label)

        patterns = [
            (r"^(.*)_L$", r"\1_R"),
            (r"^(.*)_R$", r"\1_L"),
            (r"^(.*)_Left$", r"\1_Right"),
            (r"^(.*)_Right$", r"\1_Left"),
            (r"^(.*)Left$", r"\1Right"),
            (r"^(.*)Right$", r"\1Left"),
        ]

        for pattern, replacement in patterns:
            if re.match(pattern, label):
                return re.sub(pattern, replacement, label)

        return None

    def _build_symmetric_axial_layout(self, roi_centroids_3d):
        """
        Construye una proyección axial simétrica SOLO para visualización.

        Importante:
        - NO modifica los centroides 3D reales del atlas.
        - Devuelve:
            * pos2d: posiciones (x, y) simétricas para dibujar
            * depth: profundidad z simétrica para colorear
        - Funciona bien con AAL por nombre (_L/_R).
        - Para atlas sin nombres anatómicos reales (por ejemplo ROI_1...ROI_n), hace un emparejamiento automático izquierda/derecha usando cercanía en (y, z).
        """
        if not roi_centroids_3d:
            return {}, {}

        raw = {
            node: (
                float(coords[0]),
                float(coords[1]),
                float(coords[2]),
            )
            for node, coords in roi_centroids_3d.items()
        }

        # Layout base: si algo no se puede emparejar, se queda como está
        pos2d = {node: (coords[0], coords[1]) for node, coords in raw.items()}
        depth = {node: coords[2] for node, coords in raw.items()}

        processed = set()

        # ------------------------------------------------------
        # 1) Emparejamiento explícito por nombre (_L / _R)
        # ------------------------------------------------------
        for node in raw:
            partner = self._infer_lr_partner(node)

            if partner is None or partner not in raw:
                continue

            if node in processed or partner in processed:
                continue

            left, right = node, partner

            # Aseguramos que "left" sea el que tenga x menor
            if raw[left][0] > raw[right][0]:
                left, right = right, left

            x_left, y_left, z_left = raw[left]
            x_right, y_right, z_right = raw[right]

            mirrored_x = 0.5 * (abs(x_left) + abs(x_right))
            avg_y = 0.5 * (y_left + y_right)
            avg_z = 0.5 * (z_left + z_right)

            pos2d[left] = (-mirrored_x, avg_y)
            pos2d[right] = (mirrored_x, avg_y)

            depth[left] = avg_z
            depth[right] = avg_z

            processed.add(left)
            processed.add(right)

        # ------------------------------------------------------
        # 2) Fallback automático para atlas sin nombres L/R
        # ------------------------------------------------------
        remaining_left = [
            node for node, (x, _, _) in raw.items()
            if x < 0 and node not in processed
        ]
        remaining_right = [
            node for node, (x, _, _) in raw.items()
            if x > 0 and node not in processed
        ]

        if remaining_left and remaining_right:
            left_yz = np.array(
                [[raw[node][1], raw[node][2]] for node in remaining_left],
                dtype=float
            )
            right_yz = np.array(
                [[raw[node][1], raw[node][2]] for node in remaining_right],
                dtype=float
            )

            # Emparejamos cada ROI izquierda con la derecha más parecida en (y, z)
            cost = np.sum((left_yz[:, None, :] - right_yz[None, :, :]) ** 2, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)

            for i, j in zip(row_ind, col_ind):
                left = remaining_left[i]
                right = remaining_right[j]

                x_left, y_left, z_left = raw[left]
                x_right, y_right, z_right = raw[right]

                mirrored_x = 0.5 * (abs(x_left) + abs(x_right))
                avg_y = 0.5 * (y_left + y_right)
                avg_z = 0.5 * (z_left + z_right)

                pos2d[left] = (-mirrored_x, avg_y)
                pos2d[right] = (mirrored_x, avg_y)

                depth[left] = avg_z
                depth[right] = avg_z

        return pos2d, depth

    def _build_graph_from_matrix(self, connectivity_matrix, roi_labels=None, roi_centroids_3d=None, centroid_coordinate_space="world", projection="coronal"):
        """
        Convierte la matriz ROI x ROI en un grafo NetworkX.

        Para fMRI:
        - guardamos pos3d completa,
        - proyectamos a 2D con una vista anatómica fija,
        - y guardamos una profundidad separada para colorear los nodos.

        En la vista axial aplicamos una simetrización SOLO visual para que
        los nodos izquierda/derecha queden espejados respecto a la línea media.
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

        G.graph["modality"] = "fmri"
        G.graph["coordinate_space"] = centroid_coordinate_space
        G.graph["projection"] = projection

        if roi_centroids_3d:
            pos2d = {}
            pos3d = {}
            depth = {}

            # Guardamos SIEMPRE la posición 3D real
            for node in G.nodes():
                centroid = roi_centroids_3d.get(node)
                if centroid is None:
                    continue

                x, y, z = centroid
                pos3d[node] = (float(x), float(y), float(z))

            # --------------------------------------------------
            # Proyección 2D
            # --------------------------------------------------
            if projection == "axial":
                # Vista superior: plano X-Y
                # Aquí aplicamos la simetría solo para visualización.
                axial_centroids = {
                    node: pos3d[node]
                    for node in G.nodes()
                    if node in pos3d
                }

                pos2d, depth = self._build_symmetric_axial_layout(axial_centroids)

                G.graph["depth_axis"] = "z"
                G.graph["depth_legend"] = "Profundidad Z (inferior ↔ superior)"

            elif projection == "coronal":
                for node in G.nodes():
                    if node not in pos3d:
                        continue

                    x, y, z = pos3d[node]
                    pos2d[node] = (x, z)
                    depth[node] = y

                G.graph["depth_axis"] = "y"
                G.graph["depth_legend"] = "Profundidad Y (posterior ↔ anterior)"

            elif projection == "sagittal":
                for node in G.nodes():
                    if node not in pos3d:
                        continue

                    x, y, z = pos3d[node]
                    pos2d[node] = (y, z)
                    depth[node] = x

                G.graph["depth_axis"] = "x"
                G.graph["depth_legend"] = "Profundidad X (izquierda ↔ derecha)"

            else:
                raise ValueError(f"Proyección MRI no soportada: {projection}")

            if pos3d:
                nx.set_node_attributes(G, pos3d, "pos3d")
            if pos2d:
                nx.set_node_attributes(G, pos2d, "pos")
            if depth:
                nx.set_node_attributes(G, depth, "depth")

        for u, v, data in G.edges(data=True):
            peso = float(data.get("weight", 1.0))
            grosor = max(0.5, abs(peso) * 6)
            data["thickness"] = grosor

        return G

    def connectivity_workflow(self, bands=[None], window_size=1.0, threshold=None, atlas_data=None, atlas_config=None, preprocess_config=None, denoise_config=None, connectivity_config=None):
        """
        Ejecuta el workflow completo MRI.

        Importante:
        - 'bands' se acepta por compatibilidad con EEG, pero en MRI no se usa en esta versión.
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
        np.fill_diagonal(connectivity_matrix, 0.0) # Eliminamos autoconexiones para visualización y grafo

        roi_labels = self.connectivity_bundle.roi_labels

        # Actualizamos ch_names con las etiquetas ROI reales
        if roi_labels:
            self.ch_names = list(roi_labels)

        roi_centroids_3d = getattr(self.transform_bundle, "roi_centroids_3d", None)

        centroid_coordinate_space = getattr(
            self.transform_bundle,
            "centroid_coordinate_space",
            None
        )

        if centroid_coordinate_space is None:
            centroid_coordinate_space = self.transform_bundle.transform_metadata.get(
                "centroid_coordinate_space",
                "world"
            )

        G = self._build_graph_from_matrix(
            connectivity_matrix=connectivity_matrix,
            roi_labels=roi_labels,
            roi_centroids_3d=roi_centroids_3d,
            centroid_coordinate_space=centroid_coordinate_space,
            projection="axial",
            #projection="coronal"
        )

        return G, connectivity_matrix

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