from .importData import *
from .modelateData import *
from .tools import *

class Graph:
    
    def __init__(self):
        self.data = None
        self.ch_names = None
        self.modality = None
        self.metadata = {}
        
    def load_data(self, path, exclude=[None], electrode_montage_path=None, modality=None, **kwargs):
        """
        Carga los datos de entrada.

        PARTE EEG (comportamiento original):
        - Si modality='eeg', se ejecuta exactamente el flujo antiguo.

        PARTE fMRI (nuevo comportamiento añadido):
        - Si modality='fmri', delega la carga a una clase específica de mrigraph
          para no contaminar el flujo original de EEG.

        Parámetros:
        - path: ruta al fichero de entrada.
        - exclude: canales a excluir (solo EEG).
        - electrode_montage_path: ruta al montaje de electrodos (solo EEG).
        - modality: 'eeg' por defecto, o 'fmri' para usar el flujo nuevo.
        - **kwargs: parámetros adicionales para fMRI , por ejemplo:
            - auxiliary_paths
            - config
        """
        if modality == 'mri':
            modality = 'fmri' # Normalizamos el nombre de la modalidad para evitar confusiones

        self.modality = modality # Guardamos la modalidad para usarla en otros métodos

        # EEG (comportamiento original)
        if modality == 'eeg':
            input_data = InputData(path, exclude)
            self.data = input_data.load()
            
            self.ch_names=self.data.ch_names
            if(electrode_montage_path):
                self.ch_names=input_data.set_montage(electrode_montage_path)
            
            input_data.display_info(self.ch_names)

            # En EEG no necesitamos metadata adicional
            self.metadata = {}

        # fMRI (nuevo comportamiento añadido)
        elif modality == 'fmri':
            input_data = InputfMRIData(path, **kwargs)

            # En MRI, la carga devuelve el MRIInputBundle
            self.data = input_data.load()

            # En esta fase todavía no existen roi_labels reales, por eso ch_names queda vacío hasta la transformación.
            self.ch_names = input_data.get_ch_names()

            # Guardamos metadatos útiles del bundle cargado
            self.metadata = input_data.get_metadata(self.data)

            # Mostramos información del bundle cargado
            input_data.display_info(self.data)
        
        # Si la modalidad no es ni EEG ni fMRI, lanzamos un error claro
        else:
            raise ValueError(
                "Unsupported modality. Use 'eeg' or 'fmri'."
            )

    def modelate(self, window_size, connectivity, bands=[None], threshold=None, **kwargs):
        """
        Modela los datos como grafos.

        PARTE EEG:
        - Se conserva exactamente la lógica existente.

        PARTE fMRI:
        - Se usa ModelMRIData definido en eegraph.modelateData.
        - Esta clase orquesta internamente:
            1. preprocess
            2. denoise
            3. transform
            4. conectividad

        Parámetros:
        - window_size: tamaño de ventana
        - connectivity: medida de conectividad
        - bands: se mantiene por compatibilidad con EEG
        - threshold: umbral opcional
        - **kwargs: parámetros adicionales para MRI, por ejemplo:
            - atlas_data
            - atlas_config
            - preprocess_config
            - denoise_config
            - connectivity_config
        """
        print('\033[1m' + 'Model Data.' + '\033[0m')

        # EEG (comportamiento original)
        if self.modality == 'eeg':
            print(search(connectivity_measures, connectivity))
            
            model_data = ModelData(self.data, self.ch_names, eval(search(connectivity_measures, connectivity)))  
            G, connectivity_matrix = model_data.connectivity_workflow(bands, window_size, threshold)
            
            return G, connectivity_matrix
        
        # fMRI (nuevo comportamiento añadido)
        elif self.modality == 'fmri':
            model_data = ModelMRIData(data=self.data, ch_names=self.ch_names, connectivity=connectivity, **kwargs)
            connectivity_matrix, G = model_data.connectivity_workflow(bands=bands, window_size=window_size, threshold=threshold, **kwargs)

            # Tras la transformación MRI ya existen roi_labels reales. Si ModelMRIData las ha generado, las guardamos en Graph.
            if hasattr(model_data, 'ch_names') and model_data.ch_names is not None:
                self.ch_names = model_data.ch_names

            # Guardamos también información adicional si más adelante queremos inspeccionar bundles intermedios desde Graph.
            self.metadata["modelate_stage"] = "completed"

            if hasattr(model_data, 'transform_bundle') and model_data.transform_bundle is not None:
                self.metadata["transform_bundle"] = model_data.transform_bundle

            if hasattr(model_data, 'connectivity_bundle') and model_data.connectivity_bundle is not None:
                self.metadata["connectivity_bundle"] = model_data.connectivity_bundle

            return connectivity_matrix, G
        
        # Si la modalidad no es ni EEG ni fMRI, lanzamos un error claro
        else:
            raise ValueError(
                "Unsupported modality. Use 'eeg' or 'fmri'."
            )

    def visualize_html(self, graph, name, auto_open = True):
        fig = draw_graph(graph)
        fig.update_layout(title='', plot_bgcolor='white' ) 
        fig.write_html(str(name) + '_plot.html', auto_open=auto_open, default_height='100%', default_width='100%')
        
        
    def visualize_png(self, graph, name):
        fig = draw_graph(graph)
        fig.update_layout(title='', plot_bgcolor='white' ) 
        fig.write_image(str(name) + '.png', format='png',height=1000,width=1800)
