import mne
import pandas as pd
from .tools import process_channel_names, search_input, input_format
    
class InputData:
    def __init__(self, path, exclude):
        self.path = path
        self.exclude = exclude
        
    def load(self):       
        #Split the path in two parts, left and right of the dot. 
        file_type = self.path.split(".")

        #https://mne.tools/0.17/manual/io.html
        #Check the extension of the file in the input format dictionary, and use the proper MNE method. 
        self.data = eval(search_input(input_format, file_type[-1]))
        
        return self.data
    
    def set_montage(self, electrode_montage_path):
        nodes = process_channel_names(self.data.ch_names)
        df = pd.read_csv(electrode_montage_path, delimiter= "\s+|;|:", engine='python')

        
        positions_number = []
        for column in df:
            counter = 0
            for item in list(df[column]):
                if(str(item) in nodes):
                    counter+=1
                    if(counter > 4):
                        positions_number = list(df[column])
        
        standard_electrodes = ['Cz', 'Pz', 'Oz', 'Fz', 'Nz']
        positions_labels = []
        for column in df:
            for item in list(df[column]):
                if(str(item) in standard_electrodes):
                    positions_labels = list(df[column])
            
        
        new_channel_names= []
        for node in nodes:
            for i in range(len(positions_number)):
                if(str(node) == str(positions_number[i])):
                    new_channel_names.append(positions_labels[i])
                
        return new_channel_names
        

    def display_info(self, ch_names):
        #Extract the raw_data and info with mne methods. 
        self.raw_data = self.data.get_data()
        self.info = self.data.info
        
        #Display information from the data. 
        print('\n\033[1m' + 'EEG Information.')
        print('\033[0m' + "Number of Channels:", self.info['nchan'])
        print("Sample rate:", self.info['sfreq'], "Hz.")
        print("Duration:", round(self.data.times.max(),3), "seconds.")
        print("Channel Names:", ch_names)


class InputfMRIData:
    """
    Clase añadida para integrar la carga de fMRI dentro de eegraph.importData
    sin modificar la clase original InputData de EEG.

    Esta clase actúa como adaptador del loader MRI que ya existe en mrigraph.

    Importante:
    - NO sustituye a InputData.
    - NO rompe el flujo EEG.
    - Solo se usará cuando Graph trabaje con modality=fmri.
    """

    def __init__(self, path, auxiliary_paths=None, config=None):
        """
        Parámetros:
        - path: ruta al fMRI principal
        - auxiliary_paths: lista opcional de rutas auxiliares
        - config: InputConfig opcional de mrigraph
        """
        self.path = path
        self.auxiliary_paths = auxiliary_paths if auxiliary_paths is not None else []
        self.config = config
        self.bundle = None
        self._backend = None

    def load(self):
        """
        Carga el fMRI principal y los auxiliares usando mrigraph.

        Devuelve:
        - MRIInputBundle
        """
        from mrigraph.importMRIData import InputMRIData

        self._backend = InputMRIData(
            fmri_path=self.path,
            auxiliary_paths=self.auxiliary_paths,
            config=self.config
        )

        self.bundle = self._backend.load()
        return self.bundle

    def display_info(self, bundle=None):
        """
        Muestra información de la entrada MRI cargada.

        Si no se pasa bundle, usa el último cargado.
        """
        if bundle is None:
            bundle = self.bundle

        if bundle is None:
            raise ValueError(
                "No hay ningún MRIInputBundle cargado. "
                "Llama antes a load()."
            )

        # Reutilizamos la lógica ya existente en MRIGraph
        if self._backend is not None:
            self._backend.display_info(bundle)
            return

        # Fallback defensivo
        print("\n[EEGraph] Datos fMRI cargados")
        print(f"fMRI principal: {bundle.fmri_path}")

        if bundle.fmri_metadata is not None:
            print(f"Shape fMRI: {bundle.fmri_metadata.get('shape')}")
            print(f"Número de dimensiones: {bundle.fmri_metadata.get('ndim')}")

        if bundle.auxiliary_files:
            print("Archivos auxiliares detectados:")
            for file_name in bundle.auxiliary_files:
                print(f" - {file_name}")
        else:
            print("No se han proporcionado archivos auxiliares.")

    def get_ch_names(self):
        """
        Devuelve los nombres de nodos disponibles en esta fase.

        Importante:
        Tras la importación MRI todavía no existen etiquetas ROI.
        Esas aparecen después de la fase de transformación.
        """
        return []

    def get_metadata(self, bundle=None):
        """
        Expone metadatos útiles del bundle MRI en un formato cómodo para EEGraph.
        """
        if bundle is None:
            bundle = self.bundle

        if bundle is None:
            return {}

        return {
            "fmri_path": bundle.fmri_path,
            "fmri_metadata": bundle.fmri_metadata,
            "auxiliary_files": bundle.auxiliary_files,
            "input_stage": "import",
        }