from pathlib import Path
import pickle
import numpy as np
from eegraph.graph import Graph
from mrigraph.config import AtlasConfig

RUTA_BASE = Path("./z")
SALIDA_BASE = Path("./z/graph")
SALIDA_BASE.mkdir(parents=True, exist_ok=True)

atlas_config = AtlasConfig(atlas_name="aal")
# Alternativas válidas:
# atlas_config = AtlasConfig(atlas_name="schaefer_100")
# atlas_config = AtlasConfig(atlas_name="schaefer_200")
# atlas_config = AtlasConfig(atlas_name="schaefer_400")

for ruta_fmri in sorted(RUTA_BASE.rglob("*_bold.nii.gz")):
    if SALIDA_BASE in ruta_fmri.parents:
        continue

    nombre = ruta_fmri.name.split("_", 1)[0]
    salida = SALIDA_BASE / nombre
    salida.mkdir(parents=True, exist_ok=True)

    print("Procesando:", ruta_fmri)

    try:
        G = Graph()
        G.load_data(
            path=str(ruta_fmri),
            modality="fmri",
            # auxiliary_paths=["./datos/confounds.csv", "./datos/events.tsv"]
        )

        print("Modalidad:", G.modality)
        print("Metadata inicial disponible:", list(G.metadata.keys()))

        grafo_fmri, connectivity_matrix = G.modelate(
            window_size=1.0,
            connectivity="pearson",
            threshold=0.7,
            atlas_config=atlas_config,
        )

        print("Shape de la matriz de conectividad:", np.array(connectivity_matrix).shape)
        print("Número de nodos:", grafo_fmri.number_of_nodes())
        print("Número de aristas:", grafo_fmri.number_of_edges())

        ruta_grafo = salida / "grafo_fmri.pkl"
        with open(ruta_grafo, "wb") as f:
            pickle.dump(grafo_fmri, f, protocol=pickle.HIGHEST_PROTOCOL)

        ruta_matriz = salida / "connectivity_matrix_fmri.npz"
        np.savez(ruta_matriz, connectivity_matrix=np.array(connectivity_matrix))

        print("Grafo guardado en:", ruta_grafo)
        print("Matriz de conectividad guardada en:", ruta_matriz)

        transform_bundle = G.metadata["transform_bundle"]
        connectivity_bundle = G.metadata["connectivity_bundle"]

        print("Atlas usado:", transform_bundle.atlas_name)
        print("Atlas resampleado:", transform_bundle.atlas_resampled)
        print("Espacio de centroides:", transform_bundle.centroid_coordinate_space)
        print("Número de ROIs:", transform_bundle.transform_metadata["num_rois"])
        print("Primeras labels ROI:", transform_bundle.roi_labels[:10])

        G.visualize_html(
            grafo_fmri,
            salida / "fmri_ejemplo",
            auto_open=False,
        )

        print("Archivo generado:", salida / "fmri_ejemplo_plot.html")
    except Exception as e:
        print("[ERROR] Fallo procesando:", ruta_fmri)
        print("        Detalle:", e)