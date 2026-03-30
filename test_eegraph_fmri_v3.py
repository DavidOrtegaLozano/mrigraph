from pathlib import Path
import json
import numpy as np

from eegraph.graph import Graph
from mrigraph.transform import build_synthetic_atlas
from mrigraph.config import AtlasConfig

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Necesitas instalar nibabel: pip install nibabel")


# ============================================================
# TEST V3 DE INTEGRACIÓN EEGRAPH + fMRI
#
# Qué valida este script:
# 1. crea un fMRI sintético 4D
# 2. crea archivos auxiliares de prueba
# 3. carga los datos usando la API pública de EEGraph
# 4. prueba tres medidas:
#    - pearson
#    - cross_correlation
#    - corrected cross-correlation
# 5. comprueba:
#    - shape de la matriz
#    - simetría
#    - tipo de grafo (Graph / DiGraph)
#    - nodos y labels ROI
# ============================================================


# ------------------------------------------------------------
# BLOQUE 1
# Carpeta temporal de prueba
# ------------------------------------------------------------
carpeta_test = Path("tmp_eegraph_fmri_test_v3")
carpeta_test.mkdir(exist_ok=True)


# # ------------------------------------------------------------
# # BLOQUE 2
# # Crear un fMRI sintético 4D
# # ------------------------------------------------------------
np.random.seed(42)

# datos_fmri = np.random.rand(6, 6, 6, 100).astype(np.float32)
# affine = np.eye(4)
# imagen = nib.Nifti1Image(datos_fmri, affine)

# ruta_fmri = carpeta_test / "fmri_fake.nii.gz"
# nib.save(imagen, str(ruta_fmri))

# ------------------------------------------------------------
# BLOQUE 2
# Usar un fMRI real ya descargado
# ------------------------------------------------------------
ruta_fmri = Path("./tmp_eegraph_fmri_test_v3/a/tfMRI_EMOTION_LR_hp0_clean_rclean_tclean.nii.gz")
imagen = nib.load(str(ruta_fmri))
datos_fmri = imagen.get_fdata()

print("\n[TEST] Shape del archivo real cargado:")
print(datos_fmri.shape)
print("[TEST] Número de dimensiones:")
print(datos_fmri.ndim)

num_timepoints = datos_fmri.shape[3]


# ------------------------------------------------------------
# BLOQUE 3
# Crear auxiliares
# ------------------------------------------------------------
ruta_csv = carpeta_test / "confounds.csv"
np.savetxt(ruta_csv, np.random.rand(num_timepoints, 3), delimiter=",")

ruta_tsv = carpeta_test / "events.tsv"
np.savetxt(ruta_tsv, np.random.rand(5, 2), delimiter="\t")

ruta_json = carpeta_test / "metadata.json"
ruta_json.write_text(
    json.dumps({"subject": "sub-01", "session": "ses-01"}, indent=2),
    encoding="utf-8"
)


# ------------------------------------------------------------
# BLOQUE 4
# Atlas real AAL
# ------------------------------------------------------------


config_atlas = AtlasConfig(atlas_name="AAL")


# # ------------------------------------------------------------
# # BLOQUE 2
# # Crear un fMRI sintético 4D con la misma shape del atlas real
# # ------------------------------------------------------------

# np.random.seed(42)

# datos_fmri = np.random.rand(shape_atlas[0], shape_atlas[1], shape_atlas[2], 100).astype(np.float32) # Esto devuelve 
# affine = np.eye(4)
# imagen = nib.Nifti1Image(datos_fmri, affine)

# ruta_fmri = carpeta_test / "fmri_fake.nii.gz"
# nib.save(imagen, str(ruta_fmri))

from pathlib import Path
import nibabel as nib

carpeta = Path("./tmp_eegraph_fmri_test_v3/a")

for ruta in carpeta.rglob("*.nii.gz"):
    nombre = ruta.name.lower()

    if any(x in nombre for x in [
        "zstat", "sbref", "mean", "mask", "dropout",
        "jacobian", "phase", "bias", "reference"
    ]):
        continue

    try:
        img = nib.load(str(ruta))
        shape = img.shape
        if len(shape) == 4:
            print("VALIDO 4D:", ruta, "shape=", shape)
        else:
            print("NO 4D:", ruta, "shape=", shape)
    except Exception as e:
        print("ERROR:", ruta, e)


# ------------------------------------------------------------
# BLOQUE 5
# Función auxiliar para probar una medida
# ------------------------------------------------------------
def probar_medida(nombre_medida, threshold=0.3):
    print("\n" + "=" * 75)
    print(f"PROBANDO MEDIDA: {nombre_medida}")
    print("=" * 75)

    G = Graph()

    G.load_data(
        path=str(ruta_fmri),
        modality="fmri",
        auxiliary_paths=[
            str(ruta_csv),
            str(ruta_tsv),
            str(ruta_json),
        ]
    )

    grafo, matriz_conectividad = G.modelate(
        window_size=1.0,
        connectivity=nombre_medida,
        threshold=threshold,
        atlas_config=config_atlas
    )

    print("\n[TEST] Shape de la matriz:")
    print(matriz_conectividad.shape)

    print("\n[TEST] Matriz de conectividad:")
    print(matriz_conectividad)

    print("\n[TEST] ¿La matriz es cuadrada?")
    print(matriz_conectividad.shape[0] == matriz_conectividad.shape[1])

    print("\n[TEST] ¿La matriz es simétrica?")
    print(np.allclose(matriz_conectividad, matriz_conectividad.T, atol=1e-5))

    print("\n[TEST] Tipo de grafo:")
    print(type(grafo))

    print("\n[TEST] ¿Es dirigido?")
    print(grafo.is_directed())

    print("\n[TEST] Número de nodos:")
    print(grafo.number_of_nodes())

    print("\n[TEST] Número de aristas:")
    print(grafo.number_of_edges())

    print("\n[TEST] ch_names guardados en Graph:")
    print(G.ch_names)

    print("\n[TEST] Primeros nodos del grafo:")
    print(list(grafo.nodes())[:10])

    print("\n[TEST] Claves de metadata en Graph:")
    print(G.metadata.keys())

    transform_bundle = G.metadata.get("transform_bundle")

    print("\n COMPROBAMOS DATOS:")
    print(transform_bundle.transform_metadata["num_rois"])
    print(transform_bundle.transform_metadata["roi_ids"][:20])
    print(transform_bundle.roi_labels[:20])
    print("--------------------------------\n")

    print("\n[TEST] Ruta del JSON de centroides:")
    print(transform_bundle.centroid_json_path)
    connectivity_bundle = G.metadata.get("connectivity_bundle")

    if transform_bundle is not None:
        print("\n[TEST] Ruta del JSON de centroides:")
        print(transform_bundle.centroid_json_path)

        print("\n[TEST] Número de centroides:")
        print(len(transform_bundle.roi_centroids_3d))

        print("\n[TEST] Primeros 5 centroides:")
        print(list(transform_bundle.roi_centroids_3d.items())[:5])

    if transform_bundle is not None:
        print("\n[TEST] Shape ROI x tiempo:")
        print(transform_bundle.roi_time_series.shape)

    if connectivity_bundle is not None:
        print("\n[TEST] Método registrado en connectivity_metadata:")
        print(connectivity_bundle.connectivity_metadata.get("method"))

        print("\n[TEST] implemented_scope:")
        print(connectivity_bundle.connectivity_metadata.get("implemented_scope"))

        print("\n[TEST] Pasos aplicados:")
        print(connectivity_bundle.applied_steps)

        print("\n[TEST] Pasos pendientes:")
        print(connectivity_bundle.pending_steps)

    import networkx as nx

    pos_attrs = nx.get_node_attributes(grafo, "pos")
    pos3d_attrs = nx.get_node_attributes(grafo, "pos3d")

    print("\n[TEST] Nodos con pos:")
    print(len(pos_attrs))

    print("\n[TEST] Nodos con pos3d:")
    print(len(pos3d_attrs))

    

    # VISUALIZAMOS el grafo para comprobar que se ha creado correctamente (opcional)
    print("\n[TEST] Visualizando el grafo...")
    G.visualize_html(grafo, carpeta_test / f"{nombre_medida}_graph_test", auto_open=True)

    return matriz_conectividad, grafo


# ------------------------------------------------------------
# BLOQUE 6
# Probar Pearson
# ------------------------------------------------------------
matriz_pearson, grafo_pearson = probar_medida("pearson", threshold=0.6)


# ------------------------------------------------------------
# BLOQUE 7
# Probar cross_correlation
# ------------------------------------------------------------
# matriz_cross, grafo_cross = probar_medida("cross_correlation", threshold=0.35)


# ------------------------------------------------------------
# BLOQUE 8
# Probar corrected cross-correlation
# ------------------------------------------------------------
# matriz_corr_cross, grafo_corr_cross = probar_medida("corrected cross-correlation", threshold=0.35)


# ------------------------------------------------------------
# BLOQUE 9
# Resumen final comparativo
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("RESUMEN FINAL")
print("=" * 75)

print("\n[RESUMEN] Pearson simétrica:")
print(np.allclose(matriz_pearson, matriz_pearson.T, atol=1e-5))

# print("\n[RESUMEN] Cross-correlation simétrica:")
# print(np.allclose(matriz_cross, matriz_cross.T, atol=1e-5))

# print("\n[RESUMEN] Corrected cross-correlation simétrica:")
# print(np.allclose(matriz_corr_cross, matriz_corr_cross.T, atol=1e-5))

print("\n[RESUMEN] Pearson dirigido:")
print(grafo_pearson.is_directed())

# print("\n[RESUMEN] Cross-correlation dirigido:")
# print(grafo_cross.is_directed())

# print("\n[RESUMEN] Corrected cross-correlation dirigido:")
# print(grafo_corr_cross.is_directed())

print("\n" + "=" * 75)
print("TEST V3 FINALIZADO")
print("=" * 75)
print("Si todo ha ido bien, la fase 1 queda cerrada.")