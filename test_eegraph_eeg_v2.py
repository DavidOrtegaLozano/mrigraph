from pathlib import Path
import numpy as np
import mne

from eegraph.graph import Graph


# ============================================================
# TEST V2 DE EEG SINTÉTICO PARA EEGRAPH
#
# Qué valida este script:
# 1. crea un EEG sintético pequeño con canales estándar
# 2. lo guarda en un archivo .edf
# 3. lo carga usando la API pública de EEGraph
# 4. calcula conectividad con Pearson
# 5. visualiza el primer grafo en HTML y PNG
# ============================================================


# ------------------------------------------------------------
# BLOQUE 1
# Carpeta de salida
# ------------------------------------------------------------
carpeta_test = Path("tmp_eegraph_eeg_test_v2")
carpeta_test.mkdir(exist_ok=True)

ruta_edf = carpeta_test / "eeg_sintetico.edf"


# ------------------------------------------------------------
# BLOQUE 2
# Crear un EEG sintético pequeño
# ------------------------------------------------------------
np.random.seed(42)

sfreq = 128.0          # frecuencia de muestreo
duracion = 10.0        # segundos
n_muestras = int(sfreq * duracion)

ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
n_canales = len(ch_names)

tiempo = np.arange(n_muestras) / sfreq

# Señales sintéticas:
# - base sinusoidal parecida en algunos canales
# - pequeñas variaciones y ruido para que no sean idénticas
datos = []

frecuencias_base = [10, 10, 12, 12, 8, 8, 6, 6]
fases = [0.0, 0.2, 0.5, 0.7, 1.0, 1.1, 1.4, 1.6]

for i in range(n_canales):
    senal = (
        20e-6 * np.sin(2 * np.pi * frecuencias_base[i] * tiempo + fases[i]) +
        8e-6 * np.sin(2 * np.pi * 2 * tiempo) +
        2e-6 * np.random.randn(n_muestras)
    )
    datos.append(senal)

datos = np.array(datos)

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=["eeg"] * n_canales
)

raw = mne.io.RawArray(datos, info)


# ------------------------------------------------------------
# BLOQUE 3
# Exportar a EDF
# ------------------------------------------------------------
# Intentamos primero con el método de Raw; si no existe, usamos la función de MNE.
try:
    raw.export(str(ruta_edf), fmt="edf", overwrite=True)
except AttributeError:
    mne.export.export_raw(str(ruta_edf), raw, fmt="edf", overwrite=True)

print("\n[TEST] Archivo EDF creado en:")
print(ruta_edf)


# ------------------------------------------------------------
# BLOQUE 4
# Cargar con EEGraph
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CARGA DE EEG DESDE EEGRAPH")
print("=" * 70)

G = Graph()

G.load_data(path=str(ruta_edf), modality="eeg")

print("\n[TEST] Modalidad en Graph:")
print(getattr(G, "modality", "eeg"))

print("\n[TEST] ch_names cargados:")
print(G.ch_names)


# ------------------------------------------------------------
# BLOQUE 5
# Modelar conectividad
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("MODELADO DE CONECTIVIDAD EEG DESDE EEGRAPH")
print("=" * 70)

# OJO:
# Con el flujo actual de EEGraph, aunque en graph.py los nombres internos estén
# algo confusos, la salida efectiva se usa como:
# graphs, connectivity_matrix = G.modelate(...)
# Por eso, aunque en graph.py se haga "return connectivity_matrix, G", aquí lo recibimos al revés para mantener la compatibilidad con el resto del código que espera esa salida.
# es decir, el orden de retorno en graph.py es (connectivity_matrix, G) pero aquí lo recibimos como (G, connectivity_matrix), y no falla porque el código de test se adapta a esa salida. Es importante no cambiar el orden de retorno en graph.py para no romper la compatibilidad con este test y otros que puedan esperar ese formato.
# si 
graphs, connectivity_matrix = G.modelate(
    window_size=2,
    connectivity="pearson_correlation"
)

print("\n[TEST] Tipo de 'graphs':")
print(type(graphs))

print("\n[TEST] Shape de la matriz de conectividad:")
print(np.array(connectivity_matrix).shape)

print("\n[TEST] Matriz de conectividad:")
print(connectivity_matrix)


# ------------------------------------------------------------
# BLOQUE 6
# Elegir un grafo para visualizar
# ------------------------------------------------------------
# Si salen varios grafos (uno por ventana), usamos el primero.
def extraer_primer_grafo(objeto):
    """
    Busca recursivamente el primer grafo real dentro de la estructura
    que devuelve EEGraph.

    Acepta:
    - grafo directo
    - lista/tupla de grafos
    - diccionario de grafos
    - estructuras anidadas
    """
    # Caso 1: ya es un grafo de NetworkX o compatible
    if hasattr(objeto, "is_directed") and callable(objeto.is_directed):
        return objeto

    # Caso 2: lista o tupla
    if isinstance(objeto, (list, tuple)):
        for elemento in objeto:
            encontrado = extraer_primer_grafo(elemento)
            if encontrado is not None:
                return encontrado

    # Caso 3: diccionario
    if isinstance(objeto, dict):
        for _, valor in objeto.items():
            encontrado = extraer_primer_grafo(valor)
            if encontrado is not None:
                return encontrado

    return None


print("\n[TEST] Tipo de 'graphs':")
print(type(graphs))

if isinstance(graphs, dict):
    print("\n[TEST] Claves principales de 'graphs':")
    print(list(graphs.keys()))

grafo_a_mostrar = extraer_primer_grafo(graphs)

if grafo_a_mostrar is None:
    raise ValueError(
        "No se ha podido extraer un grafo válido desde la salida de EEGraph."
    )

print("\n[TEST] Tipo del grafo extraído para visualizar:")
print(type(grafo_a_mostrar))


# ------------------------------------------------------------
# BLOQUE 7
# Visualización
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("VISUALIZACIÓN")
print("=" * 70)

# HTML
# Si quieres que se abra automáticamente en el navegador, usa auto_open=True
G.visualize_html(grafo_a_mostrar, carpeta_test / "eeg_graph_test", auto_open=True)

# PNG
# G.visualize_png(grafo_a_mostrar, carpeta_test / "eeg_graph_test")

print("\n[TEST] Archivos generados:")
print(carpeta_test / "eeg_graph_test_plot.html")
print(carpeta_test / "eeg_graph_test.png")


# ------------------------------------------------------------
# BLOQUE 8
# Comprobaciones rápidas del grafo
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPROBACIONES DEL GRAFO")
print("=" * 70)

print("\n[TEST] Número de nodos:")
print(grafo_a_mostrar.number_of_nodes())

print("\n[TEST] Número de aristas:")
print(grafo_a_mostrar.number_of_edges())

print("\n[TEST] Nombres de nodos:")
print(list(grafo_a_mostrar.nodes()))


# ------------------------------------------------------------
# BLOQUE 9
# Final
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST EEG FINALIZADO")
print("=" * 70)
print("Si todo ha ido bien, EEGraph sigue funcionando con EEG y además visualiza.")

