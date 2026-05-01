"""Random search con cross validation para GNN en Parkinson control vs patient.

Este script reutiliza el pipeline de carga y modelado del script base, pero:
- reserva por completo ds005892 para test
- hace random search sobre 30 combinaciones
- evalua cada combinacion con StratifiedKFold de 5 folds
- guarda un CSV con todos los resultados agregados por combinacion
- guarda un CSV top10 ordenado por accuracy media y menor desviacion
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from GNN_openneuro_para_test import (
    SEMILLA,
    USAR_ABSOLUTO,
    UMBRAL,
    fijar_semilla,
    crear_tabla_muestras,
    filtrar_matrices_tamano_fijo,
    convertir_tabla_a_lista_grafos,
    normalizar_features_grafos,
    RedGCNBinaria,
    entrenar_una_epoca,
    evaluar_modelo,
    balancear_por_dataset,
)


# ============================================================================
# HIPERPARÁMETROS A BUSCAR
# ============================================================================

LEARNING_RATE_LIST = [0.00003, 0.0001, 0.0003, 0.0005, 0.001]
DROPOUT_LIST = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]
HIDDEN_DIM_LIST = [16, 32, 64]
K_VECINOS_LIST = [15, 20, 25]
EPOCHS_LIST = [50]
PACIENCIA_LIST = [5]
DECAY_LIST = [1e-5, 1e-4, 1e-3]
BATCH_SIZE_LIST = [8, 16, 32]
N_RANDOM_TRIALS = 70
CV_FOLDS = 5


# ============================================================================
# CONFIGURACION FIJA
# ============================================================================

RAIZ_DATOS = Path("./a")
CARPETA_CONTROL = RAIZ_DATOS / "graphs_threshold05" / "parkinson_control"
CARPETA_PATIENT = RAIZ_DATOS / "graphs_threshold05" / "parkinson_patient"

RESULTADOS_CSV = Path("randomsearch_kfold_results.csv")
TOP10_CSV = Path("randomsearch_kfold_top10.csv")
MIN_DELTA_ES = 1e-4


# ============================================================================
# UTILIDADES DE RESUMEN
# ============================================================================


def _resumen_clases(tabla: pd.DataFrame) -> pd.DataFrame:
	conteos = tabla["clase"].value_counts().reindex(["control", "patient"], fill_value=0)
	total = int(conteos.sum())
	return pd.DataFrame(
		{
			"n": conteos.astype(int),
			"proporcion": (conteos / total if total else 0.0).astype(float),
		}
	)


def _imprimir_resumen_clases(titulo: str, tabla: pd.DataFrame) -> None:
	resumen = _resumen_clases(tabla)
	total = int(len(tabla))
	print(titulo)
	print(f"  Total: {total}")
	for clase in ["control", "patient"]:
		n = int(resumen.loc[clase, "n"])
		prop = float(resumen.loc[clase, "proporcion"])
		print(f"  {clase.capitalize():8s} -> n={n} | proporcion={prop:.4f}")
	print()


def _agregar_media_desviacion(resultados: dict[str, list[float]], prefijo: str) -> dict[str, float]:
	salida = {}
	for clave, valores in resultados.items():
		valores_array = np.asarray(valores, dtype=float)
		salida[f"mean_{prefijo}_{clave}"] = float(np.nanmean(valores_array))
		salida[f"std_{prefijo}_{clave}"] = float(np.nanstd(valores_array))
	return salida


def _cargar_grafos_por_k(tabla: pd.DataFrame, k_vecinos: int):
	return convertir_tabla_a_lista_grafos(
		tabla,
		usar_absoluto=USAR_ABSOLUTO,
		k_vecinos=k_vecinos,
	)


def _separar_test_y_cv(tabla: pd.DataFrame):
	tabla = tabla.copy().reset_index(drop=True)
	tabla_test = tabla[tabla["es_test"]].copy().reset_index(drop=True)
	tabla_cv = tabla[~tabla["es_test"]].copy().reset_index(drop=True)
	if tabla_cv.empty:
		raise ValueError("No hay muestras para cross validation tras separar el test ds005892")
	return tabla_cv, tabla_test


def _seleccionar_combinaciones_aleatorias() -> list[tuple]:
	full_grid = list(
		product(
			LEARNING_RATE_LIST,
			DROPOUT_LIST,
			HIDDEN_DIM_LIST,
			K_VECINOS_LIST,
			EPOCHS_LIST,
			PACIENCIA_LIST,
			DECAY_LIST,
			BATCH_SIZE_LIST,
		)
	)

	rng = random.Random(SEMILLA)
	if len(full_grid) <= N_RANDOM_TRIALS:
		return full_grid
	return rng.sample(full_grid, N_RANDOM_TRIALS)


def _crear_modelo(num_features_entrada: int, hidden_dim: int, dropout: float) -> RedGCNBinaria:
	return RedGCNBinaria(
		num_features_entrada=num_features_entrada,
		hidden_dim=hidden_dim,
		dropout=dropout,
	)


def _entrenar_fold(
	train_graphs,
	val_graphs,
	test_graphs,
	device,
	learning_rate: float,
	dropout: float,
	hidden_dim: int,
	epochs: int,
	paciencia: int,
	decay: float,
	batch_size: int,
):
	train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

	num_features_entrada = train_graphs[0].x.shape[1]
	modelo = _crear_modelo(num_features_entrada, hidden_dim, dropout).to(device)

	etiquetas_train = np.array([int(g.y.item()) for g in train_graphs])
	num_positivos = max(int((etiquetas_train == 1).sum()), 1)
	num_negativos = int((etiquetas_train == 0).sum())
	peso_clase_positiva = torch.tensor([num_negativos / num_positivos], dtype=torch.float32).to(device)

	criterio = nn.BCEWithLogitsLoss(pos_weight=peso_clase_positiva)
	optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=decay)

	mejor_val_f1 = float("-inf")
	mejor_val_auc = float("-inf")
	mejor_estado_modelo = None
	epocas_sin_mejora = 0

	for epoca in range(1, epochs + 1):
		metricas_train = entrenar_una_epoca(modelo, train_loader, optimizer, criterio, device, UMBRAL)
		metricas_val = evaluar_modelo(modelo, val_loader, criterio, device, UMBRAL)

		val_f1_actual = metricas_val["f1"]
		val_auc_actual = metricas_val["auc"]
		if np.isnan(val_auc_actual):
			val_auc_actual = float("-inf")

		mejora_f1 = val_f1_actual > mejor_val_f1 + MIN_DELTA_ES
		empate_f1 = abs(val_f1_actual - mejor_val_f1) <= MIN_DELTA_ES
		mejora_auc_en_empate = val_auc_actual > mejor_val_auc + MIN_DELTA_ES

		if mejora_f1 or (empate_f1 and mejora_auc_en_empate):
			mejor_val_f1 = val_f1_actual
			mejor_val_auc = val_auc_actual
			mejor_estado_modelo = deepcopy(modelo.state_dict())
			epocas_sin_mejora = 0
		else:
			epocas_sin_mejora += 1

		if epocas_sin_mejora >= paciencia:
			break

	if mejor_estado_modelo is not None:
		modelo.load_state_dict(mejor_estado_modelo)

	metricas_train_final = evaluar_modelo(modelo, train_loader, criterio, device, UMBRAL)
	metricas_val_final = evaluar_modelo(modelo, val_loader, criterio, device, UMBRAL)
	metricas_test = evaluar_modelo(modelo, test_loader, criterio, device, UMBRAL)

	return {
		"train": metricas_train_final,
		"eval": metricas_val_final,
		"test": metricas_test,
	}


def _evaluar_combinacion(
	combo: tuple,
	tabla_cv: pd.DataFrame,
	tabla_test: pd.DataFrame,
	grafos_cv_por_k: dict[int, list],
	grafos_test_por_k: dict[int, list],
	device,
) -> dict[str, float]:
	learning_rate, dropout, hidden_dim, k_vecinos, epochs, paciencia, decay, batch_size = combo

	skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMILLA)
	y = tabla_cv["label"].to_numpy()
	base_graphs = grafos_cv_por_k[k_vecinos]
	test_graphs_base = grafos_test_por_k[k_vecinos]

	fold_metrics = {
		"train": {k: [] for k in ["loss", "accuracy", "precision", "recall", "f1", "auc"]},
		"eval": {k: [] for k in ["loss", "accuracy", "precision", "recall", "f1", "auc"]},
		"test": {k: [] for k in ["loss", "accuracy", "precision", "recall", "f1", "auc"]},
	}

	for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
		train_graphs = [deepcopy(base_graphs[i]) for i in train_idx]
		val_graphs = [deepcopy(base_graphs[i]) for i in val_idx]
		test_graphs = [deepcopy(g) for g in test_graphs_base]

		train_graphs, val_graphs, test_graphs = normalizar_features_grafos(train_graphs, val_graphs, test_graphs)

		resultado_fold = _entrenar_fold(
			train_graphs=train_graphs,
			val_graphs=val_graphs,
			test_graphs=test_graphs,
			device=device,
			learning_rate=learning_rate,
			dropout=dropout,
			hidden_dim=hidden_dim,
			epochs=epochs,
			paciencia=paciencia,
			decay=decay,
			batch_size=batch_size,
		)

		for bloque in ["train", "eval", "test"]:
			for clave in fold_metrics[bloque].keys():
				fold_metrics[bloque][clave].append(float(resultado_fold[bloque][clave]))

		print(
			f"  Fold {fold_idx}/{CV_FOLDS} | "
			f"TRAIN Acc={resultado_fold['train']['accuracy']:.4f} F1={resultado_fold['train']['f1']:.4f} | "
			f"EVAL Acc={resultado_fold['eval']['accuracy']:.4f} F1={resultado_fold['eval']['f1']:.4f} | "
			f"TEST Acc={resultado_fold['test']['accuracy']:.4f} F1={resultado_fold['test']['f1']:.4f}"
		)

	salida = {
		"learning_rate": learning_rate,
		"dropout": dropout,
		"hidden_dim": hidden_dim,
		"k_vecinos": k_vecinos,
		"epochs": epochs,
		"paciencia": paciencia,
		"decay": decay,
		"batch_size": batch_size,
		"cv_folds": CV_FOLDS,
		"n_train_subjects": int(len(tabla_cv)),
		"n_test_subjects": int(len(tabla_test)),
	}

	salida.update(_agregar_media_desviacion(fold_metrics["train"], "train"))
	salida.update(_agregar_media_desviacion(fold_metrics["eval"], "eval"))
	salida.update(_agregar_media_desviacion(fold_metrics["test"], "test"))

	salida["rank_score"] = salida["mean_test_accuracy"] - salida["std_test_accuracy"]
	salida["rank_order"] = 0.0
	return salida


def _ordenar_resultados(df: pd.DataFrame) -> pd.DataFrame:
	return df.sort_values(
		by=["rank_score", "mean_test_accuracy", "std_test_accuracy"],
		ascending=[False, False, True],
	).reset_index(drop=True)


def main() -> None:
	fijar_semilla(SEMILLA)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print(f"Dispositivo: {device}")
	print(f"CUDA disponible: {torch.cuda.is_available()}")
	print(f"Semilla: {SEMILLA}")
	print()

	tabla_muestras = crear_tabla_muestras()
	tabla_116 = filtrar_matrices_tamano_fijo(tabla_muestras, tamano_objetivo=(116, 116))

	print("Resumen tras filtrado 116x116:")
	_imprimir_resumen_clases("  Total dataset", tabla_116)
	
	print("Aplicando balanceo por dataset...")
	tabla_116 = balancear_por_dataset(tabla_116, semilla=SEMILLA)
	print("Resumen tras balanceo:")
	_imprimir_resumen_clases("  Total balanceado", tabla_116)

	tabla_cv, tabla_test = _separar_test_y_cv(tabla_116)
	print("Resumen para cross validation (sin test ds005892):")
	_imprimir_resumen_clases("  CV base", tabla_cv)
	print("Resumen de test reservado (ds005892):")
	_imprimir_resumen_clases("  Test", tabla_test)

	combinaciones = _seleccionar_combinaciones_aleatorias()
	print(f"Total de combinaciones aleatorias a probar: {len(combinaciones)}")
	print()

	resultados_previos = []
	if RESULTADOS_CSV.exists():
		resultados_previos = pd.read_csv(RESULTADOS_CSV).to_dict("records")
		print(f"Cargados {len(resultados_previos)} resultados previos desde {RESULTADOS_CSV}")
		print()

	completadas = {
		(
			row.get("learning_rate"),
			row.get("dropout"),
			row.get("hidden_dim"),
			row.get("k_vecinos"),
			row.get("epochs"),
			row.get("paciencia"),
			row.get("decay"),
			row.get("batch_size"),
		)
		for row in resultados_previos
	}

	grafos_cv_por_k = {k: _cargar_grafos_por_k(tabla_cv, k) for k in K_VECINOS_LIST}
	grafos_test_por_k = {k: _cargar_grafos_por_k(tabla_test, k) for k in K_VECINOS_LIST}

	resultados = list(resultados_previos)

	for idx, combo in enumerate(combinaciones, start=1):
		if combo in completadas:
			continue

		learning_rate, dropout, hidden_dim, k_vecinos, epochs, paciencia, decay, batch_size = combo
		print(
			f"[{idx}/{len(combinaciones)}] LR={learning_rate}, DO={dropout}, HD={hidden_dim}, "
			f"KV={k_vecinos}, EP={epochs}, PA={paciencia}, DECAY={decay}, BS={batch_size}"
		)

		try:
			resultado = _evaluar_combinacion(
				combo=combo,
				tabla_cv=tabla_cv,
				tabla_test=tabla_test,
				grafos_cv_por_k=grafos_cv_por_k,
				grafos_test_por_k=grafos_test_por_k,
				device=device,
			)

			resultados.append(resultado)
			df_resultados = pd.DataFrame(resultados)
			df_resultados = _ordenar_resultados(df_resultados)
			df_resultados.to_csv(RESULTADOS_CSV, index=False)

			print(
				f"  => mean_test_acc={resultado['mean_test_accuracy']:.4f} | "
				f"std_test_acc={resultado['std_test_accuracy']:.4f} | rank={resultado['rank_score']:.4f}"
			)
			print()

		except Exception as exc:
			print(f"  ✗ Error en la combinacion: {exc}")
			print()

	if not resultados:
		raise RuntimeError("No se han obtenido resultados para ninguna combinacion.")

	df_resultados = pd.DataFrame(resultados)
	df_resultados = _ordenar_resultados(df_resultados)
	df_resultados.to_csv(RESULTADOS_CSV, index=False)

	columnas_top10 = [
		"learning_rate",
		"dropout",
		"hidden_dim",
		"k_vecinos",
		"epochs",
		"paciencia",
		"decay",
		"batch_size",
		"rank_score",
		"mean_test_accuracy",
		"std_test_accuracy",
		"mean_test_f1",
		"std_test_f1",
		"mean_test_auc",
		"std_test_auc",
		"mean_eval_accuracy",
		"std_eval_accuracy",
		"mean_train_accuracy",
		"std_train_accuracy",
	]
	columnas_top10 = [col for col in columnas_top10 if col in df_resultados.columns]
	df_top10 = df_resultados.head(10)[columnas_top10]
	df_top10.to_csv(TOP10_CSV, index=False)

	print()
	print("Resumen final:")
	print(df_top10.to_string(index=False))
	print()
	print(f"Resultados completos guardados en: {RESULTADOS_CSV}")
	print(f"Top 10 guardado en: {TOP10_CSV}")


if __name__ == "__main__":
	main()
