"""
Analisis comparativo: lee los resultados de Python y C++ y genera las graficas
y tablas del reporte tecnico.

Entradas: results/python_results.json, results/cpp_results.json,
          results/python_loss.csv, results/cpp_loss.csv,
          results/python_weights.csv, results/cpp_weights.csv,
          results/cpp_locality.csv, results/batch_sweep.csv
Salidas:  results/figures/cmp_*.png, results/figures/loc_*.png,
          results/tabla_comparativa.csv / .md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

LABELS = {
    "sklearn_lbfgs": "Scikit-Learn\nL-BFGS",
    "sklearn_sgd": "Scikit-Learn\nSGD",
    "numpy_gd": "Python/NumPy\nGD (manual)",
    "cpp_eigen_gd": "C++/Eigen\nGD",
}
COLORS = {"sklearn_lbfgs": "#4C72B0", "sklearn_sgd": "#64B5CD",
          "numpy_gd": "#DD8452", "cpp_eigen_gd": "#55A868"}


def load_models():
    py = json.loads((RES / "python_results.json").read_text())
    cpp = json.loads((RES / "cpp_results.json").read_text())
    return py, cpp, py["modelos"] + cpp["modelos"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def bar(ax, models, values, errs=None, fmt="{:.3f}"):
    keys = [m["modelo"] for m in models]
    x = np.arange(len(keys))
    bars = ax.bar(x, values, yerr=errs, capsize=4, color=[COLORS[k] for k in keys])
    ax.set_xticks(x, [LABELS[k] for k in keys])
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points")


def main() -> None:
    py, cpp, models = load_models()
    keys = [m["modelo"] for m in models]

    # --------------------------------------------------------------- tabla
    tabla = pd.DataFrame([{
        "Modelo": LABELS[m["modelo"]].replace("\n", " "),
        "Lenguaje": "C++" if m["modelo"].startswith("cpp") else "Python",
        "Entrenamiento (s)": m["fit_time_s"]["mean"],
        "Desv. (s)": m["fit_time_s"]["std"],
        "Prediccion (ms)": m["predict_time_ms"]["mean"],
        "RSS pico (MB)": m["rss_pico_MB"],
        "RSS extra entrenamiento (MB)": m["rss_extra_entrenamiento_MB"],
        "Accuracy": m["test"]["accuracy"],
        "Precision": m["test"]["precision"],
        "Recall": m["test"]["recall"],
        "F1": m["test"]["f1"],
    } for m in models])
    tabla.to_csv(RES / "tabla_comparativa.csv", index=False)
    (RES / "tabla_comparativa.md").write_text(tabla.to_markdown(index=False, floatfmt=".4f"))
    print(tabla.to_string(index=False))

    # --------------------------------------------- 1. tiempo de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bar(axes[0], models, [m["fit_time_s"]["mean"] for m in models],
        [m["fit_time_s"]["std"] for m in models])
    axes[0].set_ylabel("Segundos")
    axes[0].set_title("Tiempo de entrenamiento (media de repeticiones)")
    bar(axes[1], models, [m["predict_time_ms"]["mean"] for m in models], fmt="{:.2f}")
    axes[1].set_ylabel("Milisegundos")
    axes[1].set_title(f"Tiempo de prediccion ({py['datos']['n_test']} imagenes)")
    save(fig, "cmp_01_tiempos.png")

    # ----------------------------------------------------------- 2. memoria
    fig, ax = plt.subplots(figsize=(9, 4.8))
    base = np.array([m["rss_tras_imports_MB"] for m in models])
    data = np.array([m["rss_tras_cargar_datos_MB"] for m in models]) - base
    extra = np.array([m["rss_extra_entrenamiento_MB"] for m in models])
    x = np.arange(len(models))
    ax.bar(x, base, label="Proceso + librerias (antes de datos)", color="#8C8C8C")
    ax.bar(x, data, bottom=base, label="Carga de datos", color="#4C72B0")
    ax.bar(x, extra, bottom=base + data, label="Extra durante entrenamiento", color="#C44E52")
    for i, tot in enumerate(base + data + extra):
        ax.annotate(f"{tot:.1f} MB", (i, tot), ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=9)
    ax.set_xticks(x, [LABELS[k] for k in keys])
    ax.set_ylabel("Memoria residente (MB)")
    ax.set_title("Consumo de memoria: pico de RSS del proceso")
    ax.legend()
    save(fig, "cmp_02_memoria.png")

    # ----------------------------------------------------------- 3. metricas
    met = pd.DataFrame([{"Modelo": LABELS[m["modelo"]], "Metrica": k.capitalize(),
                         "Valor": m["test"][k]}
                        for m in models for k in ("accuracy", "precision", "recall", "f1")])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.barplot(data=met, x="Metrica", y="Valor", hue="Modelo", ax=ax,
                palette=[COLORS[k] for k in keys])
    lo = met["Valor"].min()
    ax.set_ylim(lo - 0.02, min(1.0, met["Valor"].max() + 0.01))
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.4f", fontsize=7, rotation=90, padding=2)
    ax.set_title("Metricas de rendimiento en el conjunto de prueba")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "cmp_03_metricas.png")

    # ----------------------------------------------- 4. matrices de confusion
    neg, pos = json.loads((RES / "eda_resumen.json").read_text())["digitos"].values()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    for ax, m in zip(axes, models):
        c = m["test"]["confusion"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=[str(neg), str(pos)], yticklabels=[str(neg), str(pos)])
        ax.set_title(LABELS[m["modelo"]].replace("\n", " "), fontsize=10)
        ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    save(fig, "cmp_04_confusion.png")

    # -------------------------------------------------- 5. curva de perdida
    lp = pd.read_csv(RES / "python_loss.csv")["loss"]
    lc = pd.read_csv(RES / "cpp_loss.csv")["loss"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(lp, label="Python/NumPy", lw=3, alpha=0.6, color=COLORS["numpy_gd"])
    axes[0].plot(lc, label="C++/Eigen", lw=1.2, ls="--", color=COLORS["cpp_eigen_gd"])
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Perdida (log-loss + L2)")
    axes[0].set_title("Curva de entrenamiento"); axes[0].legend()
    axes[1].plot(np.abs(lp - lc), color="#C44E52")
    axes[1].set_yscale("log"); axes[1].set_xlabel("Epoca")
    axes[1].set_title("|perdida Python - perdida C++| (diferencias de redondeo float32)")
    save(fig, "cmp_05_perdida.png")

    # --------------------------------------------------------- 6. pesos
    wp = pd.read_csv(RES / "python_weights.csv")
    wc = pd.read_csv(RES / "cpp_weights.csv")
    ws = {"Scikit-Learn L-BFGS": wp["sklearn_lbfgs"].to_numpy(),
          "Python/NumPy GD": wp["numpy_gd"].to_numpy(),
          "C++/Eigen GD": wc["cpp_eigen_gd"].to_numpy()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, w) in zip(axes, ws.items()):
        lim = np.abs(w).max()
        im = ax.imshow(w.reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(name); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Pesos aprendidos w (rojo -> {pos}, azul -> {neg})")
    save(fig, "cmp_06_pesos.png")
    corr_np_cpp = float(np.corrcoef(ws["Python/NumPy GD"], ws["C++/Eigen GD"])[0, 1])
    maxdiff = float(np.abs(ws["Python/NumPy GD"] - ws["C++/Eigen GD"]).max())

    # ------------------------------------------------- 7. barrido de lote
    sw = pd.read_csv(RES / "batch_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(sw["batch"], sw["python_ms_por_epoca"], "o-", label="Python/NumPy",
                 color=COLORS["numpy_gd"])
    axes[0].plot(sw["batch"], sw["cpp_ms_por_epoca"], "s-", label="C++/Eigen",
                 color=COLORS["cpp_eigen_gd"])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Tamano de mini-lote"); axes[0].set_ylabel("ms por epoca")
    axes[0].set_title("Tiempo por epoca vs tamano de lote"); axes[0].legend()
    axes[1].bar(sw["batch"].astype(str), sw["speedup"], color="#8172B3")
    for i, v in enumerate(sw["speedup"]):
        axes[1].annotate(f"x{v:.1f}", (i, v), ha="center", va="bottom", xytext=(0, 3),
                         textcoords="offset points")
    axes[1].set_xlabel("Tamano de mini-lote"); axes[1].set_ylabel("Speedup (Python / C++)")
    axes[1].set_title("Aceleracion de C++ sobre Python (mismo algoritmo)")
    save(fig, "cmp_07_barrido_lote.png")

    # -------------------------------------------- 8. localidad (C++)
    loc = pd.read_csv(RES / "cpp_locality.csv")

    e1 = loc[loc.experimento == "E1"]
    e2 = loc[loc.experimento == "E2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, e, title in ((axes[0], e1, "E1 Localidad espacial: g = X^T e"),
                         (axes[1], e2, "E2 Localidad temporal: gradiente completo")):
        b = ax.barh(e.variante, e.ms_mediana, color=sns.color_palette("Set2", len(e)))
        ax.bar_label(b, fmt="%.2f ms", padding=3)
        ax.set_xlabel("Tiempo (ms, mediana)"); ax.set_title(title)
        ax.set_xlim(0, e.ms_mediana.max() * 1.25)
        ax.invert_yaxis()
    save(fig, "loc_01_espacial_temporal.png")

    e3 = loc[loc.experimento == "E3"]
    n_train = py["datos"]["n_train"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(e3.parametro, e3.ms_mediana, "o-", color=COLORS["cpp_eigen_gd"])
    ax.set_xscale("log")
    ax.set_xlabel("Tamano de mini-lote (filas)"); ax.set_ylabel("ms por epoca")
    ax.set_title("E3 Tiempo de una epoca vs tamano de lote (C++/Eigen)")
    kb_row = py["datos"]["d"] * 4 / 1024
    for cache, kb in (("L1d 32 KiB", 32), ("L2 512 KiB", 512), ("L3 4 MiB", 4096)):
        rows = kb / kb_row
        if rows < n_train:
            ax.axvline(rows, ls=":", color="gray")
            ax.text(rows, ax.get_ylim()[1] * 0.97, f" {cache}\n (~{rows:.0f} filas)",
                    fontsize=8, va="top", color="gray")
    save(fig, "loc_02_tamano_lote.png")

    e4 = loc[loc.experimento == "E4"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    sns.barplot(data=e4, x="parametro", y="ms_mediana", hue="variante", ax=ax,
                palette=["#55A868", "#C44E52"])
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
    ax.set_xlabel("Tamano de mini-lote"); ax.set_ylabel("ms por epoca")
    ax.set_title("E4 Orden de almacenamiento: RowMajor vs ColMajor")
    save(fig, "loc_03_rowmajor_colmajor.png")

    e5 = loc[loc.experimento == "E5"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    b = ax.barh(e5.variante, e5.ms_mediana, color=["#C44E52", "#55A868", "#55A868", "#55A868"])
    ax.bar_label(b, fmt="%.4f ms", padding=3)
    ax.set_xlim(0, e5.ms_mediana.max() * 1.3)
    ax.set_xlabel("Tiempo por llamada (ms)")
    ax.set_title("E5 Pasar X (35.8 MB) a una funcion: valor vs referencia vs puntero")
    ax.invert_yaxis()
    save(fig, "loc_04_paso_parametros.png")

    # --------------------------------------------------- resumen numerico
    get = {m["modelo"]: m for m in models}
    resumen = {
        "speedup_fit_cpp_vs_numpy": get["numpy_gd"]["fit_time_s"]["mean"] / get["cpp_eigen_gd"]["fit_time_s"]["mean"],
        "speedup_fit_cpp_vs_sklearn_lbfgs": get["sklearn_lbfgs"]["fit_time_s"]["mean"] / get["cpp_eigen_gd"]["fit_time_s"]["mean"],
        "speedup_fit_cpp_vs_sklearn_sgd": get["sklearn_sgd"]["fit_time_s"]["mean"] / get["cpp_eigen_gd"]["fit_time_s"]["mean"],
        "speedup_predict_cpp_vs_sklearn": get["sklearn_lbfgs"]["predict_time_ms"]["mean"] / get["cpp_eigen_gd"]["predict_time_ms"]["mean"],
        "ratio_memoria_python_vs_cpp": get["numpy_gd"]["rss_pico_MB"] / get["cpp_eigen_gd"]["rss_pico_MB"],
        "correlacion_pesos_numpy_cpp": corr_np_cpp,
        "max_diff_pesos_numpy_cpp": maxdiff,
        "diff_loss_final": abs(float(lp.iloc[-1]) - float(lc.iloc[-1])),
        "localidad_speedup_fila_vs_columna": float(e1.ms_mediana.iloc[1] / e1.ms_mediana.iloc[0]),
        "localidad_speedup_fusion": float(e2.ms_mediana.iloc[0] / e2.ms_mediana.iloc[1]),
        "lote_optimo_E3": int(e3.loc[e3.ms_mediana.idxmin(), "parametro"]),
    }
    (RES / "resumen_comparativo.json").write_text(json.dumps(resumen, indent=2))
    print(json.dumps(resumen, indent=2))


if __name__ == "__main__":
    main()
