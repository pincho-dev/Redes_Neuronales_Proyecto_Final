"""
Fase A.3 - Prototipo y baseline en Python.

Entrena y evalua tres modelos de regresion logistica sobre el mismo dataset:
  1. sklearn_lbfgs : sklearn.linear_model.LogisticRegression (optimizador L-BFGS) -> baseline
  2. sklearn_sgd   : sklearn.linear_model.SGDClassifier(loss="log_loss")
  3. numpy_gd      : gradiente descendente por lotes escrito a mano con NumPy,
                     EXACTAMENTE el mismo algoritmo e hiperparametros que la version C++.
                     Permite comparar lenguaje contra lenguaje con el mismo algoritmo.

Para una comparacion justa se fija 1 hilo en BLAS (igual que en C++).

Memoria: cada modelo se ejecuta en un subproceso independiente para medir su pico de
memoria residente (RSS) sin contaminarse con los demas, igual que se mide en C++ (VmHWM).

Salidas:
  results/python_results.json   metricas, tiempos y memoria
  results/python_loss.csv       curva de perdida de numpy_gd
  results/python_weights.csv    pesos aprendidos (sklearn_lbfgs y numpy_gd)

Uso:  python baseline.py [--reps 5] [--epochs 200] [--lr 0.5] [--batch 256] [--l2 1e-4]
"""
import os

# Fijar hilos ANTES de importar numpy
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


# ----------------------------------------------------------------------------
# Implementacion manual (misma que en C++)
# ----------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    # Version numericamente estable: evita overflow de exp para z muy negativos
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class NumpyLogisticRegression:
    """Regresion logistica con gradiente descendente por mini-lotes y regularizacion L2.

    Perdida:   L(w,b) = -1/n * sum[y log p + (1-y) log(1-p)] + (l2/2)||w||^2
    Gradiente: dL/dw = X^T (p - y) / n + l2 * w ;  dL/db = mean(p - y)
    """

    def __init__(self, lr=0.5, epochs=200, batch=256, l2=1e-4):
        self.lr, self.epochs, self.batch, self.l2 = lr, epochs, batch, l2
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def loss(self, X, y):
        p = np.clip(sigmoid(X @ self.w + self.b), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                     + 0.5 * self.l2 * self.w @ self.w)

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float32)
        self.b = np.float32(0.0)
        self.loss_history = []
        lr, l2 = np.float32(self.lr), np.float32(self.l2)
        for _ in range(self.epochs):
            # Recorremos los lotes en orden (mismo orden que C++, sin barajar,
            # para que ambas implementaciones sean deterministas y comparables)
            for start in range(0, n, self.batch):
                Xb = X[start:start + self.batch]       # vista, no copia
                yb = y[start:start + self.batch]
                err = sigmoid(Xb @ self.w + self.b) - yb
                grad_w = Xb.T @ err / np.float32(len(yb)) + l2 * self.w
                grad_b = err.mean()
                self.w -= lr * grad_w
                self.b -= lr * grad_b
            self.loss_history.append(self.loss(X, y))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.float32)


# ----------------------------------------------------------------------------
# Utilidades de medicion
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def current_rss_mb() -> float:
    """Memoria residente ACTUAL (VmRSS), igual que current_rss_mb() en C++."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def peak_rss_mb() -> float:
    # ru_maxrss esta en KiB en Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def benchmark(name, make_model, Xtr, ytr, Xte, yte, reps):
    fit_times, pred_times, mem_peaks = [], [], []
    model = None
    for r in range(reps):
        model = make_model()
        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_times.append(time.perf_counter() - t0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_peaks.append(peak / 2**20)

        t0 = time.perf_counter()
        y_pred = model.predict(Xte)
        pred_times.append(time.perf_counter() - t0)
    res = {
        "modelo": name,
        "fit_time_s": {"mean": float(np.mean(fit_times)), "std": float(np.std(fit_times)),
                        "min": float(np.min(fit_times)), "all": fit_times},
        "predict_time_ms": {"mean": float(np.mean(pred_times) * 1e3),
                             "std": float(np.std(pred_times) * 1e3)},
        "train_alloc_peak_MB": float(np.mean(mem_peaks)),
        "test": metrics(yte, y_pred),
        "train": metrics(ytr, model.predict(Xtr)),
    }
    print(f"{name:14s} fit {res['fit_time_s']['mean']:.3f}s +- {res['fit_time_s']['std']:.3f} | "
          f"acc {res['test']['accuracy']:.4f} | f1 {res['test']['f1']:.4f} | "
          f"RSS pico {peak_rss_mb():.1f} MB")
    return res, model


MODEL_NAMES = ("sklearn_lbfgs", "sklearn_sgd", "numpy_gd")


def run_single(name, args):
    """Ejecuta UN modelo (dentro de un subproceso) e imprime su resultado en JSON."""
    rss_start = current_rss_mb()
    d = np.load(next((ROOT / "data").glob("mnist_*_*.npz")))
    Xtr, ytr = d["X_train"], d["y_train"]
    Xte, yte = d["X_test"], d["y_test"]
    rss_data = current_rss_mb()
    n = len(ytr)
    # L2 equivalente para sklearn: su perdida es C*sum(logloss) + 0.5||w||^2
    # La nuestra es mean(logloss) + 0.5*l2||w||^2  =>  C = 1 / (l2 * n)
    C = 1.0 / (args.l2 * n)

    if name == "sklearn_lbfgs":
        make = lambda: LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    elif name == "sklearn_sgd":
        make = lambda: SGDClassifier(loss="log_loss", alpha=args.l2, max_iter=args.epochs,
                                     tol=1e-4, random_state=0)
    else:
        make = lambda: NumpyLogisticRegression(args.lr, args.epochs, args.batch, args.l2)

    res, model = benchmark(name, make, Xtr, ytr, Xte, yte, args.reps)
    res["rss_tras_imports_MB"] = rss_start
    res["rss_tras_cargar_datos_MB"] = rss_data
    res["rss_pico_MB"] = peak_rss_mb()
    res["rss_extra_entrenamiento_MB"] = res["rss_pico_MB"] - rss_data
    if name == "sklearn_lbfgs":
        res["iteraciones"] = int(model.n_iter_[0])
        np.save(RES / "_w_lbfgs.npy", model.coef_.ravel())
    elif name == "sklearn_sgd":
        res["iteraciones"] = int(model.n_iter_)
    else:
        res["loss_final"] = model.loss_history[-1]
        np.savetxt(RES / "python_loss.csv", np.array(model.loss_history), delimiter=",",
                   header="loss", comments="")
        np.save(RES / "_w_numpy.npy", model.w)
    print("@@JSON@@" + json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--only", choices=MODEL_NAMES, help=argparse.SUPPRESS)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)

    if args.only:
        run_single(args.only, args)
        return

    d = np.load(next((ROOT / "data").glob("mnist_*_*.npz")))
    n, dim = d["X_train"].shape
    print(f"Datos: train {d['X_train'].shape}, test {d['X_test'].shape}, dtype {d['X_train'].dtype}")
    results = {
        "lenguaje": "Python",
        "versiones": {"numpy": np.__version__,
                      "sklearn": __import__("sklearn").__version__,
                      "python": __import__("platform").python_version()},
        "hiperparametros": {k: v for k, v in vars(args).items() if k != "only"}
                           | {"C_sklearn": 1.0 / (args.l2 * n)},
        "datos": {"n_train": int(n), "n_test": int(len(d["y_test"])), "d": int(dim),
                  "MB": float(sum(d[k].nbytes for k in ("X_train", "X_test", "y_train", "y_test")) / 2**20)},
        "modelos": [],
    }
    base = [sys.executable, "-I", __file__] + [f"--{k}={v}" for k, v in vars(args).items()
                                                if k != "only"]
    for name in MODEL_NAMES:
        out = subprocess.run(base + ["--only", name], capture_output=True, text=True, check=True)
        for line in out.stdout.splitlines():
            if line.startswith("@@JSON@@"):
                results["modelos"].append(json.loads(line[8:]))
            else:
                print(line)

    (RES / "python_results.json").write_text(json.dumps(results, indent=2))
    w_l, w_n = np.load(RES / "_w_lbfgs.npy"), np.load(RES / "_w_numpy.npy")
    np.savetxt(RES / "python_weights.csv", np.column_stack([w_l, w_n]), delimiter=",",
               header="sklearn_lbfgs,numpy_gd", comments="")
    (RES / "_w_lbfgs.npy").unlink(); (RES / "_w_numpy.npy").unlink()
    print("Resultados en", RES / "python_results.json")


if __name__ == "__main__":
    main()
