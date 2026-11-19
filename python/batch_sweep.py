"""
Barrido de tamano de mini-lote: mismo algoritmo (gradiente descendente) en
NumPy y en C++/Eigen. Mide el tiempo por epoca para cada tamano de lote.

Con lotes pequenos hay muchas iteraciones del bucle de Python por epoca, y el
costo fijo del interprete (~microsegundos por llamada a NumPy) domina; en C++
ese costo es practicamente cero. Con lotes grandes ambos dependen del producto
matriz-vector y la diferencia se reduce.

Salida: results/batch_sweep.csv
Requiere haber compilado cpp/ (cpp/build/logreg).
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from baseline import NumpyLogisticRegression

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CPP_BIN = ROOT / "cpp" / "build" / "logreg"
BATCHES = [16, 64, 256, 1024, 4096, 11982]
EPOCHS = 20
REPS = 3


def main() -> None:
    d = np.load(next((ROOT / "data").glob("mnist_*_*.npz")))
    Xtr, ytr = d["X_train"], d["y_train"]
    rows = []
    for bs in BATCHES:
        # --- Python / NumPy ---
        times = []
        for _ in range(REPS):
            m = NumpyLogisticRegression(lr=0.5, epochs=EPOCHS, batch=bs, l2=1e-4)
            t0 = time.perf_counter()
            m.fit(Xtr, ytr)
            times.append(time.perf_counter() - t0)
        py = min(times) / EPOCHS * 1e3

        # --- C++ / Eigen ---
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([str(CPP_BIN), f"--epochs={EPOCHS}", f"--batch={bs}",
                            f"--reps={REPS}", f"--out={tmp}"],
                           check=True, capture_output=True)
            r = json.loads((Path(tmp) / "cpp_results.json").read_text())
        cpp = r["modelos"][0]["fit_time_s"]["min"] / EPOCHS * 1e3

        rows.append({"batch": bs, "python_ms_por_epoca": py, "cpp_ms_por_epoca": cpp,
                     "speedup": py / cpp})
        print(f"batch {bs:6d} | NumPy {py:8.2f} ms/epoca | C++ {cpp:7.2f} ms/epoca | "
              f"speedup x{py / cpp:5.1f}")

    with open(RES / "batch_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
