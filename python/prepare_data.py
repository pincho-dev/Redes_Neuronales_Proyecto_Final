"""
Fase A.1 - Preparacion del dataset.

Descarga MNIST (archivos IDX originales de LeCun, mirror de Google), filtra dos digitos para clasificacion binaria,
normaliza los pixeles a [0, 1] y guarda:
  - data/mnist_<a>_<b>.npz            -> para los scripts de Python
  - data/{X,y}_{train,test}.bin       -> para el programa en C++

Formato binario (little-endian):
  int32 filas, int32 columnas, luego filas*columnas float32 en orden row-major.

Uso:  python prepare_data.py [--pos 8] [--neg 3]
"""
import argparse
import gzip
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download() -> None:
    """Descarga los 4 archivos IDX si no existen todavia."""
    RAW.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        dest = RAW / fname
        if not dest.exists():
            print("Descargando", fname)
            urllib.request.urlretrieve(MIRROR + fname, dest)


def read_idx(path: Path) -> np.ndarray:
    """Lee un archivo IDX (formato de MNIST): cabecera big-endian + bytes uint8."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def save_bin(path: Path, arr: np.ndarray) -> None:
    """Guarda una matriz float32 con cabecera (filas, columnas) para C++."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    with open(path, "wb") as f:
        np.array(arr.shape, dtype=np.int32).tofile(f)
        arr.tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=8, help="digito de la clase positiva (y=1)")
    ap.add_argument("--neg", type=int, default=3, help="digito de la clase negativa (y=0)")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    download()

    # Particion estandar de MNIST: 60000 entrenamiento, 10000 prueba
    splits = ("train", "test")
    out = {}
    for name in splits:
        Xs = read_idx(RAW / FILES[f"{name}_images"]).reshape(-1, 784).astype(np.float32) / 255.0
        ls = read_idx(RAW / FILES[f"{name}_labels"]).astype(np.int64)
        mask = (ls == args.pos) | (ls == args.neg)
        out[f"X_{name}"] = Xs[mask]
        out[f"y_{name}"] = (ls[mask] == args.pos).astype(np.float32)
        out[f"digits_{name}"] = ls[mask]

    np.savez_compressed(DATA / f"mnist_{args.neg}_{args.pos}.npz",
                        pos=args.pos, neg=args.neg, **out)
    for name in splits:
        save_bin(DATA / f"X_{name}.bin", out[f"X_{name}"])
        save_bin(DATA / f"y_{name}.bin", out[f"y_{name}"])
        n = len(out[f"y_{name}"])
        print(f"{name}: {n} muestras | positivos ({args.pos}) = {int(out[f'y_{name}'].sum())}"
              f" | negativos ({args.neg}) = {n - int(out[f'y_{name}'].sum())}")
    print("Datos guardados en", DATA)


if __name__ == "__main__":
    main()
