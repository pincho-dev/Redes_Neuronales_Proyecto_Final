"""
Fase A.2 - Analisis Exploratorio de Datos (EDA).

Genera en results/figures/:
  eda_01_muestras.png         ejemplos de cada clase
  eda_02_balance_clases.png   distribucion de clases (train/test)
  eda_03_hist_pixeles.png     distribucion de intensidades de pixel
  eda_04_imagen_media.png     imagen promedio por clase y su diferencia
  eda_05_varianza.png         mapa de varianza por pixel (pixeles informativos)
  eda_06_correlacion.png      correlacion entre caracteristicas (pixeles de mayor varianza)
  eda_07_corr_objetivo.png    correlacion de cada pixel con la etiqueta
  eda_08_pca.png              proyeccion PCA en 2D
y un resumen numerico en results/eda_resumen.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def load():
    path = next((ROOT / "data").glob("mnist_*_*.npz"))
    d = np.load(path)
    return d, int(d["pos"]), int(d["neg"])


def main() -> None:
    d, pos, neg = load()
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    names = {0: f"Digito {neg} (y=0)", 1: f"Digito {pos} (y=1)"}

    # DataFrame con los pixeles como columnas (Pandas para el resumen estadistico)
    df = pd.DataFrame(Xtr, columns=[f"px{i}" for i in range(Xtr.shape[1])])
    df["clase"] = ytr.astype(int)

    # 1. Muestras
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    rng = np.random.default_rng(0)
    for row, c in enumerate((0, 1)):
        idx = rng.choice(np.where(ytr == c)[0], 8, replace=False)
        for ax, i in zip(axes[row], idx):
            ax.imshow(Xtr[i].reshape(28, 28), cmap="gray_r")
            ax.axis("off")
        axes[row, 0].set_title(names[c], loc="left", fontsize=10)
    fig.suptitle("Ejemplos del dataset (28x28 = 784 caracteristicas)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_01_muestras.png", dpi=150)
    plt.close(fig)

    # 2. Balance de clases
    bal = pd.DataFrame({
        "conjunto": ["Entrenamiento"] * len(ytr) + ["Prueba"] * len(yte),
        "clase": [names[int(v)] for v in np.concatenate([ytr, yte])],
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=bal, x="conjunto", hue="clase", ax=ax, palette="Set2")
    for cont in ax.containers:
        ax.bar_label(cont)
    ax.set_title("Distribucion de clases")
    ax.set_ylabel("Numero de imagenes")
    fig.tight_layout()
    fig.savefig(FIG / "eda_02_balance_clases.png", dpi=150)
    plt.close(fig)

    # 3. Histograma de intensidades
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in (0, 1):
        vals = Xtr[ytr == c].ravel()
        axes[0].hist(vals, bins=50, alpha=0.6, label=names[c], log=True)
        nz = vals[vals > 0]
        axes[1].hist(nz, bins=50, alpha=0.6, label=names[c], density=True)
    axes[0].set_title("Intensidad de todos los pixeles (escala log)")
    axes[1].set_title("Intensidad de pixeles no nulos")
    for ax in axes:
        ax.set_xlabel("Intensidad normalizada [0,1]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_03_hist_pixeles.png", dpi=150)
    plt.close(fig)

    # 4. Imagen media por clase y diferencia
    m0 = Xtr[ytr == 0].mean(0).reshape(28, 28)
    m1 = Xtr[ytr == 1].mean(0).reshape(28, 28)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(m0, cmap="gray_r"); axes[0].set_title(f"Media {names[0]}")
    axes[1].imshow(m1, cmap="gray_r"); axes[1].set_title(f"Media {names[1]}")
    lim = np.abs(m1 - m0).max()
    im = axes[2].imshow(m1 - m0, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(f"Diferencia ({pos} - {neg})")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "eda_04_imagen_media.png", dpi=150)
    plt.close(fig)

    # 5. Varianza por pixel
    var = Xtr.var(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(var.reshape(28, 28), cmap="viridis")
    axes[0].set_title("Varianza por pixel"); axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].plot(np.sort(var)[::-1])
    axes[1].set_title("Varianza ordenada (784 pixeles)")
    axes[1].set_xlabel("Pixel (ordenado)"); axes[1].set_ylabel("Varianza")
    fig.tight_layout()
    fig.savefig(FIG / "eda_05_varianza.png", dpi=150)
    plt.close(fig)

    # 6. Correlacion entre los 30 pixeles de mayor varianza
    top = np.argsort(var)[::-1][:30]
    corr = df[[f"px{i}" for i in top]].corr()
    corr.index = corr.columns = [f"({i // 28},{i % 28})" for i in top]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"label": "Correlacion de Pearson"})
    ax.set_title("Correlacion entre los 30 pixeles de mayor varianza (fila,col)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_06_correlacion.png", dpi=150)
    plt.close(fig)

    # 7. Correlacion de cada pixel con la etiqueta
    corr_y = df.drop(columns="clase").corrwith(df["clase"]).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    lim = np.abs(corr_y).max()
    im = ax.imshow(corr_y.reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(f"Correlacion pixel-etiqueta (rojo -> {pos})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "eda_07_corr_objetivo.png", dpi=150)
    plt.close(fig)

    # 8. PCA 2D
    pca = PCA(n_components=50, random_state=0).fit(Xtr)
    Z = pca.transform(Xtr)[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sub = rng.choice(len(Z), 3000, replace=False)
    for c in (0, 1):
        s = sub[ytr[sub] == c]
        axes[0].scatter(Z[s, 0], Z[s, 1], s=5, alpha=0.5, label=names[c])
    axes[0].set_title("Proyeccion PCA (2 componentes)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend()
    axes[1].plot(np.cumsum(pca.explained_variance_ratio_), marker=".")
    axes[1].set_title("Varianza explicada acumulada")
    axes[1].set_xlabel("Numero de componentes")
    fig.tight_layout()
    fig.savefig(FIG / "eda_08_pca.png", dpi=150)
    plt.close(fig)

    resumen = {
        "digitos": {"negativo_y0": neg, "positivo_y1": pos},
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "n_caracteristicas": int(Xtr.shape[1]),
        "proporcion_positivos_train": float(ytr.mean()),
        "proporcion_positivos_test": float(yte.mean()),
        "pixeles_siempre_cero_train": int((Xtr.max(0) == 0).sum()),
        "fraccion_pixeles_cero": float((Xtr == 0).mean()),
        "intensidad_media": float(Xtr.mean()),
        "tinta_media_por_clase": {names[c]: float(Xtr[ytr == c].sum(1).mean()) for c in (0, 1)},
        "max_abs_corr_pixel_etiqueta": float(np.abs(corr_y).max()),
        "pca_varianza_2_componentes": float(pca.explained_variance_ratio_[:2].sum()),
        "pca_varianza_50_componentes": float(pca.explained_variance_ratio_.sum()),
        "memoria_X_train_MB": float(Xtr.nbytes / 2**20),
    }
    (ROOT / "results" / "eda_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
