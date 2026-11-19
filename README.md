# Benchmarking de Regresión Logística: C++ (Eigen) vs. Python (Scikit-Learn)

Proyecto final 2 – Redes Neuronales. Clasificación binaria de dígitos MNIST (**3 vs 8**) con
regresión logística, implementada en Python (Scikit-Learn + NumPy) y en C++ (Eigen), con
comparación de tiempo, memoria, métricas y un estudio de localidad de caché.

## Resultados principales

| Modelo | Entrenamiento | Memoria pico | Exactitud | F1 |
|---|---|---|---|---|
| Scikit-Learn L-BFGS (baseline) | 0,59 s | 208 MB | 96,72 % | 0,9666 |
| Scikit-Learn SGD | 1,86 s | 206 MB | 96,57 % | 0,9651 |
| Python/NumPy GD (manual) | 3,36 s | 206 MB | 96,88 % | 0,9681 |
| **C++/Eigen GD** | **1,03 s** | **46 MB** | **96,88 %** | **0,9681** |

- Mismo algoritmo: C++ es **×3,3 más rápido** y usa **×4,4 menos memoria** (resultados idénticos).
- Recorrer la matriz en su orden de almacenamiento: **×6,4** más rápido que a contrapaso.
- Informe completo: [`informe/informe.pdf`](informe/informe.pdf) (fuente editable: `informe/informe.docx` / `informe/informe.md`).

## Estructura

```
python/
  prepare_data.py     Fase A: descarga MNIST, filtra 3/8, exporta .npz y .bin para C++
  eda.py              Fase A: análisis exploratorio (Pandas, Matplotlib, Seaborn)
  baseline.py         Fase A: Scikit-Learn (L-BFGS, SGD) + GD manual en NumPy
  batch_sweep.py      Fase C: tiempo por época vs tamaño de lote, Python vs C++
  compare.py          Gráficas y tablas comparativas
cpp/                  Fase B (proyecto CMake, se abre en Qt Creator)
  include/  src/
    dataset           tipos de datos (RowMatrix) y carga binaria
    logistic_regression  motor matemático: sigmoide + gradiente descendente
    metrics           precisión, recall, F1, matriz de confusión
    utils             cronómetro, memoria (/proc), argumentos
    main.cpp          ejecutable `logreg`: entrena, evalúa y mide
    locality_benchmark.cpp  Fase C: experimentos de localidad E1–E5
results/              JSON/CSV de mediciones + figures/ (gráficas)
informe/              informe técnico (md, docx, pdf) y conversor md→docx
run_all.sh            ejecuta todo el pipeline
```

## Cómo ejecutar

Requisitos (Arch Linux): `sudo pacman -S --needed eigen qtcreator cmake gcc python`

```bash
./run_all.sh                 # crea el venv si no existe y ejecuta todo (REPS=5)
```

Paso a paso:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python python/prepare_data.py          # --pos 8 --neg 3 para cambiar dígitos
.venv/bin/python python/eda.py
.venv/bin/python python/baseline.py --reps 5
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j
./cpp/build/logreg --reps 5                       # --epochs --lr --batch --l2
./cpp/build/locality_benchmark
.venv/bin/python python/batch_sweep.py
.venv/bin/python python/compare.py
```

### Qt Creator

`File > Open File or Project…` → `cpp/CMakeLists.txt` → seleccionar el kit y la
configuración **Release** → ejecutar `logreg` o `locality_benchmark` (▶). Los ejecutables
encuentran `data/` y `results/` solos (ruta del proyecto definida en CMake), así que no hace
falta configurar argumentos. Primero hay que haber ejecutado `prepare_data.py`.

### Regenerar el informe

```bash
.venv/bin/python informe/build_docx.py
cd informe && soffice --headless --convert-to pdf informe.docx
```
