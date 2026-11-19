#!/usr/bin/env bash
# Ejecuta el pipeline completo: datos -> EDA -> Python -> C++ -> localidad -> comparativa
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
REPS=${REPS:-5}

if [ ! -x "$PY" ]; then
  python -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

echo "== [1/7] Preparando datos";            $PY python/prepare_data.py
echo "== [2/7] EDA";                         $PY python/eda.py > /dev/null
echo "== [3/7] Compilando C++ (Release)"
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build cpp/build -j
echo "== [4/7] Python: baseline y GD manual"; $PY python/baseline.py --reps "$REPS"
echo "== [5/7] C++/Eigen";                   ./cpp/build/logreg --reps "$REPS"
echo "== [6/7] Localidad (C++) y barrido de lote"
./cpp/build/locality_benchmark
$PY python/batch_sweep.py
echo "== [7/7] Graficas y tablas comparativas"; $PY python/compare.py
echo "Listo. Graficas en results/figures/, tabla en results/tabla_comparativa.md"
