// ============================================================================
// locality_benchmark.cpp - Fase C: estudio de localidad espacial y temporal.
//
// Experimentos (todos sobre X_train, 11982 x 784 floats = 35.8 MB, 1 hilo):
//
//  E1 Localidad ESPACIAL - orden de recorrido sobre una matriz row-major.
//     Se calcula el gradiente g = X^T e con bucles manuales y punteros crudos:
//       a) fila por fila  (accesos contiguos, stride 1)
//       b) columna por columna (salto de 784 floats = 3136 B entre accesos:
//          cada lectura usa 4 B de una linea de cache de 64 B)
//
//  E2 Localidad TEMPORAL - fusion de las dos pasadas del gradiente.
//       a) dos pasadas: z = X w  y luego  g = X^T e  (X se lee 2 veces de RAM)
//       b) una pasada fusionada: por cada fila x_i se calcula z_i y se acumula
//          e_i * x_i mientras x_i sigue en cache L1 (X se lee 1 vez)
//       c) Eigen (dos productos matriz-vector optimizados con SIMD)
//
//  E3 Tamano de mini-lote: tiempo por epoca y por muestra segun el tamano del
//     bloque de filas reutilizado entre z = Xb w y g = Xb^T e.
//
//  E4 Orden de almacenamiento en Eigen: RowMajor vs ColMajor para el paso de
//     gradiente completo con lotes.
//
//  E5 Paso de parametros: por valor (copia de 35.8 MB), por referencia
//     constante y por Eigen::Ref.
//
// Salida: results/cpp_locality.csv  (experimento, variante, parametro, ms)
// ============================================================================
#include "dataset.hpp"
#include "logistic_regression.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

using namespace logreg;

namespace {

struct Row { std::string exp, variant; long param; double ms_median, ms_min; };
std::vector<Row> g_rows;

// Ejecuta f() varias veces y devuelve la mediana en ms (descartando 1 calentamiento)
Stats time_ms(const std::function<void()>& f, int reps) {
    f();  // calentamiento: carga caches / TLB / paginas
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        Timer tm;
        f();
        t.push_back(tm.seconds() * 1e3);
    }
    return stats(t);
}

void record(const std::string& e, const std::string& v, long p, const Stats& s) {
    g_rows.push_back({e, v, p, s.median, s.min});
    std::printf("  %-4s %-28s %8ld  mediana %9.3f ms  (min %9.3f)\n", e.c_str(), v.c_str(), p,
                s.median, s.min);
}

inline float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// ---------------------------------------------------------------------------
// E1: g = X^T e con punteros crudos
// ---------------------------------------------------------------------------
// a) Recorrido fila por fila: el puntero avanza de forma contigua (stride 1)
void grad_row_order(const float* X, const float* e, float* g, long n, long d) {
    for (long j = 0; j < d; ++j) g[j] = 0.0f;
    for (long i = 0; i < n; ++i) {
        const float* xi = X + i * d;   // inicio de la fila i
        const float ei = e[i];
        for (long j = 0; j < d; ++j) g[j] += ei * xi[j];
    }
}

// b) Recorrido columna por columna sobre la MISMA matriz row-major:
//    entre dos accesos consecutivos hay d*4 = 3136 bytes -> fallo de cache casi seguro
void grad_col_order(const float* X, const float* e, float* g, long n, long d) {
    for (long j = 0; j < d; ++j) {
        float acc = 0.0f;
        const float* col = X + j;
        for (long i = 0; i < n; ++i) acc += e[i] * col[i * d];
        g[j] = acc;
    }
}

// ---------------------------------------------------------------------------
// E2: dos pasadas vs una pasada fusionada (gradiente completo de la regresion)
// ---------------------------------------------------------------------------
void grad_two_pass(const float* X, const float* y, const float* w, float b, float* z, float* g,
                   long n, long d) {
    for (long i = 0; i < n; ++i) {               // pasada 1: z = X w + b ; e = p - y
        const float* xi = X + i * d;
        float acc = b;
        for (long j = 0; j < d; ++j) acc += xi[j] * w[j];
        z[i] = sigmoid(acc) - y[i];
    }
    grad_row_order(X, z, g, n, d);               // pasada 2: g = X^T e
}

void grad_fused(const float* X, const float* y, const float* w, float b, float* g, long n,
                long d) {
    for (long j = 0; j < d; ++j) g[j] = 0.0f;
    for (long i = 0; i < n; ++i) {
        const float* xi = X + i * d;
        float acc = b;
        for (long j = 0; j < d; ++j) acc += xi[j] * w[j];
        const float ei = sigmoid(acc) - y[i];
        for (long j = 0; j < d; ++j) g[j] += ei * xi[j];   // x_i aun esta en L1
    }
}

// ---------------------------------------------------------------------------
// E3/E4: una epoca de gradiente descendente con Eigen para un layout dado
// ---------------------------------------------------------------------------
template <class Mat>
void eigen_epoch(const Mat& X, const Vector& y, Vector& w, float& b, Vector& zbuf, Vector& g,
                 long bs, float lr) {
    const long n = X.rows();
    for (long s = 0; s < n; s += bs) {
        const long m = std::min(bs, n - s);
        auto Xb = X.middleRows(s, m);
        auto z = zbuf.head(m);
        z.noalias() = Xb * w;
        z.array() += b;
        z = (1.0f + (-z.array()).exp()).inverse().matrix();
        z -= y.segment(s, m);
        g.noalias() = Xb.transpose() * z;
        w -= (lr / m) * g;
        b -= lr * z.mean();
    }
}

// ---------------------------------------------------------------------------
// E5: paso de parametros
// ---------------------------------------------------------------------------
__attribute__((noinline)) float sum_by_value(RowMatrix X) { return X(0, 0) + X(X.rows() - 1, 0); }
__attribute__((noinline)) float sum_by_ref(const RowMatrix& X) { return X(0, 0) + X(X.rows() - 1, 0); }
__attribute__((noinline)) float sum_by_eigen_ref(const Eigen::Ref<const RowMatrix>& X) {
    return X(0, 0) + X(X.rows() - 1, 0);
}
__attribute__((noinline)) float sum_by_pointer(const float* X, long n, long d) {
    return X[0] + X[(n - 1) * d];
}

}  // namespace

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    const std::string data_dir = get_arg(args, "data", std::string(PROJECT_ROOT) + "/data");
    const std::string out_dir  = get_arg(args, "out", std::string(PROJECT_ROOT) + "/results");
    const int reps = std::stoi(get_arg(args, "reps", "15"));

    const Dataset tr = load_dataset(data_dir, "train");
    const long n = tr.rows(), d = tr.cols();
    std::printf("X_train %ld x %ld (%.1f MB), %d repeticiones por medicion\n", n, d,
                n * d * 4 / 1048576.0, reps);

    // Vectores de prueba (valores fijos y reproducibles)
    Vector e = Vector::LinSpaced(n, -0.5f, 0.5f);
    Vector w = Vector::Constant(d, 0.001f);
    Vector g(d), z(n);
    const float* Xp = tr.X.data();   // puntero crudo al bloque contiguo

    // ---------------- E1 ----------------
    std::puts("\nE1 Localidad espacial: g = X^T e (bucles manuales con punteros)");
    record("E1", "fila_por_fila (stride 1)", d,
           time_ms([&] { grad_row_order(Xp, e.data(), g.data(), n, d); do_not_optimize(g[0]); }, reps));
    Vector g2(d);
    record("E1", "columna_por_columna (stride d)", d,
           time_ms([&] { grad_col_order(Xp, e.data(), g2.data(), n, d); do_not_optimize(g2[0]); }, reps));
    std::printf("  diferencia maxima entre resultados: %.2e\n", (g - g2).cwiseAbs().maxCoeff());
    record("E1", "eigen X.transpose()*e", d,
           time_ms([&] { g.noalias() = tr.X.transpose() * e; do_not_optimize(g[0]); }, reps));

    // ---------------- E2 ----------------
    std::puts("\nE2 Localidad temporal: gradiente completo, 2 pasadas vs 1 pasada fusionada");
    record("E2", "manual_dos_pasadas", n,
           time_ms([&] { grad_two_pass(Xp, tr.y.data(), w.data(), 0.f, z.data(), g.data(), n, d);
                         do_not_optimize(g[0]); }, reps));
    record("E2", "manual_fusionado", n,
           time_ms([&] { grad_fused(Xp, tr.y.data(), w.data(), 0.f, g.data(), n, d);
                         do_not_optimize(g[0]); }, reps));
    record("E2", "eigen_dos_productos", n, time_ms([&] {
        z.noalias() = tr.X * w;
        z = (1.0f + (-z.array()).exp()).inverse().matrix() - tr.y;
        g.noalias() = tr.X.transpose() * z;
        do_not_optimize(g[0]);
    }, reps));

    // ---------------- E3 ----------------
    std::puts("\nE3 Tamano de mini-lote (Eigen, row-major): tiempo de UNA epoca");
    const std::vector<long> batches = {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, n};
    for (long bs : batches) {
        Vector ww = Vector::Zero(d), zb(bs), gg(d);
        float b = 0.f;
        const int r = bs < 16 ? std::max(3, reps / 3) : reps;
        record("E3", "epoca_rowmajor", bs, time_ms([&] {
            eigen_epoch(tr.X, tr.y, ww, b, zb, gg, bs, 0.5f);
            do_not_optimize(ww[0]);
        }, r));
    }

    // ---------------- E4 ----------------
    std::puts("\nE4 Orden de almacenamiento en Eigen (una epoca)");
    const ColMatrix Xc = tr.X;   // copia con layout column-major
    for (long bs : {32L, 256L, 2048L, n}) {
        Vector w1 = Vector::Zero(d), w2 = Vector::Zero(d), zb(bs), gg(d);
        float b1 = 0.f, b2 = 0.f;
        record("E4", "RowMajor", bs, time_ms([&] { eigen_epoch(tr.X, tr.y, w1, b1, zb, gg, bs, 0.5f);
                                                   do_not_optimize(w1[0]); }, reps));
        record("E4", "ColMajor", bs, time_ms([&] { eigen_epoch(Xc, tr.y, w2, b2, zb, gg, bs, 0.5f);
                                                   do_not_optimize(w2[0]); }, reps));
    }

    // ---------------- E5 ----------------
    std::puts("\nE5 Paso de la matriz X (35.8 MB) a una funcion");
    record("E5", "por_valor (copia)", 0, time_ms([&] { do_not_optimize(sum_by_value(tr.X)); }, reps));
    record("E5", "referencia_const", 0, time_ms([&] { do_not_optimize(sum_by_ref(tr.X)); }, reps));
    record("E5", "Eigen::Ref", 0, time_ms([&] { do_not_optimize(sum_by_eigen_ref(tr.X)); }, reps));
    record("E5", "puntero_crudo", 0,
           time_ms([&] { do_not_optimize(sum_by_pointer(tr.X.data(), n, d)); }, reps));

    std::ofstream f(out_dir + "/cpp_locality.csv");
    f << "experimento,variante,parametro,ms_mediana,ms_min\n";
    for (const auto& r : g_rows)
        f << r.exp << "," << r.variant << "," << r.param << "," << r.ms_median << "," << r.ms_min << "\n";
    std::cout << "\nResultados en " << out_dir << "/cpp_locality.csv\n";
    return 0;
}
