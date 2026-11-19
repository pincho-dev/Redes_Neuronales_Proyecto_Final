// ============================================================================
// main.cpp - Entrena la regresion logistica en C++/Eigen, evalua y mide
// tiempo de entrenamiento, tiempo de prediccion y memoria.
//
// Uso:  ./logreg [--data DIR] [--out DIR] [--reps 5] [--epochs 200]
//               [--lr 0.5] [--batch 256] [--l2 1e-4]
// Sin argumentos usa <raiz del proyecto>/data y <raiz>/results.
// ============================================================================
#include "dataset.hpp"
#include "logistic_regression.hpp"
#include "metrics.hpp"
#include "utils.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace logreg;

namespace {

// Eigen 5 conserva EIGEN_WORLD_VERSION=3 por compatibilidad; usar la cadena real
std::string eigen_version() {
#ifdef EIGEN_VERSION_STRING
    return EIGEN_VERSION_STRING;
#else
    return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
           std::to_string(EIGEN_MINOR_VERSION);
#endif
}

// Conjuntos SIMD activos (SimdInstructionSetsInUse() no informa AVX2/FMA)
std::string simd_sets() {
    std::string s;
#ifdef EIGEN_VECTORIZE_AVX2
    s += "AVX2 ";
#endif
#ifdef EIGEN_VECTORIZE_FMA
    s += "FMA ";
#endif
    return s + Eigen::SimdInstructionSetsInUse();
}

std::string metrics_json(const Metrics& m) {
    std::ostringstream o;
    o << std::setprecision(10)
      << "{\"accuracy\": " << m.accuracy << ", \"precision\": " << m.precision
      << ", \"recall\": " << m.recall << ", \"f1\": " << m.f1
      << ", \"confusion\": {\"tn\": " << m.tn << ", \"fp\": " << m.fp
      << ", \"fn\": " << m.fn << ", \"tp\": " << m.tp << "}}";
    return o.str();
}

std::string vec_json(const std::vector<double>& v) {
    std::ostringstream o;
    o << std::setprecision(10) << "[";
    for (std::size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << v[i];
    return o.str() + "]";
}

}  // namespace

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    const std::string data_dir = get_arg(args, "data", std::string(PROJECT_ROOT) + "/data");
    const std::string out_dir  = get_arg(args, "out", std::string(PROJECT_ROOT) + "/results");
    const int reps = std::stoi(get_arg(args, "reps", "5"));

    Hyperparams hp;
    hp.epochs = std::stoi(get_arg(args, "epochs", "200"));
    hp.lr     = std::stof(get_arg(args, "lr", "0.5"));
    hp.batch  = std::stoi(get_arg(args, "batch", "256"));
    hp.l2     = std::stof(get_arg(args, "l2", "1e-4"));

    const double rss_start = current_rss_mb();
    Dataset train, test;
    try {
        train = load_dataset(data_dir, "train");
        test  = load_dataset(data_dir, "test");
    } catch (const std::exception& e) {
        std::cerr << "Error cargando datos: " << e.what()
                  << "\nEjecute primero: python python/prepare_data.py\n";
        return 1;
    }
    const double rss_data = current_rss_mb();
    std::cout << "Datos: train " << train.rows() << "x" << train.cols()
              << ", test " << test.rows() << "x" << test.cols()
              << " | " << (train.bytes() + test.bytes()) / 1048576.0 << " MB\n"
              << "Eigen " << eigen_version() << " | hilos: " << Eigen::nbThreads()
              << " | SIMD: " << simd_sets() << "\n";

    std::vector<double> fit_times, pred_times;
    LogisticRegression model(hp);
    Vector y_pred;
    for (int r = 0; r < reps; ++r) {
        Timer t;
        model.fit(train.X, train.y);          // pasa por referencia: sin copia
        fit_times.push_back(t.seconds());

        t.reset();
        y_pred = model.predict(test.X);
        pred_times.push_back(t.seconds() * 1e3);
        std::printf("  rep %d: fit %.3f s | predict %.3f ms | loss %.5f\n", r + 1,
                    fit_times.back(), pred_times.back(), model.loss_history().back());
    }
    const double rss_peak = peak_rss_mb();

    const Metrics mt = compute_metrics(test.y, y_pred);
    const Metrics mtr = compute_metrics(train.y, model.predict(train.X));
    const Stats sf = stats(fit_times), sp = stats(pred_times);

    std::printf("cpp_eigen_gd   fit %.3fs +- %.3f | acc %.4f | f1 %.4f | RSS pico %.1f MB\n",
                sf.mean, sf.stdev, mt.accuracy, mt.f1, rss_peak);

    // ---- Guardar resultados -------------------------------------------------
    std::ofstream js(out_dir + "/cpp_results.json");
    js << std::setprecision(10) << "{\n"
       << "  \"lenguaje\": \"C++\",\n"
       << "  \"versiones\": {\"eigen\": \"" << eigen_version() << "\", \"compilador\": \"g++ "
       << __VERSION__ << "\", \"simd\": \"" << simd_sets() << "\"},\n"
       << "  \"hiperparametros\": {\"epochs\": " << hp.epochs << ", \"lr\": " << hp.lr
       << ", \"batch\": " << hp.batch << ", \"l2\": " << hp.l2 << ", \"reps\": " << reps << "},\n"
       << "  \"datos\": {\"n_train\": " << train.rows() << ", \"n_test\": " << test.rows()
       << ", \"d\": " << train.cols() << ", \"MB\": " << (train.bytes() + test.bytes()) / 1048576.0
       << "},\n"
       << "  \"modelos\": [{\n"
       << "    \"modelo\": \"cpp_eigen_gd\",\n"
       << "    \"fit_time_s\": {\"mean\": " << sf.mean << ", \"std\": " << sf.stdev
       << ", \"min\": " << sf.min << ", \"all\": " << vec_json(fit_times) << "},\n"
       << "    \"predict_time_ms\": {\"mean\": " << sp.mean << ", \"std\": " << sp.stdev << "},\n"
       << "    \"rss_tras_imports_MB\": " << rss_start << ",\n"
       << "    \"rss_tras_cargar_datos_MB\": " << rss_data << ",\n"
       << "    \"rss_pico_MB\": " << rss_peak << ",\n"
       << "    \"rss_extra_entrenamiento_MB\": " << rss_peak - rss_data << ",\n"
       << "    \"loss_final\": " << model.loss_history().back() << ",\n"
       << "    \"test\": " << metrics_json(mt) << ",\n"
       << "    \"train\": " << metrics_json(mtr) << "\n"
       << "  }]\n}\n";

    std::ofstream lf(out_dir + "/cpp_loss.csv");
    lf << "loss\n" << std::setprecision(9);
    for (float l : model.loss_history()) lf << l << "\n";

    std::ofstream wf(out_dir + "/cpp_weights.csv");
    wf << "cpp_eigen_gd\n" << std::setprecision(9);
    for (Eigen::Index i = 0; i < model.weights().size(); ++i) wf << model.weights()[i] << "\n";

    std::cout << "Resultados en " << out_dir << "/cpp_results.json\n";
    return 0;
}
