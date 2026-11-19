// ============================================================================
// dataset.cpp - Carga de datos binarios.
// ============================================================================
#include "dataset.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace logreg {

RowMatrix load_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("No se pudo abrir " + path);

    std::int32_t shape[2];
    in.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!in || shape[0] <= 0 || shape[1] <= 0)
        throw std::runtime_error("Cabecera invalida en " + path);

    RowMatrix M(shape[0], shape[1]);
    // M.data() es un puntero al bloque contiguo de memoria de la matriz:
    // leemos el archivo directamente ahi, sin buffers intermedios.
    const std::streamsize nbytes =
        static_cast<std::streamsize>(M.size()) * static_cast<std::streamsize>(sizeof(float));
    in.read(reinterpret_cast<char*>(M.data()), nbytes);
    if (in.gcount() != nbytes) throw std::runtime_error("Archivo truncado: " + path);
    return M;  // NRVO / move: no se copia la matriz
}

Dataset load_dataset(const std::string& dir, const std::string& split) {
    Dataset ds;
    ds.X = load_matrix(dir + "/X_" + split + ".bin");
    RowMatrix yy = load_matrix(dir + "/y_" + split + ".bin");
    if (yy.rows() != ds.X.rows())
        throw std::runtime_error("X e y tienen distinto numero de filas en " + split);
    ds.y = Eigen::Map<Vector>(yy.data(), yy.rows());
    return ds;
}

}  // namespace logreg
