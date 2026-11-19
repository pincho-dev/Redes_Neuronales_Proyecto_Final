#pragma once
// ============================================================================
// dataset.hpp - Tipos de datos y carga del dataset binario generado en Python.
// ============================================================================
#include <Eigen/Dense>
#include <string>

namespace logreg {

// Matriz de caracteristicas en orden ROW-MAJOR: cada fila es una imagen de 784
// pixeles almacenada de forma contigua. Como el algoritmo procesa imagen por
// imagen (y por lotes de filas consecutivas), este orden maximiza la localidad
// espacial: al leer un pixel, la linea de cache trae los 15 pixeles siguientes.
using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vector    = Eigen::VectorXf;

struct Dataset {
    RowMatrix X;   // n x d   (imagenes)
    Vector    y;   // n       (etiquetas 0/1)

    Eigen::Index rows() const { return X.rows(); }
    Eigen::Index cols() const { return X.cols(); }
    // Memoria ocupada por los datos (bytes)
    std::size_t bytes() const {
        return static_cast<std::size_t>(X.size() + y.size()) * sizeof(float);
    }
};

// Lee un archivo con cabecera int32 (filas, columnas) + float32 row-major.
// Se lee directamente sobre el buffer de la matriz (sin copias intermedias).
RowMatrix load_matrix(const std::string& path);

// Carga X_<split>.bin y y_<split>.bin desde el directorio indicado.
Dataset load_dataset(const std::string& dir, const std::string& split);

}  // namespace logreg
