// ============================================================================
// logistic_regression.cpp - Implementacion del motor matematico con Eigen.
// ============================================================================
#include "logistic_regression.hpp"

#include <algorithm>
#include <cmath>

namespace logreg {

LogisticRegression::LogisticRegression(const Hyperparams& hp) : hp_(hp) {}

void LogisticRegression::sigmoid_inplace(Eigen::Ref<Vector> z) {
    // 1/(1+e^{-z}). Para z muy negativo e^{-z} -> inf y el resultado -> 0
    // (no hay NaN en float IEEE-754). Eigen vectoriza esta expresion con SIMD.
    z = (1.0f + (-z.array()).exp()).inverse().matrix();
}

void LogisticRegression::step(const Eigen::Ref<const RowMatrix>& Xb,
                              const Eigen::Ref<const Vector>& yb) {
    const Eigen::Index m = Xb.rows();
    // Vistas sobre los buffers preasignados (head() no reserva memoria)
    auto z = z_buf_.head(m);

    // Propagacion: z = Xb * w + b   (.noalias(): escribir directo en z, sin temporal)
    z.noalias() = Xb * w_;
    z.array() += b_;
    sigmoid_inplace(z);           // z <- p
    z -= yb;                      // z <- p - y   (error)

    // Gradiente: Xb^T (p - y) / m + lambda w
    grad_buf_.noalias() = Xb.transpose() * z;
    grad_buf_ *= 1.0f / static_cast<float>(m);
    grad_buf_ += hp_.l2 * w_;
    const float grad_b = z.mean();

    // Actualizacion de parametros
    w_ -= hp_.lr * grad_buf_;
    b_ -= hp_.lr * grad_b;
}

void LogisticRegression::fit(const Eigen::Ref<const RowMatrix>& X,
                             const Eigen::Ref<const Vector>& y) {
    const Eigen::Index n = X.rows();
    const Eigen::Index d = X.cols();
    const Eigen::Index bs = (hp_.batch <= 0) ? n : std::min<Eigen::Index>(hp_.batch, n);

    w_.setZero(d);
    b_ = 0.0f;
    z_buf_.resize(bs);
    grad_buf_.resize(d);
    loss_history_.clear();
    loss_history_.reserve(static_cast<std::size_t>(hp_.epochs));

    for (int epoch = 0; epoch < hp_.epochs; ++epoch) {
        for (Eigen::Index start = 0; start < n; start += bs) {
            const Eigen::Index m = std::min(bs, n - start);
            // middleRows devuelve un BLOQUE (vista) de filas consecutivas: en
            // row-major es un trozo contiguo de memoria -> no hay copia.
            step(X.middleRows(start, m), y.segment(start, m));
        }
        if (hp_.track_loss) loss_history_.push_back(loss(X, y));
    }
}

Vector LogisticRegression::predict_proba(const Eigen::Ref<const RowMatrix>& X) const {
    Vector p = X * w_;
    p.array() += b_;
    sigmoid_inplace(p);
    return p;
}

Vector LogisticRegression::predict(const Eigen::Ref<const RowMatrix>& X, float threshold) const {
    return (predict_proba(X).array() >= threshold).cast<float>().matrix();
}

float LogisticRegression::loss(const Eigen::Ref<const RowMatrix>& X,
                               const Eigen::Ref<const Vector>& y) const {
    constexpr float eps = 1e-7f;
    const Eigen::ArrayXf p = predict_proba(X).array().max(eps).min(1.0f - eps);
    const float ce = -(y.array() * p.log() + (1.0f - y.array()) * (1.0f - p).log()).mean();
    return ce + 0.5f * hp_.l2 * w_.squaredNorm();
}

}  // namespace logreg
