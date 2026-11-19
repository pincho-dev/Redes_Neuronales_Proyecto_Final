#pragma once
// ============================================================================
// logistic_regression.hpp - Motor matematico: regresion logistica binaria
// entrenada con gradiente descendente por mini-lotes y regularizacion L2.
//
//   p      = sigma(X w + b),       sigma(z) = 1 / (1 + e^{-z})
//   L(w,b) = -1/n sum[ y log p + (1-y) log(1-p) ] + (lambda/2) ||w||^2
//   dL/dw  = X^T (p - y) / n + lambda w
//   dL/db  = mean(p - y)
//   w <- w - eta dL/dw ;  b <- b - eta dL/db
// ============================================================================
#include "dataset.hpp"

#include <vector>

namespace logreg {

struct Hyperparams {
    float lr      = 0.5f;     // tasa de aprendizaje (eta)
    int   epochs  = 200;      // pasadas completas sobre los datos
    int   batch   = 256;      // tamano de mini-lote (<=0 => lote completo)
    float l2      = 1e-4f;    // regularizacion L2 (lambda)
    bool  track_loss = true;  // calcular la perdida al final de cada epoca
};

class LogisticRegression {
public:
    explicit LogisticRegression(const Hyperparams& hp = {});

    // Las matrices se reciben por REFERENCIA CONSTANTE (Eigen::Ref): no se copia
    // nada y ademas se aceptan bloques/sub-matrices (p.ej. X.middleRows()).
    void fit(const Eigen::Ref<const RowMatrix>& X, const Eigen::Ref<const Vector>& y);

    Vector predict_proba(const Eigen::Ref<const RowMatrix>& X) const;
    Vector predict(const Eigen::Ref<const RowMatrix>& X, float threshold = 0.5f) const;
    float  loss(const Eigen::Ref<const RowMatrix>& X, const Eigen::Ref<const Vector>& y) const;

    // Acceso de solo lectura a los parametros (referencia, no copia)
    const Vector& weights() const { return w_; }
    float bias() const { return b_; }
    const std::vector<float>& loss_history() const { return loss_history_; }
    const Hyperparams& hyperparams() const { return hp_; }

    // Funcion sigmoide aplicada elemento a elemento (in-place sobre z)
    static void sigmoid_inplace(Eigen::Ref<Vector> z);

private:
    // Un paso de gradiente sobre un mini-lote. Usa los buffers preasignados.
    void step(const Eigen::Ref<const RowMatrix>& Xb, const Eigen::Ref<const Vector>& yb);

    Hyperparams hp_;
    Vector w_;
    float  b_ = 0.0f;
    std::vector<float> loss_history_;

    // Buffers de trabajo reservados UNA vez en fit(): evitan reservar memoria
    // dinamica en cada iteracion del bucle de entrenamiento.
    Vector z_buf_;     // X_b w + b   -> luego p - y
    Vector grad_buf_;  // gradiente de w
};

}  // namespace logreg
