#pragma once
// ============================================================================
// metrics.hpp - Metricas de clasificacion binaria.
// ============================================================================
#include "dataset.hpp"

namespace logreg {

struct Metrics {
    long tp = 0, tn = 0, fp = 0, fn = 0;
    double accuracy = 0, precision = 0, recall = 0, f1 = 0;
};

// y_true e y_pred contienen 0/1. Se pasan por referencia constante.
Metrics compute_metrics(const Eigen::Ref<const Vector>& y_true,
                        const Eigen::Ref<const Vector>& y_pred);

}  // namespace logreg
