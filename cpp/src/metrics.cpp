// ============================================================================
// metrics.cpp
// ============================================================================
#include "metrics.hpp"

namespace logreg {

Metrics compute_metrics(const Eigen::Ref<const Vector>& y_true,
                        const Eigen::Ref<const Vector>& y_pred) {
    Metrics m;
    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        const bool t = y_true[i] > 0.5f;
        const bool p = y_pred[i] > 0.5f;
        if (t && p) ++m.tp;
        else if (!t && !p) ++m.tn;
        else if (!t && p) ++m.fp;
        else ++m.fn;
    }
    const double n = static_cast<double>(y_true.size());
    m.accuracy  = (m.tp + m.tn) / n;
    m.precision = (m.tp + m.fp) ? static_cast<double>(m.tp) / (m.tp + m.fp) : 0.0;
    m.recall    = (m.tp + m.fn) ? static_cast<double>(m.tp) / (m.tp + m.fn) : 0.0;
    m.f1 = (m.precision + m.recall) > 0
               ? 2 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
    return m;
}

}  // namespace logreg
