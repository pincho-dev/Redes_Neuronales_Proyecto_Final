#pragma once
// ============================================================================
// utils.hpp - Medicion de tiempo y memoria, estadisticas simples y argumentos.
// ============================================================================
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace logreg {

// Cronometro de alta resolucion (reloj monotono)
class Timer {
public:
    Timer() : t0_(std::chrono::steady_clock::now()) {}
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

// Memoria del proceso leida de /proc/self/status (Linux), en MB.
//   VmRSS: memoria residente actual.  VmHWM: pico de memoria residente.
double current_rss_mb();
double peak_rss_mb();

struct Stats { double mean, stdev, min, median; };
Stats stats(const std::vector<double>& v);

// Parser minimo de argumentos "--clave=valor" o "--clave valor"
std::map<std::string, std::string> parse_args(int argc, char** argv);
std::string get_arg(const std::map<std::string, std::string>& a, const std::string& k,
                    const std::string& def);

// Evita que el compilador elimine calculos cuyo resultado no se usa (benchmarks)
template <class T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

}  // namespace logreg
