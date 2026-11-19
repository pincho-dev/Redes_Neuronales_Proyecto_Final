// ============================================================================
// utils.cpp
// ============================================================================
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

namespace logreg {

static double read_status_kb(const std::string& key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key + ":", 0) == 0) {
            std::istringstream iss(line.substr(key.size() + 1));
            double kb = 0;
            iss >> kb;
            return kb;
        }
    }
    return 0.0;
}

double current_rss_mb() { return read_status_kb("VmRSS") / 1024.0; }
double peak_rss_mb()    { return read_status_kb("VmHWM") / 1024.0; }

Stats stats(const std::vector<double>& v) {
    Stats s{0, 0, 0, 0};
    if (v.empty()) return s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.stdev = std::sqrt(acc / v.size());
    std::vector<double> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    s.min = sorted.front();
    s.median = sorted[sorted.size() / 2];
    return s;
}

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> out;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) continue;
        a = a.substr(2);
        const auto eq = a.find('=');
        if (eq != std::string::npos) {
            out[a.substr(0, eq)] = a.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            out[a] = argv[++i];
        } else {
            out[a] = "1";
        }
    }
    return out;
}

std::string get_arg(const std::map<std::string, std::string>& a, const std::string& k,
                    const std::string& def) {
    const auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

}  // namespace logreg
