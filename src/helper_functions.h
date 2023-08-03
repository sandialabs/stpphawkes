#ifndef STPP_HELPER_FUNCTIONS_H
#define STPP_HELPER_FUNCTIONS_H

#include <RcppArmadillo.h>

#include <Rcpp.h>

#include <random>

#ifdef _OPENMP
// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

using namespace Rcpp;

inline double beta_tk(double t, double beta) {
    if ((t >= 0) && (beta > 0)) {
        return beta * std::exp(-beta * t);
    }
    return 0.0;
}

inline double Beta_tk(double t, double beta) { return 1 - std::exp(-beta * t); }

inline std::mt19937 GenerateMersenneTwister() {
#ifndef USE_NO_RANDOM_DEVICE
    std::random_device rd;
    return std::mt19937(rd());
#else
    return std::mt19937(0);
#endif
}

inline std::vector<double> insertSimulatedTimes(const std::vector<double>& t, const std::vector<double>& z_curr) {
    std::vector<double> inserted_times = t;
    inserted_times.insert(inserted_times.end(), z_curr.begin(), z_curr.end());
    std::sort(inserted_times.begin(), inserted_times.end());

    return inserted_times;
}

/**
 * @brief Finds smallest time such that all previous times are a distance far enough away such that exp(-beta*t) <
 * epsilon
 * @param[in] t is a vector of times
 * @param[in] epsilon specifies a time amount that acts as a minimum acceptable time
 *
 * @return Returns an integer index into t such that all j < min_i, t[j] is not relevant for calculations
 */
inline int findMinimumRelevantTime(const std::vector<double>& t, const double epsilon) {
    int min_i;
    int n = t.size();

    for (min_i = n - 1; min_i >= 0; --min_i) {
        if (t[min_i] < epsilon) {
            break;
        }
    }

    return min_i;
}

inline std::vector<int> findMinimumRelevantTimes(const std::vector<double>& t, const double epsilon) {
    int n = t.size();
    std::vector<int> min_is(n);
    min_is[0] = 0;

    for (int i = 1; i < n; ++i) {
        const double minimum_time = t[i] - epsilon;

        if (minimum_time < 0) {
            min_is[i] = 0;
        } else {
            int min_i;
            for (min_i = min_is[i - 1]; min_i < i; ++min_i) {
                if (t[min_i] > minimum_time) {
                    break;
                }
            }
            if (min_i != i) {
                min_is[i] = min_i;
            } else {
                min_is[i] = 0;
            }
        }
    }

    return min_is;
}

#endif  // STPP_HELPER_FUNCTIONS_H
