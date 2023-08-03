#include "temporal_common.h"

#include "simulate_temporal_hawkes.h"

namespace temporal {
double temporalLogLikelihood(const std::vector<double>& t, double mu, double alpha, double beta, double t_max) {
    int n = t.size();

    double part1 = 0;

    const double alpha_beta_product = alpha * beta;

    std::vector<size_t> min_is(n);
    min_is[0] = 0;

    for (int i = 1; i < n; ++i) {
        double minimum_time = t[i] - 36 / beta;

        if (minimum_time < 0) {
            min_is[i] = 0;
        } else {
            int min_i;
            for (min_i = min_is[i - 1]; min_i < i; ++min_i) {
                if (t[min_i] > minimum_time) {
                    break;
                }
            }

            min_is[i] = min_i;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for shared(t, mu, alpha, beta, n, min_is) reduction(+ : part1)
#else
#endif
    for (int i = 1; i < n; i++) {
        double temp = 0;
        const double t_i = t[i];
        for (int j = min_is[i]; j < i; ++j) {
            temp += std::exp(-beta * (t_i - t[j]));
        }
        part1 += std::log(alpha_beta_product * temp + mu);
    }
    part1 += std::log(mu);

    double part2 = mu * t_max;

    double part3 = 0;

    double epsilon = 1e-15;

    double minimum_time = std::log(epsilon) / beta + t_max;
    size_t min_i;
    for (min_i = n - 1; min_i >= 0; --min_i) {
        if (t[min_i] < minimum_time) {
            break;
        }
    }

// Find first index of time walking backwards where this minimum time occurs;

#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, alpha, beta, n) reduction(+ : part3)
#else
#endif
    for (int i = min_i + 1; i < n; i++) {
        part3 += alpha * Beta_tk(t_max - t[i], beta);
    }
    part3 += min_i * alpha;

    return (part1 - part2 - part3);
}

double beta_posterior(const std::vector<double>& t, double t_max, double alpha, double beta,
                      const std::vector<double>& beta_param, const std::vector<double>& z) {
    // The overall equation we are evaluating here is:
    // loglik = alpha*(sum exp(-beta*(t_max-t_i)) -N)  - beta* sum( z) + size(z)* log(beta)

    if (alpha < beta) {
        int n = t.size();

        double epsilon = t_max + 1 / beta * (-36.0 - std::log(alpha));

        int min_i = findMinimumRelevantTime(t, epsilon);

        double loglik = 0;

        for (int i = n - 1; i >= min_i; --i) {
            loglik += std::exp(beta * (t[i] - t_max));
        }

        loglik -= n;
        loglik *= alpha;

        double loglik2 = 0;
        // OpenMP seems to be overkill for this loop
        for (const auto zval : z) {
            loglik2 -= zval;
        }

        loglik2 *= beta;

        loglik += z.size() * std::log(beta) + loglik2;
        loglik += (beta_param[0] - 1) * std::log(beta) - beta * beta_param[1];

        return loglik;
    } else {
        return -INFINITY;
    }
}

double sample_mu(double t_max, int numbackground, const std::vector<double>& mu_param) {
    auto gen = GenerateMersenneTwister();

    std::gamma_distribution<> rgamma(mu_param[0] + numbackground, 1 / (mu_param[1] + t_max));
    return rgamma(gen);
}

/**
 * @brief Combines time vector with simulated temporal vector and calls sample_y
 */
std::vector<int> sample_y(double alpha_curr, double beta_curr, double mu_curr, const std::vector<double>& t_tmp) {
    int n = t_tmp.size();
    std::vector<int> y_curr(n);
    y_curr[0] = 0;

    std::vector<size_t> min_is(n);
    min_is[0] = 0;
    double alpha_beta_product = alpha_curr * beta_curr;
    double log_alpha_beta_product = std::log(alpha_beta_product);

    for (int i = 1; i < n; ++i) {
        double minimum_time = t_tmp[i] - 30 / beta_curr - log_alpha_beta_product / beta_curr;

        if (minimum_time < 0) {
            min_is[i] = 0;
        } else {
            int min_i;
            for (min_i = min_is[i - 1]; min_i < i; ++min_i) {
                if (t_tmp[min_i] > minimum_time) {
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

#ifdef _OPENMP
#pragma omp parallel for
#else
#endif
    for (int i = 1; i < n; i++) {
        auto gen = GenerateMersenneTwister();
        std::vector<double> probs(i + 1, 0);

        probs[0] = mu_curr;

        double t_i = t_tmp[i];
        for (int j = min_is[i]; j < i; ++j) {
            double temp = alpha_beta_product * std::exp(-beta_curr * (t_i - t_tmp[j]));
            probs[j + 1] = temp;
        }

        std::discrete_distribution<> d(probs.begin(), probs.end());

        int parent = d(gen);
        y_curr[i] = parent;
    }

    return y_curr;
}

double sample_alpha(const std::vector<double>& t, const int sum_numtriggered, const double t_max,
                    const double beta_curr, const double alpha_a, const double alpha_b) {
    double exponential_sum = 0;
    for (const auto t_i : t) {
        exponential_sum += Beta_tk(t_max - t_i, beta_curr);
    }

    // Compute Gamma(sum_numtriggered + alpha_a, sum(Beta_tk(t_max-t,beta)) + alpha_b)
    auto gen = GenerateMersenneTwister();

    std::gamma_distribution<> rgamma(sum_numtriggered + alpha_a, 1.0 / (exponential_sum + alpha_b));

    double alpha_curr = 0.0;
    bool a_constraint = true;
    while (a_constraint) {
        alpha_curr = rgamma(gen);
        if (alpha_curr < 1 && alpha_curr < beta_curr) {
            a_constraint = false;
        }
    }
    return alpha_curr;
}

double sample_beta(double alpha_curr, double beta_curr, double t_max, double sig_beta, const std::vector<double>& t,
                   const std::vector<double>& beta_param, const std::vector<double>& z) {
    auto gen = GenerateMersenneTwister();

    std::normal_distribution<> rnorm(0, sig_beta);
    std::uniform_real_distribution<> runif(0, 1);

    double bottom = beta_posterior(t, t_max, alpha_curr, beta_curr, beta_param, z);

    for (size_t i = 0; i < 100; i++) {
        double beta_prop = beta_curr + rnorm(gen);
        double top = beta_posterior(t, t_max, alpha_curr, beta_prop, beta_param, z);
        if (runif(gen) < std::exp(top - bottom)) {
            beta_curr = beta_prop;
            bottom = top;
        }
    }

    return beta_curr;
}

std::vector<std::vector<double>> simulateMissingTimes(const std::vector<double>& times, const arma::mat& t_missing,
                                                      const double mu_curr, const double alpha_curr,
                                                      const double beta_curr) {
    // Vector of vectors where each subvector contains a sampled set of missing times
    // There is one vector per missing time interval, hence z_currs.size() = n_mis
    std::vector<std::vector<double>> z_currs;
    int n_mis = t_missing.n_rows;
    z_currs.resize(n_mis);

    std::vector<double> z_currt;
    std::vector<double> z_curr;  // All z sampled times in one single vector
    int cnt = 0;
    for (auto& j : z_currs) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_missing(cnt, 0);
        t_mis1(1) = t_missing(cnt, 1);
        arma::vec z_curr1 = simulate_temporal(mu_curr, alpha_curr, beta_curr, t_mis1, times);
        j = arma::conv_to<std::vector<double>>::from(z_curr1);

        z_curr.insert(z_curr.end(), j.begin(), j.end());
        cnt++;
    }

    return z_currs;
}

std::pair<int, std::vector<int>> calculateNumTriggered(const std::vector<double>& t, const std::vector<int>& y_curr,
                                                       std::vector<double>& z) {
    std::vector<int> numtriggered(t.size(), 0);

    z.clear();
    int numbackground = 0;
    for (size_t i = 0; i < t.size(); i++) {
        if (y_curr[i] > 0) {
            numtriggered[y_curr[i] - 1]++;
            z.push_back(t[i] - t[y_curr[i] - 1]);
        } else {
            numbackground++;
        }
    }

    return std::make_pair(numbackground, std::move(numtriggered));
}

}  // end namespace temporal
