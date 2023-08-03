#include <cmath>
#include <vector>

#include "helper_functions.h"

namespace stpp {

constexpr double one_over_pi() { return 1 / M_PI; }

inline double gamma_k(const double x, const double y, const double sig) {
    const double one_over_two_sig = .5 * 1.0 / sig;

    return one_over_pi() * one_over_two_sig * std::exp(-(x * x + y * y) * one_over_two_sig);
}

inline double sample_mu(const double t_max, const int numbackground,
                        const std::vector<double>& mu_param) {
    auto gen = GenerateMersenneTwister();
    std::gamma_distribution<> rgamma(mu_param[0] + numbackground, 1 / (mu_param[1] + t_max));
    return rgamma(gen);
}

/**
 * @brief Helper function for sample_a that does the deterministic for loops and math
 *
 * @note Does not contain the random sampling step
 */
double sample_a_accumulate(const std::vector<double>& t, double t_max, double b_curr);

double sample_a(const std::vector<double>& t, const std::vector<double>& z_t, double t_max, const double a_curr,
                double b_curr, const std::vector<double>& a_param);

double sample_b(const std::vector<double>& t, const std::vector<double>& z_t, const double t_max, const double a_curr,
                double b_curr, const double sig_b, const std::vector<double>& b_param);

std::vector<int> sample_y(const std::vector<double>& t, const std::vector<double>& x, const std::vector<double>& y,
                          const double mu_curr, const double a_curr, const double b_curr, const double sig_curr, arma::mat poly);

double b_posterior(const std::vector<double>& t, double t_max, double a, double b, const std::vector<double>& z_t,
                   const std::vector<double>& b_param);

double sig_posterior(double sig, const std::vector<double>& z_x, const std::vector<double>& z_y,
                     const std::vector<double>& sig_param);

double sample_sig(const std::vector<double>& z_x, const std::vector<double>& z_y, double sig_curr, const double sig_sig,
                  const std::vector<double>& sig_param);

double sample_sig_gibbs(const std::vector<double>& z_x, const std::vector<double>& z_y, double sig_curr,
                  const std::vector<double>& sig_param);

namespace missing_data {

double log_lik(std::vector<double>& x, std::vector<double>& y, std::vector<double>& t, double mu, double a, double b,
               double sig, double t_max, arma::mat poly);
}
}
