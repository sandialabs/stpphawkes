#ifndef STPP_TEMPORAL_COMMON_H
#define STPP_TEMPORAL_COMMON_H

#include <RcppArmadillo.h>

#include <Rcpp.h>

#include "helper_functions.h"
namespace temporal {

/**
 * @brief Computes the log likelihood for temporal missing data
 *
 */
double temporalLogLikelihood(const std::vector<double>& t, double mu, double alpha, double beta, double t_max);

double beta_posterior(const std::vector<double>& t, double t_max, double alpha, double beta,
                      const std::vector<double>& beta_param, const std::vector<double>& z);

std::vector<int> sample_y(double alpha_curr, double beta_curr, double mu_curr, const std::vector<double>& t);

double sample_mu(double t_max, int numbackground, const std::vector<double>& mu_param);

double sample_alpha(const std::vector<double>& t, const int sum_numtriggered, double t_max, double beta_curr,
                    double alpha_a, double alpha_b);

double sample_beta(double alpha_curr, double beta_curr, double t_max, double sig_beta, const std::vector<double>& t,
                   const std::vector<double>& beta_param, const std::vector<double>& z);

std::pair<int, std::vector<int>> calculateNumTriggered(const std::vector<double>& t, const std::vector<int>& y_curr,
                                                       std::vector<double>& z);
/**
 * @brief Simulates time values for missing time intervals
 *
 * @param[in] times is a vector of known times
 * @param[in] t_missing is a matrix characterizing missing time gap intervals
 * @param[in] mu_curr is current value estimated for mu
 * @param[in] alpha_curr is current value estimated for alpha
 * @param[in] beta_curr is current value estimated for beta
 *
 * @return Returns a vector of vectors, where each sub vector contains simulated times for
 * a corresponding time interval gap specified from a row of t_missing
 *
 */
std::vector<std::vector<double>> simulateMissingTimes(const std::vector<double>& times, const arma::mat& t_missing,
                                                      const double mu_curr, const double alpha_curr,
                                                      const double beta_curr);
}
#endif  // STPP_TEMPORAL_COMMON_H
