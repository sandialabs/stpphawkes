#ifndef STPP_CONDINT_MCMC_TEMPORAL_BRANCHING_MD_H
#define STPP_CONDINT_MCMC_TEMPORAL_BRANCHING_MD_H

#include <RcppArmadillo.h>
#include <vector>

std::tuple<arma::vec, arma::vec, arma::vec, arma::vec> test_method(
    const std::vector<double>& ti, const arma::mat& t_misi, double t_maxi, const std::vector<int>& y_init,
    double mu_init, double alpha_init, double beta_init, const std::vector<double>& mu_parami,const std::vector<double>& alpha_parami,const std::vector<double>& beta_parami,
    double sig_betai, int n_mcmc, int n_burn, bool print);

#endif  // STPP_CONDINT_MCMC_TEMPORAL_BRANCHING_MD_H
