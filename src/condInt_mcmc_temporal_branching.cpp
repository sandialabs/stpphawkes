#include <RcppArmadillo.h>

#include <Rcpp.h>
// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

// Protect against compilers without OpenMP
#ifdef _OPENMP
// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

#include "helper_functions.h"
#include "temporal_common.h"

using namespace Rcpp;

// Bayesian Estimation of Temporal Hawkes Model with Missing Data using branching structure
// [[Rcpp::export]]
DataFrame condInt_mcmc_temporal_branching(std::vector<double> ti, double t_maxi, std::vector<int> y_init,
                                          double mu_init, double alpha_init, double beta_init,
                                          std::vector<double> mu_parami, std::vector<double> alpha_parami, std::vector<double> beta_parami,
                                          double sig_betai, int n_mcmc, int n_burn, bool print) {
    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    // initialize parameters
    double t_max = t_maxi;
    std::vector<int> y_curr = y_init;
    double mu_curr = mu_init;
    double alpha_curr = alpha_init;
    double beta_curr = beta_init;
    std::vector<double> mu_param = mu_parami;
    std::vector<double> alpha_param = alpha_parami;
    std::vector<double> beta_param = beta_parami;
    double sig_beta = sig_betai;
    std::vector<double> t = ti;
    arma::vec mu_samps(n_mcmc);
    arma::vec alpha_samps(n_mcmc);
    arma::vec beta_samps(n_mcmc);

    int n = t.size();
    std::vector<double> z;
    z.reserve(n);

    std::vector<int> numtriggered;
    numtriggered.resize(n);

    int numbackground;  

    // begin mcmc
    Progress p(n_mcmc, print);
    for (int iter = 0; iter < n_mcmc; iter++) {
        if (Progress::check_abort())
            return -1.0;
        y_curr = temporal::sample_y(alpha_curr, beta_curr, mu_curr, t);

        z.clear();
        std::fill(numtriggered.begin(), numtriggered.end(), 0);
        numbackground = 0;
        for (int i = 0; i < n; i++) {
            if (y_curr[i] > 0) {
                numtriggered[y_curr[i] - 1]++;
                z.push_back(t[i] - t[y_curr[i] - 1]);
            } else {
                numbackground++;
            }
        }

        mu_curr = temporal::sample_mu(t_max, numbackground, mu_param);
        alpha_curr = temporal::sample_alpha(t, z.size(), t_max, beta_curr, alpha_param[0], alpha_param[1]);
        beta_curr = temporal::sample_beta(alpha_curr, beta_curr, t_max, sig_beta, t, beta_param, z);
        mu_samps(iter) = mu_curr;
        alpha_samps(iter) = alpha_curr;
        beta_samps(iter) = beta_curr;
        p.increment();  // update progress
    }

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec alpha_sampso = alpha_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec beta_sampso = beta_samps.subvec(n_burn, n_mcmc - 1);

    DataFrame df = DataFrame::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("alpha") = alpha_sampso,
                                     Rcpp::Named("beta") = beta_sampso);

    return (df);
}
