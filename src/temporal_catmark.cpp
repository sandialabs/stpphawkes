#include <gsl/gsl_randist.h>

#include "helper_functions.h"
#include "temporal_common.h"

#include "temporal_catmark_common.h"
// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::depends(RcppGSL)]]
#include <RcppGSL.h>

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

// Protect against compilers without OpenMP
#ifdef _OPENMP
// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

// Bayesian Estimation of Temporal Hawkes Model with Categorical Marks
// [[Rcpp::export]]
DataFrame CatMarkMcMc(const std::vector<double>& t, const double t_max, const std::vector<int>& marks,
                      const double mu_init, const double alpha_init, const double beta_init,
                      const std::vector<double> mu_params, std::vector<double>& alpha_param,
                      const std::vector<double>& beta_param, const std::vector<double>& p_param, const double sig_beta,
                      const size_t n_mcmc = 1e4, const size_t n_burn = 5e3, bool print = 1) {
    double alpha_a = alpha_param[0];
    double alpha_b = alpha_param[1];

    double beta_a = beta_param[0];
    double beta_b = beta_param[1];

    double mu_curr = mu_init;
    double alpha_curr = alpha_init;
    double beta_curr = beta_init;

    std::vector<int> ntriggered(t.size(), 0);

    size_t kk = p_param.size();

    arma::vec mu_samps(n_mcmc);
    arma::vec alpha_samps(n_mcmc);
    arma::vec beta_samps(n_mcmc);
    arma::vec z_samps(n_mcmc);
    arma::mat p_samps(n_mcmc, kk);

    std::vector<double> z;
    z.reserve(t.size());

    // Begin MCMC
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);  // Arguably, we should std::unique_ptr<> this
    Progress p(n_mcmc, print);
    for (size_t iter = 0; iter < n_mcmc; ++iter) {
        if (Progress::check_abort()) {
            return -1.0;
        }
        auto y_curr = temporal::sample_y(alpha_curr, beta_curr, mu_curr, t);

        int numbackground;
        std::vector<int> numtriggered;
        std::tie(numbackground, numtriggered) = temporal::calculateNumTriggered(t, y_curr, z);

        auto p_curr = catmark::sampleP(marks, p_param, rng);

        mu_curr = temporal::sample_mu(t_max, numbackground, mu_params);

        // Please note z.size() = sum(numtriggered)
        alpha_curr = temporal::sample_alpha(t, z.size(), t_max, beta_curr, alpha_a, alpha_b);

        beta_curr = catmark::sampleBeta(alpha_curr, beta_curr, t_max, sig_beta, t, z, numtriggered, beta_a, beta_b);

        mu_samps(iter) = mu_curr;
        alpha_samps(iter) = alpha_curr;
        beta_samps(iter) = beta_curr;
        p_samps.row(iter) = arma::conv_to<arma::rowvec>::from(p_curr);
        p.increment();  // update progress
    }
    // Release random number generator
    gsl_rng_free(rng);

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec alpha_sampso = alpha_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec beta_sampso = beta_samps.subvec(n_burn, n_mcmc - 1);
    arma::mat p_sampso = p_samps.rows(n_burn, n_mcmc - 1);

    DataFrame df = DataFrame::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("alpha") = alpha_sampso,
                                     Rcpp::Named("beta") = beta_sampso, Rcpp::Named("p") = p_sampso);

    return df;
};
