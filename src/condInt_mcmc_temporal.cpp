#include <RcppArmadillo.h>

#include <Rcpp.h>

// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

// Protect against compilers without OpenMP
#ifdef _OPENMP
// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// Global Variables
static double t_max, mu_curr, alpha_curr, beta_curr;
static double sig_mu, sig_alpha, sig_beta;
static arma::vec t;

// Function declarations
static double beta_tk(double t, double beta) {
    double out;
    if ((t >= 0) & (beta > 0)) {
        out = beta * exp(-beta * t);
    } else {
        out = 0.0;
    }

    return (out);
}

static double Beta_tk(double t, double beta) {
    double out;
    out = 1 - exp(-beta * t);
    return (out);
}

static arma::vec lambda_str(arma::vec& t, double mu, double alpha, double beta) {
    int n = t.n_elem;
    int n1;
    arma::vec out(n);
    double tmp;
    arma::uvec ind;

#ifdef _OPENMP
#pragma omp parallel for shared(t, out, mu, alpha, beta, n) private(ind, n1, tmp) default(none) schedule(auto)
#else
#endif
    for (int i = 0; i < n; i++) {
        ind = find(t < t(i));
        n1 = ind.n_elem;
        if (n1 > 0) {
            tmp = 0.0;
            for (int j = 0; j < n1; j++) {
                tmp += beta_tk(t(i) - t(ind(j)), beta);
            }
            out(i) = mu + alpha * tmp;
        } else {
            out(i) = mu;
        }
    }

    return (out);
}

static double log_lik(arma::vec& t, double mu, double alpha, double beta, double t_max) {
    int n = t.n_elem;
    double M_T, lambda_T;
    double tmp, s_log_lambda_str;
    arma::vec loglik(n);

    if ((mu > 0) & (alpha > 0) & (beta > 0)) {
        M_T = mu * t_max;
        tmp = 0.0;
        for (int i = 0; i < n; i++) {
            tmp += Beta_tk(t_max - t[i], beta);
        }
        lambda_T = M_T + alpha * tmp;
        loglik = log(lambda_str(t, mu, alpha, beta));
        s_log_lambda_str = sum(loglik);
        s_log_lambda_str -= lambda_T;
    } else {
        s_log_lambda_str = -INFINITY;
    }

    return (s_log_lambda_str);
}

static double log_prior(double mu, double alpha, double beta) {
    double out = 0.0;

    out += R::dexp(mu, 1 / 0.01, 1);
    out += R::dexp(alpha, 1 / 0.01, 1);
    out += R::dexp(beta, 1 / 0.01, 1);

    return (out);
}

static double sample_mu() {
    double mu_prop = R::rnorm(mu_curr, sig_mu);
    double top, bottom, rat;
    top = log_prior(mu_prop, alpha_curr, beta_curr) + log_lik(t, mu_prop, alpha_curr, beta_curr, t_max);
    bottom = log_prior(mu_curr, alpha_curr, beta_curr) + log_lik(t, mu_curr, alpha_curr, beta_curr, t_max);
    rat = exp(top - bottom);
    double U = R::runif(0, 1);
    if (U < rat) {
        mu_curr = mu_prop;
    }

    return (mu_curr);
}

static double sample_alpha() {
    double alpha_prop = R::rnorm(alpha_curr, sig_alpha);
    double top, bottom, rat;
    top = log_prior(mu_curr, alpha_prop, beta_curr) + log_lik(t, mu_curr, alpha_prop, beta_curr, t_max);
    bottom = log_prior(mu_curr, alpha_curr, beta_curr) + log_lik(t, mu_curr, alpha_curr, beta_curr, t_max);
    rat = exp(top - bottom);
    double U = R::runif(0, 1);
    if (U < rat) {
        alpha_curr = alpha_prop;
    }

    return (alpha_curr);
}

static double sample_beta() {
    double beta_prop = R::rnorm(beta_curr, sig_beta);
    double top, bottom, rat;
    top = log_prior(mu_curr, alpha_curr, beta_prop) + log_lik(t, mu_curr, alpha_curr, beta_prop, t_max);
    bottom = log_prior(mu_curr, alpha_curr, beta_curr) + log_lik(t, mu_curr, alpha_curr, beta_curr, t_max);
    rat = exp(top - bottom);
    double U = R::runif(0, 1);
    if (U < rat) {
        beta_curr = beta_prop;
    }

    return (beta_curr);
}

// Bayesian Estimation of Temporal Hawkes Model with Missing Data
// [[Rcpp::export]]
DataFrame condInt_mcmc_temporal(arma::vec ti, double t_maxi, double mu_init, double alpha_init, double beta_init,
                                double sig_mui, double sig_alphai, double sig_betai, int n_mcmc, int n_burn,
                                bool print) {
    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    // initialize parameters
    t_max = t_maxi;
    mu_curr = mu_init;
    alpha_curr = alpha_init;
    beta_curr = beta_init;
    sig_mu = sig_mui;
    sig_alpha = sig_alphai;
    sig_beta = sig_betai;
    t = ti;
    arma::vec mu_samps(n_mcmc);
    arma::vec alpha_samps(n_mcmc);
    arma::vec beta_samps(n_mcmc);

    // begin mcmc
    Progress p(n_mcmc, print);
    for (int iter = 0; iter < n_mcmc; iter++) {
        if (Progress::check_abort())
            return -1.0;
        mu_curr = sample_mu();
        alpha_curr = sample_alpha();
        beta_curr = sample_beta();
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
