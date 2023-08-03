#include <RcppArmadillo.h>
#include "areapl.h"
#include "utilities.h"
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

// Global Variables
static double t_max, mu_curr, a_curr, b_curr, sig_curr, W;
static double sig_mu, sig_a, sig_b, sig_sig;
static arma::vec x, y, t;

// Function declarations
static double beta_tk(double t, double b) {
    double out;
    if ((t >= 0) & (b > 0)) {
        out = b * exp(-b * t);
    } else {
        out = 0.0;
    }

    return (out);
}

static double Beta_tk(double t, double b) {
    double out;
    out = 1 - exp(-b * t);
    return (out);
}

static double alpha_k(double x, double y, double sig) {
    double out;
    out = 1 / sqrt(2 * M_PI * pow(sig, 2)) * exp(-pow(x, 2) / (2 * pow(sig, 2))) * 1 / sqrt(2 * M_PI * pow(sig, 2)) *
          exp(-pow(y, 2) / (2 * pow(sig, 2)));
    return (out);
}

static double log_lambda_str(arma::vec& t, double mu, double a, double b) {
    int n = t.n_elem;
    int n1;
    arma::vec out(n);
    double tmp;
    arma::uvec ind;

    if ((mu > 0) & (a > 0) & (b > 0)) {
#ifdef _OPENMP
#pragma omp parallel for shared(t, out, mu, a, b, n) private(ind, n1, tmp) default(none) schedule(auto)
#else
#endif
        for (int i = 0; i < n; i++) {
            ind = find(t < t(i));
            n1 = ind.n_elem;
            if (n1 > 0) {
                tmp = 0.0;
                for (int j = 0; j < n1; j++) {
                    tmp += beta_tk(t(i) - t(ind(j)), b);
                }
                out(i) = log(mu + a * tmp);
            } else {
                out(i) = log(mu);
            }
        }
    } else {
        out(0) = -INFINITY;
    }

    return (sum(out));
}

static double log_gamma_str(arma::vec& x, arma::vec& y, arma::vec& t, double mu, double a, double b, double sig,
                            double W) {
    int n = t.n_elem;
    int n1;
    double tmp;
    arma::vec out(n);
    arma::uvec ind;

    if ((mu > 0) & (a > 0) & (b > 0)) {
#ifdef _OPENMP
#pragma omp parallel for shared(x, y, t, out, mu, a, b, sig, n, W) private(ind, n1, tmp) default(none) schedule(auto)
#else
#endif
        for (int i = 0; i < n; i++) {
            ind = find(t < t(i));
            n1 = ind.n_elem;
            if (n1 > 0) {
                tmp = 0.0;
                for (int j = 0; j < n1; j++) {
                    tmp += beta_tk(t(i) - t(ind(j)), b) * alpha_k(x(i) - x(ind(j)), y(i) - y(ind(j)), sig);
                }
                out(i) = log(mu / W + a * tmp);
            } else {
                out(i) = log(mu / W);
            }
        }
    } else {
        out(0) = -INFINITY;
    }

    return (sum(out));
}

static double log_prior(double mu, double a, double b, double sig) {
    double out = 0.0;

    out += R::dexp(mu, 1 / 0.01, 1);
    out += R::dexp(a, 1 / 0.01, 1);
    out += R::dexp(b, 1 / 0.01, 1);
    out += R::dexp(sig, 1 / 0.01, 1);

    return (out);
}

static double sample_mu() {
    double mu_prop = R::rnorm(mu_curr, sig_mu);
    double top, bottom, rat;
    top = log_prior(mu_prop, a_curr, b_curr, sig_curr) + log_lambda_str(t, mu_prop, a_curr, b_curr) - mu_prop * t_max;
    bottom =
        log_prior(mu_curr, a_curr, b_curr, sig_curr) + log_lambda_str(t, mu_curr, a_curr, b_curr) - mu_curr * t_max;
    rat = exp(top - bottom);
    if (!std::isnan(rat)) {
        double U = R::runif(0, 1);
        if (U < rat) {
            mu_curr = mu_prop;
        }
    }

    return (mu_curr);
}

static double sample_a() {
    double a_prop = R::rnorm(a_curr, sig_a);
    double tmp1 = 0.0;
    int n = t.n_elem;
    for (int i = 0; i < n; i++) {
        tmp1 += Beta_tk(t_max - t[i], b_curr);
    }
    double top, bottom, rat;
    top = log_prior(mu_curr, a_prop, b_curr, sig_curr) + log_lambda_str(t, mu_curr, a_prop, b_curr) - a_prop * tmp1;
    bottom = log_prior(mu_curr, a_curr, b_curr, sig_curr) + log_lambda_str(t, mu_curr, a_curr, b_curr) - a_curr * tmp1;
    rat = exp(top - bottom);
    if (!std::isnan(rat)) {
        double U = R::runif(0, 1);
        if (U < rat) {
            a_curr = a_prop;
        }
    }

    return (a_curr);
}

static double sample_b() {
    double b_prop = R::rnorm(b_curr, sig_b);
    double top, bottom, rat;
    double tmp1 = 0.0;
    int n = t.n_elem;
    for (int i = 0; i < n; i++) {
        tmp1 += Beta_tk(t_max - t[i], b_prop);
    }
    double tmp2 = 0.0;
    for (int i = 0; i < n; i++) {
        tmp2 += Beta_tk(t_max - t[i], b_curr);
    }
    top = log_prior(mu_curr, a_curr, b_prop, sig_curr) + log_lambda_str(t, mu_curr, a_curr, b_prop) - a_curr * tmp1;
    bottom = log_prior(mu_curr, a_curr, b_curr, sig_curr) + log_lambda_str(t, mu_curr, a_curr, b_curr) - a_curr * tmp2;
    rat = exp(top - bottom);
    if (!std::isnan(rat)) {
        double U = R::runif(0, 1);
        if (U < rat) {
            b_curr = b_prop;
        }
    }

    return (b_curr);
}

static double sample_sig() {
    double sig_prop = R::rnorm(sig_curr, sig_sig);
    double top, bottom, rat;
    top = log_prior(mu_curr, a_curr, b_curr, sig_prop) + log_gamma_str(x, y, t, mu_curr, a_curr, b_curr, sig_prop, W);
    bottom =
        log_prior(mu_curr, a_curr, b_curr, sig_curr) + log_gamma_str(x, y, t, mu_curr, a_curr, b_curr, sig_curr, W);
    rat = exp(top - bottom);
    if (!std::isnan(rat)) {
        double U = R::runif(0, 1);
        if (U < rat) {
            sig_curr = sig_prop;
        }
    }

    return (sig_curr);
}

// Bayesian Estimation of STPP Hawkes Model
// [[Rcpp::export]]
DataFrame condInt_mcmc_stpp(DataFrame data, double t_maxi, double mu_init, double a_init, double b_init,
                            double sig_init, arma::mat poly, double sig_mui, double sig_ai, double sig_bi,
                            double sig_sigi, int n_mcmc, int n_burn, bool print) {
    x = as<arma::vec>(data["x"]);
    y = as<arma::vec>(data["y"]);
    t = as<arma::vec>(data["t"]);

    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    // initialize parameters
    W = areapl(poly);
    t_max = t_maxi;
    mu_curr = mu_init;
    a_curr = a_init;
    b_curr = b_init;
    sig_curr = sig_init;
    sig_mu = sig_mui;
    sig_a = sig_ai;
    sig_b = sig_bi;
    sig_sig = sig_sigi;
    arma::vec mu_samps(n_mcmc);
    arma::vec a_samps(n_mcmc);
    arma::vec b_samps(n_mcmc);
    arma::vec sig_samps(n_mcmc);

    // begin mcmc
    Progress p(n_mcmc, print);
    for (int iter = 0; iter < n_mcmc; iter++) {
        if (Progress::check_abort())
            return -1.0;
        mu_curr = sample_mu();
        a_curr = sample_a();
        b_curr = sample_b();
        sig_curr = sample_sig();
        mu_samps(iter) = mu_curr;
        a_samps(iter) = a_curr;
        b_samps(iter) = b_curr;
        sig_samps(iter) = sig_curr;
        p.increment();  // update progress
    }

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec a_sampso = a_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec b_sampso = b_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec sig_sampso = sig_samps.subvec(n_burn, n_mcmc - 1);

    DataFrame df = DataFrame::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("a") = a_sampso,
                                     Rcpp::Named("b") = b_sampso, Rcpp::Named("sigma") = sig_sampso);

    return (df);
}
