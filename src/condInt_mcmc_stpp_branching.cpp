#include <RcppArmadillo.h>
#include "areapl.h"
#include "utilities.h"
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
#include "spatio_temporal_common.h"

using namespace Rcpp;

// Bayesian Estimation of STPP Hawkes Model using branching structure
// [[Rcpp::export]]
List condInt_mcmc_stpp_branching(DataFrame data, double t_maxi, std::vector<int> y_init, double mu_init, double a_init,
                                 double b_init, double sig_init, arma::mat poly, std::vector<double> mu_parami,
                                 std::vector<double> a_parami, std::vector<double> sig_parami,
                                 std::vector<double> b_parami, double sig_bi, double sig_sigi, int n_mcmc, int n_burn,
                                 bool print) {
    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    std::vector<double> x = as<std::vector<double> >(data["x"]);
    std::vector<double> y = as<std::vector<double> >(data["y"]);
    std::vector<double> t = as<std::vector<double> >(data["t"]);

    double t_max = t_maxi;

    auto y_curr = y_init;
    double mu_curr = mu_init;
    double a_curr = a_init;
    double b_curr = b_init;
    auto mu_param = mu_parami;

    auto a_param = a_parami;
    auto b_param = b_parami;
    auto sig_param = sig_parami;
    double sig_curr = sig_init;
    double sig_b = sig_bi;

    double sig_sig = sig_sigi;

    arma::vec mu_samps(n_mcmc);
    arma::vec a_samps(n_mcmc);
    arma::vec b_samps(n_mcmc);
    arma::vec sig_samps(n_mcmc);

    int n = t.size();
    std::vector<double> z_t;
    std::vector<double> z_x;
    std::vector<double> z_y;
    z_t.reserve(n);
    z_x.reserve(n);
    z_y.reserve(n);

    int numbackground = 0;

    // begin mcmc
    Progress p(n_mcmc, print);
    for (int iter = 0; iter < n_mcmc; iter++) {
        if (Progress::check_abort()) {
            return -1.0;
        }

        y_curr = stpp::sample_y(t, x, y, mu_curr, a_curr, b_curr, sig_curr, poly);

        z_t.clear();
        z_x.clear();
        z_y.clear();

        std::vector<int> numtriggered(n, 0);
        numbackground = 0;
        for (int i = 0; i < n; i++) {
            if (y_curr[i] > 0) {
                numtriggered[y_curr[i] - 1]++;
                z_t.push_back(t[i] - t[y_curr[i] - 1]);
                z_x.push_back(x[i] - x[y_curr[i] - 1]);
                z_y.push_back(y[i] - y[y_curr[i] - 1]);
            } else {
                numbackground++;
            }
        }

        mu_curr = stpp::sample_mu(t_max, numbackground, mu_param);
        a_curr = stpp::sample_a(t, z_t, t_max, a_curr, b_curr, a_param);
        b_curr = stpp::sample_b(t, z_t, t_max, a_curr, b_curr, sig_b, b_param);
        sig_curr = stpp::sample_sig(z_x, z_y, sig_curr, sig_sig, sig_param);

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

    List df = List::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("a") = a_sampso, Rcpp::Named("b") = b_sampso,
                           Rcpp::Named("sigma") = sig_sampso, Rcpp::Named("z") = y_curr);

    return (df);
}
