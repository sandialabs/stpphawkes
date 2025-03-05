#include <RcppArmadillo.h>

#include <Rcpp.h>

#include "helper_functions.h"
#include "simulate_temporal_hawkes.h"
#include "temporal_common.h"
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

using namespace Rcpp;


std::vector<std::vector<double>> sample_z(double alpha_curr, double beta_curr, double mu_curr, double t_max,
                                          const std::vector<double>& t, std::vector<std::vector<double>>& z_currs,
                                          const arma::mat& t_mis) {
    auto gen = GenerateMersenneTwister();

    std::uniform_real_distribution<> runif(0, 1);
    double alpha_str = alpha_curr * beta_curr;

    arma::vec z_prop1;
    std::vector<double> z_prop;
    arma::vec ts = arma::conv_to<arma::vec>::from(t);
    int cnt = 0;
    for (auto& j : z_currs) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_mis(cnt, 0);
        t_mis1(1) = t_mis(cnt, 1);
        if (alpha_str < beta_curr) {
            z_prop1 = simulate_temporal(mu_curr, alpha_curr, beta_curr, t_mis1, ts);
            z_prop = arma::conv_to<std::vector<double>>::from(z_prop1);
        } else {
            z_prop = j;
        }

        // Find index of time vector up to first missing time spot.
        unsigned int idx = 0;
        for (unsigned int i = 0; i < t.size(); i++) {
            if (t[i] <= t_mis1(0)) {
                idx = i;
            }
        }

        std::vector<double> t_prop = t;
        int cnt1 = 0;
        // Insert the z_currs into t_prop provided cnt != cnt1. So ignore exactly one of the intervals
        for (auto& k : z_currs) {
            if (cnt != cnt1) {
                t_prop.insert(t_prop.end(), k.begin(), k.end());
            }
            cnt1++;
        }
        // Insert the proposed z_prop matching the cnt variable I suppose.
        t_prop.insert(t_prop.end(), z_prop.begin(), z_prop.end());
        std::sort(t_prop.begin(), t_prop.end());

        // t_curr is just t + the z_currs inside of it
        std::vector<double> t_curr = t;
        for (auto& k : z_currs) {
            t_curr.insert(t_curr.end(), k.begin(), k.end());
        }
        std::sort(t_curr.begin(), t_curr.end());

        std::vector<double>::const_iterator first = t.begin();
        std::vector<double>::const_iterator last = t.begin() + idx + 1;
        std::vector<double> t_prop_sub(first, last);
        cnt1 = 0;
        // Make t_prop_sub which uses all the z_currs up to the current missing time, and use z_prop there
        for (auto& k : z_currs) {
            if (cnt1 < cnt) {
                t_prop_sub.insert(t_prop_sub.end(), k.begin(), k.end());
            }
            cnt1++;
        }
        t_prop_sub.insert(t_prop_sub.end(), z_prop.begin(), z_prop.end());
        std::sort(t_prop_sub.begin(), t_prop_sub.end());
        // t_curr_sub is just like t_prop_sub, but uses the z_curr at cnt rather than z_prop
        std::vector<double> t_curr_sub(first, last);
        cnt1 = 0;
        for (auto& k : z_currs) {
            if (cnt1 <= cnt) {
                t_curr_sub.insert(t_curr_sub.end(), k.begin(), k.end());
            }
            cnt1++;
        }
        std::sort(t_curr_sub.begin(), t_curr_sub.end());

        double top = temporal::temporalLogLikelihood(t_prop, mu_curr, alpha_curr, beta_curr, t_max) +
                     temporal::temporalLogLikelihood(t_curr_sub, mu_curr, alpha_curr, beta_curr, t_mis1(1));
        double bottom = temporal::temporalLogLikelihood(t_curr, mu_curr, alpha_curr, beta_curr, t_max) +
                        temporal::temporalLogLikelihood(t_prop_sub, mu_curr, alpha_curr, beta_curr, t_mis1(1));

        if (runif(gen) < exp(top - bottom)) {
            j.clear();
            j = z_prop;
        }
        cnt++;
    }


    return (z_currs);
}

std::vector<int> sample_y_missing_data(double alpha_curr, double beta_curr, double mu_curr,
                                       const std::vector<double>& z_curr, const std::vector<double>& t) {
    auto t_tmp = insertSimulatedTimes(t, z_curr);

    return temporal::sample_y(alpha_curr, beta_curr, mu_curr, t_tmp);
}

double sample_alpha_missing_data(double alpha_curr, double beta_curr, double t_max,
                                 const std::vector<double>& alpha_param, const int sum_numtriggered,
                                 const std::vector<double>& z_curr, const std::vector<double>& t) {
    double alpha_a = alpha_param[0];
    double alpha_b = alpha_param[1];
    auto t_tmp = insertSimulatedTimes(t, z_curr);

    return temporal::sample_alpha(t_tmp, sum_numtriggered, t_max, beta_curr, alpha_a, alpha_b);
}

double sample_beta_missing_data(double alpha_curr, double beta_curr, double t_max, double sig_beta,
                                const std::vector<double>& beta_param, const std::vector<double>& z_curr,
                                const std::vector<double>& t, const std::vector<double>& z) {
    auto t_tmp = insertSimulatedTimes(t, z_curr);

    return temporal::sample_beta(alpha_curr, beta_curr, t_max, sig_beta, t_tmp, beta_param, z);
}

std::tuple<arma::vec, arma::vec, arma::vec, arma::vec> test_method(
    const std::vector<double>& ti, const arma::mat& t_misi, double t_maxi, const std::vector<int>& y_init,
    double mu_init, double alpha_init, double beta_init, const std::vector<double>& mu_parami,
    const std::vector<double>& alpha_parami, const std::vector<double>& beta_parami, double sig_betai, int n_mcmc,
    int n_burn, bool print) {
    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    // initialize parameters
    double t_max = t_maxi;
    arma::mat t_mis = t_misi;
    std::vector<int> y_curr = y_init;
    double mu_curr = mu_init;
    double alpha_curr = alpha_init;
    double beta_curr = beta_init;

    std::vector<double> t;
    arma::vec ts = arma::conv_to<arma::vec>::from(t);

    // Save ti to dataset
    arma::vec ti_data = arma::conv_to<arma::vec>::from(ti);

    int n_mis = t_mis.n_rows;
    std::vector<std::vector<double>> z_currs;
    z_currs.resize(n_mis);
    static std::vector<double> z_currt;

    std::vector<double> z_curr;  // All z sampled times in one single vector
    int cnt = 0;
    for (auto& j : z_currs) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_mis(cnt, 0);
        t_mis1(1) = t_mis(cnt, 1);
        arma::vec z_curr1 = simulate_temporal(mu_curr, alpha_curr, beta_curr, t_mis1, ts);
        z_currt = arma::conv_to<std::vector<double>>::from(z_curr1);
        j = z_currt;
        z_curr.insert(z_curr.end(), z_currt.begin(), z_currt.end());
        cnt++;
    }

    std::vector<double> mu_param = mu_parami;
    std::vector<double> alpha_param = alpha_parami;
    std::vector<double> beta_param = beta_parami;
    double sig_beta = sig_betai;
    t = ti;
    arma::vec mu_samps(n_mcmc);
    arma::vec alpha_samps(n_mcmc);
    arma::vec beta_samps(n_mcmc);
    arma::vec z_samps(n_mcmc);

    int n = t.size();

    std::vector<double> z;
    z.reserve(n);

    // begin mcmc
    Progress p(n_mcmc, print);
    for (int iter = 0; iter < n_mcmc; iter++) {
        y_curr = sample_y_missing_data(alpha_curr, beta_curr, mu_curr, z_curr, t);

        std::vector<double> t_tmp = t;
        t_tmp.insert(t_tmp.end(), z_curr.begin(), z_curr.end());
        std::sort(t_tmp.begin(), t_tmp.end());

        int numbackground;
        std::vector<int> numtriggered;
        std::tie(numbackground, numtriggered) = temporal::calculateNumTriggered(t_tmp, y_curr, z);

        mu_curr = temporal::sample_mu(t_max, numbackground, mu_param);
        alpha_curr = sample_alpha_missing_data(alpha_curr, beta_curr, t_max, alpha_param, z.size(), z_curr, t);
        beta_curr = sample_beta_missing_data(alpha_curr, beta_curr, t_max, sig_beta, beta_param, z_curr, t, z);
        z_currs = sample_z(alpha_curr, beta_curr, mu_curr, t_max, t, z_currs, t_mis);
        z_curr.clear();
        for (auto& j : z_currs) {
            z_curr.insert(z_curr.end(), j.begin(), j.end());
        }
        mu_samps(iter) = mu_curr;
        alpha_samps(iter) = alpha_curr;
        beta_samps(iter) = beta_curr;
        z_samps(iter) = z_curr.size();
        p.increment();  // update progress
    }

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec alpha_sampso = alpha_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec beta_sampso = beta_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec z_sampso = z_samps.subvec(n_burn, n_mcmc - 1);

    return std::make_tuple(mu_sampso, alpha_sampso, beta_sampso, z_sampso);
}

// Bayesian Estimation of Temporal Hawkes Model with Missing Data using branching
// [[Rcpp::export]]
List condInt_mcmc_temporal_branching_md(std::vector<double> ti, arma::mat t_misi, double t_maxi,
                                             std::vector<int> y_init, double mu_init, double alpha_init,
                                             double beta_init, std::vector<double> mu_parami,
                                             std::vector<double> alpha_parami, std::vector<double> beta_parami,
                                             double sig_betai, int n_mcmc, int n_burn, bool print) {
    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    // Save ti to dataset
    // arma::vec ti_data = arma::vonc_to<arma::vec>::from(ti);

    // initialize parameters
    double t_max = t_maxi;
    arma::mat t_mis = t_misi;
    std::vector<int> y_curr = y_init;
    double mu_curr = mu_init;
    double alpha_curr = alpha_init;
    double beta_curr = beta_init;

    std::vector<double> t;

    arma::vec ts = arma::conv_to<arma::vec>::from(t);

    // Save ti to dataset
    arma::vec ti_data = arma::conv_to<arma::vec>::from(ti);

    int n_mis = t_mis.n_rows;
    std::vector<std::vector<double>> z_currs;
    z_currs.resize(n_mis);
    std::vector<double> z_currt;

    std::vector<double> z_curr;
    int cnt = 0;
    for (auto& j : z_currs) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_mis(cnt, 0);
        t_mis1(1) = t_mis(cnt, 1);
        arma::vec z_curr1 = simulate_temporal(mu_curr, alpha_curr, beta_curr, t_mis1, ts);
        z_currt = arma::conv_to<std::vector<double>>::from(z_curr1);
        j = z_currt;
        z_curr.insert(z_curr.end(), z_currt.begin(), z_currt.end());
        cnt++;
    }

    std::vector<double> mu_param = mu_parami;
    std::vector<double> alpha_param = alpha_parami;
    std::vector<double> beta_param = beta_parami;
    double sig_beta = sig_betai;
    t = ti;
    arma::vec mu_samps(n_mcmc);
    arma::vec alpha_samps(n_mcmc);
    arma::vec beta_samps(n_mcmc);
    arma::vec z_samps(n_mcmc);

    int n = t.size();
    std::vector<double> z;
    z.reserve(n);

    // begin mcmc
    Progress p(n_mcmc, print);
    List z_sampsallo(n_mcmc-n_burn);
    List y_sampso(n_mcmc-n_burn);
    int cnt_iter = 0;
    for (int iter = 0; iter < n_mcmc; iter++) {
        if (Progress::check_abort())
            return -1.0;
        y_curr = sample_y_missing_data(alpha_curr, beta_curr, mu_curr, z_curr, t);

        std::vector<double> t_tmp = t;
        t_tmp.insert(t_tmp.end(), z_curr.begin(), z_curr.end());
        std::sort(t_tmp.begin(), t_tmp.end());

        int numbackground;
        std::vector<int> numtriggered;
        std::tie(numbackground, numtriggered) = temporal::calculateNumTriggered(t_tmp, y_curr, z);

        mu_curr = temporal::sample_mu(t_max, numbackground, mu_param);
        alpha_curr = sample_alpha_missing_data(alpha_curr, beta_curr, t_max, alpha_param, z.size(), z_curr, t);
        beta_curr = sample_beta_missing_data(alpha_curr, beta_curr, t_max, sig_beta, beta_param, z_curr, t, z);
        z_currs = sample_z(alpha_curr, beta_curr, mu_curr, t_max, t, z_currs, t_mis);
        z_curr.clear();
        for (auto& j : z_currs) {
            z_curr.insert(z_curr.end(), j.begin(), j.end());
        }
        mu_samps(iter) = mu_curr;
        alpha_samps(iter) = alpha_curr;
        beta_samps(iter) = beta_curr;
        z_samps(iter) = z_curr.size();
        if (iter > n_burn){
          z_sampsallo[cnt_iter] = z_curr;
          y_sampso[cnt_iter] = y_curr;
          cnt_iter++;
        }

        p.increment();  // update progress
    }

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec alpha_sampso = alpha_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec beta_sampso = beta_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec z_sampso = z_samps.subvec(n_burn, n_mcmc - 1);

    DataFrame df = DataFrame::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("alpha") = alpha_sampso,
                                     Rcpp::Named("beta") = beta_sampso, Rcpp::Named("n_missing") = z_sampso);

    List out = List::create(Named("samps") = df , _["branching"] = y_sampso, _["zsamps"] = z_sampsallo);

    return (out);
}
