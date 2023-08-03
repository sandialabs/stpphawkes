#include <RcppArmadillo.h>

#include <Rcpp.h>

#include <gsl/gsl_randist.h>

#include "helper_functions.h"
#include "simulate_temporal_hawkes.h"
#include "temporal_common.h"

#include "temporal_catmark_common.h"

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

std::vector<int> initializeMarks(std::vector<double>& z_curr, std::vector<double>& p_curr, gsl_rng* rng) {
    std::vector<unsigned int> results(p_curr.size());
    //     R::rmultinom(1, p_curr.data(), z_curr.size(), results.data());

    size_t K = p_curr.size();
    size_t N = z_curr.size();

    gsl_ran_multinomial(rng, K, N, p_curr.data(), results.data());

    std::vector<int> final_results(results.begin(), results.end());

    return final_results;
}

std::vector<int> sampleMark(std::vector<double>& z_curr, std::vector<double>& p_curr, gsl_rng* rng) {
    size_t num_marks = p_curr.size();

    std::vector<unsigned int> mark_num(num_marks);

    std::random_device rd;
    std::mt19937 g(rd());

    //     R::Rf_rmultinom(1, p_curr.data(), z_curr.size(), mark_num.data());
    size_t K = p_curr.size();
    size_t N = z_curr.size();

    gsl_ran_multinomial(rng, K, N, p_curr.data(), mark_num.data());

    std::vector<int> mark_samp;
    for (size_t i = 0; i < mark_num.size(); ++i) {
        // insert mark "i" mark_num[i] times
        std::vector<int> mark_values(mark_num[i], i);
        mark_samp.insert(mark_samp.end(), mark_values.begin(), mark_values.end());
    }
    std::shuffle(mark_samp.begin(), mark_samp.end(), g);

    return mark_samp;
}

std::vector<std::vector<double>> sampleZ(const std::vector<double>& t, std::vector<std::vector<double>>& z_currs,
                                         const arma::mat& t_mis, const double t_max, const double mu_curr,
                                         const double alpha_curr, const double beta_curr) {
    arma::vec ts = arma::conv_to<arma::vec>::from(t);

    auto gen = GenerateMersenneTwister();
    std::uniform_real_distribution<> runif(0, 1);
    double alpha_str = alpha_curr * beta_curr;

    int cnt = 0;
    std::vector<double> z_prop;
    // Walk along each missing time interval
    for (auto& j : z_currs) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_mis(cnt, 0);
        t_mis1(1) = t_mis(cnt, 1);
        // Get a new proposed z_prop for this time interval
        if (alpha_str < beta_curr) {
            z_prop =
                arma::conv_to<std::vector<double>>::from(simulate_temporal(mu_curr, alpha_curr, beta_curr, t_mis1, ts));
        } else {
            z_prop = j;
        }

        // We will define 4 different time values:
        // 1: t_prop will be the known times t + z_currs, except for this interval, where z_prop is used
        // 2: t_curr will be known times t + the z_currs we have so far, but no proposed ones
        // 3: t_curr_sub will be known times t up to the current time interval with z_curr inserted
        // 4: t_prop_sub will be the known times t up to the current time interval with z_prop inserted

        // The ratio is going to be (LL(t_prop) + LL(t_curr_sub)) / (LL(t_curr) + LL(t_prop_sub))
        unsigned int idx = 0;
        for (unsigned int i = 0; i < t.size(); i++) {
            if (t[i] <= t_mis1(0)) {
                idx = i;
            }
        }

        // t_prop should be t with z_prop inserted into him
        std::vector<double> t_prop = t;
        int cnt1 = 0;
        for (auto& k : z_currs) {
            if (cnt != cnt1) {
                t_prop.insert(t_prop.end(), k.begin(), k.end());
            }
            cnt1++;
        }
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

        std::vector<double> t_curr_sub(first, last);
        cnt1 = 0;
        for (auto& k : z_currs) {
            if (cnt1 <= cnt) {
                t_curr_sub.insert(t_curr_sub.end(), k.begin(), k.end());
            }
            cnt1++;
        }
        std::sort(t_curr_sub.begin(), t_curr_sub.end());
        // t_curr_sub = c(t[1:idx]), z_curr)
        double top = temporal::temporalLogLikelihood(t_prop, mu_curr, alpha_curr, beta_curr, t_max) +
                     temporal::temporalLogLikelihood(t_curr_sub, mu_curr, alpha_curr, beta_curr, t_mis1[1]);

        double bottom = temporal::temporalLogLikelihood(t_curr, mu_curr, alpha_curr, beta_curr, t_max) +
                        temporal::temporalLogLikelihood(t_prop_sub, mu_curr, alpha_curr, beta_curr, t_mis1[1]);

        if (runif(gen) < std::exp(top - bottom)) {
            j.clear();
            j = z_prop;  // would std::move be smarter here?
        }
        cnt++;
    }
    return z_currs;
}

// Bayesian Estimation of Temporal Hawkes Model with Categorical Marks and Missing Data
// [[Rcpp::export]]
DataFrame CatMarkMcMcMissingData(const std::vector<double>& t, const arma::mat& t_missing, const double t_max,
                                 const std::vector<int>& marks, const double mu_init, const double alpha_init,
                                 const double beta_init, const std::vector<double> p_init,
                                 const std::vector<double> mu_params, const std::vector<double>& alpha_params,
                                 const std::vector<double>& beta_params, const std::vector<double>& p_params,
                                 const double sig_beta, const size_t n_mcmc = 1e4, const size_t n_burn = 5e3,
                                 bool print = 1) {
    double alpha_a = alpha_params[0];
    double alpha_b = alpha_params[1];
    double beta_a = beta_params[0];
    double beta_b = beta_params[1];

    // Make vector of length p_param with 1/ length p_param as each entry

    size_t kk = p_params.size();

    double mu_curr = mu_init;
    double alpha_curr = alpha_init;
    double beta_curr = beta_init;
    std::vector<double> p_curr = p_init;

    int n_mis = t_missing.n_rows;

    // Vector of vectors where each subvector contains a sampled set of missing times
    // There is one vector per missing time interval, hence z_currs.size() = n_mis
    std::vector<std::vector<double>> z_currs;
    z_currs.resize(n_mis);
    arma::vec ts = arma::conv_to<arma::vec>::from(t);

    std::vector<double> z_currt;
    std::vector<double> z_curr;  // All z sampled times in one single vector
    int cnt = 0;
    for (auto& j : z_currs) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_missing(cnt, 0);
        t_mis1(1) = t_missing(cnt, 1);
        arma::vec z_curr1 = simulate_temporal(mu_curr, alpha_curr, beta_curr, t_mis1, ts);
        z_currt = arma::conv_to<std::vector<double>>::from(z_curr1);

        j = z_currt;
        z_curr.insert(z_curr.end(), z_currt.begin(), z_currt.end());
        cnt++;
    }

    std::vector<int> ntriggered(t.size(), 0);

    arma::vec mu_samps(n_mcmc);
    arma::vec alpha_samps(n_mcmc);
    arma::vec beta_samps(n_mcmc);
    arma::vec z_samps(n_mcmc);
    arma::mat p_samps(n_mcmc, kk);

    std::vector<double> z;
    z.reserve(t.size());
    // Begin MCMC
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);  // Arguably, we should std::unique_ptr<> this
    std::vector<int> mark_curr = initializeMarks(z_curr, p_curr, rng);

    Progress p(n_mcmc, print);
    for (size_t iter = 0; iter < n_mcmc; ++iter) {
        if (Progress::check_abort()) {
            return -1.0;
        }
        // Insert the Z samples into the time vector t_tmp
        std::vector<double> t_tmp = insertSimulatedTimes(t, z_curr);

        auto y_curr = temporal::sample_y(alpha_curr, beta_curr, mu_curr, t_tmp);

        int numbackground;
        std::vector<int> numtriggered;
        std::tie(numbackground, numtriggered) = temporal::calculateNumTriggered(t_tmp, y_curr, z);

        mark_curr = sampleMark(z_curr, p_curr, rng);
        mu_curr = temporal::sample_mu(t_max, numbackground, mu_params);

        // Pass in t_tmp, which contains t + sampled z times
        alpha_curr = temporal::sample_alpha(t_tmp, z.size(), t_max, beta_curr, alpha_a, alpha_b);

        beta_curr = catmark::sampleBeta(alpha_curr, beta_curr, t_max, sig_beta, t_tmp, z, numtriggered, beta_a, beta_b);

        p_curr = catmark::sampleP(marks, p_params, rng);
        z_currs = sampleZ(t, z_currs, t_missing, t_max, mu_curr, alpha_curr, beta_curr);
        z_curr.clear();

        for (auto& j : z_currs) {
            z_curr.insert(z_curr.end(), j.begin(), j.end());
        }

        mu_samps(iter) = mu_curr;
        alpha_samps(iter) = alpha_curr;
        beta_samps(iter) = beta_curr;

        z_samps(iter) = z_curr.size();
        p_samps.row(iter) = arma::conv_to<arma::rowvec>::from(p_curr);
        p.increment();  // update progress
    }
    // Release random number generator
    gsl_rng_free(rng);

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec alpha_sampso = alpha_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec beta_sampso = beta_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec z_sampso = z_samps.subvec(n_burn, n_mcmc - 1);
    arma::mat p_sampso = p_samps.rows(n_burn, n_mcmc - 1);

    DataFrame df =
        DataFrame::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("alpha") = alpha_sampso,
                          Rcpp::Named("beta") = beta_sampso, Rcpp::Named("z") = z_sampso, Rcpp::Named("p") = p_sampso);

    return df;
};
