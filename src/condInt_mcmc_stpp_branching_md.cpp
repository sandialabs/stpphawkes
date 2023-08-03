#include <RcppArmadillo.h>
#include "areapl.h"
#include "simulate_stpp_hawkes.h"
#include "utilities.h"
#include "vector_utilities.h"

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

void sample_z(const std::vector<double>& t, const std::vector<double>& x, const std::vector<double>& y,
              std::vector<std::vector<double>>& z_curr_x, std::vector<std::vector<double>>& z_curr_y,
              std::vector<std::vector<double>>& z_curr_t, const double t_max, const arma::mat& t_mis,
              const double mu_curr, const double a_curr, const double b_curr, const double sig_curr, bool sp_clip,
              arma::mat poly, DataFrame history) {
    auto gen = GenerateMersenneTwister();

    std::uniform_real_distribution<> runif(0, 1);
    size_t num_missing_time_intervals = t_mis.n_rows;
    for (size_t j = 0; j < num_missing_time_intervals; ++j) {
        std::vector<double> z_prop_x;
        std::vector<double> z_prop_y;
        std::vector<double> z_prop_t;
        arma::vec t_mis1(2);
        t_mis1(0) = t_mis(j, 0);
        t_mis1(1) = t_mis(j, 1);
        if (a_curr < b_curr) {
            arma::mat z_prop =
                simulate_hawkes_stpp_c(mu_curr, a_curr, b_curr, sig_curr, poly, t_mis1, history, sp_clip);

            z_prop_x = arma::conv_to<std::vector<double>>::from(z_prop.col(0));
            z_prop_y = arma::conv_to<std::vector<double>>::from(z_prop.col(1));
            z_prop_t = arma::conv_to<std::vector<double>>::from(z_prop.col(2));
        } else {
            z_prop_x = z_curr_x[j];
            z_prop_y = z_curr_y[j];
            z_prop_t = z_curr_t[j];
        }

        unsigned int idx = 0;
        for (unsigned int i = 0; i < t.size(); i++) {
            if (t[i] <= t_mis1(0)) {
                idx = i;
            }
        }

        // t_curr = t + z_curr_t
        std::vector<double> t_curr = t;  // Start by copying t
        std::vector<double> x_curr = x;
        std::vector<double> y_curr = y;
        std::vector<size_t> idx1;

        for (size_t i = 0; i < num_missing_time_intervals; ++i) {
            // Insert simulated times

            idx1 = insertSimulatedTimesAndIndex(z_curr_t[i], t_curr);

            // Now that we've inserted one of them, we can insert the corresponding
            // z_curr_x
            insertSimulatedSpatialPoints(z_curr_x[i], idx1, x_curr);
            insertSimulatedSpatialPoints(z_curr_y[i], idx1, y_curr);
        }

        // t_prop = t + z_prop_t

        std::vector<double> t_prop = t;
        std::vector<double> x_prop = x;
        std::vector<double> y_prop = y;
        for (size_t i = 0; i < num_missing_time_intervals; ++i) {
            if (i != j) {
                // Insert this time just like before
                // Insert simulated times

                idx1 = insertSimulatedTimesAndIndex(z_curr_t[i], t_prop);

                // Now that we've inserted one of them, we can insert the corresponding
                // z_curr_x
                insertSimulatedSpatialPoints(z_curr_x[i], idx1, x_prop);
                insertSimulatedSpatialPoints(z_curr_y[i], idx1, y_prop);

            } else {
                // Insert t_prop, x_prop, y_prop

                idx1 = insertSimulatedTimesAndIndex(z_prop_t, t_prop);

                // Now that we've inserted one of them, we can insert the corresponding
                // z_curr_x
                insertSimulatedSpatialPoints(z_prop_x, idx1, x_prop);
                insertSimulatedSpatialPoints(z_prop_y, idx1, y_prop);
            }
        }

        // If only we had C++17, I wouldn't have to do this absurd pre-declaration
        // of t,x,y, currs vectors...

        // t_currs = t[1:idx] + z_curr_t
        std::vector<double>::const_iterator first = t.begin();
        std::vector<double>::const_iterator last = t.begin() + idx + 1;
        std::vector<double> t_currs(first, last);
        std::vector<double> t_props(first, last);
        first = x.begin();
        last = x.begin() + idx + 1;
        std::vector<double> x_currs(first, last);
        std::vector<double> x_props(first, last);
        first = y.begin();
        last = y.begin() + idx + 1;
        std::vector<double> y_currs(first, last);
        std::vector<double> y_props(first, last);

        // t_currs inserts the simulated times up until missing interval currently
        for (size_t i = 0; i <= j; ++i) {
            // Insert simulated times

            idx1 = insertSimulatedTimesAndIndex(z_curr_t[i], t_currs);

            // Now that we've inserted one of them, we can insert the corresponding
            // z_curr_x
            insertSimulatedSpatialPoints(z_curr_x[i], idx1, x_currs);
            insertSimulatedSpatialPoints(z_curr_y[i], idx1, y_currs);
        }

        // t_props = t[1:idx] + z_prop_t
        for (size_t i = 0; i < j; ++i) {
            // Insert simulated times
            idx1 = insertSimulatedTimesAndIndex(z_curr_t[i], t_props);

            // Now that we've inserted one of them, we can insert the corresponding
            // z_curr_x
            insertSimulatedSpatialPoints(z_curr_x[i], idx1, x_props);
            insertSimulatedSpatialPoints(z_curr_y[i], idx1, y_props);
        }

        // Also insert t_prop, x_prop, y_prop
        idx1 = insertSimulatedTimesAndIndex(z_prop_t, t_props);

        // Now that we've inserted one of them, we can insert the corresponding
        // z_curr_x
        insertSimulatedSpatialPoints(z_prop_x, idx1, x_props);
        insertSimulatedSpatialPoints(z_prop_y, idx1, y_props);

        double top =
            stpp::missing_data::log_lik(x_prop, y_prop, t_prop, mu_curr, a_curr, b_curr, sig_curr, t_max, poly) +
            stpp::missing_data::log_lik(x_currs, y_currs, t_currs, mu_curr, a_curr, b_curr, sig_curr, t_mis1(1), poly);
        double bottom =
            stpp::missing_data::log_lik(x_curr, y_curr, t_curr, mu_curr, a_curr, b_curr, sig_curr, t_max, poly) +
            stpp::missing_data::log_lik(x_props, y_props, t_props, mu_curr, a_curr, b_curr, sig_curr, t_mis1(1), poly);

        double lrat_curr = std::exp(top - bottom);

        if (runif(gen) < lrat_curr) {
            z_curr_x[j] = z_prop_x;
            z_curr_y[j] = z_prop_y;
            z_curr_t[j] = z_prop_t;
        }
    }
}

// Bayesian Estimation of STPP Hawkes Model with missing data using branching structure
// [[Rcpp::export]]
List condInt_mcmc_stpp_branching_md(DataFrame data, arma::mat t_misi, double t_maxi, std::vector<int> y_init,
                                    double mu_init, double a_init, double b_init, double sig_init, arma::mat poly,
                                    std::vector<double> mu_parami, std::vector<double> a_parami,
                                    std::vector<double> sig_parami, std::vector<double> b_parami, double sig_bi,
                                    double sig_sigi, int n_mcmc, int n_burn, bool print, bool sp_clip) {
    if (t_maxi < 0) {
        stop("t_max must be larger than 0");
    }

    std::vector<double> x = as<std::vector<double>>(data["x"]);
    std::vector<double> y = as<std::vector<double>>(data["y"]);
    std::vector<double> t = as<std::vector<double>>(data["t"]);

    // initialize parameters
    double t_max = t_maxi;
    arma::mat t_mis = t_misi;
    std::vector<int> y_curr = y_init;
    double mu_curr = mu_init;
    double a_curr = a_init;
    double b_curr = b_init;
    double sig_curr = sig_init;

    // Make z_curr x y and t a std::vector<std::vector<double>>
    using MissingVector = std::vector<std::vector<double>>;

    int n_mis = t_mis.n_rows;  // How many missing time intervals do we have?
    MissingVector z_curr_x;
    z_curr_x.reserve(n_mis);
    MissingVector z_curr_y;
    z_curr_y.reserve(n_mis);
    MissingVector z_curr_t;
    z_curr_t.reserve(n_mis);

    std::vector<double> z_curr_t_all;
    std::vector<double> z_curr_x_all;
    std::vector<double> z_curr_y_all;

    for (int i = 0; i < n_mis; ++i) {
        arma::vec t_mis1(2);
        t_mis1(0) = t_mis(i, 0);
        t_mis1(1) = t_mis(i, 1);
        auto z_curr = simulate_hawkes_stpp_c(mu_curr, a_curr, b_curr, sig_curr, poly, t_mis1, data, sp_clip);
        z_curr_x.emplace_back(arma::conv_to<std::vector<double>>::from(z_curr.col(0)));
        z_curr_y.emplace_back(arma::conv_to<std::vector<double>>::from(z_curr.col(1)));
        z_curr_t.emplace_back(arma::conv_to<std::vector<double>>::from(z_curr.col(2)));

        // We place all of the z_curr_x's into a single large vector z_curr_x_all,
        // and same for
        // y and t. These are easily insertable into x,y, and t variables in the
        // MCMC for loop
        z_curr_x_all.insert(z_curr_x_all.end(), z_curr_x[i].begin(), z_curr_x[i].end());
        z_curr_y_all.insert(z_curr_y_all.end(), z_curr_y[i].begin(), z_curr_y[i].end());
        z_curr_t_all.insert(z_curr_t_all.end(), z_curr_t[i].begin(), z_curr_t[i].end());
    }

    auto mu_param = mu_parami;
    auto sig_b = sig_bi;
    auto sig_sig = sig_sigi;

    auto a_param = a_parami;
    auto b_param = b_parami;
    auto sig_param = sig_parami;

    arma::vec mu_samps(n_mcmc);
    arma::vec a_samps(n_mcmc);
    arma::vec b_samps(n_mcmc);
    arma::vec sig_samps(n_mcmc);
    arma::vec z_samps(n_mcmc);

    arma::vec zlratios(n_mcmc);

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

        // Populate t_tmp, x_tmp, y_tmp with simulated data
        std::vector<double> t_tmp = t;

        std::vector<size_t> idx = insertSimulatedTimesAndIndex(z_curr_t_all, t_tmp);

        std::vector<double> x_tmp = insertSimulatedSpatialPoints(x, z_curr_x_all, idx);
        std::vector<double> y_tmp = insertSimulatedSpatialPoints(y, z_curr_y_all, idx);

        y_curr = stpp::sample_y(t_tmp, x_tmp, y_tmp, mu_curr, a_curr, b_curr, sig_curr, poly);
        n = t_tmp.size();

        z_t.clear();
        z_x.clear();
        z_y.clear();

        std::vector<int> numtriggered(n, 0);
        numbackground = 0;
        for (int i = 0; i < n; ++i) {
            if (y_curr[i] > 0) {
                numtriggered[y_curr[i] - 1]++;
                z_t.push_back(t_tmp[i] - t_tmp[y_curr[i] - 1]);
                z_x.push_back(x_tmp[i] - x_tmp[y_curr[i] - 1]);
                z_y.push_back(y_tmp[i] - y_tmp[y_curr[i] - 1]);
            } else {
                numbackground++;
            }
        }

        mu_curr = stpp::sample_mu(t_max, numbackground, mu_param);
        a_curr = stpp::sample_a(t_tmp, z_t, t_max, a_curr, b_curr, a_param);
        b_curr = stpp::sample_b(t_tmp, z_t, t_max, a_curr, b_curr, sig_b, b_param);
        sig_curr = stpp::sample_sig(z_x, z_y, sig_curr, sig_sig, sig_param);
        sample_z(t, x, y, z_curr_x, z_curr_y, z_curr_t, t_max, t_mis, mu_curr, a_curr, b_curr, sig_curr, sp_clip, poly,
                 data);

        // Reassign the z_curr_x_all, z_curr_y_all, and z_curr_t_all variables with
        // the new sampled data
        z_curr_x_all.clear();
        z_curr_y_all.clear();
        z_curr_t_all.clear();

        for (int i = 0; i < n_mis; ++i) {
            z_curr_x_all.insert(z_curr_x_all.end(), z_curr_x[i].begin(), z_curr_x[i].end());
            z_curr_y_all.insert(z_curr_y_all.end(), z_curr_y[i].begin(), z_curr_y[i].end());
            z_curr_t_all.insert(z_curr_t_all.end(), z_curr_t[i].begin(), z_curr_t[i].end());
        }

        mu_samps(iter) = mu_curr;
        a_samps(iter) = a_curr;
        b_samps(iter) = b_curr;
        sig_samps(iter) = sig_curr;
        z_samps(iter) = z_curr_t_all.size();
        p.increment();  // update progress
    }

    arma::vec mu_sampso = mu_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec a_sampso = a_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec b_sampso = b_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec sig_sampso = sig_samps.subvec(n_burn, n_mcmc - 1);
    arma::vec z_sampso = z_samps.subvec(n_burn, n_mcmc - 1);

    List df = List::create(Rcpp::Named("mu") = mu_sampso, Rcpp::Named("a") = a_sampso, Rcpp::Named("b") = b_sampso,
                           Rcpp::Named("sigma") = sig_sampso, Rcpp::Named("z") = z_sampso);

    return df;
}
