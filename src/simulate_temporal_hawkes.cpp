#include "RcppArmadillo.h"

#include "utilities.h"
// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

//' Calculate intensity function for temporal Hawkes
//'
//' @param mu - background parameter
//' @param alpha - alpha parameter
//' @param beta - beta parameter
//' @param times - history of previous times
//' @param evalpt - point to evaluate
//' @return lambda - intensity at evalpt
//' @export
// [[Rcpp::export]]
double intensity_temporal(double mu, double alpha, double beta, arma::vec& times, double evalpt) {
    double ci = 0;
    if (times.n_elem > 0) {
        arma::uvec use = find(times <= evalpt);

        if (use.n_elem > 0) {
            ci = alpha * sum(exp(-beta * (evalpt - times(use))));
        } else {
            ci = 0;
        }
    } else {
        ci = 0;
    }

    ci = mu + ci;
    return (ci);
}

//' Simulates a temporal Hawkes process with an exponential correlation function
//'
//' @param mu - background parameter
//' @param alpha - \eqn{\alpha} parameter
//' @param beta - \eqn{\beta} parameter
//' @param tt - vector of two elements defining time span (e.g., c(0,10))
//' @param times - history of previous times (e.g., numeric())
//' @param seed - value to seed random number generation (default = -1)
//' @return arrivals - vector of arrival times
//' @export
//' @examples 
//'     times = simulate_temporal(.5,.1,.5,c(0,10),numeric())
// [[Rcpp::export]]
arma::vec simulate_temporal(const double mu, double alpha, const double beta, const arma::vec& tt,
                            const arma::vec& times, int seed = -1) {
    if (alpha > beta) {
        stop("Unstable. You must have alpha < beta");
    }

    alpha *= beta;

    if (seed != -1) {
        set_seed((unsigned int)seed);
    }

    arma::vec arrivals;
    double t_max = tt(1);
    double s, t, dlambda, lambda_star;
    double U, u0, idx;
    s = t = dlambda = lambda_star = 0;

    if (times.n_elem == 0 || min(times) > tt(0)) {
        s = 0;
        t = 0;
        dlambda = 0;
        lambda_star = mu;
        U = R::runif(0, 1);
        u0 = -(1.0 / lambda_star) * log(U);
        if (u0 > t_max) {
            return (arrivals);
        }

        s += u0;
        t = s;
        dlambda = alpha;
        arrivals.resize(1);
        arrivals(0) = t;
    } else {
        idx = 0;
        if (max(times) <= tt(0)) {
            idx = times.n_elem - 1;
        } else {
            for (unsigned int i = 0; i < times.n_elem; i++) {
                if (times(i) <= tt(0)) {
                    idx = i;
                }
            }
        }

        s = times(idx);
        t = times(idx);
        dlambda = 0;
        arma::vec time_sub = times.subvec(0, idx);
        lambda_star = intensity_temporal(mu, alpha, beta, time_sub, s);
        dlambda = alpha;
        for (int i = 1; i <= idx; i++) {
            dlambda = alpha + dlambda * exp(-beta * (times(i) - times(i - 1)));
        }
    }

    while (s < t_max) {
        lambda_star = mu + dlambda * exp(-beta * (s - t));
        U = R::runif(0, 1);
        s = s - (1.0 / lambda_star) * log(U);
        if (s > t_max) {
            arma::uvec id = find(arrivals >= tt(0), 1, "first");
            if (id.n_elem > 0) {
                arma::vec arrivalso = arrivals.subvec(id[0], arrivals.n_elem - 1);
                return (arrivalso);
            } else {
                return (arrivals);
            }
        }

        U = R::runif(0, 1);
        if (U <= (mu + dlambda * exp(-beta * (s - t))) / lambda_star) {
            dlambda = alpha + dlambda * exp(-beta * (s - t));
            t = s;
            arrivals.insert_rows(arrivals.n_elem, 1);
            arrivals(arrivals.n_elem - 1) = t;
        }
    }

    arma::uvec id = find(arrivals >= tt(0), 1, "first");
    if (id.n_elem > 0) {
        arma::vec arrivalso = arrivals.subvec(id[0], arrivals.n_elem - 1);
        return (arrivalso);
    } else {
        return (arrivals);
    }
}
