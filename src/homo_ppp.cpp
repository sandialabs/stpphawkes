#include <RcppArmadillo.h>
#include "polygon.h"
#include "utilities.h"
// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

arma::mat homog_STPP(double mu, arma::mat poly, arma::vec t_region, double xfrac, double yfrac, int seed = -1) {
    if (seed != -1) {
        set_seed((unsigned int)seed);
    }

    arma::mat lr = larger_region(poly, xfrac, yfrac);

    t_region = sort(t_region);
    double t_area = t_region(1) - t_region(0);

    int npts = R::rpois(mu * t_area);

    NumericVector x = runif(npts, lr(0, 0), lr(1, 0));
    NumericVector y = runif(npts, lr(0, 1), lr(1, 1));
    NumericVector times = runif(npts, t_region(0), t_region(1));
    IntegerVector idx = seq_len(npts) - 1;
    IntegerVector samp = sample(idx, npts, false);
    times = times[samp];
    std::sort(times.begin(), times.end());

    NumericMatrix out(npts, 3);
    out(_, 0) = x;
    out(_, 1) = y;
    out(_, 2) = times;

    arma::mat out1 = as<arma::mat>(out);

    return out1;
}
