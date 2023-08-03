#include <RcppArmadillo.h>
using namespace Rcpp;

void set_seed(unsigned int seed) {
    Environment base_env("package:base");
    Function set_seed_r = base_env["set.seed"];
    set_seed_r(seed);
}

arma::mat DFtoMat(DataFrame x) {
    int nRows = x.nrows();
    NumericMatrix y(nRows, x.size());
    for (int i = 0; i < x.size(); i++) {
        y(_, i) = NumericVector(x[i]);
    }

    arma::mat out = as<arma::mat>(y);
    return out;
}
