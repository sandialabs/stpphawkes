#include <RcppArmadillo.h>
using namespace Rcpp;

void set_seed(unsigned int seed) {
    Environment base_env("package:base");
    Function set_seed_r = base_env["set.seed"];
    set_seed_r(seed);
}

arma::mat DFtoMat(DataFrame x) {
    int nRows = x.nrows();

    // Only the numeric columns can go into the matrix. A data frame produced by homog.STPP also
    // carries a character "type" column, and coercing that with NumericVector throws.
    std::vector<int> numeric_cols;
    for (int i = 0; i < x.size(); i++) {
        SEXP col = x[i];
        if (Rf_isFactor(col)) {
            continue;
        }
        switch (TYPEOF(col)) {
            case REALSXP:
            case INTSXP:
            case LGLSXP:
                numeric_cols.push_back(i);
                break;
            default:
                break;
        }
    }

    NumericMatrix y(nRows, numeric_cols.size());
    for (size_t i = 0; i < numeric_cols.size(); i++) {
        y(_, i) = NumericVector(x[numeric_cols[i]]);
    }

    arma::mat out = as<arma::mat>(y);
    return out;
}
