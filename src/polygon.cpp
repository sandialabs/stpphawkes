#include <RcppArmadillo.h>
#include "ptinpoly.h"
// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


// [[Rcpp::export]]
arma::mat bbox(const arma::mat& poly) {
    arma::vec x = {min(poly.col(0)), max(poly.col(0))};
    arma::vec y = {min(poly.col(1)), max(poly.col(1))};
    arma::mat z(2, 2);
    z.col(0) = x;
    z.col(1) = y;
    return z;
}

// [[Rcpp::export]]
arma::mat bboxx(const arma::mat& poly) {
    arma::mat z(4, 2);
    z(0, 0) = poly(0, 0);
    z(1, 0) = poly(0, 1);
    z(2, 0) = poly(0, 1);
    z(3, 0) = poly(0, 0);
    z(0, 1) = poly(1, 0);
    z(1, 1) = poly(1, 0);
    z(2, 1) = poly(1, 1);
    z(3, 1) = poly(1, 1);
    return z;
}

// [[Rcpp::export]]
arma::mat sbox(const arma::mat& poly, const double xfrac, const double yfrac) {
    arma::vec xr = {min(poly.col(0)), max(poly.col(0))};
    arma::vec yr = {min(poly.col(1)), max(poly.col(1))};

    double xw = xr(1) - xr(0);
    xr(0) -= xfrac * xw;
    xr(1) += xfrac * xw;

    double yw = yr(1) - yr(0);
    yr(0) -= yfrac * yw;
    yr(1) += yfrac * yw;

    arma::mat out(4, 2);
    out(0, 0) = xr(0);
    out(1, 0) = xr(1);
    out(2, 0) = xr(1);
    out(3, 0) = xr(0);
    out(0, 1) = yr(0);
    out(1, 1) = yr(0);
    out(2, 1) = yr(1);
    out(3, 1) = yr(1);

    return out;
}

// [[Rcpp::export]]
arma::mat buffer_region(const arma::mat& poly, const double d) {
    arma::mat out = poly;
    out.row(0) += d;
    out.row(2) -= d;
    out(1, 0) += d;
    out(1, 1) -= d;
    out(3, 0) -= d;
    out(3, 1) += d;

    return out;
}

// [[Rcpp::export]]
arma::mat larger_region(const arma::mat& poly, const double xfrac, const double yfrac) {
    arma::mat out = sbox(poly, xfrac, yfrac);
    arma::mat lr(2, 2);
    lr(0, 0) = min(out.col(0));
    lr(1, 0) = max(out.col(0));
    lr(0, 1) = min(out.col(1));
    lr(1, 1) = max(out.col(1));
    return lr;
}

// [[Rcpp::export]]
arma::uvec inout(arma::vec& x, arma::vec& y, arma::mat& poly, bool bound) {
    arma::vec xp = poly.col(0);
    xp.insert_rows(poly.n_rows, 1);
    xp(poly.n_rows) = poly(0, 0);
    arma::vec yp = poly.col(1);
    yp.insert_rows(poly.n_rows, 1);
    yp(poly.n_rows) = poly(0, 1);

    arma::mat bb = larger_region(poly, .1, .1);

    arma::ivec result = ptinpoly(x, y, xp, yp, bb);

    arma::uvec z(x.n_elem);
    z.fill(0);
    if (bound) {
        arma::uvec id = find(result <= 0);
        z(id).fill(1);
    } else {
        arma::uvec id = find(result < 0);
        z(id).fill(1);
    }

    return z;
}
