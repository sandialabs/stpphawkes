#ifndef STPP_POLYGON_H
#define STPP_POLYGON_H

#include <RcppArmadillo.h>

using namespace Rcpp;

arma::mat bbox(const arma::mat& poly);
arma::mat bboxx(const arma::mat& poly);
arma::mat sbox(const arma::mat& poly, const double xfrac, const double yfrac);
arma::mat larger_region(const arma::mat& poly, const double xfrac, const double yfrac);
arma::mat buffer_region(const arma::mat& poly, const double d);
arma::uvec inout(arma::vec& x, arma::vec& y, arma::mat& poly, bool bound);

#endif  // STPP_POLYGON_H
