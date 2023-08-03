#ifndef STPP_SIMULATE_STPP_HAWKES
#define STPP_SIMULATE_STPP_HAWKES

#include <RcppArmadillo.h>

Rcpp::DataFrame simulate_hawkes_stpp(Rcpp::List params, arma::mat poly, arma::vec t_region, Rcpp::DataFrame history,
                                     int seed = -1);

arma::mat simulate_hawkes_stpp_c(double mu, double a, double b, double sig, arma::mat poly, arma::vec t_region,
                                 Rcpp::DataFrame history, bool sp_clip);


#endif  // STPP_SIMULATE_STPP_HAWKES
