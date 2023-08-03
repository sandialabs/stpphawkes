#ifndef STPP_HOMO_PPP_H
#define STPP_HOMO_PPP_H

#include <RcppArmadillo.h>

arma::mat homog_STPP(double mu, arma::mat poly, arma::vec t_region, double xfrac, double yfrac, int seed = -1);

#endif  // STPP_HOMO_PPP_H
