#ifndef STPP_NONUNIF_PPP_H
#define STPP_NONUNIF_PPP_H

#include <RcppArmadillo.h>

arma::mat nonunif_STPP(double mu, double mux, double muy, double sigx, double sigy, arma::mat poly, arma::vec t_region, double xfrac, double yfrac,int seed = -1);

#endif  // STPP_INHOMO_PPP_H
