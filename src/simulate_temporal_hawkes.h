#ifndef STPP_SIMULATE_TEMPORAL_HAWKES_H
#define STPP_SIMULATE_TEMPORAL_HAWKES_H

#include "RcppArmadillo.h"

double intensity_temporal(double mu, double alpha, double beta, arma::vec& times, double evalpt);

arma::vec simulate_temporal(const double mu, double alpha, const double beta, const arma::vec& tt,
                            const arma::vec& times, int seed = -1);

#endif  // STPP_SIMULATE_TEMPORAL_HAWKES_H
