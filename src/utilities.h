#ifndef STPP_UTILITIES_H
#define STPP_UTILITIES_H

#include <RcppArmadillo.h>


void set_seed(unsigned int seed);
arma::mat DFtoMat(Rcpp::DataFrame x);

inline double normalCDF(const double value) { return 0.5 * erfc(-value * M_SQRT1_2); }



#endif  // STPP_UTILITIES_H
