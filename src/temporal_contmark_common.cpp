#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppGSL)]]
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <RcppGSL.h>
#include "helper_functions.h"

namespace contmark {


double sample_wscale(const std::vector<double>& marks, const std::vector<double>& wscale_param, const double wshape) {
  double alpha_wscale=wscale_param[0]+marks.size();
  double summark=0;
  for(int i = 0; i < marks.size(); ++i){
    summark+=pow(marks[i],wshape);
  }
  double beta_wscale=wscale_param[1]+summark;

  auto gen = GenerateMersenneTwister();
  std::gamma_distribution<> rgamma(alpha_wscale, 1.0/beta_wscale);
  double gam_samp = rgamma(gen);

  double out = 1/gam_samp;
  // Extract out the estimated parameter values

  // Return sampled wscale
  return out;

}


}  // end namespace catmark
