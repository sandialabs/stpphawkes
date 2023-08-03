#ifndef TEMPORAL_CATMARK_COMMON_H
#define TEMPORAL_CATMARK_COMMON_H

#include <gsl/gsl_rng.h>
#include <vector>

namespace catmark {

  double betaPosterior(const std::vector<double>& t, const std::vector<double>& z, const std::vector<int>& numtriggered,
                       const double t_max, const double alpha_curr, const double beta, const double beta_a,
                       const double beta_b);

  std::vector<double> sampleP(const std::vector<int>& marks, const std::vector<double>& p_param, gsl_rng* rng);

  double sampleAlpha(const std::vector<double>& t, int sum_numtriggered, const double t_max, const double beta_curr,
                     const double alpha_a, const double alpha_b);

  double sampleBeta(const double alpha_curr, double beta_curr, const double t_max, const double sig_beta,
                    const std::vector<double>& t, const std::vector<double>& z, const std::vector<int>& numtriggered,
                    const double beta_a, const double beta_b);

  std::vector<int> sampleY(double alpha_curr, double beta_curr, double mu_curr, const std::vector<double>& t_tmp);

  std::vector<int> countMarks(const std::vector<int>& marks, const size_t kk);
}

#endif
