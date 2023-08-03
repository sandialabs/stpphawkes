#ifndef TEMPORAL_CONTMARK_COMMON_H
#define TEMPORAL_CONTMARK_COMMON_H

#include <gsl/gsl_rng.h>
#include <vector>

namespace contmark {

double sample_wscale(const std::vector<double>& marks, const std::vector<double>& wscale_param, const double wshape);

}

#endif
