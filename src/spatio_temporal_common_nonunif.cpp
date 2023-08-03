#include "spatio_temporal_common_nonunif.h"
#include "utilities.h"
#include "vector_utilities.h"
#include "areapl.h"

namespace stpp_nonunif {

double sample_a_accumulate(const std::vector<double>& t, double t_max, double b_curr) {
  size_t n = t.size();
  double Ba = 0;

  // sum_{i=0}^{n-1} Beta_tk = \sum (1 - exp(-beta*(t_max-t_i)))
  // = n - sum exp(-beta*(t_max - t_i));
  // When exp(-beta*(t_max - t_i)) <= 1e-16, this is no longer relevant
  // This is equivalent to when t_max - t_i <= ln(1e-16)/beta = 36/beta;
  double epsilon = t_max - 36.0 / b_curr;
    // int min_relevant_time = findMinimumRelevantTime(t, epsilon);
    // for (size_t i = min_relevant_time; i < n; ++i) {
  for (size_t i = 0; i < n; ++i) {
    Ba -= std::exp(-b_curr * (t_max - t[i]));
  }

  Ba += n;

  return Ba;
}

double sample_a(const std::vector<double>& t, const std::vector<double>& z_t, double t_max, const double a_curr,
                double b_curr, const std::vector<double>& a_param) {
  double Ba = sample_a_accumulate(t, t_max, b_curr);
  auto gen = GenerateMersenneTwister();
  std::gamma_distribution<> rgamma(a_param[0] + z_t.size(), 1 / (a_param[1] + Ba));
  return rgamma(gen);
}

double sample_b(const std::vector<double>& t, const std::vector<double>& z_t, const double t_max, const double a_curr,
                double b_curr, const double sig_b, const std::vector<double>& b_params) {
  auto gen = GenerateMersenneTwister();
  std::normal_distribution<> rnorm(0, sig_b);
  std::uniform_real_distribution<> runif(0, 1);
  double bottom = b_posterior(t, t_max, a_curr, b_curr, z_t, b_params);

  double b_prop = b_curr + rnorm(gen);
  while (b_prop < a_curr) {
    b_prop = b_curr + rnorm(gen);
  }
  double top = b_posterior(t, t_max, a_curr, b_prop, z_t, b_params);
  double rat =
    std::exp(top - bottom) * (1 - normalCDF(a_curr - b_curr / sig_b)) / (1 - normalCDF(a_curr - b_prop / sig_b));
  if (runif(gen) < rat) {
    b_curr = b_prop;
  }

  return b_curr;
}

// sample mux or muy
double sample_muxy(const std::vector<double>& xpa, const int nparents, const double sigx,
                   const std::vector<double>& mux_params) {
  double sum_xpa = 0;
    for (int i = 0; i < nparents; ++i) {
    sum_xpa += xpa[i];
  }
    double mux_var = 1 / (1 / mux_params[1] + nparents / sigx);
    double mux_mean = mux_var * (mux_params[0] / mux_params[1] + sum_xpa / sigx);
  auto gen = GenerateMersenneTwister();
  std::normal_distribution<> rnorm(mux_mean, sqrt(mux_var));
  double out;
  out = rnorm(gen);
  return out;
}

// sample variances sigx or sigy
double sample_sigxy(const std::vector<double>& xpa, const int nparents, const double mux,
                    const std::vector<double>& sigx_params) {
  double sumsq_xpa = 0;
    for (int i = 0; i < nparents; ++i) {
        sumsq_xpa += (xpa[i] - mux) * (xpa[i] - mux);
  }

  auto gen = GenerateMersenneTwister();
  std::gamma_distribution<> rgamma(sigx_params[0] + nparents / 2.0,
                                   1.0 / (sigx_params[1] + sumsq_xpa / 2.0));  // scale parameter
  double sigx_new = rgamma(gen);
  sigx_new = 1.0 / sigx_new;
  return (sigx_new);
}

std::vector<int> sample_y(const std::vector<double>& t, const std::vector<double>& x, const std::vector<double>& y,
                          const double mu_curr, const double a_curr, const double b_curr, const double sig_curr,
                          const double mux_curr, const double muy_curr, const double sigx_curr,
                          const double sigy_curr) {
  std::vector<int> y_curr;
  int n = t.size();
  y_curr.resize(n);
  y_curr[0] = 0;

  // Find minimum relevant times

  double epsilon = 36.0 / b_curr;
    // const std::vector<int> min_relevant_times = findMinimumRelevantTimes(t,
    // epsilon);

  const double sigxy = sqrt(sigx_curr * sigy_curr);
  const double scale_factor_I = 1.0 / (2 * sigxy * M_PI);
  const double one_over_two_sigx = 1.0 / (2 * sigx_curr);
  const double one_over_two_sigy = 1.0 / (2 * sigy_curr);
  const double scale_factor = a_curr * b_curr * 1.0 / (2 * M_PI * sig_curr);
  const double one_over_two_sig = 1.0 / (2 * sig_curr);

#ifdef _OPENMP
#pragma omp parallel for
#else
#endif
  for (int i = 1; i < n; ++i) {
    auto gen = GenerateMersenneTwister();

    std::vector<double> probs(i + 1, 0);

    const double t_i = t[i];
    const double x_i = x[i];
    const double y_i = y[i];

    double mu_str = mu_curr * scale_factor_I * std::exp(-(one_over_two_sigx * (x_i - mux_curr) * (x_i - mux_curr) +
                                                        one_over_two_sigy * (y_i - muy_curr) * (y_i - muy_curr)));
    probs[0] = mu_str;

    // int min_relevant_time = min_relevant_times[i];
    // for (int j = min_relevant_time; j < i; ++j) {
    for (int j = 0; j < i; ++j) {
      probs[j + 1] =
        scale_factor * std::exp(-b_curr * (t_i - t[j]) -
        one_over_two_sig * ((x_i - x[j]) * (x_i - x[j]) + (y_i - y[j]) * (y_i - y[j])));
    }

    std::discrete_distribution<> d(probs.begin(), probs.end());

    int parent = d(gen);
    y_curr[i] = parent;
   }

  return y_curr;
}

double b_posterior(const std::vector<double>& t, double t_max, double a, double b, const std::vector<double>& z_t,
                   const std::vector<double>& b_param) {
  if (b < a) {
    return (-INFINITY);
  }
  size_t n = t.size();

  double epsilon = t_max - 38 / b;
  // int min_relevant_time = findMinimumRelevantTime(t, epsilon);

  double loglik = 0;
  // for (size_t i = min_relevant_time; i < n; ++i) {
  for (size_t i = 0; i < n; ++i) {
    // ORIGINAL loglik -= a * Beta_tk(t_max - t[i], b);
    // loglik -=  Beta_tk(t_max - t[i], b);
    loglik -= std::exp(-b * (t_max - t[i]));
  }

  loglik += n;
  loglik *= -a;

  // sum log(beta_tk(x,b)) = sum log(b * exp(-b*x)) = N*log(b) + -b*sum(x)
  double part_two_sum = 0.0;

  part_two_sum = std::accumulate(z_t.begin(), z_t.end(), 0.0);

  part_two_sum *= -b;
  part_two_sum += z_t.size() * std::log(b);

  loglik += part_two_sum;

  loglik += ((b_param[0] - 1) * std::log(b) - b * b_param[1]);
  return loglik;
}

double sig_posterior(double sig, const std::vector<double>& z_x, const std::vector<double>& z_y,
                     const std::vector<double>& sig_param) {
  double loglik = 0;
  for (size_t i = 0; i < z_x.size(); ++i) {
    // loglik += std::log(gamma_k(z_x[i], z_y[i], sig));
    loglik -= (z_x[i] * z_x[i] + z_y[i] * z_y[i]);
  }

  loglik *= 1.0 / (2 * sig);

  loglik += z_x.size() * std::log(1 / (2 * M_PI * sig));

  loglik += (-sig_param[0] - 1) * std::log(sig) - sig / sig_param[1];  // input params are shape+rate of gamma,
                                                                       // inverse-gamma=1/gamma
  return loglik;
}

double sample_sig(const std::vector<double>& z_x, const std::vector<double>& z_y, double sig_curr, const double sig_sig,
                  const std::vector<double>& sig_param) {
  auto gen = GenerateMersenneTwister();
  std::normal_distribution<> rnorm(0, sig_sig);
  std::uniform_real_distribution<> runif(0, 1);

  double bottom = sig_posterior(sig_curr, z_x, z_y, sig_param);
  double sig_prop = sig_curr + rnorm(gen);
  while (sig_prop < 0) {
    sig_prop = sig_curr + rnorm(gen);
  }

  double top = sig_posterior(sig_prop, z_x, z_y, sig_param);
  double rat = std::exp(top - bottom) * (1 - normalCDF(-sig_curr / sig_sig)) / (1 - normalCDF(-sig_prop / sig_sig));

  if (runif(gen) < rat) {
    sig_curr = sig_prop;
  }
  return sig_curr;
}

// likelihood should now depend on branching structure??
namespace missing_data {
double log_lik(std::vector<double>& x, std::vector<double>& y, std::vector<double>& t, double mu, double a, double b,
               double sig, double mux, double muy, double sigx, double sigy, double t_max) {
   int n = t.size();

   double temp0;
   double temp = 0.0;

   const double sigxy = sqrt(sigx * sigy);
   const double scale_factor_I = 1.0 / (2 * sigxy * M_PI);
   const double one_over_two_sigx = 1.0 / (2 * sigx);
   const double one_over_two_sigy = 1.0 / (2 * sigy);
   double mu_str = mu * scale_factor_I;
    double part1 = std::log(mu_str) -
                   (one_over_two_sigx * (x[0] - mux) * (x[0] - mux) + one_over_two_sigy * (y[0] - muy) * (y[0] - muy));

   double epsilon = 36.0 / b;
   // std::vector<int> min_relevant_times = findMinimumRelevantTimes(t, epsilon);

   const double scale_factor = a * b / (2 * sig * M_PI);
   const double one_over_two_sig = 1.0 / (2 * sig);
#ifdef _OPENMP
#pragma omp parallel for shared(x, y, t, mu, a, b, sig, n) private(temp) reduction(+ : part1)
#else
#endif
   for (int i = 1; i < n; ++i) {
     const double t_i = t[i];
     const double x_i = x[i];
     const double y_i = y[i];
      temp0 = std::exp(-(one_over_two_sigx * (x_i - mux) * (x_i - mux) + one_over_two_sigy * (y_i - muy) * (y_i - muy)));
     // int min_relevant_time = min_relevant_times[i];
     temp = 0.0;
     for (int j = 0; j < i; ++j) {
       temp += std::exp(-b * (t_i - t[j]) -
         one_over_two_sig * ((x_i - x[j]) * (x_i - x[j]) + (y_i - y[j]) * (y_i - y[j])));
     }
     part1 += log(mu_str * temp0 + scale_factor * temp);
   }

   double part2 = mu * t_max;

   double part3 = 0;

    // int min_relevant_time = findMinimumRelevantTime(t, t_max - 36.0 / b);
   for (int i = 0; i < n; ++i) {
     // part3 -= exp(-b * (t_max - t[i]));
     part3 += Beta_tk(t_max - t[i], b);
   }
    // part3 += n;
   part3 *= a;

   return (part1 - part2 - part3);
 }

}  // end namespace missing_data

}  // end namespace stpp
