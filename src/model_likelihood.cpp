#include <RcppArmadillo.h>
#include "areapl.h"

#include <Rcpp.h>

// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

// Protect against compilers without OpenMP
#ifdef _OPENMP
// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

static double beta_tk(double t, double beta);
static double Beta_tk(double t, double beta);
static double gamma_k(double x, double y, double sig);
static double gamma_i(double x, double y, double mux, double muy, double sigx, double sigy);

// [[Rcpp::export]]
double temporal_likelihood(arma::vec& t, double mu, double alpha, double beta, double t_max) {
    int n = t.n_elem;
    arma::vec out(n);
    double tmp;
    double s_log_lambda_str;
    arma::uvec ind;
    int n1;

#ifdef _OPENMP
#pragma omp parallel for shared(t, out, mu, alpha, beta, n) private(ind, n1, tmp) default(none) schedule(auto)
#else
#endif
    for (int i = 0; i < n; i++) {
        ind = find(t < t(i));
        n1 = ind.n_elem;
        if (n1 > 0) {
            tmp = 0.0;
            for (int j = 0; j < n1; j++) {
                tmp += beta_tk(t(i) - t(ind(j)), beta);
            }
            out(i) = mu + alpha * tmp;
        } else {
            out(i) = mu;
        }
    }

    s_log_lambda_str = sum(log(out));

    s_log_lambda_str -= mu * t_max;

    tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, beta, n) reduction(+ : tmp)
#else
#endif
    for (int i = 0; i < n; i++) {
        tmp += Beta_tk(t_max - t[i], beta);
    }

    s_log_lambda_str -= alpha * tmp;

    return (s_log_lambda_str);
}

// [[Rcpp::export]]
double stpp_likelihood(arma::vec& x, arma::vec& y, arma::vec& t, arma::mat& poly, double mu, double a, double b, double sig, double t_max) {
  int n = t.n_elem;
  arma::vec out(n);
  double tmp;
  double W = areapl(poly);
  double s_log_lambda_str;
  arma::uvec ind;
  int n1;

#ifdef _OPENMP
#pragma omp parallel for shared(x, y, t, out, mu, a, b, sig, n, W) private(ind, n1, tmp) default(none) schedule(auto)
#else
#endif
  for (int i = 0; i < n; i++) {
    ind = find(t < t(i));
    n1 = ind.n_elem;
    if (n1 > 0) {
      tmp = 0.0;
      for (int j = 0; j < n1; j++) {
                tmp += beta_tk(t(i) - t(ind(j)), b) * gamma_k(x(i) - x(ind(j)), y(i) - y(ind(j)), sig);
      }
            out(i) = mu / W + a * tmp;
    } else {
            out(i) = mu / W;
    }
  }

  s_log_lambda_str = sum(log(out));

  s_log_lambda_str -= mu * t_max;

  tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, b, n) reduction(+ : tmp)
#else
#endif
  for (int i = 0; i < n; i++) {
    tmp += Beta_tk(t_max - t[i], b);
  }

  s_log_lambda_str -= a * tmp;

  return (s_log_lambda_str);
}

// [[Rcpp::export]]
double stpp_likelihood_nonunif(arma::vec& x, arma::vec& y, arma::vec& t, double mu, double a, double b, double sig, double mux, double muy, double sigx, double sigy, double t_max) {
  int n = t.n_elem;
  arma::vec out(n);
  double tmp;
  double tmp0;
  double mu_str;
  double s_log_lambda_str;
  arma::uvec ind;
  int n1;

#ifdef _OPENMP
#pragma omp parallel for shared(x, y, t, out, mu, a, b, sig, mux, muy, sigx, sigy, \
                                n) private(ind, n1, tmp, tmp0, mu_str) default(none) schedule(auto)
#else
#endif
  for (int i = 0; i < n; i++) {
    ind = find(t < t(i));
    n1 = ind.n_elem;
        mu_str = mu * gamma_i(x(i), y(i), mux, muy, sigx, sigy);
        if (mu_str == 0) {
      mu_str = 1E-200;
    }
    if (n1 > 0) {
      tmp = 0.0;
      for (int j = 0; j < n1; j++) {
        tmp0 = beta_tk(t(i) - t(ind(j)), b) * gamma_k(x(i) - x(ind(j)), y(i) - y(ind(j)), sig);
        tmp += tmp0;
      }
      out(i) = mu_str + a * tmp;
    } else {
      out(i) = mu_str;
    }
  }

  s_log_lambda_str = sum(log(out));

  s_log_lambda_str -= mu * t_max;

  tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, b, n) reduction(+ : tmp)
#else
#endif
  for (int i = 0; i < n; i++) {
    tmp += Beta_tk(t_max - t[i], b);
  }

  s_log_lambda_str -= a * tmp;

  return (s_log_lambda_str);
}

double gamma_k(double x, double y, double sig) {
  double out;

    out = 1.0 / (2.0 * M_PI * sig) * exp(-(x * x + y * y) / (2 * sig));

  return (out);
}

double gamma_i(double x, double y, double mux, double muy, double sigx, double sigy) {
  double out;

  double sigxy = sqrt(sigx * sigy);
  out = 1.0 / (2.0 * M_PI * sigxy) * exp(-((x - mux) * (x - mux) / sigx + (y - muy) * (y - muy) / sigy) / 2);

  return (out);
}

double beta_tk(double t, double beta) {
    double out;
    if ((t >= 0) & (beta > 0)) {
        out = beta * exp(-beta * t);
    } else {
        out = 0.0;
    }

    return (out);
}

double Beta_tk(double t, double beta) {
    double out;
    out = 1 - exp(-beta * t);
    return (out);
}
