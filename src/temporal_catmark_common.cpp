#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppGSL)]]
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <RcppGSL.h>
#include "helper_functions.h"

namespace catmark {
double betaPosterior(const std::vector<double>& t, const std::vector<double>& z, const std::vector<int>& numtriggered,
                     const double t_max, const double alpha_curr, const double beta, const double beta_a,
                     const double beta_b) {
    // The overall equation we are evaluating here is:
    // loglik = log(beta)*sum(ntriggered)  (Part 3)
    //        + alpha*(sum exp(-beta*(t_max-t_i)) -N) (Part 1)
    //        - beta* sum( z)   (Part 2)

    // Begin calculating Part 1: Partition up the exponentials
    if (alpha_curr < beta) {
        int n = t.size();

        double epsilon = t_max + 1 / beta * (-36 - log(alpha_curr));

        int min_i;
        for (min_i = n - 1; min_i >= 0; --min_i) {
            if (t[min_i] < epsilon) {
                break;
            }
        }
        // Part 1 Calculation:

        double loglik = 0;
        // Compute -1*alpha* sum(Beta_tk(t_max - t_i, beta)
        // Re-write this as alpha* (sum( exp(beta*(t_i - t_max)))  - n)
        for (int i = n - 1; i >= min_i; --i) {
            loglik += std::exp(beta * (t[i] - t_max));
        }

        loglik -= n;
        loglik *= alpha_curr;

        // Part 2 Calculation:

        // Next, we compute the term beta*sum(z)
        double loglik2 = 0;
        // OpenMP seems to be overkill for this loop
        for (const auto zval : z) {
            loglik2 -= zval;
        }

        loglik2 *= beta;

        // Part 3 Calculation:

        // Add in log(beta) * sum(ntriggered)
        loglik += std::log(beta) * z.size() + loglik2;

        // Calculate prior ber @jrlewi's code
        // prior = dgamma(beta, beta_a, beta_b, log = TRUE);
        // http://dirk.eddelbuettel.com/code/rcpp/html/namespaceR.html#a673235717f0c019f9d2c9b5528847e20
        double prior = R::dgamma(beta, beta_a, 1.0 / beta_b, 1);

        return loglik + prior;
    } else {
        return -INFINITY;
    }
}

// Multinomial Mark Distribution

std::vector<int> countMarks(const std::vector<int>& marks, const size_t kk) {
    // Identify the marks, count them up, and store that in a vector.
    // Example: marks = {A,A,B,A,C,D,C}
    // Desired Output: A: 3, B: 1, C: 2, D : 1

    // With Integer Labels
    // Example: marks = {1,1,2,3,4,1,2,3}
    // Desired Output: {3,2,2,1} which is equivalent to 1: 3, 2:2, 3:2, 4:1 (mark : count)

    // Key is the "mark" name, and value is the count associated with that mark
    std::map<int, size_t> mark_counts;

    // initalize mark_counts to kk
    for (size_t ii = 1; ii <= kk; ++ii) {
        mark_counts[ii] = 0;
    }

    for (const auto mark : marks) {
        mark_counts[mark]++;
    }

    std::vector<int> counts;
    for (const auto& kv : mark_counts) {
        counts.emplace_back(kv.second);
    }
    return counts;
}

std::vector<double> sampleP(const std::vector<int>& marks, const std::vector<double>& p_param, gsl_rng* rng) {
    // Convert Marks vector into a vectorized map of counts
    size_t kk = p_param.size();
    auto number_of_counts_per_mark = countMarks(marks, kk);
    std::vector<double> dirichlet_parameters(number_of_counts_per_mark.size());
    if (p_param.size() != number_of_counts_per_mark.size()) {
        throw;
    }
    for (size_t i = 0; i < kk; ++i) {
        dirichlet_parameters[i] += number_of_counts_per_mark[i] + p_param[i];
    }

    std::vector<double> results(kk);

    // Allocate random number generator
    gsl_ran_dirichlet(rng, kk, dirichlet_parameters.data(), results.data());
    // Release random number generator

    return results;
}

double sampleBeta(const double alpha_curr, double beta_curr, const double t_max, const double sig_beta,
                  const std::vector<double>& t, const std::vector<double>& z, const std::vector<int>& numtriggered,
                  const double beta_a, const double beta_b) {
    auto gen = GenerateMersenneTwister();

    std::normal_distribution<> rnorm(0, sig_beta);
    std::uniform_real_distribution<> runif(0, 1);

    double top;
    double bottom = betaPosterior(t, z, numtriggered, t_max, alpha_curr, beta_curr, beta_a, beta_b);
    double beta_prop;
    for (size_t i = 0; i < 100; i++) {
        beta_prop = beta_curr + rnorm(gen);  // Sample from normal distribution
        if (beta_prop > 0) {
            top = betaPosterior(t, z, numtriggered, t_max, alpha_curr, beta_prop, beta_a, beta_b);
            if (runif(gen) < std::exp(top - bottom)) {
                beta_curr = beta_prop;
                bottom = top;
            }
        }
    }

    return beta_curr;
}

}  // end namespace catmark
