#include <celero/Celero.h>

#include <algorithm>
#include <iostream>
#include "helper_functions.h"
#include "spatio_temporal_common.h"

CELERO_MAIN;

class StppPerfTest : public celero::TestFixture {
protected:
    StppPerfTest() {}

    std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override { return {{0, 0}}; }

    void setUp(int64_t experimentValue) override {
        for (size_t i = 0; i < 500; ++i) {
            t.push_back(i / 5.0);
            xvec.push_back(i / 500.0);
        }
        t_max = *std::max_element(t.begin(), t.end());
        sig_param.push_back(1.0);
        sig_param.push_back(2.0);
        b_param.push_back(1.0);
        b_param.push_back(1.2);
    }
    std::vector<double> t;
    std::vector<double> xvec;
    std::vector<double> sig_param;
    std::vector<double> b_param;
    double x = .5;
    double y = .4;
    double sig = 1.2;
    double a_curr = 1.1;
    double b_curr = 1.3;
    double mu_curr = .8;
    double sig_curr = 1.4;
    double t_max;
};

inline double gamma_k_slow(const double x, const double y, const double sig) {
    return 1 / (2 * M_PI * sig) * exp(-(x * x + y * y) / (2 * sig));
}

double sig_posterior_slow(double sig, const std::vector<double>& z_x, const std::vector<double>& z_y,
                          const std::vector<double>& sig_param) {
    double loglik = 0;
#ifdef _OPENMP
#pragma omp parallel for shared(z_x, z_y, sig) reduction(+ : loglik)
#else
#endif
    for (size_t i = 0; i < z_x.size(); ++i) {
        loglik += std::log(stpp::gamma_k(z_x[i], z_y[i], sig));
    }

    loglik += (-sig_param[0] - 1) * std::log(sig) -
              sig / sig_param[1];  // input params are shape+rate of gamma, inverse-gamma=1/gamma
    return loglik;
}

double b_posterior_slow(const std::vector<double>& t, double t_max, double a, double b, const std::vector<double>& z_t,
                        const std::vector<double>& b_param) {
    if (b < a) {
        return (-INFINITY);
    }
    double loglik = 0;
    size_t n = t.size();
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, a, b, n) reduction(- : loglik)
#else
#endif
    for (size_t i = 0; i < n; ++i) {
        loglik -= a * Beta_tk(t_max - t[i], b);
    }

#ifdef _OPENMP
#pragma omp parallel for shared(z_t, b) reduction(+ : loglik)
#else
#endif
    for (size_t i = 0; i < z_t.size(); ++i) {
        loglik += std::log(beta_tk(z_t[i], b));
    }

    loglik += ((b_param[0] - 1) * std::log(b) - b * b_param[1]);
    return loglik;
}

#define PERFORMANCE_TEST_RUN_LOL 1
#define RUN_GAMMA_K_TESTS 0
#define RUN_SIG_POSTERIOR_TESTS 0
#define RUN_BETA_POSTERIOR_TESTS 0
#define RUN_SAMPLE_A_TESTS 0
#define RUN_SAMPLE_Y_TESTS 0
#define RUN_LOG_LIK_TESTS 1

#if PERFORMANCE_TEST_RUN_LOL

#define NUM_SAMPLES_YO 30
#define NUM_ITERATIONS_BRO 100
#define PRINT_STUFF 0

#else
#define NUM_SAMPLES_YO 1
#define NUM_ITERATIONS_BRO 1
#define PRINT_STUFF 1

#endif

#if RUN_GAMMA_K_TESTS
BASELINE_F(GammaTests, GammaKFunction, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = gamma_k_slow(x, y, sig));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}
BENCHMARK_F(GammaTests, GammaKFunctionFast, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = stpp::gamma_k(x, y, sig));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}

#endif  // RUN_GAMMA_K_TESTS

#if RUN_SIG_POSTERIOR_TESTS
BASELINE_F(SigTests, SigPosterior, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = stpp::sig_posterior(sig, t, t, sig_param));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}
BENCHMARK_F(SigTests, SigPosteriorSlow, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = sig_posterior_slow(sig, t, t, sig_param));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}

#endif  // RUN_SIG_POSTERIOR_TESTS

#if RUN_BETA_POSTERIOR_TESTS

BASELINE_F(BetaPosterior, b_posterior_slow, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = b_posterior_slow(t, t_max, 1.0, 1.5, t, b_param));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}
BENCHMARK_F(BetaPosterior, b_posterior, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = stpp::b_posterior(t, t_max, 1.0, 1.5, t, b_param));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}

#endif  // RUN_BETA_POSTERIOR_TESTS

double sample_a_slow(const std::vector<double>& t, double t_max, double b_curr) {
    size_t n = t.size();
    double Ba = 0;
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, b_curr, n) reduction(+ : Ba)
#else
#endif
    for (size_t i = 0; i < n; ++i) {
        Ba += Beta_tk(t_max - t[i], b_curr);
    }

    return Ba;
}

#if RUN_SAMPLE_A_TESTS

BASELINE_F(SampleA, sample_a_slow, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = sample_a_slow(t, t_max, b_curr));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}
BENCHMARK_F(SampleA, sample_a, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans = 0;
    celero::DoNotOptimizeAway(ans = stpp::sample_a_accumulate(t, t_max, b_curr));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}

#endif  // RUN_SAMPLE_A_TESTS

std::vector<int> sample_y_slow(const std::vector<double>& t, const std::vector<double>& x, const std::vector<double>& y,
                               const double mu_curr, const double a_curr, const double b_curr, const double sig_curr) {
    std::vector<int> y_curr;
    int n = t.size();
    y_curr.resize(n);
    y_curr[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for
#else
#endif
    for (int i = 1; i < n; i++) {
        auto gen = GenerateMersenneTwister();
        double temp;

        std::vector<double> probs;
        probs.reserve(n);
        probs.push_back(mu_curr);

        for (int j = 0; j < i; j++) {
            temp = a_curr * beta_tk(t[i] - t[j], b_curr) * stpp::gamma_k(x[i] - x[j], y[i] - y[j], sig_curr);
            probs.push_back(temp);
        }

        std::discrete_distribution<> d(probs.begin(), probs.end());

        int parent = d(gen);
        y_curr[i] = parent;
    }

    return y_curr;
}

#if RUN_SAMPLE_Y_TESTS

BASELINE_F(SampleA, sample_y_slow, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    std::vector<int> ans;
    celero::DoNotOptimizeAway(ans = sample_y_slow(t, xvec, xvec, mu_curr, 100.0, b_curr, sig_curr));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << std::accumulate(ans.begin(), ans.end(), 0) << std::endl;
#endif
}
BENCHMARK_F(SampleA, sample_y, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    std::vector<int> ans;
    celero::DoNotOptimizeAway(ans = stpp::sample_y(t, xvec, xvec, mu_curr, 100.0, b_curr, sig_curr));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << std::accumulate(ans.begin(), ans.end(), 0) << std::endl;
#endif
}

#endif  // RUN_SAMPLE_Y_TESTS

double log_lik_slow(std::vector<double>& x, std::vector<double>& y, std::vector<double>& t, double mu, double a,
                    double b, double sig, double t_max, const double W) {
    int n = t.size();

    double part1, part2, part3, temp;

    part1 = std::log(mu);
#ifdef _OPENMP
#pragma omp parallel for shared(x, y, t, mu, a, b, sig, n) private(temp) reduction(+ : part1)
#else
#endif
    for (int i = 1; i < n; i++) {
        temp = mu;
        for (int j = 0; j < i; j++) {
            temp += a * beta_tk(t[i] - t[j], b) * stpp::gamma_k(x[i] - x[j], y[i] - y[j], sig);
        }
        part1 += log(temp);
    }

    part2 = mu * t_max * W;

    part3 = 0;
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, a, b, n) reduction(+ : part3)
#else
#endif
    for (int i = 0; i < n; i++) {
        part3 += a * Beta_tk(t_max - t[i], b);
    }

    return (part1 - part2 - part3);
}

#if RUN_LOG_LIK_TESTS

BASELINE_F(LogLikelihoodTests, log_lik_slow, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans;
    celero::DoNotOptimizeAway(ans = log_lik_slow(xvec, xvec, t, mu_curr, a_curr, b_curr, sig_curr, t_max, 12.0));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}
BENCHMARK_F(LogLikelihoodTests, log_lik, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double ans;
    celero::DoNotOptimizeAway(
        ans = stpp::missing_data::log_lik(xvec, xvec, t, mu_curr, a_curr, b_curr, sig_curr, t_max, 12.0));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "The ans is " << ans << std::endl;
#endif
}

#endif  // RUN_LOG_LIK_TESTS