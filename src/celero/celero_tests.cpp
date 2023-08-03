#include <celero/Celero.h>

#include <algorithm>
#include <iostream>
#include "condInt_mcmc_temporal_branching_md.h"
#include "helper_functions.h"
#ifdef COMPILE_CELERO_TESTS

CELERO_MAIN;

class StppPerfTest : public celero::TestFixture {
    // class StppPerfTest {
protected:
    // public:

    StppPerfTest() {}

    std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override { return {{0, 0}}; }

    void setUp(int64_t experimentValue) override {
        for (size_t i = 0; i < 500; ++i) {
            t.push_back(i / 50.0);
        }
    }

    double log_lik(const std::vector<double>& t, double mu, double alpha, double beta, double t_max) {
        int n = t.size();

        double temp;

        double part1 = log(mu);
#ifdef _OPENMP
#pragma omp parallel for shared(t, mu, alpha, beta, n) private(temp) reduction(+ : part1)
#else
#endif
        for (int i = 1; i < n; i++) {
            temp = mu;
            for (int j = 0; j < i; j++) {
                temp += alpha * beta_tk(t[i] - t[j], beta);
            }
            part1 += log(temp);
        }

        double part2 = mu * t_max;

        double part3 = 0;
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, alpha, beta, n) reduction(+ : part3)
#else
#endif
        for (int i = 0; i < n; i++) {
            part3 += alpha * Beta_tk(t_max - t[i], beta);
        }

        return (part1 - part2 - part3);
    }

    double log_lik_break(const std::vector<double>& t, double mu, double alpha, double beta, double t_max) {
        int n = t.size();

        double part1 = 0;

        const double alpha_beta_product = alpha * beta;

        std::vector<size_t> min_is(n);
        min_is[0] = 0;

        for (int i = 1; i < n; ++i) {
            //@todo srowe: Also consider walking forward from min_i[i-1]
            double minimum_time = t[i] - 30 / beta;

            if (minimum_time < 0) {
                min_is[i] = 0;
            } else {
                int min_i;
                for (min_i = min_is[i - 1]; min_i < i; ++min_i) {
                    if (t[min_i] > minimum_time) {
                        break;
                    }
                }

                min_is[i] = min_i;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for shared(t, mu, alpha, beta, n, min_is) reduction(+ : part1)
#else
#endif
        for (int i = 1; i < n; i++) {
            double temp = 0;
            const double t_i = t[i];
            for (int j = min_is[i]; j < i; ++j) {
                temp += exp(-beta * (t_i - t[j]));
            }
            part1 += log(alpha_beta_product * temp + mu);
        }
        part1 += log(mu);

        double part2 = mu * t_max;

        double part3 = 0;

        double epsilon = 1e-15;

        double minimum_time = log(epsilon) / beta + t_max;
        size_t min_i;
        for (min_i = n - 1; min_i >= 0; --min_i) {
            if (t[min_i] < minimum_time) {
                break;
            }
        }

// Find first index of time walking backwards where this minimum time occurs;

#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, alpha, beta, n) reduction(+ : part3)
#else
#endif
        for (int i = min_i + 1; i < n; i++) {
            part3 += alpha * Beta_tk(t_max - t[i], beta);
        }
        part3 += (min_i + 1) * alpha;

        return (part1 - part2 - part3);
    }

    double mu = .5;
    double alpha = .9;
    double beta = 14.768;
    double t_max = 100;
    std::vector<double> t;
    std::vector<long double> tlong;

    std::vector<double> t_y_test{
        0.481305, 1.44156, 1.77284, 1.82171, 3.51204, 3.57801, 3.59475, 3.60665, 3.6494,  3.65586, 3.70341, 3.71742,
        3.73639,  3.73729, 3.79619, 3.80896, 3.81536, 3.81833, 3.82625, 3.82938, 3.83127, 3.83711, 3.87083, 3.87206,
        3.87292,  3.89881, 3.90684, 3.92419, 3.93074, 3.95098, 3.95111, 3.95293, 3.95995, 3.98144, 4.00502, 4.0089,
        4.03402,  4.03881, 4.07318, 4.08133, 4.09287, 4.14017, 4.19463, 4.20433, 4.23339, 4.23624, 4.38049, 4.53713,
        4.53881,  4.55417, 4.56676, 4.60646, 4.60775, 4.64287, 4.68863, 4.70935, 4.71661, 4.72065, 4.7266,  4.75352,
        4.76531,  4.79299, 4.79649, 4.83907, 4.84225, 4.86099, 4.87258, 4.89156, 4.89543, 4.89596, 4.89924, 4.9493,
        4.95484,  4.99947, 5.01171, 5.01334, 5.02319, 5.02856, 5.07625, 5.13449, 5.24221, 5.28383, 5.31296, 5.34891,
        5.39757,  5.4303,  5.44652, 5.45322, 5.90533, 5.9069,  5.99492, 6.05788, 6.05812, 6.06539, 6.14791, 6.16696,
        6.23217,  8.69357, 8.69546, 8.73127, 9.9571,  31.1508, 31.2521, 31.2652, 31.3588, 31.4224, 31.4395, 31.4404,
        31.5361,  31.5386, 31.5654, 31.6835, 32.0381, 32.0428, 32.179,  32.3284, 32.4477, 32.4571, 32.4739, 32.4833,
        32.508,   32.5113, 32.5451, 32.5528, 32.558,  32.5623, 32.5672, 32.5781, 32.6145, 32.6289, 32.6344, 32.6749,
        32.7217,  32.7341, 32.7686, 32.8156, 32.9184, 33.004,  33.0246, 36.8765, 36.904,  38.3209, 42.1813, 42.2354,
        42.2677,  42.2887, 42.3032, 42.3272, 42.3326, 42.37,   42.4353, 42.471,  42.4714, 42.5452, 42.5982, 42.606,
        42.6168,  42.6229, 42.6369, 42.6373, 42.6489, 42.656,  42.6573, 42.6626, 42.6754, 42.7138, 42.7176, 42.7211,
        42.7229,  42.7249, 42.7363, 42.7578, 42.7653, 42.7775, 42.8039, 42.81,   42.8156, 42.8181, 42.8234, 42.862,
        42.8778,  42.8843, 42.8864, 42.8944, 42.9103, 42.9168, 42.9306, 42.9553, 42.9779, 42.9798, 43.0204, 43.04,
        43.0897,  43.1258, 43.1606, 43.1699, 43.2171, 43.2348, 43.3374, 43.3381, 43.3512, 43.3734, 43.3799, 43.4101,
        43.4391,  43.4413, 43.5057, 43.546,  43.5487, 43.5929, 43.6123, 43.6376, 43.6476, 43.656,  43.6623, 43.6745,
        43.7019,  43.7048, 43.7243, 43.7474, 43.8006, 43.802,  43.816,  43.8553, 44.0187, 44.1282, 71.5453, 71.627,
        71.6518,  71.7164, 71.8052, 72.0752, 78.5647, 79.0447, 79.0572, 79.0602, 87.0648, 87.0776, 87.0831, 87.1379,
        87.6084,  87.9361, 87.9453, 90.3554, 90.3735, 90.4196, 90.4755, 90.736,  90.8395, 90.8888, 90.9167, 90.9896,
        91.0284,  91.0547, 91.0792, 91.0856, 91.1564, 91.1874, 91.2767, 91.3509, 91.3606, 91.3694, 91.3846, 91.4163,
        91.4716,  91.4999, 91.5104, 91.5955, 91.6377, 91.7239, 91.7351, 91.8266, 91.8809, 91.9032, 92.1026, 92.2009,
        92.6741,  92.725,  92.7861, 92.7923, 92.7939, 92.8139, 92.8361, 92.8395, 92.8399, 92.8693, 92.8732, 92.8892,
        92.9098,  92.9599, 92.9645, 92.9943, 93.0287, 93.0655, 93.163,  93.2169, 93.2483, 93.2788, 93.3242, 93.3357,
        93.3962,  93.4512, 93.4865, 93.4967, 93.5075, 93.5618, 93.5687, 93.5998, 93.6258, 93.6656, 93.7532, 93.7863,
        93.8326,  93.8569, 93.9061, 94.045,  94.0737, 94.0897, 94.139,  94.1548, 94.1703, 94.1878, 94.2535, 94.2686,
        94.2745,  94.2857, 94.2886, 94.2888, 94.2892, 94.2894, 94.2919, 94.2926, 94.3315, 94.3428, 94.3479, 94.3531,
        94.3548,  94.3572, 94.39,   94.3915, 94.393,  94.4169, 94.4243, 94.4443, 94.4462, 94.4732, 94.4751, 94.5021,
        94.5749,  94.801,  95.0073, 95.1161, 95.1707, 95.1795, 95.2221, 95.2224, 95.2351, 95.2384, 95.2423, 95.3655,
        95.6047,  97.9832, 98.8664, 98.8851,
    };

    std::vector<double> z_curr{
        50.5633, 52.971, 55.3788, 57.7866, 60.1944, 62.6021, 65.0099, 67.4177, 69.8255,
    };

    std::vector<double> t_beta{
        0.481305, 1.44156, 1.77284, 1.82171, 3.51204, 3.57801, 3.59475, 3.60665, 3.6494,  3.65586, 3.70341, 3.71742,
        3.73639,  3.73729, 3.79619, 3.80896, 3.81536, 3.81833, 3.82625, 3.82938, 3.83127, 3.83711, 3.87083, 3.87206,
        3.87292,  3.89881, 3.90684, 3.92419, 3.93074, 3.95098, 3.95111, 3.95293, 3.95995, 3.98144, 4.00502, 4.0089,
        4.03402,  4.03881, 4.07318, 4.08133, 4.09287, 4.14017, 4.19463, 4.20433, 4.23339, 4.23624, 4.38049, 4.53713,
        4.53881,  4.55417, 4.56676, 4.60646, 4.60775, 4.64287, 4.68863, 4.70935, 4.71661, 4.72065, 4.7266,  4.75352,
        4.76531,  4.79299, 4.79649, 4.83907, 4.84225, 4.86099, 4.87258, 4.89156, 4.89543, 4.89596, 4.89924, 4.9493,
        4.95484,  4.99947, 5.01171, 5.01334, 5.02319, 5.02856, 5.07625, 5.13449, 5.24221, 5.28383, 5.31296, 5.34891,
        5.39757,  5.4303,  5.44652, 5.45322, 5.90533, 5.9069,  5.99492, 6.05788, 6.05812, 6.06539, 6.14791, 6.16696,
        6.23217,  8.69357, 8.69546, 8.73127, 9.9571,  31.1508, 31.2521, 31.2652, 31.3588, 31.4224, 31.4395, 31.4404,
        31.5361,  31.5386, 31.5654, 31.6835, 32.0381, 32.0428, 32.179,  32.3284, 32.4477, 32.4571, 32.4739, 32.4833,
        32.508,   32.5113, 32.5451, 32.5528, 32.558,  32.5623, 32.5672, 32.5781, 32.6145, 32.6289, 32.6344, 32.6749,
        32.7217,  32.7341, 32.7686, 32.8156, 32.9184, 33.004,  33.0246, 36.8765, 36.904,  38.3209, 42.1813, 42.2354,
        42.2677,  42.2887, 42.3032, 42.3272, 42.3326, 42.37,   42.4353, 42.471,  42.4714, 42.5452, 42.5982, 42.606,
        42.6168,  42.6229, 42.6369, 42.6373, 42.6489, 42.656,  42.6573, 42.6626, 42.6754, 42.7138, 42.7176, 42.7211,
        42.7229,  42.7249, 42.7363, 42.7578, 42.7653, 42.7775, 42.8039, 42.81,   42.8156, 42.8181, 42.8234, 42.862,
        42.8778,  42.8843, 42.8864, 42.8944, 42.9103, 42.9168, 42.9306, 42.9553, 42.9779, 42.9798, 43.0204, 43.04,
        43.0897,  43.1258, 43.1606, 43.1699, 43.2171, 43.2348, 43.3374, 43.3381, 43.3512, 43.3734, 43.3799, 43.4101,
        43.4391,  43.4413, 43.5057, 43.546,  43.5487, 43.5929, 43.6123, 43.6376, 43.6476, 43.656,  43.6623, 43.6745,
        43.7019,  43.7048, 43.7243, 43.7474, 43.8006, 43.802,  43.816,  43.8553, 44.0187, 44.1282, 50.5633, 52.971,
        55.3788,  57.7866, 60.1944, 62.6021, 65.0099, 67.4177, 69.8255, 71.5453, 71.627,  71.6518, 71.7164, 71.8052,
        72.0752,  78.5647, 79.0447, 79.0572, 79.0602, 87.0648, 87.0776, 87.0831, 87.1379, 87.6084, 87.9361, 87.9453,
        90.3554,  90.3735, 90.4196, 90.4755, 90.736,  90.8395, 90.8888, 90.9167, 90.9896, 91.0284, 91.0547, 91.0792,
        91.0856,  91.1564, 91.1874, 91.2767, 91.3509, 91.3606, 91.3694, 91.3846, 91.4163, 91.4716, 91.4999, 91.5104,
        91.5955,  91.6377, 91.7239, 91.7351, 91.8266, 91.8809, 91.9032, 92.1026, 92.2009, 92.6741, 92.725,  92.7861,
        92.7923,  92.7939, 92.8139, 92.8361, 92.8395, 92.8399, 92.8693, 92.8732, 92.8892, 92.9098, 92.9599, 92.9645,
        92.9943,  93.0287, 93.0655, 93.163,  93.2169, 93.2483, 93.2788, 93.3242, 93.3357, 93.3962, 93.4512, 93.4865,
        93.4967,  93.5075, 93.5618, 93.5687, 93.5998, 93.6258, 93.6656, 93.7532, 93.7863, 93.8326, 93.8569, 93.9061,
        94.045,   94.0737, 94.0897, 94.139,  94.1548, 94.1703, 94.1878, 94.2535, 94.2686, 94.2745, 94.2857, 94.2886,
        94.2888,  94.2892, 94.2894, 94.2919, 94.2926, 94.3315, 94.3428, 94.3479, 94.3531, 94.3548, 94.3572, 94.39,
        94.3915,  94.393,  94.4169, 94.4243, 94.4443, 94.4462, 94.4732, 94.4751, 94.5021, 94.5749, 94.801,  95.0073,
        95.1161,  95.1707, 95.1795, 95.2221, 95.2224, 95.2351, 95.2384, 95.2423, 95.3655, 95.6047, 97.9832, 98.8664,
        98.8851,
    };

    std::vector<double> z_beta{
        0.0488688, 0.0659669,  0.0167377,  0.0118994,  0.0546497, 0.0492168,  0.054008,   0.0615562,   0.0329818,
        0.0198734, 0.0787649,  0.0725736,  0.0191776,  0.0221466, 0.017286,   0.0140196,  0.0159092,   0.0187758,
        0.0445802, 0.0426748,  0.0416472,  0.067538,   0.0360167, 0.0533659,  0.0586778,  0.0780637,   0.0522997,
        0.0460862, 0.035761,   0.0507019,  0.0540376,  0.0577942, 0.081085,   0.0788567,  0.0917456,   0.0724265,
        0.0588537, 0.101357,   0.121442,   0.111461,   0.0932207, 0.0416152,  0.176162,   0.300889,    0.00167759,
        0.0170401, 0.0279557,  0.0522915,  0.0409845,  0.0364103, 0.082172,   0.0664834,  0.0279778,   0.0112904,
        0.0172499, 0.036908,   0.0446685,  0.066382,   0.0429679, 0.0737608,  0.0492626,  0.0645023,   0.0335008,
        0.0493085, 0.0531775,  0.0349665,  0.0266687,  0.0577466, 0.0594133,  0.103511,   0.0624077,   0.0584956,
        0.0237208, 0.0290887,  0.0645414,  0.121156,   0.213659,  0.149337,   0.0707457,  0.0650802,   0.0846108,
        0.081392,  0.0489487,  0.0229148,  0.592366,   0.0015744, 0.0880156,  0.062966,   0.000236704, 0.00750898,
        0.089794,  0.101568,   0.0842581,  0.00188584, 0.0358179, 0.101313,   0.0131046,  0.0936204,   0.0635296,
        0.0171708, 0.0180185,  0.0965745,  0.0982374,  0.0293198, 0.14488,    0.501977,   0.00466899,  0.136269,
        0.149401,  0.119321,   0.00935867, 0.0167867,  0.02622,   0.0341556,  0.0280193,  0.0370967,   0.0447662,
        0.0466437, 0.0171994,  0.0143625,  0.0253239,  0.0565393, 0.0666029,  0.0672235,  0.0967203,   0.107215,
        0.0997079, 0.0937085,  0.0938439,  0.184264,   0.188392,  0.0206284,  0.0274641,  0.0541313,   0.0322609,
        0.0210048, 0.0355424,  0.0384863,  0.0294224,  0.0428044, 0.108085,   0.101017,   0.0361572,   0.0741747,
        0.126748,  0.0608107,  0.0186219,  0.0169436,  0.0309684, 0.0205276,  0.0259526,  0.0191026,   0.0204,
        0.0252669, 0.0265757,  0.0577259,  0.0602321,  0.0585439, 0.047463,   0.011173,   0.0225032,   0.0402033,
        0.0441318, 0.0546044,  0.0789845,  0.073698,   0.057805,  0.0528702,  0.045917,   0.0580411,   0.0678256,
        0.0687238, 0.0682411,  0.0709446,  0.048331,   0.0390018, 0.0462757,  0.0689306,  0.0835533,   0.0694848,
        0.103566,  0.0846951,  0.111804,   0.105444,   0.120657,  0.0801751,  0.0912458,  0.0741798,   0.120336,
        0.103291,  0.0137911,  0.0353242,  0.0287186,  0.0589317, 0.0656092,  0.0613618,  0.0956169,   0.106931,
        0.042962,  0.0469019,  0.0636186,  0.0446684,  0.0353174, 0.0436399,  0.0247739,  0.0268681,   0.0459054,
        0.0425038, 0.0619818,  0.0728662,  0.0987082,  0.0777353, 0.0685978,  0.054729,   0.216696,    0.109477,
        0.0816447, 0.0248124,  0.0645956,  0.0887674,  0.358821,  0.012452,   0.00302954, 0.0127461,   0.0055299,
        0.0603242, 0.858552,   0.00916301, 0.0181083,  0.0460857, 0.0559177,  0.316362,   0.10348,     0.0493451,
        0.0279183, 0.100816,   0.0388297,  0.0650658,  0.0507265, 0.0308821,  0.0772113,  0.101859,    0.120319,
        0.0742149, 0.00968956, 0.0184527,  0.0240359,  0.0469413, 0.0869817,  0.0836102,  0.0387867,   0.0955997,
        0.127302,  0.128414,   0.0111207,  0.102655,   0.054309,  0.0223201,  0.221725,   0.0982961,   0.0508308,
        0.0611325, 0.00616506, 0.00781918, 0.0216638,  0.0422035, 0.0255722,  0.0259725,  0.0331707,   0.0337323,
        0.0492599, 0.0405353,  0.0906125,  0.0753589,  0.084473,  0.0687772,  0.101,      0.16873,     0.151378,
        0.0852681, 0.0619302,  0.0759306,  0.0568719,  0.0719768, 0.115449,   0.0903056,  0.0455297,   0.0209548,
        0.0651236, 0.0612675,  0.0380326,  0.0570954,  0.0658013, 0.127394,   0.120628,   0.0793552,   0.0705821,
        0.0734604, 0.188137,   0.0287074,  0.0160316,  0.0653097, 0.0651125,  0.0312663,  0.0329554,   0.0831828,
        0.0808349, 0.0210363,  0.0171,     0.0199224,  0.0142546, 0.00344729, 0.00366758, 0.00333573,  0.00400324,
        0.0427153, 0.0535797,  0.0585224,  0.0611902,  0.0622098, 0.0257756,  0.0472239,  0.0436173,   0.0398715,
        0.0638401, 0.0670242,  0.0543107,  0.0546343,  0.0802534, 0.0581638,  0.0778785,  0.130575,    0.354803,
        0.206355,  0.108805,   0.0545758,  0.00875419, 0.0514093, 0.042898,   0.0129607,  0.0160707,   0.0199696,
        0.130361,  0.366295,   0.0187009,
    };

    std::vector<int> n_triggered_alpha{
        0, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 2, 1, 0, 2, 1, 2, 1, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
        1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
        1, 1, 1, 2, 1, 1, 2, 1, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 0, 0, 1, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1,
        1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        2, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 1, 0, 1, 1, 2, 0, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1,
        1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 1, 0,
    };

    std::vector<double> t_alpha{
        0.481305, 1.44156, 1.77284, 1.82171, 3.51204, 3.57801, 3.59475, 3.60665, 3.6494,  3.65586, 3.70341, 3.71742,
        3.73639,  3.73729, 3.79619, 3.80896, 3.81536, 3.81833, 3.82625, 3.82938, 3.83127, 3.83711, 3.87083, 3.87206,
        3.87292,  3.89881, 3.90684, 3.92419, 3.93074, 3.95098, 3.95111, 3.95293, 3.95995, 3.98144, 4.00502, 4.0089,
        4.03402,  4.03881, 4.07318, 4.08133, 4.09287, 4.14017, 4.19463, 4.20433, 4.23339, 4.23624, 4.38049, 4.53713,
        4.53881,  4.55417, 4.56676, 4.60646, 4.60775, 4.64287, 4.68863, 4.70935, 4.71661, 4.72065, 4.7266,  4.75352,
        4.76531,  4.79299, 4.79649, 4.83907, 4.84225, 4.86099, 4.87258, 4.89156, 4.89543, 4.89596, 4.89924, 4.9493,
        4.95484,  4.99947, 5.01171, 5.01334, 5.02319, 5.02856, 5.07625, 5.13449, 5.24221, 5.28383, 5.31296, 5.34891,
        5.39757,  5.4303,  5.44652, 5.45322, 5.90533, 5.9069,  5.99492, 6.05788, 6.05812, 6.06539, 6.14791, 6.16696,
        6.23217,  8.69357, 8.69546, 8.73127, 9.9571,  31.1508, 31.2521, 31.2652, 31.3588, 31.4224, 31.4395, 31.4404,
        31.5361,  31.5386, 31.5654, 31.6835, 32.0381, 32.0428, 32.179,  32.3284, 32.4477, 32.4571, 32.4739, 32.4833,
        32.508,   32.5113, 32.5451, 32.5528, 32.558,  32.5623, 32.5672, 32.5781, 32.6145, 32.6289, 32.6344, 32.6749,
        32.7217,  32.7341, 32.7686, 32.8156, 32.9184, 33.004,  33.0246, 36.8765, 36.904,  38.3209, 42.1813, 42.2354,
        42.2677,  42.2887, 42.3032, 42.3272, 42.3326, 42.37,   42.4353, 42.471,  42.4714, 42.5452, 42.5982, 42.606,
        42.6168,  42.6229, 42.6369, 42.6373, 42.6489, 42.656,  42.6573, 42.6626, 42.6754, 42.7138, 42.7176, 42.7211,
        42.7229,  42.7249, 42.7363, 42.7578, 42.7653, 42.7775, 42.8039, 42.81,   42.8156, 42.8181, 42.8234, 42.862,
        42.8778,  42.8843, 42.8864, 42.8944, 42.9103, 42.9168, 42.9306, 42.9553, 42.9779, 42.9798, 43.0204, 43.04,
        43.0897,  43.1258, 43.1606, 43.1699, 43.2171, 43.2348, 43.3374, 43.3381, 43.3512, 43.3734, 43.3799, 43.4101,
        43.4391,  43.4413, 43.5057, 43.546,  43.5487, 43.5929, 43.6123, 43.6376, 43.6476, 43.656,  43.6623, 43.6745,
        43.7019,  43.7048, 43.7243, 43.7474, 43.8006, 43.802,  43.816,  43.8553, 44.0187, 44.1282, 50.5633, 52.971,
        55.3788,  57.7866, 60.1944, 62.6021, 65.0099, 67.4177, 69.8255, 71.5453, 71.627,  71.6518, 71.7164, 71.8052,
        72.0752,  78.5647, 79.0447, 79.0572, 79.0602, 87.0648, 87.0776, 87.0831, 87.1379, 87.6084, 87.9361, 87.9453,
        90.3554,  90.3735, 90.4196, 90.4755, 90.736,  90.8395, 90.8888, 90.9167, 90.9896, 91.0284, 91.0547, 91.0792,
        91.0856,  91.1564, 91.1874, 91.2767, 91.3509, 91.3606, 91.3694, 91.3846, 91.4163, 91.4716, 91.4999, 91.5104,
        91.5955,  91.6377, 91.7239, 91.7351, 91.8266, 91.8809, 91.9032, 92.1026, 92.2009, 92.6741, 92.725,  92.7861,
        92.7923,  92.7939, 92.8139, 92.8361, 92.8395, 92.8399, 92.8693, 92.8732, 92.8892, 92.9098, 92.9599, 92.9645,
        92.9943,  93.0287, 93.0655, 93.163,  93.2169, 93.2483, 93.2788, 93.3242, 93.3357, 93.3962, 93.4512, 93.4865,
        93.4967,  93.5075, 93.5618, 93.5687, 93.5998, 93.6258, 93.6656, 93.7532, 93.7863, 93.8326, 93.8569, 93.9061,
        94.045,   94.0737, 94.0897, 94.139,  94.1548, 94.1703, 94.1878, 94.2535, 94.2686, 94.2745, 94.2857, 94.2886,
        94.2888,  94.2892, 94.2894, 94.2919, 94.2926, 94.3315, 94.3428, 94.3479, 94.3531, 94.3548, 94.3572, 94.39,
        94.3915,  94.393,  94.4169, 94.4243, 94.4443, 94.4462, 94.4732, 94.4751, 94.5021, 94.5749, 94.801,  95.0073,
        95.1161,  95.1707, 95.1795, 95.2221, 95.2224, 95.2351, 95.2384, 95.2423, 95.3655, 95.6047, 97.9832, 98.8664,
        98.8851,
    };

    double alpha_curr = .9;
    double beta_curr = 10;
    double mu_curr = .5;
};
namespace celery {

std::vector<int> sample_y(double alpha_curr, double beta_curr, double mu_curr, const std::vector<double>& z_curr,
                          const std::vector<double>& t) {
    std::vector<double> t_tmp = t;
    t_tmp.insert(t_tmp.end(), z_curr.begin(), z_curr.end());
    std::sort(t_tmp.begin(), t_tmp.end());

    int n = t_tmp.size();

    std::vector<int> y_curr(n);
    y_curr[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#else
#endif
    for (int i = 1; i < n; i++) {
        std::random_device rd;
#if USE_RANDOM_DEVICE > 0
        std::mt19937 gen(rd());
#else
        std::mt19937 gen(0);
#endif

        std::vector<double> probs;
        probs.reserve(n);
        probs.push_back(mu_curr);

        for (int j = 0; j < i; j++) {
            double temp = alpha_curr * beta_tk(t_tmp[i] - t_tmp[j], beta_curr);
            probs.push_back(temp);
        }
        std::discrete_distribution<> d(probs.begin(), probs.end());

        int parent = d(gen);
        y_curr[i] = parent;
    }

    return y_curr;
}

std::vector<int> sample_y_fast(double alpha_curr, double beta_curr, double mu_curr, const std::vector<double>& z_curr,
                               const std::vector<double>& t) {
    std::vector<double> t_tmp;
    t_tmp.reserve(z_curr.size() + t.size());
    std::merge(t.begin(), t.end(), z_curr.begin(), z_curr.end(), std::back_inserter(t_tmp));
    int n = t_tmp.size();

    std::vector<int> y_curr(n);
    y_curr[0] = 0;

    std::vector<size_t> min_is(n);
    min_is[0] = 0;
    double alpha_beta_product = alpha_curr * beta_curr;

    for (int i = 1; i < n; ++i) {
        double minimum_time = t[i] - 25 / beta_curr - log(alpha_beta_product) / beta_curr;

        if (minimum_time < 0) {
            min_is[i] = 0;
        } else {
            int min_i;
            for (min_i = min_is[i - 1]; min_i < i; ++min_i) {
                if (t_tmp[min_i] > minimum_time) {
                    break;
                }
            }
            if (min_i != i) {
                min_is[i] = min_i;
            } else {
                min_is[i] = 0;
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for
#else
#endif
    for (int i = 1; i < n; i++) {
        std::random_device rd;
#if USE_RANDOM_DEVICE > 0
        std::mt19937 gen(rd());
#else
        std::mt19937 gen(0);
#endif
        ///@todo srowe; Would it be smarter to allocate this once outside and just clear as needed?
        std::vector<double> probs(i + 1, 0);

        probs[0] = mu_curr;

        double t_i = t_tmp[i];
        for (int j = min_is[i]; j < i; ++j) {
            double temp = alpha_beta_product * std::exp(-beta_curr * (t_i - t_tmp[j]));
            probs[j + 1] = temp;
        }
        std::discrete_distribution<> d(probs.begin(), probs.end());

        int parent = d(gen);
        y_curr[i] = parent;
    }

    return y_curr;
}

double beta_posterior(const std::vector<double>& t, double t_max, double alpha, double beta,
                      const std::vector<double>& z) {
    if (beta < 0 || beta > 15) {
        return (-INFINITY);
    }

    double loglik = 0;
    int n = t.size();
#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, alpha, beta, n) reduction(- : loglik)
#else
#endif
    for (int i = 0; i < n; i++) {
        loglik -= alpha * Beta_tk(t_max - t[i], beta);
    }

#ifdef _OPENMP
#pragma omp parallel for shared(z, beta) reduction(+ : loglik)
#else
#endif
    for (auto i = 0; i < z.size(); i++) {
        loglik += log(beta_tk(z[i], beta));
        // loglik += log( beta* exp(-beta*z[i])) = log(beta) -beta*z_i
    }
    return (loglik);
}

double beta_posterior_fast(const std::vector<double>& t, double t_max, double alpha, double beta,
                           const std::vector<double>& z) {
    if (beta < 0 || beta > 15) {
        return (-INFINITY);
    }
    int n = t.size();

    double epsilon = t_max + 1 / beta * (-36 - log(alpha));

    int min_i;
    for (min_i = n - 1; min_i >= 0; --min_i) {
        if (t[min_i] < epsilon) {
            break;
        }
    }

    double loglik = 0;

#ifdef _OPENMP
#pragma omp parallel for simd shared(t, t_max, beta, n) reduction(+ : loglik)
#else
#endif
    for (int i = n - 1; i >= min_i; --i) {
        loglik += std::exp(beta * (t[i] - t_max));
    }

    loglik -= n;
    loglik *= alpha;

    double loglik2 = 0;
    // OpenMP seems to be overkill for this loop
    for (const auto zval : z) {
        loglik2 -= zval;
    }
    loglik2 *= beta;

    loglik += z.size() * log(beta) + loglik2;

    return loglik;
}

double alpha_posterior(std::vector<double>& t, double t_max, double alpha, double beta,
                       const std::vector<int>& ntriggered) {
    if (alpha < 0 || alpha > 15) {
        return (-INFINITY);
    }

    double loglik = 0;
    int n = t.size();

#ifdef _OPENMP
#pragma omp parallel for shared(t, t_max, alpha, beta, n, ntriggered) reduction(+ : loglik)
#else
#endif
    for (int i = 0; i < n; i++) {
        loglik += ntriggered[i] * log(alpha) - alpha * Beta_tk(t_max - t[i], beta);
    }

    return loglik;
}

double alpha_posterior_fast(std::vector<double>& t, double t_max, double alpha, double beta,
                            const std::vector<int>& ntriggered) {
    if (alpha < 0 || alpha > 15) {
        return (-INFINITY);
    }

    double loglik = 0;
    int n = t.size();
    double log_alpha = log(alpha);

    // Loop over t_i backwards and find minimum value until less than epsilon
    // -36 is approximately log(1e-16) where log is log base e
    double epsilon = t_max + 1 / beta * (-36 - log(alpha));

    int min_i;
    for (min_i = n - 1; min_i >= 0; --min_i) {
        if (t[min_i] < epsilon) {
            break;
        }
    }

    for (int i = n - 1; i >= min_i; --i) {
        loglik -= alpha * Beta_tk(t_max - t[i], beta);
    }

    loglik -= alpha * min_i;

    for (int i = 0; i < n; ++i) {
        loglik += log_alpha * ntriggered[i];
    }

    return loglik;
}

}  // end namespace celery

#define PERFORMANCE_TEST_RUN_LOL 0
#define RUN_LOG_LIK_TESTS 0
#define RUN_SAMPLE_Y 0
#define RUN_BETA_POSTERIOR 0
#define RUN_ALPHA_POSTERIOR 1

#if PERFORMANCE_TEST_RUN_LOL

#define NUM_SAMPLES_YO 30
#define NUM_ITERATIONS_BRO 100
#define PRINT_STUFF 0

#else
#define NUM_SAMPLES_YO 1
#define NUM_ITERATIONS_BRO 1
#define PRINT_STUFF 1

#endif

#if RUN_LOG_LIK_TESTS

BASELINE_F(StppTests, LogLikBaseline, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double x = 0;
    celero::DoNotOptimizeAway(x = celery::log_lik(t, mu, alpha, beta, t_max));

#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "THe x is " << x << std::endl;
#endif
}

BENCHMARK_F(StppTests, IgnoreSmallExp, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double x = 0;

    celero::DoNotOptimizeAway(x = celery::log_lik_break(t, mu, alpha, beta, t_max));
#if PRINT_STUFF
    std::cout.precision(16);

    std::cout << "THe x is " << x << std::endl;
#endif
}

#endif

#if RUN_SAMPLE_Y

BASELINE_F(SampleYTest, SampleYBaseline, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    std::vector<int> y;
    celero::DoNotOptimizeAway(y = celery::sample_y(alpha_curr, beta_curr, mu_curr, z_curr, t_y_test));

#if PRINT_STUFF
    std::cout.precision(16);
    for (size_t i = 0; i < 70; ++i) {
        std::cout << "Baseline y[" << i << "] = " << y[i] << std::endl;
    }
#endif
}

BENCHMARK_F(SampleYTest, SampleYFaster, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    std::vector<int> y;

    celero::DoNotOptimizeAway(y = celery::sample_y_fast(alpha_curr, beta_curr, mu_curr, z_curr, t_y_test));
#if PRINT_STUFF
    std::cout.precision(16);

    for (size_t i = 0; i < 70; ++i) {
        std::cout << "Fast y[" << i << "] = " << y[i] << std::endl;
    }

#endif  // PRINT_STUFF
}

#endif  // SAMPLE_Y

#if RUN_BETA_POSTERIOR

BASELINE_F(BetaPosteriorTests, BetaBaseline, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double x;
    celero::DoNotOptimizeAway(x = celery::beta_posterior(t_beta, t_max, alpha, beta, z_beta));

#if PRINT_STUFF
    std::cout.precision(16);

    std::cout << "x = " << x << std::endl;
#endif  // PRINT_STUFF
}

BENCHMARK_F(BetaPosteriorTests, BetaFaster, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double x;

    celero::DoNotOptimizeAway(x = celery::beta_posterior_fast(t_beta, t_max, alpha, beta, z_beta));
#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "x = " << x << std::endl;

#endif  // PRINT_STUFF
}

#endif  // RUN_BETA_POSTERIOR

#if RUN_ALPHA_POSTERIOR

BASELINE_F(AlphaPosteriorTests, AlphaBaseline, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double x;
    celero::DoNotOptimizeAway(x = celery::alpha_posterior(t_alpha, t_max, alpha_curr, beta_curr, n_triggered_alpha));

#if PRINT_STUFF
    std::cout.precision(16);

    std::cout << "x = " << x << std::endl;
#endif  // PRINT_STUFF
}

BENCHMARK_F(AlphaPosteriorTests, AlphaFaster, StppPerfTest, NUM_SAMPLES_YO, NUM_ITERATIONS_BRO) {
    double x;

    celero::DoNotOptimizeAway(
        x = celery::alpha_posterior_fast(t_alpha, t_max, alpha_curr, beta_curr, n_triggered_alpha));
#if PRINT_STUFF
    std::cout.precision(16);
    std::cout << "x = " << x << std::endl;

#endif  // PRINT_STUFF
}

#endif  // RUN_ALPHA_POSTERIOR

#endif  // COMPILE_CELERO_TESTS
