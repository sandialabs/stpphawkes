#ifndef STPP_RNG_SEED_H
#define STPP_RNG_SEED_H

#include <R_ext/Random.h>

#include <atomic>
#include <cstdint>
#include <random>

/**
 * @brief Base seed for the C++ generators, drawn from R's RNG so that set.seed() is honoured.
 */
inline std::uint32_t& RngBaseSeed() {
    static std::uint32_t base_seed = 0;
    return base_seed;
}

/**
 * @brief Counter distinguishing successive generators created in sequential code.
 */
inline std::atomic<std::uint64_t>& RngStreamCounter() {
    static std::atomic<std::uint64_t> counter{0};
    return counter;
}

/**
 * @brief Seeds the C++ generators from R's RNG.
 *
 * Must be called once at the start of every exported function that samples, so results respond
 * to set.seed(). Relies on the RNGScope that Rcpp's generated wrappers already establish.
 */
inline void SeedRngFromR() {
    RngBaseSeed() = static_cast<std::uint32_t>(::unif_rand() * 4294967296.0);
    RngStreamCounter() = 0;
}

/**
 * @brief Returns a generator for a distinct stream derived from the R-derived base seed.
 *
 * Sequential callers get a fresh stream each call. Inside a parallel region pass an explicit
 * stream id (e.g. the loop index) so the draws do not depend on thread scheduling.
 */
inline std::mt19937 GenerateMersenneTwister(std::uint64_t stream) {
    std::seed_seq seq{static_cast<std::uint32_t>(RngBaseSeed()), static_cast<std::uint32_t>(stream & 0xFFFFFFFFu),
                      static_cast<std::uint32_t>(stream >> 32)};
    return std::mt19937(seq);
}

inline std::mt19937 GenerateMersenneTwister() {
    return GenerateMersenneTwister(RngStreamCounter()++);
}

#endif  // STPP_RNG_SEED_H
