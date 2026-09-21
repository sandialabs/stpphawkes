#ifndef STPP_GSL_RNG_HOLDER_H
#define STPP_GSL_RNG_HOLDER_H

#include <gsl/gsl_rng.h>

#include <memory>

#include "rng_seed.h"

/**
 * @brief Owns a GSL generator that is seeded from R's RNG and freed on every exit path.
 *
 * The seed comes from RngBaseSeed(), which SeedRngFromR() fills from R's RNG, so draws taken
 * from this generator respond to set.seed() like the rest of the package. Holding the generator
 * in a unique_ptr also means an early return (Progress::check_abort(), an exception, an
 * Rcpp::stop) releases it instead of leaking it.
 */
class GslRngHolder {
   public:
    GslRngHolder() : rng_(gsl_rng_alloc(gsl_rng_mt19937), &gsl_rng_free) {
        gsl_rng_set(rng_.get(), static_cast<unsigned long>(RngBaseSeed()));
    }

    gsl_rng* get() const { return rng_.get(); }

   private:
    std::unique_ptr<gsl_rng, void (*)(gsl_rng*)> rng_;
};

#endif  // STPP_GSL_RNG_HOLDER_H
