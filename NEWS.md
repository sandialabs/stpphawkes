# stpphawkes (development version)

## Correctness fixes

* The compensator in the temporal Hawkes log-likelihood was short by one `alpha`
  on every call (the truncated tail contributed `min_i` terms where `min_i + 1`
  were skipped), which biased the estimates of `alpha`. The corrected version now
  matches an untruncated reference implementation exactly.
* `homog.STPP()`'s C++ twin drew the background from `rpois(mu * t_area)`, ignoring
  the area of the region the points are laid down on. It now uses
  `rpois(mu * s_area * t_area)`, matching the R implementation.
* `simulate_hawkes_stpp()` never clipped its output to the study polygon, so it
  returned the background points generated on the enlarged region used to avoid
  edge effects.
* `mcmc_temporal(..., t_mis =, branching = TRUE)` did not seed its generator from
  R's RNG, so it ignored `set.seed()`.
* The inverse-gamma prior term in `sig_posterior()` used `-sig/scale` instead of
  `-scale/sig`. The Metropolis update now targets the same posterior as the
  conjugate `sample_sig_gibbs()`.
* `mcmc_stpp_nonunif()` passed `t_max` positionally into `stpp.mle.nonunif()`'s
  `poly` argument, so a user-supplied `t_max` was silently replaced by `max(data$t)`.
* In the branching missing-data sampler the initial missing times were drawn
  against an empty history instead of the observed times.
* The offspring burn-in window used `-b * log(fraction)` instead of
  `-log(fraction) / b`, so it scaled the wrong way with `b`.
* A stray mark code in the categorical-mark sampler hit a bare `throw;`, which
  terminated the R session; it now raises an R condition.
* The GSL generator used for the Dirichlet draws was never seeded, so those draws
  ignored `set.seed()`; it is now seeded from R's RNG and freed on every exit path.
* `sample_alpha()`'s rejection loop could run forever; it now falls back to an exact
  inverse-CDF draw from the truncated gamma after a bounded number of attempts.
* `pip()` is exported (its `@export` tag was indented and had been folded into
  `@return`).
* `predict_hawkes_t()` squared `beta` by pre-multiplying `alpha`, which
  `simulate_temporal()` already does internally.
* `intensity_stpp()` treated `sig` as a standard deviation; the package convention
  is that it is the variance. Same fix in `condInt_mcmc_stpp.cpp`.
* `inhomog.PPP()` called the temporal `homog.PPP()` and then indexed the result as a
  matrix; it now calls `homog.SPPP()`.
* `simulate_hawkes_stpp_inhom()` checked `ndims(params$mu)` rather than `ndims(mu)`,
  and dropped the whole background catalog from its output.
* Reading `.Random.seed` before any RNG draw no longer errors in a fresh session.
* `DFtoMat()` skips non-numeric columns, so a `homog.STPP()` data frame can be passed
  as `history`; histories with a different number of columns than the background are
  lined up instead of failing in `join_cols`.
* Guarded several loops and index-0 writes against empty inputs.

## Packaging

* Fixed the malformed `Date` field in `DESCRIPTION`.
* `src/Makevars.win` uses `RcppGSL:::CFlags()`/`LdFlags()` instead of the obsolete
  `LIB_GSL` variable, which is unset on current Rtools.
* `.Rbuildignore` patterns corrected; `src/celero/` and `src/tests/` no longer ship
  in the tarball.
* `NAMESPACE` regenerated in roxygen's own format.

# stpphawkes 0.2.4
* memory bugfixes

# stpphawkes 0.2.2
* bugfixes
* add additional outputs to mcmc to include missing data and branching structure

# stpphawkes 0.2.1
* bugfixes

# stpphawkes 0.2.0
* initial release of code

