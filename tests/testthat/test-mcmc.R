# Tests for the MCMC wrappers in R/mcmc.R.
#
# The samplers themselves are expensive, so most tests here exercise the
# cheap input-validation branches (pure R). A small number of end-to-end
# runs use a tiny n_mcmc and are wrapped in skip_on_cran() so they do not
# slow down CRAN checks.

# ---- Input validation (fast, no sampling) ----

test_that("mcmc_temporal rejects n_burn >= n_mcmc", {
  bad <- list(n_mcmc = 100, n_burn = 100, sig_mu = .5, sig_alpha = .5,
              sig_beta = .5, mu_param = c(.1, .1), alpha_param = c(.1, .1),
              beta_param = c(.1, .1))
  expect_error(mcmc_temporal(c(1, 2, 3), mcmc_param = bad),
               "n_burn must be less than n_mcmc")
})

test_that("mcmc_temporal rejects a non-matrix t_mis", {
  ok <- list(n_mcmc = 100, n_burn = 50, sig_mu = .5, sig_alpha = .5,
             sig_beta = .5, mu_param = c(.1, .1), alpha_param = c(.1, .1),
             beta_param = c(.1, .1))
  init <- list(mu = 0.5, alpha = 0.1, beta = 0.5)
  expect_error(
    mcmc_temporal(c(1, 2, 3), t_mis = c(1, 2), param_init = init, mcmc_param = ok),
    "t_mis must be a matrix"
  )
})

test_that("mcmc_temporal rejects a t_mis with the wrong number of columns", {
  ok <- list(n_mcmc = 100, n_burn = 50, sig_mu = .5, sig_alpha = .5,
             sig_beta = .5, mu_param = c(.1, .1), alpha_param = c(.1, .1),
             beta_param = c(.1, .1))
  init <- list(mu = 0.5, alpha = 0.1, beta = 0.5)
  expect_error(
    mcmc_temporal(c(1, 2, 3), t_mis = matrix(1:3, ncol = 3),
                  param_init = init, mcmc_param = ok),
    "incorrect number of columns"
  )
})

test_that("mcmc_temporal_catmark requires marks to be a factor", {
  expect_error(
    mcmc_temporal_catmark(c(1, 2, 3), marks = c("a", "b", "c")),
    "marks must be a factor"
  )
})

test_that("mcmc_temporal_catmark validates mcmc_param element lengths", {
  marks <- factor(c("a", "b", "a"))
  bad <- list(n_mcmc = 100, n_burn = 50, sig_beta = .5,
              alpha_param = 1,            # should be length 2
              beta_param = c(.1, .1),
              p_param = c(.5, .5))
  expect_error(
    mcmc_temporal_catmark(c(1, 2, 3), marks = marks, mcmc_param = bad),
    "alpha_param must be numeric, length 2"
  )
})

test_that("mcmc_temporal_contmark requires numeric marks", {
  expect_error(
    mcmc_temporal_contmark(c(1, 2, 3), marks = factor(c("a", "b", "c")),
                           wshape = 1),
    "marks must be numeric"
  )
})

test_that("mcmc_temporal_contmark requires initial parameters", {
  expect_error(
    mcmc_temporal_contmark(c(1, 2, 3), marks = c(1, 2, 3), wshape = 1),
    "Initial parameters needed"
  )
})

test_that("mcmc_stpp rejects n_burn >= n_mcmc", {
  df <- data.frame(x = c(.1, .2), y = c(.1, .2), t = c(1, 2))
  bad <- list(n_mcmc = 100, n_burn = 200)
  expect_error(mcmc_stpp(df, unit_square(), mcmc_param = bad),
               "n_burn must be less than n_mcmc")
})

# ---- End-to-end smoke tests (slow; skipped on CRAN) ----

test_that("mcmc_temporal runs and returns posterior samples", {
  skip_on_cran()
  skip_if_no_dll()
  set.seed(321)
  times <- simulate_temporal(0.5, 0.1, 0.5, c(0, 100), numeric(), seed = 321)
  skip_if(length(times) < 20, "too few events")
  param <- list(n_mcmc = 50, n_burn = 10, sig_mu = .5, sig_alpha = .5,
                sig_beta = .5, mu_param = c(.1, .1), alpha_param = c(.1, .1),
                beta_param = c(.1, .1))
  init <- list(mu = 0.5, alpha = 0.1, beta = 0.5)
  out <- mcmc_temporal(times, param_init = init, mcmc_param = param,
                       print = FALSE)
  expect_true(is.list(out) || is.data.frame(out))
})
