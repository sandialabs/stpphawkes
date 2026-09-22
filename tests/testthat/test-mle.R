# Tests for maximum-likelihood estimation in R/mle.R and the compiled
# likelihood functions they call.

test_that("temporal_likelihood returns a finite scalar", {
  skip_if_no_dll()
  t <- sort(runif(20, 0, 10))
  lik <- temporal_likelihood(t, 0.5, 0.1, 0.5, 10)
  expect_length(lik, 1)
  expect_true(is.finite(lik))
})

test_that("temporal.mle recovers plausible parameters from simulated data", {
  skip_if_no_dll()
  set.seed(2024)
  t <- simulate_temporal(0.5, 0.1, 0.5, c(0, 200), numeric(), seed = 2024)
  skip_if(length(t) < 20, "too few events to fit reliably")
  fit <- temporal.mle(t, print = FALSE)
  expect_named(fit, c("mu", "alpha", "beta", "loglik"))
  expect_gt(fit$mu, 0)
  expect_gte(fit$alpha, 0)
  expect_gte(fit$beta, 0)
  expect_true(is.finite(fit$loglik))
})

test_that("temporal.mle honours a supplied t_max", {
  skip_if_no_dll()
  set.seed(11)
  t <- simulate_temporal(0.5, 0.1, 0.5, c(0, 50), numeric(), seed = 11)
  skip_if(length(t) < 10, "too few events")
  fit <- temporal.mle(t, t_max = 60, print = FALSE)
  expect_true(is.finite(fit$loglik))
})

test_that("temporal.catmark.mle returns per-mark probabilities summing to 1", {
  skip_if_no_dll()
  set.seed(5)
  t <- simulate_temporal(0.5, 0.1, 0.5, c(0, 100), numeric(), seed = 5)
  skip_if(length(t) < 10, "too few events")
  marks <- sample(c("a", "b", "c"), length(t), replace = TRUE)
  fit <- temporal.catmark.mle(t, marks, print = FALSE)
  expect_true("p" %in% names(fit))
  expect_equal(sum(fit$p), 1, tolerance = 1e-8)
})

test_that("stpp.mle returns the full parameter set", {
  skip_if_no_dll()
  set.seed(3)
  params <- list(mu = 0.5, a = 0.1, b = 0.5, sig = 0.01)
  data <- simulate_hawkes_stpp(params, square(10), c(0, 20),
                               d = 0.5, history = numeric(), seed = 3)
  skip_if(nrow(data) < 10, "too few events to fit")
  fit <- stpp.mle(data, square(10), print = FALSE)
  expect_named(fit, c("mu", "a", "b", "sig", "loglik"))
  expect_gt(fit$mu, 0)
  expect_gt(fit$sig, 0)
})
