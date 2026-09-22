# Tests for the compiled Hawkes simulators and the intensity function
# (R/RcppExports.R -> src/). All require the shared library.

test_that("simulate_temporal returns arrival times within the window", {
  skip_if_no_dll()
  arrivals <- simulate_temporal(0.5, 0.1, 0.5, c(0, 10), numeric(), seed = 1)
  expect_type(arrivals, "double")
  if (length(arrivals) > 0) {
    expect_true(all(arrivals >= 0 & arrivals <= 10))
    expect_false(is.unsorted(arrivals))
  }
})

test_that("simulate_temporal is reproducible under a fixed seed", {
  skip_if_no_dll()
  a <- simulate_temporal(0.5, 0.1, 0.5, c(0, 10), numeric(), seed = 123)
  b <- simulate_temporal(0.5, 0.1, 0.5, c(0, 10), numeric(), seed = 123)
  expect_equal(a, b)
})

test_that("intensity_temporal returns the background rate with no history", {
  skip_if_no_dll()
  # with no prior events the intensity at evalpt is just mu
  lambda <- intensity_temporal(0.5, 0.1, 0.5, numeric(), 5)
  expect_equal(lambda, 0.5, tolerance = 1e-10)
})

test_that("intensity_temporal is non-negative and finite with history", {
  skip_if_no_dll()
  hist <- c(1, 2, 3)
  lambda <- intensity_temporal(0.5, 0.2, 0.5, hist, 4)
  expect_true(is.finite(lambda))
  expect_gte(lambda, 0)
})

test_that("simulate_hawkes_stpp returns a data frame with x,y,t", {
  skip_if_no_dll()
  params <- list(mu = 0.5, a = 0.1, b = 0.5, sig = 0.01)
  out <- simulate_hawkes_stpp(params, square(10), c(0, 10),
                              d = 0.5, history = numeric(), seed = 1)
  expect_true(is.data.frame(out) || is.list(out))
  expect_true(all(c("x", "y", "t") %in% names(out)))
})
