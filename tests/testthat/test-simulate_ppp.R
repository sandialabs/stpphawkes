# Tests for Poisson-process simulators in R/homog_ppp.R and R/inhomo_ppp.R.
# These are stochastic, so assertions check structure, invariants, and
# reproducibility under a fixed seed rather than exact values.

test_that("homog.STPP returns a well-formed data frame", {
  skip_if_no_dll()
  set.seed(42)
  out <- homog.STPP(0.5, unit_square(), c(0, 10))
  expect_s3_class(out, "data.frame")
  expect_named(out, c("x", "y", "t", "type"))
  expect_type(out$type, "character")
  # times are returned sorted
  expect_false(is.unsorted(out$t))
})

test_that("homog.STPP is reproducible under a fixed seed", {
  skip_if_no_dll()
  set.seed(1)
  a <- homog.STPP(1, unit_square(), c(0, 5))
  set.seed(1)
  b <- homog.STPP(1, unit_square(), c(0, 5))
  expect_equal(a, b)
})

test_that("homog.STPP remove=TRUE keeps points inside the polygon", {
  skip_if_no_dll()
  set.seed(7)
  out <- homog.STPP(2, unit_square(), c(0, 10), remove = TRUE)
  if (nrow(out) > 0) {
    inside <- inout(out$x, out$y, unit_square(), TRUE)
    expect_true(all(as.logical(inside)))
  } else {
    succeed("no points generated; nothing to check")
  }
})

test_that("homog.STPP with checkpoly=FALSE skips polygon validation", {
  skip_if_no_dll()
  # checkpoly guards against malformed polygons; with checkpoly=FALSE that
  # guard is bypassed, so even a thin sliver runs without a "malformed" error.
  sliver <- matrix(c(0,    0,
                     0,    0.001,
                     10,   0.001,
                     10,   0),
                   ncol = 2, byrow = TRUE)
  set.seed(3)
  expect_no_error(homog.STPP(1, sliver, c(0, 1), checkpoly = FALSE))
})

test_that("homog.PPP (R helper) returns sorted times within the region", {
  out <- stpphawkes:::homog.PPP(5, c(0, 10), seed = 123)
  expect_true(all(out >= 0 & out <= 10))
  expect_false(is.unsorted(out))
})

test_that("inhomog.STPP works with a functional intensity", {
  skip_if_no_dll()
  set.seed(99)
  # Use a high constant intensity over a sizeable domain so the expected
  # point count is large; otherwise rpois can return 0 and the simulator
  # legitimately errors with "there is no data to thin".
  mu_fun <- function(x, y, t) rep(5, length(x))  # constant intensity
  out <- stpphawkes:::inhomog.STPP(mu_fun, square(10), c(0, 5),
                                   nx = 21, ny = 21, nt = 21)
  expect_s3_class(out, "data.frame")
  expect_named(out, c("x", "y", "t", "type"))
})
