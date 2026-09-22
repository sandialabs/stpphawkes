# Tests for pure-R utility functions in R/utility_functions.R
# These do not touch compiled code and can run without the shared library.

test_that("is.scalar identifies length-1 numerics only", {
  expect_true(stpphawkes:::is.scalar(1))
  expect_true(stpphawkes:::is.scalar(3.14))
  expect_false(stpphawkes:::is.scalar(c(1, 2)))
  expect_false(stpphawkes:::is.scalar("a"))
  expect_false(stpphawkes:::is.scalar(numeric(0)))
})

test_that("is.length2 identifies length-2 numerics only", {
  expect_true(stpphawkes:::is.length2(c(1, 2)))
  expect_false(stpphawkes:::is.length2(1))
  expect_false(stpphawkes:::is.length2(c(1, 2, 3)))
  expect_false(stpphawkes:::is.length2(c("a", "b")))
})

test_that("ndims returns the number of dimensions", {
  expect_equal(stpphawkes:::ndims(1:5), 0)          # vectors have no dim
  expect_equal(stpphawkes:::ndims(matrix(0, 2, 3)), 2)
  expect_equal(stpphawkes:::ndims(array(0, c(2, 3, 4))), 3)
})

test_that("repmat tiles a vector like MATLAB", {
  out <- stpphawkes:::repmat(c(1, 2), 2, 3)
  expect_equal(dim(out), c(2, 6))
  expect_equal(as.numeric(out[1, ]), c(1, 2, 1, 2, 1, 2))
})

test_that("repmat tiles a matrix", {
  m <- matrix(1:4, 2, 2)
  out <- stpphawkes:::repmat(m, 2, 2)
  expect_equal(dim(out), c(4, 4))
})

test_that("trapz integrates a linear function exactly", {
  x <- seq(0, 1, length.out = 101)
  y <- 2 * x                      # integral over [0,1] is 1
  expect_equal(stpphawkes:::trapz(x, y), 1, tolerance = 1e-8)
})

test_that("trapz integrates a constant exactly", {
  x <- seq(0, 5, length.out = 50)
  y <- rep(3, 50)                 # integral is 3 * 5 = 15
  expect_equal(stpphawkes:::trapz(x, y), 15, tolerance = 1e-8)
})

test_that("trapz errors on dimension mismatch", {
  expect_error(stpphawkes:::trapz(1:3, matrix(0, 4, 2)), "Dimension Mismatch")
})

test_that("cumtrapz returns cumulative integral with leading zero", {
  x <- seq(0, 1, length.out = 101)
  y <- rep(2, 101)
  z <- stpphawkes:::cumtrapz(x, y)
  expect_equal(z[1], 0)
  expect_equal(as.numeric(z[length(z)]), 2, tolerance = 1e-8)
})

test_that("long2UTM maps longitudes into 1..60", {
  expect_equal(stpphawkes:::long2UTM(-180), 1)
  expect_equal(stpphawkes:::long2UTM(0), 31)
  expect_true(stpphawkes:::long2UTM(179) %in% 1:60)
})
