# Shared fixtures and helpers for the test suite.
# Files named helper*.R are sourced by testthat before running any tests.

# A simple unit-square polygon used across many tests.
#
# Vertices are wound in the same orientation as the package's own examples
# (e.g. `homog.STPP(0.5, matrix(c(0,0,1,1,0,1,1,0), ncol = 2), c(0,10))`):
# (0,0) -> (0,1) -> (1,1) -> (1,0). areapl() returns a *signed* area, so this
# winding yields a positive area and passes the checkpoly test; the opposite
# winding gives a negative area and is falsely rejected as "malformed".
unit_square <- function() {
  matrix(c(0, 0,
           0, 1,
           1, 1,
           1, 0),
         ncol = 2, byrow = TRUE)
}

# A larger, well-conditioned square polygon, wound the same way as unit_square().
square <- function(side = 10) {
  matrix(c(0,    0,
           0,    side,
           side, side,
           side, 0),
         ncol = 2, byrow = TRUE)
}

# Skip a test unless the compiled shared object is loadable. Pure-R helper
# functions can be tested even when the C++ has not been compiled, but any
# test that touches .Call() must be skipped in that case.
skip_if_no_dll <- function() {
  testthat::skip_if_not(
    is.loaded("_stpphawkes_areapl"),
    "compiled stpphawkes shared library not available"
  )
}
