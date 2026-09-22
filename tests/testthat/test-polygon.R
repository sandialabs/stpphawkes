# Tests for polygon geometry: pure-R helpers in R/polygon.R and the
# compiled routines they depend on (areapl, inout, bbox, sbox, ...).

test_that("sbox (R helper) enlarges the bounding box by the given fraction", {
  poly <- unit_square()
  out <- stpphawkes:::sbox(poly, xfrac = 0.1, yfrac = 0.1)
  expect_equal(dim(out), c(4, 2))
  # unit square expanded by 10% each side -> x,y in [-0.1, 1.1]
  expect_equal(min(out[, 1]), -0.1, tolerance = 1e-12)
  expect_equal(max(out[, 1]),  1.1, tolerance = 1e-12)
  expect_equal(min(out[, 2]), -0.1, tolerance = 1e-12)
  expect_equal(max(out[, 2]),  1.1, tolerance = 1e-12)
})

test_that("larger.region (R helper) returns min/max corners", {
  poly <- unit_square()
  lr <- stpphawkes:::larger.region(poly, xfrac = 0.1, yfrac = 0.1)
  expect_equal(dim(lr), c(2, 2))
  expect_true(lr[1, 1] < lr[2, 1])  # xmin < xmax
  expect_true(lr[1, 2] < lr[2, 2])  # ymin < ymax
})

test_that("make.grid builds a grid of the requested size", {
  g <- stpphawkes:::make.grid(5, 5, unit_square())
  expect_length(g$x, 5)
  expect_length(g$y, 5)
  expect_equal(dim(g$mask), c(5, 5))
  expect_type(g$mask, "logical")
})

test_that("make.grid rejects grids smaller than 2x2", {
  expect_error(stpphawkes:::make.grid(1, 5, unit_square()), "at least")
})

# ---- Compiled routines ----

test_that("areapl computes polygon area (magnitude)", {
  skip_if_no_dll()
  # areapl returns a *signed* area whose sign depends on vertex winding, so
  # compare magnitudes.
  expect_equal(abs(areapl(unit_square())), 1, tolerance = 1e-10)
  expect_equal(abs(areapl(square(10))), 100, tolerance = 1e-10)
})

test_that("areapl sign follows vertex winding", {
  skip_if_no_dll()
  # Same square, reversed winding -> area of equal magnitude, opposite sign.
  cw  <- unit_square()
  ccw <- cw[nrow(cw):1, ]
  expect_equal(areapl(cw), -areapl(ccw), tolerance = 1e-10)
})

test_that("inout classifies interior and exterior points", {
  skip_if_no_dll()
  poly <- unit_square()
  res <- inout(c(0.5, 5), c(0.5, 5), poly, TRUE)
  expect_true(as.logical(res[1]))   # interior
  expect_false(as.logical(res[2]))  # exterior
})

test_that("pip keeps only points inside the polygon", {
  skip_if_no_dll()
  poly <- unit_square()
  x <- c(0.5, 0.25, 5, -1)
  y <- c(0.5, 0.75, 5, -1)
  out <- pip(x, y, poly)
  expect_true(all(out$x >= 0 & out$x <= 1))
  expect_true(all(out$y >= 0 & out$y <= 1))
  expect_length(out$x, 2)
})

test_that("bbox returns the [min,max] corners of the polygon", {
  skip_if_no_dll()
  poly <- square(10)
  bb <- bbox(poly)
  # bbox() returns a 2x2 matrix: column 1 = c(xmin, xmax), column 2 = c(ymin, ymax).
  expect_equal(dim(bb), c(2L, 2L))
  expect_equal(bb[, 1], c(0, 10), tolerance = 1e-10)   # x range
  expect_equal(bb[, 2], c(0, 10), tolerance = 1e-10)   # y range
})

test_that("bboxx expands a 2-row corner matrix into 4 vertices", {
  skip_if_no_dll()
  # bboxx() takes a 2-row matrix (first row one corner, second row the other)
  # and returns the 4 vertices of the axis-aligned box, as used by checkpoly.
  corners <- matrix(c(0, 0,
                      10, 10), ncol = 2, byrow = TRUE)
  bx <- bboxx(corners)
  expect_equal(dim(bx), c(4L, 2L))
})

test_that("ptinpoly flags points inside the polygon", {
  skip_if_no_dll()
  poly <- unit_square()
  xp <- c(poly[, 1], poly[1, 1])
  yp <- c(poly[, 2], poly[1, 2])
  bb <- bbox(poly)
  res <- ptinpoly(c(0.5), c(0.5), xp, yp, bb)
  expect_true(as.logical(res[1]))
})
