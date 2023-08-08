#' Marked Hawkes Process with Missing Data
#'
#' A library for estimation of spatio-temporal Hawkes process parameters with
#' missing data support
#'
#' @name stpphawkes
#'
#' @references J. D. Tucker, L. Shand, and J. R. Lewis, “Handling Missing Data in Self-Exciting Point Process Models,”
#'   Spatial Statistics, vol. 29. pp. 160-176, 2019.
#'
#' @docType package
#' @useDynLib stpphawkes
#' @importFrom Rcpp evalCpp
#' @importFrom grDevices chull
#' @importFrom graphics abline par
#' @importFrom stats optim predict qnorm rexp rnorm rpois runif var
#' @aliases stpphawkes stpphawkes-package
NULL
