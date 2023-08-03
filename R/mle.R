#' MLE Estimation of Temporal Hawkes Model Parameters
#'
#' Maximum likelihood estimation of the parameters of a temporal exponential decay Hawkes model
#'
#'
#' @param t - vector of arrival times
#' @param t_max - maximum time value (default = max(times))
#' @param initval - vector of two elements describing missing time range (default = NA)
#' @param print - print progress (default = TRUE)
#' @return A list containing the parameter values and likelihood value
#' @export
temporal.mle <- function(t, t_max=max(t), initval=NA, print=TRUE){
  if (is.na(t_max))
    t_max <- max(t)

  fn <- function(params){
    mu <- params[1]
    alpha <- params[2]
    beta <- params[3]
    if (mu <= 0 || alpha < 0 || beta < 0)
      return(Inf)

    if (alpha > beta)
      return(Inf)

    lik <- temporal_likelihood(t, mu, alpha, beta, t_max)
    if (print)
      print(c(mu,alpha,beta,lik))

    -lik
  }

  if (is.na(initval[1])) {
    initval <- c(length(t)/t_max, .5, 1)
  }

  temp <- optim(initval, fn, control=list(trace=FALSE))

  return(list(mu=temp$par[1], alpha=temp$par[2], beta=temp$par[3], loglik=-temp$value))

}

#' MLE Estimation of Temporal Hawkes Model Parameters with Categorical Marks
#'
#' Maximum likelihood estimation of the parameters of a temporal exponential decay Hawkes model
#'
#'
#' @param t - vector of arrival times
#' @param marks - vector of marks
#' @param t_max - maximum time value (default = max(times))
#' @param initval - initial parameter values for likelihood optimization
#' @param print - print progress (default = TRUE)
#' @return A list containing the parameter values and likelihood value
#' @export
temporal.catmark.mle <- function(t, marks, t_max=max(t), initval=NA, print=TRUE){
  if (is.na(t_max))
    t_max <- max(t)

  fn <- function(params){
    mu <- params[1]
    alpha <- params[2]
    beta <- params[3]
    if (mu <= 0 || alpha < 0 || beta < 0)
      return(Inf)

    if (alpha > beta)
      return(Inf)

    lik <- temporal_likelihood(t, mu, alpha, beta, t_max)
    if (print)
      print(c(mu,alpha,beta,lik))

    -lik
  }

  if (is.na(initval[1])) {
    initval <- c(length(t)/t_max, .5, 1)
  }

  temp <- optim(initval, fn, control=list(trace=FALSE))

  p_temp <-as.numeric(table(marks)/length(marks))

  return(list(mu=temp$par[1], alpha=temp$par[2], beta=temp$par[3],p=p_temp, loglik=-temp$value))

}

#' MLE Estimation of Spatio-Temporal Hawkes Model Parameters
#'
#' Maximum likelihood estimation of the parameters of a spatio-temporal exponential decay Hawkes model.
#'
#' @param data - A DataFrame containing \eqn{x},\eqn{y}, and \eqn{t}
#' @param poly - a matrix defining the polygon
#' @param t_max - maximum time value (default = max(times))
#' @param initval - vector of two elements describing missing time range (default = NA)
#' @param print - print progress (default = TRUE)
#' @return A list containing the parameter values and likelihood value
#' @export
stpp.mle <- function(data, poly, t_max=max(data$t), initval=NA, print=TRUE){

  x <- data$x
  y <- data$y
  t <- data$t

  fn <- function(params){
    mu <- params[1]
    a <- params[2]
    b <- params[3]
    sig <- params[4]
    if (mu <= 0 || a < 0 || b < 0 || sig <= 0)
      return(Inf)

    if (a > b)
      return(Inf)

    lik <- stpp_likelihood(x, y, t, poly, mu, a, b, sig, t_max)
    if (print)
      print(c(mu,a,b,sig,lik))

    -lik
  }

  if (is.na(initval[1])) {
    initval <- c(length(t)/t_max, .5, 1, .01)
  }

  temp <- optim(initval, fn, control=list(trace=FALSE))

  return(list(mu=temp$par[1], a=temp$par[2], b=temp$par[3], sig=temp$par[4], loglik=-temp$value))
}

#' MLE Estimation of Nonuniform Spatio-Temporal Hawkes Model Parameters
#'
#' Maximum likelihood estimation of the parameters of a spatio-temporal exponential decay Hawkes model.
#'
#' @param data - A DataFrame containing \eqn{x},\eqn{y}, and \eqn{t}
#' @param poly - a matrix defining the polygon
#' @param t_max - maximum time value (default = max(times))
#' @param initval - vector of two elements describing missing time range (default = NA)
#' @param print - print progress (default = TRUE)
#' @return A list containing the parameter values and likelihood value
#' @export
stpp.mle.nonunif <- function(data, poly, t_max=max(data$t), initval=NA, print=TRUE){

  x <- data$x
  y <- data$y
  t <- data$t

  fn <- function(params){
    mu <- params[1]
    a <- params[2]
    b <- params[3]
    sig <- params[4]
    mux <- params[5]
    muy <- params[6]
    sigx <- params[7]
    sigy <- params[8]

    if (mu <= 0 || a < 0 || b < 0)
      return(Inf)

    if (a >= b)
      return(Inf)

    if (sig <= 0 | sigx <= 0 | sigy <= 0)
      return(Inf)

    lik <- stpp_likelihood_nonunif(x, y, t, mu, a, b, sig,mux, muy, sigx, sigy, t_max)
    if (print)
      print(c(mu,a,b,sig,lik))

    -lik
  }

  if (is.na(initval[1])) {
    initval <- c(length(t)/t_max, .5, 1, 0.1,mean(x),mean(y),var(x),var(y))
  }

  temp <- optim(initval, fn, control=list(trace=FALSE))

  return(list(mu=temp$par[1], a=temp$par[2], b=temp$par[3], sig=temp$par[4], mux=temp$par[5], muy=temp$par[6], sigx=temp$par[7], sigy=temp$par[8],loglik=-temp$value))
}

