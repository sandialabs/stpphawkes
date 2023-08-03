predict.hawkes_t <- function(post_samps, times, evalpt, t_mis=NULL){
  param <- colMeans(post_samps)

  if (!is.null(t_mis)){
    z <- simulate_temporal(param["mu"], param["alpha"]*param["beta"],
                             param["beta"], t_mis, times, -1)
    times<-sort(c(times,z))
  }

  lambda <- intensity_temporal(param["mu"], param["alpha"]*param["beta"],
                          param["beta"], times, evalpt)

  return(lambda)
}

predict.hawkes_st <- function(post_samps, x, y, times, poly, evalpt){
  param <- colMeans(post_samps)
  lambda <- intensity_stpp(param["mu"], param["a"], param["b"], param["sig"], x, y, times,
                           poly, evalpt)

  return(lambda)
}

predict.hawkes_np <- function(obj, x, y, t, v, mu, g) {
    term1 <- predict(v, x = t)*predict(mu, x = c(x,y))
    k <- which(obj$t < t)
    k <- k[length(k)]
    tk <- t - obj$t[1:k]
    xk <- x - obj$x[1:k]
    yk <- y - obj$y[1:k]
    term2 <- sum(predict(g, x = cbind(xk,yk,tk)))
    lambda <- term1 + term2
    return(lambda)
}

intensity_stpp <- function(mu, a, b, sig, x, y, times, poly, evalpt) {
  # area of polygon
  W <- areapl(poly)

  # determine if evalpt is in poly
  inoutv <- inout(evalpt[1], evalpt[2], poly, T)
  if (!inoutv){
    warning("evalpt is not in bounding region specificied by polygon")
    ci <- 0
    return(ci)
  }

  if (length(times) > 0){
    use <- times <= evalpt[3]
    if (sum(use)>0){
      beta <- b*exp(-b*(evalpt[3]-times[use]))
      alpha <- 1/(2*pi*sig^2)*exp(-((evalpt[1]-x[use])^2+(evalpt[2]-y[use])^2)/(2*sig^2))
      ci <- a * sum(beta*alpha)
    } else
      ci <- 0
  } else
    ci <- 0

  if (evalpt[3]>0)
    mu1 <- mu/W
  else
    mu1 <- 0

  ci <- mu1 + ci

  return(ci)
}
