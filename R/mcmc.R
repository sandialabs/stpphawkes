#' Bayesian Estimation of Temporal Hawkes Model Parameters
#'
#' This function computes the posterior of the parameters of a temporal exponential decay Hawkes model
#' using Metropolis-with-in-Gibbs sampling.
#'
#' The default is to estimate the branching structure which is much more computationally efficient. The model will also account to missing data if \code{t_mis} is provided.
#'
#' Branching models specify gamma priors for mu, alpha and beta parameters.
#'
#' @param times - vector of arrival times
#' @param t_max - maximum time value (default = max(times))
#' @param t_mis - mx2 matrix, mth row contains two elements describing the mth missing time range (default = NULL)
#' @param param_init - list of parameters of initial guess (default = NULL, will start with MLE)
#' @param mcmc_param - list of mcmc parameters
#' @param branching - using branching structure in estimation (default = TRUE)
#' @param print - print progress (default = TRUE)
#' @return A DataFrame containing the mcmc samples
#' @examples
#' times = simulate_temporal(.5,.1,.5,c(0,10),numeric()) 
#' out = mcmc_temporal(times)
#' @export
mcmc_temporal <- function(times, t_max=max(times), t_mis=NULL, param_init=NULL, mcmc_param=NULL,
                          branching=TRUE, print=TRUE){

  if (is.null(mcmc_param)){
    mcmc_param <- list(n_mcmc=2000, n_burn=500, sig_mu=.5, sig_alpha=0.5, sig_beta=.5,mu_param=c(.1,.1),alpha_param=c(.1,.1),
                       beta_param=c(.1,.1))
  }

  if (is.null(param_init)){
    message("No default value for param_init, use MLE as starting point\n")
    param_init<-temporal.mle(times, t_max, print=FALSE)
  }

  if (is.null(t_mis)){

    if(branching){
      y_init <- rep(0, length(times))
      post_samps <- condInt_mcmc_temporal_branching(times, t_max, y_init,param_init$mu,
                                                    param_init$alpha, param_init$beta,
                                                    mcmc_param$mu_param, mcmc_param$alpha_param, mcmc_param$beta_param,
                                                    mcmc_param$sig_beta, mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    } else {
      post_samps <- condInt_mcmc_temporal(times, t_max, param_init$mu, param_init$alpha, param_init$beta,
                                          mcmc_param$sig_mu, mcmc_param$sig_alpha, mcmc_param$sig_beta,
                                          mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    }

  } else {
    if(!is.matrix(t_mis)){stop('t_mis must be a matrix')}
    if(dim(t_mis)[2]!=2){stop('t_mis has incorrect number of columns')}
    if(branching){
      y_init <- rep(0, length(times))
      post_samps <- condInt_mcmc_temporal_branching_md(times, t_mis, t_max, y_init, param_init$mu, param_init$alpha, param_init$beta,
                                                       mcmc_param$mu_param, mcmc_param$alpha_param, mcmc_param$beta_param,
                                                       mcmc_param$sig_beta, mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    } else {
      post_samps <- condInt_mcmc_temporal_md(times, t_mis, t_max, param_init$mu, param_init$alpha,
                                             param_init$beta, mcmc_param$sig_mu, mcmc_param$sig_alpha, mcmc_param$sig_beta,
                                             mcmc_param$n_mcmc, mcmc_param$n_burn, print, FALSE)

    }

  }

  return(post_samps)
}


#' Bayesian Estimation of Temporal Hawkes Model Parameters with Categorical Marks
#'
#' This function computes the posterior of the parameters of a temporal exponential decay Hawkes model
#' using Metropolis-with-in-Gibbs sampling.
#'
#' The default is to estimate the branching structure which is much more computationally efficient. The model will also account to missing data if \code{t_mis} is provided.
#'
#'
#' @param times - vector of arrival times
#' @param marks - vector of marks
#' @param t_max - maximum time value (default = max(times))
#' @param t_mis - mx2 matrix, mth row contains two elements describing the mth missing time range (default = NULL)
#' @param param_init - list of parameters of initial guess (default = NULL, will start with MLE)
#' @param mcmc_param - list of mcmc parameters
#' @param branching - using branching structure in estimation (default = TRUE)
#' @param print - print progress (default = TRUE)
#' @return A DataFrame containing the mcmc samples
#' @export
mcmc_temporal_catmark <- function(times, marks, t_max=max(times), t_mis=NULL, param_init=NULL, mcmc_param=NULL,
                          branching=TRUE, print=TRUE){
  # check inputs
  if(!is.factor(marks)){stop('marks is a factor, please specify number of levels')}

  if(mcmc_param$n_burn >= mcmc_param$n_mcmc) {stop('n_burn must be less than n_mcmc')}

  if (is.null(mcmc_param)){
    mcmc_param <- list(n_mcmc=2000, n_burn=500, sig_mu=.5, sig_alpha=.5, sig_beta=.5,mu_param=c(.1,.1), alpha_param=c(.1,.1),
                       beta_param=c(.1,.1),p_param=as.numeric(table(marks)))
  } else {
    if(!is.length2(mcmc_param$alpha_param)) {stop('alpha_param must be numeric, length 2')}
    if(!is.length2(mcmc_param$beta_param)) {stop('beta_param must be numeric, length 2')}
    if(!is.scalar(mcmc_param$sig_beta)) {stop('sig_beta must be numeric, length 1')}
    if(!is.scalar(mcmc_param$n_mcmc)) {stop('n_mcmc must be numeric, length 1')}
    if(length(mcmc_param$p_param) != length(levels(marks))) {stop('p_param must have length equal to the number of levels in marks')}
  }

  if (is.null(param_init)){
    message("No default value for param_init, use MLE as starting point\n")
    param_init<-temporal.catmark.mle(times, marks, t_max, print=FALSE)
  } else {
    if(!is.scalar(param_init$mu)) {stop('mu_init must be numeric, length 1')}
    if(!is.scalar(param_init$alpha)) {stop('alpha_init must be numeric, length 1')}
    if(!is.scalar(param_init$beta)) {stop('beta_init must be numeric, length 1')}
  }

  if (is.null(t_mis)){

    if(branching){
      post_samps <- CatMarkMcMc(times, t_max, marks, param_init$mu,
                                param_init$alpha, param_init$beta, mcmc_param$mu_param,
                                mcmc_param$alpha_param, mcmc_param$beta_param, mcmc_param$p_param,
                                mcmc_param$sig_beta, mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    } else {
      stop("Not Implemented\n")

    }

  } else {
    if(!is.matrix(t_mis)){stop('t_mis must be a matrix')}
    if(dim(t_mis)[2]!=2){stop('t_mis has incorrect number of columns')}
    if(length(param_init$p) != length(levels(marks))) {stop('p_init must have length equal to the number of levels in marks')}
    if(branching){
      post_samps <- CatMarkMcMcMissingData(times, t_mis, t_max, marks, param_init$mu,
                                           param_init$alpha, param_init$beta, param_init$p, mcmc_param$mu_param,
                                           mcmc_param$alpha_param, mcmc_param$beta_param, mcmc_param$p_param,
                                           mcmc_param$sig_beta, mcmc_param$n_mcmc, mcmc_param$n_burn, print)


    } else {
      stop("Not Implemented\n")

    }

  }

  return(post_samps)
}

#' Bayesian Estimation of Temporal Hawkes Model Parameters with Categorical Marks
#'
#' This function computes the posterior of the parameters of a temporal exponential decay Hawkes model
#' using Metropolis-with-in-Gibbs sampling.
#'
#' The default is to estimate the branching structure which is much more computationally efficient. The model will also account to missing data if \code{t_mis} is provided.
#'
#'
#' @param times - vector of arrival times
#' @param marks - vector of continuous marks
#' @param wshape - fixed weibull shape parameter
#' @param t_max - maximum time value (default = max(times))
#' @param t_mis - mx2 matrix, mth row contains two elements describing the mth missing time range (default = NULL)
#' @param param_init - list of parameters of initial guess (default = NULL, will start with MLE)
#' @param mcmc_param - list of mcmc parameters
#' @param branching - using branching structure in estimation (default = TRUE)
#' @param dist - distribution for marks string (default = "Weibull")
#' @param print - print progress (default = TRUE)
#' @return A DataFrame containing the mcmc samples
#' @export
mcmc_temporal_contmark <- function(times, marks, wshape,t_max=max(times), t_mis=NULL, param_init=NULL, mcmc_param=NULL,
                                  branching=TRUE, dist="Weibull",print=TRUE){
  # check inputs
  if(!is.numeric(marks)){stop('marks must be numeric')}

  if(mcmc_param$n_burn >= mcmc_param$n_mcmc) {stop('n_burn must be less than n_mcmc')}

  if (is.null(mcmc_param)){
    mcmc_param <- list(n_mcmc=2000, n_burn=500, sig_mu=.5, sig_alpha=.5, sig_beta=.5,mu_param=c(.1,.1), alpha_param=c(.1,.1),
                       beta_param=c(.1,.1),wscale_param=c(1,3))
  } else {
    if(!is.length2(mcmc_param$alpha_param)) {stop('alpha_param must be numeric, length 2')}
    if(!is.length2(mcmc_param$beta_param)) {stop('beta_param must be numeric, length 2')}
    if(!is.scalar(mcmc_param$sig_beta)) {stop('sig_beta must be numeric, length 1')}
    if(!is.scalar(mcmc_param$n_mcmc)) {stop('n_mcmc must be numeric, length 1')}
    #if(length(mcmc_param$p_param) != length(levels(marks))) {stop('p_param must have length equal to the number of levels in marks')}
  }

  if (is.null(param_init)){
    stop("Initial parameters needed")
  } else {
    if(!is.scalar(param_init$mu)) {stop('mu_init must be numeric, length 1')}
    if(!is.scalar(param_init$alpha)) {stop('alpha_init must be numeric, length 1')}
    if(!is.scalar(param_init$beta)) {stop('beta_init must be numeric, length 1')}
  }

  if (is.null(t_mis)){

    if(branching){
      if(dist=="Weibull"){
      post_samps <- WeibullMarkMcMc(times, t_max, marks, wshape, param_init$mu,
                                param_init$alpha, param_init$beta, param_init$wscale,mcmc_param$mu_param,
                                mcmc_param$alpha_param, mcmc_param$beta_param, mcmc_param$wscale_param,
                                mcmc_param$sig_beta, mcmc_param$n_mcmc, mcmc_param$n_burn, print)
      } else{
        stop("Not Implemented\n")
      }

    } else {
      stop("Not Implemented\n")

    }

  } else {
    #if(!is.matrix(t_mis)){stop('t_mis must be a matrix')}
    #if(dim(t_mis)[2]!=2){stop('t_mis has incorrect number of columns')}
    #if(length(param_init$p) != length(levels(marks))) {stop('p_init must have length equal to the number of levels in marks')}
    #if(branching){
    #  post_samps <- CatMarkMcMcMissingData(times, t_mis, t_max, marks, param_init$mu,
    #                                       param_init$alpha, param_init$beta, param_init$p, mcmc_param$mu_param,
    #                                       mcmc_param$alpha_param, mcmc_param$beta_param, mcmc_param$p_param,
    #                                       mcmc_param$sig_beta, mcmc_param$n_mcmc, mcmc_param$n_burn, print)


    #} else {
      stop("Not Implemented\n")

    #}

  }

  return(post_samps)
}


#' Bayesian Estimation of Spatio-Temporal Hawkes Model Parameters
#'
#' This function computes the posterior of a spatio-temporal exponential decay Hawkes model
#' using Metropolis-with-in-Gibbs sampling.
#'
#' The default is to estimate the branching structure.
#' The model will also account to missing data if \code{t_mis} is provided.
#'
#' @param data - A DataFrame containing \eqn{x},\eqn{y},\eqn{t}
#' @param poly - matrix defining polygon (\eqn{N} x \eqn{2})
#' @param t_max - maximum time value (default = max(times))
#' @param t_mis - vector of two elements describing missing time range (default = NULL)
#' @param param_init - list of parameters of initial guess (default = NULL, will start with MLE)
#' @param mcmc_param - list of mcmc parameters
#' @param branching - using branching structure in estimation (default = TRUE)
#' @param print - print progress (default = TRUE)
#' @param sp_clip - when simulating missing data spatial points, clip spatial region back to observed region (default = TRUE)
#' @return A DataFrame containing the mcmc samples
#' @export
mcmc_stpp <- function(data, poly, t_max=max(data$t), t_mis=NULL, param_init=NULL,mcmc_param=NULL,
                      branching=TRUE, print=TRUE, sp_clip=TRUE){

  if (is.null(mcmc_param)){
    mcmc_param <- list(n_mcmc=2000, n_burn=500, sig_mu=.5, sig_b=.5, sig_sig=.1, mu_param=c(.1,.1),a_param=c(0.1,0.1),b_param=c(0.1,0.1),sig_param=c(0.1,0.1))
  }

  if (is.null(param_init)){
    message("No default value for param_init, use MLE as starting point\n")
    param_init<-stpp.mle(data, poly, t_max, print=FALSE)
  }

  if (is.null(t_mis)){

    if(branching){
      y_init <- rep(0, length(data$t))
      post_samps <- condInt_mcmc_stpp_branching(data, t_max, y_init, param_init$mu, param_init$a, param_init$b,param_init$sig,
                                                poly, mcmc_param$mu_param,mcmc_param$a_param,mcmc_param$sig_param,mcmc_param$b_param,mcmc_param$sig_b,mcmc_param$sig_sig,
                                                mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    } else {
      stop("intensity function mcmc not yet implemented, need branching=TRUE")
      #post_samps <- condInt_mcmc_stpp(data, t_max, param_init$mu, param_init$a, param_init$b, param_init$sig, poly,
      #                                mcmc_param$sig_mu, mcmc_param$sig_a, mcmc_param$sig_b, mcmc_param$sig_sig,
      #                                mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    }

  } else {
    if(!is.matrix(t_mis)){stop('t_mis must be a matrix')}
    if(dim(t_mis)[2]!=2){stop('t_mis has incorrect number of columns')}
    if(branching){
      y_init <- rep(0, length(data$t))
      post_samps <- condInt_mcmc_stpp_branching_md(data, t_mis, t_max, y_init, param_init$mu, param_init$a, param_init$b,param_init$sig,
                                                   poly, mcmc_param$mu_param,mcmc_param$a_param,mcmc_param$sig_param,mcmc_param$b_param,mcmc_param$sig_b,mcmc_param$sig_sig,
                                                   mcmc_param$n_mcmc, mcmc_param$n_burn, print, sp_clip)

    } else {
      stop("missing data is not implemented, need branching=TRUE")

    }

  }

  return(post_samps)
}

#' Bayesian Estimation of Spatio-Temporal Hawkes Model Parameters with non uniform spatial locations
#'
#' This function computes the posterior of a spatio-temporal exponential decay Hawkes model
#' using Metropolis-with-in-Gibbs sampling.
#'
#' The default is to estimate the branching structure.
#' The model will also account to missing data if \code{t_mis} is provided.
#'
#' @param data - A DataFrame containing \eqn{x},\eqn{y},\eqn{t}
#' @param poly - matrix defining polygon (\eqn{N} x \eqn{2})
#' @param t_max - maximum time value (default = max(times))
#' @param t_mis - vector of two elements describing missing time range (default = NULL)
#' @param param_init - list of parameters of initial guess (default = NULL, will start with MLE)
#' @param mcmc_param - list of mcmc parameters
#' @param branching - using branching structure in estimation (default = TRUE)
#' @param print - print progress (default = TRUE)
#' @param sp_clip - when simulating missing data spatial points, clip spatial region back to observed region (default = TRUE)
#' @return A DataFrame containing the mcmc samples
#' @export
mcmc_stpp_nonunif <- function(data, poly, t_max=max(data$t), t_mis=NULL, param_init=NULL,mcmc_param=NULL,
                      branching=TRUE, print=TRUE, sp_clip=TRUE){

  if (is.null(mcmc_param)){
    mcmc_param <- list(n_mcmc=2000, n_burn=500, sig_mu=.5, sig_b=.5, sig_sig=.1, mu_param=c(.1,.1),a_param=c(0.1,0.1),b_param=c(0.1,0.1),sig_param=c(0.1,0.1),
                       mux_param=c(mean(data$x),var(data$x)),muy_param=c(mean(data$y),var(data$y)),sigx_param=c(0.1,0.1),sigy_param=c(0.1,0.1))
  }

  if (is.null(param_init)){
    message("No default value for param_init, use MLE as starting point\n")
    param_init<-stpp.mle.nonunif(data, t_max, print=FALSE)
  }

  if (is.null(t_mis)){

    if(branching){
      y_init <- rep(0, length(data$t))
      post_samps <- condInt_mcmc_stpp_branching_nonunif(data, t_max, y_init, param_init$mu, param_init$a, param_init$b, param_init$sig,
                                                   param_init$mux, param_init$muy, param_init$sigx,param_init$sigy,
                                                   poly, mcmc_param$mu_param,mcmc_param$a_param,mcmc_param$sig_param,mcmc_param$b_param,mcmc_param$sig_b,mcmc_param$sig_sig,
                                                   mcmc_param$mux_param,mcmc_param$muy_param, mcmc_param$sigx_param,mcmc_param$sigy_param,
                                                   mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    } else {
      stop("intensity function mcmc not yet implemented, need branching=TRUE")
      #post_samps <- condInt_mcmc_stpp(data, t_max, param_init$mu, param_init$a, param_init$b, param_init$sig, poly,
      #                                mcmc_param$sig_mu, mcmc_param$sig_a, mcmc_param$sig_b, mcmc_param$sig_sig,
      #                                mcmc_param$n_mcmc, mcmc_param$n_burn, print)

    }

  } else {
    if(!is.matrix(t_mis)){stop('t_mis must be a matrix')}
    if(dim(t_mis)[2]!=2){stop('t_mis has incorrect number of columns')}
    if(branching){
      y_init <- rep(0, length(data$t))
      post_samps <- condInt_mcmc_stpp_branching_nonunif_md(data, t_mis, t_max, y_init, param_init$mu, param_init$a, param_init$b,param_init$sig,
                                                   param_init$mux, param_init$muy, param_init$sigx,param_init$sigy,
                                                   poly, mcmc_param$mu_param,mcmc_param$a_param,mcmc_param$sig_param,mcmc_param$b_param,mcmc_param$sig_b,mcmc_param$sig_sig,
                                                   mcmc_param$mux_param,mcmc_param$muy_param, mcmc_param$sigx_param,mcmc_param$sigy_param,
                                                   mcmc_param$n_mcmc, mcmc_param$n_burn, print, sp_clip)

    } else {
      stop("missing data is not implemented, need branching=TRUE")

    }

  }

  return(post_samps)
}
