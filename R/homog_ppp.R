#' Simulate a homogenous space-time Poisson process
#'
#' This function simulates a homogenous space-time Poisson process on \eqn{W}, defined by polygon
#'
#' @param mu - background parameter
#' @param poly - matrix defining polygon (\eqn{N} x \eqn{2})
#' @param t.region - vector of two elements describing time span
#' @param xfrac - x fractional increase of polygon to handle boundary effects (default = .1)
#' @param yfrac - y fractional increase (default = .1)
#' @param remove - remove points outside polygon (default = FALSE)
#' @param checkpoly - check if polygon is proper (default = TRUE)
#' @param showplot - plot points (default = FALSE)
#' @return A DataFrame containing \eqn{x},\eqn{y},\eqn{t}
#' @export
homog.STPP <- function(mu, poly, t.region, xfrac=.1, yfrac=.1, 
                       remove=FALSE, checkpoly=TRUE, showplot=FALSE){

  if (checkpoly) {
    xar <- areapl(poly)
    yar <- areapl(bboxx(bbox(poly)))
    zar <- areapl(poly[chull(poly),])
    if (xar < (yar/5) || xar < (zar/5))
      stop("Polygon argument may be malformed")
  }

  lr <- larger.region(poly, xfrac, yfrac)
  s.area <- (lr[2]-lr[1])*(lr[4]-lr[3])
  t.region <- sort(t.region)
  t.area <- t.region[2]-t.region[1]

  npts <- round(rpois(n=1,lambda=mu * s.area * t.area))

  x <- runif(npts,lr[1],lr[2])
  y <- runif(npts,lr[3],lr[4])
  times <- runif(npts,min=t.region[1],max=t.region[2])
  samp <- sample(1:npts,npts,replace=F)
  times <- times[samp]
  times <- sort(times)

  if (remove){
    inoutv = inout(x, y, poly, T)
    x <- x[inoutv]
    y <- y[inoutv]
    times <- times[inoutv]
  }


  if (showplot){
    plot(x, y)
  }

  #if(length(unqiue(out$t))<length(out$t)){
  #  t=t+rnorm(length(t),0,1E-16)
  #}
  out = data.frame(x=x,y=y,t=times,type=as.character(1:length(times)),stringsAsFactors=F)

  return(out)
}


homog.SPPP <- function(mu, poly, checkpoly=T, showplot=T){
    # simulate a homogenous Poisson process on W, defined by poly

    if (checkpoly) {
	    xar <- areapl(poly)
	    yar <- areapl(bboxx(bbox(poly)))
	    zar <- areapl(poly[chull(poly),])
	    if (xar < (yar/5) || xar < (zar/5))
	        stop("Polygon argument may be malformed")
	}

    lr <- larger.region(poly)
    area <- (lr[2]-lr[1])*(lr[4]-lr[3])

    mu <- mu * area
    npts <- rpois(1,mu)

    x <- runif(npts,lr[1],lr[2])
    y <- runif(npts,lr[3],lr[4])

    out <- pip(x, y, poly)

    if (showplot){
        plot(out$x, out$y)
    }

    return(cbind(out$x,out$y))
}

homog.PPP <- function(mu, t.region=c(0,1), seed=NULL){

  if (is.null(seed))
    seed <- .Random.seed
  else
    set.seed(seed)

  t.region <- sort(t.region)
  t.area <- t.region[2]-t.region[1]

  npts <- round(rpois(n=1,lambda=mu * t.area))

  times <- runif(npts,min=t.region[1],max=t.region[2])
  times <- sort(times)

  return(times)
}
