inhomog.STPP <- function(mu, poly, t.region, nx=101, ny=101, nt=101, xfrac=.1,
                         yfrac=.1, remove=F, checkpoly=T, showplot=F, ...){
  # simulate a inhomogenous space-time Poisson process on W, defined by poly

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

  if (is.array(mu)){
    if (length(dim(mu))!=3) stop ("mu must be a 3D-array")
    nx <- dim(mu)[1]
    ny <- dim(mu)[2]
    nt <- dim(mu)[3]
    Mu <- mu
    # create grid for mu
    s.grid <- make.grid(nx,ny,poly)
    s.grid$mask <- matrix(as.logical(s.grid$mask),nx,ny)
    t.grid <- list(times=seq(t.region[1],t.region[2],length=nt),tinc=(t.area/(nt-1)))

  } else if(is.function(mu)){
    # create grid for mu
    s.grid <- make.grid(nx,ny,poly)
    s.grid$mask <- matrix(as.logical(s.grid$mask),nx,ny)
    t.grid <- list(times=seq(t.region[1],t.region[2],length.out=nt),tinc=(t.area/(nt-1)))
    Mu <- array(NaN,dim=c(nx,ny,nt))
    for(it in 1:nt){
      mu1 <- mu(as.vector(s.grid$X),as.vector(s.grid$Y),t.grid$times[it],...)
      M <- matrix(mu1,ncol=ny,nrow=nx,byrow=TRUE)
      M[!(s.grid$mask)] <- NaN
      Mu[,,it] <- M
    }
  }

  # calculate number of points
  en <- sum(Mu,na.rm=TRUE)*s.grid$xinc*s.grid$yinc*t.grid$tinc
  npoints <- round(rpois(n=1,lambda=en),0)
  mu.max <- max(Mu,na.rm=TRUE)
  npts <- npoints
  if (npts==0) stop("there is no data to thin")

  times.init <- runif(nt,min=t.region[1],max=t.region[2])
  samp <- sample(1:nt,npts,replace=T)
  times <- times.init[samp]

  # generate points
  x <- runif(npts,lr[1],lr[2])
  y <- runif(npts,lr[3],lr[4])

  # Accpetance/Rejection of points
  if (is.matrix(mu)){
    prob <- interp::bicubic(s.grid$x, s.grid$y, Mu, x, y)$z
    u <- runif(npts)
    retain <- u <= prob
    if (sum(retain==F)==length(retain)) stop ("no point was retained at the first iteration, please check your parameters")

    x <- x[retain]
    y <- y[retain]
    samp <- samp[retain]
    samp.remain <- (1:nt)[-samp]
    times <- times[retain]

  } else if(is.function(mu)){
    prob <- mu(x,y,times,...)/mu.max
    u <- runif(npts)
    retain <- u <= prob
    if (sum(retain==F)==length(retain)) stop ("no point was retained at the first iteration, please check your parameters")

    x <- x[retain]
    y <- y[retain]
    samp <- samp[retain]
    samp.remain <- (1:nt)[-samp]
    times <- times[retain]
  }

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

  out = data.frame(x=x,y=y,t=times,type=as.character(1:length(times)),stringsAsFactors=F)

  return(out)
}

inhomog.PPP <- function(mu, lambda, poly, l.x, l.y, checkpoly = T, showplot=T){
    # Given an intesity function lambda, simulate an inhomogenous PPP
    # it is assumed that lambda is defined on a grid larger than the polygon

    if (checkpoly) {
	    xar <- areapl(poly)
	    yar <- areapl(bboxx(bbox(poly)))
	    zar <- areapl(poly[chull(poly),])
	    if (xar < (yar/5) || xar < (zar/5))
	        stop("Polygon argument may be malformed")
	}

    p.homog <- homog.PPP(mu, poly)
    nmax <- nrow(p.homog)
    P <- lambda/max(lambda)

    # Accpetance/Rejection of points
    p_keep <- interp::bicubic(l.x, l.y, P, p.homog[,1], p.homog[,2])$z
    u <- runif(nmax)
    p <- p.homog[which(u < p_keep),]

    if (showplot) {
        plot(p[,1],p[,2])
    }

    return(p)
}
