simulate_hawkes_stpp_inhom <- function(mu, params, poly, t.region=NULL, seed=NULL, ...){
  # This simulates event times, called "times", according to a self-exciting
  # point process with paraemters param

  if (is.null(seed))
    seed <- .Random.seed
  else
    set.seed(seed)

  if (ndims(params$mu!=0))
    stop("mu needs to be a matrix")
  if (params$a<=0)
    stop("a needs to be greater than 0")
  if (params$b<=0)
    stop("b needs to be greater than 0")

  # work out the temporal length we should run each offspring sequence
  fraction <- 0.01 ## this is the fraction of offspring we want
  ## each sequence to be short by, on average
  n <- params$a
  time.ext <- -params$b*log(fraction)

  # Generate the background catalog as a Poisson process with the background intensity µ
  # do this on larger region in space and time to overcome edge effects
  rng <- qnorm(.95, sd=params$sig)
  xr <- c(min(poly[,1]),max(poly[,1]))
  yr <- c(min(poly[,2]),max(poly[,2]))
  xw <- xr[2] - xr[1]
  yw <- yr[2] - yr[1]
  if (is.null(t.region))
    t.region1 <- c(0-2*n,2*time.ext)
  else if (t.region[2]<=2*time.ext)
    t.region1 <- c(0-2*n,2*time.ext)
  else
    t.region1 <- c(0-2*n,t.region[2])
  bgrd <- inhomog.STPP(mu, poly, t.region1, nx=params$nx, ny=params$ny,
                       nt=params$nt, xfrac=rng/xw, yfrac=rng/yw, ...)
  l <- 1
  G <- list()
  G[[l]] <- bgrd

  repeat
  {
    # For each event in catalog G^l, simulate its N_j offspring where N_j is rpois(a)
    ti <- c()
    xi <- c()
    yi <- c()
    typei <- c()
    for (ii in 1:nrow(G[[l]])){
      npts <- rpois(1,params$a)
      if (npts>0){
        for (jj in 1:npts){
          ti <- c(ti, G[[l]]$t[ii] + rexp(1,rate=params$b))
          xi <- c(xi, G[[l]]$x[ii] + rnorm(1, sd=params$sig))
          yi <- c(yi, G[[l]]$y[ii] + rnorm(1, sd=params$sig))
          new.index <- paste(G[[l]]$type[ii], ".", jj, sep="")
          typei <- c(typei, new.index)
        }
      }

    }

    if (length(ti)==0)
      break
    # sort offspring times
    ix <- sort(ti, index.return=T)$ix
    ti <- ti[ix]
    xi <- xi[ix]
    yi <- yi[ix]
    typei <- typei[ix]

    # remove any offspring whose time is beyond time window
    id <- ti < t.region1[2]
    l <- l+1

    # check if any offspring generated, if not terminate
    if (sum(id)!=0)
      G[[l]] <- data.frame(x=xi[id],y=yi[id],t=ti[id],type=typei[id], stringsAsFactors = F)

    if (sum(id)==0) break
  }

  # Combine all the generated points
  ti <- c()
  xi <- c()
  yi <- c()
  typei <- c()
  for (ii in 2:length(G)){
    ti <- c(ti, G[[ii]]$t)
    xi <- c(xi, G[[ii]]$x)
    yi <- c(yi, G[[ii]]$y)
    typei <- c(typei, G[[ii]]$type)
  }

  # Check for events that are outside polygon
  inoutv <- inout(xi, yi, poly, T)
  xi <- xi[inoutv]
  yi <- yi[inoutv]
  ti <- ti[inoutv]
  typei <- typei[inoutv]

  # check for events that are earlier than the burn-in period
  ix <- ti >= time.ext
  ti <- ti[ix]
  xi <- xi[ix]
  yi <- yi[ix]
  typei <- typei[ix]

  # remove points beyond user time region
  if (!is.null(t.region)){
    ix <- ti >= t.region[1] & ti <= t.region[2]
    ti <- ti[ix]
    xi <- xi[ix]
    yi <- yi[ix]
    typei <- typei[ix]
  }

  # sort events
  ix <- sort(ti, index.return=T)$ix
  ti <- ti[ix]
  xi <- xi[ix]
  yi <- yi[ix]
  typei <- typei[ix]

  # return data
  out <- data.frame(x=xi,y=yi,t=ti,type=typei,stringsAsFactors = F)
  attr(out, "seed") <- seed

  return(out)

}

simulate_hawkes_stpp_lomax <- function(params, poly, t.region=NULL){
  # This simulates event times, called "times", according to a self-exciting
  # point process with paraemters param
  # coords = c(x1, y1, x2, y2)

  if (params$c<0)
    stop("c needs to be greater than 0")
  if (params$mu<0)
    stop("mu needs to be greater than 0")
  if (params$a<=0)
    stop("a needs to be greater than 0")
  if (params$b<=0)
    stop("b needs to be greater than 0")

  # work out the temporal length we should run each offspring sequence
  fraction <- 0.95 ## this is the fraction of offspring we want
  ## each sequence to be short by, on average
  n <- params$a
  time.ext <- ((1-fraction)^(1/-params$c)-1)/params$b

  # Generate the background catalog as a Poisson process with the background intensity µ
  # do this on larger region in space and time to overcome edge effects
  rng <- qnorm(.95, sd=params$sig)
  xr <- c(min(poly[,1]),max(poly[,1]))
  yr <- c(min(poly[,2]),max(poly[,2]))
  xw <- xr[2] - xr[1]
  yw <- yr[2] - yr[1]
  t.region1 <- c(0-2*n,time.ext)
  bgrd <- homog.STPP(params$mu, poly, t.region1, xfrac=rng/xw, yfrac=rng/yw)
  l <- 1
  G <- list()
  G[[l]] <- bgrd

  repeat
  {
    # For each event in catalog G^l, simulate its N_j offspring where N_j is rpois(a)
    ti <- c()
    xi <- c()
    yi <- c()
    typei <- c()
    for (ii in 1:nrow(G[[l]])){
      npts <- rpois(1,params$a)
      if (npts>0){
        for (jj in 1:npts){
          ti <- c(ti, G[[l]]$t[ii] + extraDistr::rlomax(1,params$b,params$c))
          xi <- c(xi, G[[l]]$x[ii] + rnorm(1, sd=params$sig))
          yi <- c(yi, G[[l]]$y[ii] + rnorm(1, sd=params$sig))
          new.index <- paste(G[[l]]$type[ii], ".", jj, sep="")
          typei <- c(typei, new.index)
        }
      }

    }

    if (length(ti)==0)
      break
    # sort offspring times
    ix <- sort(ti, index.return=T)$ix
    ti <- ti[ix]
    xi <- xi[ix]
    yi <- yi[ix]
    typei <- typei[ix]

    # remove any offspring whose time is beyond time window
    id <- ti < t.region1[2]
    l <- l+1

    # check if any offspring generated, if not terminate
    if (sum(id)!=0)
      G[[l]] <- data.frame(x=xi[id],y=yi[id],t=ti[id],type=typei[id], stringsAsFactors = F)

    if (sum(id)==0) break
  }

  # Combine all the generated points
  ti <- c()
  xi <- c()
  yi <- c()
  typei <- c()
  for (ii in 1:length(G)){
    ti <- c(ti, G[[ii]]$t)
    xi <- c(xi, G[[ii]]$x)
    yi <- c(yi, G[[ii]]$y)
    typei <- c(typei, G[[ii]]$type)
  }

  # Check for events that are outside polygon
  inoutv <- inout(xi, yi, poly, T)
  xi <- xi[inoutv]
  yi <- yi[inoutv]
  ti <- ti[inoutv]
  typei <- typei[inoutv]

  # check for events that are earlier than t.region[1]
  ix <- ti >= 0
  ti <- ti[ix]
  xi <- xi[ix]
  yi <- yi[ix]
  typei <- typei[ix]

  # remove points beyond user time region
  if (!is.null(t.region)){
    ix <- ti >= t.region[1] & ti <= t.region[2]
    ti <- ti[ix]
    xi <- xi[ix]
    yi <- yi[ix]
    typei <- typei[ix]
  }

  # sort events
  ix <- sort(ti, index.return=T)$ix
  ti <- ti[ix]
  xi <- xi[ix]
  yi <- yi[ix]
  typei <- typei[ix]

  # return data
  out <- data.frame(x=xi,y=yi,t=ti,type=typei,stringsAsFactors = F)

  return(out)

}
