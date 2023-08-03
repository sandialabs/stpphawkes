#' Point in polygon
#'
#' Determines if a point is in a polygon or on a polygon boundary
#'
#' @param x - vector of x positions
#' @param y - vector of y positions
#' @param poly - matrix defining polygon (\eqn{N} x \eqn{2})
#' @return A list containing the x and y coordiantes of the points inside the polygon
#'  @export
pip <- function(x, y, poly){
  inoutv = inout(x, y, poly, T)
  x1 = x[inoutv]
  y1 = y[inoutv]
  return(list(x=x1,y=y1))
}


larger.region <- function(poly, xfrac=.1, yfrac=.1){
  out <- sbox(poly,xfrac,yfrac)
  lr <- matrix(0,2,2)
  lr[1,1] <- min(out[,1])
  lr[2,1] <- max(out[,1])
  lr[1,2] <- min(out[,2])
  lr[2,2] <- max(out[,2])

  return(lr)
}

sbox <- function(poly, xfrac=.1, yfrac=.1){
  xr <- c(min(poly[,1]),max(poly[,1]))
  yr <- c(min(poly[,2]),max(poly[,2]))

  xw <- xr[2] - xr[1]
  xr[1] <- xr[1] - xfrac*xw
  xr[2] <- xr[2] + xfrac*xw

  yw <- yr[2] - yr[1]
  yr[1] <- yr[1] - yfrac*yw
  yr[2] <- yr[2] + yfrac*yw

  out <- matrix(0,4,2)
  out[1,1] <- xr[1]
  out[2,1] <- xr[2]
  out[3,1] <- xr[2]
  out[4,1] <- xr[1]
  out[1,2] <- yr[1]
  out[2,2] <- yr[1]
  out[3,2] <- yr[2]
  out[4,2] <- yr[2]

  return(out)
}


inpip <- function (x, y, poly, bound=T){
	seq(1:length(x))[inout(x, y, poly, bound)]
}

make.grid <- function(nx,ny,poly){
  if (missing(poly)) poly <- matrix(c(0,0,1,0,1,1,0,1),4,2,T)

  if ((nx < 2) || (ny < 2)) stop("the grid must be at least of size 2x2")

  xrang <- range(poly[, 1], na.rm = TRUE)
  yrang <- range(poly[, 2], na.rm = TRUE)
  xmin <- xrang[1]
  xmax <- xrang[2]
  ymin <- yrang[1]
  ymax <- yrang[2]

  xinc <- (xmax-xmin)/nx
  yinc <- (ymax-ymin)/ny

  xc <- xmin-xinc/2
  yc <- ymin-yinc/2
  xgrid <- rep(0,nx)
  ygrid <- rep(0,ny)
  xgrid[1] <- xc + xinc
  ygrid[1] <- yc + yinc

  for (i in 2:nx){
    xgrid[i] <- xgrid[i-1]+xinc
  }
  for (i in 2:ny){
    ygrid[i] <- ygrid[i-1]+yinc
  }

  yy <- matrix(xgrid,nx,ny)
  xx <- t(yy)
  yy <- matrix(ygrid,nx,ny)

  X <- as.vector(xx)
  Y <- as.vector(yy)

  poly <- rbind(poly,poly[1,])
  pts <- inpip(X,Y,poly)

  X[pts] <- TRUE
  X[X!=TRUE] <- FALSE
  mask <- matrix(X,ncol=ny,nrow=nx,byrow=TRUE)

  return(list(x=xgrid,y=ygrid,X=xx,Y=yy,pts=pts,xinc=xinc,yinc=yinc,mask=matrix(as.logical(mask),nx,ny)))
}
