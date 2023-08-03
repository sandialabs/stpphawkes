is.scalar <- function(x) is.numeric(x) && length(x) == 1L

is.length2 <- function(x) is.numeric(x) && length(x) == 2L

ndims <- function(x){
    return(length(dim(x)))
}

trapz <- function(x,y,dims=1){
    if ((dims-1)>0){
        perm = c(dims:max(ndims(y),dims), 1:(dims-1))
    } else {
        perm = c(dims:max(ndims(y),dims))
    }

    if (ndims(y) == 0){
        m = 1
    } else {
        if (length(x) != dim(y)[dims])
            stop('Dimension Mismatch')
        y = aperm(y, perm)
        m = nrow(y)
    }

    if (m==1){
        M = length(y)
        out = sum(diff(x)*(y[-M]+y[-1])/2)
    } else {
        slice1 = y[as.vector(outer(1:(m-1), dim(y)[1]*( 1:prod(dim(y)[-1])-1 ), '+')) ]
        dim(slice1) = c(m-1, length(slice1)/(m-1))
        slice2 = y[as.vector(outer(2:m, dim(y)[1]*( 1:prod(dim(y)[-1])-1 ), '+'))]
        dim(slice2) = c(m-1, length(slice2)/(m-1))
        out = t(diff(x)) %*% (slice1+slice2)/2.
        siz = dim(y)
        siz[1] = 1
        out = array(out, siz)
        perm2 = rep(0, length(perm))
        perm2[perm] = 1:length(perm)
        out = aperm(out, perm2)
        ind = which(dim(out) != 1)
        out = array(out, dim(out)[ind])
    }

    return(out)
}

cumtrapz <- function(x,y,dims=1){
  if ((dims-1)>0){
    perm = c(dims:max(ndims(y),dims), 1:(dims-1))
  } else {
    perm = c(dims:max(ndims(y),dims))
  }

  if (ndims(y) == 0){
    n = 1
    m = length(y)
  } else {
    if (length(x) != dim(y)[dims])
      stop('Dimension Mismatch')
    y = aperm(y, perm)
    m = nrow(y)
    n = ncol(y)
  }

  if (n==1){
    dt = diff(x)/2.0
    z = c(0, cumsum(dt*(y[1:(m-1)] + y[2:m])))
    dim(z) = c(m,1)
  } else {
    tmp = diff(x)
    dim(tmp) = c(m-1,1)
    dt = repmat(tmp/2.0,1,n)
    z = rbind(rep(0,n), apply(dt*(y[1:(m-1),] + y[2:m,]),2,cumsum))
    perm2 = rep(0, length(perm))
    perm2[perm] = 1:length(perm)
    z = aperm(z, perm2)
  }

  return(z)
}

repmat <- function(X,m,n){
  ##R equivalent of repmat (matlab)
  mx = dim(X)[1]
  if (is.null(mx)){
    mx = 1
    nx = length(X)
    mat = matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
  }else {
    nx = dim(X)[2]
    mat = matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
  }

  return(mat)
}

long2UTM <- function(long) {
    (floor((long + 180)/6) %% 60) + 1
}

plot_trace_1d<-function(post_samps, mu, alpha, beta){
  par(mfrow=c(3,1))
  plot(post_samps$mu, type='l', ylim=range(c(post_samps$mu, mu)))
  abline(h=mu, col=2, lwd=2)
  plot(post_samps$alpha, type='l', ylim=range(c(post_samps$alpha, alpha)))
  abline(h=alpha, col=2, lwd=2)
  plot(post_samps$beta, type='l', ylim=range(c(post_samps$beta, beta)))
  abline(h=beta, col=2, lwd=2)
  par(mfrow=c(1,1))
}
