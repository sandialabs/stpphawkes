#include <RcppArmadillo.h>
using namespace Rcpp;

#define FALSE 0
#define TRUE !FALSE

void frset_pip(double xl, double xu, double yl, double yu);
void dscale_pip(double xo, double yo, double* xs, double* ys);
void ptinpoly2(int* presult, double xpt, double ypt, double* xbuf, double* ybuf, int numpts);
void ptinpoly1(int* presult, double* xpt, double* ypt, double* xbuf, double* ybuf, int numpts, double* bb, int npts);
double fmax2(double x, double y);
double fmin2(double x, double y);
arma::ivec ptinpoly(arma::vec& x, arma::vec& y, arma::vec& xp, arma::vec& yp,
                    arma::mat& bb);
