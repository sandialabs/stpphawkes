#include "ptinpoly.h"
#include <RcppArmadillo.h>
using namespace Rcpp;

static double pxl1, pxu1, pyl1, pyu1, pxl2, pyl2;

//' Calculate if points are in the polynomial
//'
//' @param x - vector of x coordinates
//' @param y - vector of y coordinates
//' @param xp - vector of x coordinates of polynomial
//' @param yp - vector of y coordinates of polynomial
//' @param bb - matrix of bounding box of polynomial
//' @return inout - vector of 1 if point is in polynomial and 0 if not
//' @export
// [[Rcpp::export]]
arma::ivec ptinpoly(arma::vec& x, arma::vec& y, arma::vec& xp, arma::vec& yp,
                    arma::mat& bb){
int np = xp.n_elem;
int npts = x.n_elem;
arma::ivec result(npts);
arma::vec bbi(4);
bbi(0) = bb(0, 0);
bbi(1) = bb(1, 0);
bbi(2) = bb(0, 1);
bbi(3) = bb(1, 1);

int* _result = result.memptr();
double* _x = x.memptr();
double* _y = y.memptr();
double* _xp = xp.memptr();
double* _yp = yp.memptr();
double* _bb = bbi.memptr();

ptinpoly1(_result, _x, _y, _xp, _yp, np, _bb, npts);

return (result);
}

void ptinpoly1(int* presult, double* xpt, double* ypt, double* xbuf, double* ybuf, int numpts, double* bb, int npts) {
    int i;
    double *xs, *ys;
    double xpts, ypts;
    xs = (double*)malloc(numpts * sizeof(double));
    ys = (double*)malloc(numpts * sizeof(double));

    frset_pip(bb[0], bb[1], bb[2], bb[3]);

    for (i = 0; i < numpts; i++)
        dscale_pip(xbuf[i], ybuf[i], &xs[i], &ys[i]);
    for (i = 0; i < npts; i++) {
        dscale_pip(xpt[i], ypt[i], &xpts, &ypts);
        ptinpoly2(&presult[i], xpts, ypts, xs, ys, numpts);
    }

    free(xs);
    free(ys);

    return;
}

void frset_pip(double xl, double xu, double yl, double yu) {
    pxl1 = xl;
    pyl1 = yl;
    pxu1 = xu;
    pyu1 = yu;
    pxl2 = (pxu1 + pxl1) / 2;
    pyl2 = (pyu1 + pyl1) / 2;
}

void dscale_pip(double xo, double yo, double* xs, double* ys) {
    /* Scales (xo, yo) to (xs, ys) ( -1 < xs, ys < 1 ) within bounding box */
    *xs = (xo - pxl2) / (pxu1 - pxl2);
    *ys = (yo - pyl2) / (pyu1 - pyl2);
}

/* Function to determine whether a point lies inside, outside or on the
* boundary of a polygon.
* Count the number of times a horizontal line from the point to minus
* infinity crosses segments of the polygon. Special cases arise when
* the polygon points lie on this horizontal line and the direction
* (in Y) of the lines on either side of the offending points have to
* be considered. Further complications arise if whole polygon segments
* lie along the horizontal line. In this case the directions of the
* polygon segments at either end of the one or more horizontal
* segments have to be considered. Assume the 1st and last polygon
* points are the same.
*/
void ptinpoly2(int* presult, double xpt, double ypt, double* xbuf, double* ybuf, int numpts) {
    /*    int   i;*/
    int numcrosses;
    int ptr;
    double ratio;
    double xcross;
    double ydif;
    int thisyup = FALSE, lastyup;

    /* First decide on the direction of the segments leading from the
    last points to the first point in the polygon (taking any horizontal
    segments into account. */
    ptr = numpts - 2;
    while ((ybuf[0] == ybuf[ptr]) && (ptr != 0))
        ptr--;
    lastyup = FALSE; /* down */
    if (ybuf[0] > ybuf[ptr])
        lastyup = TRUE; /* up */

    /* Loop through each segment of the polygon - only stopping
    prematurely if the point is found to lie on one of the polygon
    segments. */
    numcrosses = 0;
    *presult = 1;
    ptr = 0;
    while ((ptr != numpts - 1) && (*presult != 0)) {
        /* Does this polygon segment go up or down? */
        if (ybuf[ptr] < ybuf[ptr + 1])
            thisyup = TRUE;
        if (ybuf[ptr] > ybuf[ptr + 1])
            thisyup = FALSE;

        /* Does the point lie within the Y bounds of the segment? */
        if ((ypt < fmax2(ybuf[ptr], ybuf[ptr + 1])) && (ypt > fmin2(ybuf[ptr], ybuf[ptr + 1]))) {
            /* Could the horz line from the point possibly cross
            the segment? */
            if (xpt >= fmin2(xbuf[ptr], xbuf[ptr + 1])) {
                /* Does the horz line from the point definitely cross
                the segment? */
                if (xpt <= fmax2(xbuf[ptr], xbuf[ptr + 1])) {
                    /* Work out whether the horz line from the point does
                    in fact cross the polygon segment. If the segment
                    is horizontal then the point must lie on the
                    polygon segment. If it is not horizontal then the
                    crossing point of the two lines must be worked
                    out and examined to see if it is to the right or
                    the left of the data point. Rounding errors are
                    significant and have to be dealt with. */
                    ydif = ybuf[ptr + 1] - ybuf[ptr];
                    if (ydif != 0.0) {
                        ratio = (ypt - ybuf[ptr]) / ydif;
                        xcross = xbuf[ptr] + (ratio * (xbuf[ptr + 1] - xbuf[ptr]));
                        if (xcross < xpt)
                            numcrosses++;
                        if ((xcross - xpt < 0.000001) && (xcross - xpt > -0.000001))
                            *presult = 0;
                    } else {
                        numcrosses++;
                        *presult = 0;
                    }
                } else
                    numcrosses++;
            }
        } else {
            /* The Y value of the data point does not lie inside those of
            the current polygon boundary segment. Does the current
            polygon boundary point have the same Y-value as the data
            point? */
            if (ypt == ybuf[ptr]) {
                /* Yes. Is the data point the same as the current polygon
                boundary point? */
                if (xpt == xbuf[ptr])
                    *presult = 0;
                else {
                    /* If the segment is horizontal then work out whether
                    the data point lies on the line? If not then we
                    just ignore this segment. */
                    if (ybuf[ptr] == ybuf[ptr + 1]) {
                        if ((xpt >= fmin2(xbuf[ptr], xbuf[ptr + 1])) && (xpt <= fmax2(xbuf[ptr], xbuf[ptr + 1])))
                            *presult = 0;
                    } else {
                        /* Could the horz line possibly cross the segment? */
                        if (xpt > xbuf[ptr]) {
                            /* The segment is not horz so simply check
                            whether this polygon boundary point is an
                            extrema or not. */
                            if (thisyup == lastyup)
                                numcrosses++;
                        }
                    }
                }
            }
        }
        lastyup = thisyup;
        ptr++;
    }

    /* Now we can decide whether the point is inside, outside, or on the
    edge of the polygon. */
    if (*presult != 0) {
        if (numcrosses % 2 == 0)
            *presult = 1;
        else
            *presult = -1;
    }
} /* end ptinpoly */

double fmax2(double x, double y) {
#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(y))
        return x + y;
#endif
    return (x < y) ? y : x;
}

double fmin2(double x, double y) {
#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(y))
        return x + y;
#endif
    return (x < y) ? x : y;
}
