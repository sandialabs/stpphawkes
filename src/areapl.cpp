#include <RcppArmadillo.h>

using namespace Rcpp;

//' Calculate area of polynomial
//'
//' @param poly - matrix describing polynomial
//' @return W - area of polynomial
//' @export
// [[Rcpp::export]]
double areapl(const arma::mat& poly) {
    int np = poly.n_rows;
    double totare = 0;
    double x1, y1, x2, y2;
    arma::vec polyx(np + 1);
    arma::vec polyy(np + 1);
    for (int i = 0; i < np; i++) {
        polyx[i] = poly(i, 0);
        polyy[i] = poly(i, 1);
    }
    polyx[np] = poly(0, 0);
    polyy[np] = poly(0, 1);

    for (int is = 0; is <= np; is++) {
        x1 = polyx[is];
        y1 = polyy[is];

        if (is == np) {
            x2 = polyx[0];
            y2 = polyy[0];
        } else {
            x2 = polyx[is + 1];
            y2 = polyy[is + 1];
        }

        // Find the area of the trapezium
        totare += (x2 - x1) * (y2 + y1) / 2.0;
    }

    return (totare);
}
