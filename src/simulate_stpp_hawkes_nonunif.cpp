#include <RcppArmadillo.h>
#include "nonunif_ppp.h"
#include "polygon.h"
#include "utilities.h"
// Correctly setup the build environment
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


//' Simulate inhomogenous spatio-temporal hawkes model
//'
//' @param params - list containing params (\eqn{\mu}, \eqn{a}, \eqn{b}, \eqn{\sigma},\eqn{\mu x}, \eqn{\mu y}, \eqn{\sigma x}, \eqn{\sigma y} )
//' @param poly - matrix defining polygon (\eqn{N} x \eqn{2})
//' @param t_region - vector of two elements describing time region (e.g., c(0,10))
//' @param d - generate parents on larger polygon by expanded observed polygon by d (default = R::qnorm(.95, 0, sig, 1, 0))
//' @param history - history of process (e.g., numeric())
//' @param seed - set random number seed (default=-1)
//' @return A DataFrame containing \eqn{x},\eqn{y},\eqn{t}
//' @export
// [[Rcpp::export]]
DataFrame simulate_hawkes_stpp_nonunif(List params, arma::mat poly, arma::vec t_region, double d, DataFrame history,
                               int seed = -1) {
    arma::mat history1 = DFtoMat(history);

    if (seed != -1) {
        set_seed((unsigned int)seed);
    }

    double mu = params["mu"];
    double a = params["a"];
    double b = params["b"];
    double sig = params["sig"];
    double mux = params["mux"];
    double muy = params["muy"];
    double sigx = params["sigx"];
    double sigy = params["sigy"];

    sig = sqrt(sig);

    if (std::isnan(d)) {
        d = R::qnorm(.95, 0, sig, 1, 0);
    }

    if (mu < 0) {
        stop("mu needs to be greater than 0");
    }
    if (a < 0) {
        stop("a needs to be greater than 0");
    }
    if (b < 0) {
        stop("b needs to be greater than 0");
    }

    if (a >= b) {
        stop("b needs to be greater than a: UNSTABLE");
    }

    if (sigx < 0) {
      stop("sigx needs to be greater than 0");
    }
    if (sigy < 0) {
      stop("sigy needs to be greater than 0");
    }

    // work out the temporal length we should run each offspring sequence
    // this is the fraction of offspring we want
    // each sequence to be short by, on average
    double fraction = 0.01;
    double time_ext = -b * log(fraction);

    // Generate the background catalog as a Poisson process with the background intensity µ
    // do this on larger region in space and time to overcome edge effects
     double rng = d;
     //rng = R::qnorm(.95, 0, sig, 1, 0);
    //double rng = d;
    arma::vec xr = {min(poly.col(0)), max(poly.col(0))};
    arma::vec yr = {min(poly.col(1)), max(poly.col(1))};
    double xw = xr[1] - xr[0];
    double yw = yr[1] - yr[0];

    arma::vec t_region1(2);
    if (t_region[1] <= 2 * time_ext) {
        t_region1(0) = t_region[0];
        t_region1(1) = 2 * time_ext;
    } else {
        t_region1(0) = t_region[0] - time_ext;
        t_region1(1) = t_region[1];
    }

    if (history1.n_elem > 0) {
        t_region1 = t_region;
    }


    arma::mat bgrd = nonunif_STPP(mu, mux, muy, sigx, sigy, poly, t_region1, rng / xw, rng / yw);

    if (history1.n_elem > 0) {
        arma::vec t_tmp = history1.col(2);
        arma::uvec idx1 = find(t_tmp < t_region[0], 1, "last");

        if (idx1.n_elem > 0) {
            arma::mat history_sub = history1.rows(0, idx1[0]);
            bgrd = join_cols(history_sub, bgrd);
        }
    }

    bgrd.insert_cols(3, 1);

    int l = 0;
    std::vector<arma::mat> G;
    G.push_back(bgrd);

    std::vector<double> ti;
    std::vector<double> xi;
    std::vector<double> yi;
    std::vector<double> zi;

    int npts;
    int N = 1;

    do {
        // For each event in catalog G^l, simulate its N_j offspring where N_j is rpois(a)
        ti.clear();
        xi.clear();
        yi.clear();
        zi.clear();

        for (unsigned int ii = 0; ii < G[l].n_rows; ii++) {
            npts = R::rpois(a);
            for (int jj = 0; jj < npts; jj++) {
                ti.push_back(G[l](ii, 2) + R::rexp(1.0 / b));
                xi.push_back(G[l](ii, 0) + R::rnorm(0, sig));
                yi.push_back(G[l](ii, 1) + R::rnorm(0, sig));
                zi.push_back(l + 1);
            }
        }

        if (ti.empty()) {
            break;
        }
        // sort offspring times
        arma::vec tif = arma::conv_to<arma::vec>::from(ti);
        arma::vec xif = arma::conv_to<arma::vec>::from(xi);
        arma::vec yif = arma::conv_to<arma::vec>::from(yi);
        arma::vec zif = arma::conv_to<arma::vec>::from(zi);
        arma::uvec idx = sort_index(tif);
        tif = tif(idx);
        xif = xif(idx);
        yif = yif(idx);
        zif = zif(idx);

        // continue if offspring in time region were generated
        idx = find(tif <= t_region[1]);
        N = idx.n_elem;

        if (N > 0) {
            arma::mat tmp(tif.n_elem, 4);
            tmp.col(0) = xif;
            tmp.col(1) = yif;
            tmp.col(2) = tif;
            tmp.col(3) = zif;

            G.push_back(tmp);
            l++;
        }

    } while (N > 0);

    // Combine all the generated points
    arma::mat out;

    if (l == 0) {
        arma::mat outf(0, 3);
        return (outf);
    }

    for (int i = 0; i < l; i++) {
        out = join_cols(out, G[i]);
    }

    // only remove points that are outisde time region
    arma::uvec ind = find((out.col(2) >= t_region(0)) && (out.col(2) <= t_region(1)));
    arma::mat out2 = out.rows(ind);

    // sort events
    ind = sort_index(out2.col(2));
    arma::vec out_x = out2.col(0);
    out_x = out_x(ind);
    arma::vec out_y = out2.col(1);
    out_y = out_y(ind);
    arma::vec out_t = out2.col(2);
    out_t = out_t(ind);
    arma::vec out_z = out2.col(3);
    out_z = out_z(ind);

    DataFrame df = DataFrame::create(Rcpp::Named("x") = out_x, Rcpp::Named("y") = out_y, Rcpp::Named("t") = out_t,
                                     Rcpp::Named("z") = out_z);

    return (df);
}


arma::mat simulate_hawkes_nonunif_stpp_c(double mu, double a, double b, double sig, double mux, double muy, double sigx, double sigy, arma::mat poly, arma::vec t_region,
                                     DataFrame history, bool sp_clip) {
      arma::mat history1 = DFtoMat(history);
      sig = sqrt(sig);

      if (mu < 0) {
        stop("mu needs to be greater than 0");
      }
      if (a < 0) {
        stop("a needs to be greater than 0");
      }
      if (b < 0) {
        stop("b needs to be greater than 0");
      }

      if (a >= b) {
        stop("b needs to be greater than a: UNSTABLE");
      }
      if (sigx < 0) {
        stop("sigx needs to be greater than 0");
      }
      if (sigy < 0) {
        stop("sigy needs to be greater than 0");
      }

      // work out the temporal length we should run each offspring sequence
      // this is the fraction of offspring we want
      // each sequence to be short by, on average
      double fraction = 0.01;
      double time_ext = -b * log(fraction);

      // Generate the background catalog as a Poisson process with the background intensity µ
      // do this on larger region in space and time to overcome edge effects
      double rng = R::qnorm(.95, 0, sig, 1, 0);

      arma::vec xr = {min(poly.col(0)), max(poly.col(0))};
      arma::vec yr = {min(poly.col(1)), max(poly.col(1))};
      double xw = xr[1] - xr[0];
      double yw = yr[1] - yr[0];

      arma::vec t_region1(2);
      if (t_region[1] <= 2 * time_ext) {
        t_region1(0) = t_region[0];
        t_region1(1) = 2 * time_ext;
      } else {
        t_region1(0) = t_region[0] - time_ext;
        t_region1(1) = t_region[1];
      }

      if (history1.n_elem > 0) {
        t_region1 = t_region;
      }

      arma::mat bgrd = nonunif_STPP(mu, mux, muy, sigx, sigy, poly, t_region1, rng / xw, rng / yw);
      if (history1.n_elem > 0) {
        arma::vec t_tmp = history1.col(2);
        arma::uvec idx1 = find(t_tmp < t_region[0], 1, "last");

        if (idx1.n_elem > 0) {
          arma::mat history_sub = history1.rows(0, idx1[0]);
          bgrd = join_cols(history_sub, bgrd);
        }
      }

      int l = 0;
      std::vector<arma::mat> G;
      G.push_back(bgrd);

      std::vector<double> ti;
      std::vector<double> xi;
      std::vector<double> yi;

        int npts;
        int N = 1;
        do {
          // For each event in catalog G^l, simulate its N_j offspring where N_j is rpois(a)
          ti.clear();
          xi.clear();
          yi.clear();

          for (unsigned int ii = 0; ii < G[l].n_rows; ii++) {
            npts = R::rpois(a);
            for (int jj = 0; jj < npts; jj++) {
              ti.push_back(G[l](ii, 2) + R::rexp(1.0 / b));
              xi.push_back(G[l](ii, 0) + R::rnorm(0, sig));
              yi.push_back(G[l](ii, 1) + R::rnorm(0, sig));
            }
          }

          if (ti.empty()) {
            break;
          }

          // sort offspring times
          arma::vec tif = arma::conv_to<arma::vec>::from(ti);
          arma::vec xif = arma::conv_to<arma::vec>::from(xi);
          arma::vec yif = arma::conv_to<arma::vec>::from(yi);
          arma::uvec idx = sort_index(tif);
          tif = tif(idx);
          xif = xif(idx);
          yif = yif(idx);

          // continue if offspring in time region were generated
          idx = find(tif <= t_region[1]);
          N = idx.n_elem;

          if (N > 0) {
            arma::mat tmp(xif.n_elem, 3);
            tmp.col(0) = xif;
            tmp.col(1) = yif;
            tmp.col(2) = tif;
            G.push_back(tmp);
            l++;
          }

        } while (N > 0);

        // Combine all the generated points
        arma::mat out;

        if (l == 0) {
          arma::mat outf(0, 3);
          return (outf);
        }

        for (int i = 0; i < l; i++) {
          out = join_cols(out, G[i]);
        }

        if (sp_clip) {
          // remove points outside polygon
          arma::vec tmpx = out.col(0);
          arma::vec tmpy = out.col(1);
          arma::uvec inoutv = inout(tmpx, tmpy, poly, true);
          arma::uvec ind = find(inoutv > 0);
          arma::mat out1 = out.rows(ind);

          // and remove points that are outisde time region
          ind = find((out1.col(2) >= t_region(0)) && (out1.col(2) <= t_region(1)));
          arma::mat out2 = out1.rows(ind);

          // sort events
          ind = sort_index(out2.col(2));
          arma::vec out_x = out2.col(0);
          out_x = out_x(ind);
          arma::vec out_y = out2.col(1);
          out_y = out_y(ind);
          arma::vec out_t = out2.col(2);
          out_t = out_t(ind);

          arma::mat outf(out_t.n_elem, 3);
          outf.col(0) = out_x;
          outf.col(1) = out_y;
          outf.col(2) = out_t;

          return (outf);

        } else {
          // only remove points that are outisde time region
          arma::uvec ind = find((out.col(2) >= t_region(0)) && (out.col(2) <= t_region(1)));
          arma::mat out2 = out.rows(ind);

          // sort events
          ind = sort_index(out2.col(2));
          arma::vec out_x = out2.col(0);
          out_x = out_x(ind);
          arma::vec out_y = out2.col(1);
          out_y = out_y(ind);
          arma::vec out_t = out2.col(2);
          out_t = out_t(ind);

          arma::mat outf(out_t.n_elem, 3);
          outf.col(0) = out_x;
          outf.col(1) = out_y;
          outf.col(2) = out_t;

          return (outf);
        }
      }
