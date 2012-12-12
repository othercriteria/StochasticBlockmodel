/* Efficient numerical code to support BinaryMatrix.py

   Daniel Klein, 12/4/2012 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define G_ind(k,i,j) (G[(k)*m*(n-1) + (i)*(n-1) + (j)])
#define wopt_ind(i,j) (wopt[(i)*n + (j)])
#define logwopt_ind(i,j) (logwopt[(i)*n + (j)])

void fill_G(int *r, int r_max, int m, int n,
	    double *wopt, double *logwopt, double *G) {
  int i, j, k;

  for(i = 0; i < m; ++i) {
    int ri = r[i];
    for(j = n-2; j > 0; --j) {
      double wij = logwopt_ind(i,j);
      for(k = 1; k < ri+1; ++k) {
	double b = G_ind(k-1,i,j) + wij;
	double a = G_ind(k,i,j);
	if ((a == -INFINITY) && (b == -INFINITY)) {
	  continue;
	}
	if (a > b) {
	  G_ind(k,i,j-1) = a + log(1.0 + exp(b-a));
	} else {
	  G_ind(k,i,j-1) = b + log(1.0 + exp(a-b));
	}
      }
    }
    for(j = 0; j < n-1; ++j) {
      double Gk_num, Gk_den;
      for(k = 0; k < r_max; ++k) {
	Gk_num = G_ind(k,i,j);
	Gk_den = G_ind(k+1,i,j);
	if isinf(Gk_den) {
	  G_ind(k,i,j) = -1.0;
	} else {
	  G_ind(k,i,j) = wopt_ind(i,j) * exp(Gk_num-Gk_den) * \
	    ((n-j-k-1.0)/(k+1.0));
	}
      }
      if isinf(Gk_den) {
	G_ind(r_max,i,j) = -1.0;
      }
    }
  }
}
