/* Efficient numerical code to support BinaryMatrix.py

   Daniel Klein, 12/4/2012 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define G_ind(k,i,j) (G[(k)*m*(n-1) + (i)*(n-1) + (j)])
#define G_ind_n_init(k,i,j) (G[(k)*m*(n_init-1) + (i)*(n_init-1) + (j)])
#define S_ind(j,i) (S[(j)*m + (i)])
#define A_ind_n_init(r,c) (A[(r)*n_init + (c)])
#define B1_ind(p) (B[(2*p)])
#define B2_ind(p) (B[(2*p+1)])
#define wopt_ind(i,j) (wopt[(i)*n + (j)])
#define logwopt_ind(i,j) (logwopt[(i)*n + (j)])
#define logw_ind_n_init(i,j) (logw[(i)*n_init + (j)])

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
	if (isinf(Gk_den)) {
	  G_ind(k,i,j) = -1.0;
	} else {
	  G_ind(k,i,j) = wopt_ind(i,j) * exp(Gk_num-Gk_den) * \
	    ((n-j-k-1.0)/(k+1.0));
	}
      }
      if (isinf(Gk_den)) {
	G_ind(r_max,i,j) = -1.0;
      }
    }
  }
}

double core_cnll(int *A,
		 int count, int m, int n,
		 int *r, int *rndx, int *irndx,
		 int *csort, int *cndx, int *cconj,
		 double *G,
		 double *S, double *SS, int *B) {
  int i, j, c1;
  double SSS;
  
  int place = -1;

  double cnll = 0.0;

  const int n_init = n;
  for(c1 = 0; c1 < n_init; ++c1) {
    int placestart = place + 1;

    int clabel = cndx[c1];
    int colval = csort[c1];
    if (count == 0) {
      break;
    }

    for(i = 0; i < colval; ++i) {
      cconj[i] -= 1;
    }

    n -= 1;

    int smin = colval;
    int smax = colval;
    int cumsums = count;
    int cumconj = count - colval;

    count -= colval;

    SS[colval] = 0.0;
    SS[colval+1] = 1.0;
    SS[colval+2] = 0.0;

    for(i = (m-1); i >= 0; --i) {
      int rlabel = rndx[i];
      int val = r[rlabel];

      double p = val / (n + 1.0);
      double q = 1.0 - p;

      if ((n > 0) && (val > 0)) {
	double Gk = G_ind_n_init(val-1, rlabel, c1);
	if ((Gk < 0) || (q <= 0) || (p >= 1)) {
	  q = 0.0;
	  p = 1.0;
	} else {
	  p = p / (1.0 - p) * Gk;
	  p = p / (1.0 + p);
	  q = 1.0 - p;
	}
      }

      cumsums -= val;
      cumconj -= cconj[i];

      smin -= 1;
      if ((cumsums - cumconj) > smin) {
	smin = cumsums - cumconj;
      }
      if (smin < 0) {
	smin = 0;
      }
      if (i < smax) {
	smax = i;
      }

      SSS = 0.0;
      SS[smin] = 0.0;
      for(j = (smin+1); j < (smax+2); ++j) {
	double a = SS[j] * q;
	double b = SS[j+1] * p;
	double apb = a + b;
	SSS += apb;
	SS[j] = apb;
	S_ind(j,i) = b / (apb + DBL_EPSILON);
      }
      SS[smax+2] = 0.0;

      if (SSS <= 0) {
	break;
      }

      for(j = (smin+1); j < (smax+2); ++j) {
	SS[j] = SS[j] / SSS;
      }
    }

    if (SSS <= 0) {
      break;
    }
  
    j = 1;
    int jmax = colval + 1;

    if (j < jmax) {
      for(i = 0; i < m; ++i) {
	double p = S_ind(j,i);
	int rlabel = rndx[i];
	if (A_ind_n_init(rlabel,clabel) == 1) {
	  int val = r[rlabel];
	  r[rlabel] -= 1;

	  place += 1;
	  cnll -= log(p);
	  B1_ind(place) = rlabel;
	  B2_ind(place) = clabel;
	  j += 1;

	  if (j == jmax) {
	    break;
	  }
	} else {
	  cnll -= log(1.0 - p);
	}
      }
    }
    if (count == 0) {
      break;
    }

    for(j = place; j >= placestart; --j) {
      int k = B1_ind(j);
      int val = r[k];
      int irndxk = irndx[k];

      int irndxk1 = irndxk + 1;
      if ((irndxk1 >= m) || r[rndx[irndxk1]] <= val) {
	continue;
      }

      irndxk1 +=1;
      while((irndxk1 < m) && r[rndx[irndxk1]] > val) {
	irndxk1 += 1;
      }
      irndxk1 -= 1;

      int rndxk1 = rndx[irndxk1];
      rndx[irndxk] = rndxk1;
      rndx[irndxk1] = k;
      irndx[k] = irndxk1;
      irndx[rndxk1] = irndxk;
    }
  }

  return cnll;
}

void core_sample(double *logw,
		 int count, int m, int n,
		 int *r, int *rndx, int *irndx,
		 int *csort, int *cndx, int *cconj,
		 double *G, double *rvs,
		 double *S, double *SS,
		 int *B, double *logQ, double *logP) {
  int i, j, c1;
  double SSS;

  int rand_p = 0;
  
  int place = -1;

  const int n_init = n;
  for(c1 = 0; c1 < n_init; ++c1) {
    int placestart = place + 1;

    int clabel = cndx[c1];
    int colval = csort[c1];
    if (count == 0) {
      break;
    }

    for(i = 0; i < colval; ++i) {
      cconj[i] = cconj[i] - 1;
    }

    n -= 1;

    int smin = colval;
    int smax = colval;
    int cumsums = count;
    int cumconj = count - colval;

    count -= colval;

    SS[colval] = 0.0;
    SS[colval+1] = 1.0;
    SS[colval+2] = 0.0;

    for(i = (m-1); i >= 0; --i) {
      int rlabel = rndx[i];
      int val = r[rlabel];

      double p = val / (n + 1.0);
      double q = 1.0 - p;

      if ((n > 0) && (val > 0)) {
	double Gk = G_ind_n_init(val-1, rlabel, c1);
	if ((Gk < 0) || (q <= 0) || (p >= 1)) {
	  q = 0.0;
	  p = 1.0;
	} else {
	  p = p / (1.0 - p) * Gk;
	  p = p / (1.0 + p);
	  q = 1.0 - p;
	}
      }

      cumsums -= val;
      cumconj -= cconj[i];

      smin -= 1;
      if ((cumsums - cumconj) > smin) {
	smin = cumsums - cumconj;
      }
      if (smin < 0) {
	smin = 0;
      }
      if (i < smax) {
	smax = i;
      }

      SSS = 0.0;
      SS[smin] = 0.0;
      for(j = (smin+1); j < (smax+2); ++j) {
	double a = SS[j] * q;
	double b = SS[j+1] * p;
	double apb = a + b;
	SSS += apb;
	SS[j] = apb;
	S_ind(j,i) = b / (apb + DBL_EPSILON);
      }
      SS[smax+2] = 0.0;

      if (SSS <= 0) {
	break;
      }

      for(j = (smin+1); j < (smax+2); ++j) {
	SS[j] = SS[j] / SSS;
      }
    }

    if (SSS <= 0) {
      break;
    }
  
    j = 1;
    int jmax = colval + 1;

    if (j < jmax) {
      for(i = 0; i < m; ++i) {
	double p = S_ind(j,i);
	double rv = rvs[rand_p++];
	if (rv < p) {
	  int rlabel = rndx[i];
	  int val = r[rlabel];
	  r[rlabel] = r[rlabel] - 1;

	  place += 1;
	  B1_ind(place) = rlabel;
	  B2_ind(place) = clabel;
	  j += 1;

	  *logQ += log(p);
	  *logP += logw_ind_n_init(rlabel, clabel);
	  
	  if (j == jmax) {
	    break;
	  }
	} else {
	  *logQ += log(1.0 - p);
	}
      }
    }
    if (count == 0) {
      break;
    }

    for(j = place; j >= placestart; --j) {
      int k = B1_ind(j);
      int val = r[k];
      int irndxk = irndx[k];

      int irndxk1 = irndxk + 1;
      if ((irndxk1 >= m) || r[rndx[irndxk1]] <= val) {
	continue;
      }

      irndxk1 +=1;
      while((irndxk1 < m) && r[rndx[irndxk1]] > val) {
	irndxk1 += 1;
      }
      irndxk1 -= 1;

      int rndxk1 = rndx[irndxk1];
      rndx[irndxk] = rndxk1;
      rndx[irndxk1] = k;
      irndx[k] = irndxk1;
      irndx[rndxk1] = irndxk;
    }
  }
}
