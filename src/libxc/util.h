/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef _LDA_H
#define _LDA_H

/* These are generic header files that are needed basically everywhere */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "xc.h"

/* we include the references also */
#include "references.h"

/* need config to figure out what needs to be defined or not */
#include "config.h"

#ifndef M_E
# define M_E            2.7182818284590452354   /* e */
#endif
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif
#ifndef M_SQRT2
# define M_SQRT2        1.41421356237309504880  /* sqrt(2) */
#endif

#define POW_2(x) ((x)*(x))
#define POW_3(x) ((x)*(x)*(x))

#define POW_1_2(x) sqrt(x)
#define POW_1_4(x) sqrt(sqrt(x))
#define POW_3_2(x) ((x)*sqrt(x))

#ifdef HAVE_CBRT
#define CBRT(x)    cbrt(x)
#define POW_1_3(x) cbrt(x)
#define POW_2_3(x) (cbrt(x)*cbrt(x))
#define POW_4_3(x) ((x)*cbrt(x))
#define POW_5_3(x) ((x)*cbrt(x)*cbrt(x))
#define POW_7_3(x) ((x)*(x)*cbrt(x))
#else
#define CBRT(x) pow((x), 1.0/3.0)
#define POW_1_3(x) pow((x), 1.0/3.0)
#define POW_2_3(x) pow((x), 2.0/3.0)
#define POW_4_3(x) pow((x), 4.0/3.0)
#define POW_5_3(x) pow((x), 5.0/3.0)
#define POW_7_3(x) pow((x), 7.0/3.0)
#endif

#define M_SQRTPI        1.772453850905516027298167483341145182798L
#define M_SQRT3         1.732050807568877293527446341505872366943L
#define M_CBRT2         1.259921049894873164767210607278228350570L
#define M_CBRT3         1.442249570307408382321638310780109588392L
#define M_CBRT4         1.587401051968199474751705639272308260391L
#define M_CBRT5         1.709975946676696989353108872543860109868L
#define M_CBRT6         1.817120592832139658891211756327260502428L
#define M_CBRT7         1.912931182772389101199116839548760282862L
#define M_CBRT9         2.080083823051904114530056824357885386338L

/* Very useful macros */
#ifndef min
#define min(x,y)  (((x)<(y)) ? (x) : (y))
#endif
#ifndef max
#define max(x,y)  (((x)<(y)) ? (y) : (x))
#endif

/* some useful constants */
#define LOG_DBL_MIN   (log(DBL_MIN))
#define LOG_DBL_MAX   (log(DBL_MAX))
#define SQRT_DBL_EPSILON   (sqrt(DBL_EPSILON))

/* special functions */
#define Heaviside(x) (((x) >= 0) ? 1.0 : 0.0)
#ifndef DEVICE
double LambertW(double z);
double xc_dilogarithm(const double x);
#else
__device__ double LambertW(double z);
__device__ double xc_dilogarithm(const double x);
#endif

/* we define this function here, so it can be properly inlined by all compilers */
#ifndef DEVICE
static inline double
xc_cheb_eval(const double x, const double *cs, const int N)
#else
__device__ static inline  double xc_cheb_eval(const double x, const double *cs, const int N)
#endif
{
  int i;
  double twox, b0, b1, b2;

  b2 = b1 = b0 = 0.0;

  twox = 2.0*x;
  for(i=N-1; i>=0; i--){
    b2 = b1;
    b1 = b0;
    b0 = twox*b1 - b2 + cs[i];
  }

  return 0.5*(b0 - b2);
}

#ifndef DEVICE
double xc_bessel_I0_scaled(const double x);
double xc_bessel_I0(const double x);
double xc_bessel_K0_scaled(const double x);
double xc_bessel_K0(const double x);
double xc_bessel_K1_scaled(const double x);
double xc_bessel_K1(const double x);
double xc_expint_e1_impl(double x, const int scale);
static inline double expint_e1(const double x)         { return  xc_expint_e1_impl( x, 0); }
static inline double expint_e1_scaled(const double x)  { return  xc_expint_e1_impl( x, 1); }
static inline double expint_Ei(const double x)         { return -xc_expint_e1_impl(-x, 0); }
#define Ei(x) expint_Ei(x)
static inline double expint_Ei_scaled(const double x)  { return -xc_expint_e1_impl(-x, 1); }
#else
double xc_bessel_I0_scaled(const double x);
double xc_bessel_I0(const double x);
double xc_bessel_K0_scaled(const double x);
double xc_bessel_K0(const double x);
double xc_bessel_K1_scaled(const double x);
double xc_bessel_K1(const double x);
__device__ double xc_expint_e1_impl(double x, const int scale);
__device__ static __inline__ double expint_e1(const double x)         { return  xc_expint_e1_impl( x, 0); }
__device__ static __inline__ double expint_e1_scaled(const double x)  { return  xc_expint_e1_impl( x, 1); }
__device__ static __inline__ double expint_Ei(const double x)         { return -xc_expint_e1_impl(-x, 0); }
#define Ei(x) expint_Ei(x)
__device__ static __inline__ double expint_Ei_scaled(const double x)  { return -xc_expint_e1_impl(-x, 1); }
#endif

/* integration */
typedef void integr_fn(double *x, int n, void *ex);
double xc_integrate(integr_fn func, void *ex, double a, double b);
void xc_rdqagse(integr_fn f, void *ex, double *a, double *b, 
	     double *epsabs, double *epsrel, int *limit, double *result,
	     double *abserr, int *neval, int *ier, double *alist__,
	     double *blist, double *rlist, double *elist, int *iord, int *last);
  
typedef struct xc_functional_key_t {
  char name[256];
  int  number;
} xc_functional_key_t;


#define M_C 137.0359996287515 /* speed of light */

#define RS_FACTOR      0.6203504908994000166680068120477781673508     /* (3/(4*Pi))^1/3        */
#define X_FACTOR_C     0.9305257363491000250020102180716672510262     /* 3/8*cur(3/pi)*4^(2/3) */
#define X_FACTOR_2D_C  1.504505556127350098528211870828726895584      /* 8/(3*sqrt(pi))        */
#define K_FACTOR_C     4.557799872345597137288163759599305358515      /* 3/10*(6*pi^2)^(2/3)   */
#define MU_GE          0.1234567901234567901234567901234567901235     /* 10/81                 */
#define X2S            0.1282782438530421943003109254455883701296     /* 1/(2*(6*pi^2)^(1/3))  */
#define X2S_2D         0.1410473958869390717370198628901931464610     /* 1/(2*(4*pi)^(1/2))    */
#define FZETAFACTOR    0.5198420997897463295344212145564567011405     /* 2^(4/3) - 2           */

#define RS(x)          (RS_FACTOR/CBRT(x))
#define FZETA(x)       ((pow(1.0 + (x),  4.0/3.0) + pow(1.0 - (x),  4.0/3.0) - 2.0)/FZETAFACTOR)
#define DFZETA(x)      ((CBRT(1.0 + (x)) - CBRT(1.0 - (x)))*(4.0/3.0)/FZETAFACTOR)
#define D2FZETA(x)     ((4.0/9.0)/FZETAFACTOR)* \
  (fabs(x)==1.0 ? (FLT_MAX) : (pow(1.0 + (x), -2.0/3.0) + pow(1.0 - (x), -2.0/3.0)))
#define D3FZETA(x)     (-(8.0/27.0)/FZETAFACTOR)* \
  (fabs(x)==1.0 ? (FLT_MAX) : (pow(1.0 + (x), -5.0/3.0) - pow(1.0 - (x), -5.0/3.0)))

#define MIN_GRAD             5.0e-13
#define MIN_TAU              5.0e-13
#define MIN_ZETA             5.0e-13

/* The following inlines confuse the xlc compiler */
void xc_rho2dzeta(int nspin, const double *rho, double *d, double *zeta);
void xc_fast_fzeta(const double x, const int nspin, const int order, double * fz);
void xc_mix_init(xc_func_type *p, int n_funcs, const int *funcs_id, const double *mix_coef);

/* LDAs */
void xc_lda_init(xc_func_type *p);
void xc_lda_end (xc_func_type *p);

typedef struct xc_lda_work_t {
  int   order; /* to which order should I return the derivatives */
  double rs, z;

  double f;                                   /* energy per unit particle */
  double dfdrs, dfdz;                         /*  first derivatives of e  */
  double d2fdrs2, d2fdrsz, d2fdz2;            /* second derivatives of e  */
  double d3fdrs3, d3fdrs2z, d3fdrsz2, d3fdz3; /*  third derivatives of e  */

#if defined CUDA || defined CUDA_MPIV
  int nspin;
#endif
} xc_lda_work_t;

#if defined CUDA || defined CUDA_MPIV
void xc_lda_fxc_fd(const xc_func_type *p, int np, const double *rho, double *fxc, void *gpu_work_params);
void xc_lda_kxc_fd(const xc_func_type *p, int np, const double *rho, double *kxc, void *gpu_work_params);
#else
void xc_lda_fxc_fd(const xc_func_type *p, int np, const double *rho, double *fxc);
void xc_lda_kxc_fd(const xc_func_type *p, int np, const double *rho, double *kxc);
#endif

/* the different possibilities for screening the interaction */
#define XC_RSF_ERF      0
#define XC_RSF_ERF_GAU  1
#define XC_RSF_YUKAWA   2

typedef void xc_lda_func_type (const xc_func_type *p, xc_lda_work_t *r);

void xc_lda_x_attenuation_function_erf(int order, double aa, double *f, double *df, double *d2f, double *d3f);
void xc_lda_x_attenuation_function_erf_gau(int order, double aa, double *f, double *df, double *d2f, double *d3f);
void xc_lda_x_attenuation_function_yukawa(int order, double aa, double *f, double *df, double *d2f, double *d3f);
void xc_lda_x_attenuation_function(int interaction, int order, double aa, double *f, double *df, double *d2f, double *d3f);

/* direct access to the internal functions */
void xc_lda_x_func     (const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_x_erf_func (const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_c_hl_func  (const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_c_vwn_func (const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_c_pw_func  (const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_c_pz_func  (const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_c_rc04_func(const xc_func_type *p, xc_lda_work_t *r);
void xc_lda_c_2d_amgb_func(const xc_func_type *p, xc_lda_work_t *r);

/* GGAs */
typedef struct xc_gga_work_x_t {
  int   order; /* to which order should I return the derivatives */
  double x;

  double f;          /* enhancement factor       */
  double dfdx;       /* first derivatives of f  */
  double d2fdx2;     /* second derivatives of zk */
  double d3fdx3;
} xc_gga_work_x_t;

void work_gga_becke_init(xc_func_type *p);

/* exchange enhancement factors: if you add one, please add it also to the util.c */
typedef void(*xc_gga_enhancement_t)(const xc_func_type *, xc_gga_work_x_t *r);
xc_gga_enhancement_t xc_get_gga_enhancement_factor(int func_id);

void xc_gga_x_wc_enhance   (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_pbe_enhance  (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_pw91_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_rpbe_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_htbs_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_b86_enhance  (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_b88_enhance  (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_g96_enhance  (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_pw86_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_airy_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_ak13_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_bayesian_enhance(const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_bpccac_enhance(const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_c09x_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_am05_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_dk87_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_herman_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_lg93_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_lv_rpw86_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_mpbe_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_optx_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_sogga11_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_ssb_sw_enhance (const xc_func_type *p, xc_gga_work_x_t *r);
void xc_gga_x_vmt_enhance (const xc_func_type *p, xc_gga_work_x_t *r);

/* these functions are used in more than one functional */
void xc_lda_c_pw_g(int func, int order, int k, double *rs, double *f, double *dfdrs, double *d2fdrs2, double *d3fdrs3);
void xc_beta_Hu_Langreth (double r, int order, double *b, double *dbdr, double *d2bdr2);

typedef struct xc_gga_work_c_t {
  int   order; /* to which order should I return the derivatives */

  double dens, ds[2], sigmat, sigmas[3];
  double rs, z, xt, xs[2];

  double f;

  double dfdrs, dfdz, dfdxt, dfdxs[2];
  double d2fdrs2, d2fdrsz, d2fdrsxt, d2fdrsxs[2], d2fdz2, 
    d2fdzxt, d2fdzxs[2], d2fdxt2, d2fdxtxs[2], d2fdxs2[3];

  double d3fdrs3, d3fdz3, d3fdxt3, d3fdxs3[4]; /* uuu, uud, udd, ddd */
  double d3fdrs2z, d3fdrs2xt, d3fdrs2xs[2];
  double d3fdrsz2, d3fdz2xt, d3fdz2xs[2];
  double d3fdrsxt2, d3fdzxt2, d3fdxt2xs[2];
  double d3fdrsxs2[3], d3fdzxs2[3],d3fdxtxs2[3];
  double d3fdrszxt, d3fdrszxs[2], d3fdrsxtxs[2], d3fdzxtxs[2];
} xc_gga_work_c_t;

void xc_gga_c_pw91_func(const xc_func_type *p, xc_gga_work_c_t *r);
void xc_gga_c_pbe_func (const xc_func_type *p, xc_gga_work_c_t *r);
void xc_gga_c_pbeloc_func (const xc_func_type *p, xc_gga_work_c_t *r);
void xc_gga_c_regtpss_func (const xc_func_type *p, xc_gga_work_c_t *r);
void xc_gga_c_scan_e0_func (const xc_func_type *p, xc_gga_work_c_t *r);
void xc_gga_c_q2d_func (const xc_func_type *p, xc_gga_work_c_t *r);

/* meta GGAs */
typedef struct xc_mgga_work_x_t {
  int   order; /* to which order should I return the derivatives */
  double rs, zeta, x, t, u;

  double f;                                   /* enhancement factor       */
  double dfdrs, dfdx, dfdt, dfdu;             /* first derivatives of f  */
  double d2fdrs2, d2fdx2, d2fdt2, d2fdu2;     /* second derivatives of zk */
  double d2fdrsx, d2fdrst, d2fdrsu, d2fdxt, d2fdxu, d2fdtu;
} xc_mgga_work_x_t;

typedef struct xc_mgga_work_c_t {
  int   order; /* to which order should I return the derivatives */

  double dens, ds[2], sigmat, sigmas[3];
  double rs, z, xt, xs[2], ts[2], us[2];

  double f;
  double dfdrs, dfdz, dfdxt, dfdxs[2], dfdts[2], dfdus[2];
  double d2fdrs2, d2fdrsz, d2fdrsxt, d2fdrsxs[2], d2fdrsts[2], d2fdrsus[2];
  double d2fdz2, d2fdzxt, d2fdzxs[2], d2fdzts[2], d2fdzus[2];
  double d2fdxt2, d2fdxtxs[2], d2fdxtts[2], d2fdxtus[2];
  double d2fdxs2[3], d2fdxsts[4], d2fdxsus[4];
  double d2fdts2[3], d2fdtsus[4];
  double d2fdus2[3];
  double d3fdrs3, d3fdrs2z, d3fdrsz2, d3fdrszxt, d3fdrszxs[2], d3fdrszts[2], d3fdrszus[2];
  double d3fdrs2xt, d3fdrsxt2, d3fdrsxtxs[2], d3fdrsxtts[2], d3fdrsxtus[2], d3fdrs2xs[2];
  double d3fdrsxs2[3], d3fdrsxsts[4], d3fdrsxsus[4], d3fdrs2ts[2], d3fdrsts2[3];
  double d3fdrstsus[4], d3fdrs2us[2], d3fdrsus2[3];
  double d3fdz3, d3fdz2xt, d3fdzxt2, d3fdzxtxs[2], d3fdzxtts[2], d3fdzxtus[2];
  double d3fdz2xs[2], d3fdzxs2[3], d3fdzxsts[4], d3fdzxsus[4], d3fdz2ts[2], d3fdzts2[3];
  double d3fdztsus[4], d3fdz2us[2], d3fdzus2[3];
  double d3fdxt3, d3fdxt2xs[2], d3fdxtxs2[3], d3fdxtxsts[4], d3fdxtxsus[4], d3fdxt2ts[2];
  double d3fdxtts2[3], d3fdxttsus[4], d3fdxt2us[2], d3fdxtus2[3];
  double d3fdxs3[4], d3fdxs2ts[6], d3fdxs2us[6], d3fdxsts2[6], d3fdxstsus[8], d3fdxsus2[6];
  double d3fdts3[4], d3fdts2us[6], d3fdtsus2[6], d3fdus3[4];
} xc_mgga_work_c_t;


void xc_mgga_x_scan_falpha(int order, double a, double c1, double c2, double dd, double *f, double *dfda);

/* useful MACROS */
#define DFRACTION(num, dnum, den, dden) \
  (((dnum)*(den) - (num)*(dden))/((den)*(den)))
#define D2FRACTION(num, dnum, d2num, den, dden, d2den) \
  ((2.0*(num)*(dden)*(dden) - 2.0*(den)*(dden)*(dnum) - (den)*(num)*(d2den) + (den)*(den)*(d2num))/((den)*(den)*(den)))
#define D3FRACTION(num, dnum, d2num, d3num, den, dden, d2den, d3den)	\
  ((-(num)*(6.0*(dden)*(dden)*(dden) - 6.0*(den)*(dden)*(d2den) + (den)*(den)*(d3den)) + \
    (den)*(6.0*(dden)*(dden)*(dnum) - 3.0*(den)*(dden)*(d2num) + (den)*(-3.0*(dnum)*(d2den) + (den)*(d3num))))/((den)*(den)*(den)*(den)))

/* Some useful functions */
const char *get_kind(const xc_func_type *func);
const char *get_family(const xc_func_type *func);


#if defined CUDA || defined CUDA_MPIV
#include "gpu.h"
#endif

#endif
