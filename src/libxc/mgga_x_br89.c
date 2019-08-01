/* 
 Copyright (C) 2006-2009 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_X_BR89         206 /* Becke-Roussel 89  */ 
#define XC_MGGA_X_BJ06         207 /* Becke & Johnson correction to Becke-Roussel 89  */ 
#define XC_MGGA_X_TB09         208 /* Tran & Blaha correction to Becke & Johnson  */ 
#define XC_MGGA_X_RPP09        209 /* Rasanen, Pittalis, and Proetto correction to Becke & Johnson  */ 
#define XC_MGGA_X_B00          284 /* Becke 2000 */ 
 
typedef struct{ 
  double c; 
} mgga_x_tb09_params; 
 
static double br89_gamma = 0.8; 
static double b00_at     = 0.928; 
 
static void  
mgga_x_tb09_init(xc_func_type *p) 
{ 
  mgga_x_tb09_params *params; 
 
  p->params = malloc(sizeof(mgga_x_tb09_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(mgga_x_tb09_params); 
#endif 
  params = (mgga_x_tb09_params *)p->params; 
 
  params->c = 0; 
 
  switch(p->info->number){ 
  case XC_MGGA_X_BR89: 
    break; 
  case XC_MGGA_X_BJ06:   
    params->c = 1.0; 
    break; 
  case XC_MGGA_X_TB09: 
    /* the value of c should be passed by the calling code */ 
    break; 
  case XC_MGGA_X_RPP09: 
    params->c = 1.0; 
    break; 
  case XC_MGGA_X_B00: 
    break; 
  } 
} 
 
 
/* This code follows the inversion done in the PINY_MD package */ 
static double 
br_newt_raph(double a, double tol,  double * res, int *ierr) 
{ 
  int count; 
  double x, f; 
  static int max_iter = 50; 
 
   *ierr = 1; 
   if(a == 0.0) 
     return 0.0; 
    
   /* starting point */ 
   x = (a < 0.0) ? -1.0 : 1.0; 
 
   count = 0; 
   do { 
     double arg, eee, xm2, fp; 
 
     xm2 = x - 2.0; 
     arg = 2.0*x/3.0; 
     eee = exp(-arg)/a; 
 
     f  = x*eee - xm2; 
     fp = eee*(1.0 - 2.0/3.0*x) - 1.0; 
 
     x -= f/fp; 
     x  = fabs(x); 
 
     count ++; 
     *res = fabs(f); 
   } while((*res > tol) && (count < max_iter)); 
 
   if(count == max_iter) *ierr=0;  
   return x; 
} 
 
static double 
br_bisect(double a, double tol, int *ierr) {  
  int count;  
  double f, x, x1, x2;  
  static int max_iter = 500;  
 	  
  *ierr = 1;  
  if(a == 0.0)  
    return 0.0;  
		    
  /* starting interval */  
  if(a > 0.0) {  
    x1 = 2.0 + tol;  
    x2 = 1.0/a + 2.0; 
  }else{  
    x2 = 2.0 - tol;  
    x1 = 0.0;  
  }  
	 	  
  /* bisection */  
  count = 0;  
  do{  
    double arg, eee, xm2;  
    x   = 0.5*(x1 + x2);  
    xm2 = x - 2.0;  
    arg = 2.0*x/3.0;  
    eee = exp(-arg);  
    f   = x*eee - a*xm2;  
	 	  
    if(f > 0.0) x1 = x;  
    if(f < 0.0) x2 = x;  
	 	  
    count++;  
  }while((fabs(f) > tol)  && (count < max_iter));  
 	  
  if(count == max_iter) *ierr=0;   
  return x;  
}  
	 	  
double xc_mgga_x_br89_get_x(double Q) 
{ 
  double rhs, br_x, tol, res; 
  int ierr; 
 
  tol = 5e-12; 
 
  /* build right-hand side of the non-linear equation  
     Remember we use a different definition of tau */ 
  rhs = 2.0/3.0*pow(M_PI, 2.0/3.0)/Q; 
 
  br_x = br_newt_raph(rhs, tol, &res, &ierr); 
  if(ierr == 0){ 
    br_x = br_bisect(rhs, tol, &ierr); 
    if(ierr == 0){ 
      fprintf(stderr,  
	      "Warning: Convergence not reached in Becke-Roussel functional\n" 
	      "For rhs = %e (residual = %e)\n", rhs, res); 
    } 
  } 
 
  return br_x; 
} 
 
/* Eq. (22) */ 
void 
xc_mgga_b00_fw(int order, double t, double *fw, double *dfwdt) 
{ 
  double w, w2; 
   
  w = (K_FACTOR_C - t)/(K_FACTOR_C + t); 
  w2 = w*w; 
   
  *fw = w*(1.0 - 2.0*w2 + w2*w2); 
   
  if(order < 1) return; 
   
  *dfwdt = 1.0 - 6.0*w2 + 5.0*w2*w2; 
  *dfwdt *= -2.0*K_FACTOR_C/((K_FACTOR_C + t)*(K_FACTOR_C + t)); 
} 
 
 
static void  
func(const xc_func_type *pt, xc_mgga_work_x_t *r) 
{ 
  double Q, br_x, v_BR, dv_BRdbx, d2v_BRdbx2, dxdQ, d2xdQ2, ff, dffdx, d2ffdx2; 
  double cnst, c_TB09, c_HEG, exp1, exp2, gamma, fw, dfwdt, min_Q; 
 
  min_Q = 5.0e-13; 
 
  gamma = (pt->info->number == XC_MGGA_X_B00) ? 1.0 : br89_gamma; 
 
  Q = (r->u - 4.0*gamma*r->t + 0.5*gamma*r->x*r->x)/6.0; 
  if(fabs(Q) < min_Q) Q = (Q < 0) ? -min_Q : min_Q; 
 
  br_x = xc_mgga_x_br89_get_x(Q); 
 
  cnst = -2.0*CBRT(M_PI)/X_FACTOR_C; 
  exp1 = exp(br_x/3.0); 
  exp2 = exp(-br_x); 
 
  v_BR = (fabs(br_x) > pt->dens_threshold) ? 
    exp1*(1.0 - exp2*(1.0 + br_x/2.0))/br_x : 
    1.0/2.0 + br_x/6.0 - br_x*br_x/18.0; 
 
  v_BR *= cnst; 
 
  if(pt->info->number == XC_MGGA_X_BR89 || pt->info->number == XC_MGGA_X_B00){ 
    /* we have also to include the factor 1/2 from Eq. (9) */ 
    r->f = - v_BR / 2.0; 
 
    if(pt->info->number == XC_MGGA_X_B00){ 
      xc_mgga_b00_fw(r->order, r->t, &fw, &dfwdt); 
      r->f *= 1.0 + b00_at*fw; 
    } 
  }else{ /* XC_MGGA_X_BJ06 & XC_MGGA_X_TB09 */ 
    r->f = 0.0; 
  } 
 
  if(r->order < 1) return; 
 
  if(pt->info->number == XC_MGGA_X_BR89 || r->order > 1){ 
    dv_BRdbx = (fabs(br_x) > pt->dens_threshold) ? 
      (3.0 + br_x*(br_x + 2.0) + (br_x - 3.0)/exp2) / (3.0*exp1*exp1*br_x*br_x) : 
      1.0/6.0 - br_x/9.0; 
    dv_BRdbx *= cnst; 
     
    ff    = br_x*exp(-2.0/3.0*br_x)/(br_x - 2); 
    dffdx = ff*(-2.0/3.0 + 1.0/br_x - 1.0/(br_x - 2.0)); 
    dxdQ  = -ff/(Q*dffdx); 
  } 
 
  if(pt->info->number == XC_MGGA_X_BR89 || pt->info->number == XC_MGGA_X_B00){ 
    r->dfdx = -r->x*gamma*dv_BRdbx*dxdQ/12.0; 
    r->dfdt =   4.0*gamma*dv_BRdbx*dxdQ/12.0; 
    r->dfdu =            -dv_BRdbx*dxdQ/12.0; 
 
    if(pt->info->number == XC_MGGA_X_B00){ 
      r->dfdx *= 1.0 + b00_at*fw; 
      r->dfdt  = r->dfdt*(1.0 + b00_at*fw) - v_BR*b00_at*dfwdt/2.0; 
      r->dfdu *= 1.0 + b00_at*fw; 
    } 
  }else{ 
    assert(pt->params != NULL); 
    c_TB09 = ((mgga_x_tb09_params *) (pt->params))->c; 
 
    r->dfdrs = -c_TB09*v_BR; 
 
    c_HEG  = (3.0*c_TB09 - 2.0)*sqrt(5.0/12.0)/(X_FACTOR_C*M_PI); 
     
    if(pt->info->number == XC_MGGA_X_BJ06 || pt->info->number == XC_MGGA_X_TB09) 
      r->dfdrs -= c_HEG*sqrt(2.0*r->t); 
    else /* XC_MGGA_X_RPP09 */ 
      r->dfdrs -= c_HEG*sqrt(max(2.0*r->t - r->x*r->x/4.0, 0.0)); 
 
    r->dfdrs /= -r->rs; /* due to the definition of dfdrs */ 
  } 
 
  if(r->order < 2) return; 
   
  if(pt->info->number == XC_MGGA_X_BR89 || r->order > 2){ 
    d2v_BRdbx2 = (fabs(br_x) > pt->dens_threshold) ? 
      ((18.0 + (br_x - 6.0)*br_x)/exp2 - 2.0*(9.0 + br_x*(6.0 + br_x*(br_x + 2.0))))  
      / (9.0*exp1*exp1*br_x*br_x*br_x) : 
      -1.0/9.0; 
    d2v_BRdbx2 *= cnst; 
 
    d2ffdx2 = dffdx*dffdx/ff + ff*(-1.0/(br_x*br_x) + 1.0/((br_x - 2.0)*(br_x - 2.0))); 
    d2xdQ2 = -(2.0*dxdQ/Q + d2ffdx2*dxdQ*dxdQ/dffdx); 
  } 
 
  if(pt->info->number == XC_MGGA_X_BR89){ 
    double aux1 = d2v_BRdbx2*dxdQ*dxdQ + dv_BRdbx*d2xdQ2; 
 
    r->d2fdx2 = -(aux1*gamma*r->x*r->x/6.0 + dv_BRdbx*dxdQ)*gamma/12.0; 
    r->d2fdxt =  aux1*gamma*gamma*r->x/18.0; 
    r->d2fdxu = -aux1*gamma*r->x/72.0; 
    r->d2fdt2 = -aux1*2.0*gamma*gamma/9.0; 
    r->d2fdtu =  aux1*gamma/18.0; 
    r->d2fdu2 = -aux1/72.0; 
  }else{ 
     
  } 
 
} 
 
#include "work_mgga_x.c" 
 
const xc_func_info_type xc_func_info_mgga_x_br89 = { 
  XC_MGGA_X_BR89, 
  XC_EXCHANGE, 
  "Becke-Roussel 89", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Becke1989_3761, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1.0e-12, 
  0, NULL, NULL, 
  NULL, NULL, 
  NULL, NULL,        /* this is not an LDA                   */ 
  work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_mgga_x_bj06 = { 
  XC_MGGA_X_BJ06, 
  XC_EXCHANGE, 
  "Becke & Johnson 06", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Becke2006_221101, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_x_tb09_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
 
static const func_params_type ext_params[] = { 
  {1.0, "Value of the c parameter"}, 
}; 
 
static void  
set_ext_params(xc_func_type *p, const double *ext_params) 
{ 
  mgga_x_tb09_params *params; 
  double ff; 
 
  assert(p != NULL && p->params != NULL); 
  params = (mgga_x_tb09_params *) (p->params); 
 
  ff = (ext_params == NULL) ? p->info->ext_params[0].value : ext_params[0]; 
  params->c = ff; 
} 
 
const xc_func_info_type xc_func_info_mgga_x_tb09 = { 
  XC_MGGA_X_TB09, 
  XC_EXCHANGE, 
  "Tran & Blaha 09", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Tran2009_226401, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_VXC, 
  1.0e-23, 
  1, ext_params, set_ext_params, 
  mgga_x_tb09_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_mgga_x_rpp09 = { 
  XC_MGGA_X_RPP09, 
  XC_EXCHANGE, 
  "Rasanen, Pittalis & Proetto 09", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Rasanen2010_044112, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_x_tb09_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_mgga_x_b00 = { 
  XC_MGGA_X_B00, 
  XC_EXCHANGE, 
  "Becke 2000", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Becke2000_4020, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1.0e-23, 
  0, NULL, NULL, 
  mgga_x_tb09_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
