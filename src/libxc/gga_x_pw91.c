/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_PW91         109 /* Perdew & Wang 91 */ 
#define XC_GGA_X_MPW91        119 /* Modified form of PW91 by Adamo & Barone */ 
#define XC_GGA_K_LC94         521 /* Lembarki & Chermette */ 
 
typedef struct{ 
  double a, b, c, d, f, alpha, expo; 
} gga_x_pw91_params; 
 
 
static void  
gga_x_pw91_init(xc_func_type *p) 
{ 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_pw91_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_x_pw91_params); 
#endif 
 
  switch(p->info->number){ 
  case XC_GGA_X_PW91: 
    /* b_PW91 ~ 0.0042 */ 
    xc_gga_x_pw91_set_params(p, 0.19645, 7.7956, 0.2743, -0.1508, 0.004, 100.0, 4.0); 
    break; 
  case XC_GGA_X_MPW91: 
    /* 
      === from nwchem source (xc_xmpw91.F) === 
      C. Adamo confirmed that there is a typo in the JCP paper 
      b_mPW91 is 0.00426 instead of 0.0046 
       
      also the power seems to be 3.72 and not 3.73 
    */ 
    xc_gga_x_pw91_set_params2(p, 0.00426, 100.0, 3.72); 
    break; 
  case XC_GGA_K_LC94: 
    xc_gga_x_pw91_set_params(p, 0.093907, 76.320, 0.26608, -0.0809615, 0.000057767, 100.0, 4.0); 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_pw91\n"); 
    exit(1); 
  }  
} 
 
void  
xc_gga_x_pw91_set_params(xc_func_type *p, double a, double b, double c, double d, double f, double alpha, double expo) 
{ 
  gga_x_pw91_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_pw91_params *) (p->params); 
 
  params->a     = a; 
  params->b     = b; 
  params->c     = c; 
  params->d     = d; 
  params->f     = f; 
  params->alpha = alpha; 
  params->expo  = expo; 
} 
 
void  
xc_gga_x_pw91_set_params2(xc_func_type *p, double bt, double alpha, double expo) 
{ 
  double beta; 
  double a, b, c, d, f; 
 
  beta =  5.0*pow(36.0*M_PI,-5.0/3.0); 
  a    =  6.0*bt/X2S; 
  b    =  1.0/X2S; 
  c    =  bt/(X_FACTOR_C*X2S*X2S); 
  d    = -(bt - beta)/(X_FACTOR_C*X2S*X2S); 
  f    = 1.0e-6/(X_FACTOR_C*pow(X2S, expo)); 
 
  xc_gga_x_pw91_set_params(p, a, b, c, d, f, alpha, expo); 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_pw91.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_pw91 = { 
  XC_GGA_X_PW91, 
  XC_EXCHANGE, 
  "Perdew & Wang 91", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew1991, &xc_ref_Perdew1992_6671, &xc_ref_Perdew1992_6671_err, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  gga_x_pw91_init, 
  NULL, NULL, 
  work_gga_x, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_mpw91 = { 
  XC_GGA_X_MPW91, 
  XC_EXCHANGE, 
  "mPW91 of Adamo & Barone", 
  XC_FAMILY_GGA, 
  {&xc_ref_Adamo1998_664, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-31, 
  0, NULL, NULL, 
  gga_x_pw91_init, 
  NULL, NULL, 
  work_gga_x, 
  NULL 
}; 
 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_lc94 = { 
  XC_GGA_K_LC94, 
  XC_KINETIC, 
  "Lembarki & Chermette", 
  XC_FAMILY_GGA, 
  {&xc_ref_Lembarki1994_5328, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-21, 
  0, NULL, NULL, 
  gga_x_pw91_init, 
  NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
