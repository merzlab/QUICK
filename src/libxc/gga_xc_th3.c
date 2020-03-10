/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_XC_TH3          156 /* Tozer and Handy v. 3 */ 
#define XC_GGA_XC_TH4          157 /* Tozer and Handy v. 4 */ 
 
typedef struct{ 
  double *omega; 
} gga_xc_th3_params; 
 
 
/* parameters for TH3 */ 
static double omega_TH3[] =  
  {-0.142542e+00, -0.783603e+00, -0.188875e+00, +0.426830e-01, -0.304953e+00, +0.430407e+00,  
   -0.997699e-01, +0.355789e-02, -0.344374e-01, +0.192108e-01, -0.230906e-02, +0.235189e-01,  
   -0.331157e-01, +0.121316e-01, +0.441190e+00, -0.227167e+01, +0.403051e+01, -0.228074e+01, 
   +0.360204e-01}; 
 
/* parameters for TH4 */ 
static double omega_TH4[] =  
  {+0.677353e-01, -0.106763e+01, -0.419018e-01, +0.226313e-01, -0.222478e+00, +0.283432e+00, 
   -0.165089e-01, -0.167204e-01, -0.332362e-01, +0.162254e-01, -0.984119e-03, +0.376713e-01, 
   -0.653419e-01, +0.222835e-01, +0.375782e+00, -0.190675e+01, +0.322494e+01, -0.168698e+01, 
   -0.235810e-01}; 
 
 
static void  
gga_xc_th3_init(xc_func_type *p) 
{ 
  gga_xc_th3_params *params; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_xc_th3_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_xc_th3_params); 
#endif 
  params = (gga_xc_th3_params *)p->params; 
 
  switch(p->info->number){ 
  case XC_GGA_XC_TH3: 
    params->omega = omega_TH3; 
    break; 
 
  case XC_GGA_XC_TH4: 
    params->omega = omega_TH4; 
    break; 
 
  default: 
    fprintf(stderr, "Internal error in gga_xc_th3\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_xc_th3.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_xc_th3 = { 
  XC_GGA_XC_TH3, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. 3", 
  XC_FAMILY_GGA, 
  {&xc_ref_Handy1998_707, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-18, 
  0, NULL, NULL, 
  gga_xc_th3_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_xc_th4 = { 
  XC_GGA_XC_TH4, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. 4", 
  XC_FAMILY_GGA, 
  {&xc_ref_Handy1998_707, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-15, 
  0, NULL, NULL, 
  gga_xc_th3_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
