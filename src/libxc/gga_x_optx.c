/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_OPTX         110 /* Handy & Cohen OPTX 01                          */ 
 
typedef struct{ 
  double a, b, gamma; 
} gga_x_optx_params; 
 
 
static void  
gga_x_optx_init(xc_func_type *p) 
{ 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_optx_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_x_optx_params); 
#endif 
 
  xc_gga_x_optx_set_params(p, 1.05151, 1.43169/X_FACTOR_C, 0.006); 
} 
 
 
void  
xc_gga_x_optx_set_params(xc_func_type *p, double a, double b, double gamma) 
{ 
  gga_x_optx_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_optx_params *) (p->params); 
 
  params->a     = a; 
  params->b     = b; 
  params->gamma = gamma; 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_optx.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_optx = { 
  XC_GGA_X_OPTX, 
  XC_EXCHANGE, 
  "Handy & Cohen OPTX 01", 
  XC_FAMILY_GGA, 
  {&xc_ref_Handy2001_403, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-22, 
  0, NULL, NULL, 
  gga_x_optx_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
