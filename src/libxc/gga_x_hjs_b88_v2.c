/* 
 Copyright (C) 2017 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_HJS_B88_V2   46 /* HJS screened exchange corrected B88 version */ 
 
typedef struct{ 
  double omega; 
} gga_x_hjs_b88_v2_params; 
 
static void 
gga_x_hjs_init(xc_func_type *p) 
{ 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_x_hjs_b88_v2_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_x_hjs_b88_v2_params); 
#endif 
 
  /* we take 0.11 as the default for hjs_b88_v2 */ 
  xc_gga_x_hjs_b88_v2_set_params(p, 0.11); 
} 
 
void  
xc_gga_x_hjs_b88_v2_set_params(xc_func_type *p, double omega) 
{ 
  gga_x_hjs_b88_v2_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_hjs_b88_v2_params *) (p->params); 
 
  params->omega = omega; 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_hjs_b88_v2.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_x_hjs_b88_v2 = { 
  XC_GGA_X_HJS_B88_V2, 
  XC_EXCHANGE, 
  "HJS screened exchange B88 corrected version", 
  XC_FAMILY_GGA, 
  {&xc_ref_Weintraub2009_754, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-6, /* densities smaller than 1e-6 yield NaNs */ 
  0, NULL, NULL, 
  gga_x_hjs_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
