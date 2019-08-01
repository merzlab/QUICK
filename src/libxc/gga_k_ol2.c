/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_K_OL2          513 /* Ou-Yang and Levy v.2 */ 
#define XC_GGA_X_OL2          183 /* Exchange form based on Ou-Yang and Levy v.2 */ 
 
typedef struct{ 
  double aa, bb, cc; 
} gga_k_ol2_params; 
 
static void  
gga_k_ol2_init(xc_func_type *p) 
{ 
  gga_k_ol2_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_k_ol2_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_k_ol2_params); 
#endif 
  params = (gga_k_ol2_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_GGA_K_OL2: 
    params->aa = 1.0; 
    params->bb = 1.0/K_FACTOR_C; 
    params->cc = 0.00887/K_FACTOR_C; 
    break; 
  case XC_GGA_X_OL2: 
    params->aa = M_CBRT2*0.07064/X_FACTOR_C; 
    params->bb = M_CBRT2*0.07064/X_FACTOR_C; 
    params->cc = M_CBRT2*M_CBRT2*0.07064*34.0135/X_FACTOR_C; 
    break; 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_k_ol2.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_ol2 = { 
  XC_GGA_X_OL2, 
  XC_EXCHANGE, 
  "Exchange form based on Ou-Yang and Levy v.2", 
  XC_FAMILY_GGA, 
  {&xc_ref_Fuentealba1995_31, &xc_ref_OuYang1991_379, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  5e-26, 
  0, NULL, NULL, 
  gga_k_ol2_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_ol2 = { 
  XC_GGA_K_OL2, 
  XC_KINETIC, 
  "Ou-Yang and Levy v.2", 
  XC_FAMILY_GGA, 
  {&xc_ref_OuYang1991_379, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  5e-26, 
  0, NULL, NULL, 
  gga_k_ol2_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
