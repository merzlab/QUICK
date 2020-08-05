/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_VMT84_PBE        69 /* VMT{8,4} with constraint satisfaction with mu = mu_PBE  */ 
#define XC_GGA_X_VMT84_GE         68 /* VMT{8,4} with constraint satisfaction with mu = mu_GE  */ 
 
typedef struct{ 
  double mu; 
  double alpha; 
} gga_x_vmt84_params; 
 
static void  
gga_x_vmt84_init(xc_func_type *p) 
{ 
  gga_x_vmt84_params *params; 
 
  assert(p != NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_vmt84_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_x_vmt84_params); 
#endif 
  params = (gga_x_vmt84_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_GGA_X_VMT84_PBE: 
    params->mu    = 0.2195149727645171; 
    params->alpha = 0.000074; 
    break; 
  case XC_GGA_X_VMT84_GE: 
    params->mu = 10.0/81.0; 
    params->alpha = 0.000023; 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_vmt84\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_vmt84.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_vmt84_pbe = { 
  XC_GGA_X_VMT84_PBE, 
  XC_EXCHANGE, 
  "VMT{8,4} with constraint satisfaction with mu = mu_PBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Vela2012_144115, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_vmt84_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_vmt84_ge = { 
  XC_GGA_X_VMT84_GE, 
  XC_EXCHANGE, 
  "VMT{8,4} with constraint satisfaction with mu = mu_GE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Vela2012_144115, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_vmt84_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
