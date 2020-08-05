/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_HYB_MGGA_X_M11          297 /* M11 hybrid exchange functional from Minnesota        */ 
 
typedef struct{ 
  const double a[12], b[12]; 
} mgga_x_m11_params; 
 
static const mgga_x_m11_params par_m11 = { 
  { 
    -0.18399900e+00, -1.39046703e+01,  1.18206837e+01,  3.10098465e+01, -5.19625696e+01,  1.55750312e+01, 
    -6.94775730e+00, -1.58465014e+02, -1.48447565e+00,  5.51042124e+01, -1.34714184e+01,  0.00000000e+00 
  }, { 
     0.75599900e+00,  1.37137944e+01, -1.27998304e+01, -2.93428814e+01,  5.91075674e+01, -2.27604866e+01, 
    -1.02769340e+01,  1.64752731e+02,  1.85349258e+01, -5.56825639e+01,  7.47980859e+00,  0.00000000e+00 
  } 
}; 
 
 
static void 
mgga_x_m11_init(xc_func_type *p) 
{ 
  mgga_x_m11_params *params; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(mgga_x_m11_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(mgga_x_m11_params); 
#endif 
  params = (mgga_x_m11_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_HYB_MGGA_X_M11: 
    memcpy(params, &par_m11, sizeof(mgga_x_m11_params)); 
    p->cam_alpha = 1.0; 
    p->cam_beta  = -(1.0 - 0.428); 
    p->cam_omega = 0.25; 
    break; 
  default: 
    fprintf(stderr, "Internal error in mgga_x_m11\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/mgga_x_m11.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_m11 = { 
  XC_HYB_MGGA_X_M11, 
  XC_EXCHANGE, 
  "Minnesota M11 hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Peverati2011_2810, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-11, 
  0, NULL, NULL, 
  mgga_x_m11_init, NULL,  
  NULL, NULL, work_mgga_c, 
}; 
