/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_HYB_MGGA_X_M08_HX   295 /* M08-HX exchange functional from Minnesota  */ 
#define XC_HYB_MGGA_X_M08_SO   296 /* M08-SO exchange functional from Minnesota  */ 
 
typedef struct{ 
  const double a[12], b[12]; 
} mgga_x_m08_params; 
 
static const mgga_x_m08_params par_m08_hx = { 
  { 
     1.3340172e+00, -9.4751087e+00, -1.2541893e+01,  9.1369974e+00,  3.4717204e+01,  5.8831807e+01, 
     7.1369574e+01,  2.3312961e+01,  4.8314679e+00, -6.5044167e+00, -1.4058265e+01,  1.2880570e+01 
  }, { 
    -8.5631823e-01,  9.2810354e+00,  1.2260749e+01, -5.5189665e+00, -3.5534989e+01, -8.2049996e+01, 
    -6.8586558e+01,  3.6085694e+01, -9.3740983e+00, -5.9731688e+01,  1.6587868e+01,  1.3993203e+01 
  } 
}; 
 
static const mgga_x_m08_params par_m08_so = { 
  { 
    -3.4888428e-01, -5.8157416e+00,  3.7550810e+01,  6.3727406e+01, -5.3742313e+01, -9.8595529e+01, 
     1.6282216e+01,  1.7513468e+01, -6.7627553e+00,  1.1106658e+01,  1.5663545e+00,  8.7603470e+00 
  }, { 
     7.8098428e-01,  5.4538178e+00, -3.7853348e+01, -6.2295080e+01,  4.6713254e+01,  8.7321376e+01, 
     1.6053446e+01,  2.0126920e+01, -4.0343695e+01, -5.8577565e+01,  2.0890272e+01,  1.0946903e+01 
  } 
}; 
 
 
static void 
mgga_x_m08_init(xc_func_type *p) 
{ 
  mgga_x_m08_params *params; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(mgga_x_m08_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(mgga_x_m08_params); 
#endif 
  params = (mgga_x_m08_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_HYB_MGGA_X_M08_HX: 
    memcpy(params, &par_m08_hx, sizeof(mgga_x_m08_params)); 
    p->cam_alpha = 0.5223; 
  break; 
  case XC_HYB_MGGA_X_M08_SO: 
    memcpy(params, &par_m08_so, sizeof(mgga_x_m08_params)); 
    p->cam_alpha = 0.5679; 
  break; 
  default: 
    fprintf(stderr, "Internal error in mgga_x_m08\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/mgga_x_m08.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_x.c" 
 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_m08_hx = { 
  XC_HYB_MGGA_X_M08_HX, 
  XC_EXCHANGE, 
  "Minnesota M08-HX hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Zhao2008_1849, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-20, 
  0, NULL, NULL, 
  mgga_x_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_m08_so = { 
  XC_HYB_MGGA_X_M08_SO, 
  XC_EXCHANGE, 
  "Minnesota M08-SO hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Zhao2008_1849, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-20, 
  0, NULL, NULL, 
  mgga_x_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_x, 
}; 
