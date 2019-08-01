/* 
 Copyright (C) 2008 Lara Ferrigni, Georg Madsen, M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_X_M06_L         203 /* M06-L exchange functional from Minnesota          */ 
#define XC_HYB_MGGA_X_M06_HF    444 /* M06-HF hybrid exchange functional from Minnesota  */ 
#define XC_HYB_MGGA_X_M06       449 /* M06 hybrid exchange functional from Minnesota     */ 
#define XC_MGGA_X_REVM06_L      293 /* revised M06-L exchange functional from Minnesota  */ 
 
static const double a_m06l[12] = { 
  0.3987756, 0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615, 
  30.97273,  -23.18489, -56.73480, 21.60364, 34.21814, -9.049762 
}; 
static const double d_m06l[6] = {0.6012244, 0.004748822, -0.008635108, -0.000009308062, 0.00004482811, 0.0}; 
 
static const double a_m06hf[12] = { 
   1.179732e-01, -1.066708e+00, -1.462405e-01,  7.481848e+00,  3.776679e+00, -4.436118e+01,  
  -1.830962e+01,  1.003903e+02,  3.864360e+01, -9.806018e+01, -2.557716e+01,  3.590404e+01 
}; 
static const double d_m06hf[6] = {-1.179732e-01, -2.500000e-03, -1.180065e-02, 0.0, 0.0, 0.0}; 
 
static const double a_m06[12] = { 
   5.877943e-01, -1.371776e-01,  2.682367e-01, -2.515898e+00, -2.978892e+00,  8.710679e+00, 
   1.688195e+01, -4.489724e+00, -3.299983e+01, -1.449050e+01,  2.043747e+01,  1.256504e+01 
}; 
static const double d_m06[6] = {1.422057e-01, 7.370319e-04, -1.601373e-02, 0.0, 0.0, 0.0}; 
 
static const double a_revm06l[12] = { 
  1.423227252,  0.471820438, -0.167555701, -0.250154262,  0.062487588,  0.733501240, 
 -2.359736776, -1.436594372,  0.444643793,  1.529925054,  2.053941717, -0.036536031 
}; 
static const double d_revm06l[6] = {-0.423227252, 0.0, 0.003724234, 0.0, 0.0, 0.0}; 
 
typedef struct{ 
  const double *a, *d; 
} mgga_x_m06l_params; 
 
 
static void 
mgga_x_m06l_init(xc_func_type *p) 
{ 
  mgga_x_m06l_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(mgga_x_m06l_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(mgga_x_m06l_params); 
#endif 
  params = (mgga_x_m06l_params *)p->params; 
 
  switch(p->info->number){ 
  case XC_MGGA_X_M06_L: 
    params->a = a_m06l; 
    params->d = d_m06l; 
    break; 
  case XC_HYB_MGGA_X_M06_HF: 
    params->a = a_m06hf; 
    params->d = d_m06hf; 
    p->cam_alpha = 1.0; 
    break; 
  case XC_HYB_MGGA_X_M06: 
    params->a = a_m06; 
    params->d = d_m06; 
    p->cam_alpha = 0.27; 
    break; 
  case XC_MGGA_X_REVM06_L: 
    params->a = a_revm06l; 
    params->d = d_revm06l; 
    break; 
  default: 
    fprintf(stderr, "Internal error in mgga_x_m06l\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/mgga_x_m06l.c" 
#endif 
 
#define func xc_mgga_x_m06l_enhance 
#include "work_mgga_x.c" 
 
const xc_func_info_type xc_func_info_mgga_x_m06_l = { 
  XC_MGGA_X_M06_L, 
  XC_EXCHANGE, 
  "Minnesota M06-L exchange functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Zhao2006_194101, &xc_ref_Zhao2008_215, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1.0e-22, 
  0, NULL, NULL, 
  mgga_x_m06l_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_m06_hf = { 
  XC_HYB_MGGA_X_M06_HF, 
  XC_EXCHANGE, 
  "Minnesota M06-HF hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Zhao2006_13126, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1.0e-32, 
  0, NULL, NULL, 
  mgga_x_m06l_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_m06 = { 
  XC_HYB_MGGA_X_M06, 
  XC_EXCHANGE, 
  "Minnesota M06 hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Zhao2008_215, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1.0e-32, 
  0, NULL, NULL, 
  mgga_x_m06l_init, NULL,  
  NULL, NULL, work_mgga_x, 
}; 
 
const xc_func_info_type xc_func_info_mgga_x_revm06_l = { 
  XC_MGGA_X_REVM06_L, 
  XC_EXCHANGE, 
  "Minnesota revM06-L exchange functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Wang2017_8487, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  5.0e-13, 
  0, NULL, NULL, 
  mgga_x_m06l_init, NULL, 
  NULL, NULL, work_mgga_x, 
}; 
