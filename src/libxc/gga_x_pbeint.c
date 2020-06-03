/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_PBEINT        60 /* PBE for hybrid interfaces                      */ 
#define XC_GGA_K_APBEINT       54 /* interpolated version of APBE                   */ 
#define XC_GGA_K_REVAPBEINT    53 /* interpolated version of REVAPBE                */ 
 
 
typedef struct{ 
  double kappa, alpha, muPBE, muGE; 
} gga_x_pbeint_params; 
 
 
static void  
gga_x_pbe_init(xc_func_type *p) 
{ 
  gga_x_pbeint_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_pbeint_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_x_pbeint_params); 
#endif 
  params = (gga_x_pbeint_params *) (p->params); 
  
  switch(p->info->number){ 
  case XC_GGA_X_PBEINT: 
    xc_gga_x_pbeint_set_params(p, 0.8040, 0.197, 0.2195149727645171, MU_GE); 
    break; 
  case XC_GGA_K_APBEINT: 
    xc_gga_x_pbeint_set_params(p, 0.8040, 5.0/3.0, 0.23899, 5.0/27.0); 
    break; 
  case XC_GGA_K_REVAPBEINT: 
    xc_gga_x_pbeint_set_params(p, 1.245, 5.0/3.0, 0.23899, 5.0/27.0); 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_pbeint\n"); 
    exit(1); 
  } 
} 
 
 
void  
xc_gga_x_pbeint_set_params(xc_func_type *p, double kappa, double alpha, double muPBE, double muGE) 
{ 
  gga_x_pbeint_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_pbeint_params *) (p->params); 
 
  params->kappa = kappa; 
  params->alpha = alpha; 
  params->muPBE = muPBE; 
  params->muGE  = muGE; 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_pbeint.c" 
#endif 
 
#define func xc_gga_x_pbeint_enhance 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_pbeint = { 
  XC_GGA_X_PBEINT, 
  XC_EXCHANGE, 
  "PBE for hybrid interfaces", 
  XC_FAMILY_GGA, 
  {&xc_ref_Fabiano2010_113104, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_apbeint = { 
  XC_GGA_K_APBEINT, 
  XC_KINETIC, 
  "interpolated version of APBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Laricchia2011_2439, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_revapbeint = { 
  XC_GGA_K_REVAPBEINT, 
  XC_KINETIC, 
  "interpolated version of revAPBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Laricchia2011_2439, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
