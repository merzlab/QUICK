/* 
 Copyright (C) 2017 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_C_ZPBEINT       61 /* spin-dependent gradient correction to PBEint       */ 
#define XC_GGA_C_ZPBESOL       63 /* spin-dependent gradient correction to PBEsol       */ 
 
typedef struct{ 
  double beta, alpha; 
} gga_c_zpbeint_params; 
 
static void  
gga_c_zpbeint_init(xc_func_type *p) 
{ 
  gga_c_zpbeint_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_c_zpbeint_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_c_zpbeint_params); 
#endif 
  params = (gga_c_zpbeint_params *) (p->params); 
  
  switch(p->info->number){ 
  case XC_GGA_C_ZPBEINT: 
    params->beta  = 0.052; 
    params->alpha = 2.4; 
    break; 
  case XC_GGA_C_ZPBESOL: 
    params->beta  = 0.046; 
    params->alpha = 4.8; 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_c_zpbeint\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_c_zpbeint.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_c_zpbeint = { 
  XC_GGA_C_ZPBEINT, 
  XC_CORRELATION, 
  "spin-dependent gradient correction to PBEint", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2011_233103, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_zpbeint_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_zpbesol = { 
  XC_GGA_C_ZPBESOL, 
  XC_CORRELATION, 
  "spin-dependent gradient correction to PBEsol", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2011_233103, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_zpbeint_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
