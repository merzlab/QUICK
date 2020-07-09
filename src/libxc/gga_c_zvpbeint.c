/* 
 Copyright (C) 2017 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_C_ZVPBEINT       557 /* another spin-dependent correction to PBEint       */ 
#define XC_GGA_C_ZVPBESOL       558 /* another spin-dependent correction to PBEsol       */ 
 
typedef struct{ 
  double beta, alpha, omega; 
} gga_c_zvpbeint_params; 
 
static void  
gga_c_zvpbeint_init(xc_func_type *p) 
{ 
  gga_c_zvpbeint_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_c_zvpbeint_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_c_zvpbeint_params); 
#endif 
  params = (gga_c_zvpbeint_params *) (p->params); 
  
  switch(p->info->number){ 
  case XC_GGA_C_ZVPBEINT: 
    params->beta  = 0.052; 
    params->alpha = 1.0; 
    params->omega = 4.5; 
    break; 
  case XC_GGA_C_ZVPBESOL: 
    params->beta  = 0.046; 
    params->alpha = 1.8; 
    params->omega = 4.5; 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_c_zvpbeint\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_c_zvpbeint.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_c_zvpbeint = { 
  XC_GGA_C_ZVPBEINT, 
  XC_CORRELATION, 
  "another spin-dependent correction to PBEint", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2012_194105, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-10, 
  0, NULL, NULL, 
  gga_c_zvpbeint_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_zvpbesol = { 
  XC_GGA_C_ZVPBESOL, 
  XC_CORRELATION, 
  "another spin-dependent correction to PBEsol", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2012_194105, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-10, 
  0, NULL, NULL, 
  gga_c_zvpbeint_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
