/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_DK87_R1      111 /* dePristo & Kress 87 (version R1)               */ 
#define XC_GGA_X_DK87_R2      112 /* dePristo & Kress 87 (version R2)               */ 
 
typedef struct { 
  double a1, b1, alpha; 
} gga_x_dk87_params; 
 
static const gga_x_dk87_params par_dk87_r1 = { 
  0.861504, 0.044286, 1.0 
}; 
 
static const gga_x_dk87_params par_dk87_r2 = { 
  0.861213, 0.042076, 0.98 
}; 
 
static void  
gga_x_dk87_init(xc_func_type *p) 
{ 
  gga_x_dk87_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_dk87_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(gga_x_dk87_params); 
#endif 
  params = (gga_x_dk87_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_GGA_X_DK87_R1:  
    memcpy(params, &par_dk87_r1, sizeof(gga_x_dk87_params)); 
    break; 
  case XC_GGA_X_DK87_R2: 
    memcpy(params, &par_dk87_r2, sizeof(gga_x_dk87_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_dk87\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_dk87.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_dk87_r1 = { 
  XC_GGA_X_DK87_R1, 
  XC_EXCHANGE, 
  "dePristo & Kress 87 version R1", 
  XC_FAMILY_GGA, 
  {&xc_ref_DePristo1987_1425, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  gga_x_dk87_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_dk87_r2 = { 
  XC_GGA_X_DK87_R2, 
  XC_EXCHANGE, 
  "dePristo & Kress 87 version R2", 
  XC_FAMILY_GGA, 
  {&xc_ref_DePristo1987_1425, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  gga_x_dk87_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
