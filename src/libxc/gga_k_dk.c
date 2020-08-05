/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_K_DK          516 /* DePristo and Kress                    */ 
#define XC_GGA_K_PERDEW      517 /* Perdew                                */ 
#define XC_GGA_K_VSK         518 /* Vitos, Skriver, and Kollar            */ 
#define XC_GGA_K_VJKS        519 /* Vitos, Johansson, Kollar, and Skriver */ 
#define XC_GGA_K_ERNZERHOF   520 /* Ernzerhof */ 
 
typedef struct{ 
  double aa[5], bb[5]; 
} gga_k_dk_params; 
 
static void  
gga_k_dk_init(xc_func_type *p) 
{ 
  int i; 
  double ff, *aa, *bb; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_k_dk_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_k_dk_params); 
#endif 
 
  /* shortcuts for a and b */ 
  aa  = ((gga_k_dk_params *) (p->params))->aa; 
  bb  = ((gga_k_dk_params *) (p->params))->bb; 
 
  /* initialize parameters to zero */ 
  for(i=0; i<5; i++){ 
    aa[i] = 0.0; 
    bb[i] = 0.0; 
  } 
 
  switch(p->info->number){ 
  case XC_GGA_K_DK: 
    ff = 5.0*X2S*X2S/27.0; /* = t2/t0 = 1.0/(72.0*K_FACTOR_C) */ 
 
    bb[0] =  1.0; 
    bb[1] = -0.05   *ff; 
    bb[2] =  9.99802*(ff*ff); 
    bb[3] =  2.96085*(ff*ff*ff); 
 
    aa[0] =   1.0; 
    aa[1] =   0.95   *ff; 
    aa[2] =  14.28111*(ff*ff); 
    aa[3] = -19.57962*(ff*ff*ff); 
    aa[4] =   9.0*bb[3]*ff; 
 
    break; 
 
  case XC_GGA_K_PERDEW: 
    ff = X2S*X2S; 
 
    bb[0] =  1.0; 
    bb[1] = 88.3960*ff; 
    bb[2] = 16.3683*(ff*ff); 
 
    aa[0] =   1.0; 
    aa[1] =  88.2108*ff; 
 
    break; 
 
  case XC_GGA_K_VSK: 
    ff = 5.0*X2S*X2S/27.0; /* = t2/t0 = 1.0/(72.0*K_FACTOR_C) */ 
 
    bb[0] =  1.0; 
    bb[1] = -0.05     *ff; 
    bb[2] =  0.396    *(ff*ff); 
 
    aa[0] =  1.0; 
    aa[1] =  0.95     *ff; 
    aa[3] =  9.0*bb[2]*ff; 
 
    break; 
 
  case XC_GGA_K_VJKS: 
    ff = X2S*X2S; 
 
    bb[0] =  1.0; 
    bb[1] =  0.6511 *ff; 
    bb[2] =  0.0431 *(ff*ff); 
 
    aa[0] =  1.0; 
    aa[1] =  0.8944 *ff; 
    aa[3] = -bb[2]  *ff; 
 
    break; 
 
  case XC_GGA_K_ERNZERHOF: 
    ff = X2S*X2S; 
 
    bb[0] =  135.0; 
    bb[1] =    3.0*ff; 
 
    aa[0] =  135.0; 
    aa[1] =   28.0*ff; 
    aa[2] =    5.0*(ff*ff); 
 
    break; 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_k_dk.c" 
#endif 
 
#define func maple2c_func 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_dk = { 
  XC_GGA_K_DK, 
  XC_KINETIC, 
  "DePristo and Kress", 
  XC_FAMILY_GGA, 
  {&xc_ref_DePristo1987_438, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_dk_init, 
  NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_perdew = { 
  XC_GGA_K_PERDEW, 
  XC_KINETIC, 
  "Perdew", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew1992_79, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_k_dk_init, 
  NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_vsk = { 
  XC_GGA_K_VSK, 
  XC_KINETIC, 
  "Vitos, Skriver, and Kollar", 
  XC_FAMILY_GGA, 
  {&xc_ref_Vitos1998_12611, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_dk_init, 
  NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_vjks = { 
  XC_GGA_K_VJKS, 
  XC_KINETIC, 
  "Vitos, Johansson, Kollar, and Skriver", 
  XC_FAMILY_GGA, 
  {&xc_ref_Vitos2000_052511, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_dk_init, 
  NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_ernzerhof = { 
  XC_GGA_K_ERNZERHOF, 
  XC_KINETIC, 
  "Ernzerhof", 
  XC_FAMILY_GGA, 
  {&xc_ref_Ernzerhof2000_59, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_dk_init, 
  NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
