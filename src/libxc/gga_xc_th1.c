/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_XC_TH_FL        196 /* Tozer and Handy v. FL  */ 
#define XC_GGA_XC_TH_FC        197 /* Tozer and Handy v. FC  */ 
#define XC_GGA_XC_TH_FCFO      198 /* Tozer and Handy v. FCFO */ 
#define XC_GGA_XC_TH_FCO       199 /* Tozer and Handy v. FCO */ 
#define XC_GGA_XC_TH1          154 /* Tozer and Handy v. 1 */ 
 
typedef struct{ 
  const double *omega; 
} gga_xc_th1_params; 
 
static const double omega_TH_FL[] =  
  {-0.106141e01, +0.898203e00, -0.134439e01, +0.302369e00, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0}; 
 
static const double omega_TH_FC[] =  
  {-0.864448e+00, +0.565130e+00, -0.127306e+01, +0.309681e+00, -0.287658e+00, +0.588767e+00, 
   -0.252700e+00, +0.223563e-01, +0.140131e-01, -0.826608e-01, +0.556080e-01, -0.936227e-02, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0}; 
 
static const double omega_TH_FCFO[] =  
  {-0.864448e+00, +0.565130e+00, -0.127306e+01, +0.309681e+00, -0.287658e+00, +0.588767e+00, 
   -0.252700e+00, +0.223563e-01, +0.140131e-01, -0.826608e-01, +0.556080e-01, -0.936227e-02, 
   -0.677146e-02, +0.515199e-01, -0.874213e-01, +0.423827e-01, +0.431940e+00, -0.691153e+00, 
   -0.637866e+00, +0.107565e+01, 0.0}; 
 
static const double omega_TH_FCO[] =  
  {-0.962998e+00, +0.860233e+00, -0.154092e+01, +0.381602e+00, -0.210208e+00, +0.391496e+00, 
   -0.107660e+00, -0.105324e-01, +0.837384e-02, -0.617859e-01, +0.383072e-01, -0.526905e-02, 
   -0.381514e-02, +0.321541e-01, -0.568280e-01, +0.288585e-01, +0.368326e+00, -0.328799e+00, 
   -0.122595e+01, +0.136412e+01, 0.0}; 
 
static const double omega_TH1[] =  
  {-0.728255e+00, +0.331699e+00, -0.102946e+01, +0.235703e+00, -0.876221e-01, +0.140854e+00, 
   +0.336982e-01, -0.353615e-01, +0.497930e-02, -0.645900e-01, +0.461795e-01, -0.757191e-02, 
   -0.242717e-02, +0.428140e-01, -0.744891e-01, +0.386577e-01, -0.352519e+00, +0.219805e+01 
   -0.372927e+01, +0.194441e+01, +0.128877e+00}; 
 
 
static void  
gga_xc_th1_init(xc_func_type *p) 
{ 
  gga_xc_th1_params *params; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_xc_th1_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(gga_xc_th1_params); 
#endif 
  params = (gga_xc_th1_params *)p->params; 
 
  switch(p->info->number){ 
  case XC_GGA_XC_TH_FL: 
    params->omega = omega_TH_FL; 
    break; 
 
  case XC_GGA_XC_TH_FC: 
    params->omega = omega_TH_FC; 
    break; 
 
  case XC_GGA_XC_TH_FCFO: 
    params->omega = omega_TH_FCFO; 
    break; 
 
  case XC_GGA_XC_TH_FCO: 
    params->omega = omega_TH_FCO;  
    break; 
 
  case XC_GGA_XC_TH1: 
    params->omega = omega_TH1; 
    break; 
 
  default: 
    fprintf(stderr, "Internal error in gga_xc_th1\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_xc_th1.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_xc_th_fl = { 
  XC_GGA_XC_TH_FL, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. FL", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tozer1997_183, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_th1_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_xc_th_fc = { 
  XC_GGA_XC_TH_FC, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. FC", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tozer1997_183, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_th1_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_xc_th_fcfo = { 
  XC_GGA_XC_TH_FCFO, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. FCFO", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tozer1997_183, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_th1_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_xc_th_fco = { 
  XC_GGA_XC_TH_FCO, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. FCO", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tozer1997_183, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_th1_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_xc_th1 = { 
  XC_GGA_XC_TH1, 
  XC_EXCHANGE_CORRELATION, 
  "Tozer and Handy v. 1", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tozer1998_2545, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_th1_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
