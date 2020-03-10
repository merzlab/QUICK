/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_C_TPSS          231 /* Tao, Perdew, Staroverov & Scuseria correlation */ 
 
typedef struct{ 
  double beta, d; 
  double C0_c[4]; 
} mgga_c_tpss_params; 
 
 
static void  
mgga_c_tpss_init(xc_func_type *p) 
{ 
 
  assert(p != NULL && p->params == NULL); 
  p->params = malloc(sizeof(mgga_c_tpss_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(mgga_c_tpss_params); 
#endif 
 
  switch(p->info->number){ 
  case XC_MGGA_C_TPSS: 
    xc_mgga_c_tpss_set_params(p, 0.06672455060314922, 2.8, 0.53, 0.87, 0.50, 2.26); 
    break; 
  default: 
    fprintf(stderr, "Internal error in mgga_c_tpss\n"); 
    exit(1); 
  } 
} 
 
void 
xc_mgga_c_tpss_set_params 
     (xc_func_type *p, double beta, double d, double C0_0, double C0_1, double C0_2, double C0_3) 
{ 
  mgga_c_tpss_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (mgga_c_tpss_params *) (p->params); 
 
  params->beta    = beta; 
  params->d       = d; 
  params->C0_c[0] = C0_0; 
  params->C0_c[1] = C0_1; 
  params->C0_c[2] = C0_2; 
  params->C0_c[3] = C0_3; 
} 
 
#ifndef DEVICE 
#include "maple2c/mgga_c_tpss.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
 
const xc_func_info_type xc_func_info_mgga_c_tpss = { 
  XC_MGGA_C_TPSS, 
  XC_CORRELATION, 
  "Tao, Perdew, Staroverov & Scuseria", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Tao2003_146401, &xc_ref_Perdew2004_6898, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, /* densities smaller than 1e-26 give NaNs */ 
  0, NULL, NULL, 
  mgga_c_tpss_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
