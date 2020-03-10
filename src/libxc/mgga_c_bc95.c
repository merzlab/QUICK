/* 
 Copyright (C) 2008 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_C_BC95          240 /* Becke correlation 95 */ 
 
typedef struct{ 
  double css, copp; 
} mgga_c_bc95_params; 
 
 
static void  
mgga_c_bc95_init(xc_func_type *p) 
{ 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(mgga_c_bc95_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(mgga_c_bc95_params); 
#endif 
 
  xc_mgga_c_bc95_set_params(p, 0.038, 0.0031); 
} 
 
 
void  
xc_mgga_c_bc95_set_params(xc_func_type *p, double css, double copp) 
{ 
  mgga_c_bc95_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (mgga_c_bc95_params *) (p->params); 
 
  params->css  = css; 
  params->copp = copp; 
} 
 
 
#ifndef DEVICE 
#include "maple2c/mgga_c_bc95.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
const xc_func_info_type xc_func_info_mgga_c_bc95 = { 
  XC_MGGA_C_BC95, 
  XC_CORRELATION, 
  "Becke correlation 95", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Becke1996_1040, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_bc95_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
