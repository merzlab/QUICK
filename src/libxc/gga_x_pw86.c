/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_PW86         108 /* Perdew & Wang 86 */ 
#define XC_GGA_X_RPW86        144 /* refitted Perdew & Wang 86 */ 
#define XC_GGA_K_FR_PW86      515 /* Fuentealba & Reyes (PW86 version) */ 
 
typedef struct{ 
  double aa, bb, cc; 
} gga_x_pw86_params; 
 
 
static const gga_x_pw86_params par_pw86 = { 
  1.296, 14.0, 0.2 
}; 
 
static const gga_x_pw86_params par_rpw86 = { 
  15.0*0.1234, 17.33, 0.163, 
}; 
 
static const gga_x_pw86_params par_fr_pw86 = { 
  2.208, 9.27, 0.2 
}; 
 
static void  
gga_x_pw86_init(xc_func_type *p) 
{ 
  gga_x_pw86_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_pw86_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_x_pw86_params); 
#endif 
  params = (gga_x_pw86_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_GGA_X_PW86:  
    memcpy(params, &par_pw86, sizeof(gga_x_pw86_params)); 
    break; 
  case XC_GGA_X_RPW86: 
    memcpy(params, &par_rpw86, sizeof(gga_x_pw86_params)); 
    break; 
  case XC_GGA_K_FR_PW86: 
    memcpy(params, &par_fr_pw86, sizeof(gga_x_pw86_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_pw86\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_pw86.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_pw86 = { 
  XC_GGA_X_PW86, 
  XC_EXCHANGE, 
  "Perdew & Wang 86", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew1986_8800, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  gga_x_pw86_init, NULL, NULL, 
  work_gga_x, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_rpw86 = { 
  XC_GGA_X_RPW86, 
  XC_EXCHANGE, 
  "Refitted Perdew & Wang 86", 
  XC_FAMILY_GGA, 
  {&xc_ref_Murray2009_2754, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  gga_x_pw86_init, NULL, NULL, 
  work_gga_x, 
  NULL 
}; 
 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_fr_pw86 = { 
  XC_GGA_K_FR_PW86, 
  XC_KINETIC, 
  "Fuentealba & Reyes (PW86 version)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Fuentealba1995_31, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pw86_init, NULL, NULL, 
  work_gga_k, 
  NULL 
}; 
