/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_MPBE         122 /* Adamo & Barone modification to PBE             */ 
#define XC_GGA_K_PBE3         595 /* Three parameter PBE-like expansion             */ 
#define XC_GGA_K_PBE4         596 /* Four  parameter PBE-like expansion             */ 
 
typedef struct{ 
  double a; 
  double c1, c2, c3; 
} gga_x_mpbe_params; 
 
 
static void  
gga_x_mpbe_init(xc_func_type *p) 
{ 
  gga_x_mpbe_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_mpbe_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_x_mpbe_params); 
#endif 
  params = (gga_x_mpbe_params *) (p->params); 
  
  switch(p->info->number){ 
  case XC_GGA_X_MPBE: 
    params->a  =  0.157; 
    params->c1 =  0.21951; 
    params->c2 = -0.015; 
    params->c3 =  0.0; 
    break; 
  case XC_GGA_K_PBE3: 
    params->a  =  4.1355; 
    params->c1 = -3.7425; 
    params->c2 = 50.258; 
    params->c3 =  0.0; 
    break; 
  case XC_GGA_K_PBE4: 
    params->a  =   1.7107; 
    params->c1 =  -7.2333; 
    params->c2 =  61.645; 
    params->c3 = -93.683; 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_mpbe\n"); 
    exit(1); 
  } 
} 
 
 
#ifndef DEVICE 
#include "maple2c/gga_x_mpbe.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_mpbe = { 
  XC_GGA_X_MPBE, 
  XC_EXCHANGE, 
  "Adamo & Barone modification to PBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Adamo2002_5933, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-21, 
  0, NULL, NULL, 
  gga_x_mpbe_init, NULL, 
  NULL, work_gga_x, NULL 
}; 
 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_pbe3 = { 
  XC_GGA_K_PBE3, 
  XC_KINETIC, 
  "Three parameter PBE-like expansion", 
  XC_FAMILY_GGA, 
  {&xc_ref_Karasiev2006_111, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-21, 
  0, NULL, NULL, 
  gga_x_mpbe_init, NULL, 
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_pbe4 = { 
  XC_GGA_K_PBE4, 
  XC_KINETIC, 
  "Four parameter PBE-like expansion", 
  XC_FAMILY_GGA, 
  {&xc_ref_Karasiev2006_111, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-21, 
  0, NULL, NULL, 
  gga_x_mpbe_init, NULL, 
  NULL, work_gga_k, NULL 
}; 
