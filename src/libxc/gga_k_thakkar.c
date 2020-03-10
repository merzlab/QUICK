/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_K_THAKKAR      523 /* Thakkar 1992 */ 
 
#ifndef DEVICE 
#include "maple2c/gga_k_thakkar.c" 
#endif 
 
#define func xc_gga_k_thakkar_enhance 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_thakkar = { 
  XC_GGA_K_THAKKAR, 
  XC_KINETIC, 
  "Thakkar 1992", 
  XC_FAMILY_GGA, 
  {&xc_ref_Thakkar1992_6920, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  5e-26, 
  0, NULL, NULL, 
  NULL, NULL, 
  NULL, work_gga_k, NULL 
}; 
