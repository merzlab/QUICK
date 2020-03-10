/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_C09X         158 /* C09x to be used with the VdW of Rutgers-Chalmers     */ 
 
#ifndef DEVICE 
#include "maple2c/gga_x_c09x.c" 
#endif 
 
#define func xc_gga_x_c09x_enhance 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_c09x = { 
  XC_GGA_X_C09X, 
  XC_EXCHANGE, 
  "C09x to be used with the VdW of Rutgers-Chalmers", 
  XC_FAMILY_GGA, 
  {&xc_ref_Cooper2010_161104, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  NULL, NULL, NULL, 
  work_gga_x, 
  NULL 
}; 
