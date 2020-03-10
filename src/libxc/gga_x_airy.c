/* 
 Copyright (C) 2008 Georg Madsen 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_AIRY  192 /* Constantin et al based on the Airy gas */ 
 
#ifndef DEVICE 
#include "maple2c/gga_x_airy.c" 
#endif 
 
#define func xc_gga_x_airy_enhance 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_airy = { 
  XC_GGA_X_AIRY, 
  XC_EXCHANGE, 
  "Constantin et al based on the Airy gas", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2009_035125, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  NULL, NULL,  
  NULL, work_gga_x, NULL 
}; 
