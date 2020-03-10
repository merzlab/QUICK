/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_AK13  56 /* Armiento & Kuemmel 2013 */ 
 
static const double B1 =  1.74959015598863046792081721182; /* 3*muGE/5 + 8 pi/15 */ 
static const double B2 = -1.62613336586517367779736042170; /* muGE - B1 */ 
 
double xc_gga_ak13_get_asymptotic (double homo) 
{ 
  double Qx, aa, aa2, factor; 
 
  Qx = sqrt(2.0)*B1/(3.0*CBRT(3.0*M_PI*M_PI)); 
 
  aa  = X_FACTOR_C*Qx; 
  aa2 = aa*aa; 
 
  factor = (homo < 0.0) ? -1.0 : 1.0; 
     
  return (aa2/2.0)*(1.0 + factor*sqrt(1.0 - 4.0*homo/aa2)); 
} 
 
 
#ifndef DEVICE 
#include "maple2c/gga_x_ak13.c" 
#endif 
 
#define func xc_gga_x_ak13_enhance 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_ak13 = { 
  XC_GGA_X_AK13, 
  XC_EXCHANGE, 
  "Armiento & Kuemmel 2013", 
  XC_FAMILY_GGA, 
  {&xc_ref_Armiento2013_036402, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  NULL, NULL, NULL, 
  work_gga_x, 
  NULL 
}; 
 
