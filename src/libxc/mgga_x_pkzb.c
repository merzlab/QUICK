/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_X_PKZB          213 /* Perdew, Kurth, Zupan, and Blaha */ 
 
#ifndef DEVICE 
#include "maple2c/mgga_x_pkzb.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_x.c" 
 
 
const xc_func_info_type xc_func_info_mgga_x_pkzb = { 
  XC_MGGA_X_PKZB, 
  XC_EXCHANGE, 
  "Perdew, Kurth, Zupan, and Blaha", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Perdew1999_2544, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-23, 
  0, NULL, NULL, 
  NULL, 
  NULL, NULL, NULL, 
  work_mgga_x, 
}; 
