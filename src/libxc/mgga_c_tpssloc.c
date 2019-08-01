/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_C_TPSSLOC       247 /* Semilocal dynamical correlation */ 
 
#ifndef DEVICE 
#include "maple2c/mgga_c_tpssloc.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
const xc_func_info_type xc_func_info_mgga_c_tpssloc = { 
  XC_MGGA_C_TPSSLOC, 
  XC_CORRELATION, 
  "Semilocal dynamical correlation", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Constantin2012_035130, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-9, /* densities smaller than 1e-26 give NaNs */ 
  0, NULL, NULL, 
  NULL, NULL, 
  NULL, NULL, work_mgga_c 
}; 
 
