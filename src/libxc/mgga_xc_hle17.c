/* 
 Copyright (C) 2017 Susi Lehtola 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_MGGA_XC_HLE17      288  /* high local exchange 2017   */ 
 
static void 
mgga_xc_hle17_init(xc_func_type *p) 
{ 
  static int   funcs_id  [2] = {XC_MGGA_X_TPSS, XC_MGGA_C_TPSS}; 
  static double funcs_coef[2] = {1.25, 0.5}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
} 
 
const xc_func_info_type xc_func_info_mgga_xc_hle17 = { 
  XC_MGGA_XC_HLE17, 
  XC_EXCHANGE_CORRELATION, 
  "high local exchange 2017", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Verma2017_7144, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_xc_hle17_init, 
  NULL, NULL, NULL, NULL 
}; 
