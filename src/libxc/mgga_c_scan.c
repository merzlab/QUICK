/* 
 Copyright (C) 2016 Susi Lehtola 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_C_SCAN          267 /* SCAN correlation */ 
#define XC_MGGA_C_SCAN_RVV10    292 /* SCAN correlation + rVV10 correlation */ 
#define XC_MGGA_C_SCAN_VV10     584 /* SCAN correlation +  VV10 correlation */ 
 
#ifndef DEVICE 
#include "maple2c/mgga_c_scan.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
const xc_func_info_type xc_func_info_mgga_c_scan = { 
  XC_MGGA_C_SCAN, 
  XC_CORRELATION, 
  "SCAN correlation of Sun, Ruzsinszky, and Perdew", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Sun2015_036402, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-26, 
  0, NULL, NULL, 
  NULL, NULL,  
  NULL, NULL, work_mgga_c, 
}; 
 
 
static void 
mgga_c_scan_rvv10_init(xc_func_type *p) 
{ 
  static int   funcs_id  [1] = {XC_MGGA_C_SCAN}; 
  static double funcs_coef[1] = {1.0}; 
 
  xc_mix_init(p, 1, funcs_id, funcs_coef); 
 
  p->nlc_b = 15.7; 
  p->nlc_C = 0.0093; 
} 
 
const xc_func_info_type xc_func_info_mgga_c_scan_rvv10 = { 
  XC_MGGA_C_SCAN_RVV10, 
  XC_CORRELATION, 
  "SCAN + rVV10 correlation", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Peng2016_041005, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_VV10 | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-32, 
  0, NULL, NULL, 
  mgga_c_scan_rvv10_init, 
  NULL, NULL, NULL, NULL 
}; 
 
static void 
mgga_c_scan_vv10_init(xc_func_type *p) 
{ 
  static int   funcs_id  [1] = {XC_MGGA_C_SCAN}; 
  static double funcs_coef[1] = {1.0}; 
 
  xc_mix_init(p, 1, funcs_id, funcs_coef); 
 
  p->nlc_b = 14.0; 
  p->nlc_C = 0.0093; 
} 
 
const xc_func_info_type xc_func_info_mgga_c_scan_vv10 = { 
  XC_MGGA_C_SCAN_VV10, 
  XC_CORRELATION, 
  "SCAN + VV10 correlation", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Brandenburg2016_115144, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_VV10 | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-32, 
  0, NULL, NULL, 
  mgga_c_scan_vv10_init, 
  NULL, NULL, NULL, NULL 
}; 
