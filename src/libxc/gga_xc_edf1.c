/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_XC_EDF1        165 /* Empirical functionals from Adamson, Gill, and Pople */ 
#define XC_GGA_X_OPTPBE_VDW   141 /* PBE reparametrization for vdW */ 
#define XC_GGA_XC_MOHLYP      194 /* Functional for organometallic chemistry */ 
#define XC_GGA_XC_MOHLYP2     195 /* Functional for barrier heights */ 
#define XC_GGA_X_SOGGA        150 /* Second-order generalized gradient approximation */ 
 
static void 
gga_xc_edf1_init(xc_func_type *p) 
{ 
  static int   funcs_id  [4] = {XC_LDA_X, XC_GGA_X_B88, XC_GGA_X_B88, XC_GGA_C_LYP}; 
  static double funcs_coef[4] = {1.030952 - 10.4017 + 8.44793, 10.4017, -8.44793, 1.0}; 
 
  xc_mix_init(p, 4, funcs_id, funcs_coef);   
 
  xc_gga_x_b88_set_params(p->func_aux[1], 0.0035, 6.0); 
  xc_gga_x_b88_set_params(p->func_aux[2], 0.0042, 6.0); 
  xc_gga_c_lyp_set_params(p->func_aux[3], 0.055, 0.158, 0.25, 0.3505); 
} 
 
const xc_func_info_type xc_func_info_gga_xc_edf1 = { 
  XC_GGA_XC_EDF1, 
  XC_EXCHANGE_CORRELATION, 
  "EDF1", 
  XC_FAMILY_GGA, 
  {&xc_ref_Adamson1998_6, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_edf1_init,  
  NULL, NULL, NULL, NULL 
}; 
 
 
static void 
gga_x_optpbe_vdw_init(xc_func_type *p) 
{ 
  static int   funcs_id  [2] = {XC_GGA_X_PBE, XC_GGA_X_RPBE}; 
  static double funcs_coef[2] = {1.0 - 0.054732, 0.054732}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef);   
 
  xc_gga_x_pbe_set_params (p->func_aux[0], 1.04804, 0.175519); 
  xc_gga_x_rpbe_set_params(p->func_aux[1], 1.04804, 0.175519); 
} 
 
const xc_func_info_type xc_func_info_gga_x_optpbe_vdw = { 
  XC_GGA_X_OPTPBE_VDW, 
  XC_EXCHANGE, 
  "Reparametrized PBE for vdW", 
  XC_FAMILY_GGA, 
  {&xc_ref_Klimes2010_022201, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_optpbe_vdw_init,  
  NULL, NULL, NULL, NULL 
}; 
 
 
static void 
gga_xc_mohlyp_init(xc_func_type *p) 
{ 
  static int   funcs_id  [3] = {XC_GGA_X_OPTX, XC_LDA_C_VWN, XC_GGA_C_LYP}; 
  static double funcs_coef[3] = {1.0, 0.5, 0.5}; 
 
  xc_mix_init(p, 3, funcs_id, funcs_coef); 
 
  xc_gga_x_optx_set_params(p->func_aux[0], 1.0, 1.292/X_FACTOR_C, 0.006); 
} 
 
const xc_func_info_type xc_func_info_gga_xc_mohlyp = { 
  XC_GGA_XC_MOHLYP, 
  XC_EXCHANGE_CORRELATION, 
  "Functional for organometallic chemistry", 
  XC_FAMILY_GGA, 
  {&xc_ref_Schultz2005_11127, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_mohlyp_init, 
  NULL, NULL, NULL, NULL 
}; 
 
static void 
gga_xc_mohlyp2_init(xc_func_type *p) 
{ 
  static int   funcs_id  [2] = {XC_GGA_X_OPTX, XC_GGA_C_LYP}; 
  static double funcs_coef[2] = {1.0, 0.5}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
 
  xc_gga_x_optx_set_params(p->func_aux[0], 1.05151, 1.8497564/X_FACTOR_C, 0.006); 
} 
 
const xc_func_info_type xc_func_info_gga_xc_mohlyp2 = { 
  XC_GGA_XC_MOHLYP2, 
  XC_EXCHANGE_CORRELATION, 
  "Functional for barrier heights", 
  XC_FAMILY_GGA, 
  {&xc_ref_Zheng2009_808, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_mohlyp2_init, 
  NULL, NULL, NULL, NULL 
}; 
 
static void 
gga_x_sogga_init(xc_func_type *p) 
{ 
  static int   funcs_id  [2] = {XC_GGA_X_PBE, XC_GGA_X_RPBE}; 
  static double funcs_coef[2] = {0.5, 0.5}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
 
  xc_gga_x_pbe_set_params (p->func_aux[0], 0.552, 10.0/81.0); 
  xc_gga_x_rpbe_set_params(p->func_aux[1], 0.552, 10.0/81.0); 
} 
 
const xc_func_info_type xc_func_info_gga_x_sogga = { 
  XC_GGA_X_SOGGA, 
  XC_EXCHANGE, 
  "Second-order generalized gradient approximation", 
  XC_FAMILY_GGA, 
  {&xc_ref_Zhao2008_184109, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_sogga_init, 
  NULL, NULL, NULL, NULL 
}; 
 
