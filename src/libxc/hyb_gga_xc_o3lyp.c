/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_HYB_GGA_XC_O3LYP   404 /* hybrid using the optx functional */ 
#define XC_HYB_GGA_XC_X3LYP   411 /* hybrid by Xu and Goddard */ 
 
 
/*************************************************************/ 
static void 
gga_xc_o3lyp_init(xc_func_type *p) 
{ 
  const double a = 0.1161, b = 0.9262, c = 0.8133, CC = 0.81, a1 = 1.05151; 
  static int funcs_id[4] = {XC_LDA_X, XC_GGA_X_OPTX, XC_LDA_C_VWN, XC_GGA_C_LYP}; 
  double funcs_coef[4]; 
 
  /* 
    Delta OPTX is defined as the GGA correction in OPTX, and O3LYP has 
    been defined in the paper not in terms of OPTX but in terms of 
    Delta OPTX. 
 
    To get the functional in terms of full LDA and full OPTX, we need 
    to remove the double counting of the LDA contribution. 
   */ 
  funcs_coef[0] = b - a1*c; 
  funcs_coef[1] = c; 
  funcs_coef[2] = 1.0 - CC; 
  funcs_coef[3] = CC; 
 
  xc_mix_init(p, 4, funcs_id, funcs_coef); 
  p->cam_alpha = a; 
} 
 
const xc_func_info_type xc_func_info_hyb_gga_xc_o3lyp = { 
  XC_HYB_GGA_XC_O3LYP, 
  XC_EXCHANGE_CORRELATION, 
  "O3LYP", 
  XC_FAMILY_HYB_GGA, 
  {&xc_ref_Cohen2001_607, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_o3lyp_init, 
  NULL, NULL, NULL, NULL 
}; 
 
 
/*************************************************************/ 
static void 
gga_xc_x3lyp_init(xc_func_type *p) 
{ 
  const double a1=0.765, a2=0.235; 
  const double a0=0.218, ax=0.709, ac=0.871; 
 
  static int funcs_id[5] = {XC_LDA_X, XC_GGA_X_B88, XC_GGA_X_PW91, XC_LDA_C_VWN_RPA, XC_GGA_C_LYP}; 
  double funcs_coef[5]; 
 
  funcs_coef[0] = 1.0 - a0 - ax*(a1 + a2); 
  funcs_coef[1] = ax*a1; 
  funcs_coef[2] = ax*a2; 
  funcs_coef[3] = 1.0 - ac; 
  funcs_coef[4] = ac; 
 
  xc_mix_init(p, 5, funcs_id, funcs_coef); 
  p->cam_alpha = a0; 
} 
 
const xc_func_info_type xc_func_info_hyb_gga_xc_x3lyp = { 
  XC_HYB_GGA_XC_X3LYP, 
  XC_EXCHANGE_CORRELATION, 
  "X3LYP", 
  XC_FAMILY_HYB_GGA, 
  {&xc_ref_Xu2004_2673, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_x3lyp_init, 
  NULL, NULL, NULL, NULL 
}; 
