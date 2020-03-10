/* 
 Copyright (C) 2013 Rolf Wuerdemann, M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_HYB_GGA_XC_CAMY_B3LYP        470 /* B3LYP with Yukawa screening */ 
 
void 
xc_hyb_gga_xc_camy_b3lyp_init(xc_func_type *p) 
{ 
  static double ac = 0.81; 
  static int   funcs_id  [4] = {XC_GGA_X_B88, XC_GGA_X_SFAT, XC_LDA_C_VWN, XC_GGA_C_LYP}; 
  static double funcs_coef[4]; 
 
  /* Need temp variables since cam_ parameters are initialized in mix_init */ 
  static double omega, alpha, beta; 
 
  /* N.B. The notation used in the original reference uses a different 
     convention for alpha and beta.  In libxc, alpha is the weight for 
     HF exchange, which in the original reference is alpha+beta. 
  */ 
  omega = 0.34; 
  alpha = 0.65; 
  beta  = -0.46; 
 
  funcs_coef[0] = 1.0 - alpha; 
  funcs_coef[1] = -beta; 
  funcs_coef[2] = 1.0 - ac; 
  funcs_coef[3] = ac; 
   
  xc_mix_init(p, 4, funcs_id, funcs_coef); 
  xc_gga_x_sfat_set_params(p->func_aux[1], XC_GGA_X_B88, omega); 
   
  p->cam_omega = omega; 
  p->cam_alpha = alpha; 
  p->cam_beta  = beta; 
} 
 
const xc_func_info_type xc_func_info_hyb_gga_xc_camy_b3lyp = { 
  XC_HYB_GGA_XC_CAMY_B3LYP, 
  XC_EXCHANGE_CORRELATION, 
  "CAMY version of B3LYP", 
  XC_FAMILY_HYB_GGA, 
  {&xc_ref_Seth2012_901, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HYB_CAMY | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-32, 
  0, NULL, NULL, 
  xc_hyb_gga_xc_camy_b3lyp_init, 
  NULL, NULL, NULL, NULL 
}; 
 
