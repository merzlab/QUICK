/* 
 Copyright (C) 2013 Rolf Wuerdemann, M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define  XC_HYB_GGA_XC_LCY_PBE 467  /* PBE with yukawa screening */ 
 
void 
xc_hyb_gga_xc_lcy_pbe_init(xc_func_type *p) 
{ 
  static int   funcs_id  [2] = {XC_GGA_X_SFAT, XC_GGA_C_PBE}; 
  static double funcs_coef[2]; 
  static double gamma; 
 
  gamma = 0.75; /* Use omega for gamma */ 
 
  funcs_coef[0] = 1.0; 
  funcs_coef[1] = 1.0; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
 
  xc_gga_x_sfat_set_params(p->func_aux[0], XC_GGA_X_PBE, gamma); 
  p->cam_omega = gamma; 
  p->cam_alpha = 1.0; 
  p->cam_beta  = -1.0; 
} 
 
const xc_func_info_type xc_func_info_hyb_gga_xc_lcy_pbe = { 
  XC_HYB_GGA_XC_LCY_PBE, 
  XC_EXCHANGE_CORRELATION, 
  "LCY version of PBE", 
  XC_FAMILY_HYB_GGA, 
  {&xc_ref_Seth2012_901, &xc_ref_Seth2013_2286, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HYB_LCY | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-32, 
  0, NULL, NULL, 
  xc_hyb_gga_xc_lcy_pbe_init, 
  NULL, NULL, NULL, NULL 
}; 
