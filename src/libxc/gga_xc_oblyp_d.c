/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_XC_OBLYP_D       67  /* oBLYP-D functional of Goerigk and Grimme  */ 
#define XC_GGA_XC_OPWLYP_D      66  /* oPWLYP-D functional of Goerigk and Grimme */ 
#define XC_GGA_XC_OPBE_D        65  /* oPBE_D functional of Goerigk and Grimme   */ 
 
static void 
gga_xc_oblyp_d_init(xc_func_type *p) 
{ 
  static int   funcs_id  [4] = {XC_GGA_X_B88, XC_GGA_C_LYP}; 
  static double funcs_coef[4] = {1.0, 1.0}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
  xc_gga_x_b88_set_params(p->func_aux[0], 0.00401, 6.0); 
  xc_gga_c_lyp_set_params(p->func_aux[1], 0.05047, 0.140, 0.2196, 0.363); 
} 
 
const xc_func_info_type xc_func_info_gga_xc_oblyp_d = { 
  XC_GGA_XC_OBLYP_D, 
  XC_EXCHANGE_CORRELATION, 
  "oBLYP-D functional of Goerigk and Grimme", 
  XC_FAMILY_GGA, 
  {&xc_ref_Goerigk2010_107, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_oblyp_d_init, 
  NULL, NULL, NULL, NULL 
}; 
 
 
static void 
gga_xc_opwlyp_d_init(xc_func_type *p) 
{ 
  static int   funcs_id  [4] = {XC_GGA_X_PW91, XC_GGA_C_LYP}; 
  static double funcs_coef[4] = {1.0, 1.0}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
  xc_gga_x_pw91_set_params2(p->func_aux[0], 0.00402, 0.8894/(X2S*X2S), 0.79); 
  xc_gga_c_lyp_set_params(p->func_aux[1], 0.04960, 0.144, 0.2262, 0.346); 
} 
 
const xc_func_info_type xc_func_info_gga_xc_opwlyp_d = { 
  XC_GGA_XC_OPWLYP_D, 
  XC_EXCHANGE_CORRELATION, 
  "oPWLYP-D functional of Goerigk and Grimme", 
  XC_FAMILY_GGA, 
  {&xc_ref_Goerigk2010_107, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_opwlyp_d_init, 
  NULL, NULL, NULL, NULL 
}; 
 
 
static void 
gga_xc_opbe_d_init(xc_func_type *p) 
{ 
  static int   funcs_id  [4] = {XC_GGA_X_PBE, XC_GGA_C_PBE}; 
  static double funcs_coef[4] = {1.0, 1.0}; 
 
  xc_mix_init(p, 2, funcs_id, funcs_coef); 
  xc_gga_x_pbe_set_params(p->func_aux[0], 1.2010, 0.21198); 
  xc_gga_c_pbe_set_params(p->func_aux[1], 0.04636); 
} 
 
const xc_func_info_type xc_func_info_gga_xc_opbe_d = { 
  XC_GGA_XC_OPBE_D, 
  XC_EXCHANGE_CORRELATION, 
  "oPBE-D functional of Goerigk and Grimme", 
  XC_FAMILY_GGA, 
  {&xc_ref_Goerigk2010_107, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_xc_opbe_d_init, 
  NULL, NULL, NULL, NULL 
}; 
 
