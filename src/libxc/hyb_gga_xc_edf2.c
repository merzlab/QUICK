/*
 Copyright (C) 2006-2007 M.A.L. Marques and
                    2015 Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_HYB_GGA_XC_EDF2        476 /* Empirical functional from Lin, George and Gill */

static void
hyb_gga_xc_edf2_init(xc_func_type *p)
{
  static int   funcs_id  [6] = {XC_LDA_X, XC_GGA_X_B88, XC_GGA_X_B88, XC_LDA_C_VWN, XC_GGA_C_LYP, XC_GGA_C_LYP};
  static double funcs_coef[6] = {0.2811, 0.6227, -0.0551, 0.3029, 0.5998, -0.0053};

  xc_mix_init(p, 6, funcs_id, funcs_coef);  
  xc_gga_x_b88_set_params(p->func_aux[2], 0.0035, 6.0);
  xc_gga_c_lyp_set_params(p->func_aux[5], 0.055, 0.158, 0.25, 0.3505);
  p->cam_alpha = 0.1695;
}

const xc_func_info_type xc_func_info_hyb_gga_xc_edf2 = {
  XC_HYB_GGA_XC_EDF2,
  XC_EXCHANGE_CORRELATION,
  "EDF2",
  XC_FAMILY_HYB_GGA,
  {&xc_ref_Lin2004_365, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  hyb_gga_xc_edf2_init, 
  NULL, NULL, NULL, NULL
};
