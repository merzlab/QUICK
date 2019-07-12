/*
 Copyright (C) 2016 Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_X_CAP         270 /* Correct Asymptotic Potential */
#define XC_HYB_GGA_XC_CAP0   477 /* Correct Asymptotic Potential hybrid */

#include "maple2c/gga_x_cap.c"

#define func maple2c_func
#include "work_gga_x.c"

const xc_func_info_type xc_func_info_gga_x_cap = {
  XC_GGA_X_CAP,
  XC_EXCHANGE,
  "Correct Asymptotic Potential",
  XC_FAMILY_GGA,
  {&xc_ref_Carmona2015_054105, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-24,
  0, NULL, NULL,
  NULL, NULL, NULL,
  work_gga_x,
  NULL
};

void
xc_hyb_gga_xc_cap0_init(xc_func_type *p)
{
  static int   funcs_id  [2] = {XC_GGA_X_CAP, XC_GGA_C_PBE};
  static double funcs_coef[2] = {1.0, 1.0};
  /* Can't init this above */
  static const double a0 = 1.0/4.0;
  funcs_coef[0]=1.0-a0;
  
  xc_mix_init(p, 2, funcs_id, funcs_coef);
  /* C functional is PBE C with β = (3/4)β PBE */
  xc_gga_c_pbe_set_params(p->func_aux[1],0.75*0.06672455060314922);
  
  p->cam_alpha = a0;
}

const xc_func_info_type xc_func_info_hyb_gga_xc_cap0 = {
  XC_HYB_GGA_XC_CAP0,
  XC_EXCHANGE_CORRELATION,
  "Correct Asymptotic Potential hybrid",
  XC_FAMILY_HYB_GGA,
  {&xc_ref_Carmona2016_120, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  xc_hyb_gga_xc_cap0_init,
  NULL, NULL, NULL, NULL
};
