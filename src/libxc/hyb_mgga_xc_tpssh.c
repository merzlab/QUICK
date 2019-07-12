/*
 Copyright (C) 2014 Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_HYB_MGGA_XC_TPSSH       457 /*    TPSS hybrid */
#define XC_HYB_MGGA_XC_REVTPSSH    458 /* revTPSS hybrid */

static void
hyb_mgga_xc_tpssh_init(xc_func_type *p)
{
  static int   funcs_id  [2] = {XC_MGGA_X_TPSS, XC_MGGA_C_TPSS};
  static double funcs_coef[2] = {0.9, 1.0};

  xc_mix_init(p, 2, funcs_id, funcs_coef);
  p->cam_alpha = 0.10;
}


const xc_func_info_type xc_func_info_hyb_mgga_xc_tpssh = {
  XC_HYB_MGGA_XC_TPSSH,
  XC_EXCHANGE_CORRELATION,
  "TPSSh",
  XC_FAMILY_HYB_MGGA,
  {&xc_ref_Staroverov2003_12129, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  hyb_mgga_xc_tpssh_init,
  NULL, NULL, NULL, NULL /* this is taken care of by the generic routine */
};


static void
hyb_mgga_xc_revtpssh_init(xc_func_type *p)
{
  static int   funcs_id  [2] = {XC_MGGA_X_REVTPSS, XC_MGGA_C_REVTPSS};
  static double funcs_coef[2] = {0.9, 1.0};

  xc_mix_init(p, 2, funcs_id, funcs_coef);
  p->cam_alpha = 0.10;
}


const xc_func_info_type xc_func_info_hyb_mgga_xc_revtpssh = {
  XC_HYB_MGGA_XC_REVTPSSH,
  XC_EXCHANGE_CORRELATION,
  "revTPSSh",
  XC_FAMILY_HYB_MGGA,
  {&xc_ref_Csonka2010_3688, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  hyb_mgga_xc_revtpssh_init,
  NULL, NULL, NULL, NULL /* this is taken care of by the generic routine */
};
