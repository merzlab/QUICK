/*
 Copyright (C) 2008 Georg Madsen

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_MGGA_C_KCIS         562 /* Krieger, Chen, Iafrate, and Savin */
#define XC_HYB_MGGA_XC_B0KCIS  563 /* Hybrid based on KCIS */

#include "maple2c/mgga_c_kcis.c"

#define func maple2c_func
#include "work_mgga_c.c"

const xc_func_info_type xc_func_info_mgga_c_kcis = {
  XC_MGGA_C_KCIS,
  XC_CORRELATION,
  "Krieger, Chen, Iafrate, and Savin",
  XC_FAMILY_MGGA,
  {&xc_ref_Rey1998_581, &xc_ref_Krieger1999_463, &xc_ref_Krieger2001_48, &xc_ref_Toulouse2002_10465, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_DEVELOPMENT,
  1e-24,
  0, NULL, NULL,
  NULL, NULL, 
  NULL, NULL, work_mgga_c
};

/*************************************************************/
void
xc_hyb_mgga_xc_b0kcis_init(xc_func_type *p)
{
  static int   funcs_id  [2] = {XC_GGA_X_B88, XC_MGGA_C_KCIS};
  static double funcs_coef[2] = {1.0 - 0.25, 1.0};

  xc_mix_init(p, 2, funcs_id, funcs_coef);
  p->cam_alpha = 0.25;
}

const xc_func_info_type xc_func_info_hyb_mgga_xc_b0kcis = {
  XC_HYB_MGGA_XC_B0KCIS,
  XC_EXCHANGE_CORRELATION,
  "Hybrid based on KCIS",
  XC_FAMILY_HYB_MGGA,
  {&xc_ref_Toulouse2002_10465, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_DEVELOPMENT,
  1e-32,
  0, NULL, NULL,
  xc_hyb_mgga_xc_b0kcis_init, NULL, 
  NULL, NULL, work_mgga_c
};
