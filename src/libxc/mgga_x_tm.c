/*
 Copyright (C) 2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_MGGA_X_TM          540 /* Tao and Mo 2016 */

#include "maple2c/mgga_x_tm.c"

#define func xc_mgga_x_tm_enhance
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_tm = {
  XC_MGGA_X_TM,
  XC_EXCHANGE,
  "Tao and Mo 2016",
  XC_FAMILY_MGGA,
  {&xc_ref_Tao2016_073001, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1.0e-23,
  0, NULL, NULL,
  NULL, NULL,
  NULL, NULL, work_mgga_x,
};
