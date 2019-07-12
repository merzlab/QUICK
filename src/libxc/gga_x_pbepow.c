/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_X_PBEpow         539 /* PBE power */

#include "maple2c/gga_x_pbepow.c"

#define func xc_gga_x_pbepow_enhance
#include "work_gga_x.c"

const xc_func_info_type xc_func_info_gga_x_pbepow = {
  XC_GGA_X_PBEpow,
  XC_EXCHANGE,
  "PBE power",
  XC_FAMILY_GGA,
  {&xc_ref_Bremond2016_244102, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  5e-3, /* Gives NaN for densities as small as 1e-3 */
  0, NULL, NULL,
  NULL, NULL, 
  NULL, work_gga_x, NULL
};

