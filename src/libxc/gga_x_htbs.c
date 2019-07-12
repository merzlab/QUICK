/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_X_HTBS         191 /* Haas, Tran, Blaha, and Schwarz  */

#include "maple2c/gga_x_htbs.c"

#define func maple2c_func
#include "work_gga_x.c"

const xc_func_info_type xc_func_info_gga_x_htbs = {
  XC_GGA_X_HTBS,
  XC_EXCHANGE,
  "Haas, Tran, Blaha, and Schwarz",
  XC_FAMILY_GGA,
  {&xc_ref_Haas2011_205117, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-16,
  0, NULL, NULL,
  NULL, NULL, 
  NULL, work_gga_x, NULL
};
