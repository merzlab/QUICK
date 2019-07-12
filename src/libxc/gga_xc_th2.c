/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_XC_TH2          155 /* Tozer and Handy v. 2 */

#include "maple2c/gga_xc_th2.c"

#define func maple2c_func
#include "work_gga_c.c"

const xc_func_info_type xc_func_info_gga_xc_th2 = {
  XC_GGA_XC_TH2,
  XC_EXCHANGE_CORRELATION,
  "Tozer and Handy v. 2",
  XC_FAMILY_GGA,
  {&xc_ref_Tozer1998_3162, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-17,
  0, NULL, NULL,
  NULL, NULL, 
  NULL, work_gga_c, NULL
};
