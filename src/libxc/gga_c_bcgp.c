/*
 Copyright (C) 2017 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_C_BCGP          39 /* Burke, Cancio, Gould, and Pittalis                 */

#include "maple2c/gga_c_bcgp.c"

#define func maple2c_func
#include "work_gga_c.c"

const xc_func_info_type xc_func_info_gga_c_bcgp = {
  XC_GGA_C_BCGP,
  XC_CORRELATION,
  "Burke, Cancio, Gould, and Pittalis",
  XC_FAMILY_GGA,
  {&xc_ref_Burke2014_4834, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-25,
  0, NULL, NULL,
  NULL, NULL, 
  NULL, work_gga_c, NULL
};
