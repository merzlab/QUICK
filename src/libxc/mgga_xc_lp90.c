/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_XC_LP90          564 /* Lee & Parr, Eq. (56) */

#include "maple2c/mgga_xc_lp90.c"

#define func maple2c_func
#include "work_mgga_c.c"

const xc_func_info_type xc_func_info_mgga_xc_lp90 = {
  XC_MGGA_XC_LP90,
  XC_EXCHANGE_CORRELATION,
  "Lee & Parr, Eq. (56)",
  XC_FAMILY_MGGA,
  {&xc_ref_Lee1990_193, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-24,
  0, NULL, NULL,
  NULL, NULL, 
  NULL, NULL, work_mgga_c,
};
