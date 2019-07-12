/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_K_MEYER        57 /* Meyer,  Wang, and Young */

#include "maple2c/gga_k_meyer.c"

#define func maple2c_func
#define XC_KINETIC_FUNCTIONAL
#include "work_gga_x.c"

const xc_func_info_type xc_func_info_gga_k_meyer = {
  XC_GGA_K_MEYER,
  XC_KINETIC,
  "Meyer,  Wang, and Young",
  XC_FAMILY_GGA,
  {&xc_ref_Meyer1976_898, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  5e-14,
  0, NULL, NULL,
  NULL,
  NULL, NULL,
  work_gga_k,
  NULL
};
