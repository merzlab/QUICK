/*
 Copyright (C) 2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_MGGA_X_GVT4          204 /* GVT4 from Van Voorhis and Scuseria */

#include "maple2c/mgga_x_gvt4.c"

#define func xc_mgga_x_gvt4_enhance
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_gvt4 = {
  XC_MGGA_X_GVT4,
  XC_EXCHANGE,
  "GVT4 (X part of VSXC)",
  XC_FAMILY_MGGA,
  {&xc_ref_VanVoorhis1998_400, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1.0e-23,
  0, NULL, NULL,
  NULL, NULL,
  NULL, NULL,        /* this is not an LDA                   */
  work_mgga_x,
};
