/*
 Copyright (C) 2017 Miguel Marques, Mario Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_X_GX          575 /* GX functional of Loos */

#include "maple2c/mgga_x_gx.c"

#define func maple2c_func
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_gx = {
  XC_MGGA_X_GX,
  XC_EXCHANGE,
  "GX functional of Loos",
  XC_FAMILY_MGGA,
  {&xc_ref_Loos2017_114108, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1e-20,
  0, NULL, NULL,
  NULL, NULL,
  NULL, NULL, work_mgga_x,
};
