/*
 Copyright (C) 2014 Jess Wellendorff, M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_X_MBEEFVDW       250 /* mBEEF-vdW exchange */

#include "maple2c/mgga_x_mbeefvdw.c"

#define func maple2c_func
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_mbeefvdw = {
  XC_MGGA_X_MBEEFVDW,
  XC_EXCHANGE,
  "mBEEF-vdW exchange",
  XC_FAMILY_MGGA,
  {&xc_ref_Lundgaard2016_235162, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1e-10,
  0, NULL, NULL,
  NULL, NULL, 
  NULL, NULL, work_mgga_x,
};
