/*
 Copyright (C) 2006-2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

/* Local tau approximation */

#define XC_MGGA_X_LTA          201 /* Local tau approximation of Ernzerhof & Scuseria */

#include "maple2c/mgga_x_lta.c"

#define func maple2c_func
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_lta = {
  XC_MGGA_X_LTA,
  XC_EXCHANGE,
  "Local tau approximation",
  XC_FAMILY_MGGA,
  {&xc_ref_Ernzerhof1999_911, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1.0e-23,
  0, NULL, NULL,
  NULL, NULL,
  NULL, NULL,        /* this is not an LDA                   */
  work_mgga_x,
};
