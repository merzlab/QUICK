/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_HYB_MGGA_XC_B98         598 /* Becke 98 */

static void
hyb_mgga_xc_b98_init(xc_func_type *p)
{
  p->cam_alpha = 0.1985;
}

#include "maple2c/mgga_xc_b98.c"

#define func maple2c_func
#include "work_mgga_c.c"

const xc_func_info_type xc_func_info_hyb_mgga_xc_b98 = {
  XC_HYB_MGGA_XC_B98,
  XC_EXCHANGE_CORRELATION,
  "Becke 98",
  XC_FAMILY_HYB_MGGA,
  {&xc_ref_Becke1998_2092, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-23,
  0, NULL, NULL,
  hyb_mgga_xc_b98_init, NULL,
  NULL, NULL, work_mgga_c,
};

