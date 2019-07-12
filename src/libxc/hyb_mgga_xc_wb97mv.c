/*
 Copyright (C) 2015 Narbe Mardirossian and Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_HYB_MGGA_XC_WB97M_V   531 /* Mardirossian and Head-Gordon */

static void
hyb_mgga_xc_wb97mv_init(xc_func_type *p)
{
  p->cam_omega =  0.3;
  p->cam_alpha =  1.0;
  p->cam_beta  = -(1.0 - 0.15);
  p->nlc_b = 6.0;
  p->nlc_C = 0.01;
}

#include "maple2c/hyb_mgga_xc_wb97mv.c"

#define func maple2c_func
#include "work_mgga_c.c"

const xc_func_info_type xc_func_info_hyb_mgga_xc_wb97m_v = {
  XC_HYB_MGGA_XC_WB97M_V,
  XC_EXCHANGE_CORRELATION,
  "wB97M-V exchange-correlation functional",
  XC_FAMILY_HYB_MGGA,
  {&xc_ref_Mardirossian2016_214110, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_VV10 | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-13,
  0, NULL, NULL,
  hyb_mgga_xc_wb97mv_init,
  NULL, NULL, NULL,
  work_mgga_c,
};
