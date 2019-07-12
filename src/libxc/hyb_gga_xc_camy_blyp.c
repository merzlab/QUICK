/*
 Copyright (C) 2013 Rolf Wuerdemann, M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define  XC_HYB_GGA_XC_CAMY_BLYP 455  /* BLYP with yukawa screening */

void
xc_hyb_gga_xc_camy_blyp_init(xc_func_type *p)
{
  static int   funcs_id  [3] = {XC_GGA_X_B88, XC_GGA_X_SFAT, XC_GGA_C_LYP};
  static double funcs_coef[3];

  /* N.B. The notation used in the original reference uses a different
     convention for alpha and beta.  In libxc, alpha is the weight for
     HF exchange, which in the original reference is alpha+beta.
  */
  static double alpha, beta, omega;
  
  alpha = 1.00;
  beta  =-0.80;
  omega = 0.44;	/* we use omega for gamma here, 'cause
		   both denote dampening parameters for
	       	   range related interactions */
  
  funcs_coef[0] = 1.0 - alpha;
  funcs_coef[1] =-beta;
  funcs_coef[2] = 1.0;

  xc_mix_init(p, 3, funcs_id, funcs_coef);
  xc_gga_x_sfat_set_params(p->func_aux[1], XC_GGA_X_B88, omega);

  p->cam_omega=omega;
  p->cam_alpha=alpha;
  p->cam_beta=beta;
}

const xc_func_info_type xc_func_info_hyb_gga_xc_camy_blyp = {
  XC_HYB_GGA_XC_CAMY_BLYP,
  XC_EXCHANGE_CORRELATION,
  "CAMY version of BLYP",
  XC_FAMILY_HYB_GGA,
  {&xc_ref_Akinaga2008_348, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HYB_CAMY | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  xc_hyb_gga_xc_camy_blyp_init,
  NULL, NULL, NULL, NULL
};

