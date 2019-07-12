/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_HYB_GGA_XC_CAM_B3LYP        433 /* CAM version of B3LYP */
#define XC_HYB_GGA_XC_TUNED_CAM_B3LYP  434 /* CAM version of B3LYP tuned for excitations*/
#define XC_HYB_GGA_XC_CAM_QTP_01       482 /* CAM-QTP(01): CAM-B3LYP retuned using ionization potentials of water */

void
xc_hyb_gga_xc_cam_b3lyp_init(xc_func_type *p)
{
  double ac;
  static int   funcs_id  [4] = {XC_GGA_X_B88, XC_GGA_X_ITYH, XC_LDA_C_VWN, XC_GGA_C_LYP};
  static double funcs_coef[4];

  /* Need temp variables since cam_ parameters are initialized in mix_init */
  static double omega, alpha, beta;
  
  switch(p->info->number){
  case XC_HYB_GGA_XC_CAM_B3LYP:
    /* N.B. The notation used in Yanai et al uses a different
       convention for alpha and beta.  In libxc, alpha is the weight
       for HF exchange, which in Yanai et al is alpha+beta, so:

       alpha_libxc = alpha_Yanai + beta_Yanai
       beta_libxc  = - beta_Yanai
     */
    omega = 0.33;
    alpha = 0.65;
    beta  =-0.46;
    ac = 0.81;
    break;
  case XC_HYB_GGA_XC_TUNED_CAM_B3LYP:
    /* The same note applies here. */
    omega = 0.150;
    alpha = 1.0000;
    beta  =-0.9201;
    ac = 0.81;
    break;
  case XC_HYB_GGA_XC_CAM_QTP_01:
    /* The same note applies here. */
    omega = 0.31;
    alpha = 1.00;
    beta  =-0.77;
    ac = 0.80;
    break;
  default:
    fprintf(stderr,"Internal error in hyb_gga_xc_cam_b3lyp_init.\n");
    exit(1);
  }

  funcs_coef[0] = 1.0 - alpha;
  funcs_coef[1] = -beta;
  funcs_coef[2] = 1.0 - ac;
  funcs_coef[3] = ac;

  xc_mix_init(p, 4, funcs_id, funcs_coef);
  xc_gga_x_ityh_set_params(p->func_aux[1], XC_GGA_X_B88, omega);

  p->cam_omega = omega;
  p->cam_alpha = alpha;
  p->cam_beta  = beta;
}

const xc_func_info_type xc_func_info_hyb_gga_xc_cam_b3lyp = {
  XC_HYB_GGA_XC_CAM_B3LYP,
  XC_EXCHANGE_CORRELATION,
  "CAM version of B3LYP",
  XC_FAMILY_HYB_GGA,
  {&xc_ref_Yanai2004_51, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  xc_hyb_gga_xc_cam_b3lyp_init,
  NULL, NULL, NULL, NULL
};

const xc_func_info_type xc_func_info_hyb_gga_xc_tuned_cam_b3lyp = {
  XC_HYB_GGA_XC_TUNED_CAM_B3LYP,
  XC_EXCHANGE_CORRELATION,
  "CAM version of B3LYP, tuned for excitations and properties",
  XC_FAMILY_HYB_GGA,
  {&xc_ref_Okuno2012_29, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  xc_hyb_gga_xc_cam_b3lyp_init,
  NULL, NULL, NULL, NULL
};

const xc_func_info_type xc_func_info_hyb_gga_xc_cam_qtp_01 = {
  XC_HYB_GGA_XC_CAM_QTP_01,
  XC_EXCHANGE_CORRELATION,
  "CAM-B3LYP retuned using ionization potentials of water",
  XC_FAMILY_HYB_GGA,
  {&xc_ref_Jin2016_034107, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  xc_hyb_gga_xc_cam_b3lyp_init,
  NULL, NULL, NULL, NULL
};
