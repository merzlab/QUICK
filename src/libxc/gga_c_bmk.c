/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_C_N12           80 /* N12 functional from Minnesota            */
#define XC_GGA_C_N12_SX        79 /* N12-SX functional from Minnesota         */
#define XC_GGA_C_GAM           33 /* GAM functional from Minnesota            */
#define XC_GGA_C_BMK          280 /* Boese-Martin for kinetics                */
#define XC_GGA_C_TAU_HCTH     281 /* correlation part of tau-hcth             */
#define XC_GGA_C_HYB_TAU_HCTH 283 /* correlation part of hyb_tau-hcth         */

typedef struct {
  double c_ss[5], c_ab[5];
} gga_c_bmk_params;

/* c_ss and c_ab coefficients flipped in original paper! */
static const gga_c_bmk_params par_n12 = {
  { 1.00000e+00, -5.53170e+00,  3.07958e+01, -5.64196e+01,  3.21250e+01},
  { 1.00000e+00,  3.24511e+00, -2.52893e+01,  1.44407e+01,  1.96870e+01}
};

/* c_ss and c_ab coefficients flipped in original paper! */
static const gga_c_bmk_params par_n12_sx = {
  { 2.63373e+00, -1.05450e+00, -7.29853e-01,  4.94024e+00, -7.31760e+00},
  { 8.33615e-01,  3.24128e+00, -1.06407e+01, -1.60471e+01,  2.51047e+01}
};

static const gga_c_bmk_params par_gam = {
  { 0.231765,  0.575592, -3.43391, -5.77281,   9.52448},
  { 0.860548, -2.94135,  15.4176,  -5.99825, -23.4119}
};

static const gga_c_bmk_params par_bmk = {
  {-2.19098, 23.8939, -44.3303,  22.5982, 0.0},
  { 1.22334, -3.4631,  10.0731, -11.1974, 0.0}
};

static const gga_c_bmk_params par_tau_hcth = {
  { 0.41385, -0.9086, -0.0549, 1.7480, 0.0},
  { 0.65262, 6.3638, -14.080, -3.3755, 0.0}
};

static const gga_c_bmk_params par_hyb_tau_hcth = {
  { 0.18600, 3.9782, -7.0694, 3.4747, 0.0},
  { 0.80490, 3.8388, -13.547, 3.9133, 0.0}
};

static void 
gga_c_bmk_init(xc_func_type *p)
{
  gga_c_bmk_params *params;

  assert(p->params == NULL);
  p->params = malloc(sizeof(gga_c_bmk_params));
  params = (gga_c_bmk_params *)(p->params);

  switch(p->info->number){
  case XC_GGA_C_N12:
    memcpy(params, &par_n12, sizeof(gga_c_bmk_params));
    break;
  case XC_GGA_C_N12_SX:
    memcpy(params, &par_n12_sx, sizeof(gga_c_bmk_params));
    break;
  case XC_GGA_C_GAM:
    memcpy(params, &par_gam, sizeof(gga_c_bmk_params));
    break;
  case XC_GGA_C_BMK:
    memcpy(params, &par_bmk, sizeof(gga_c_bmk_params));
    break;
  case XC_GGA_C_TAU_HCTH:
    memcpy(params, &par_tau_hcth, sizeof(gga_c_bmk_params));
    break;
  case XC_GGA_C_HYB_TAU_HCTH:
    memcpy(params, &par_hyb_tau_hcth, sizeof(gga_c_bmk_params));
    break;
  default:
    fprintf(stderr, "Internal error in gga_c_bmk\n");
    exit(1);
    break;
  }
}

#include "maple2c/gga_c_bmk.c"

#define func maple2c_func
#include "work_gga_c.c"

const xc_func_info_type xc_func_info_gga_c_n12 = {
  XC_GGA_C_N12,
  XC_CORRELATION,
  "Minnesota N12 functional",
  XC_FAMILY_GGA,
  {&xc_ref_Peverati2012_2310, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-20,
  0, NULL, NULL,
  gga_c_bmk_init, NULL, 
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_c_n12_sx = {
  XC_GGA_C_N12_SX,
  XC_CORRELATION,
  "Minnesota N12-SX functional",
  XC_FAMILY_GGA,
  {&xc_ref_Peverati2012_16187, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-20,
  0, NULL, NULL,
  gga_c_bmk_init, NULL, 
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_c_gam = {
  XC_GGA_C_GAM,
  XC_CORRELATION,
  "GAM functional from Minnesota",
  XC_FAMILY_GGA,
  {&xc_ref_Yu2015_12146, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC | XC_FLAGS_DEVELOPMENT,
  1e-23,
  0, NULL, NULL,
  gga_c_bmk_init, NULL,
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_c_bmk = {
  XC_GGA_C_BMK,
  XC_CORRELATION,
  "Boese-Martin for kinetics",
  XC_FAMILY_GGA,
  {&xc_ref_Boese2004_3405, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-20,
  0, NULL, NULL,
  gga_c_bmk_init, NULL,
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_c_tau_hcth = {
  XC_GGA_C_TAU_HCTH,
  XC_CORRELATION,
  "correlation part of tau-hcth",
  XC_FAMILY_GGA,
  {&xc_ref_Boese2002_9559, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-20,
  0, NULL, NULL,
  gga_c_bmk_init, NULL,
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_c_hyb_tau_hcth = {
  XC_GGA_C_HYB_TAU_HCTH,
  XC_CORRELATION,
  "correlation part of hyb-tau-hcth",
  XC_FAMILY_GGA,
  {&xc_ref_Boese2002_9559, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-20,
  0, NULL, NULL,
  gga_c_bmk_init, NULL,
  NULL, work_gga_c, NULL
};
