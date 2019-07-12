/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_X_FT97_A       114 /* Filatov & Thiel 97 (version A) */
#define XC_GGA_X_FT97_B       115 /* Filatov & Thiel 97 (version B) */

typedef struct{
  double beta0, beta1, beta2;
} gga_x_ft97_params;

static const gga_x_ft97_params par_ft97_a = {
  0.00293, 0.0, 0.0
};

static const gga_x_ft97_params par_ft97_b = {
  /* These parameters are what Filatov and Thiel actually used, not
     the ones they published in the paper... the differences being that
     beta1 has one more digit, and beta2 is squared: 2501.149^2 */
  0.002913644, 0.0009474169, 6255746.320201
};

static void 
gga_x_ft97_init(xc_func_type *p)
{
  gga_x_ft97_params *params;

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(gga_x_ft97_params));
  params = (gga_x_ft97_params *) (p->params);

  switch(p->info->number){
  case XC_GGA_X_FT97_A: 
    memcpy(params, &par_ft97_a, sizeof(gga_x_ft97_params));
    break;
  case XC_GGA_X_FT97_B:
    memcpy(params, &par_ft97_b, sizeof(gga_x_ft97_params));
    break;
  default:
    fprintf(stderr, "Internal error in gga_x_ft97\n");
    exit(1);
  }
}

#include "maple2c/gga_x_ft97.c"

#define func maple2c_func
#include "work_gga_c.c"

const xc_func_info_type xc_func_info_gga_x_ft97_a = {
  XC_GGA_X_FT97_A,
  XC_EXCHANGE,
  "Filatov & Thiel 97 (version A)",
  XC_FAMILY_GGA,
  {&xc_ref_Filatov1997_847, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-22,
  0, NULL, NULL,
  gga_x_ft97_init, NULL, 
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_x_ft97_b = {
  XC_GGA_X_FT97_B,
  XC_EXCHANGE,
  "Filatov & Thiel 97 (version B)",
  XC_FAMILY_GGA,
  {&xc_ref_Filatov1997_847, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-22,
  0, NULL, NULL,
  gga_x_ft97_init, NULL,
  NULL, work_gga_c, NULL
};
