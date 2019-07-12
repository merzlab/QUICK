/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_X_M11_L        226 /* M11-L exchange functional from Minnesota  */

typedef struct{
  const double a[12], b[21], c[12], d[12];
} mgga_x_m11_l_params;

static const mgga_x_m11_l_params par_m11_l = {
  {
     8.121131e-01,  1.738124e+01,  1.154007e+00,  6.869556e+01,  1.016864e+02, -5.887467e+00, 
     4.517409e+01, -2.773149e+00, -2.617211e+01,  0.000000e+00,  0.000000e+00,  0.000000e+00
  }, {
     1.878869e-01, -1.653877e+01,  6.755753e-01, -7.567572e+01, -1.040272e+02,  1.831853e+01,
    -5.573352e+01, -3.520210e+00,  3.724276e+01,  0.000000e+00,  0.000000e+00,  0.000000e+00
  }, {
    -4.386615e-01, -1.214016e+02, -1.393573e+02, -2.046649e+00,  2.804098e+01, -1.312258e+01,
    -6.361819e+00, -8.055758e-01,  3.736551e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00
  }, {
     1.438662e+00,  1.209465e+02,  1.328252e+02,  1.296355e+01,  5.854866e+00, -3.378162e+00,
    -4.423393e+01,  6.844475e+00,  1.949541e+01,  0.000000e+00,  0.000000e+00,  0.000000e+00
  }
};

static void
mgga_x_m11_l_init(xc_func_type *p)
{
  mgga_x_m11_l_params *params;

  assert(p->params == NULL);
  p->params = malloc(sizeof(mgga_x_m11_l_params));
  params = (mgga_x_m11_l_params *) (p->params);

  switch(p->info->number){
  case XC_MGGA_X_M11_L:
    memcpy(params, &par_m11_l, sizeof(mgga_x_m11_l_params));
    p->cam_omega = 0.25;
    break;
  default:
    fprintf(stderr, "Internal error in mgga_x_m11_l\n");
    exit(1);
  }
}

#include "maple2c/mgga_x_m11_l.c"

#define func maple2c_func
#include "work_mgga_c.c"

const xc_func_info_type xc_func_info_mgga_x_m11_l = {
  XC_MGGA_X_M11_L,
  XC_EXCHANGE,
  "Minnesota M11-L exchange functional",
  XC_FAMILY_MGGA,
  {&xc_ref_Peverati2012_117, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-13,
  0, NULL, NULL,
  mgga_x_m11_l_init, NULL, 
  NULL, NULL, work_mgga_c,
};
