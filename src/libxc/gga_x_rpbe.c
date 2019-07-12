/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_GGA_X_RPBE  117 /* Hammer, Hansen & Norskov (PBE-like) */


typedef struct{
  double rpbe_kappa, rpbe_mu;
} gga_x_rpbe_params;


static void 
gga_x_rpbe_init(xc_func_type *p)
{
  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(gga_x_rpbe_params));

  /* same parameters as standard PBE */
  xc_gga_x_rpbe_set_params(p, 0.8040, 0.2195149727645171);
}


void 
xc_gga_x_rpbe_set_params(xc_func_type *p, double kappa, double mu)
{
  gga_x_rpbe_params *params;

  assert(p != NULL && p->params != NULL);
  params = (gga_x_rpbe_params *) (p->params);

  params->rpbe_kappa = kappa;
  params->rpbe_mu    = mu;
}

#include "maple2c/gga_x_rpbe.c"

#define func xc_gga_x_rpbe_enhance
#include "work_gga_x.c"

const xc_func_info_type xc_func_info_gga_x_rpbe = {
  XC_GGA_X_RPBE,
  XC_EXCHANGE,
  "Hammer, Hansen, and Norskov",
  XC_FAMILY_GGA,
  {&xc_ref_Hammer1999_7413, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_rpbe_init, NULL, 
  NULL, work_gga_x, NULL
};
