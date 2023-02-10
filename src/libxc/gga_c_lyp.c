/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_GGA_C_LYP    131  /* Lee, Yang & Parr */
#define XC_GGA_C_TM_LYP 559  /* Takkar and McCarthy reparametrization */

typedef struct{
  double A, B, c, d;
} gga_c_lyp_params;

void xc_gga_c_lyp_init(xc_func_type *p)
{
  assert(p->params == NULL);

  p->params = malloc(sizeof(gga_c_lyp_params));
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  p->params_byte_size = sizeof(gga_c_lyp_params); //Assign the value for param size
#endif
  /* values of constants in standard LYP functional */
  switch(p->info->number){
  case XC_GGA_C_LYP:
    xc_gga_c_lyp_set_params(p, 0.04918, 0.132, 0.2533, 0.349);
    break;
  case XC_GGA_C_TM_LYP:
    xc_gga_c_lyp_set_params(p, 0.0393, 0.21, 0.41, 0.15);
    break;
  default:
    fprintf(stderr, "Internal error in gga_c_lyp\n");
    exit(1);
  }
}


void xc_gga_c_lyp_set_params(xc_func_type *p, double A, double B, double c, double d)
{
  gga_c_lyp_params *params;

  assert(p != NULL && p->params != NULL);
  params = (gga_c_lyp_params *) (p->params);

  params->A = A;
  params->B = B;
  params->c = c;
  params->d = d;
}

#ifndef DEVICE
#include "maple2c/gga_c_lyp.c"
#endif

#define func maple2c_func
#include "work_gga_c.c"

const xc_func_info_type xc_func_info_gga_c_lyp = {
  XC_GGA_C_LYP,
  XC_CORRELATION,
  "Lee, Yang & Parr",
  XC_FAMILY_GGA,
  {&xc_ref_Lee1988_785, &xc_ref_Miehlich1989_200, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  xc_gga_c_lyp_init, NULL,
  NULL, work_gga_c, NULL
};

const xc_func_info_type xc_func_info_gga_c_tm_lyp = {
  XC_GGA_C_TM_LYP,
  XC_CORRELATION,
  "Takkar and McCarthy reparametrization",
  XC_FAMILY_GGA,
  {&xc_ref_Thakkar2009_134109, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  xc_gga_c_lyp_init, NULL,
  NULL, work_gga_c, NULL
};
