/*
 Copyright (C) 2014 Jess Wellendorff, M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_X_MBEEF          249 /* mBEEF exchange */

static const double coefs_mbeef[64] = {
         1.18029330e+00,   8.53027860e-03,  -1.02312143e-01,
         6.85757490e-02,  -6.61294786e-03,  -2.84176163e-02,
         5.54283363e-03,   3.95434277e-03,  -1.98479086e-03,
         1.00339208e-01,  -4.34643460e-02,  -1.82177954e-02,
         1.62638575e-02,  -8.84148272e-03,  -9.57417512e-03,
         9.40675747e-03,   6.37590839e-03,  -8.79090772e-03,
        -1.50103636e-02,   2.80678872e-02,  -1.82911291e-02,
        -1.88495102e-02,   1.69805915e-07,  -2.76524680e-07,
         1.44642135e-03,  -3.03347141e-03,   2.93253041e-03,
        -8.45508103e-03,   6.31891628e-03,  -8.96771404e-03,
        -2.65114646e-08,   5.05920757e-08,   6.65511484e-04,
         1.19130546e-03,   1.82906057e-03,   3.39308972e-03,
        -7.90811707e-08,   1.62238741e-07,  -4.16393106e-08,
         5.54588743e-08,  -1.16063796e-04,   8.22139896e-04,
        -3.51041030e-04,   8.96739466e-04,   2.09603871e-08,
        -3.76702959e-08,   2.36391411e-08,  -3.38128188e-08,
        -5.54173599e-06,  -5.14204676e-05,   6.68980219e-09,
        -2.16860568e-08,   9.12223751e-09,  -1.38472194e-08,
         6.94482484e-09,  -7.74224962e-09,   7.36062570e-07,
        -9.40351563e-06,  -2.23014657e-09,   6.74910119e-09,
        -4.93824365e-09,   8.50272392e-09,  -6.91592964e-09,
         8.88525527e-09 };

static const double coefs_mbeefvdw[25] = {
         1.17114923e+00,   1.15594371e-01,  -5.32167416e-02,
        -2.01131648e-02,   1.41417107e-03,  -6.76157938e-02,
         4.53837246e-02,  -2.22650139e-02,   1.92374554e-02,
         9.19317034e-07,   1.48659502e-02,   3.18024096e-02,
        -5.21818079e-03,   1.33707403e-07,  -5.00749348e-07,
         1.40794142e-03,  -6.08338264e-03,  -6.57949254e-07,
        -5.49909413e-08,   5.74317889e-08,   1.41530486e-04,
        -1.00478906e-07,   2.01895739e-07,   3.97324768e-09,
        -3.40722258e-09 };

typedef struct{
  int legorder;
  const double *coefs;
} mgga_x_mbeef_params;

static void
mgga_x_mbeef_init(xc_func_type *p)
{
  mgga_x_mbeef_params *params;

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(mgga_x_mbeef_params));
  params = (mgga_x_mbeef_params *)p->params;

  switch(p->info->number){
  case XC_MGGA_X_MBEEF:
    params->legorder = 8;
    params->coefs = coefs_mbeef;
    break;
  case XC_MGGA_X_MBEEFVDW:
    params->legorder = 5;
    params->coefs = coefs_mbeefvdw;
    break;
  default:
    fprintf(stderr, "Internal error in mgga_x_mbeef\n");
    exit(1);
  }
}

#include "maple2c/mgga_x_mbeef.c"

#define func maple2c_func
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_mbeef = {
  XC_MGGA_X_MBEEF,
  XC_EXCHANGE,
  "mBEEF exchange",
  XC_FAMILY_MGGA,
  {&xc_ref_Wellendorff2014_144107, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  5e-7,
  0, NULL, NULL,
  mgga_x_mbeef_init,
  NULL, NULL, NULL,
  work_mgga_x,
};
