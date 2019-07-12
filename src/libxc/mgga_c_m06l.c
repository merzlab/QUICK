/*
 Copyright (C) 2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_C_M06_L         233 /* M06-L correlation functional from Minnesota          */
#define XC_MGGA_C_M06_HF        234 /* M06-HF correlation functional from Minnesota         */
#define XC_MGGA_C_M06           235 /* M06 correlation functional from Minnesota            */
#define XC_MGGA_C_M06_2X        236 /* M06-2X correlation functional from Minnesota         */
#define XC_MGGA_C_REVM06_L      294 /* Revised M06-L correlation functional from Minnesota  */

typedef struct{
  double gamma_ss, gamma_ab, alpha_ss, alpha_ab;
  const double css[5], cab[5], dss[6], dab[6];
} mgga_c_m06l_params;

static const mgga_c_m06l_params par_m06l = {
  0.06, 0.0031, 0.00515088, 0.00304966,
  { 5.349466e-01,  5.396620e-01, -3.161217e+01,  5.149592e+01, -2.919613e+01},
  { 6.042374e-01,  1.776783e+02, -2.513252e+02,  7.635173e+01, -1.255699e+01},
  { 4.650534e-01,  1.617589e-01,  1.833657e-01,  4.692100e-04, -4.990573e-03,  0.000000e+00},
  { 3.957626e-01, -5.614546e-01,  1.403963e-02,  9.831442e-04, -3.577176e-03,  0.000000e+00}
};

static const mgga_c_m06l_params par_m06hf = {
  0.06, 0.0031, 0.00515088, 0.00304966,
  { 1.023254e-01, -2.453783e+00,  2.913180e+01, -3.494358e+01,  2.315955e+01},
  { 1.674634e+00,  5.732017e+01,  5.955416e+01, -2.311007e+02,  1.255199e+02},
  { 8.976746e-01, -2.345830e-01,  2.368173e-01, -9.913890e-04, -1.146165e-02,  0.000000e+00},
  {-6.746338e-01, -1.534002e-01, -9.021521e-02, -1.292037e-03, -2.352983e-04,  0.000000e+00}
};

static const mgga_c_m06l_params par_m06 = {
  0.06, 0.0031, 0.00515088, 0.00304966,  
  { 5.094055e-01, -1.491085e+00,  1.723922e+01, -3.859018e+01,  2.845044e+01},
  { 3.741539e+00,  2.187098e+02, -4.531252e+02,  2.936479e+02, -6.287470e+01},
  { 4.905945e-01, -1.437348e-01,  2.357824e-01,  1.871015e-03, -3.788963e-03,  0.000000e+00},
  {-2.741539e+00, -6.720113e-01, -7.932688e-02,  1.918681e-03, -2.032902e-03,  0.000000e+00}
};

static const mgga_c_m06l_params par_m062x = {
  0.06, 0.0031, 0.00515088, 0.00304966,  
  { 3.097855e-01, -5.528642e+00,  1.347420e+01, -3.213623e+01,  2.846742e+01},
  { 8.833596e-01,  3.357972e+01, -7.043548e+01,  4.978271e+01, -1.852891e+01},
  { 6.902145e-01,  9.847204e-02,  2.214797e-01, -1.968264e-03, -6.775479e-03,  0.000000e+00},
  { 1.166404e-01, -9.120847e-02, -6.726189e-02,  6.720580e-05,  8.448011e-04,  0.000000e+00}
};

static const mgga_c_m06l_params par_revm06l = {
  0.06, 0.0031, 0.00515088, 0.00304966,
  { 1.227659748,  0.855201283, -3.113346677, -2.239678026,  0.354638962},
  { 0.344360696, -0.557080242, -2.009821162, -1.857641887, -1.076639864},
  {-0.538821292, -0.028296030,  0.023889696,   0.0, 0.0,   -0.002437902},
  { 0.400714600,  0.015796569, -0.032680984,   0.0, 0.0,    0.001260132}
};

static void 
mgga_c_vsxc_init(xc_func_type *p)
{
  mgga_c_m06l_params *params;

  assert(p != NULL);

  p->n_func_aux  = 1;
  p->func_aux    = (xc_func_type **) malloc(1*sizeof(xc_func_type *));
  p->func_aux[0] = (xc_func_type *)  malloc(  sizeof(xc_func_type));

  xc_func_init(p->func_aux[0], XC_LDA_C_PW_MOD, XC_POLARIZED);

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(mgga_c_m06l_params));
  params = (mgga_c_m06l_params *)p->params;

  switch(p->info->number){
  case XC_MGGA_C_M06_L:
    memcpy(params, &par_m06l, sizeof(mgga_c_m06l_params));
    break;
  case XC_MGGA_C_M06_HF: 
    memcpy(params, &par_m06hf, sizeof(mgga_c_m06l_params));
    break;
  case XC_MGGA_C_M06:
    memcpy(params, &par_m06, sizeof(mgga_c_m06l_params));
    break;
  case XC_MGGA_C_M06_2X:
    memcpy(params, &par_m062x, sizeof(mgga_c_m06l_params));
    break;
  case XC_MGGA_C_REVM06_L:
    memcpy(params, &par_revm06l, sizeof(mgga_c_m06l_params));
    break;
  default:
    fprintf(stderr, "Internal error in mgga_c_m06l\n");
    exit(1);
  }  
}

#include "maple2c/mgga_c_m06l.c"

#define func maple2c_func
#include "work_mgga_c.c"

const xc_func_info_type xc_func_info_mgga_c_m06_l = {
  XC_MGGA_C_M06_L,
  XC_CORRELATION,
  "Minnesota M06-L correlation functional",
  XC_FAMILY_MGGA,
  {&xc_ref_Zhao2006_194101, &xc_ref_Zhao2008_215, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1.0e-23,
  0, NULL, NULL,
  mgga_c_vsxc_init, NULL,
  NULL, NULL, work_mgga_c,
};

const xc_func_info_type xc_func_info_mgga_c_m06_hf = {
  XC_MGGA_C_M06_HF,
  XC_CORRELATION,
  "Minnesota M06-HF correlation functional",
  XC_FAMILY_MGGA,
  {&xc_ref_Zhao2006_13126, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1.0e-23,
  0, NULL, NULL,
  mgga_c_vsxc_init, NULL, 
  NULL, NULL, work_mgga_c,
};

const xc_func_info_type xc_func_info_mgga_c_m06 = {
  XC_MGGA_C_M06,
  XC_CORRELATION,
  "Minnesota M06 correlation functional",
  XC_FAMILY_MGGA,
  {&xc_ref_Zhao2008_215, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1.0e-23,
  0, NULL, NULL,
  mgga_c_vsxc_init, NULL,
  NULL, NULL, work_mgga_c,
};

const xc_func_info_type xc_func_info_mgga_c_m06_2x = {
  XC_MGGA_C_M06_2X,
  XC_CORRELATION,
  "Minnesota M06-2X correlation functional",
  XC_FAMILY_MGGA,
  {&xc_ref_Zhao2008_215, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1.0e-23,
  0, NULL, NULL,
  mgga_c_vsxc_init, NULL,
  NULL, NULL, work_mgga_c,
};

const xc_func_info_type xc_func_info_mgga_c_revm06_l = {
  XC_MGGA_C_REVM06_L,
  XC_CORRELATION,
  "Minnesota revM06-L correlation functional",
  XC_FAMILY_MGGA,
  {&xc_ref_Wang2017_8487, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  5.0e-13,
  0, NULL, NULL,
  mgga_c_vsxc_init, NULL,
  NULL, NULL, work_mgga_c,
};
