/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_C_M08_HX       78 /* M08-HX correlation functional from Minnesota  */ 
#define XC_MGGA_C_M08_SO       77 /* M08-SO correlation functional from Minnesota  */ 
#define XC_MGGA_C_M11          76 /* M11 correlation functional from Minnesota     */ 
#define XC_MGGA_C_M11_L        75 /* M11-L correlation functional from Minnesota   */ 
#define XC_MGGA_C_MN12_L       74 /* MN12-L correlation functional from Minnesota  */ 
#define XC_MGGA_C_MN12_SX      73 /* MN12-SX correlation functional from Minnesota */ 
#define XC_MGGA_C_MN15_L      261 /* MN15-L correlation functional from Minnesota  */ 
#define XC_MGGA_C_MN15        269 /* MN15 correlation functional from Minnesota    */ 
 
typedef struct{ 
  const double m08_a[12], m08_b[12]; 
} mgga_c_m08_params; 
 
 
static const mgga_c_m08_params par_m08_hx = { 
  { 
    1.0000000e+00, -4.0661387e-01, -3.3232530e+00,  1.5540980e+00,  4.4248033e+01, -8.4351930e+01, 
    -1.1955581e+02,  3.9147081e+02,  1.8363851e+02, -6.3268223e+02, -1.1297403e+02,  3.3629312e+02 
  }, { 
    1.3812334e+00, -2.4683806e+00, -1.1901501e+01, -5.4112667e+01,  1.0055846e+01,  1.4800687e+02, 
    1.1561420e+02,  2.5591815e+02,  2.1320772e+02, -4.8412067e+02, -4.3430813e+02,  5.6627964e+01 
  } 
}; 
 
static const mgga_c_m08_params par_m08_so = { 
  {  
    1.0000000e+00,  0.0000000e+00, -3.9980886e+00,  1.2982340e+01,  1.0117507e+02, -8.9541984e+01, 
    -3.5640242e+02,  2.0698803e+02,  4.6037780e+02, -2.4510559e+02, -1.9638425e+02,  1.1881459e+02 
  }, { 
    1.0000000e+00, -4.4117403e+00, -6.4128622e+00,  4.7583635e+01,  1.8630053e+02, -1.2800784e+02, 
    -5.5385258e+02,  1.3873727e+02,  4.1646537e+02, -2.6626577e+02,  5.6676300e+01,  3.1673746e+02 
  } 
}; 
 
static const mgga_c_m08_params par_m11 = { 
  { 
    1.0000000e+00,  0.0000000e+00, -3.8933250e+00, -2.1688455e+00,  9.3497200e+00, -1.9845140e+01, 
    2.3455253e+00,  7.9246513e+01,  9.6042757e+00, -6.7856719e+01, -9.1841067e+00,  0.0000000e+00 
  }, { 
    7.2239798e-01,  4.3730564e-01, -1.6088809e+01, -6.5542437e+01,  3.2057230e+01,  1.8617888e+02, 
    2.0483468e+01, -7.0853739e+01,  4.4483915e+01, -9.4484747e+01, -1.1459868e+02,  0.0000000e+00 
  } 
}; 
 
static const mgga_c_m08_params par_m11_l = { 
  { 
    1.000000e+00,  0.000000e+00,  2.750880e+00, -1.562287e+01,  9.363381e+00,  2.141024e+01, 
    -1.424975e+01, -1.134712e+01,  1.022365e+01,  0.000000e+00,  0.000000e+00,  0.000000e+00 
  }, { 
    1.000000e+00, -9.082060e+00,  6.134682e+00, -1.333216e+01, -1.464115e+01,  1.713143e+01, 
    2.480738e+00, -1.007036e+01, -1.117521e-01,  0.000000e+00,  0.000000e+00,  0.000000e+00 
  } 
}; 
 
static const mgga_c_m08_params par_mn12_l = { 
  { 
    8.844610e-01, -2.202279e-01,  5.701372e+00, -2.562378e+00, -9.646827e-01,  1.982183e-01, 
    1.019976e+01,  9.789352e-01, -1.512722e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00 
  }, { 
    5.323948e-01, -5.831909e+00,  3.882386e+00,  5.878488e+00,  1.493228e+01, -1.374636e+01, 
    -8.492327e+00, -2.486548e+00, -1.822346e+01,  0.000000e+00,  0.000000e+00,  0.000000e+00 
  } 
}; 
 
static const mgga_c_m08_params par_mn12_sx = { 
  { 
    7.171161e-01, -2.380914e+00,  5.793565e+00, -1.243624e+00,  1.364920e+01, -2.110812e+01, 
    -1.598767e+01,  1.429208e+01,  6.149191e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00 
  }, {  
    4.663699e-01, -9.110685e+00,  8.705051e+00, -1.813949e+00, -4.147211e-01, -1.021527e+01, 
    8.240270e-01,  4.993815e+00, -2.563930e+01,  0.000000e+00,  0.000000e+00,  0.000000e+00 
  } 
}; 
 
static const mgga_c_m08_params par_mn15_l = { 
  { 
    0.952058087, -0.756954364,  5.677396094, -5.017104782, -5.10654071, -4.812053335, 
    3.397640087,  1.980041517, 10.1231046,    0.0,          0.0,         0.0 
  }, { 
    0.819504932, -7.689358913, -0.70532663,  -0.600096421, 11.03332527,  5.861969337, 
    8.913865465,  5.74529876,   4.254880837,  0.0,          0.0,         0.0 
  } 
}; 
 
static const mgga_c_m08_params par_mn15 = { 
  { 
    1.093250748, -0.269735037, 6.368997613, -0.245337101, -1.587103441, 0.124698862, 
    1.605819855,  0.466206031, 3.484978654,  0.0,          0.0,         0.0 
  }, { 
    1.427424993, -3.57883682,  7.398727547,  3.927810559,  2.789804639, 4.988320462, 
    3.079464318,  3.521636859, 4.769671992,  0.0,          0.0,         0.0 
  } 
}; 
 
 
static void 
mgga_c_m08_init(xc_func_type *p) 
{ 
  mgga_c_m08_params *params; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(mgga_c_m08_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(mgga_c_m08_params); 
#endif 
  params = (mgga_c_m08_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_MGGA_C_M08_HX:  
    memcpy(params, &par_m08_hx, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_M08_SO: 
    memcpy(params, &par_m08_so, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_M11: 
    memcpy(params, &par_m11, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_M11_L: 
    memcpy(params, &par_m11_l, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_MN12_L: 
    memcpy(params, &par_mn12_l, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_MN12_SX: 
    memcpy(params, &par_mn12_sx, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_MN15_L: 
    memcpy(params, &par_mn15_l, sizeof(mgga_c_m08_params)); 
    break; 
  case XC_MGGA_C_MN15: 
    memcpy(params, &par_mn15, sizeof(mgga_c_m08_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in mgga_c_m08\n"); 
    exit(1); 
  } 
} 
 
 
#ifndef DEVICE 
#include "maple2c/mgga_c_m08.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
 
const xc_func_info_type xc_func_info_mgga_c_m08_hx = { 
  XC_MGGA_C_M08_HX, 
  XC_CORRELATION, 
  "Minnesota M08 correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Zhao2008_1849, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_m08_so = { 
  XC_MGGA_C_M08_SO, 
  XC_CORRELATION, 
  "Minnesota M08-SO correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Zhao2008_1849, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_m11 = { 
  XC_MGGA_C_M11, 
  XC_CORRELATION, 
  "Minnesota M11 correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Peverati2011_2810, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_m11_l = { 
  XC_MGGA_C_M11_L, 
  XC_CORRELATION, 
  "Minnesota M11-L correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Peverati2012_117, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_mn12_l = { 
  XC_MGGA_C_MN12_L, 
  XC_CORRELATION, 
  "Minnesota MN12-L correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Peverati2012_13171, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_mn12_sx = { 
  XC_MGGA_C_MN12_SX, 
  XC_CORRELATION, 
  "Minnesota MN12-SX correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Peverati2012_16187, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_mn15_l = { 
  XC_MGGA_C_MN15_L, 
  XC_CORRELATION, 
  "Minnesota MN15-L correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Yu2016_1280, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_mgga_c_mn15 = { 
  XC_MGGA_C_MN15, 
  XC_CORRELATION, 
  "Minnesota MN15 correlation functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Yu2016_5032, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_c_m08_init, 
  NULL, NULL, NULL, 
  work_mgga_c, 
}; 
