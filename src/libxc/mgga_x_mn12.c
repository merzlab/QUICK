/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_MGGA_X_MN12_L        227 /* MN12-L exchange functional from Minnesota          */ 
#define XC_HYB_MGGA_X_MN12_SX   248 /* MN12-SX hybrid exchange functional from Minnesota  */ 
#define XC_MGGA_X_MN15_L        260 /* MN15-L exhange functional from Minnesota           */ 
#define XC_HYB_MGGA_X_MN15      268 /* MN15 hybrid exchange functional from Minnesota     */ 
 
typedef struct{ 
  const double c[40]; 
} mgga_x_mn12_params; 
 
/* the ordering is 
CC000 [ 0], CC001 [ 1], CC002 [ 2], CC003 [ 3], CC004 [ 4], CC005 [ 5] 
CC010 [ 6], CC011 [ 7], CC012 [ 8], CC013 [ 9], CC014 [10], 
CC020 [11], CC021 [12], CC022 [13], CC023 [14], 
CC030 [15], CC031 [16], CC032 [17], 
CC100 [18], CC101 [19], CC102 [20], CC103 [21], CC104 [22] 
CC110 [23], CC111 [24], CC112 [25], CC113 [26], 
CC120 [27], CC121 [28], CC122 [29], 
CC200 [30], CC201 [31], CC202 [32], CC203 [33], 
CC210 [34], CC211 [35], CC212 [36], 
CC300 [37], CC301 [38], CC302 [39] 
*/ 
static const mgga_x_mn12_params par_mn12_l = {{ 
    6.735981e-01, -2.270598e+00, -2.613712e+00,  3.993609e+00,  4.635575e+00, 1.250676e+00, 
    8.444920e-01, -1.301173e+01, -1.777730e+01, -4.627211e+00,  5.976605e+00, 
    1.142897e+00, -2.040226e+01, -2.382843e+01,  7.119109e+00, 
    -2.335726e+01, -1.622633e+01,  1.482732e+01, 
    1.449285e+00,  1.020598e+01,  4.407450e+00, -2.008193e+01, -1.253561e+01, 
    -5.435031e+00,  1.656736e+01,  2.000229e+01, -2.513105e+00, 
    9.658436e+00, -3.825281e+00, -2.500000e+01, 
    -2.070080e+00, -9.951913e+00,  8.731211e-01,  2.210891e+01, 
    8.822633e+00,  2.499949e+01,  2.500000e+01, 
    6.851693e-01, -7.406948e-02, -6.788000e-01 
  }}; 
 
static const mgga_x_mn12_params par_mn12_sx = {{ 
   5.226556e-01, -2.681208e-01, -4.670705e+00,  3.067320e+00,  4.095370e+00,  2.653023e+00, 
   5.165969e-01, -2.035442e+01, -9.946472e+00,  2.938637e+00,  1.131100e+01, 
   4.752452e+00, -3.061331e+00, -2.523173e+01,  1.710903e+01, 
  -2.357480e+01, -2.727754e+01,  1.603291e+01, 
   1.842503e+00,  1.927120e+00,  1.107987e+01, -1.182087e+01, -1.117768e+01, 
  -5.821000e+00,  2.266545e+01,  8.246708e+00, -4.778364e+00, 
   5.329122e-01, -6.666755e+00,  1.671429e+00, 
  -3.311409e+00,  3.415913e-01, -6.413076e+00,  1.038584e+01, 
   9.026277e+00,  1.929689e+01,  2.669232e+01, 
   1.517278e+00, -3.442503e+00,  1.100161e+00 
  }}; 
 
static const mgga_x_mn12_params par_mn15_l = {{ 
   0.670864162, -0.822003903, -1.022407046,  1.689460986, -0.00562032,  -0.110293849, 
   0.972245178, -6.697641991, -4.322814495, -6.786641376, -5.687461462, 
   9.419643818, 11.83939406,   5.086951311,  4.302369948, 
  -8.07344065,   2.429988978, 11.09485698, 
   1.247333909,  3.700485291,  0.867791614, -0.591190518, -0.295305435, 
  -5.825759145,  2.537532196,  3.143390933,  2.939126332, 
   0.599342114,  2.241702738,  2.035713838, 
  -1.525344043, -2.325875691,  1.141940663, -1.563165026, 
   7.882032871, 11.93400684,   9.852928303, 
   0.584030245, -0.720941131, -2.836037078 
  }}; 
 
static const mgga_x_mn12_params par_mn15 = {{ 
   0.073852235, -0.839976156, -3.082660125, -1.02881285, -0.811697255,   -0.063404387, 
   2.54805518,  -5.031578906,  0.31702159,  2.981868205, -0.749503735, 
   0.231825661,  1.261961411,  1.665920815, 7.483304941, 
  -2.544245723,  1.384720031,  6.902569885, 
   1.657399451, 2.98526709,    6.89391326,   2.489813993,  1.454724691, 
  -5.054324071, 2.35273334,    1.299104132,  1.203168217, 
   0.121595877,  8.048348238, 21.91203659, 
  -1.852335832, -3.4722735,   -1.564591493, -2.29578769, 
   3.666482991, 10.87074639,  9.696691388, 
   0.630701064, -0.505825216, -3.562354535 
  }}; 
 
static void 
mgga_x_mn12_init(xc_func_type *p) 
{ 
  mgga_x_mn12_params *params; 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(mgga_x_mn12_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(mgga_x_mn12_params); 
#endif 
  params = (mgga_x_mn12_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_MGGA_X_MN12_L: 
    memcpy(params, &par_mn12_l, sizeof(mgga_x_mn12_params)); 
    break; 
  case XC_HYB_MGGA_X_MN12_SX: 
    memcpy(params, &par_mn12_sx, sizeof(mgga_x_mn12_params)); 
    p->cam_alpha = 0.00; 
    p->cam_beta  = 0.25; 
    p->cam_omega = 0.11; 
    break; 
  case XC_MGGA_X_MN15_L: 
    memcpy(params, &par_mn15_l, sizeof(mgga_x_mn12_params)); 
    break; 
  case XC_HYB_MGGA_X_MN15: 
    memcpy(params, &par_mn15, sizeof(mgga_x_mn12_params)); 
    p->cam_alpha = 0.44; 
    p->cam_beta  = 0.00; 
    p->cam_omega = 0.00; 
    break; 
  default: 
    fprintf(stderr, "Internal error in mgga_x_mn12\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/mgga_x_mn12.c" 
#endif 
 
#define func maple2c_func 
#include "work_mgga_c.c" 
 
 
const xc_func_info_type xc_func_info_mgga_x_mn12_l = { 
  XC_MGGA_X_MN12_L, 
  XC_EXCHANGE, 
  "Minnesota MN12-L exchange functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Peverati2012_13171, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_x_mn12_init, NULL, 
  NULL, NULL, work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_mn12_sx = { 
  XC_HYB_MGGA_X_MN12_SX, 
  XC_EXCHANGE, 
  "Minnesota MN12-SX hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Peverati2012_16187, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-32, 
  0, NULL, NULL, 
  mgga_x_mn12_init, NULL, 
  NULL, NULL, work_mgga_c 
}; 
 
const xc_func_info_type xc_func_info_mgga_x_mn15_l = { 
  XC_MGGA_X_MN15_L, 
  XC_EXCHANGE, 
  "Minnesota MN15-L exchange functional", 
  XC_FAMILY_MGGA, 
  {&xc_ref_Yu2016_1280, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-23, 
  0, NULL, NULL, 
  mgga_x_mn12_init, NULL, 
  NULL, NULL, work_mgga_c, 
}; 
 
const xc_func_info_type xc_func_info_hyb_mgga_x_mn15 = { 
  XC_HYB_MGGA_X_MN15, 
  XC_EXCHANGE, 
  "Minnesota MN15 hybrid exchange functional", 
  XC_FAMILY_HYB_MGGA, 
  {&xc_ref_Yu2016_5032, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-32, 
  0, NULL, NULL, 
  mgga_x_mn12_init, NULL, 
  NULL, NULL, work_mgga_c, 
}; 
