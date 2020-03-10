/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_PBE          101 /* Perdew, Burke & Ernzerhof exchange             */ 
#define XC_GGA_X_PBE_R        102 /* Perdew, Burke & Ernzerhof exchange (revised)   */ 
#define XC_GGA_X_PBE_SOL      116 /* Perdew, Burke & Ernzerhof exchange (solids)    */ 
#define XC_GGA_X_XPBE         123 /* xPBE reparametrization by Xu & Goddard         */ 
#define XC_GGA_X_PBE_JSJR     126 /* JSJR reparametrization by Pedroza, Silva & Capelle */ 
#define XC_GGA_X_PBEK1_VDW    140 /* PBE reparametrization for vdW                  */ 
#define XC_GGA_X_APBE         184 /* mu fixed from the semiclassical neutral atom   */ 
#define XC_GGA_X_PBE_TCA       59 /* PBE revised by Tognetti et al                  */ 
#define XC_GGA_X_PBE_MOL       49 /* Del Campo, Gazquez, Trickey and Vela (PBE-like) */ 
#define XC_GGA_X_LAMBDA_LO_N   45 /* lambda_LO(N) version of PBE                    */ 
#define XC_GGA_X_LAMBDA_CH_N   44 /* lambda_CH(N) version of PBE                    */ 
#define XC_GGA_X_LAMBDA_OC2_N  40 /* lambda_OC2(N) version of PBE                   */ 
#define XC_GGA_X_BCGP          38 /* Burke, Cancio, Gould, and Pittalis             */ 
#define XC_GGA_X_PBEFE        265 /* PBE for formation energies                     */ 
 
#define XC_GGA_K_APBE         185 /* mu fixed from the semiclassical neutral atom   */ 
#define XC_GGA_K_TW1          187 /* Tran and Wesolowski set 1 (Table II)           */ 
#define XC_GGA_K_TW2          188 /* Tran and Wesolowski set 2 (Table II)           */ 
#define XC_GGA_K_TW3          189 /* Tran and Wesolowski set 3 (Table II)           */ 
#define XC_GGA_K_TW4          190 /* Tran and Wesolowski set 4 (Table II)           */ 
#define XC_GGA_K_REVAPBE       55 /* revised APBE                                   */ 
 
 
typedef struct{ 
  double kappa, mu; 
 
  /* parameter used in the Odashima & Capelle versions */ 
  double lambda; 
} gga_x_pbe_params; 
 
 
static void  
gga_x_pbe_init(xc_func_type *p) 
{ 
  gga_x_pbe_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_x_pbe_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_x_pbe_params); 
#endif 
  params = (gga_x_pbe_params *) (p->params); 
  
  params->lambda = 0.0; 
 
  switch(p->info->number){ 
  case XC_GGA_X_PBE: 
    /* PBE: mu = beta*pi^2/3, beta = 0.06672455060314922 */ 
    xc_gga_x_pbe_set_params(p, 0.8040, 0.2195149727645171); 
    break; 
  case XC_GGA_X_PBE_R: 
    xc_gga_x_pbe_set_params(p, 1.245, 0.2195149727645171); 
    break; 
  case XC_GGA_X_PBE_SOL: 
    xc_gga_x_pbe_set_params(p, 0.804, MU_GE); 
    break; 
  case XC_GGA_X_XPBE: 
    xc_gga_x_pbe_set_params(p, 0.91954, 0.23214); 
    break; 
  case XC_GGA_X_PBE_JSJR: 
    xc_gga_x_pbe_set_params(p, 0.8040, 0.046*M_PI*M_PI/3.0); 
    break; 
  case XC_GGA_X_PBEK1_VDW: 
    xc_gga_x_pbe_set_params(p, 1.0, 0.2195149727645171); 
    break; 
  case XC_GGA_X_APBE: 
    xc_gga_x_pbe_set_params(p, 0.8040, 0.260); 
    break; 
  case XC_GGA_K_APBE: 
    xc_gga_x_pbe_set_params(p, 0.8040, 0.23889); 
    break; 
  case XC_GGA_K_TW1: 
    xc_gga_x_pbe_set_params(p, 0.8209, 0.2335); 
    break; 
  case XC_GGA_K_TW2: 
    xc_gga_x_pbe_set_params(p, 0.6774, 0.2371); 
    break; 
  case XC_GGA_K_TW3: 
    xc_gga_x_pbe_set_params(p, 0.8438, 0.2319); 
    break; 
  case XC_GGA_K_TW4: 
    xc_gga_x_pbe_set_params(p, 0.8589, 0.2309); 
    break; 
  case XC_GGA_X_PBE_TCA: 
    xc_gga_x_pbe_set_params(p, 1.227, 0.2195149727645171); 
    break; 
  case XC_GGA_K_REVAPBE: 
    xc_gga_x_pbe_set_params(p, 1.245, 0.23889); 
    break; 
  case XC_GGA_X_PBE_MOL: 
    xc_gga_x_pbe_set_params(p, 0.8040, 0.27583); 
    break; 
  case XC_GGA_X_LAMBDA_LO_N: 
    xc_gga_x_pbe_set_params(p, -1.0, 0.2195149727645171); 
    params->lambda = 2.273; 
    break; 
  case XC_GGA_X_LAMBDA_CH_N: 
    xc_gga_x_pbe_set_params(p, -1.0, 0.2195149727645171); 
    params->lambda = 2.215; 
    break; 
  case XC_GGA_X_LAMBDA_OC2_N: 
    xc_gga_x_pbe_set_params(p, -1.0, 0.2195149727645171); 
    params->lambda = 2.00; 
    break; 
  case XC_GGA_X_BCGP: 
    xc_gga_x_pbe_set_params(p, 0.8040, 0.249); 
    break; 
  case XC_GGA_X_PBEFE: 
    xc_gga_x_pbe_set_params(p, 0.437, 0.346); 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_x_pbe\n"); 
    exit(1); 
  } 
} 
 
 
void  
xc_gga_x_pbe_set_params(xc_func_type *p, double kappa, double mu) 
{ 
  gga_x_pbe_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_pbe_params *) (p->params); 
 
  params->kappa = kappa; 
  params->mu    = mu; 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_x_pbe.c" 
#endif 
 
#define func xc_gga_x_pbe_enhance 
#include "work_gga_x.c" 
 
 
const xc_func_info_type xc_func_info_gga_x_pbe = { 
  XC_GGA_X_PBE, 
  XC_EXCHANGE, 
  "Perdew, Burke & Ernzerhof", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew1996_3865, &xc_ref_Perdew1996_3865_err, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbe_r = { 
  XC_GGA_X_PBE_R, 
  XC_EXCHANGE, 
  "Revised PBE from Zhang & Yang", 
  XC_FAMILY_GGA, 
  {&xc_ref_Zhang1998_890, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbe_sol = { 
  XC_GGA_X_PBE_SOL, 
  XC_EXCHANGE, 
  "Perdew, Burke & Ernzerhof SOL", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew2008_136406, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_xpbe = { 
  XC_GGA_X_XPBE, 
  XC_EXCHANGE, 
  "Extended PBE by Xu & Goddard III", 
  XC_FAMILY_GGA, 
  {&xc_ref_Xu2004_4068, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbe_jsjr = { 
  XC_GGA_X_PBE_JSJR, 
  XC_EXCHANGE, 
  "Reparametrized PBE by Pedroza, Silva & Capelle", 
  XC_FAMILY_GGA, 
  {&xc_ref_Pedroza2009_201106, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbek1_vdw = { 
  XC_GGA_X_PBEK1_VDW, 
  XC_EXCHANGE, 
  "Reparametrized PBE for vdW", 
  XC_FAMILY_GGA, 
  {&xc_ref_Klimes2010_022201, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_apbe = { 
  XC_GGA_X_APBE, 
  XC_EXCHANGE, 
  "mu fixed from the semiclassical neutral atom", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbe_tca = { 
  XC_GGA_X_PBE_TCA, 
  XC_EXCHANGE, 
  "PBE revised by Tognetti et al", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tognetti2008_536, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
static const func_params_type ext_params[] = { 
  {1e23, "Number of electrons"}, 
}; 
 
static void  
set_ext_params(xc_func_type *p, const double *ext_params) 
{ 
  const double lambda_1 = 1.48; 
 
  gga_x_pbe_params *params; 
  double lambda, ff; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_pbe_params *) (p->params); 
 
  ff = (ext_params == NULL) ? p->info->ext_params[0].value : ext_params[0]; 
 
  lambda = (1.0 - 1.0/ff)*params->lambda + lambda_1/ff; 
  params->kappa = lambda/M_CBRT2 - 1.0; 
} 
 
 
const xc_func_info_type xc_func_info_gga_x_lambda_lo_n = { 
  XC_GGA_X_LAMBDA_LO_N, 
  XC_EXCHANGE, 
  "lambda_LO(N) version of PBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Odashima2009_798, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  1, ext_params, set_ext_params, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_lambda_ch_n = { 
  XC_GGA_X_LAMBDA_CH_N, 
  XC_EXCHANGE, 
  "lambda_CH(N) version of PBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Odashima2009_798, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  1, ext_params, set_ext_params, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_lambda_oc2_n = { 
  XC_GGA_X_LAMBDA_OC2_N, 
  XC_EXCHANGE, 
  "lambda_OC2(N) version of PBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Odashima2009_798, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  1, ext_params, set_ext_params, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbe_mol = { 
  XC_GGA_X_PBE_MOL, 
  XC_EXCHANGE, 
  "Reparametrized PBE by del Campo, Gazquez, Trickey & Vela", 
  XC_FAMILY_GGA, 
  {&xc_ref_delCampo2012_104108, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_bcgp = { 
  XC_GGA_X_BCGP, 
  XC_EXCHANGE, 
  "Burke, Cancio, Gould, and Pittalis", 
  XC_FAMILY_GGA, 
  {&xc_ref_Burke2014_4834, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_x_pbefe = { 
  XC_GGA_X_PBEFE, 
  XC_EXCHANGE, 
  "PBE for formation energies", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perez2015_3844, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_x, NULL 
}; 
 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_k_apbe = { 
  XC_GGA_K_APBE, 
  XC_KINETIC, 
  "mu fixed from the semiclassical neutral atom", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_revapbe = { 
  XC_GGA_K_REVAPBE, 
  XC_KINETIC, 
  "revised APBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_tw1 = { 
  XC_GGA_K_TW1, 
  XC_KINETIC, 
  "Tran and Wesolowski set 1 (Table II)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_tw2 = { 
  XC_GGA_K_TW2, 
  XC_KINETIC, 
  "Tran and Wesolowski set 2 (Table II)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_tw3 = { 
  XC_GGA_K_TW3, 
  XC_KINETIC, 
  "Tran and Wesolowski set 3 (Table II)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_tw4 = { 
  XC_GGA_K_TW4, 
  XC_KINETIC, 
  "Tran and Wesolowski set 4 (Table II)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_pbe_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
