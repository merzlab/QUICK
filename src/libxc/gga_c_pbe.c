/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
/************************************************************************ 
 Implements Perdew, Burke & Ernzerhof Generalized Gradient Approximation 
 correlation functional. 
 
 I based this implementation on a routine from L.C. Balbas and J.M. Soler 
************************************************************************/ 
 
#define XC_GGA_C_PBE          130 /* Perdew, Burke & Ernzerhof correlation              */ 
#define XC_GGA_C_PBE_SOL      133 /* Perdew, Burke & Ernzerhof correlation SOL          */ 
#define XC_GGA_C_XPBE         136 /* xPBE reparametrization by Xu & Goddard             */ 
#define XC_GGA_C_PBE_JRGX     138 /* JRGX reparametrization by Pedroza, Silva & Capelle */ 
#define XC_GGA_C_RGE2         143 /* Regularized PBE                                    */ 
#define XC_GGA_C_APBE         186 /* mu fixed from the semiclassical neutral atom       */ 
#define XC_GGA_C_SPBE          89 /* PBE correlation to be used with the SSB exchange   */ 
#define XC_GGA_C_PBEINT        62 /* PBE for hybrid interfaces                          */ 
#define XC_GGA_C_PBEFE        258 /* PBE for formation energies                         */ 
#define XC_GGA_C_PBE_MOL      272 /* Del Campo, Gazquez, Trickey and Vela (PBE-like)    */ 
#define XC_GGA_C_TM_PBE       560  /* Thakkar and McCarthy reparametrization */ 
 
typedef struct{ 
  double beta, gamma, BB; 
} gga_c_pbe_params; 
 
 
static void gga_c_pbe_init(xc_func_type *p) 
{ 
  gga_c_pbe_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(gga_c_pbe_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(gga_c_pbe_params); 
#endif 
  params = (gga_c_pbe_params *) (p->params); 
 
  /* most functionals have the same gamma and B */ 
  params->gamma = (1.0 - log(2.0))/(M_PI*M_PI); 
  params->BB = 1.0;  
 
  switch(p->info->number){ 
  case XC_GGA_C_PBE: 
    params->beta = 0.06672455060314922; 
    break; 
  case XC_GGA_C_PBE_SOL: 
    params->beta = 0.046; 
    break; 
  case XC_GGA_C_XPBE: 
    params->beta  = 0.089809; 
    params->gamma = params->beta*params->beta/(2.0*0.197363); 
    break; 
  case XC_GGA_C_PBE_JRGX: 
    params->beta = 3.0*10.0/(81.0*M_PI*M_PI); 
    break; 
  case XC_GGA_C_RGE2: 
    params->beta = 0.053; 
    break; 
  case XC_GGA_C_APBE: 
    params->beta = 3.0*0.260/(M_PI*M_PI); 
    break; 
  case XC_GGA_C_SPBE: 
    params->beta = 0.06672455060314922; 
    /* the sPBE functional contains one term less than the original PBE, so we set it to zero */ 
    params->BB = 0.0;  
    break; 
  case XC_GGA_C_PBEINT: 
    params->beta = 0.052; 
    break; 
  case XC_GGA_C_PBEFE: 
    params->beta = 0.043; 
    break; 
  case XC_GGA_C_PBE_MOL: 
    params->beta = 0.08384; 
    break; 
  case XC_GGA_C_TM_PBE: 
    params->gamma = -0.0156; 
    params->beta  = 3.38*params->gamma; 
    break; 
  default: 
    fprintf(stderr, "Internal error in gga_c_pbe\n"); 
    exit(1); 
  } 
} 
 
 
void  
xc_gga_c_pbe_set_params(xc_func_type *p, double beta) 
{ 
  gga_c_pbe_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_c_pbe_params *) (p->params); 
 
  params->beta = beta; 
} 
 
 
#ifndef DEVICE 
#include "maple2c/gga_c_pbe.c" 
#endif 
 
#define func maple2c_func 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_c_pbe = { 
  XC_GGA_C_PBE, 
  XC_CORRELATION, 
  "Perdew, Burke & Ernzerhof", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew1996_3865, &xc_ref_Perdew1996_3865_err, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_pbe_sol = { 
  XC_GGA_C_PBE_SOL, 
  XC_CORRELATION, 
  "Perdew, Burke & Ernzerhof SOL", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perdew2008_136406, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_xpbe = { 
  XC_GGA_C_XPBE, 
  XC_CORRELATION, 
  "Extended PBE by Xu & Goddard III", 
  XC_FAMILY_GGA, 
  {&xc_ref_Xu2004_4068, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_pbe_jrgx = { 
  XC_GGA_C_PBE_JRGX, 
  XC_CORRELATION, 
  "Reparametrized PBE by Pedroza, Silva & Capelle", 
  XC_FAMILY_GGA, 
  {&xc_ref_Pedroza2009_201106, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_rge2 = { 
  XC_GGA_C_RGE2, 
  XC_CORRELATION, 
  "Regularized PBE", 
  XC_FAMILY_GGA, 
  {&xc_ref_Ruzsinszky2009_763, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_apbe = { 
  XC_GGA_C_APBE, 
  XC_CORRELATION, 
  "mu fixed from the semiclassical neutral atom", 
  XC_FAMILY_GGA, 
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_spbe = { 
  XC_GGA_C_SPBE, 
  XC_CORRELATION, 
  "PBE correlation to be used with the SSB exchange", 
  XC_FAMILY_GGA, 
  {&xc_ref_Swart2009_094103, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_pbeint = { 
  XC_GGA_C_PBEINT, 
  XC_CORRELATION, 
  "PBE for hybrid interfaces", 
  XC_FAMILY_GGA, 
  {&xc_ref_Fabiano2010_113104, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_pbefe = { 
  XC_GGA_C_PBEFE, 
  XC_CORRELATION, 
  "PBE for formation energies", 
  XC_FAMILY_GGA, 
  {&xc_ref_Perez2015_3844, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_pbe_mol = { 
  XC_GGA_C_PBE_MOL, 
  XC_CORRELATION, 
  "Reparametrized PBE by del Campo, Gazquez, Trickey & Vela", 
  XC_FAMILY_GGA, 
  {&xc_ref_delCampo2012_104108, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_c_tm_pbe = { 
  XC_GGA_C_TM_PBE, 
  XC_CORRELATION, 
  "Thakkar and McCarthy reparametrization", 
  XC_FAMILY_GGA, 
  {&xc_ref_Thakkar2009_134109, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-12, 
  0, NULL, NULL, 
  gga_c_pbe_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
