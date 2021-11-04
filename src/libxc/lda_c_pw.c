/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
/************************************************************************ 
 Correlation energy per-particle and potential of a HEG as parameterized  
 by  
   J.P. Perdew & Y. Wang 
   Ortiz & Ballone 
 
Note that the PW modified corresponds to the version of PW used in the  
original PBE routine. This amounts to adding some more digits in some of 
the constants of PW. 
************************************************************************/ 
 
#define XC_LDA_C_PW     12   /* Perdew & Wang                */ 
#define XC_LDA_C_PW_MOD 13   /* Perdew & Wang (Modified)     */ 
#define XC_LDA_C_OB_PW  14   /* Ortiz & Ballone (PW)         */ 
#define XC_LDA_C_PW_RPA 25   /* Perdew & Wang fit of the RPA */ 
 
typedef struct { 
  double pp[3], a[3], alpha1[3]; 
  double beta1[3], beta2[3], beta3[3], beta4[3]; 
  double fz20; 
} lda_c_pw_params; 
 
static const lda_c_pw_params par_pw = { 
  {1.0,  1.0,  1.0}, 
  {0.031091,  0.015545,   0.016887}, 
  {0.21370,  0.20548,  0.11125}, 
  {7.5957, 14.1189, 10.357}, 
  {3.5876, 6.1977, 3.6231}, 
  {1.6382, 3.3662, 0.88026}, 
  {0.49294, 0.62517, 0.49671}, 
  1.709921 
}; 
 
static const lda_c_pw_params par_pw_mod = { 
  {1.0,  1.0,  1.0}, 
  {0.0310907, 0.01554535, 0.0168869}, 
  {0.21370,  0.20548,  0.11125}, 
  {7.5957, 14.1189, 10.357}, 
  {3.5876, 6.1977, 3.6231}, 
  {1.6382, 3.3662,  0.88026}, 
  {0.49294, 0.62517, 0.49671}, 
  1.709920934161365617563962776245 
}; 
 
static const lda_c_pw_params par_ob = { 
  {1.0,  1.0,  1.0}, 
  {0.031091,  0.015545, 0.016887}, 
  {0.026481, 0.022465, 0.11125}, 
  {7.5957, 14.1189, 10.357}, 
  {3.5876, 6.1977, 3.6231}, 
  {-0.46647, -0.56043, 0.88026}, 
  {0.13354, 0.11313, 0.49671}, 
  1.709921 
}; 
 
static const lda_c_pw_params par_pw_rpa = { 
  {0.75, 0.75, 1.0}, 
  {0.031091,  0.015545,   0.016887}, 
  {0.082477, 0.035374, 0.028829}, 
  { 5.1486, 6.4869, 10.357}, 
  {1.6483, 1.3083, 3.6231}, 
  {0.23647, 0.15180, 0.47990}, 
  {0.20614, 0.082349, 0.12279}, 
  1.709921 
}; 
 
static void  
lda_c_pw_init(xc_func_type *p) 
{   
  lda_c_pw_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(lda_c_pw_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(lda_c_pw_params); 
#endif 
  params = (lda_c_pw_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_LDA_C_PW: 
    memcpy(params, &par_pw, sizeof(lda_c_pw_params)); 
    break; 
  case XC_LDA_C_PW_MOD: 
    memcpy(params, &par_pw_mod, sizeof(lda_c_pw_params)); 
    break; 
  case XC_LDA_C_OB_PW: 
    memcpy(params, &par_ob, sizeof(lda_c_pw_params)); 
    break; 
  case XC_LDA_C_PW_RPA: 
    memcpy(params, &par_pw_rpa, sizeof(lda_c_pw_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in lda_c_pw\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/lda_c_pw.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_c_pw = { 
  XC_LDA_C_PW, 
  XC_CORRELATION, 
  "Perdew & Wang", 
  XC_FAMILY_LDA, 
  {&xc_ref_Perdew1992_13244, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pw_init, /* init */ 
  NULL,     /* end  */ 
  work_lda, /* lda  */ 
  NULL, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_pw_mod = { 
  XC_LDA_C_PW_MOD, 
  XC_CORRELATION, 
  "Perdew & Wang (modified)", 
  XC_FAMILY_LDA, 
  {&xc_ref_Perdew1992_13244_mod, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pw_init, /* init */ 
  NULL,     /* end  */ 
  work_lda, /* lda  */ 
  NULL, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_ob_pw = { 
  XC_LDA_C_OB_PW, 
  XC_CORRELATION, 
  "Ortiz & Ballone (PW parametrization)", 
  XC_FAMILY_LDA, 
  {&xc_ref_Ortiz1994_1391, &xc_ref_Ortiz1994_1391_err, &xc_ref_Perdew1992_13244_mod, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pw_init, /* init */ 
  NULL,     /* end  */ 
  work_lda, /* lda  */ 
  NULL, 
  NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_pw_rpa = { 
  XC_LDA_C_PW_RPA, 
  XC_CORRELATION, 
  "Perdew & Wang (fit to the RPA energy)", 
  XC_FAMILY_LDA, 
  {&xc_ref_Perdew1992_13244, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pw_init, /* init */ 
  NULL,     /* end  */ 
  work_lda, /* lda  */ 
  NULL, 
  NULL 
}; 
