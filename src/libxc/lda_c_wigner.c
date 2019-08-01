/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_LDA_C_WIGNER    2   /* Wigner parametrization       */ 
#define XC_LDA_XC_LP_A   547   /* Lee-Parr reparametrization B */ 
#define XC_LDA_XC_LP_B   548   /* Lee-Parr reparametrization B */ 
#define XC_LDA_C_MCWEENY 551   /* McWeeny 76 */ 
#define XC_LDA_C_BR78    552   /* Brual & Rothstein 78 */ 
#define XC_LDA_C_OW_LYP  573   /* Wigner with corresponding LYP parameters */ 
#define XC_LDA_C_OW      574   /* Optimized Wigner */ 
 
typedef struct { 
  double a, b; 
} lda_c_wigner_params; 
 
static void  
lda_c_wigner_init(xc_func_type *p) 
{ 
  lda_c_wigner_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(lda_c_wigner_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(lda_c_wigner_params); 
#endif 
  params = (lda_c_wigner_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_LDA_C_WIGNER: 
    params->a = -0.44; 
    params->b =  7.8; 
    break; 
  case XC_LDA_XC_LP_A: 
    params->a = -0.8626*RS_FACTOR; 
    params->b = 0.0; 
    break; 
  case XC_LDA_XC_LP_B: 
    params->a = -0.906*RS_FACTOR; 
    params->b =  2.1987e-2*RS_FACTOR; 
    break; 
  case XC_LDA_C_MCWEENY: 
    params->a = -RS_FACTOR/2.946; 
    params->b =  RS_FACTOR*9.652/2.946; 
    break; 
  case XC_LDA_C_BR78: 
    params->a = -RS_FACTOR/21.437; 
    params->b =  RS_FACTOR*9.810/21.437; 
    break; 
  case XC_LDA_C_OW_LYP: 
    params->a = -0.04918*RS_FACTOR/0.349; 
    params->b = RS_FACTOR/0.349; 
    break; 
  case XC_LDA_C_OW: 
    params->a = -0.0526*RS_FACTOR/0.349; 
    params->b = RS_FACTOR/0.349; 
    break; 
  default: 
    fprintf(stderr, "Internal error in lda_c_wigner\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/lda_c_wigner.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_c_wigner = { 
  XC_LDA_C_WIGNER, 
  XC_CORRELATION, 
  "Wigner", 
  XC_FAMILY_LDA, 
  {&xc_ref_Wigner1938_678, &xc_ref_Stewart1995_4337, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_xc_lp_a = { 
  XC_LDA_XC_LP_A, 
  XC_EXCHANGE_CORRELATION, 
  "Lee-Parr reparametrization A", 
  XC_FAMILY_LDA, 
  {&xc_ref_Lee1990_193, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_xc_lp_b = { 
  XC_LDA_XC_LP_B, 
  XC_EXCHANGE_CORRELATION, 
  "Lee-Parr reparametrization B", 
  XC_FAMILY_LDA, 
  {&xc_ref_Lee1990_193, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_mcweeny = { 
  XC_LDA_C_MCWEENY, 
  XC_CORRELATION, 
  "McWeeny 76", 
  XC_FAMILY_LDA, 
  {&xc_ref_McWeeny1976_3, &xc_ref_Brual1978_1177, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_br78 = { 
  XC_LDA_C_BR78, 
  XC_CORRELATION, 
  "Brual & Rothstein 78", 
  XC_FAMILY_LDA, 
  {&xc_ref_Brual1978_1177, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_ow_lyp = { 
  XC_LDA_C_OW_LYP, 
  XC_CORRELATION, 
  "Wigner with corresponding LYP parameters", 
  XC_FAMILY_LDA, 
  {&xc_ref_Stewart1995_4337, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_ow = { 
  XC_LDA_C_OW, 
  XC_CORRELATION, 
  "Optimized Wigner", 
  XC_FAMILY_LDA, 
  {&xc_ref_Stewart1995_4337, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_wigner_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
