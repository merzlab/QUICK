/* 
 Copyright (C) 2006-2009 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
/* these functionals is for the soft-Coulomb interaction with alpha=1 */ 
 
#define XC_LDA_XC_1D_EHWLRG_1     536 /* LDA constructed from slab-like systems of 1 electron  */ 
#define XC_LDA_XC_1D_EHWLRG_2     537 /* LDA constructed from slab-like systems of 2 electrons */ 
#define XC_LDA_XC_1D_EHWLRG_3     538 /* LDA constructed from slab-like systems of 3 electrons */ 
 
typedef struct { 
  double alpha; 
  double a1, a2, a3; 
} lda_xc_1d_ehwlrg_params; 
 
static void  
lda_xc_1d_ehwlrg_init(xc_func_type *p) 
{ 
  lda_xc_1d_ehwlrg_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(lda_xc_1d_ehwlrg_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(lda_xc_1d_ehwlrg_params); 
#endif 
  params = (lda_xc_1d_ehwlrg_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_LDA_XC_1D_EHWLRG_1: 
    params->alpha =  0.638; 
    params->a1    = -0.803; 
    params->a2    =  0.82; 
    params->a3    = -0.47; 
    break; 
  case XC_LDA_XC_1D_EHWLRG_2: 
    params->alpha =  0.604; 
    params->a1    = -0.74; 
    params->a2    =  0.68; 
    params->a3    = -0.38; 
    break; 
  case XC_LDA_XC_1D_EHWLRG_3: 
    params->alpha =  0.61; 
    params->a1    = -0.77; 
    params->a2    =  0.79; 
    params->a3    = -0.48; 
    break; 
  } 
} 
 
#define XC_DIMENSIONS 1 
 
#ifndef DEVICE 
#include "maple2c/lda_xc_1d_ehwlrg.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_xc_1d_ehwlrg_1 = { 
  XC_LDA_XC_1D_EHWLRG_1, 
  XC_EXCHANGE_CORRELATION, 
  "LDA constructed from slab-like systems of 1 electron", 
  XC_FAMILY_LDA, 
  {&xc_ref_Entwistle2016_205134, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_1D |  XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  lda_xc_1d_ehwlrg_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_xc_1d_ehwlrg_2 = { 
  XC_LDA_XC_1D_EHWLRG_2, 
  XC_EXCHANGE_CORRELATION, 
  "LDA constructed from slab-like systems of 2 electrons", 
  XC_FAMILY_LDA, 
  {&xc_ref_Entwistle2016_205134, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_1D |  XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  lda_xc_1d_ehwlrg_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
 
const xc_func_info_type xc_func_info_lda_xc_1d_ehwlrg_3 = { 
  XC_LDA_XC_1D_EHWLRG_3, 
  XC_EXCHANGE_CORRELATION, 
  "LDA constructed from slab-like systems of 3 electrons", 
  XC_FAMILY_LDA, 
  {&xc_ref_Entwistle2016_205134, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_1D |  XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  lda_xc_1d_ehwlrg_init, NULL, 
  work_lda, NULL, NULL 
}; 
