/* 
 Copyright (C) 2006-2016 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_LDA_C_CHACHIYO  287   /* Chachiyo simple 2 parameter correlation   */ 
#define XC_LDA_C_KARASIEV  579   /* Karasiev reparameterization of Chachiyo   */ 
 
typedef struct { 
  double ap, bp, af, bf; 
} lda_c_chachiyo_params; 
 
static lda_c_chachiyo_params par_chachiyo = {-0.01554535, 20.4562557, -0.007772675, 27.4203609}; 
static lda_c_chachiyo_params par_karasiev = {-0.01554535, 21.7392245, -0.007772675, 28.3559732}; 
 
static void  
lda_c_chachiyo_init(xc_func_type *p) 
{ 
  lda_c_chachiyo_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(lda_c_chachiyo_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(lda_c_chachiyo_params); 
#endif 
  params = (lda_c_chachiyo_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_LDA_C_CHACHIYO: 
    memcpy(params, &par_chachiyo, sizeof(lda_c_chachiyo_params)); 
    break; 
  case XC_LDA_C_KARASIEV: 
    memcpy(params, &par_karasiev, sizeof(lda_c_chachiyo_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in lda_c_chachiyo\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/lda_c_chachiyo.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_c_chachiyo = { 
  XC_LDA_C_CHACHIYO, 
  XC_CORRELATION, 
  "Chachiyo simple 2 parameter correlation", 
  XC_FAMILY_LDA, 
  {&xc_ref_Chachiyo2016_021101, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_chachiyo_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_karasiev = { 
  XC_LDA_C_KARASIEV, 
  XC_CORRELATION, 
  "Karasiev reparameterization of Chachiyo", 
  XC_FAMILY_LDA, 
  {&xc_ref_Karasiev2016_157101, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_chachiyo_init, NULL, 
  work_lda, NULL, NULL 
}; 
