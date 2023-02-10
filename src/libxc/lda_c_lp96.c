/* 
 Copyright (C) 2017 Susi Lehtola 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_LDA_C_LP96      289   /* Liu-Parr correlation */ 
#define XC_LDA_K_LP96      580   /* Liu-Parr kinetic */ 
 
typedef struct { 
  double C1, C2, C3; 
} lda_c_lp96_params; 
 
static lda_c_lp96_params c_lp96 = {-0.0603,   0.0175, -0.00053}; 
static lda_c_lp96_params k_lp96 = { 0.03777, -0.01002, 0.00039}; 
 
static void  
lda_c_lp96_init(xc_func_type *p) 
{ 
  lda_c_lp96_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(lda_c_lp96_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(lda_c_lp96_params); 
#endif 
  params = (lda_c_lp96_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_LDA_C_LP96: 
    memcpy(params, &c_lp96, sizeof(lda_c_lp96_params)); 
    break; 
  case XC_LDA_K_LP96: 
    memcpy(params, &k_lp96, sizeof(lda_c_lp96_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in lda_c_lp96\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/lda_c_lp96.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_c_lp96 = { 
  XC_LDA_C_LP96, 
  XC_CORRELATION, 
  "Liu-Parr correlation", 
  XC_FAMILY_LDA, 
  {&xc_ref_Liu1996_2211, &xc_ref_Liu2000_29, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-16, 
  0, NULL, NULL, 
  lda_c_lp96_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_k_lp96 = { 
  XC_LDA_K_LP96, 
  XC_KINETIC, 
  "Liu-Parr kinetic", 
  XC_FAMILY_LDA, 
  {&xc_ref_Liu1996_2211, &xc_ref_Liu2000_29, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-16, 
  0, NULL, NULL, 
  lda_c_lp96_init, NULL, 
  work_lda, NULL, NULL 
}; 
