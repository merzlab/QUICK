/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
/************************************************************************ 
 Correlation energy per particle and potential of a HEG as parametrized  
 by  
   Perdew & Zunger 
   Ortiz & Ballone 
************************************************************************/ 
 
#define XC_LDA_C_PZ       9   /* Perdew & Zunger              */ 
#define XC_LDA_C_PZ_MOD  10   /* Perdew & Zunger (Modified)   */ 
#define XC_LDA_C_OB_PZ   11   /* Ortiz & Ballone (PZ)         */ 
 
typedef struct { 
  double gamma[2]; 
  double beta1[2]; 
  double beta2[2]; 
  double a[2], b[2], c[2], d[2]; 
} lda_c_pz_params; 
 
static lda_c_pz_params pz_original = { 
  {-0.1423, -0.0843},  /* gamma */ 
  { 1.0529,  1.3981},  /* beta1 */ 
  { 0.3334,  0.2611},  /* beta2 */ 
  { 0.0311,  0.01555}, /*  a    */ 
  {-0.048,  -0.0269},  /*  b    */ 
  { 0.0020,  0.0007},  /*  c    */ 
  {-0.0116, -0.0048}   /*  d    */ 
}; 
 
static lda_c_pz_params pz_modified = { 
  {-0.1423, -0.0843},    
  { 1.0529,  1.3981},  
  { 0.3334,  0.2611},  
  { 0.0311,  0.01555}, 
  {-0.048,  -0.0269},    
  { 0.0020191519406228,  0.00069255121311694}, 
  {-0.0116320663789130, -0.00480126353790614} 
}; 
 
static lda_c_pz_params pz_ob = { 
  {-0.103756, -0.065951}, 
  { 0.56371,   1.11846}, 
  { 0.27358,   0.18797}, 
  { 0.031091,  0.015545}, 
  {-0.046644, -0.025599}, 
  { 0.00419,   0.00329},  /* the sign of c[0] and c[1] is different from [2], but is consistent 
                             with the continuity requirement. There is nothing in [3] about this. */ 
  {-0.00983,  -0.00300} 
}; 
 
static void  
lda_c_pz_init(xc_func_type *p) 
{ 
  lda_c_pz_params *params; 
 
  assert(p!=NULL && p->params == NULL); 
  p->params = malloc(sizeof(lda_c_pz_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(lda_c_pz_params); 
#endif 
  params = (lda_c_pz_params *) (p->params); 
 
  switch(p->info->number){ 
  case XC_LDA_C_PZ: 
    memcpy(params, &pz_original, sizeof(lda_c_pz_params)); 
    break; 
  case XC_LDA_C_PZ_MOD: 
    memcpy(params, &pz_modified, sizeof(lda_c_pz_params)); 
    break; 
  case XC_LDA_C_OB_PZ: 
    memcpy(params, &pz_ob, sizeof(lda_c_pz_params)); 
    break; 
  default: 
    fprintf(stderr, "Internal error in lda_c_pz\n"); 
    exit(1); 
  } 
} 
 
#ifndef DEVICE 
#include "maple2c/lda_c_pz.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_c_pz = { 
  XC_LDA_C_PZ, 
  XC_CORRELATION, 
  "Perdew & Zunger", 
  XC_FAMILY_LDA, 
  {&xc_ref_Perdew1981_5048, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pz_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_pz_mod = { 
  XC_LDA_C_PZ_MOD, 
  XC_CORRELATION, 
  "Perdew & Zunger (Modified)", 
  XC_FAMILY_LDA, 
  {&xc_ref_Perdew1981_5048_mod, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pz_init, NULL, 
  work_lda, NULL, NULL 
}; 
 
const xc_func_info_type xc_func_info_lda_c_ob_pz = { 
  XC_LDA_C_OB_PZ, 
  XC_CORRELATION, 
  "Ortiz & Ballone (PZ parametrization)", 
  XC_FAMILY_LDA, 
  {&xc_ref_Ortiz1994_1391, &xc_ref_Ortiz1994_1391_err, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  lda_c_pz_init, NULL, 
  work_lda, NULL, NULL 
}; 
