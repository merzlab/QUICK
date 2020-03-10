/* 
 Copyright (C) 2017 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_LDA_X_REL   532   /* Relativistic exchange        */ 
 
#ifndef DEVICE 
#include "maple2c/lda_x_rel.c" 
#endif 
 
#define func maple2c_func 
#include "work_lda.c" 
 
const xc_func_info_type xc_func_info_lda_x_rel = { 
  XC_LDA_X_REL, 
  XC_EXCHANGE, 
  "Slater exchange with relativistic corrections", 
  XC_FAMILY_LDA, 
  {&xc_ref_Rajagopal1978_L943, &xc_ref_MacDonald1979_2977, &xc_ref_Engel1995_2750, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-24, 
  0, NULL, NULL, 
  NULL, NULL,  
  work_lda, NULL, NULL 
}; 
