/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_LDA_X_2D  19 /* Exchange in 2D */

#include "maple2c/lda_x_2d.c"

#define func maple2c_func
#define XC_DIMENSIONS 2
#include "work_lda.c"

const xc_func_info_type xc_func_info_lda_x_2d = {
  XC_LDA_X_2D,
  XC_EXCHANGE,
  "Slater exchange",
  XC_FAMILY_LDA,
  {&xc_ref_Dirac1930_376, &xc_ref_Bloch1929_545, NULL, NULL, NULL},
  XC_FLAGS_2D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-24,
  0, NULL, NULL,
  NULL, NULL,
  work_lda, NULL,  NULL
};

