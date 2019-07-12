/*
 Copyright (C) 2006-2009 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_LDA_C_1D_LOOS          26 /* P-F Loos correlation LDA     */

#include "maple2c/lda_c_1d_loos.c"

#define func maple2c_func
#define XC_DIMENSIONS 1
#include "work_lda.c"

const xc_func_info_type xc_func_info_lda_c_1d_loos = {
  XC_LDA_C_1D_LOOS,
  XC_CORRELATION,
  "P-F Loos correlation LDA",
  XC_FAMILY_LDA,
  {&xc_ref_Loos2013_064108, NULL, NULL, NULL, NULL},
  XC_FLAGS_1D |  XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  5e-28,
  0, NULL, NULL,
  NULL, NULL,
  work_lda, NULL, NULL
};
