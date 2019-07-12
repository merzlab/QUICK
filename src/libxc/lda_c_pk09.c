/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_LDA_C_PK09   554   /* Proynov and Kong 2009 */

#include "maple2c/lda_c_pk09.c"

#define func maple2c_func
#include "work_lda.c"

const xc_func_info_type xc_func_info_lda_c_pk09 = {
  XC_LDA_C_PK09,
  XC_CORRELATION,
  "Proynov and Kong 2009",
  XC_FAMILY_LDA,
  {&xc_ref_Proynov2009_014103, &xc_ref_Proynov2017_059904, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  5e-24,
  0, NULL, NULL,
  NULL, NULL,
  work_lda, NULL, NULL
};
