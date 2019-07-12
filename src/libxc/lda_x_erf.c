/*
 Copyright (C) 2017 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

#define XC_LDA_X_ERF   546   /* Attenuated exchange LDA (erf) */

static void lda_x_erf_init(xc_func_type *p)
{
  /* initialize omega to something reasonable */
  p->cam_omega = 0.3;
}


#include "maple2c/lda_x_erf.c"

#define func maple2c_func
#include "work_lda.c"

const xc_func_info_type xc_func_info_lda_x_erf = {
  XC_LDA_X_ERF,
  XC_EXCHANGE,
  "Attenuated exchange LDA (erf)",
  XC_FAMILY_LDA,
  {&xc_ref_Toulouse2004_1047, &xc_ref_Tawada2004_8425, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-13,
  0, NULL, NULL,
  lda_x_erf_init, NULL, 
  work_lda, NULL, NULL
};
