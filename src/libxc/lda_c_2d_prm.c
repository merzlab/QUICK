/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

/************************************************************************
Correlation functional by Pittalis, Rasanen & Marques for the 2D electron gas
************************************************************************/

/* TODO: convert this to an (rs, zeta) expression */

#define XC_LDA_C_2D_PRM  16   /* Pittalis, Rasanen & Marques correlation in 2D */

typedef struct{
  double N;
  double c;
} lda_c_2d_prm_params;

/* Initialization */
static void
lda_c_2d_prm_init(xc_func_type *p)
{
  assert(p != NULL && p->params == NULL);

  p->params = malloc(sizeof(lda_c_2d_prm_params));
}

#include "maple2c/lda_c_2d_prm.c"

#define func maple2c_func
#define XC_DIMENSIONS 2
#include "work_lda.c"

static const func_params_type ext_params[] = {
  {2.0, "Number of electrons"},
};

static void 
set_ext_params(xc_func_type *p, const double *ext_params)
{
  static double prm_q = 3.9274; /* 2.258 */
  lda_c_2d_prm_params *params;
  double ff;

  assert(p != NULL && p->params != NULL);
  params = (lda_c_2d_prm_params *) (p->params);

  ff = (ext_params == NULL) ? p->info->ext_params[0].value : ext_params[0];
  params->N = ff;

  if(params->N <= 1.0){
    fprintf(stderr, "PRM functional cannot be used for N_electrons <= 1\n");
    exit(1);
  }

  params->c = M_PI/(2.0*(params->N - 1.0)*prm_q*prm_q); /* Eq. (13) */
}

const xc_func_info_type xc_func_info_lda_c_2d_prm = {
  XC_LDA_C_2D_PRM,
  XC_CORRELATION,
  "PRM (for 2D systems)",
  XC_FAMILY_LDA,
  {&xc_ref_Pittalis2008_195322, NULL, NULL, NULL, NULL},
  XC_FLAGS_2D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  1, ext_params, set_ext_params,
  lda_c_2d_prm_init, NULL,
  work_lda, NULL, NULL
};
