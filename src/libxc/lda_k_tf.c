/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_LDA_K_TF      50   /* Thomas-Fermi kinetic energy functional */
#define XC_LDA_K_LP      51   /* Lee and Parr Gaussian ansatz           */

typedef struct {
  double ax;
} lda_k_tf_params;

static void 
lda_k_tf_init(xc_func_type *p)
{
  lda_k_tf_params *params;

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(lda_k_tf_params));
  params = (lda_k_tf_params *) (p->params);

  switch(p->info->number){
  case XC_LDA_K_TF:
    /* 3/10*(3*M_PI^2)^(2/3) * (3/4 pi)^(2/3) = 3/10*pow(9*M_PI/4, 2/3) */
    params->ax = 1.104950565705860002098832079519635692942;
    break;
  case XC_LDA_K_LP:
    /* 3*M_PI/2^(5/3) * (3/4 pi)^(2/3) = 3*M_PI*pow(3/(8*M_PI), 2/3)*/
    params->ax = 1.142427709758666675644309251677891925671;
    break;
  default:
    fprintf(stderr, "Internal error in lda_k_tf\n");
    exit(1);
  }
}

#include "maple2c/lda_k_tf.c"

#define func maple2c_func
#include "work_lda.c"

const xc_func_info_type xc_func_info_lda_k_tf = {
  XC_LDA_K_TF,
  XC_KINETIC,
  "Thomas-Fermi kinetic energy",
  XC_FAMILY_LDA,
  {&xc_ref_Thomas1927_542, &xc_ref_Fermi1927_602, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-24,
  0, NULL, NULL,
  lda_k_tf_init, NULL,
  work_lda, NULL, NULL
};

const xc_func_info_type xc_func_info_lda_k_lp = {
  XC_LDA_K_LP,
  XC_KINETIC,
  "Lee and Parr Gaussian ansatz for the kinetic energy",
  XC_FAMILY_LDA,
  {&xc_ref_Lee1987_2377, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-24,
  0, NULL, NULL,
  lda_k_tf_init, NULL,
  work_lda, NULL, NULL
};

