/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

#define XC_MGGA_X_BR89_EXPLICIT  586 /* Becke-Roussel 89 with an explicit inversion of x(y) */

#include "maple2c/mgga_x_br89_explicit.c"

#define func maple2c_func
#include "work_mgga_x.c"

const xc_func_info_type xc_func_info_mgga_x_br89_explicit = {
  XC_MGGA_X_BR89_EXPLICIT,
  XC_EXCHANGE,
  "Becke-Roussel 89",
  XC_FAMILY_MGGA,
  {&xc_ref_Becke1989_3761, &xc_ref_Proynov2008_103, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1.0e-12,
  0, NULL, NULL,
  NULL, NULL,
  NULL, NULL, work_mgga_x,
};
