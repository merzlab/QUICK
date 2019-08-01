/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_2D_PBE          129 /* Perdew, Burke & Ernzerhof exchange in 2D          */ 
 
#ifndef DEVICE 
#include "maple2c/gga_x_2d_pbe.c" 
#endif 
 
#define func maple2c_func 
#define XC_DIMENSIONS 2 
#include "work_gga_x.c" 
 
const xc_func_info_type xc_func_info_gga_x_2d_pbe = { 
  XC_GGA_X_2D_PBE, 
  XC_EXCHANGE, 
  "Perdew, Burke & Ernzerhof in 2D", 
  XC_FAMILY_GGA, 
  {&xc_ref_Vilhena2014, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_2D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  NULL, NULL,  
  NULL, work_gga_x, NULL 
}; 
