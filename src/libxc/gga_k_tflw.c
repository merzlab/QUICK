/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
/* for a review on the values of lambda and gamma, please see EV 
Ludena and VV Karasiev, in "Reviews of Modern Quantum Chemistry: a 
Celebration of the Contributions of Robert G. Parr, edited by KD Sen 
(World Scientific, Singapore, 2002), p. 612. 
 */ 
 
#define XC_GGA_K_TFVW          52  /* Thomas-Fermi plus von Weiszaecker correction */ 
#define XC_GGA_K_VW            500 /* von Weiszaecker functional */ 
#define XC_GGA_K_GE2           501 /* Second-order gradient expansion (l = 1/9) */ 
#define XC_GGA_K_GOLDEN        502 /* TF-lambda-vW form by Golden (l = 13/45) */ 
#define XC_GGA_K_YT65          503 /* TF-lambda-vW form by Yonei and Tomishima (l = 1/5) */ 
#define XC_GGA_K_BALTIN        504 /* TF-lambda-vW form by Baltin (l = 5/9) */ 
#define XC_GGA_K_LIEB          505 /* TF-lambda-vW form by Lieb (l = 0.185909191) */ 
#define XC_GGA_K_ABSP1         506 /* gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)] */ 
#define XC_GGA_K_ABSP2         507 /* gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)] */ 
#define XC_GGA_K_ABSP3         277 /* gamma-TFvW form by Acharya et al [g = 1 - 1.513/N^0.35] */ 
#define XC_GGA_K_ABSP4         278 /* gamma-TFvW form by Acharya et al [g = l = 1/(1 + 1.332/N^(1/3))] */ 
#define XC_GGA_K_GR            508 /* gamma-TFvW form by Gazquez and Robles */ 
#define XC_GGA_K_LUDENA        509 /* gamma-TFvW form by Ludena */ 
#define XC_GGA_K_GP85          510 /* gamma-TFvW form by Ghosh and Parr */ 
 
typedef struct{ 
  double gamma, lambda; 
} gga_k_tflw_params; 
 
 
/* for automatically assigning lambda and gamma set them to -1 */ 
static void  
gga_k_tflw_set_params(xc_func_type *p, double gamma, double lambda, double N) 
{ 
  gga_k_tflw_params *params; 
  double C0 = CBRT(M_PI/3.0); 
  double C1 = CBRT(M_PI*M_PI/36.0)/6.0 - CBRT(M_PI*M_PI/9.0)/4.0; 
   
  assert(p != NULL && p->params != NULL); 
  params = (gga_k_tflw_params *) (p->params); 
 
  params->gamma = 1.0; 
  if(gamma > 0.0){ 
    params->gamma = gamma; 
  }else if(N > 0.0){ 
    switch(p->info->number){ 
    case XC_GGA_K_TFVW: 
      params->gamma = 1.0; 
      break; 
    case XC_GGA_K_VW: 
      params->gamma = 0.0; 
      break; 
    case XC_GGA_K_ABSP1:      /* Ref. 79 */ 
      params->gamma = 1.0 - 1.412/CBRT(N); 
      break; 
    case XC_GGA_K_ABSP2:      /* Ref. 79 */ 
      params->gamma = 1.0 - 1.332/CBRT(N); 
      break; 
    case XC_GGA_K_ABSP3:      /* Ref. 79 */ 
      params->gamma = 1.0 - 1.513/pow(N, 0.35); 
      break; 
    case XC_GGA_K_ABSP4:      /* Ref. 79 */ 
      params->gamma = 1.0/(1.0 + 1.332/CBRT(N)); 
      break; 
    case XC_GGA_K_GR:         /* Ref. 80 */ 
      params->gamma = (1.0 - 2.0/N)*(1.0 - C0/CBRT(N) + C1*CBRT(N*N)); 
      break; 
    case XC_GGA_K_LUDENA:     /* Ref. 82 */ 
      params->gamma = CBRT(6.0*M_PI)*M_PI*M_PI*(1.0 - 1.0/(N*N)); 
	break; 
    case XC_GGA_K_GP85:       /* Ref. 86 */ 
      params->gamma = CBRT(6.0*M_PI*M_PI)*M_PI*M_PI/4.0* 
	(1.0 - 1.0/N)*(1.0 + 1.0/N + 6.0/(N*N)); 
      break; 
    } 
  } 
 
  params->lambda = 1.0; 
  if(lambda > 0.0){ 
    params->lambda  = lambda; 
  }else{ 
    switch(p->info->number){ 
    case XC_GGA_K_TFVW: 
      params->lambda = 1.0; 
      break; 
    case XC_GGA_K_GE2: 
      params->lambda = 1.0/9.0; 
      break; 
    case XC_GGA_K_GOLDEN:     /* Ref. 33 */ 
      params->lambda = 13.0/45.0; 
      break; 
    case XC_GGA_K_YT65:       /* Ref. 57 */ 
      params->lambda = 1.0/5.0; 
      break; 
    case XC_GGA_K_BALTIN:     /* Ref. 66 */ 
      params->lambda = 5.0/9.0; 
      break; 
    case XC_GGA_K_LIEB:       /* Ref. 12 */ 
      params->lambda = 0.185909191;   /* 1/5.37897... */ 
      break; 
    case XC_GGA_K_ABSP4:      /* Ref. 79 */ 
      params->lambda = 1.0/(1.0 + 1.332/CBRT(N)); 
      break; 
    } 
  } 
} 
 
 
static void  
gga_k_tflw_init(xc_func_type *p) 
{ 
 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_k_tflw_params)); 
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
  p->params_byte_size = sizeof(gga_k_tflw_params); 
#endif 
 
  /* This automatically sets gamma and lambda depending on the functional chosen. 
     We put by default N = 1.0 */ 
  gga_k_tflw_set_params(p, -1.0, -1.0, 1.0); 
} 
 
#ifndef DEVICE 
#include "maple2c/gga_k_tflw.c" 
#endif 
 
#define func maple2c_func 
#define XC_KINETIC_FUNCTIONAL 
#include "work_gga_x.c" 
 
static const func_params_type tfvw_ext_params[] = { 
  {1.0, "Lambda"}, 
  {1.0, "Gamma"}, 
}; 
 
static void  
tfvw_set_ext_params(xc_func_type *p, const double *ext_params) 
{ 
  double ff, lambda, gamma; 
 
  ff = (ext_params == NULL) ? p->info->ext_params[0].value : ext_params[0]; 
  lambda = ff; 
  ff = (ext_params == NULL) ? p->info->ext_params[1].value : ext_params[1]; 
  gamma = ff; 
 
  gga_k_tflw_set_params(p, gamma, lambda, 1.0); 
} 
 
const xc_func_info_type xc_func_info_gga_k_tfvw = { 
  XC_GGA_K_TFVW, 
  XC_KINETIC, 
  "Thomas-Fermi plus von Weiszaecker correction", 
  XC_FAMILY_GGA, 
  {&xc_ref_Weizsacker1935_431, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  2, tfvw_ext_params, tfvw_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_vw = { 
  XC_GGA_K_VW, 
  XC_KINETIC, 
  "von Weiszaecker correction to Thomas-Fermi", 
  XC_FAMILY_GGA, 
  {&xc_ref_Weizsacker1935_431, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_ge2 = { 
  XC_GGA_K_GE2, 
  XC_KINETIC, 
  "Second-order gradient expansion of the kinetic energy density", 
  XC_FAMILY_GGA, 
  {&xc_ref_Kompaneets1956_427, &xc_ref_Kirznits1957_115, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_golden = { 
  XC_GGA_K_GOLDEN, 
  XC_KINETIC, 
  "TF-lambda-vW form by Golden (l = 13/45)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Golden1957_604, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_yt65 = { 
  XC_GGA_K_YT65, 
  XC_KINETIC, 
  "TF-lambda-vW form by Yonei and Tomishima (l = 1/5)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Yonei1965_1051, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_baltin = { 
  XC_GGA_K_BALTIN, 
  XC_KINETIC, 
  "TF-lambda-vW form by Baltin (l = 5/9)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Baltin1972_1176, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_lieb = { 
  XC_GGA_K_LIEB, 
  XC_KINETIC, 
  "TF-lambda-vW form by Lieb (l = 0.185909191)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Lieb1981_603, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  0, NULL, NULL, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
static const func_params_type N_ext_params[] = { 
  {1.0, "Number of electrons"}, 
}; 
 
static void  
N_set_ext_params(xc_func_type *p, const double *ext_params) 
{ 
  double ff, N; 
 
  ff = (ext_params == NULL) ? p->info->ext_params[0].value : ext_params[0]; 
  N = ff; 
 
  gga_k_tflw_set_params(p, -1.0, -1.0, N); 
} 
 
 
const xc_func_info_type xc_func_info_gga_k_absp1 = { 
  XC_GGA_K_ABSP1, 
  XC_KINETIC, 
  "gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)]", 
  XC_FAMILY_GGA, 
  {&xc_ref_Acharya1980_6978, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_absp2 = { 
  XC_GGA_K_ABSP2, 
  XC_KINETIC, 
  "gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)]", 
  XC_FAMILY_GGA, 
  {&xc_ref_Acharya1980_6978, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_absp3 = { 
  XC_GGA_K_ABSP3, 
  XC_KINETIC, 
  "gamma-TFvW form by Acharya et al [g = 1 - 1.513/N^0.35]", 
  XC_FAMILY_GGA, 
  {&xc_ref_Acharya1980_6978, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_absp4 = { 
  XC_GGA_K_ABSP4, 
  XC_KINETIC, 
  "gamma-TFvW form by Acharya et al [g = l = 1/(1 + 1.332/N^(1/3))]", 
  XC_FAMILY_GGA, 
  {&xc_ref_Acharya1980_6978, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_gr = { 
  XC_GGA_K_GR, 
  XC_KINETIC, 
  "gamma-TFvW form by Gazquez and Robles", 
  XC_FAMILY_GGA, 
  {&xc_ref_Gazquez1982_1467, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-25, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_ludena = { 
  XC_GGA_K_LUDENA, 
  XC_KINETIC, 
  "gamma-TFvW form by Ludena", 
  XC_FAMILY_GGA, 
  {&xc_ref_Ludena1986, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
 
const xc_func_info_type xc_func_info_gga_k_gp85 = { 
  XC_GGA_K_GP85, 
  XC_KINETIC, 
  "gamma-TFvW form by Ghosh and Parr", 
  XC_FAMILY_GGA, 
  {&xc_ref_Ghosh1985_3307, NULL, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC, 
  1e-32, 
  1, N_ext_params, N_set_ext_params, 
  gga_k_tflw_init, NULL,  
  NULL, work_gga_k, NULL 
}; 
