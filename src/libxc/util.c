/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"


/* this function converts the spin-density into total density and
	 relative magnetization */
/* inline */ void
xc_rho2dzeta(int nspin, const double *rho, double *d, double *zeta)
{
  if(nspin==XC_UNPOLARIZED){
    *d    = max(rho[0], 0.0);
    *zeta = 0.0;
  }else{
    *d = rho[0] + rho[1];
    if(*d > 0.0){
      *zeta = (rho[0] - rho[1])/(*d);
      *zeta = min(*zeta,  1.0);
      *zeta = max(*zeta, -1.0);
    }else{
      *d    = 0.0;
      *zeta = 0.0;
    }
  }
}

/* inline */ void
xc_fast_fzeta(const double x, const int nspin, const int order, double * fz){

  double aa, bb, aa2, bb2;

  if(nspin != XC_UNPOLARIZED){
    aa = CBRT(1.0 + x);
    bb = CBRT(1.0 - x);
    
    aa2 = aa*aa;
    bb2 = bb*bb;
    
    fz[0] = (aa2*aa2 + bb2*bb2 - 2.0)/FZETAFACTOR;
    if(order < 1) return;
    fz[1] = (aa - bb)*(4.0/3.0)/FZETAFACTOR;
    if(order < 2) return;
    fz[2] = ((4.0/9.0)/FZETAFACTOR)*(fabs(x)==1.0 ? (FLT_MAX) : (pow(1.0 + (x), -2.0/3.0) + pow(1.0 - (x), -2.0/3.0)));
    if(order < 3) return;
    fz[3] = (-(8.0/27.0)/FZETAFACTOR)*(fabs(x)==1.0 ? (FLT_MAX) : (pow(1.0 + (x), -5.0/3.0) - pow(1.0 - (x), -5.0/3.0)));
  } else {
    fz[0] = 0.0;
    fz[1] = 0.0;
    fz[2] = (8.0/9.0)/FZETAFACTOR;
    fz[3] = 0.0;
  }
}

/* initializes the mixing */
void 
xc_mix_init(xc_func_type *p, int n_funcs, const int *funcs_id, const double *mix_coef)
{
  int ii;

  assert(p != NULL);
  assert(p->func_aux == NULL && p->mix_coef == NULL);

  /* allocate structures needed for */
  p->n_func_aux = n_funcs;
  p->mix_coef   = (double *) malloc(n_funcs*sizeof(double));
  p->func_aux   = (xc_func_type **) malloc(n_funcs*sizeof(xc_func_type *));

  for(ii=0; ii<n_funcs; ii++){
    p->mix_coef[ii] = mix_coef[ii];
    p->func_aux[ii] = (xc_func_type *) malloc(sizeof(xc_func_type));
    xc_func_init (p->func_aux[ii], funcs_id[ii], p->nspin);
  }

  /* initialize variables */
  p->cam_omega=0.0;
  p->cam_alpha=0.0;
  p->cam_beta=0.0;
  p->nlc_b=0.0;
  p->nlc_C=0.0;
}

xc_gga_enhancement_t
xc_get_gga_enhancement_factor(int func_id)
{
  switch(func_id){
#ifndef CUDA //Madu: temporary blocked this in CUDA version
  case XC_GGA_X_WC:
    return xc_gga_x_wc_enhance;

  case XC_GGA_X_PBE:
  case XC_GGA_X_PBE_R:
  case XC_GGA_X_PBE_SOL:
  case XC_GGA_X_XPBE:
  case XC_GGA_X_PBE_JSJR:
  case XC_GGA_X_PBEK1_VDW:
  case XC_GGA_X_RGE2:
  case XC_GGA_X_APBE:
  case XC_GGA_X_PBEINT:
  case XC_GGA_X_PBE_TCA:
    return xc_gga_x_pbe_enhance;

  case XC_GGA_X_PW91:
  case XC_GGA_X_MPW91:
    return xc_gga_x_pw91_enhance;

  case XC_GGA_X_RPBE:
    return xc_gga_x_rpbe_enhance;

  case XC_GGA_X_HTBS:
    return xc_gga_x_htbs_enhance;

  case XC_GGA_X_B86:
  case XC_GGA_X_B86_MGC:
  case XC_GGA_X_B86_R:
    return xc_gga_x_b86_enhance;

  case XC_GGA_X_B88:
  case XC_GGA_X_OPTB88_VDW:
  case XC_GGA_X_MB88:
    return xc_gga_x_b88_enhance;
  case XC_GGA_X_G96:
    return xc_gga_x_g96_enhance;

  case XC_GGA_X_PW86:
  case XC_GGA_X_RPW86:
    return xc_gga_x_pw86_enhance;

  case XC_GGA_X_AIRY:
  case XC_GGA_X_LAG:
    return xc_gga_x_airy_enhance;

  case XC_GGA_X_BAYESIAN:
    return xc_gga_x_bayesian_enhance;

  case XC_GGA_X_BPCCAC:
    return xc_gga_x_bpccac_enhance;

  case XC_GGA_X_C09X:
    return xc_gga_x_c09x_enhance;

  case XC_GGA_X_AM05:
    return xc_gga_x_am05_enhance;

  case XC_GGA_X_DK87_R1:
  case XC_GGA_X_DK87_R2:
    return xc_gga_x_dk87_enhance;

  case XC_GGA_X_HERMAN:
    return xc_gga_x_herman_enhance;

  case XC_GGA_X_LG93:
    return xc_gga_x_lg93_enhance;

  case XC_GGA_X_LV_RPW86:
    return xc_gga_x_lv_rpw86_enhance;

  case XC_GGA_X_MPBE:
    return xc_gga_x_mpbe_enhance;

  case XC_GGA_X_OPTX:
    return xc_gga_x_optx_enhance;

  case XC_GGA_X_SOGGA11:
  case XC_HYB_GGA_X_SOGGA11_X:
    return xc_gga_x_sogga11_enhance;

  case XC_GGA_X_SSB_SW:
  case XC_GGA_X_SSB:
  case XC_GGA_X_SSB_D:
    return xc_gga_x_ssb_sw_enhance;

  case XC_GGA_X_VMT_PBE:
  case XC_GGA_X_VMT_GE:
  case XC_GGA_X_VMT84_PBE:
  case XC_GGA_X_VMT84_GE:
    return xc_gga_x_vmt_enhance;
#endif
  default:
    fprintf(stderr, "Internal error in get_gga_enhancement\n");
    exit(1);
  }
}


const char *get_kind(const xc_func_type *func) {
  switch(func->info->kind) {
    case(XC_EXCHANGE):
      return "XC_EXCHANGE";

    case(XC_CORRELATION):
      return "XC_CORRELATION";

    case(XC_EXCHANGE_CORRELATION):
      return "XC_EXCHANGE_CORRELATION";

    case(XC_KINETIC):
      return "XC_KINETIC";

    default:
      printf("Internal error in get_kind.\n");
      return "";
  }
}

const char *get_family(const xc_func_type *func) {
  switch(func->info->family) {
    case(XC_FAMILY_UNKNOWN):
      return "XC_FAMILY_UNKNOWN";

    case(XC_FAMILY_LDA):
      return "XC_FAMILY_LDA";

    case(XC_FAMILY_GGA):
      return "XC_FAMILY_GGA";

    case(XC_FAMILY_MGGA):
      return "XC_FAMILY_MGGA";

    case(XC_FAMILY_LCA):
      return "XC_FAMILY_LCA";

    case(XC_FAMILY_OEP):
      return "XC_FAMILY_OEP";

    case(XC_FAMILY_HYB_GGA):
      return "XC_FAMILY_HYB_GGA";

    case(XC_FAMILY_HYB_MGGA):
      return "XC_FAMILY_HYB_MGGA";

    default:
      printf("Internal error in get_family.\n");
      return "";
  }
}
