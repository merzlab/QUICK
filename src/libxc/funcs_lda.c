#include "util.h"

extern xc_func_info_type xc_func_info_lda_x;
extern xc_func_info_type xc_func_info_lda_c_wigner;
extern xc_func_info_type xc_func_info_lda_c_rpa;
extern xc_func_info_type xc_func_info_lda_c_hl;
extern xc_func_info_type xc_func_info_lda_c_gl;
extern xc_func_info_type xc_func_info_lda_c_xalpha;
extern xc_func_info_type xc_func_info_lda_c_vwn;
extern xc_func_info_type xc_func_info_lda_c_vwn_rpa;
extern xc_func_info_type xc_func_info_lda_c_pz;
extern xc_func_info_type xc_func_info_lda_c_pz_mod;
extern xc_func_info_type xc_func_info_lda_c_ob_pz;
extern xc_func_info_type xc_func_info_lda_c_pw;
extern xc_func_info_type xc_func_info_lda_c_pw_mod;
extern xc_func_info_type xc_func_info_lda_c_ob_pw;
extern xc_func_info_type xc_func_info_lda_c_2d_amgb;
extern xc_func_info_type xc_func_info_lda_c_2d_prm;
extern xc_func_info_type xc_func_info_lda_c_vbh;
extern xc_func_info_type xc_func_info_lda_c_1d_csc;
extern xc_func_info_type xc_func_info_lda_x_2d;
extern xc_func_info_type xc_func_info_lda_xc_teter93;
extern xc_func_info_type xc_func_info_lda_x_1d;
extern xc_func_info_type xc_func_info_lda_c_ml1;
extern xc_func_info_type xc_func_info_lda_c_ml2;
extern xc_func_info_type xc_func_info_lda_c_gombas;
extern xc_func_info_type xc_func_info_lda_c_pw_rpa;
extern xc_func_info_type xc_func_info_lda_c_1d_loos;
extern xc_func_info_type xc_func_info_lda_c_rc04;
extern xc_func_info_type xc_func_info_lda_c_vwn_1;
extern xc_func_info_type xc_func_info_lda_c_vwn_2;
extern xc_func_info_type xc_func_info_lda_c_vwn_3;
extern xc_func_info_type xc_func_info_lda_c_vwn_4;
extern xc_func_info_type xc_func_info_lda_xc_zlp;
extern xc_func_info_type xc_func_info_lda_k_tf;
extern xc_func_info_type xc_func_info_lda_k_lp;
extern xc_func_info_type xc_func_info_lda_xc_ksdt;
extern xc_func_info_type xc_func_info_lda_c_chachiyo;
extern xc_func_info_type xc_func_info_lda_c_lp96;
extern xc_func_info_type xc_func_info_lda_x_rel;
extern xc_func_info_type xc_func_info_lda_xc_1d_ehwlrg_1;
extern xc_func_info_type xc_func_info_lda_xc_1d_ehwlrg_2;
extern xc_func_info_type xc_func_info_lda_xc_1d_ehwlrg_3;
extern xc_func_info_type xc_func_info_lda_x_erf;
extern xc_func_info_type xc_func_info_lda_xc_lp_a;
extern xc_func_info_type xc_func_info_lda_xc_lp_b;
extern xc_func_info_type xc_func_info_lda_x_rae;
extern xc_func_info_type xc_func_info_lda_k_zlp;
extern xc_func_info_type xc_func_info_lda_c_mcweeny;
extern xc_func_info_type xc_func_info_lda_c_br78;
#ifdef COMPILE_PK09
extern xc_func_info_type xc_func_info_lda_c_pk09;
#endif
extern xc_func_info_type xc_func_info_lda_c_ow_lyp;
extern xc_func_info_type xc_func_info_lda_c_ow;
extern xc_func_info_type xc_func_info_lda_xc_gdsmfb;
extern xc_func_info_type xc_func_info_lda_c_gk72;
extern xc_func_info_type xc_func_info_lda_c_karasiev;
extern xc_func_info_type xc_func_info_lda_k_lp96;


const xc_func_info_type *xc_lda_known_funct[] = {
  &xc_func_info_lda_x,
  &xc_func_info_lda_c_wigner,
  &xc_func_info_lda_c_rpa,
  &xc_func_info_lda_c_hl,
  &xc_func_info_lda_c_gl,
  &xc_func_info_lda_c_xalpha,
  &xc_func_info_lda_c_vwn,
  &xc_func_info_lda_c_vwn_rpa,
  &xc_func_info_lda_c_pz,
  &xc_func_info_lda_c_pz_mod,
  &xc_func_info_lda_c_ob_pz,
  &xc_func_info_lda_c_pw,
  &xc_func_info_lda_c_pw_mod,
  &xc_func_info_lda_c_ob_pw,
  &xc_func_info_lda_c_2d_amgb,
  &xc_func_info_lda_c_2d_prm,
  &xc_func_info_lda_c_vbh,
  &xc_func_info_lda_c_1d_csc,
  &xc_func_info_lda_x_2d,
  &xc_func_info_lda_xc_teter93,
  &xc_func_info_lda_x_1d,
  &xc_func_info_lda_c_ml1,
  &xc_func_info_lda_c_ml2,
  &xc_func_info_lda_c_gombas,
  &xc_func_info_lda_c_pw_rpa,
  &xc_func_info_lda_c_1d_loos,
  &xc_func_info_lda_c_rc04,
  &xc_func_info_lda_c_vwn_1,
  &xc_func_info_lda_c_vwn_2,
  &xc_func_info_lda_c_vwn_3,
  &xc_func_info_lda_c_vwn_4,
  &xc_func_info_lda_xc_zlp,
  &xc_func_info_lda_k_tf,
  &xc_func_info_lda_k_lp,
  &xc_func_info_lda_xc_ksdt,
  &xc_func_info_lda_c_chachiyo,
  &xc_func_info_lda_c_lp96,
  &xc_func_info_lda_x_rel,
  &xc_func_info_lda_xc_1d_ehwlrg_1,
  &xc_func_info_lda_xc_1d_ehwlrg_2,
  &xc_func_info_lda_xc_1d_ehwlrg_3,
  &xc_func_info_lda_x_erf,
  &xc_func_info_lda_xc_lp_a,
  &xc_func_info_lda_xc_lp_b,
  &xc_func_info_lda_x_rae,
  &xc_func_info_lda_k_zlp,
  &xc_func_info_lda_c_mcweeny,
  &xc_func_info_lda_c_br78,
#ifdef COMPILE_PK09
  &xc_func_info_lda_c_pk09,
#endif
  &xc_func_info_lda_c_ow_lyp,
  &xc_func_info_lda_c_ow,
  &xc_func_info_lda_xc_gdsmfb,
  &xc_func_info_lda_c_gk72,
  &xc_func_info_lda_c_karasiev,
  &xc_func_info_lda_k_lp96,
  NULL
};
