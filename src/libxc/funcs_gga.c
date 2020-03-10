#include "util.h"

extern xc_func_info_type xc_func_info_gga_x_gam;
extern xc_func_info_type xc_func_info_gga_c_gam;
extern xc_func_info_type xc_func_info_gga_x_hcth_a;
extern xc_func_info_type xc_func_info_gga_x_ev93;
extern xc_func_info_type xc_func_info_gga_x_bcgp;
extern xc_func_info_type xc_func_info_gga_c_bcgp;
extern xc_func_info_type xc_func_info_gga_x_lambda_oc2_n;
extern xc_func_info_type xc_func_info_gga_x_b86_r;
extern xc_func_info_type xc_func_info_gga_x_lambda_ch_n;
extern xc_func_info_type xc_func_info_gga_x_lambda_lo_n;
extern xc_func_info_type xc_func_info_gga_x_hjs_b88_v2;
extern xc_func_info_type xc_func_info_gga_c_q2d;
extern xc_func_info_type xc_func_info_gga_x_q2d;
extern xc_func_info_type xc_func_info_gga_x_pbe_mol;
extern xc_func_info_type xc_func_info_gga_k_tfvw;
extern xc_func_info_type xc_func_info_gga_k_revapbeint;
extern xc_func_info_type xc_func_info_gga_k_apbeint;
extern xc_func_info_type xc_func_info_gga_k_revapbe;
extern xc_func_info_type xc_func_info_gga_x_ak13;
extern xc_func_info_type xc_func_info_gga_k_meyer;
extern xc_func_info_type xc_func_info_gga_x_lv_rpw86;
extern xc_func_info_type xc_func_info_gga_x_pbe_tca;
extern xc_func_info_type xc_func_info_gga_x_pbeint;
extern xc_func_info_type xc_func_info_gga_c_zpbeint;
extern xc_func_info_type xc_func_info_gga_c_pbeint;
extern xc_func_info_type xc_func_info_gga_c_zpbesol;
extern xc_func_info_type xc_func_info_gga_xc_opbe_d;
extern xc_func_info_type xc_func_info_gga_xc_opwlyp_d;
extern xc_func_info_type xc_func_info_gga_xc_oblyp_d;
extern xc_func_info_type xc_func_info_gga_x_vmt84_ge;
extern xc_func_info_type xc_func_info_gga_x_vmt84_pbe;
extern xc_func_info_type xc_func_info_gga_x_vmt_ge;
extern xc_func_info_type xc_func_info_gga_x_vmt_pbe;
extern xc_func_info_type xc_func_info_gga_c_n12_sx;
extern xc_func_info_type xc_func_info_gga_c_n12;
extern xc_func_info_type xc_func_info_gga_x_n12;
extern xc_func_info_type xc_func_info_gga_c_regtpss;
extern xc_func_info_type xc_func_info_gga_c_op_xalpha;
extern xc_func_info_type xc_func_info_gga_c_op_g96;
extern xc_func_info_type xc_func_info_gga_c_op_pbe;
extern xc_func_info_type xc_func_info_gga_c_op_b88;
extern xc_func_info_type xc_func_info_gga_c_ft97;
extern xc_func_info_type xc_func_info_gga_c_spbe;
extern xc_func_info_type xc_func_info_gga_x_ssb_sw;
extern xc_func_info_type xc_func_info_gga_x_ssb;
extern xc_func_info_type xc_func_info_gga_x_ssb_d;
extern xc_func_info_type xc_func_info_gga_xc_hcth_407p;
extern xc_func_info_type xc_func_info_gga_xc_hcth_p76;
extern xc_func_info_type xc_func_info_gga_xc_hcth_p14;
extern xc_func_info_type xc_func_info_gga_xc_b97_gga1;
extern xc_func_info_type xc_func_info_gga_c_hcth_a;
extern xc_func_info_type xc_func_info_gga_x_bpccac;
extern xc_func_info_type xc_func_info_gga_c_revtca;
extern xc_func_info_type xc_func_info_gga_c_tca;
extern xc_func_info_type xc_func_info_gga_x_pbe;
extern xc_func_info_type xc_func_info_gga_x_pbe_r;
extern xc_func_info_type xc_func_info_gga_x_b86;
extern xc_func_info_type xc_func_info_gga_x_herman;
extern xc_func_info_type xc_func_info_gga_x_b86_mgc;
extern xc_func_info_type xc_func_info_gga_x_b88;
extern xc_func_info_type xc_func_info_gga_x_g96;
extern xc_func_info_type xc_func_info_gga_x_pw86;
extern xc_func_info_type xc_func_info_gga_x_pw91;
extern xc_func_info_type xc_func_info_gga_x_optx;
extern xc_func_info_type xc_func_info_gga_x_dk87_r1;
extern xc_func_info_type xc_func_info_gga_x_dk87_r2;
extern xc_func_info_type xc_func_info_gga_x_lg93;
extern xc_func_info_type xc_func_info_gga_x_ft97_a;
extern xc_func_info_type xc_func_info_gga_x_ft97_b;
extern xc_func_info_type xc_func_info_gga_x_pbe_sol;
extern xc_func_info_type xc_func_info_gga_x_rpbe;
extern xc_func_info_type xc_func_info_gga_x_wc;
extern xc_func_info_type xc_func_info_gga_x_mpw91;
extern xc_func_info_type xc_func_info_gga_x_am05;
extern xc_func_info_type xc_func_info_gga_x_pbea;
extern xc_func_info_type xc_func_info_gga_x_mpbe;
extern xc_func_info_type xc_func_info_gga_x_xpbe;
extern xc_func_info_type xc_func_info_gga_x_2d_b86_mgc;
extern xc_func_info_type xc_func_info_gga_x_bayesian;
extern xc_func_info_type xc_func_info_gga_x_pbe_jsjr;
extern xc_func_info_type xc_func_info_gga_x_2d_b88;
extern xc_func_info_type xc_func_info_gga_x_2d_b86;
extern xc_func_info_type xc_func_info_gga_x_2d_pbe;
extern xc_func_info_type xc_func_info_gga_c_pbe;
extern xc_func_info_type xc_func_info_gga_c_lyp;
extern xc_func_info_type xc_func_info_gga_c_p86;
extern xc_func_info_type xc_func_info_gga_c_pbe_sol;
extern xc_func_info_type xc_func_info_gga_c_pw91;
extern xc_func_info_type xc_func_info_gga_c_am05;
extern xc_func_info_type xc_func_info_gga_c_xpbe;
extern xc_func_info_type xc_func_info_gga_c_lm;
extern xc_func_info_type xc_func_info_gga_c_pbe_jrgx;
extern xc_func_info_type xc_func_info_gga_x_optb88_vdw;
extern xc_func_info_type xc_func_info_gga_x_pbek1_vdw;
extern xc_func_info_type xc_func_info_gga_x_optpbe_vdw;
extern xc_func_info_type xc_func_info_gga_x_rge2;
extern xc_func_info_type xc_func_info_gga_c_rge2;
extern xc_func_info_type xc_func_info_gga_x_rpw86;
extern xc_func_info_type xc_func_info_gga_x_kt1;
extern xc_func_info_type xc_func_info_gga_xc_kt2;
extern xc_func_info_type xc_func_info_gga_c_wl;
extern xc_func_info_type xc_func_info_gga_c_wi;
extern xc_func_info_type xc_func_info_gga_x_mb88;
extern xc_func_info_type xc_func_info_gga_x_sogga;
extern xc_func_info_type xc_func_info_gga_x_sogga11;
extern xc_func_info_type xc_func_info_gga_c_sogga11;
extern xc_func_info_type xc_func_info_gga_c_wi0;
extern xc_func_info_type xc_func_info_gga_xc_th1;
extern xc_func_info_type xc_func_info_gga_xc_th2;
extern xc_func_info_type xc_func_info_gga_xc_th3;
extern xc_func_info_type xc_func_info_gga_xc_th4;
extern xc_func_info_type xc_func_info_gga_x_c09x;
extern xc_func_info_type xc_func_info_gga_c_sogga11_x;
extern xc_func_info_type xc_func_info_gga_x_lb;
extern xc_func_info_type xc_func_info_gga_xc_hcth_93;
extern xc_func_info_type xc_func_info_gga_xc_hcth_120;
extern xc_func_info_type xc_func_info_gga_xc_hcth_147;
extern xc_func_info_type xc_func_info_gga_xc_hcth_407;
extern xc_func_info_type xc_func_info_gga_xc_edf1;
extern xc_func_info_type xc_func_info_gga_xc_xlyp;
extern xc_func_info_type xc_func_info_gga_xc_kt1;
extern xc_func_info_type xc_func_info_gga_xc_b97_d;
extern xc_func_info_type xc_func_info_gga_xc_pbe1w;
extern xc_func_info_type xc_func_info_gga_xc_mpwlyp1w;
extern xc_func_info_type xc_func_info_gga_xc_pbelyp1w;
extern xc_func_info_type xc_func_info_gga_x_lbm;
extern xc_func_info_type xc_func_info_gga_x_ol2;
extern xc_func_info_type xc_func_info_gga_x_apbe;
extern xc_func_info_type xc_func_info_gga_k_apbe;
extern xc_func_info_type xc_func_info_gga_c_apbe;
extern xc_func_info_type xc_func_info_gga_k_tw1;
extern xc_func_info_type xc_func_info_gga_k_tw2;
extern xc_func_info_type xc_func_info_gga_k_tw3;
extern xc_func_info_type xc_func_info_gga_k_tw4;
extern xc_func_info_type xc_func_info_gga_x_htbs;
extern xc_func_info_type xc_func_info_gga_x_airy;
extern xc_func_info_type xc_func_info_gga_x_lag;
extern xc_func_info_type xc_func_info_gga_xc_mohlyp;
extern xc_func_info_type xc_func_info_gga_xc_mohlyp2;
extern xc_func_info_type xc_func_info_gga_xc_th_fl;
extern xc_func_info_type xc_func_info_gga_xc_th_fc;
extern xc_func_info_type xc_func_info_gga_xc_th_fcfo;
extern xc_func_info_type xc_func_info_gga_xc_th_fco;
extern xc_func_info_type xc_func_info_gga_c_optc;
extern xc_func_info_type xc_func_info_gga_c_pbeloc;
extern xc_func_info_type xc_func_info_gga_xc_vv10;
extern xc_func_info_type xc_func_info_gga_c_pbefe;
extern xc_func_info_type xc_func_info_gga_c_op_pw91;
extern xc_func_info_type xc_func_info_gga_x_pbefe;
extern xc_func_info_type xc_func_info_gga_x_cap;
extern xc_func_info_type xc_func_info_gga_x_eb88;
extern xc_func_info_type xc_func_info_gga_c_pbe_mol;
extern xc_func_info_type xc_func_info_gga_k_absp3;
extern xc_func_info_type xc_func_info_gga_k_absp4;
extern xc_func_info_type xc_func_info_gga_c_bmk;
extern xc_func_info_type xc_func_info_gga_c_tau_hcth;
extern xc_func_info_type xc_func_info_gga_c_hyb_tau_hcth;
extern xc_func_info_type xc_func_info_gga_x_beefvdw;
extern xc_func_info_type xc_func_info_gga_xc_beefvdw;
extern xc_func_info_type xc_func_info_gga_x_pbetrans;
extern xc_func_info_type xc_func_info_gga_x_chachiyo;
extern xc_func_info_type xc_func_info_gga_k_vw;
extern xc_func_info_type xc_func_info_gga_k_ge2;
extern xc_func_info_type xc_func_info_gga_k_golden;
extern xc_func_info_type xc_func_info_gga_k_yt65;
extern xc_func_info_type xc_func_info_gga_k_baltin;
extern xc_func_info_type xc_func_info_gga_k_lieb;
extern xc_func_info_type xc_func_info_gga_k_absp1;
extern xc_func_info_type xc_func_info_gga_k_absp2;
extern xc_func_info_type xc_func_info_gga_k_gr;
extern xc_func_info_type xc_func_info_gga_k_ludena;
extern xc_func_info_type xc_func_info_gga_k_gp85;
extern xc_func_info_type xc_func_info_gga_k_pearson;
extern xc_func_info_type xc_func_info_gga_k_ol1;
extern xc_func_info_type xc_func_info_gga_k_ol2;
extern xc_func_info_type xc_func_info_gga_k_fr_b88;
extern xc_func_info_type xc_func_info_gga_k_fr_pw86;
extern xc_func_info_type xc_func_info_gga_k_dk;
extern xc_func_info_type xc_func_info_gga_k_perdew;
extern xc_func_info_type xc_func_info_gga_k_vsk;
extern xc_func_info_type xc_func_info_gga_k_vjks;
extern xc_func_info_type xc_func_info_gga_k_ernzerhof;
extern xc_func_info_type xc_func_info_gga_k_lc94;
extern xc_func_info_type xc_func_info_gga_k_llp;
extern xc_func_info_type xc_func_info_gga_k_thakkar;
extern xc_func_info_type xc_func_info_gga_x_wpbeh;
extern xc_func_info_type xc_func_info_gga_x_hjs_pbe;
extern xc_func_info_type xc_func_info_gga_x_hjs_pbe_sol;
extern xc_func_info_type xc_func_info_gga_x_hjs_b88;
extern xc_func_info_type xc_func_info_gga_x_hjs_b97x;
extern xc_func_info_type xc_func_info_gga_x_ityh;
extern xc_func_info_type xc_func_info_gga_x_sfat;
extern xc_func_info_type xc_func_info_gga_x_sg4;
extern xc_func_info_type xc_func_info_gga_c_sg4;
extern xc_func_info_type xc_func_info_gga_x_gg99;
extern xc_func_info_type xc_func_info_gga_x_pbepow;
extern xc_func_info_type xc_func_info_gga_x_kgg99;
extern xc_func_info_type xc_func_info_gga_xc_hle16;
extern xc_func_info_type xc_func_info_gga_c_scan_e0;
extern xc_func_info_type xc_func_info_gga_c_gapc;
extern xc_func_info_type xc_func_info_gga_c_gaploc;
extern xc_func_info_type xc_func_info_gga_c_zvpbeint;
extern xc_func_info_type xc_func_info_gga_c_zvpbesol;
extern xc_func_info_type xc_func_info_gga_c_tm_lyp;
extern xc_func_info_type xc_func_info_gga_c_tm_pbe;
extern xc_func_info_type xc_func_info_gga_c_w94;
extern xc_func_info_type xc_func_info_gga_c_cs1;
extern xc_func_info_type xc_func_info_gga_x_b88m;
extern xc_func_info_type xc_func_info_gga_k_pbe3;
extern xc_func_info_type xc_func_info_gga_k_pbe4;
extern xc_func_info_type xc_func_info_gga_k_exp4;


const xc_func_info_type *xc_gga_known_funct[] = {
  &xc_func_info_gga_x_gam,
  &xc_func_info_gga_c_gam,
  &xc_func_info_gga_x_hcth_a,
  &xc_func_info_gga_x_ev93,
  &xc_func_info_gga_x_bcgp,
  &xc_func_info_gga_c_bcgp,
  &xc_func_info_gga_x_lambda_oc2_n,
  &xc_func_info_gga_x_b86_r,
  &xc_func_info_gga_x_lambda_ch_n,
  &xc_func_info_gga_x_lambda_lo_n,
  &xc_func_info_gga_x_hjs_b88_v2,
  &xc_func_info_gga_c_q2d,
  &xc_func_info_gga_x_q2d,
  &xc_func_info_gga_x_pbe_mol,
  &xc_func_info_gga_k_tfvw,
  &xc_func_info_gga_k_revapbeint,
  &xc_func_info_gga_k_apbeint,
  &xc_func_info_gga_k_revapbe,
  &xc_func_info_gga_x_ak13,
  &xc_func_info_gga_k_meyer,
  &xc_func_info_gga_x_lv_rpw86,
  &xc_func_info_gga_x_pbe_tca,
  &xc_func_info_gga_x_pbeint,
  &xc_func_info_gga_c_zpbeint,
  &xc_func_info_gga_c_pbeint,
  &xc_func_info_gga_c_zpbesol,
  &xc_func_info_gga_xc_opbe_d,
  &xc_func_info_gga_xc_opwlyp_d,
  &xc_func_info_gga_xc_oblyp_d,
  &xc_func_info_gga_x_vmt84_ge,
  &xc_func_info_gga_x_vmt84_pbe,
  &xc_func_info_gga_x_vmt_ge,
  &xc_func_info_gga_x_vmt_pbe,
  &xc_func_info_gga_c_n12_sx,
  &xc_func_info_gga_c_n12,
  &xc_func_info_gga_x_n12,
  &xc_func_info_gga_c_regtpss,
  &xc_func_info_gga_c_op_xalpha,
  &xc_func_info_gga_c_op_g96,
  &xc_func_info_gga_c_op_pbe,
  &xc_func_info_gga_c_op_b88,
  &xc_func_info_gga_c_ft97,
  &xc_func_info_gga_c_spbe,
  &xc_func_info_gga_x_ssb_sw,
  &xc_func_info_gga_x_ssb,
  &xc_func_info_gga_x_ssb_d,
  &xc_func_info_gga_xc_hcth_407p,
  &xc_func_info_gga_xc_hcth_p76,
  &xc_func_info_gga_xc_hcth_p14,
  &xc_func_info_gga_xc_b97_gga1,
  &xc_func_info_gga_c_hcth_a,
  &xc_func_info_gga_x_bpccac,
  &xc_func_info_gga_c_revtca,
  &xc_func_info_gga_c_tca,
  &xc_func_info_gga_x_pbe,
  &xc_func_info_gga_x_pbe_r,
  &xc_func_info_gga_x_b86,
  &xc_func_info_gga_x_herman,
  &xc_func_info_gga_x_b86_mgc,
  &xc_func_info_gga_x_b88,
  &xc_func_info_gga_x_g96,
  &xc_func_info_gga_x_pw86,
  &xc_func_info_gga_x_pw91,
  &xc_func_info_gga_x_optx,
  &xc_func_info_gga_x_dk87_r1,
  &xc_func_info_gga_x_dk87_r2,
  &xc_func_info_gga_x_lg93,
  &xc_func_info_gga_x_ft97_a,
  &xc_func_info_gga_x_ft97_b,
  &xc_func_info_gga_x_pbe_sol,
  &xc_func_info_gga_x_rpbe,
  &xc_func_info_gga_x_wc,
  &xc_func_info_gga_x_mpw91,
  &xc_func_info_gga_x_am05,
  &xc_func_info_gga_x_pbea,
  &xc_func_info_gga_x_mpbe,
  &xc_func_info_gga_x_xpbe,
  &xc_func_info_gga_x_2d_b86_mgc,
  &xc_func_info_gga_x_bayesian,
  &xc_func_info_gga_x_pbe_jsjr,
  &xc_func_info_gga_x_2d_b88,
  &xc_func_info_gga_x_2d_b86,
  &xc_func_info_gga_x_2d_pbe,
  &xc_func_info_gga_c_pbe,
  &xc_func_info_gga_c_lyp,
  &xc_func_info_gga_c_p86,
  &xc_func_info_gga_c_pbe_sol,
  &xc_func_info_gga_c_pw91,
  &xc_func_info_gga_c_am05,
  &xc_func_info_gga_c_xpbe,
  &xc_func_info_gga_c_lm,
  &xc_func_info_gga_c_pbe_jrgx,
  &xc_func_info_gga_x_optb88_vdw,
  &xc_func_info_gga_x_pbek1_vdw,
  &xc_func_info_gga_x_optpbe_vdw,
  &xc_func_info_gga_x_rge2,
  &xc_func_info_gga_c_rge2,
  &xc_func_info_gga_x_rpw86,
  &xc_func_info_gga_x_kt1,
  &xc_func_info_gga_xc_kt2,
  &xc_func_info_gga_c_wl,
  &xc_func_info_gga_c_wi,
  &xc_func_info_gga_x_mb88,
  &xc_func_info_gga_x_sogga,
  &xc_func_info_gga_x_sogga11,
  &xc_func_info_gga_c_sogga11,
  &xc_func_info_gga_c_wi0,
  &xc_func_info_gga_xc_th1,
  &xc_func_info_gga_xc_th2,
  &xc_func_info_gga_xc_th3,
  &xc_func_info_gga_xc_th4,
  &xc_func_info_gga_x_c09x,
  &xc_func_info_gga_c_sogga11_x,
  &xc_func_info_gga_x_lb,
  &xc_func_info_gga_xc_hcth_93,
  &xc_func_info_gga_xc_hcth_120,
  &xc_func_info_gga_xc_hcth_147,
  &xc_func_info_gga_xc_hcth_407,
  &xc_func_info_gga_xc_edf1,
  &xc_func_info_gga_xc_xlyp,
  &xc_func_info_gga_xc_kt1,
  &xc_func_info_gga_xc_b97_d,
  &xc_func_info_gga_xc_pbe1w,
  &xc_func_info_gga_xc_mpwlyp1w,
  &xc_func_info_gga_xc_pbelyp1w,
  &xc_func_info_gga_x_lbm,
  &xc_func_info_gga_x_ol2,
  &xc_func_info_gga_x_apbe,
  &xc_func_info_gga_k_apbe,
  &xc_func_info_gga_c_apbe,
  &xc_func_info_gga_k_tw1,
  &xc_func_info_gga_k_tw2,
  &xc_func_info_gga_k_tw3,
  &xc_func_info_gga_k_tw4,
  &xc_func_info_gga_x_htbs,
  &xc_func_info_gga_x_airy,
  &xc_func_info_gga_x_lag,
  &xc_func_info_gga_xc_mohlyp,
  &xc_func_info_gga_xc_mohlyp2,
  &xc_func_info_gga_xc_th_fl,
  &xc_func_info_gga_xc_th_fc,
  &xc_func_info_gga_xc_th_fcfo,
  &xc_func_info_gga_xc_th_fco,
  &xc_func_info_gga_c_optc,
  &xc_func_info_gga_c_pbeloc,
  &xc_func_info_gga_xc_vv10,
  &xc_func_info_gga_c_pbefe,
  &xc_func_info_gga_c_op_pw91,
  &xc_func_info_gga_x_pbefe,
  &xc_func_info_gga_x_cap,
  &xc_func_info_gga_x_eb88,
  &xc_func_info_gga_c_pbe_mol,
  &xc_func_info_gga_k_absp3,
  &xc_func_info_gga_k_absp4,
  &xc_func_info_gga_c_bmk,
  &xc_func_info_gga_c_tau_hcth,
  &xc_func_info_gga_c_hyb_tau_hcth,
  &xc_func_info_gga_x_beefvdw,
  &xc_func_info_gga_xc_beefvdw,
  &xc_func_info_gga_x_pbetrans,
  &xc_func_info_gga_x_chachiyo,
  &xc_func_info_gga_k_vw,
  &xc_func_info_gga_k_ge2,
  &xc_func_info_gga_k_golden,
  &xc_func_info_gga_k_yt65,
  &xc_func_info_gga_k_baltin,
  &xc_func_info_gga_k_lieb,
  &xc_func_info_gga_k_absp1,
  &xc_func_info_gga_k_absp2,
  &xc_func_info_gga_k_gr,
  &xc_func_info_gga_k_ludena,
  &xc_func_info_gga_k_gp85,
  &xc_func_info_gga_k_pearson,
  &xc_func_info_gga_k_ol1,
  &xc_func_info_gga_k_ol2,
  &xc_func_info_gga_k_fr_b88,
  &xc_func_info_gga_k_fr_pw86,
  &xc_func_info_gga_k_dk,
  &xc_func_info_gga_k_perdew,
  &xc_func_info_gga_k_vsk,
  &xc_func_info_gga_k_vjks,
  &xc_func_info_gga_k_ernzerhof,
  &xc_func_info_gga_k_lc94,
  &xc_func_info_gga_k_llp,
  &xc_func_info_gga_k_thakkar,
  &xc_func_info_gga_x_wpbeh,
  &xc_func_info_gga_x_hjs_pbe,
  &xc_func_info_gga_x_hjs_pbe_sol,
  &xc_func_info_gga_x_hjs_b88,
  &xc_func_info_gga_x_hjs_b97x,
  &xc_func_info_gga_x_ityh,
  &xc_func_info_gga_x_sfat,
  &xc_func_info_gga_x_sg4,
  &xc_func_info_gga_c_sg4,
  &xc_func_info_gga_x_gg99,
  &xc_func_info_gga_x_pbepow,
  &xc_func_info_gga_x_kgg99,
  &xc_func_info_gga_xc_hle16,
  &xc_func_info_gga_c_scan_e0,
  &xc_func_info_gga_c_gapc,
  &xc_func_info_gga_c_gaploc,
  &xc_func_info_gga_c_zvpbeint,
  &xc_func_info_gga_c_zvpbesol,
  &xc_func_info_gga_c_tm_lyp,
  &xc_func_info_gga_c_tm_pbe,
  &xc_func_info_gga_c_w94,
  &xc_func_info_gga_c_cs1,
  &xc_func_info_gga_x_b88m,
  &xc_func_info_gga_k_pbe3,
  &xc_func_info_gga_k_pbe4,
  &xc_func_info_gga_k_exp4,
  NULL
};
