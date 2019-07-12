#include "util.h"

extern xc_func_info_type xc_func_info_mgga_c_dldf;
extern xc_func_info_type xc_func_info_mgga_xc_zlp;
extern xc_func_info_type xc_func_info_mgga_xc_otpss_d;
extern xc_func_info_type xc_func_info_mgga_c_cs;
extern xc_func_info_type xc_func_info_mgga_c_mn12_sx;
extern xc_func_info_type xc_func_info_mgga_c_mn12_l;
extern xc_func_info_type xc_func_info_mgga_c_m11_l;
extern xc_func_info_type xc_func_info_mgga_c_m11;
extern xc_func_info_type xc_func_info_mgga_c_m08_so;
extern xc_func_info_type xc_func_info_mgga_c_m08_hx;
extern xc_func_info_type xc_func_info_mgga_x_lta;
extern xc_func_info_type xc_func_info_mgga_x_tpss;
extern xc_func_info_type xc_func_info_mgga_x_m06_l;
extern xc_func_info_type xc_func_info_mgga_x_gvt4;
extern xc_func_info_type xc_func_info_mgga_x_tau_hcth;
extern xc_func_info_type xc_func_info_mgga_x_br89;
extern xc_func_info_type xc_func_info_mgga_x_bj06;
extern xc_func_info_type xc_func_info_mgga_x_tb09;
extern xc_func_info_type xc_func_info_mgga_x_rpp09;
extern xc_func_info_type xc_func_info_mgga_x_2d_prhg07;
extern xc_func_info_type xc_func_info_mgga_x_2d_prhg07_prp10;
extern xc_func_info_type xc_func_info_mgga_x_revtpss;
extern xc_func_info_type xc_func_info_mgga_x_pkzb;
extern xc_func_info_type xc_func_info_mgga_x_ms0;
extern xc_func_info_type xc_func_info_mgga_x_ms1;
extern xc_func_info_type xc_func_info_mgga_x_ms2;
extern xc_func_info_type xc_func_info_mgga_x_m11_l;
extern xc_func_info_type xc_func_info_mgga_x_mn12_l;
extern xc_func_info_type xc_func_info_mgga_xc_cc06;
extern xc_func_info_type xc_func_info_mgga_x_mk00;
extern xc_func_info_type xc_func_info_mgga_c_tpss;
extern xc_func_info_type xc_func_info_mgga_c_vsxc;
extern xc_func_info_type xc_func_info_mgga_c_m06_l;
extern xc_func_info_type xc_func_info_mgga_c_m06_hf;
extern xc_func_info_type xc_func_info_mgga_c_m06;
extern xc_func_info_type xc_func_info_mgga_c_m06_2x;
extern xc_func_info_type xc_func_info_mgga_c_m05;
extern xc_func_info_type xc_func_info_mgga_c_m05_2x;
extern xc_func_info_type xc_func_info_mgga_c_pkzb;
extern xc_func_info_type xc_func_info_mgga_c_bc95;
extern xc_func_info_type xc_func_info_mgga_c_revtpss;
extern xc_func_info_type xc_func_info_mgga_xc_tpsslyp1w;
extern xc_func_info_type xc_func_info_mgga_x_mk00b;
extern xc_func_info_type xc_func_info_mgga_x_bloc;
extern xc_func_info_type xc_func_info_mgga_x_modtpss;
extern xc_func_info_type xc_func_info_mgga_c_tpssloc;
extern xc_func_info_type xc_func_info_mgga_x_mbeef;
extern xc_func_info_type xc_func_info_mgga_x_mbeefvdw;
extern xc_func_info_type xc_func_info_mgga_xc_b97m_v;
extern xc_func_info_type xc_func_info_mgga_x_mvs;
extern xc_func_info_type xc_func_info_mgga_x_mn15_l;
extern xc_func_info_type xc_func_info_mgga_c_mn15_l;
extern xc_func_info_type xc_func_info_mgga_x_scan;
extern xc_func_info_type xc_func_info_mgga_c_scan;
extern xc_func_info_type xc_func_info_mgga_c_mn15;
extern xc_func_info_type xc_func_info_mgga_x_b00;
extern xc_func_info_type xc_func_info_mgga_xc_hle17;
extern xc_func_info_type xc_func_info_mgga_c_scan_rvv10;
extern xc_func_info_type xc_func_info_mgga_x_revm06_l;
extern xc_func_info_type xc_func_info_mgga_c_revm06_l;
extern xc_func_info_type xc_func_info_mgga_x_tm;
extern xc_func_info_type xc_func_info_mgga_x_vt84;
extern xc_func_info_type xc_func_info_mgga_x_sa_tpss;
extern xc_func_info_type xc_func_info_mgga_k_pc07;
extern xc_func_info_type xc_func_info_mgga_c_kcis;
extern xc_func_info_type xc_func_info_mgga_xc_lp90;
extern xc_func_info_type xc_func_info_mgga_c_b88;
extern xc_func_info_type xc_func_info_mgga_x_gx;
extern xc_func_info_type xc_func_info_mgga_x_pbe_gx;
extern xc_func_info_type xc_func_info_mgga_x_revscan;
extern xc_func_info_type xc_func_info_mgga_c_revscan;
extern xc_func_info_type xc_func_info_mgga_c_scan_vv10;
extern xc_func_info_type xc_func_info_mgga_c_revscan_vv10;
extern xc_func_info_type xc_func_info_mgga_x_br89_explicit;


const xc_func_info_type *xc_mgga_known_funct[] = {
  &xc_func_info_mgga_c_dldf,
  &xc_func_info_mgga_xc_zlp,
  &xc_func_info_mgga_xc_otpss_d,
  &xc_func_info_mgga_c_cs,
  &xc_func_info_mgga_c_mn12_sx,
  &xc_func_info_mgga_c_mn12_l,
  &xc_func_info_mgga_c_m11_l,
  &xc_func_info_mgga_c_m11,
  &xc_func_info_mgga_c_m08_so,
  &xc_func_info_mgga_c_m08_hx,
  &xc_func_info_mgga_x_lta,
  &xc_func_info_mgga_x_tpss,
  &xc_func_info_mgga_x_m06_l,
  &xc_func_info_mgga_x_gvt4,
  &xc_func_info_mgga_x_tau_hcth,
  &xc_func_info_mgga_x_br89,
  &xc_func_info_mgga_x_bj06,
  &xc_func_info_mgga_x_tb09,
  &xc_func_info_mgga_x_rpp09,
  &xc_func_info_mgga_x_2d_prhg07,
  &xc_func_info_mgga_x_2d_prhg07_prp10,
  &xc_func_info_mgga_x_revtpss,
  &xc_func_info_mgga_x_pkzb,
  &xc_func_info_mgga_x_ms0,
  &xc_func_info_mgga_x_ms1,
  &xc_func_info_mgga_x_ms2,
  &xc_func_info_mgga_x_m11_l,
  &xc_func_info_mgga_x_mn12_l,
  &xc_func_info_mgga_xc_cc06,
  &xc_func_info_mgga_x_mk00,
  &xc_func_info_mgga_c_tpss,
  &xc_func_info_mgga_c_vsxc,
  &xc_func_info_mgga_c_m06_l,
  &xc_func_info_mgga_c_m06_hf,
  &xc_func_info_mgga_c_m06,
  &xc_func_info_mgga_c_m06_2x,
  &xc_func_info_mgga_c_m05,
  &xc_func_info_mgga_c_m05_2x,
  &xc_func_info_mgga_c_pkzb,
  &xc_func_info_mgga_c_bc95,
  &xc_func_info_mgga_c_revtpss,
  &xc_func_info_mgga_xc_tpsslyp1w,
  &xc_func_info_mgga_x_mk00b,
  &xc_func_info_mgga_x_bloc,
  &xc_func_info_mgga_x_modtpss,
  &xc_func_info_mgga_c_tpssloc,
  &xc_func_info_mgga_x_mbeef,
  &xc_func_info_mgga_x_mbeefvdw,
  &xc_func_info_mgga_xc_b97m_v,
  &xc_func_info_mgga_x_mvs,
  &xc_func_info_mgga_x_mn15_l,
  &xc_func_info_mgga_c_mn15_l,
  &xc_func_info_mgga_x_scan,
  &xc_func_info_mgga_c_scan,
  &xc_func_info_mgga_c_mn15,
  &xc_func_info_mgga_x_b00,
  &xc_func_info_mgga_xc_hle17,
  &xc_func_info_mgga_c_scan_rvv10,
  &xc_func_info_mgga_x_revm06_l,
  &xc_func_info_mgga_c_revm06_l,
  &xc_func_info_mgga_x_tm,
  &xc_func_info_mgga_x_vt84,
  &xc_func_info_mgga_x_sa_tpss,
  &xc_func_info_mgga_k_pc07,
  &xc_func_info_mgga_c_kcis,
  &xc_func_info_mgga_xc_lp90,
  &xc_func_info_mgga_c_b88,
  &xc_func_info_mgga_x_gx,
  &xc_func_info_mgga_x_pbe_gx,
  &xc_func_info_mgga_x_revscan,
  &xc_func_info_mgga_c_revscan,
  &xc_func_info_mgga_c_scan_vv10,
  &xc_func_info_mgga_c_revscan_vv10,
  &xc_func_info_mgga_x_br89_explicit,
  NULL
};
