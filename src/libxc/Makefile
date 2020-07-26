#  !---------------------------------------------------------------------!
#  ! Refactored by Madu Manathunga on 07/25/2020                         !
#  !                                                                     !
#  ! Copyright (C) 2020-2021 Merz lab                                    !
#  ! Copyright (C) 2020-2021 Götz lab                                    !
#  !                                                                     !
#  ! This Source Code Form is subject to the terms of the Mozilla Public !
#  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
#  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
#  !_____________________________________________________________________!
#
#  !---------------------------------------------------------------------!
#  ! This Makefile builds a QUICK version from source code               !
#  !---------------------------------------------------------------------!

MAKEIN = ./make.in
include $(MAKEIN)

#  !---------------------------------------------------------------------!
#  ! Variable declarations                                               !
#  !---------------------------------------------------------------------!

FOR=$(FC) -c -I. $(FFLAGS) 

CCC = $(CC) -c -I. $(CCFLAGS) 

LIBXCOBJ = $(libxcobjfolder)/special_functions.o $(libxcobjfolder)/bessel.o\
	$(libxcobjfolder)/expint_e1.o $(libxcobjfolder)/integrate.o\
	$(libxcobjfolder)/util.o $(libxcobjfolder)/mix_func.o\
	$(libxcobjfolder)/func_info.o $(libxcobjfolder)/functionals.o\
	$(libxcobjfolder)/version.o $(libxcobjfolder)/func_reference.o\
	$(libxcobjfolder)/references.o $(libxcobjfolder)/lda.o $(libxcobjfolder)/lda_x.o\
	$(libxcobjfolder)/lda_x_1d.o $(libxcobjfolder)/lda_x_2d.o\
	$(libxcobjfolder)/lda_c_wigner.o $(libxcobjfolder)/lda_c_gombas.o\
	$(libxcobjfolder)/lda_c_rpa.o $(libxcobjfolder)/lda_c_hl.o\
	$(libxcobjfolder)/lda_c_vwn.o $(libxcobjfolder)/lda_c_vwn_1.o\
	$(libxcobjfolder)/lda_c_vwn_2.o $(libxcobjfolder)/lda_c_vwn_3.o\
	$(libxcobjfolder)/lda_c_vwn_4.o $(libxcobjfolder)/lda_c_vwn_rpa.o\
	$(libxcobjfolder)/lda_c_pz.o $(libxcobjfolder)/lda_c_pw.o\
	$(libxcobjfolder)/lda_c_pk09.o $(libxcobjfolder)/lda_c_lp96.o\
	$(libxcobjfolder)/lda_c_ml1.o $(libxcobjfolder)/lda_c_rc04.o\
	$(libxcobjfolder)/lda_xc_teter93.o $(libxcobjfolder)/lda_c_1d_csc.o\
	$(libxcobjfolder)/lda_c_1d_loos.o $(libxcobjfolder)/lda_c_2d_amgb.o\
	$(libxcobjfolder)/lda_c_2d_prm.o $(libxcobjfolder)/lda_k_tf.o\
	$(libxcobjfolder)/lda_xc_zlp.o $(libxcobjfolder)/lda_xc_ksdt.o\
	$(libxcobjfolder)/lda_c_chachiyo.o $(libxcobjfolder)/lda_xc_1d_ehwlrg.o\
	$(libxcobjfolder)/lda_x_rel.o $(libxcobjfolder)/lda_x_erf.o\
	$(libxcobjfolder)/lda_k_zlp.o $(libxcobjfolder)/lda_c_gk72.o\
	$(libxcobjfolder)/gga.o $(libxcobjfolder)/gga_x_lg93.o\
	$(libxcobjfolder)/gga_x_pbe.o $(libxcobjfolder)/gga_x_rpbe.o\
	$(libxcobjfolder)/gga_x_pbea.o $(libxcobjfolder)/gga_x_mpbe.o\
	$(libxcobjfolder)/gga_x_herman.o $(libxcobjfolder)/gga_x_b86.o\
	$(libxcobjfolder)/gga_x_b88.o $(libxcobjfolder)/gga_x_g96.o\
	$(libxcobjfolder)/gga_x_pw86.o $(libxcobjfolder)/gga_x_pw91.o\
	$(libxcobjfolder)/gga_x_optx.o $(libxcobjfolder)/gga_x_airy.o\
	$(libxcobjfolder)/gga_x_c09x.o $(libxcobjfolder)/gga_x_dk87.o\
	$(libxcobjfolder)/gga_x_ft97.o $(libxcobjfolder)/gga_x_wc.o\
	$(libxcobjfolder)/gga_x_am05.o $(libxcobjfolder)/gga_x_bayesian.o\
	$(libxcobjfolder)/gga_x_kt.o $(libxcobjfolder)/gga_x_htbs.o\
	$(libxcobjfolder)/gga_c_pbe.o $(libxcobjfolder)/gga_c_lyp.o\
	$(libxcobjfolder)/gga_c_p86.o $(libxcobjfolder)/gga_c_pw91.o\
	$(libxcobjfolder)/gga_c_am05.o $(libxcobjfolder)/gga_c_lm.o\
	$(libxcobjfolder)/gga_c_wl.o $(libxcobjfolder)/gga_c_wi.o\
	$(libxcobjfolder)/gga_x_lb.o $(libxcobjfolder)/gga_xc_b97.o\
	$(libxcobjfolder)/gga_xc_edf1.o $(libxcobjfolder)/gga_xc_1w.o\
	$(libxcobjfolder)/gga_c_optc.o $(libxcobjfolder)/gga_c_tca.o\
	$(libxcobjfolder)/gga_x_bpccac.o $(libxcobjfolder)/gga_x_sogga11.o\
	$(libxcobjfolder)/gga_c_sogga11.o $(libxcobjfolder)/gga_xc_th1.o\
	$(libxcobjfolder)/gga_x_wpbeh.o $(libxcobjfolder)/gga_x_hjs.o\
	$(libxcobjfolder)/gga_x_ityh.o $(libxcobjfolder)/gga_x_sfat.o\
	$(libxcobjfolder)/gga_x_ev93.o $(libxcobjfolder)/gga_x_ak13.o\
	$(libxcobjfolder)/gga_x_q2d.o $(libxcobjfolder)/gga_c_q2d.o\
	$(libxcobjfolder)/gga_x_ssb_sw.o $(libxcobjfolder)/gga_c_ft97.o\
	$(libxcobjfolder)/gga_x_n12.o $(libxcobjfolder)/gga_x_lv_rpw86.o\
	$(libxcobjfolder)/gga_x_2d_b86.o $(libxcobjfolder)/gga_x_2d_b86_mgc.o\
	$(libxcobjfolder)/gga_x_2d_b88.o $(libxcobjfolder)/gga_x_2d_pbe.o\
	$(libxcobjfolder)/gga_k_tflw.o $(libxcobjfolder)/gga_k_pearson.o\
	$(libxcobjfolder)/gga_k_ol1.o $(libxcobjfolder)/gga_k_ol2.o\
	$(libxcobjfolder)/gga_k_dk.o $(libxcobjfolder)/gga_k_meyer.o\
	$(libxcobjfolder)/gga_x_vmt.o $(libxcobjfolder)/gga_x_hcth_a.o\
	$(libxcobjfolder)/gga_x_cap.o $(libxcobjfolder)/gga_x_beefvdw.o\
	$(libxcobjfolder)/gga_x_sg4.o $(libxcobjfolder)/gga_x_gg99.o\
	$(libxcobjfolder)/gga_x_pbepow.o $(libxcobjfolder)/gga_x_lag.o\
	$(libxcobjfolder)/gga_k_thakkar.o $(libxcobjfolder)/gga_xc_oblyp_d.o\
	$(libxcobjfolder)/hyb_gga_xc_b3lyp.o $(libxcobjfolder)/hyb_gga_xc_o3lyp.o\
	$(libxcobjfolder)/gga_x_pbeint.o $(libxcobjfolder)/gga_x_rge2.o\
	$(libxcobjfolder)/gga_x_vmt84.o $(libxcobjfolder)/gga_x_hjs_b88_v2.o\
	$(libxcobjfolder)/gga_c_revtca.o $(libxcobjfolder)/gga_c_bcgp.o\
	$(libxcobjfolder)/gga_c_zpbeint.o $(libxcobjfolder)/gga_c_pbeloc.o\
	$(libxcobjfolder)/gga_c_sg4.o $(libxcobjfolder)/gga_c_regtpss.o\
	$(libxcobjfolder)/gga_c_scan_e0.o $(libxcobjfolder)/gga_xc_th2.o\
	$(libxcobjfolder)/gga_xc_th3.o $(libxcobjfolder)/gga_c_hcth_a.o\
	$(libxcobjfolder)/hyb_gga_xc_wb97.o $(libxcobjfolder)/gga_c_bmk.o\
	$(libxcobjfolder)/gga_c_gapc.o $(libxcobjfolder)/gga_c_gaploc.o\
	$(libxcobjfolder)/gga_c_zvpbeint.o $(libxcobjfolder)/gga_c_w94.o\
	$(libxcobjfolder)/gga_c_cs1.o $(libxcobjfolder)/gga_c_op_xalpha.o\
	$(libxcobjfolder)/gga_c_op_b88.o $(libxcobjfolder)/gga_c_op_pbe.o\
	$(libxcobjfolder)/gga_c_op_g96.o $(libxcobjfolder)/gga_c_op_pw91.o\
	$(libxcobjfolder)/gga_x_pbetrans.o $(libxcobjfolder)/gga_x_chachiyo.o\
	$(libxcobjfolder)/gga_k_exp4.o $(libxcobjfolder)/hyb_gga_xc_pbeh.o\
	$(libxcobjfolder)/hyb_gga_xc_b1wc.o $(libxcobjfolder)/hyb_gga_xc_hse.o\
	$(libxcobjfolder)/hyb_gga_xc_cam_b3lyp.o\
	$(libxcobjfolder)/hyb_gga_xc_camy_blyp.o\
	$(libxcobjfolder)/hyb_gga_xc_camy_b3lyp.o $(libxcobjfolder)/hyb_gga_xc_lcy_pbe.o\
	$(libxcobjfolder)/hyb_gga_xc_lcy_blyp.o $(libxcobjfolder)/gga_xc_vv10.o\
	$(libxcobjfolder)/hyb_gga_xc_edf2.o $(libxcobjfolder)/mgga.o\
	$(libxcobjfolder)/mgga_x_lta.o $(libxcobjfolder)/mgga_x_tpss.o\
	$(libxcobjfolder)/mgga_x_br89.o $(libxcobjfolder)/mgga_x_gvt4.o\
	$(libxcobjfolder)/mgga_x_m06l.o $(libxcobjfolder)/mgga_x_tau_hcth.o\
	$(libxcobjfolder)/mgga_x_2d_prhg07.o $(libxcobjfolder)/mgga_x_pkzb.o\
	$(libxcobjfolder)/hyb_mgga_x_m05.o $(libxcobjfolder)/mgga_x_m08.o\
	$(libxcobjfolder)/mgga_x_ms.o $(libxcobjfolder)/mgga_x_mn12.o\
	$(libxcobjfolder)/mgga_x_mk00.o $(libxcobjfolder)/mgga_c_vsxc.o\
	$(libxcobjfolder)/mgga_c_pkzb.o $(libxcobjfolder)/mgga_c_bc95.o\
	$(libxcobjfolder)/mgga_c_m08.o $(libxcobjfolder)/mgga_xc_cc06.o\
	$(libxcobjfolder)/mgga_c_cs.o $(libxcobjfolder)/mgga_xc_otpss_d.o\
	$(libxcobjfolder)/mgga_xc_zlp.o $(libxcobjfolder)/hyb_mgga_xc_b88b95.o\
	$(libxcobjfolder)/mgga_x_mbeef.o $(libxcobjfolder)/mgga_xc_b97mv.o\
	$(libxcobjfolder)/mgga_x_scan.o $(libxcobjfolder)/mgga_c_scan.o\
	$(libxcobjfolder)/hyb_mgga_xc_tpssh.o $(libxcobjfolder)/mgga_x_mvs.o\
	$(libxcobjfolder)/hyb_mgga_x_mvsh.o $(libxcobjfolder)/mgga_x_tm.o\
	$(libxcobjfolder)/mgga_x_vt84.o $(libxcobjfolder)/mgga_x_sa_tpss.o\
	$(libxcobjfolder)/mgga_k_pc07.o $(libxcobjfolder)/mgga_x_mbeefvdw.o\
	$(libxcobjfolder)/mgga_x_m11.o $(libxcobjfolder)/mgga_x_m11_l.o\
	$(libxcobjfolder)/hyb_mgga_xc_wb97mv.o $(libxcobjfolder)/mgga_c_m05.o\
	$(libxcobjfolder)/mgga_c_m06l.o $(libxcobjfolder)/hyb_mgga_x_dldf.o\
	$(libxcobjfolder)/mgga_c_tpss.o $(libxcobjfolder)/mgga_c_revtpss.o\
	$(libxcobjfolder)/mgga_c_tpssloc.o $(libxcobjfolder)/mgga_xc_hle17.o\
	$(libxcobjfolder)/mgga_c_kcis.o $(libxcobjfolder)/mgga_xc_lp90.o\
	$(libxcobjfolder)/hyb_mgga_xc_kcis.o $(libxcobjfolder)/mgga_c_b88.o\
	$(libxcobjfolder)/mgga_x_gx.o $(libxcobjfolder)/mgga_x_pbe_gx.o\
	$(libxcobjfolder)/mgga_c_revscan.o $(libxcobjfolder)/mgga_x_br89_explicit.o\
	$(libxcobjfolder)/mgga_xc_b98.o \

#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

$(LIBXCOBJ):$(libxcobjfolder)/%.o:%.c
	$(CCC) $< -o $@

libxc_f90:
	$(CCC) xc_f.c -o $(libxcobjfolder)/libxcf90_la-xc_f.o
	$(FOR) libxc_funcs.f90 -o $(libxcobjfolder)/libxcf90_la-libxc_funcs.o
	$(FOR) libxc.f90 -o $(libxcobjfolder)/libxcf90_la-libxc.o

libxc_gpu:$(LIBXCOBJ) libxc_f90
	$(CCC) gpu.c -o $(libxcobjfolder)/gpu.o
	$(ARCH) $(ARCHFLAGS) $(libfolder)/libxc.$(LIBEXT) $(libxcobjfolder)/*.o

libxc_cpu:$(LIBXCOBJ) libxc_f90
	$(ARCH) $(ARCHFLAGS) $(libfolder)/libxc.$(LIBEXT) $(libxcobjfolder)/*.o

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@-rm -f *.o *.a *.mod 
