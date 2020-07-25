#!/bin/sh

MAKEIN = ./make.in
include $(MAKEIN)

LIBXC_CUDA_FLAGS = -I$(libxcfolder)
#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
CUDALIBXCOBJ=$(libxcdevobjfolder)/gga_c_am05.o $(libxcdevobjfolder)/gga_c_bcgp.o $(libxcdevobjfolder)/gga_c_bmk.o $(libxcdevobjfolder)/gga_c_cs1.o \
	$(libxcdevobjfolder)/gga_c_ft97.o $(libxcdevobjfolder)/gga_c_gapc.o $(libxcdevobjfolder)/gga_c_gaploc.o $(libxcdevobjfolder)/gga_c_hcth_a.o \
	$(libxcdevobjfolder)/gga_c_lm.o $(libxcdevobjfolder)/gga_c_lyp.o $(libxcdevobjfolder)/gga_c_op_b88.o $(libxcdevobjfolder)/gga_c_op_g96.o \
	$(libxcdevobjfolder)/gga_c_op_pbe.o $(libxcdevobjfolder)/gga_c_op_pw91.o $(libxcdevobjfolder)/gga_c_optc.o $(libxcdevobjfolder)/gga_c_op_xalpha.o \
	$(libxcdevobjfolder)/gga_c_p86.o $(libxcdevobjfolder)/gga_c_pbe.o $(libxcdevobjfolder)/gga_c_pbeloc.o $(libxcdevobjfolder)/gga_c_pw91.o \
	$(libxcdevobjfolder)/gga_c_q2d.o $(libxcdevobjfolder)/gga_c_regtpss.o $(libxcdevobjfolder)/gga_c_revtca.o $(libxcdevobjfolder)/gga_c_scan_e0.o \
	$(libxcdevobjfolder)/gga_c_sg4.o $(libxcdevobjfolder)/gga_c_sogga11.o $(libxcdevobjfolder)/gga_c_tca.o $(libxcdevobjfolder)/gga_c_w94.o \
	$(libxcdevobjfolder)/gga_c_wi.o $(libxcdevobjfolder)/gga_c_wl.o $(libxcdevobjfolder)/gga_c_zpbeint.o $(libxcdevobjfolder)/gga_c_zvpbeint.o \
	$(libxcdevobjfolder)/gga_k_dk.o $(libxcdevobjfolder)/gga_k_exp4.o $(libxcdevobjfolder)/gga_k_meyer.o $(libxcdevobjfolder)/gga_k_ol1.o \
	$(libxcdevobjfolder)/gga_k_ol2.o $(libxcdevobjfolder)/gga_k_pearson.o $(libxcdevobjfolder)/gga_k_tflw.o $(libxcdevobjfolder)/gga_k_thakkar.o \
	$(libxcdevobjfolder)/gga_x_2d_b86.o $(libxcdevobjfolder)/gga_x_2d_b86_mgc.o $(libxcdevobjfolder)/gga_x_2d_b88.o $(libxcdevobjfolder)/gga_x_2d_pbe.o \
	$(libxcdevobjfolder)/gga_x_airy.o $(libxcdevobjfolder)/gga_x_ak13.o $(libxcdevobjfolder)/gga_x_am05.o $(libxcdevobjfolder)/gga_x_b86.o \
	$(libxcdevobjfolder)/gga_x_b88.o $(libxcdevobjfolder)/gga_x_bayesian.o $(libxcdevobjfolder)/gga_x_beefvdw.o $(libxcdevobjfolder)/gga_x_bpccac.o \
	$(libxcdevobjfolder)/gga_x_c09x.o $(libxcdevobjfolder)/gga_x_cap.o $(libxcdevobjfolder)/gga_xc_b97.o $(libxcdevobjfolder)/gga_x_chachiyo.o \
	$(libxcdevobjfolder)/gga_xc_th1.o $(libxcdevobjfolder)/gga_xc_th2.o $(libxcdevobjfolder)/gga_xc_th3.o $(libxcdevobjfolder)/gga_x_dk87.o \
	$(libxcdevobjfolder)/gga_x_eg93.o $(libxcdevobjfolder)/gga_x_ft97.o $(libxcdevobjfolder)/gga_x_g96.o $(libxcdevobjfolder)/gga_x_hcth_a.o \
	$(libxcdevobjfolder)/gga_x_herman.o $(libxcdevobjfolder)/gga_x_hjs_b88_v2.o $(libxcdevobjfolder)/gga_x_hjs.o $(libxcdevobjfolder)/gga_x_htbs.o \
	$(libxcdevobjfolder)/gga_x_kt.o $(libxcdevobjfolder)/gga_x_lag.o $(libxcdevobjfolder)/gga_x_lg93.o $(libxcdevobjfolder)/gga_x_lv_rpw86.o \
	$(libxcdevobjfolder)/gga_x_mpbe.o $(libxcdevobjfolder)/gga_x_n12.o $(libxcdevobjfolder)/gga_x_optx.o $(libxcdevobjfolder)/gga_x_pbea.o \
	$(libxcdevobjfolder)/gga_x_pbe.o $(libxcdevobjfolder)/gga_x_pbeint.o $(libxcdevobjfolder)/gga_x_pbepow.o $(libxcdevobjfolder)/gga_x_pbetrans.o \
	$(libxcdevobjfolder)/gga_x_pw86.o $(libxcdevobjfolder)/gga_x_pw91.o $(libxcdevobjfolder)/gga_x_q2d.o $(libxcdevobjfolder)/gga_x_rge2.o \
	$(libxcdevobjfolder)/gga_x_rpbe.o $(libxcdevobjfolder)/gga_x_sg4.o $(libxcdevobjfolder)/gga_x_sogga11.o $(libxcdevobjfolder)/gga_x_ssb_sw.o \
	$(libxcdevobjfolder)/gga_x_vmt84.o $(libxcdevobjfolder)/gga_x_vmt.o $(libxcdevobjfolder)/gga_x_wc.o $(libxcdevobjfolder)/hyb_gga_xc_wb97.o \
	$(libxcdevobjfolder)/lda_c_1d_csc.o $(libxcdevobjfolder)/lda_c_1d_loos.o $(libxcdevobjfolder)/lda_c_2d_amgb.o $(libxcdevobjfolder)/lda_c_2d_prm.o \
	$(libxcdevobjfolder)/lda_c_chachiyo.o $(libxcdevobjfolder)/lda_c_gk72.o $(libxcdevobjfolder)/lda_c_gombas.o $(libxcdevobjfolder)/lda_c_hl.o \
	$(libxcdevobjfolder)/lda_c_lp96.o $(libxcdevobjfolder)/lda_c_ml1.o $(libxcdevobjfolder)/lda_c_pk09.o $(libxcdevobjfolder)/lda_c_pw.o \
	$(libxcdevobjfolder)/lda_c_pz.o $(libxcdevobjfolder)/lda_c_rc04.o $(libxcdevobjfolder)/lda_c_rpa.o $(libxcdevobjfolder)/lda_c_vwn_1.o \
	$(libxcdevobjfolder)/lda_c_vwn_2.o $(libxcdevobjfolder)/lda_c_vwn_3.o $(libxcdevobjfolder)/lda_c_vwn_4.o $(libxcdevobjfolder)/lda_c_vwn.o \
	$(libxcdevobjfolder)/lda_c_vwn_rpa.o $(libxcdevobjfolder)/lda_c_wigner.o $(libxcdevobjfolder)/lda_k_tf.o $(libxcdevobjfolder)/lda_k_zlp.o \
	$(libxcdevobjfolder)/lda_x_2d.o $(libxcdevobjfolder)/lda_xc_1d_ehwlrg.o $(libxcdevobjfolder)/lda_xc_ksdt.o $(libxcdevobjfolder)/lda_xc_teter93.o \
	$(libxcdevobjfolder)/lda_x.o $(libxcdevobjfolder)/lda_xc_zlp.o $(libxcdevobjfolder)/lda_x_rel.o 
        #$(libxcdevobjfolder)/lda_x_erf.o $(libxcdevobjfolder)/hyb_mgga_xc_wb97mv.o $(libxcdevobjfolder)/hyb_mgga_x_dldf.o $(libxcdevobjfolder)/hyb_mgga_x_m05.o \
	$(libxcdevobjfolder)/mgga_c_b88.o $(libxcdevobjfolder)/mgga_c_bc95.o $(libxcdevobjfolder)/mgga_c_cs.o $(libxcdevobjfolder)/mgga_c_kcis.o \
	$(libxcdevobjfolder)/mgga_c_m05.o $(libxcdevobjfolder)/mgga_c_m06l.o $(libxcdevobjfolder)/mgga_c_m08.o $(libxcdevobjfolder)/mgga_c_pkzb.o \
	$(libxcdevobjfolder)/mgga_c_revscan.o $(libxcdevobjfolder)/mgga_c_revtpss.o $(libxcdevobjfolder)/mgga_c_scan.o $(libxcdevobjfolder)/mgga_c_tpss.o \
	$(libxcdevobjfolder)/mgga_c_tpssloc.o $(libxcdevobjfolder)/mgga_c_vsxc.o $(libxcdevobjfolder)/mgga_k_pc07.o $(libxcdevobjfolder)/mgga_x_br89_explicit.o \
	$(libxcdevobjfolder)/mgga_xc_b97mv.o $(libxcdevobjfolder)/mgga_xc_b98.o $(libxcdevobjfolder)/mgga_xc_cc06.o $(libxcdevobjfolder)/mgga_xc_lp90.o \
	$(libxcdevobjfolder)/mgga_xc_zlp.o $(libxcdevobjfolder)/mgga_x_gvt4.o $(libxcdevobjfolder)/mgga_x_gx.o $(libxcdevobjfolder)/mgga_x_lta.o \
	$(libxcdevobjfolder)/mgga_x_m06l.o $(libxcdevobjfolder)/mgga_x_m08.o $(libxcdevobjfolder)/mgga_x_m11.o $(libxcdevobjfolder)/mgga_x_m11_l.o \
	$(libxcdevobjfolder)/mgga_x_mbeef.o $(libxcdevobjfolder)/mgga_x_mbeefvdw.o $(libxcdevobjfolder)/mgga_x_mk00.o $(libxcdevobjfolder)/mgga_x_mn12.o \
	$(libxcdevobjfolder)/mgga_x_ms.o $(libxcdevobjfolder)/mgga_x_mvs.o $(libxcdevobjfolder)/mgga_x_pbe_gx.o $(libxcdevobjfolder)/mgga_x_pkzb.o \
	$(libxcdevobjfolder)/mgga_x_sa_tpss.o $(libxcdevobjfolder)/mgga_x_scan.o $(libxcdevobjfolder)/mgga_x_tau_hcth.o $(libxcdevobjfolder)/mgga_x_tm.o \
	$(libxcdevobjfolder)/mgga_x_tpss.o $(libxcdevobjfolder)/mgga_x_vt84.o

$(CUDALIBXCOBJ):$(libxcdevobjfolder)/%.o:%.cu
	$(CUDAC) -dc $(CUDA_FLAGS) $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -c $< -o $@

all: $(CUDALIBXCOBJ)

#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: 
	@-rm -f *.o 
	
