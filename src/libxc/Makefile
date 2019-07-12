#!/bin/sh

MAKEIN = ../../make.in
include $(MAKEIN)

# --- Makefile for libxc  ---
#  				- v 1.00 2019/06/03 Madu Manathunga
#	--------
#	 INDEX
#	--------
#	A. Compiler Setting			! Intel Fortran 9.0+ or GNU Fortran is recommended for single CPU Version
#								! mpif90 is recommended for MPI Multi-CPU Version
#	B. Make Object Files		! Source files --> Object files
#	C. Make Executed files		! Object files --> Executed files
#	D. Self-defined Option		! Make option

#************************************************************************
#                  A. Compiler Settings
#
#   CC specifies gcc/icc compiler
#   FOR specifies f90 compiler
#   FFLAGS are compliation options
#   
#
#************************************************************************

#Madu: intel and gnu use -module and -J options to direct module files. 
#Therefore, we use the following if condition to handle this. 

ifeq "$(FC)" "gfortran"
  FOR=$(FC) -c -I. -I$(objfolder) -J$(objfolder) 
else
  FOR = $(FC) -c -I. -O2 -module $(objfolder)
endif

CCC = $(CC) -c -I. -O2 -g
CUDACC = $(CUDAC) -c -I. -O2 -g
 
#----------------------
# quick modules and object files
#----------------------
libfolder = ../../lib
objfolder = ../../obj

SUBS =  special_functions.o bessel.o expint_e1.o integrate.o util.o     \
        mix_func.o func_info.o functionals.o version.o func_reference.o \
        references.o lda.o \
        lda_x.o lda_x_1d.o lda_x_2d.o lda_c_wigner.o lda_c_gombas.o     \
        lda_c_rpa.o lda_c_hl.o lda_c_vwn.o lda_c_vwn_1.o lda_c_vwn_2.o  \
        lda_c_vwn_3.o lda_c_vwn_4.o lda_c_vwn_rpa.o lda_c_pz.o          \
        lda_c_pw.o lda_c_pk09.o lda_c_lp96.o \
        lda_c_ml1.o lda_c_rc04.o lda_xc_teter93.o lda_c_1d_csc.o        \
        lda_c_1d_loos.o lda_c_2d_amgb.o lda_c_2d_prm.o          \
        lda_k_tf.o lda_xc_zlp.o lda_xc_ksdt.o lda_c_chachiyo.o          \
        lda_xc_1d_ehwlrg.o lda_x_rel.o lda_x_erf.o lda_k_zlp.o          \
        lda_c_gk72.o \
        gga.o \
        gga_x_lg93.o gga_x_pbe.o gga_x_rpbe.o gga_x_pbea.o              \
        gga_x_mpbe.o gga_x_herman.o gga_x_b86.o gga_x_b88.o             \
        gga_x_g96.o gga_x_pw86.o gga_x_pw91.o gga_x_optx.o              \
        gga_x_airy.o gga_x_c09x.o gga_x_dk87.o gga_x_ft97.o             \
        gga_x_wc.o gga_x_am05.o gga_x_bayesian.o gga_x_kt.o             \
        gga_x_htbs.o gga_c_pbe.o gga_c_lyp.o gga_c_p86.o gga_c_pw91.o   \
        gga_c_am05.o gga_c_lm.o gga_c_wl.o gga_c_wi.o gga_x_lb.o        \
        gga_xc_b97.o gga_xc_edf1.o gga_xc_1w.o gga_c_optc.o             \
        gga_c_tca.o gga_x_bpccac.o gga_x_sogga11.o gga_c_sogga11.o      \
        gga_xc_th1.o gga_x_wpbeh.o gga_x_hjs.o gga_x_ityh.o             \
        gga_x_sfat.o gga_x_ev93.o gga_x_ak13.o gga_x_q2d.o gga_c_q2d.o  \
        gga_x_ssb_sw.o gga_c_ft97.o gga_x_n12.o gga_x_lv_rpw86.o        \
        gga_x_2d_b86.o gga_x_2d_b86_mgc.o               \
        gga_x_2d_b88.o gga_x_2d_pbe.o gga_k_tflw.o gga_k_pearson.o      \
        gga_k_ol1.o gga_k_ol2.o gga_k_dk.o gga_k_meyer.o gga_x_vmt.o    \
        gga_x_hcth_a.o gga_x_cap.o gga_x_beefvdw.o gga_x_sg4.o          \
        gga_x_gg99.o gga_x_pbepow.o gga_x_lag.o gga_k_thakkar.o         \
        gga_xc_oblyp_d.o hyb_gga_xc_b3lyp.o hyb_gga_xc_o3lyp.o          \
        gga_x_pbeint.o gga_x_rge2.o gga_x_vmt84.o gga_x_hjs_b88_v2.o    \
        gga_c_revtca.o gga_c_bcgp.o gga_c_zpbeint.o gga_c_pbeloc.o      \
        gga_c_sg4.o gga_c_regtpss.o gga_c_scan_e0.o gga_xc_th2.o        \
        gga_xc_th3.o gga_c_hcth_a.o hyb_gga_xc_wb97.o gga_c_bmk.o       \
        gga_c_gapc.o gga_c_gaploc.o gga_c_zvpbeint.o gga_c_w94.o        \
        gga_c_cs1.o gga_c_op_xalpha.o gga_c_op_b88.o gga_c_op_pbe.o     \
        gga_c_op_g96.o gga_c_op_pw91.o gga_x_pbetrans.o \
        gga_x_chachiyo.o gga_k_exp4.o \
        hyb_gga_xc_pbeh.o hyb_gga_xc_b1wc.o                             \
        hyb_gga_xc_hse.o hyb_gga_xc_cam_b3lyp.o                         \
        hyb_gga_xc_camy_blyp.o hyb_gga_xc_camy_b3lyp.o                  \
        hyb_gga_xc_lcy_pbe.o hyb_gga_xc_lcy_blyp.o gga_xc_vv10.o        \
        hyb_gga_xc_edf2.o                                               \
        mgga.o \
        mgga_x_lta.o mgga_x_tpss.o mgga_x_br89.o mgga_x_gvt4.o          \
        mgga_x_m06l.o mgga_x_tau_hcth.o mgga_x_2d_prhg07.o                  \
        mgga_x_pkzb.o hyb_mgga_x_m05.o mgga_x_m08.o mgga_x_ms.o         \
        mgga_x_mn12.o mgga_x_mk00.o mgga_c_vsxc.o mgga_c_pkzb.o         \
        mgga_c_bc95.o mgga_c_m08.o mgga_xc_cc06.o mgga_c_cs.o             \
        mgga_xc_otpss_d.o mgga_xc_zlp.o hyb_mgga_xc_b88b95.o              \
        mgga_x_mbeef.o mgga_xc_b97mv.o mgga_x_scan.o mgga_c_scan.o      \
        hyb_mgga_xc_tpssh.o mgga_x_mvs.o hyb_mgga_x_mvsh.o              \
        mgga_x_tm.o mgga_x_vt84.o mgga_x_sa_tpss.o mgga_k_pc07.o        \
        mgga_x_mbeefvdw.o mgga_x_m11.o mgga_x_m11_l.o                   \
        hyb_mgga_xc_wb97mv.o mgga_c_m05.o mgga_c_m06l.o                 \
        hyb_mgga_x_dldf.o mgga_c_tpss.o mgga_c_revtpss.o                \
        mgga_c_tpssloc.o mgga_xc_hle17.o mgga_c_kcis.o mgga_xc_lp90.o   \
        hyb_mgga_xc_kcis.o mgga_c_b88.o mgga_x_gx.o mgga_x_pbe_gx.o     \
        mgga_c_revscan.o mgga_x_br89_explicit.o mgga_xc_b98.o gpu.o


GPUOBJ = test.o
#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************


$(SUBS):%.o:%.c
	$(CCC) $< -o $@
$(GPUOBJ):%.o:%.cu
	$(CUDACC) $< -o $@
libxcf90_la-xc_f.o:
	$(CCC) xc_f.c -o libxcf90_la-xc_f.o
libxcf90_la-libxc_funcs.o:
	$(FOR) libxc_funcs.f90 -o libxcf90_la-libxc_funcs.o
libxcf90_la-libxc.o:
	$(FOR) libxc.f90 -o libxcf90_la-libxc.o

#================= quick subroutine library =================================

#libxc_libs:$(SUBS) libxcf90_la-xc_f.o libxcf90_la-libxc_funcs.o libxcf90_la-libxc.o
libxc_libs:$(SUBS) $(GPUOBJ) libxcf90_la-xc_f.o libxcf90_la-libxc_funcs.o libxcf90_la-libxc.o
	ar -r $(libfolder)/libxc.a *.o
	ar cru $(libfolder)/libxcf90.a  libxcf90_la-xc_f.o libxcf90_la-libxc_funcs.o libxcf90_la-libxc.o	

all: libxc_libs
#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean:
	-rm -f *.o *.a *.mod 
