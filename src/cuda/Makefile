#!/bin/sh

MAKEIN = ../../make.in
include $(MAKEIN)

#all: quick_cuda

objfolder = ../../obj
libxcfolder = ../libxc
#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
CUDACOBJ=$(objfolder)/gpu.o $(objfolder)/gpu_type.o $(objfolder)/gpu_get2e.o \
	$(objfolder)/gpu_getxc.o
CUDAFOBJ= $(objfolder)/gpu_write_info.o
#CUDALIBXCOBJ=$(objfolder)/gga_x_b88.o
#CXXOBJ=$(objfolder)/octree.o $(objfolder)/grid_packer.o
#CUDALIBXCOBJ=$(objfolder)/gpu_libxc.o

#$(CUDACOBJ):$(objfolder)/%.o:%.cu
#	$(CUDAC) $(CUDA_OPT_FLAGS) $(CUDA_ARCH_FLAGS) $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -c $< -o $@

$(CUDACOBJ):
	$(CUDAC) $(CUDA_OPT_FLAGS) $(CUDA_ARCH_FLAGS) $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -c gpu.cu -o $(objfolder)/gpu.o
	$(CUDAC) $(CUDA_OPT_FLAGS) $(CUDA_ARCH_FLAGS) $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -c gpu_type.cu -o $(objfolder)/gpu_type.o
	$(CUDAC) $(CUDA_OPT_FLAGS) $(CUDA_ARCH_FLAGS) $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -c gpu_get2e.cu -o $(objfolder)/gpu_get2e.o
	$(CUDAC) $(CUDA_OPT_FLAGS) $(CUDA_ARCH_FLAGS) -dc $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -c gpu_getxc.cu -o $(objfolder)/gpu_getxc.o

FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS)

$(CUDAFOBJ):$(objfolder)/%.o:%.f90
	$(FOR) -c $< -o $@

#$(CUDALIBXCOBJ):$(objfolder)/%.o:%.cu
#	$(CUDAC) $(CUDA_FLAGS2) $(LIBXC_CUDA_FLAGS) -c $< -o $@

#$(CXXOBJ):$(objfolder)/%.o:%.cpp
#	$(CXX) -c $< -o $@

#$(CUDALIBXCOBJ):$(objfolder)/%.o:%.cu
#	$(CUDAC) $(CUDA_FLAGS2) -I. -I$(libxcfolder) -c $< -o $@

#quick_cuda: 
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu.cu
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu_type.cu
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu_get2e.cu
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu_getxc.cu
#	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c gpu_write_info.f90
#all: $(CUDACOBJ) $(CUDAFOBJ) $(CUDALIBXCOBJ)
#all: $(CUDACOBJ) $(CUDAFOBJ) $(CXXOBJ)	
all: $(CUDACOBJ) $(CUDAFOBJ)
	$(CUDAC) $(CUDA_OPT_FLAGS) $(CUDA_ARCH_FLAGS) $(CUDA_INCL_FLAGS) $(LIBXC_CUDA_FLAGS) -dlink $(objfolder)/gpu.o \
	$(objfolder)/gpu_type.o $(objfolder)/gpu_get2e.o $(objfolder)/gpu_getxc.o $(objfolder)/gga_c_am05.o \
	$(objfolder)/gga_c_bcgp.o $(objfolder)/gga_c_bmk.o $(objfolder)/gga_c_cs1.o \
	$(objfolder)/gga_c_ft97.o $(objfolder)/gga_c_gapc.o $(objfolder)/gga_c_gaploc.o $(objfolder)/gga_c_hcth_a.o \
	$(objfolder)/gga_c_lm.o $(objfolder)/gga_c_lyp.o $(objfolder)/gga_c_op_b88.o $(objfolder)/gga_c_op_g96.o \
	$(objfolder)/gga_c_op_pbe.o $(objfolder)/gga_c_op_pw91.o $(objfolder)/gga_c_optc.o $(objfolder)/gga_c_op_xalpha.o \
	$(objfolder)/gga_c_p86.o $(objfolder)/gga_c_pbe.o $(objfolder)/gga_c_pbeloc.o $(objfolder)/gga_c_pw91.o \
	$(objfolder)/gga_c_q2d.o $(objfolder)/gga_c_regtpss.o $(objfolder)/gga_c_revtca.o $(objfolder)/gga_c_scan_e0.o \
	$(objfolder)/gga_c_sg4.o $(objfolder)/gga_c_sogga11.o $(objfolder)/gga_c_tca.o $(objfolder)/gga_c_w94.o \
	$(objfolder)/gga_c_wi.o $(objfolder)/gga_c_wl.o $(objfolder)/gga_c_zpbeint.o $(objfolder)/gga_c_zvpbeint.o \
	$(objfolder)/gga_k_dk.o $(objfolder)/gga_k_exp4.o $(objfolder)/gga_k_meyer.o $(objfolder)/gga_k_ol1.o \
	$(objfolder)/gga_k_ol2.o $(objfolder)/gga_k_pearson.o $(objfolder)/gga_k_tflw.o $(objfolder)/gga_k_thakkar.o \
	$(objfolder)/gga_x_2d_b86.o $(objfolder)/gga_x_2d_b86_mgc.o $(objfolder)/gga_x_2d_b88.o $(objfolder)/gga_x_2d_pbe.o \
	$(objfolder)/gga_x_airy.o $(objfolder)/gga_x_ak13.o $(objfolder)/gga_x_am05.o $(objfolder)/gga_x_b86.o \
	$(objfolder)/gga_x_b88.o $(objfolder)/gga_x_bayesian.o $(objfolder)/gga_x_beefvdw.o $(objfolder)/gga_x_bpccac.o \
	$(objfolder)/gga_x_c09x.o $(objfolder)/gga_x_cap.o $(objfolder)/gga_xc_b97.o $(objfolder)/gga_x_chachiyo.o \
	$(objfolder)/gga_xc_th1.o $(objfolder)/gga_xc_th2.o $(objfolder)/gga_xc_th3.o $(objfolder)/gga_x_dk87.o \
	$(objfolder)/gga_x_eg93.o $(objfolder)/gga_x_ft97.o $(objfolder)/gga_x_g96.o $(objfolder)/gga_x_hcth_a.o \
	$(objfolder)/gga_x_herman.o $(objfolder)/gga_x_hjs_b88_v2.o $(objfolder)/gga_x_hjs.o $(objfolder)/gga_x_htbs.o \
	$(objfolder)/gga_x_kt.o $(objfolder)/gga_x_lag.o $(objfolder)/gga_x_lg93.o $(objfolder)/gga_x_lv_rpw86.o \
	$(objfolder)/gga_x_mpbe.o $(objfolder)/gga_x_n12.o $(objfolder)/gga_x_optx.o $(objfolder)/gga_x_pbea.o \
	$(objfolder)/gga_x_pbe.o $(objfolder)/gga_x_pbeint.o $(objfolder)/gga_x_pbepow.o $(objfolder)/gga_x_pbetrans.o \
	$(objfolder)/gga_x_pw86.o $(objfolder)/gga_x_pw91.o $(objfolder)/gga_x_q2d.o $(objfolder)/gga_x_rge2.o \
	$(objfolder)/gga_x_rpbe.o $(objfolder)/gga_x_sg4.o $(objfolder)/gga_x_sogga11.o $(objfolder)/gga_x_ssb_sw.o \
	$(objfolder)/gga_x_vmt84.o $(objfolder)/gga_x_vmt.o $(objfolder)/gga_x_wc.o $(objfolder)/hyb_gga_xc_wb97.o \
	$(objfolder)/lda_c_1d_csc.o $(objfolder)/lda_c_1d_loos.o $(objfolder)/lda_c_2d_amgb.o $(objfolder)/lda_c_2d_prm.o \
	$(objfolder)/lda_c_chachiyo.o $(objfolder)/lda_c_gk72.o $(objfolder)/lda_c_gombas.o $(objfolder)/lda_c_hl.o \
	$(objfolder)/lda_c_lp96.o $(objfolder)/lda_c_ml1.o $(objfolder)/lda_c_pk09.o $(objfolder)/lda_c_pw.o \
	$(objfolder)/lda_c_pz.o $(objfolder)/lda_c_rc04.o $(objfolder)/lda_c_rpa.o $(objfolder)/lda_c_vwn_1.o \
	$(objfolder)/lda_c_vwn_2.o $(objfolder)/lda_c_vwn_3.o $(objfolder)/lda_c_vwn_4.o $(objfolder)/lda_c_vwn.o \
	$(objfolder)/lda_c_vwn_rpa.o $(objfolder)/lda_c_wigner.o $(objfolder)/lda_k_tf.o $(objfolder)/lda_k_zlp.o \
	$(objfolder)/lda_x_2d.o $(objfolder)/lda_xc_1d_ehwlrg.o $(objfolder)/lda_xc_ksdt.o $(objfolder)/lda_xc_teter93.o \
	$(objfolder)/lda_x.o $(objfolder)/lda_xc_zlp.o $(objfolder)/lda_x_rel.o -o $(objfolder)/gpu_all.o
#	$(objfolder)/lda_x_erf.o $(objfolder)/hyb_mgga_xc_wb97mv.o $(objfolder)/hyb_mgga_x_dldf.o $(objfolder)/hyb_mgga_x_m05.o \
	$(objfolder)/mgga_c_b88.o $(objfolder)/mgga_c_bc95.o $(objfolder)/mgga_c_cs.o $(objfolder)/mgga_c_kcis.o \
	$(objfolder)/mgga_c_m05.o $(objfolder)/mgga_c_m06l.o $(objfolder)/mgga_c_m08.o $(objfolder)/mgga_c_pkzb.o \
	$(objfolder)/mgga_c_revscan.o $(objfolder)/mgga_c_revtpss.o $(objfolder)/mgga_c_scan.o $(objfolder)/mgga_c_tpss.o \
	$(objfolder)/mgga_c_tpssloc.o $(objfolder)/mgga_c_vsxc.o $(objfolder)/mgga_k_pc07.o $(objfolder)/mgga_x_br89_explicit.o \
	$(objfolder)/mgga_xc_b97mv.o $(objfolder)/mgga_xc_b98.o $(objfolder)/mgga_xc_cc06.o $(objfolder)/mgga_xc_lp90.o \
	$(objfolder)/mgga_xc_zlp.o $(objfolder)/mgga_x_gvt4.o $(objfolder)/mgga_x_gx.o $(objfolder)/mgga_x_lta.o \
	$(objfolder)/mgga_x_m06l.o $(objfolder)/mgga_x_m08.o $(objfolder)/mgga_x_m11.o $(objfolder)/mgga_x_m11_l.o \
	$(objfolder)/mgga_x_mbeef.o $(objfolder)/mgga_x_mbeefvdw.o $(objfolder)/mgga_x_mk00.o $(objfolder)/mgga_x_mn12.o \
	$(objfolder)/mgga_x_ms.o $(objfolder)/mgga_x_mvs.o $(objfolder)/mgga_x_pbe_gx.o $(objfolder)/mgga_x_pkzb.o \
	$(objfolder)/mgga_x_sa_tpss.o $(objfolder)/mgga_x_scan.o $(objfolder)/mgga_x_tau_hcth.o $(objfolder)/mgga_x_tm.o \
	$(objfolder)/mgga_x_tpss.o $(objfolder)/mgga_x_vt84.o

#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: 
	-rm -f *.o 
	
