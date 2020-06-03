#!/bin/sh

#include "config.h"
MAKEIN = ./make.in
include $(MAKEIN)

# --- Makefile for Quick Program ---
#  				- v 3.00 2019/03/30 Madu Manathunga
#				- v 2.00 2010/10/25 Yipu Miao
#				- v 1.18 2009/09/16 John Faver Exp $ 
#				- Makefile created by mkmf.pl $Id:
#	--------
#	 INDEX
#	--------
#	A. Compiler Setting			! Intel Fortran 9.0+ or GNU Fortran is recommended for single CPU Version
#	B. Make folders
#		! mpif90 is recommended for MPI Multi-CPU Version
#	C. Make Object Files		! Source files --> Object files
#	D. Make Executed files		! Object files --> Executed files
#	E. Self-defined Option		! Make option

#************************************************************************
#                  A. Compiler Settings
# 
#   FC specifies f90 compiler
#   FFLAGS are compliation options
#   LFLAGS are linking flags
#
#************************************************************************

# See compiler information and flag in make.in

#----------------------
# src file location
# #----------------------
#libxc = /users/PCS0202/bgs0374/Work/Project/Qk/libxc-4.3.4/lib
#
# #----------------------
# # obj file location
# #----------------------
# libxcinc = /users/PCS0202/bgs0374/Work/Project/Qk/libxc-4.3.4/include
#----------------------
# src file location
#----------------------
srcfolder = ./src

#----------------------
# obj file location
#----------------------
objfolder = ./obj

#----------------------
# exe file location
#----------------------
exefile = ./bin/quick

#----------------------
# exe folder location
#----------------------
exefolder = ./bin

#----------------------
# library file location
#----------------------
libfolder = ./lib

#----------------------
# config file location
#----------------------
configfolder = ./src/config

#----------------------
# subroutine file location
#----------------------
subfolder = ./src/subs

#----------------------
# BLAS file location
#----------------------
blasfolder = ./src/BLAS

#----------------------
# BLAS file location
# #----------------------
libxcfolder = ./src/libxc

#----------------------
# cuda files
#----------------------

cudafolder = ./src/cuda
cudaobj    =   $(objfolder)/gpu_write_info.o $(objfolder)/gpu.o $(objfolder)/gpu_type.o \
	$(objfolder)/gpu_get2e.o
cudaxcobj = $(objfolder)/gpu_getxc.o

cudalibxcobj=$(objfolder)/gga_c_am05.o $(objfolder)/gga_c_bcgp.o $(objfolder)/gga_c_bmk.o $(objfolder)/gga_c_cs1.o \
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
	$(objfolder)/lda_x.o $(objfolder)/lda_xc_zlp.o $(objfolder)/lda_x_rel.o 
#       $(objfolder)/lda_x_erf.o $(objfolder)/hyb_mgga_xc_wb97mv.o $(objfolder)/hyb_mgga_x_dldf.o $(objfolder)/hyb_mgga_x_m05.o \
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
cublasfolder    = $(cudafolder)/CUBLAS
cusolverfolder  = $(cudafolder)/CUSOLVER
cublasobj       = $(objfolder)/fortran_thunking.o
cusolverobj     = $(objfolder)/quick_cusolver.o 
#----------------------
# octree files
#----------------------
octfolder = ./src/octree
octobj    = $(objfolder)/grid_packer.o $(objfolder)/octree.o

#----------------------
# quick modules and object files
#----------------------
modfolder = ./src/modules

modobj= $(objfolder)/quick_mpi_module.o $(objfolder)/quick_constants_module.o $(objfolder)/quick_method_module.o \
        $(objfolder)/quick_molspec_module.o $(objfolder)/quick_gaussian_class_module.o $(objfolder)/quick_size_module.o \
        $(objfolder)/quick_amber_interface_module.o $(objfolder)/quick_basis_module.o $(objfolder)/quick_calculated_module.o \
        $(objfolder)/quick_divcon_module.o $(objfolder)/quick_ecp_module.o $(objfolder)/quick_electrondensity_module.o \
        $(objfolder)/quick_files_module.o $(objfolder)/quick_gridpoints_module.o $(objfolder)/quick_mfcc_module.o \
        $(objfolder)/quick_params_module.o $(objfolder)/quick_pb_module.o $(objfolder)/quick_scratch_module.o \
        $(objfolder)/quick_timer_module.o $(objfolder)/quick_all_module.o

OBJ =   $(objfolder)/main.o \
        $(objfolder)/initialize.o $(objfolder)/read_job_and_atom.o $(objfolder)/fmm.o \
        $(objfolder)/getMolSad.o $(objfolder)/getMol.o $(objfolder)/shell.o $(objfolder)/schwarz.o \
        $(objfolder)/quick_one_electron_integral.o $(objfolder)/getEnergy.o $(objfolder)/inidivcon.o \
        $(objfolder)/ecp.o $(objfolder)/hfoperator.o $(objfolder)/nuclear.o \
        $(objfolder)/dft.o $(objfolder)/sedftoperator.o $(objfolder)/dipole.o \
        $(objfolder)/scf.o $(objfolder)/uscf.o $(objfolder)/finalize.o $(objfolder)/uhfoperator.o \
        $(objfolder)/udftoperator.o $(objfolder)/usedftoperator.o \
        $(objfolder)/uelectdii.o $(objfolder)/mpi_setup.o $(objfolder)/quick_debug.o \
        $(objfolder)/calMP2.o $(objfolder)/optimize.o $(objfolder)/gradient.o $(objfolder)/hessian.o \
        $(objfolder)/CPHF.o $(objfolder)/frequency.o $(objfolder)/MFCC.o $(objfolder)/basis.o \
        $(objfolder)/fake_amber_interface.o $(objfolder)/scf_operator.o  


all: quick quick.cuda
#************************************************************************
# 
#                  B. Make necessary directories
#  
#************************************************************************
makefolders:
	mkdir -p $(objfolder) $(exefolder) $(libfolder)

#************************************************************************
# 
#                 C. Make Object Files
# 
#************************************************************************

#================= common subroutine library ============================
quick_subs:
	cd $(subfolder) && make all

#================= quick module library =================================
quick_modules:
	cd $(modfolder) && make all
#================= octree subroutines   =================================
octree:
	cd $(octfolder) && make all
#============= targets for cuda =========================================
quick_cuda:
	cd $(cudafolder) && make allbutxc
	cd $(cudafolder) && make xc 
		
#================= targets for BLAS =====================================
blas:
	cd $(blasfolder) && make
	cp $(blasfolder)/*.a $(libfolder)
#==================== libxc cpu library =================================
libxc_cpu:
	cd $(libxcfolder) && make libxc_cpu
#==================== libxc cpu library =================================
libxc_gpu:
	cd $(libxcfolder) && make libxc_gpu
	cd $(libxcfolder)/maple2c_device && make all
#=============== targets for CUBLAS =====================================

$(cublasobj):$(objfolder)/%.o:$(cublasfolder)/%.c
	$(CPP) $(CPP_FLAG) -c $< -o $@

#=============== targets for CUSOLVER ===================================

$(cusolverobj):$(objfolder)/%.o:$(cusolverfolder)/%.c
	$(CPP) $(CPP_FLAG) -c $< -o $@

#===================== target for main src files ========================

$(OBJ):$(objfolder)/%.o:$(srcfolder)/%.f90
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -I$(objfolder) -c $< -o $@

#==================== target configuration files ========================

cpconfig:
	cp $(configfolder)/config.h $(srcfolder)/config.h
cpconfig.cuda:
	cp $(configfolder)/config.cuda.h $(srcfolder)/config.h
cpconfig.cuda.SP:
	cp $(configfolder)/config.cuda.SP.h $(srcfolder)/config.h
cpconfig.cuda.MPI:
	cp $(configfolder)/config.cuda.MPI.h $(srcfolder)/config.h
cpconfig.MPI:
	cp $(configfolder)/config.MPI.h $(srcfolder)/config.h

#=========== targets for amber-quick interface ========================
# This is for amber-quick interface
# #amber_interface.o: amber_interface.f90 quick_modules qmmm_module.mod
# #       $(FC) -o $(objfolder)/amber_interface.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c  $(srcfolder)/amber_interface.f90
#
#===================== preprocess files ===============================
#quick_pprs:
#	$(FPP) $(srcfolder)/main.f90 > $(srcfolder)/main_.f90
#
#================= quick core subroutines ===============================


#**********************************************************************
# 
#                 C. Make Executables
# 
#**********************************************************************
quick: makefolders cpconfig libxc_cpu octree quick_modules quick_subs $(OBJ) blas 
	$(FC) -o $(exefolder)/quick $(OBJ) $(octobj) $(modobj) $(libfolder)/quicksublib.a $(libfolder)/blas.a \
	$(libfolder)/libxcf90.a $(libfolder)/libxc.a $(LDFLAGS) 

quick.cuda: makefolders cpconfig.cuda libxc_gpu octree quick_cuda quick_modules quick_subs $(OBJ) $(cusolverobj) $(cublasobj)
	$(FC) -o $(exefolder)/quick.cuda $(OBJ) $(octobj) $(modobj) $(objfolder)/gpu_xcall.o $(cudaobj) $(cudaxcobj) $(cudalibxcobj) \
	$(libfolder)/quicksublib.a $(cusolverobj) $(cublasobj) $(libfolder)/libxcf90.a $(libfolder)/libxc.a $(CFLAGS) $(LDFLAGS) 

quick.cuda.SP: makefolders cpconfig.cuda.SP quick_cuda quick_modules quick_subs quick_pprs $(OBJ) $(cusolverobj) $(cublasobj)
	$(FC) -o quick.cuda.SP $(OBJ) $(modobj) $(cudaobj) $(libfolder)/quicksublib.a $(cusolverobj) $(cublasobj) $(CFLAGS) 

quick.MPI: makefolders cpconfig.MPI libxc_cpu octree quick_modules quick_subs $(OBJ) blas 
	$(FC) -o $(exefolder)/quick.MPI  $(OBJ) $(octobj) $(modobj) $(libfolder)/quicksublib.a $(libfolder)/blas.a \
	$(libfolder)/libxcf90.a $(libfolder)/libxc.a $(LDFLAGS) 

quick.cuda.MPI: makefolders cpconfig.cuda.MPI libxc_gpu octree quick_cuda quick_modules quick_subs $(OBJ) blas $(cusolverobj) $(cublasobj)
	$(FC) -o $(exefolder)/quick.cuda.MPI  $(OBJ) $(octobj) $(modobj) $(objfolder)/gpu_xcall.o $(cudaobj) $(cudaxcobj) $(cudalibxcobj) \
	$(libfolder)/quicksublib.a $(cusolverobj) $(cublasobj) $(libfolder)/blas.a $(libfolder)/libxcf90.a $(libfolder)/libxc.a $(CFLAGS) $(LDFLAGS)

quick_lib:$(OBJ) ambermod amber_interface.o

quicklib: makefolders cpconfig libxc_cpu octree quick_modules quick_subs $(OBJ) blas
	ar -r $(libfolder)/quicklib.a $(objfolder)/*.o
	$(FC) -o test_quickapi.o test_quick_api_module.f90 test_quickapi.f90 -I$(objfolder) $(libfolder)/quicklib.a \
	$(libfolder)/quicksublib.a $(libfolder)/blas.a $(libfolder)/libxcf90.a $(libfolder)/libxc.a $(LDFLAGS) 

quickculib:makefolders cpconfig.cuda libxc_gpu octree quick_cuda quick_modules quick_subs $(OBJ) $(cusolverobj) $(cublasobj)
	ar -r $(libfolder)/quickculib.a $(objfolder)/*.o
	$(FC) -o test_quickapi.o test_quick_api_module.f90 test_quickapi.f90 -I$(objfolder) $(libfolder)/quickculib.a \
	$(libfolder)/quicksublib.a $(libfolder)/libxcf90.a $(libfolder)/libxc.a $(CFLAGS) $(LDFLAGS) 

quickmpilib: makefolders cpconfig.MPI libxc_cpu octree quick_modules quick_subs $(OBJ) blas
	ar -r $(libfolder)/quickmpilib.a $(objfolder)/*.o
	$(FC) -DQUAPI_MPIV -o test_quickapi.o test_quick_api_module.f90 test_quickapi.f90 -I$(objfolder) $(libfolder)/quickmpilib.a \
	$(libfolder)/quicksublib.a $(libfolder)/blas.a $(libfolder)/libxcf90.a $(libfolder)/libxc.a $(LDFLAGS)

quickcumpilib:makefolders cpconfig.cuda.MPI libxc_gpu octree quick_cuda quick_modules quick_subs $(OBJ) blas $(cusolverobj) $(cublasobj)
	ar -r $(libfolder)/quickcumpilib.a $(objfolder)/*.o
	$(FC) -DQUAPI_MPIV -o test_quickapi.o test_quick_api_module.f90 test_quickapi.f90 -I$(objfolder) $(libfolder)/quickcumpilib.a \
	$(libfolder)/blas.a $(libfolder)/quicksublib.a $(libfolder)/libxcf90.a $(libfolder)/libxc.a $(CFLAGS) $(LDFLAGS)

ambermod:
	cd ../../../AmberTools/src/sqm && $(MAKE) qmmm_module.o
	cp ../../../AmberTools/src/sqm/qmmm_module.mod .
	cp ../../../AmberTools/src/sqm/qmmm_vsolv_module.mod .
	cp ../../../AmberTools/src/sqm/qmmm_struct_module.mod .
	cp ../../../AmberTools/src/sqm/qmmm_nml_module.mod .
	cp ../../../AmberTools/src/sqm/qmmm_module.o .

		
#************************************************************************
# 
#                 D. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: neat
	-rm -f $(objfolder)/* $(libfolder)/* 
	cd $(cudafolder) && make clean
	cd $(subfolder) && make clean
	cd $(blasfolder) && make clean
	cd $(modfolder) && make clean
	cd $(libxcfolder) && make clean
	cd $(libxcfolder)/maple2c_device && make clean	
neat:
	-rm -f $(TMPFILES)

#Madu: Clean except libxc. Only for debugging
dryclean:
	-rm -f $(objfolder)/* $(libfolder)/*
	cd $(cudafolder) && make clean
	cd $(subfolder) && make clean
	cd $(blasfolder) && make clean
	cd $(modfolder) && make clean	

# - 2. Make tags for source files
TAGS: $(SRC)
	etags $(SRC)
tags: $(SRC)
	ctags $(SRC)

include $(srcfolder)/depend 
