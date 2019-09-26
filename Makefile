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
cudaobj    =   $(objfolder)/gpu_write_info.o $(objfolder)/gpu.o $(objfolder)/gpu_type.o $(objfolder)/gpu_getxc.o \
	       $(objfolder)/gpu_get2e.o 
cublasfolder  =$(cudafolder)/CUBLAS
cublasobj     =$(objfolder)/fortran_thunking.o
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
#============= targets for cuda =========================================
quick_cuda:
	cd $(cudafolder) && make all 
	
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
#=============== targets for CUBLAS =====================================
$(cublasobj):$(objfolder)/%.o:$(cublasfolder)/%.c
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
quick: makefolders cpconfig libxc_cpu quick_modules quick_subs $(OBJ) blas 
	$(FC) -o $(exefolder)/quick $(OBJ) $(modobj) $(libfolder)/quicklib.a $(libfolder)/blas.a \
	$(libfolder)/libxcf90.a $(libfolder)/libxc.a $(LDFLAGS) 

quick.cuda: makefolders cpconfig.cuda libxc_gpu quick_cuda quick_modules quick_subs $(OBJ) $(cublasobj)
	$(FC) -o $(exefolder)/quick.cuda $(OBJ) $(modobj) $(cudaobj) $(libfolder)/quicklib.a $(cublasobj) \
	$(libfolder)/libxcf90.a $(libfolder)/libxc.a $(CFLAGS) $(LDFLAGS) 

quick.cuda.SP: makefolders cpconfig.cuda.SP quick_cuda quick_modules quick_subs quick_pprs $(OBJ) $(cublasobj)
	$(FC) -o quick.cuda.SP $(OBJ) $(modobj) $(cudaobj) $(libfolder)/quicklib.a $(cublasobj) $(CFLAGS) 

quick.MPI: makefolders cpconfig.MPI libxc_cpu quick_modules quick_subs $(OBJ) blas 
	$(FC) -o $(exefolder)/quick.MPI  $(OBJ) $(modobj) $(libfolder)/quicklib.a $(libfolder)/blas.a \
	$(libfolder)/libxcf90.a $(libfolder)/libxc.a $(LDFLAGS) 

quick_lib:$(OBJ) ambermod amber_interface.o

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
	-rm -f $(objfolder)/* $(exefolder)/quick* $(libfolder)/* 
	cd $(cudafolder) && make clean
	cd $(subfolder) && make clean
	cd $(blasfolder) && make clean
	cd $(modfolder) && make clean
	cd $(libxcfolder) && make clean	
neat:
	-rm -f $(TMPFILES)

#Madu: Clean except libxc. Only for debugging
dryclean:
	-rm -f $(objfolder)/* $(exefolder)/quick* $(libfolder)/*
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
