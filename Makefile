#!/bin/sh

#include "config.h"
MAKEIN = ./make.in
include $(MAKEIN)

# --- Makefile for Quick Program ---
#				- v 2.00 2010/10/25 Yipu Miao
#				- v 1.18 2009/09/16 John Faver Exp $ 
#				- Makefile created by mkmf.pl $Id:
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
#   FC specifies f90 compiler
#   FFLAGS are compliation options
#   LFLAGS are linking flags
#
#************************************************************************

# See compiler information and flag in make.in

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
# cuda files
#----------------------



cudafolder = ./src/cuda
cudaobj    = gpu_write_info.o gpu.o gpu_type.o gpu_getxc.o gpu_get2e.o #vertical.o #$(CUDA_INT_OBJ)
#cudafile   = gpu.cu gpu_type.cu gpu_get2e.cu gpu_getxc.cu

#----------------------
# quick modules and object files
#----------------------
modfolder = ./src/modules

mpimod  = quick_mpi_module.f90
mpimod_o= quick_mpi_module.o

modfile0 = quick_constants_module.f90
modobj0  = quick_constants_module.o
modfile1 = quick_method_module.f90 quick_molspec_module.f90 quick_gaussian_class_module.f90 
modobj1  = quick_method_module.o quick_molspec_module.o quick_gaussian_class_module.o
modfile2 = quick_size_module.f90 quick_amber_interface_module.f90 quick_basis_module.f90 \
		   quick_calculated_module.f90 quick_divcon_module.f90 \
		   quick_ecp_module.f90 quick_electrondensity_module.f90 quick_files_module.f90 \
		   quick_gridpoints_module.f90 \
		   quick_mfcc_module.f90 quick_params_module.f90 quick_pb_module.f90 \
		   quick_scratch_module.f90 quick_timer_module.f90 quick_all_module.f90
modobj2  = quick_size_module.o quick_amber_interface_module.o quick_basis_module.o \
		   quick_calculated_module.o quick_divcon_module.o \
		   quick_ecp_module.o quick_electrondensity_module.o quick_files_module.o \
		   quick_gridpoints_module.o \
		   quick_mfcc_module.o quick_params_module.o quick_pb_module.o \
		   quick_scratch_module.o quick_timer_module.o quick_all_module.o
modobjall = $(mpimod_o) ${modobj0} ${modobj1} ${modobj2} 

#.DEFAULT:
#	-touch $@

all: quick quick.cuda

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= common subroutine library ============================
quick_subs:
	cp $(objfolder)/*.mod $(subfolder)
	cd $(subfolder) && $(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c *.f90
	ar -r $(libfolder)/quicklib.a $(subfolder)/*.o
	rm $(subfolder)/*.mod

#================= quick module library =================================
quick_modules:
	cd $(modfolder) && make
	cp $(modfolder)/*.mod $(srcfolder)
	mv $(modfolder)/*.mod $(modfolder)/*.o $(objfolder)
#=========== targets for cuda =========================================
quick_cuda:
#	cd $(cudafolder) && $(CUDAC) $(CUDA_FLAGS) -c gpu.cu
#	cd $(cudafolder) && $(CUDAC) $(CUDA_FLAGS) -c gpu_get2e.cu 
#	cd $(cudafolder) && $(CUDAC) $(CUDA_FLAGS) -c $(cudafile)
#	cd $(cudafolder) && $(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c gpu_write_info.f90
	cd $(cudafolder) && make all 
	cp $(cudafolder)/*.o $(objfolder)
	
#================= quick core subroutines ===============================
main.o: quick_modules
	$(FPP) $(srcfolder)/main.f90 > $(objfolder)/_main.f90
	$(FC) -o $(objfolder)/main.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c   $(objfolder)/_main.f90

#================= quick core subroutines ===============================
#optimize.o: quick_modules
#	$(FPP) $(srcfolder)/optimize.f90 > $(objfolder)/_optimize.f90
#	$(FC) -o $(objfolder)/optimize.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c   $(objfolder)/_optimize.f90

#================= quick dft gradient subroutines ===============================
dftgrad.o: quick_modules
	$(FPP) $(srcfolder)/dftgrad.f90 > $(objfolder)/_dftgrad.f90
	$(FC) -o $(objfolder)/dftgrad.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c   $(objfolder)/_dftgrad.f90

#================= quick pt2der subroutines ===============================
pt2der.o: quick_modules
	$(FPP) $(srcfolder)/pt2der.f90 > $(objfolder)/_pt2der.f90
	$(FC) -o $(objfolder)/pt2der.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c   $(objfolder)/_pt2der.f90

#================= quick 2eshelloptdft.o subroutines ===============================
2eshelloptdft.o: quick_modules
	$(FPP) $(srcfolder)/2eshelloptdft.f90 > $(objfolder)/_2eshelloptdft.f90
	$(FC) -o $(objfolder)/2eshelloptdft.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c   $(objfolder)/_2eshelloptdft.f90

#================= quick sswder.o subroutines ===============================
sswder.o: quick_modules
	$(FPP) $(srcfolder)/sswder.f90 > $(objfolder)/_sswder.f90
	$(FC) -o $(objfolder)/sswder.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c   $(objfolder)/_sswder.f90

#=========== targets for BLAS =====================================
blas:
	cd $(blasfolder) && make
	cp $(blasfolder)/*.a $(libfolder)

#=========== targets for CUBLAS =====================================
fortran_thunking.o:
	cd $(cudafolder)/CUBLAS && $(CPP) $(CPP_FLAG) -c  fortran_thunking.c
	cp $(cudafolder)/CUBLAS/fortran_thunking.o $(objfolder)

#=========== targets for amber-quick interface ========================
# This is a fake amber-quick interface
fake_amber_interface.o: 
	$(FC) -o $(objfolder)/fake_amber_interface.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c  $(srcfolder)/fake_amber_interface.f90

# This is for amber-quick interface
amber_interface.o: amber_interface.f90 quick_modules qmmm_module.mod
	$(FC) -o $(objfolder)/amber_interface.o $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c  $(srcfolder)/amber_interface.f90

									
OBJ =   main.o \
		initialize.o read_job_and_atom.o fmm.o \
		getMolSad.o getMol.o shell.o schwarz.o quick_one_electron_integral.o \
		getEnergy.o inidivcon.o ecp.o hfoperator.o nuclear.o \
		dft.o sedftoperator.o dipole.o \
		scf.o uscf.o finalize.o uhfoperator.o udftoperator.o usedftoperator.o \
		uelectdii.o mpi_setup.o quick_debug.o calMP2.o optimize.o \
		gradient.o hessian.o CPHF.o frequency.o MFCC.o basis.o dftgrad.o pt2der.o 2eshelloptdft.o sswder.o
cpconfig:
	cp $(configfolder)/config.h $(srcfolder)/config.h
cpconfig.cuda:
	cp $(configfolder)/config.cuda.h $(srcfolder)/config.h
cpconfig.cuda.SP:
	cp $(configfolder)/config.cuda.SP.h $(srcfolder)/config.h
cpconfig.MPI:
	cp $(configfolder)/config.MPI.h $(srcfolder)/config.h

#************************************************************************
# 
#                 C. Make Executed Files
# 
#************************************************************************

quick: cpconfig quick_modules quick_subs $(OBJ) blas fake_amber_interface.o
	mkdir -p $(libfolder) $(objfolder) $(exefolder) 
	cp $(libfolder)/quicklib.a $(objfolder)
	cp $(libfolder)/blas.a $(objfolder)
	cd $(objfolder) && $(FC) -o quick  $(OBJ) $(modobjall) quicklib.a blas.a fake_amber_interface.o $(LDFLAGS)
	mv $(objfolder)/quick $(exefile)	
	rm -f $(srcfolder)/*.mod $(srcfolder)/*.o

quick.cuda: cpconfig.cuda quick_cuda quick_modules quick_subs $(OBJ) fake_amber_interface.o fortran_thunking.o
	mkdir -p $(libfolder) $(objfolder) $(exefolder) 
	cp $(libfolder)/quicklib.a $(objfolder)
	cd $(objfolder) && $(FC) -o quick.cuda $(OBJ) $(modobjall) $(cudaobj) quicklib.a fake_amber_interface.o fortran_thunking.o $(CFLAGS) $(LDFLAGS) 
	mv $(objfolder)/quick.cuda $(exefile).cuda 
	rm -f $(srcfolder)/*.mod $(srcfolder)/*.o

quick.cuda.SP: cpconfig.cuda.SP quick_cuda quick_modules quick_subs $(OBJ) fake_amber_interface.o fortran_thunking.o
	cp $(libfolder)/quicklib.a $(objfolder)
	cd $(objfolder) && $(FC) -o quick.cuda.SP $(OBJ) $(modobjall) $(cudaobj) $(libfolder)/quicklib.a fake_amber_interface.o fortran_thunking.o $(CFLAGS) 
	mv $(objfolder)/quick.cuda.SP $(exefile).cuda.SP
	rm -f $(srcfolder)/*.mod $(srcfolder)/*.o

quick.MPI: cpconfig.MPI quick_modules quick_subs $(OBJ) blas fake_amber_interface.o
	cp $(libfolder)/quicklib.a $(objfolder)
	cp $(libfolder)/blas.a $(objfolder)
	cd $(objfolder) && $(FC) -o quick.MPI  $(OBJ) $(modobjall) quicklib.a blas.a fake_amber_interface.o $(LDFLAGS)
	mv $(objfolder)/quick.MPI $(exefile).MPI    
	rm -f $(srcfolder)/*.mod $(srcfolder)/*.o


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
	-rm -f _*.f90 .cppdefs *.mod *.o subs/*.o quick
	
neat:
	-rm -f $(TMPFILES)

# - 2. Make tags for source files
TAGS: $(SRC)
	etags $(SRC)
tags: $(SRC)
	ctags $(SRC)

include $(srcfolder)/depend 
