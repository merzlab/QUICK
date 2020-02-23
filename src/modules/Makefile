#!/bin/sh

#include "../config.h"
MAKEIN = ../../make.in
include $(MAKEIN)

# --- Makefile for Quick Program ---
#  				- v 3.00 2019/03/25 Madu Manathunga
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

#Madu: intel and gnu use -module and -J options to direct module files. 
#Therefore, we use the following if condition to handle this. 

ifeq "$(FC)" "ifort"
  FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -I$(objfolder) -module $(objfolder)
else ifeq "$(FC)" "mpiifort"
  FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -I$(objfolder) -module $(objfolder)
else
  FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -J$(objfolder)
endif

#----------------------
# quick modules and object files
#----------------------
modfolder = ./modules
objfolder = ../../obj

PPRCFILES = quick_basis_module.f90 quick_method_module.f90 quick_molspec_module.f90 quick_calculated_module.f90 quick_timer_module.f90
modobj=$(objfolder)/quick_mpi_module.o $(objfolder)/quick_constants_module.o $(objfolder)/quick_method_module.o \
		$(objfolder)/quick_molspec_module.o $(objfolder)/quick_gaussian_class_module.o $(objfolder)/quick_size_module.o \
		$(objfolder)/quick_amber_interface_module.o $(objfolder)/quick_basis_module.o $(objfolder)/quick_calculated_module.o \
		$(objfolder)/quick_divcon_module.o $(objfolder)/quick_ecp_module.o $(objfolder)/quick_electrondensity_module.o \
		$(objfolder)/quick_files_module.o $(objfolder)/quick_timer_module.o $(objfolder)/quick_gridpoints_module.o \
		$(objfolder)/quick_mfcc_module.o $(objfolder)/quick_params_module.o $(objfolder)/quick_pb_module.o \
		$(objfolder)/quick_scratch_module.o $(objfolder)/quick_all_module.o
#----------------------
# Preprocess .f90 soruce files
# #----------------------

quick_basis_module.f90: quick_basis_module_.f90 
	$(FPP) quick_basis_module_.f90 > quick_basis_module.f90 

quick_method_module.f90: quick_method_module_.f90
	$(FPP) quick_method_module_.f90 > quick_method_module.f90

quick_molspec_module.f90: quick_molspec_module_.f90
	$(FPP) quick_molspec_module_.f90 > quick_molspec_module.f90

quick_calculated_module.f90: quick_calculated_module_.f90
	$(FPP) quick_calculated_module_.f90 > quick_calculated_module.f90

quick_timer_module.f90: quick_timer_module_.f90
	$(FPP) quick_timer_module_.f90 > quick_timer_module.f90

#----------------------
# Create object and mod files
# # #----------------------

$(modobj):$(objfolder)/%.o:%.f90
	$(FOR) -c $< -o $@

all:quick_modules

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
quick_modules: $(PPRCFILES) $(modobj) 	

#quick_mpi_modules: $(mpimodobj)		
#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
.PHONY: clean neat
clean: neat
	-rm -f .cppdefs *.mod *.o subs/*.o quick 
	
neat:
	-rm -f $(TMPFILES) $(PPRCFILES) 

# - 2. Make tags for source files
TAGS: $(SRC)
	etags $(SRC)
tags: $(SRC)
	ctags $(SRC)
