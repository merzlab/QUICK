#!/bin/sh

MAKEIN = ./make.in
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

FOR=$(FC) $(FFLAGS)

#----------------------
# quick modules and object files
#----------------------

modobj=$(objfolder)/quick_mpi_module.o $(objfolder)/quick_constants_module.o $(objfolder)/quick_method_module.o \
		$(objfolder)/quick_molspec_module.o $(objfolder)/quick_gaussian_class_module.o $(objfolder)/quick_size_module.o \
		$(objfolder)/quick_amber_interface_module.o $(objfolder)/quick_basis_module.o $(objfolder)/quick_calculated_module.o \
		$(objfolder)/quick_divcon_module.o $(objfolder)/quick_ecp_module.o $(objfolder)/quick_electrondensity_module.o \
		$(objfolder)/quick_files_module.o $(objfolder)/quick_timer_module.o $(objfolder)/quick_gridpoints_module.o \
		$(objfolder)/quick_mfcc_module.o $(objfolder)/quick_params_module.o $(objfolder)/quick_pb_module.o \
		$(objfolder)/quick_scratch_module.o $(objfolder)/quick_all_module.o $(objfolder)/quick_scf_module.o \
		$(objfolder)/quick_gradient_module.o $(objfolder)/quick_api_module.o $(objfolder)/quick_api_test_module.o 

#----------------------
# Create object and mod files
# # #----------------------

$(modobj):$(objfolder)/%.o:%.f90
	$(FOR) -c $< -o $@

all:$(modobj)

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
quick_modules: $(modobj) 	

#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
.PHONY: clean neat
clean: neat
	@-rm -f .cppdefs *.mod *.o subs/*.o quick 
	
neat:
	@-rm -f $(TMPFILES) 

# - 2. Make tags for source files
TAGS: $(SRC)
	etags $(SRC)
tags: $(SRC)
	ctags $(SRC)
