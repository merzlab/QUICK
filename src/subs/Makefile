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

#----------------------
# quick modules and object files
#----------------------
libfolder = ../../lib
objfolder = ../../obj

SUBS = Angles.o copyDMat.o copySym.o degen.o denspt.o diag.o dipole.o \
	EffChar.o eigval.o eigvec.o ekinetic.o findBlock.o fmt.o getinum.o \
	getNum.o greedy_distrubute.o hrr.o iatoi.o iatoimp.o io.o iwhole.o \
	lbfgs.o Lsolve.o matComp.o matMul.o order.o orthog.o PriCol.o PriSym.o \
	PrtAct.o PrtDat.o PrtErr.o PrtLab.o PrtMsg.o PrtTim.o PrtWrn.o pteval.o quick_open.o \
	random.o rdinum.o rdnml.o rdnum.o rdword.o readPDB.o spdfgh.o ssw.o \
	sum2Mat.o transpose.o tridi.o upcase.o vett.o whatis.o whole.o \
	wrtRestart.o xnorm.o zeroMatrix.o zmake.o pt2der.o sswder.o \
	denspt_new_imp.o pteval_new_imp.o

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -I$(objfolder) 

$(SUBS):%.o:%.f90
	$(FOR) -c $< -o $@

#================= quick subroutine library =================================

quick_subs:$(SUBS)
	ar -r $(libfolder)/quicklib.a *.o

all: quick_subs	

#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean:
	-rm -f *.o  
