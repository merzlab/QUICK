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
#libfolder = ../../lib
objfolder = ../../obj

SUBS = $(objfolder)/Angles.o $(objfolder)/copyDMat.o $(objfolder)/copySym.o \
	$(objfolder)/degen.o $(objfolder)/denspt.o $(objfolder)/diag.o $(objfolder)/dipole.o \
	$(objfolder)/EffChar.o $(objfolder)/eigvec.o \
	$(objfolder)/ekinetic.o $(objfolder)/findBlock.o $(objfolder)/fmt.o $(objfolder)/getinum.o \
	$(objfolder)/getNum.o $(objfolder)/greedy_distrubute.o $(objfolder)/hrr.o $(objfolder)/iatoi.o \
	$(objfolder)/iatoimp.o $(objfolder)/io.o $(objfolder)/iwhole.o \
	$(objfolder)/lbfgs.o $(objfolder)/Lsolve.o $(objfolder)/matComp.o $(objfolder)/matMul.o \
	$(objfolder)/order.o $(objfolder)/orthog.o $(objfolder)/PriCol.o $(objfolder)/PriSym.o \
	$(objfolder)/PrtAct.o $(objfolder)/PrtDat.o $(objfolder)/PrtErr.o $(objfolder)/PrtLab.o \
	$(objfolder)/PrtMsg.o $(objfolder)/PrtTim.o $(objfolder)/PrtWrn.o $(objfolder)/pteval.o \
	$(objfolder)/quick_open.o $(objfolder)/random.o $(objfolder)/rdinum.o $(objfolder)/rdnml.o \
	$(objfolder)/rdnum.o $(objfolder)/rdword.o $(objfolder)/readPDB.o $(objfolder)/spdfgh.o \
	$(objfolder)/ssw.o $(objfolder)/sum2Mat.o $(objfolder)/transpose.o $(objfolder)/tridi.o \
	$(objfolder)/upcase.o $(objfolder)/vett.o $(objfolder)/whatis.o $(objfolder)/whole.o \
	$(objfolder)/wrtRestart.o $(objfolder)/xnorm.o $(objfolder)/zeroMatrix.o $(objfolder)/zmake.o \
	$(objfolder)/pt2der.o $(objfolder)/sswder.o $(objfolder)/denspt_new_imp.o \
	$(objfolder)/pteval_new_imp.o $(objfolder)/scaMatMul.o 

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -I$(objfolder)

$(SUBS):$(objfolder)/%.o:%.f90
	$(FOR) -c $< -o $@

#================= quick subroutine library =================================

#quick_subs:$(SUBS)
#	ar -r $(libfolder)/quicksublib.a *.o

all: $(SUBS)

#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean:
	-rm -f *.o  
