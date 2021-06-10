#  !---------------------------------------------------------------------!
#  ! Refactored by Madu Manathunga on 07/25/2020                         !
#  !                                                                     !
#  ! Previous contributors: Yipu Miao, John Faver                        !
#  !                                                                     !
#  ! Copyright (C) 2020-2021 Merz lab                                    !
#  ! Copyright (C) 2020-2021 Götz lab                                    !
#  !                                                                     !
#  ! This Source Code Form is subject to the terms of the Mozilla Public !
#  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
#  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
#  !_____________________________________________________________________!
#
#  !---------------------------------------------------------------------!
#  ! This Makefile builds a QUICK version from source code               !
#  !---------------------------------------------------------------------!

MAKEIN = ./make.in
include $(MAKEIN)

#  !---------------------------------------------------------------------!
#  ! Variable declarations                                               !
#  !---------------------------------------------------------------------!

FOR=$(FC) $(FFLAGS)

SUBS = $(objfolder)/Angles.o $(objfolder)/copyDMat.o $(objfolder)/copySym.o \
	$(objfolder)/degen.o $(objfolder)/denspt.o $(objfolder)/oshell_denspt.o \
	$(objfolder)/diag.o $(objfolder)/diagmkl.o \
	$(objfolder)/dipole.o $(objfolder)/EffChar.o $(objfolder)/eigvec.o \
	$(objfolder)/findBlock.o $(objfolder)/fmt.o $(objfolder)/getinum.o \
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

MIRP = $(objfolder)/mirp_fmt.o

#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

$(SUBS):$(objfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

$(MIRP):$(objfolder)/%.o:%.cpp
	@echo "[QUICK]  CXX $@"
	$(VB)$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(SUBS) $(MIRP)

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@-rm -f *.o
