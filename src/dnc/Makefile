#  !---------------------------------------------------------------------!
#  ! Copyright (C) 2020-2021 Merz lab                                    !
#  ! Copyright (C) 2020-2021 Götz lab                                    !
#  !                                                                     !
#  ! This Source Code Form is subject to the terms of the Mozilla Public !
#  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
#  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
#  !_____________________________________________________________________!
#
#  !---------------------------------------------------------------------!
#  ! This Makefile builds QUICK's divide-and-conquer (DnC/MFCC) sources  !
#  !---------------------------------------------------------------------!

MAKEIN = ./make.in
include $(MAKEIN)

#  !---------------------------------------------------------------------!
#  ! Variable declarations                                               !
#  !---------------------------------------------------------------------!

FOR=$(FC) $(FFLAGS)

DNC = $(objfolder)/dnc_scf.o $(objfolder)/dnc_scf_operator.o \
	$(objfolder)/dnc_uscf.o $(objfolder)/dnc_uscf_operator.o \
	$(objfolder)/inidivcon.o $(objfolder)/mfcc_init_guess.o $(objfolder)/mfcc_fragment_scf.o \
	$(objfolder)/mfcc_reorder.o $(objfolder)/mfcc_start.o $(objfolder)/calmp2divcon.o \
	$(objfolder)/dnc_shell.o

#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

$(DNC):$(objfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

all: $(DNC)

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@-rm -f *.o
