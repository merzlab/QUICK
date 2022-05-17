#  !---------------------------------------------------------------------!
#  ! Created by Madu Manathunga on 05/17/2022                            !
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

DFTD3 = $(dftd3folder)/common.o $(dftd3folder)/sizes.o $(dftd3folder)/pars.o $(dftd3folder)/core.o \
	$(dftd3folder)/api.o

#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

$(DFTD3):$(dftd3folder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

all:$(DFTD3)
        $(VB)$(ARCH) $(ARCHFLAGS) $(libfolder)/libdftd3.$(LIBEXT) $(dftd3folder)/*.o

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@-rm -f *.o
