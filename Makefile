
#  !---------------------------------------------------------------------!
#  ! Written by Madu Manathunga on 07/17/2020                            !
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
#  ! This is the main Makefile for compiling QUICK source code           !
#  !---------------------------------------------------------------------!

MAKEIN=./make.in
include $(MAKEIN)

#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

.PHONY: nobuildtypes serial mpi cuda cudampi all

all:$(BUILDTYPES)

nobuildtypes:
	@echo -e "\033[91mError: No build type found. Plesae run configure script first.\033[0m"

serial:
	@echo -e "\033[92mBuilding serial version..\033[0m"
	@cd $(buildfolder)/serial && make serial

mpi:
	@echo -e "\033[92mBuilding mpi version..\033[0m"
	@cd $(buildfolder)/mpi && make mpi

cuda:
	@echo -e "\033[92mBuilding cuda version..\033[0m"
	@cd $(buildfolder)/cuda && make cuda

cudampi:
	@echo -e "\033[92mBuilding cuda-mpi version..\033[0m"
	@cd $(buildfolder)/cudampi && make cudampi 

#  !---------------------------------------------------------------------!
#  ! Installation targets                                                !
#  !---------------------------------------------------------------------!

.PHONY: noinstall install serialinstall mpiinstall cudainstall cudampiinstall

install: $(INSTALLTYPES)
	@echo -e "\033[92mInstallation sucessful.\033[0m"

noinstall:
	@echo -e "\033[91mError: No prefix to install. You must specify a prefix during the configuration.\033[0m"
	@echo -e "\033[91m       Please find QUICK executables in $(exefolder).\033[0m"

serialinstall:
	@cp -f $(exefolder)/quick $(installfolder)/bin
	@cp -f $(buildfolder)/serial/include/* $(installfolder)/include/serial
	@cp -f $(buildfolder)/serial/lib/* $(installfolder)/lib/serial

mpiinstall:
	@cp -f $(exefolder)/quick.mpi $(installfolder)/bin
	@cp -f $(buildfolder)/mpi/include/* $(installfolder)/include/mpi
	@cp -f $(buildfolder)/mpi/lib/* $(installfolder)/lib/mpi

cudainstall:
	@cp -f $(exefolder)/quick.cuda $(installfolder)/bin
	@cp -f $(buildfolder)/cuda/include/* $(installfolder)/include/cuda
	@cp -f $(buildfolder)/cuda/lib/* $(installfolder)/lib/cuda

cudampiinstall:
	@cp -f $(exefolder)/quick.cuda.mpi $(installfolder)/bin
	@cp -f $(buildfolder)/cudampi/include/* $(installfolder)/include/cudampi
	@cp -f $(buildfolder)/cudampi/lib/* $(installfolder)/lib/cudampi

#  !---------------------------------------------------------------------!
#  ! Installation targets                                                !
#  !---------------------------------------------------------------------!

.PHONY: test buildtest installtest

test:$(TESTTYPE)

buildtest:
	@cp $(toolsfolder)/runtest $(homefolder)
	@sh $(homefolder)/runtest

installtest:
	@cp $(toolsfolder)/runtest $(installfolder)
	@cd $(installfolder)
	@sh $(installfolder)/runtest

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY:serialclean mpiclean cudaclean cudampiclean
	
clean:$(CLEANTYPES)
	@-rm -f $(homefolder)/runtest 
	@echo -e "\033[92mCleaned up successfully.\033[0m"

serialclean:
	@cd $(buildfolder)/serial && make clean 

mpiclean:
	@cd $(buildfolder)/mpi && make clean

cudaclean:
	@cd $(buildfolder)/cuda && make clean

cudampiclean:
	@cd $(buildfolder)/cudampi && make clean

distclean:
	@-rm -f $(homefolder)/runtest 
	@-rm -rf $(buildfolder) $(exefolder)

#  !---------------------------------------------------------------------!
#  ! Uninstall targets                                                   !
#  !---------------------------------------------------------------------!

.PHONY: nouninstall uninstall serialuninstall mpiuninstall cudauninstall cudampiuninstall 

uninstall: $(UNINSTALLTYPES)
	@-rm -f $(installfolder)/runtest
	@echo -e "\033[92mUninstallation sucessful.\033[0m"

nouninstall:
	@echo -e "\033[92mNothing to uninstall.\033[0m"

serialuninstall:
	@-rm -f $(installfolder)/bin/quick
	@-rm -f $(installfolder)/include/serial/*
	@-rm -f $(installfolder)/lib/serial/*

mpiuninstall:
	@-rm -f $(installfolder)/bin/quick.mpi
	@-rm -f $(installfolder)/include/mpi/*
	@-rm -f $(installfolder)/lib/mpi/*

cudauninstall:
	@-rm -f $(installfolder)/bin/quick.cuda
	@-rm -f $(installfolder)/include/cuda/*
	@-rm -f $(installfolder)/lib/cuda/*

cudampiuninstall:
	@-rm -f $(installfolder)/bin/quick.cuda.mpi
	@-rm -f $(installfolder)/include/cudampi/*
	@-rm -f $(installfolder)/lib/cudampi/*

