
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
	@echo -e "\033[92mBuilding successful.\033[0m"

nobuildtypes:
	@echo -e "\033[91mError: No build type found. Plesae run configure script first.\033[0m"

serial: checkfolders
	@echo -e "\033[92mBuilding serial version..\033[0m"
	@cd $(buildfolder)/serial && make serial

mpi: checkfolders
	@echo -e "\033[92mBuilding mpi version..\033[0m"
	@cd $(buildfolder)/mpi && make mpi

cuda: checkfolders
	@echo -e "\033[92mBuilding cuda version..\033[0m"
	@cd $(buildfolder)/cuda && make cuda

cudampi: checkfolders
	@echo -e "\033[92mBuilding cuda-mpi version..\033[0m"
	@cd $(buildfolder)/cudampi && make cudampi 

checkfolders:
	@if [ ! -d $(exefolder) ]; then echo -e "\033[91mError: $(exefolder) not found. Please configure first.\033[0m"; \
	exit 1; fi
	@if [ ! -d $(buildfolder) ]; then echo -e "\033[91mError: $(buildfolder) not found. Please configure first.\033[0m"; \
	exit 1; fi 	

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
	@if [ -x $(exefolder)/quick ]; then cp -f $(exefolder)/quick $(installfolder)/bin; \
	else echo -e "\033[91mError: Executable not found. You must run 'make' before running 'make install'.\033[m"; \
	exit 1; fi
	@cp -f $(buildfolder)/serial/include/* $(installfolder)/include/serial
	@cp -f $(buildfolder)/serial/lib/* $(installfolder)/lib/serial

mpiinstall:
	@if [ -x $(exefolder)/quick.mpi ]; then cp -f $(exefolder)/quick.mpi $(installfolder)/bin; \
        else echo -e "\033[91mError: Executable not found. You must run 'make' before running 'make install'.\033[m"; \
        exit 1; fi
	@cp -f $(buildfolder)/mpi/include/* $(installfolder)/include/mpi
	@cp -f $(buildfolder)/mpi/lib/* $(installfolder)/lib/mpi

cudainstall:
	@if [ -x $(exefolder)/quick.cuda ]; then cp -f $(exefolder)/quick.cuda $(installfolder)/bin; \
        else echo -e "\033[91mError: Executable not found. You must run 'make' before running 'make install'.\033[m"; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda $(installfolder)/bin
	@cp -f $(buildfolder)/cuda/include/* $(installfolder)/include/cuda
	@cp -f $(buildfolder)/cuda/lib/* $(installfolder)/lib/cuda

cudampiinstall:
	@if [ -x $(exefolder)/quick.cuda.mpi ]; then cp -f $(exefolder)/quick.cuda.mpi $(installfolder)/bin; \
        else echo -e "\033[91mError: Executable not found. You must run 'make' before running 'make install'.\033[m"; \
        exit 1; fi
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
	@if [ ! -x $(installfolder)/bin/quick* ]; then \
        echo -e "\033[91mError: Executables not found. You must run 'make install' before running 'make test'.\033[m"; \
        exit 1; fi	
	@cp $(toolsfolder)/runtest $(installfolder)
	@cd $(installfolder) && sh runtest

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY:serialclean mpiclean cudaclean cudampiclean makeinclean
	
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

distclean: makeinclean
	@-rm -f $(homefolder)/runtest 
	@-rm -rf $(buildfolder) $(exefolder)
	@echo -e "\033[92mRemoved build and bin directories.\033[0m"

makeinclean:
	@-rm -f $(libxcfolder)/make.in
	@-rm -f $(libxcfolder)/maple2c_device/make.in 
	@-rm -f $(subfolder)/make.in
	@-rm -f $(modfolder)/make.in  
	@-rm -f $(octfolder)/make.in 
	@-rm -f $(blasfolder)/make.in 
	@-rm -f $(cudafolder)/make.in 

#  !---------------------------------------------------------------------!
#  ! Uninstall targets                                                   !
#  !---------------------------------------------------------------------!

.PHONY: nouninstall uninstall serialuninstall mpiuninstall cudauninstall cudampiuninstall 

uninstall: $(UNINSTALLTYPES)
	@-rm -rf $(installfolder)/basis
	@-rm -rf $(installfolder)/test
	@-rm -f $(installfolder)/runtest
	@echo -e "\033[92mUninstallation sucessful.\033[0m"

nouninstall:
	@echo -e "\033[92mNothing to uninstall.\033[0m"

serialuninstall:
	@-rm -f $(installfolder)/bin/quick
	@-rm -rf $(installfolder)/include/serial
	@-rm -rf $(installfolder)/lib/serial

mpiuninstall:
	@-rm -f $(installfolder)/bin/quick.mpi
	@-rm -rf $(installfolder)/include/mpi
	@-rm -rf $(installfolder)/lib/mpi

cudauninstall:
	@-rm -f $(installfolder)/bin/quick.cuda
	@-rm -rf $(installfolder)/include/cuda
	@-rm -rf $(installfolder)/lib/cuda

cudampiuninstall:
	@-rm -f $(installfolder)/bin/quick.cuda.mpi
	@-rm -rf $(installfolder)/include/cudampi
	@-rm -rf $(installfolder)/lib/cudampi

