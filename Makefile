
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
	@echo  "Building successful."

nobuildtypes:
	@echo  "Error: No build type found. Plesae run configure script first."

serial: checkfolders
	@echo  "Building serial version.."
	@cp -f $(buildfolder)/make.serial.in $(buildfolder)/make.in
	@cd $(buildfolder) && make serial

mpi: checkfolders
	@echo  "Building mpi version.."
	@cp -f $(buildfolder)/make.mpi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make mpi

cuda: checkfolders
	@echo  "Building cuda version.."
	@cp -f $(buildfolder)/make.cuda.in $(buildfolder)/make.in
	@cd $(buildfolder) && make cuda

cudampi: checkfolders
	@echo  "Building cuda-mpi version.."
	@cp -f $(buildfolder)/make.cudampi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make cudampi 

checkfolders:
	@if [ ! -d $(exefolder) ]; then echo  "Error: $(exefolder) not found. Please configure first."; \
	exit 1; fi
	@if [ ! -d $(buildfolder) ]; then echo  "Error: $(buildfolder) not found. Please configure first."; \
	exit 1; fi 	

#  !---------------------------------------------------------------------!
#  ! Installation targets                                                !
#  !---------------------------------------------------------------------!

.PHONY: noinstall install serialinstall mpiinstall cudainstall cudampiinstall

install: $(INSTALLTYPES)
	@echo  "Installation sucessful."
	@echo  ""
	@echo  "Please add the following into your .bash_profile or .bashrc file."
	@echo  "      export QUICK_BASIS=$(installfolder)/basis"

noinstall:
	@echo  "Please find QUICK executables in $(exefolder)."

serialinstall:
	@if [ -x $(exefolder)/quick ]; then cp -f $(exefolder)/quick $(installfolder)/bin; \
	else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
	exit 1; fi
	@cp -f $(buildfolder)/include/serial/* $(installfolder)/include/serial
	@cp -f $(buildfolder)/lib/serial/* $(installfolder)/lib/serial

mpiinstall:
	@if [ -x $(exefolder)/quick.mpi ]; then cp -f $(exefolder)/quick.mpi $(installfolder)/bin; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(buildfolder)/include/mpi/* $(installfolder)/include/mpi
	@cp -f $(buildfolder)/lib/mpi/* $(installfolder)/lib/mpi

cudainstall:
	@if [ -x $(exefolder)/quick.cuda ]; then cp -f $(exefolder)/quick.cuda $(installfolder)/bin; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda $(installfolder)/bin
	@cp -f $(buildfolder)/include/cuda/* $(installfolder)/include/cuda
	@cp -f $(buildfolder)/lib/cuda/* $(installfolder)/lib/cuda

cudampiinstall:
	@if [ -x $(exefolder)/quick.cuda.mpi ]; then cp -f $(exefolder)/quick.cuda.mpi $(installfolder)/bin; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda.mpi $(installfolder)/bin
	@cp -f $(buildfolder)/include/cudampi/* $(installfolder)/include/cudampi
	@cp -f $(buildfolder)/lib/cudampi/* $(installfolder)/lib/cudampi

#  !---------------------------------------------------------------------!
#  ! Installation targets                                                !
#  !---------------------------------------------------------------------!

.PHONY: test buildtest installtest

test:$(TESTTYPE)

buildtest:
	@cp $(toolsfolder)/runtest $(homefolder)
	$(homefolder)/runtest

installtest:
	@if [ ! -x $(installfolder)/bin/quick* ]; then \
        echo  "Error: Executables not found. You must run 'make install' before running 'make test'."; \
        exit 1; fi	
	@cp $(toolsfolder)/runtest $(installfolder)
	@cd $(installfolder) && ./runtest

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY:serialclean mpiclean cudaclean cudampiclean makeinclean
	
clean:$(CLEANTYPES)
	@-rm -f $(homefolder)/runtest 
	@echo  "Cleaned up successfully."

serialclean:
	@cp -f $(buildfolder)/make.serial.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory clean 

mpiclean:
	@cp -f $(buildfolder)/make.mpi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory clean

cudaclean:
	@cp -f $(buildfolder)/make.cuda.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory clean

cudampiclean:
	@cp -f $(buildfolder)/make.cudampi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory clean

distclean: makeinclean
	@-rm -f $(homefolder)/runtest 
	@-rm -rf $(buildfolder) $(exefolder)
	@echo  "Removed build and bin directories."

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
	@if [ "$(TESTTYPE)" = 'installtest' ]; then rm -rf $(installfolder)/basis; \
	rm -rf $(installfolder)/test; fi
	@-rm -f $(installfolder)/runtest
	@echo  "Uninstallation sucessful."

nouninstall:
	@echo  "Nothing to uninstall."

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

