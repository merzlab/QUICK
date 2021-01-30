
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

ifeq "$(SHARED)" 'yes'
  libsuffix = "so"
else
  libsuffix = "a"
endif

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
	@cd $(buildfolder) && make --no-print-directory serial

mpi: checkfolders
	@echo  "Building mpi version.."
	@cp -f $(buildfolder)/make.mpi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory mpi

cuda: checkfolders
	@echo  "Building cuda version.."
	@cp -f $(buildfolder)/make.cuda.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory cuda

cudampi: checkfolders
	@echo  "Building cuda-mpi version.."
	@cp -f $(buildfolder)/make.cudampi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory cudampi

checkfolders:
	@if [ ! -d $(exefolder) ]; then echo  "Error: $(exefolder) not found. Please configure first."; \
	exit 1; fi
	@if [ ! -d $(buildfolder) ]; then echo  "Error: $(buildfolder) not found. Please configure first."; \
	exit 1; fi

#  !---------------------------------------------------------------------!
#  ! Installation targets                                                !
#  !---------------------------------------------------------------------!

.PHONY: noinstall install serialinstall mpiinstall cudainstall cudampiinstall aminstall

install: $(INSTALLTYPES)
	@echo  "Installation sucessful."
	@echo  ""
	@echo  "Please run the following command to set environment variables."
	@echo  "      source $(installfolder)/quick.rc"

noinstall: all
	@echo  "Please find QUICK executables in $(exefolder)."

serialinstall: serial
	@if [ -x $(exefolder)/quick ]; then cp -f $(exefolder)/quick $(installfolder)/bin; \
	else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
	exit 1; fi
	@cp -f $(buildfolder)/include/serial/* $(installfolder)/include/serial
	@cp -f $(buildfolder)/lib/serial/* $(installfolder)/lib/serial

mpiinstall: mpi
	@if [ -x $(exefolder)/quick.MPI ]; then cp -f $(exefolder)/quick.MPI $(installfolder)/bin; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(buildfolder)/include/mpi/* $(installfolder)/include/mpi
	@cp -f $(buildfolder)/lib/mpi/* $(installfolder)/lib/mpi

cudainstall: cuda
	@if [ -x $(exefolder)/quick.cuda ]; then cp -f $(exefolder)/quick.cuda $(installfolder)/bin; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda $(installfolder)/bin
	@cp -f $(buildfolder)/include/cuda/* $(installfolder)/include/cuda
	@cp -f $(buildfolder)/lib/cuda/* $(installfolder)/lib/cuda

cudampiinstall: cudampi
	@if [ -x $(exefolder)/quick.cuda.MPI ]; then cp -f $(exefolder)/quick.cuda.MPI $(installfolder)/bin; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda.MPI $(installfolder)/bin
	@cp -f $(buildfolder)/include/cudampi/* $(installfolder)/include/cudampi
	@cp -f $(buildfolder)/lib/cudampi/* $(installfolder)/lib/cudampi

aminstall: all
	@if [ -d $(installfolder)/lib ]; then \
	if [ -e $(buildfolder)/lib/serial/libquick.$(libsuffix) ]; then mv $(buildfolder)/lib/serial/libquick.$(libsuffix) $(installfolder)/lib/libquick.$(libsuffix); \
	mv $(buildfolder)/lib/serial/libxc.$(libsuffix) $(installfolder)/lib/libxc.$(libsuffix); fi; \
	if [ -e $(buildfolder)/lib/serial/libblas-quick.$(libsuffix) ]; then mv $(buildfolder)/lib/serial/libblas-quick.$(libsuffix) $(installfolder)/lib/libblas-quick.$(libsuffix); fi; \
	if [ -e $(buildfolder)/lib/mpi/libquick-mpi.$(libsuffix) ]; then mv $(buildfolder)/lib/mpi/libquick-mpi.$(libsuffix) $(installfolder)/lib/libquick-mpi.$(libsuffix); \
	mv $(buildfolder)/lib/mpi/libxc.$(libsuffix) $(installfolder)/lib/libxc.$(libsuffix); fi; \
	if [ -e $(buildfolder)/lib/mpi/libblas-quick.$(libsuffix) ]; then mv $(buildfolder)/lib/mpi/libblas-quick.$(libsuffix) $(installfolder)/lib/libblas-quick.$(libsuffix); fi; \
	if [ -e $(buildfolder)/lib/cuda/libquick-cuda.$(libsuffix) ]; then mv $(buildfolder)/lib/cuda/libquick-cuda.$(libsuffix) $(installfolder)/lib/libquick-cuda.$(libsuffix); \
	mv $(buildfolder)/lib/cuda/libxc-cuda.$(libsuffix) $(installfolder)/lib/libxc-cuda.$(libsuffix); fi; \
	if [ -e $(buildfolder)/lib/cudampi/libquick-cudampi.$(libsuffix) ]; then mv $(buildfolder)/lib/cudampi/libquick-cudampi.$(libsuffix) $(installfolder)/lib/libquick-cudampi.$(libsuffix); \
	mv $(buildfolder)/lib/cudampi/libxc-cuda.$(libsuffix) $(installfolder)/lib/libxc-cuda.$(libsuffix); fi; echo "Successfully installed QUICK libraries in $(installfolder)/lib folder.";\
        else echo "Error: $(installfolder)/lib folder not found."; exit 1; fi
	@if [ -d $(installfolder)/bin ]; then \
	for i in quick quick.MPI quick.cuda quick.cuda.MPI; do if [ -x $(exefolder)/$$i ]; then mv $(exefolder)/$$i $(installfolder)/bin/; fi; done; \
	echo "Successfully installed QUICK executables in $(installfolder)/bin folder."; \
	else echo  "Error: $(installfolder)/bin folder not found."; exit 1; fi


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
	@-rm -f $(homefolder)/quick.rc
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

.PHONY: nouninstall uninstall serialuninstall mpiuninstall cudauninstall cudampiuninstall amuninstall

uninstall: $(UNINSTALLTYPES)
	@if [ "$(TESTTYPE)" = 'installtest' ]; then rm -rf $(installfolder)/basis; \
	rm -rf $(installfolder)/test; fi
	@-rm -f $(installfolder)/runtest
	@-rm -f $(installfolder)/quick.rc
	@echo  "Uninstallation sucessful."

nouninstall:
	@echo  "Nothing to uninstall."

serialuninstall:
	@-rm -f $(installfolder)/bin/quick
	@-rm -rf $(installfolder)/include/serial
	@-rm -rf $(installfolder)/lib/serial

mpiuninstall:
	@-rm -f $(installfolder)/bin/quick.MPI
	@-rm -rf $(installfolder)/include/mpi
	@-rm -rf $(installfolder)/lib/mpi

cudauninstall:
	@-rm -f $(installfolder)/bin/quick.cuda
	@-rm -rf $(installfolder)/include/cuda
	@-rm -rf $(installfolder)/lib/cuda

cudampiuninstall:
	@-rm -f $(installfolder)/bin/quick.cuda.MPI
	@-rm -rf $(installfolder)/include/cudampi
	@-rm -rf $(installfolder)/lib/cudampi

amuninstall:
	@-rm -f $(installfolder)/bin/quick
	@-rm -f $(installfolder)/bin/quick.MPI
	@-rm -f $(installfolder)/bin/quick.cuda
	@-rm -f $(installfolder)/bin/quick.cuda.MPI
	@-rm -f $(installfolder)/lib/libquick*
	@-rm -f $(installfolder)/lib/libxc.*
	@-rm -f $(installfolder)/lib/libxc-cuda.*
	@echo  "Successfully removed QUICK executables and libraries from $(installfolder) folder."
