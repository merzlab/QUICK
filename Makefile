
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

.PHONY: nobuildtypes serial mpi cuda cudampi hip hipmpi all

all:$(BUILDTYPES)
	@echo  "Building successful."

nobuildtypes:
	@echo  "Error: No build type found. Plesae run configure script first."

serial: checkfolders
	@echo  "Building serial version.."
	@cp -f $(buildfolder)/make.serial.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory serial
	@mv -f $(exefolder)/test-api $(homefolder)/test/

mpi: checkfolders
	@echo  "Building mpi version.."
	@cp -f $(buildfolder)/make.mpi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory mpi
	@mv -f $(exefolder)/test-api.MPI $(homefolder)/test/

cuda: checkfolders
	@echo  "Building cuda version.."
	@cp -f $(buildfolder)/make.cuda.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory cuda
	@mv -f $(exefolder)/test-api.cuda $(homefolder)/test/

cudampi: checkfolders
	@echo  "Building cuda-mpi version.."
	@cp -f $(buildfolder)/make.cudampi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory cudampi
	@mv -f $(exefolder)/test-api.cuda.MPI $(homefolder)/test/

hip: checkfolders
	@echo  "Building hip version.."
	@cp -f $(buildfolder)/make.hip.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory hip
	@mv -f $(exefolder)/test-api.hip $(homefolder)/test/

hipmpi: checkfolders
	@echo  "Building hip-mpi version.."
	@cp -f $(buildfolder)/make.hipmpi.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory hipmpi
	@mv -f $(exefolder)/test-api.hip.MPI $(homefolder)/test/

checkfolders:
	@if [ ! -d $(exefolder) ]; then echo  "Error: $(exefolder) not found. Please configure first."; \
	exit 1; fi
	@if [ ! -d $(buildfolder) ]; then echo  "Error: $(buildfolder) not found. Please configure first."; \
	exit 1; fi

#  !---------------------------------------------------------------------!
#  ! Installation targets                                                !
#  !---------------------------------------------------------------------!

.PHONY: noinstall install serialinstall mpiinstall cudainstall cudampiinstall hipinstall hipmpiinstall aminstall

install: $(INSTALLTYPES)
	@echo  "Installation sucessful."
	@echo  ""
	@echo  "Please run the following command to set environment variables."
	@echo  "      source $(installfolder)/quick.rc"

noinstall: all
	@echo  "Please find QUICK executables in $(exefolder)."

serialinstall: serial
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -x $(exefolder)/quick ]; then cp -f $(exefolder)/quick $(installfolder)/bin; \
	cp -f $(homefolder)/test/test-api $(installfolder)/test; \
	else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
	exit 1; fi
	@cp -f $(buildfolder)/include/serial/* $(installfolder)/include/serial
	@cp -f $(buildfolder)/lib/serial/* $(installfolder)/lib/serial
	@cp -f $(toolsfolder)/sadguess $(installfolder)/bin

mpiinstall: mpi
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -x $(exefolder)/quick.MPI ]; then cp -f $(exefolder)/quick.MPI $(installfolder)/bin; \
	cp -f $(homefolder)/test/test-api.MPI $(installfolder)/test; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(buildfolder)/include/mpi/* $(installfolder)/include/mpi
	@cp -f $(buildfolder)/lib/mpi/* $(installfolder)/lib/mpi

cudainstall: cuda
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -x $(exefolder)/quick.cuda ]; then cp -f $(exefolder)/quick.cuda $(installfolder)/bin; \
	cp -f $(homefolder)/test/test-api.cuda $(installfolder)/test; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda $(installfolder)/bin
	@cp -f $(buildfolder)/include/cuda/* $(installfolder)/include/cuda
	@cp -f $(buildfolder)/lib/cuda/* $(installfolder)/lib/cuda

cudampiinstall: cudampi
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -x $(exefolder)/quick.cuda.MPI ]; then cp -f $(exefolder)/quick.cuda.MPI $(installfolder)/bin; \
	cp -f $(homefolder)/test/test-api.cuda.MPI $(installfolder)/test; \
        else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
        exit 1; fi
	@cp -f $(exefolder)/quick.cuda.MPI $(installfolder)/bin
	@cp -f $(buildfolder)/include/cudampi/* $(installfolder)/include/cudampi
	@cp -f $(buildfolder)/lib/cudampi/* $(installfolder)/lib/cudampi

hipinstall: hip
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -x $(exefolder)/quick.hip ]; then cp -f $(exefolder)/quick.hip $(installfolder)/bin; \
	cp -f $(homefolder)/test/test-api.hip $(installfolder)/test; \
	else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
	exit 1; fi
	@cp -f $(exefolder)/quick.hip $(installfolder)/bin
	@cp -f $(buildfolder)/include/hip/* $(installfolder)/include/hip
	@cp -f $(buildfolder)/lib/hip/* $(installfolder)/lib/hip

hipmpiinstall: hipmpi
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -x $(exefolder)/quick.hip.MPI ]; then cp -f $(exefolder)/quick.hip.MPI $(installfolder)/bin; \
	cp -f $(homefolder)/test/test-api.hip.MPI $(installfolder)/test; \
	else echo  "Error: Executable not found. You must run 'make' before running 'make install'."; \
	exit 1; fi
	@cp -f $(exefolder)/quick.hip.MPI $(installfolder)/bin
	@cp -f $(buildfolder)/include/hipmpi/* $(installfolder)/include/hipmpi
	@cp -f $(buildfolder)/lib/hipmpi/* $(installfolder)/lib/hipmpi 

aminstall: all
	@if [ ! -d $(installfolder)/lib64 ]; then mkdir $(installfolder)/lib64; fi;
	@if [ -e $(homefolder)/src/lapack/libblas.so ]; then \
	mv $(homefolder)/src/lapack/libblas.so $(installfolder)/lib64/libblas.so.3.12; \
	ln -s -T $(installfolder)/lib64/libblas.so.3.12 $(installfolder)/lib64/libblas.so.3; \
	ln -s -T $(installfolder)/lib64/libblas.so.3 $(installfolder)/lib64/libblas.so; fi; \
	if [ -e $(homefolder)/src/lapack/liblapack.so ]; then \
	mv $(homefolder)/src/lapack/liblapack.so $(installfolder)/lib64/liblapack.so.3.12; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3.12 $(installfolder)/lib64/liblapack.so.3; \
	ln -s -T $(installfolder)/lib64/liblapack.so.3 $(installfolder)/lib64/liblapack.so; fi;
	@if [ -d $(installfolder)/lib ]; then \
	if [ -e $(buildfolder)/lib/serial/libquick.$(libsuffix) ]; then mv $(buildfolder)/lib/serial/libquick.$(libsuffix) $(installfolder)/lib/libquick.$(libsuffix); \
	mv $(buildfolder)/lib/serial/libxc.$(libsuffix) $(installfolder)/lib/libxc.$(libsuffix); fi; \
	if [ -e $(buildfolder)/lib/mpi/libquick-mpi.$(libsuffix) ]; then mv $(buildfolder)/lib/mpi/libquick-mpi.$(libsuffix) $(installfolder)/lib/libquick-mpi.$(libsuffix); \
	mv $(buildfolder)/lib/mpi/libxc.$(libsuffix) $(installfolder)/lib/libxc.$(libsuffix); fi; \
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

.PHONY: test buildtest installtest fulltest

test:$(TESTTYPE)

buildtest:
	@cp $(toolsfolder)/runtest $(homefolder)
	@$(homefolder)/runtest

installtest:
	@if [ ! -x $(installfolder)/bin/quick ] && [ ! -x $(installfolder)/bin/quick.MPI ] && [ ! -x $(installfolder)/bin/quick.cuda ] && [ ! -x $(installfolder)/bin/quick.cuda.MPI && \
	[ ! -x $(installfolder)/bin/quick.hip ] && [ ! -x $(installfolder)/bin/quick.hip.MPI ]; then \
        echo "Error: Executables not found. You must run 'make install' before running 'make test'."; \
        exit 1; fi
	@cp $(toolsfolder)/runtest $(installfolder)
	@cd $(installfolder) && ./runtest

fulltest:
	@if [ ! -x $(installfolder)/bin/quick ] && [ ! -x $(installfolder)/bin/quick.MPI ] && [ ! -x $(installfolder)/bin/quick.cuda ] && [ ! -x $(installfolder)/bin/quick.cuda.MPI && \
	[ ! -x $(installfolder)/bin/quick.hip ] && [ ! -x $(installfolder)/bin/quick.hip.MPI ]; then \
        echo "Error: Executables not found."; \
        exit 1; fi
	@cp $(toolsfolder)/runtest $(installfolder)
	@cd $(installfolder) && ./runtest --full

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY:serialclean mpiclean cudaclean cudampiclean hipclean hipmpiclean makeinclean

clean:$(CLEANTYPES)
	@-rm -f $(homefolder)/runtest
	@-rm -f $(homefolder)/build/include/common/*
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

hipclean:
	@cp -f $(buildfolder)/make.hip.in $(buildfolder)/make.in
	@cd $(buildfolder) && make --no-print-directory clean

hipmpiclean:
	@cp -f $(buildfolder)/make.hipmpi.in $(buildfolder)/make.in
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
	@-rm -f $(cudafolder)/make.in
	@-rm -f $(hipfolder)/make.in

#  !---------------------------------------------------------------------!
#  ! Uninstall targets                                                   !
#  !---------------------------------------------------------------------!

.PHONY: nouninstall uninstall serialuninstall mpiuninstall cudauninstall cudampiuninstall hipuninstall hipmpiuninstall amuninstall

uninstall: $(UNINSTALLTYPES)
	@if [ "$(TESTTYPE)" = 'installtest' ]; then rm -rf $(installfolder)/basis; \
	rm -rf $(installfolder)/test; fi
	@-rm -f $(installfolder)/runtest
	@-rm -f $(installfolder)/quick.rc
	@-rm -rf $(installfolder)/include/common
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

hipuninstall:
	@-rm -f $(installfolder)/bin/quick.hip
	@-rm -rf $(installfolder)/include/hip
	@-rm -rf $(installfolder)/lib/hip

hipmpiuninstall:
	@-rm -f $(installfolder)/bin/quick.hip.MPI
	@-rm -rf $(installfolder)/include/hipmpi
	@-rm -rf $(installfolder)/lib/hipmpi

amuninstall:
	@-rm -f $(installfolder)/bin/quick
	@-rm -f $(installfolder)/bin/quick.MPI
	@-rm -f $(installfolder)/bin/quick.cuda
	@-rm -f $(installfolder)/bin/quick.cuda.MPI
	@-rm -f $(installfolder)/lib/libquick*
	@-rm -f $(installfolder)/lib/libxc.*
	@-rm -f $(installfolder)/lib/libxc-cuda.*
	@echo  "Successfully removed QUICK executables and libraries from $(installfolder) folder."
