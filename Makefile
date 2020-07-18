
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
#  ! Set build targets                                                   !
#  !---------------------------------------------------------------------!

.PHONY: nobuildtypes serial mpi cuda cudampi all

all:$(BUILDTYPES)

nobuildtypes:
	@echo "Error: No build type found. Plesae run configure script first."

serial:
	@echo "Building serial version.."
	@cd $(buildfolder)/serial && make serial

mpi:
	@echo "Building mpi version.."
	@cd $(buildfolder)/mpi && make mpi

cuda:
	@echo "Building cuda version.."
	@cd $(buildfolder)/cuda && make cuda

cudampi:
	@echo "Building cuda-mpi version.."
	@cd $(buildfolder)/cudampi && make cudampi 

#  !---------------------------------------------------------------------!
#  ! Set cleaning targets                                                !
#  !---------------------------------------------------------------------!

.PHONY:serialclean mpiclean cudaclean cudampiclean
	
clean:$(CLEANTYPES)
	@echo "Successfully cleaned up."

serialclean:
	@cd $(buildfolder)/serial && make clean 

mpiclean:
	@cd $(buildfolder)/mpi && make clean

cudaclean:
	@cd $(buildfolder)/cuda && make clean

cudampiclean:
	@cd $(buildfolder)/cudampi && make clean

distclean:
	@-rm -rf $(buildfolder) $(exefolder)

