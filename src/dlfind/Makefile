#  !---------------------------------------------------------------------!
#  ! Refactored by Madu Manathunga on 07/25/2020                         !
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
#FOR=$(FC) $(FFLAGS) $(DLFIND_FLAGS)

# modules used by more than one file
OBJ_MOD = $(dlfindobjfolder)/dlf_stat_module.o $(dlfindobjfolder)/dlf_global_module.o $(dlfindobjfolder)/dlf_allocate.o $(dlfindobjfolder)/dlf_util.o \
	$(dlfindobjfolder)/dlf_sort.o 

# Objects that work without numerical libraries (this distinction does not hold any more) 
OBJ_NOLAPACK = $(dlfindobjfolder)/dl_find.o $(dlfindobjfolder)/dlf_convergence.o $(dlfindobjfolder)/dlf_checkpoint.o $(dlfindobjfolder)/dlf_svnversion.o\
	$(dlfindobjfolder)/dlf_formstep.o $(dlfindobjfolder)/dlf_scalestep.o \
	$(dlfindobjfolder)/dlf_lbfgs.o $(dlfindobjfolder)/dlf_time.o \
	$(dlfindobjfolder)/dlf_task.o $(dlfindobjfolder)/dlf_neb.o $(dlfindobjfolder)/dlf_qts.o $(dlfindobjfolder)/dlf_dimer.o $(dlfindobjfolder)/dlf_conint.o $(dlfindobjfolder)/dlf_coords.o \
	$(dlfindobjfolder)/dlf_parallel_opt.o $(dlfindobjfolder)/dlf_microiter.o

# Objects that need LAPACK linked in
OBJ_LAPACK = $(dlfindobjfolder)/dlf_hdlc_matrixlib.o $(dlfindobjfolder)/dlf_hdlc_primitive.o $(dlfindobjfolder)/dlf_hdlc_constraint.o \
	$(dlfindobjfolder)/dlf_hdlc_hdlclib.o $(dlfindobjfolder)/dlf_hdlc_interface.o $(dlfindobjfolder)/dlf_linalg.o

OBJ =  $(OBJ_NOLAPACK) $(OBJ_LAPACK) 

OBJ_MAIN = $(dlfindobjfolder)/dlfind_main_driver.o $(dlfindobjfolder)/dlf_serial.o 
##OBJ_MAIN = $(dlfindobjfolder)/dlf_serial.o 


#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

$(OBJ_MOD):$(dlfindobjfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

$(OBJ):$(dlfindobjfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

$(OBJ_MAIN):$(dlfindobjfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

dlfind:$(OBJ_MOD) $(OBJ) $(OBJ_MAIN)
	$(VB) $(ARCH) $(ARCHFLAGS) $(libfolder)/libdlfind.$(LIBEXT) $(dlfindobjfolder)/*.o

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@-rm -f *.o *.a *.mod 
