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

modobj=$(objfolder)/quick_exception_module.o $(objfolder)/quick_mpi_module.o $(objfolder)/quick_files_module.o \
		$(objfolder)/quick_constants_module.o $(objfolder)/quick_input_parser_module.o $(objfolder)/quick_method_module.o \
		$(objfolder)/quick_molspec_module.o $(objfolder)/quick_gaussian_class_module.o $(objfolder)/quick_size_module.o \
		$(objfolder)/quick_basis_module.o $(objfolder)/quick_calculated_module.o \
		$(objfolder)/quick_divcon_module.o $(objfolder)/quick_ecp_module.o $(objfolder)/quick_electrondensity_module.o \
		$(objfolder)/quick_timer_module.o $(objfolder)/quick_mfcc_module.o $(objfolder)/quick_params_module.o \
		$(objfolder)/quick_pb_module.o $(objfolder)/quick_scratch_module.o $(objfolder)/quick_all_module.o \
		$(objfolder)/quick_overlap_module.o $(objfolder)/quick_gridpoints_module.o $(objfolder)/quick_cutoff_module.o \
		$(objfolder)/quick_eri_module.o $(objfolder)/oshell_quick_eri_module.o $(objfolder)/quick_lri_module.o \
		$(objfolder)/quick_eri_grad_module.o $(objfolder)/oshell_quick_eri_grad_module.o $(objfolder)/quick_lri_grad_module.o \
		$(objfolder)/quick_oei_module.o $(objfolder)/quick_dft_module.o \
		$(objfolder)/quick_scf_operator_module.o $(objfolder)/quick_scf_module.o $(objfolder)/quick_gradient_module.o \
		$(objfolder)/oshell_quick_gradient_module.o \
		$(objfolder)/quick_uscf_operator_module.o $(objfolder)/quick_uscf_module.o $(objfolder)/quick_sad_guess_module.o \
		$(objfolder)/quick_optimizer_module.o $(objfolder)/quick_cew_module.o 

apimodobj=$(objfolder)/quick_api_module.o  $(objfolder)/quick_api_test_module.o

#  !---------------------------------------------------------------------!
#  ! Build targets                                                !
#  !---------------------------------------------------------------------!

$(modobj):$(objfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	$(VB)$(FOR) -c $< -o $@

$(apimodobj):$(objfolder)/%.o:%.f90
	@echo "[QUICK]  FC $@"
	 $(VB)$(FOR) -c $< -o $@

all:$(modobj) $(apimodobj)

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@-rm -f *.mod *.o

