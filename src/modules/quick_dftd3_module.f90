
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/16/2020                            !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

! Interface for dftd3 libarary
module quick_dftd3_module

  implicit none
  private

  public :: calculateDFTD3

  interface calculateDFTD3
    module procedure calculate_dispersion_energy
  end interface calculateDFTD3

contains

  subroutine calculate_dispersion_energy

    use dftd3_api
    use quick_molspec_module
    use quick_method_module, only: quick_method
    use quick_calculated_module, only: quick_qm_struct
    implicit none

    type(dftd3_input) :: input
    type(dftd3_calc) :: dftd3    
    integer :: version
    character(len=8) :: functional

    ! set version
    if(quick_method%DFTD2) then
      version=2
    else if(quick_method%DFTD3) then
      version=3
    else if(quick_method%DFTD3BJ) then
      version=4
    else if(quick_method%DFTD3M) then
      version=5
    else if(quick_method%DFTD3MBJ) then
      version=6
    end if

    ! set dft functional
    if(quick_method%uselibxc) then
      if(quick_method%nof_functionals == 1) then
        if(quick_method%functional_id(1) == 402) functional='b3-lyp'
        if(quick_method%functional_id(1) == 404) functional='o3-lyp'
        if(quick_method%functional_id(1) == 406) functional='pbe0'
        !if(quick_method%functional_id(1) == 407) functional='b97'
      elseif(quick_method%nof_functionals==2) then
        if(quick_method%functional_id(1) == 106 .and. quick_method%functional_id(2) == 131) functional='b-lyp'
        if(quick_method%functional_id(1) == 110 .and. quick_method%functional_id(2) == 131) functional='o-lyp'
        if(quick_method%functional_id(1) == 102 .and. quick_method%functional_id(2) == 130) functional='revpbe'
        if(quick_method%functional_id(1) == 101 .and. quick_method%functional_id(2) == 130) functional='pbe'
        !if(quick_method%functional_id(1) == 106 .and. quick_method%functional_id(2) == 132) functional='b-p'
        !if(quick_method%functional_id(1) == 109 .and. quick_method%functional_id(2) == 134) functional='pw91'
      endif
    else if(quick_method%b3lyp) then    
      functional='b3-lyp'
    else if(quick_method%blyp) then
      functional='b-lyp'
    else if(quick_method%hf) then
      functional='hf'
    endif

    ! initialize dftd3
    call dftd3_init(dftd3, input)
 
    ! set functional
    call dftd3_set_functional(dftd3, functional, version, .false.)

    ! compute dispersion energy and gradient
    call dftd3_dispersion(dftd3, xyz, quick_molspec%iattype, quick_qm_struct%Edisp, & 
    quick_qm_struct%disp_gradient)

  write(*, "(A)") "*** Dispersion for non-periodic case"
  write(*, "(A,ES20.12)") "Energy [au]:", quick_qm_struct%Edisp
  write(*, "(A)") "Gradients [au]:"
  write(*, "(3ES20.12)") quick_qm_struct%disp_gradient
  write(*, *)

  end subroutine calculate_dispersion_energy

end module quick_dftd3_module

