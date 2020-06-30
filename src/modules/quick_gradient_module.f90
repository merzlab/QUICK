!---------------------------------------------------------------------!
! Created by Madu Manathunga on 06/29/2020                            !
!                                                                     ! 
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

! This module contains subroutines and data structures related to
! scf gradient calculation

#include "../config.h"

module quick_gradient_module

  implicit none
  private

  public :: allocate_quick_gradient, deallocate_quick_gradient
  public :: tmp_grad, tmp_ptchg_grad
  
  double precision, allocatable, dimension(:) :: tmp_grad
  double precision, allocatable, dimension(:) :: tmp_ptchg_grad

contains

  subroutine allocate_quick_gradient()

    use quick_molspec_module, only : quick_molspec, natom
    implicit none

    if(.not. allocated(tmp_grad)) allocate(tmp_grad(3*natom))
    tmp_grad = 0.0d0

    if(quick_molspec%nextatom.gt.0) then
      if(.not. allocated(tmp_ptchg_grad)) allocate(tmp_ptchg_grad(3*quick_molspec%nextatom))    
      tmp_ptchg_grad = 0.0d0
    endif

  end subroutine allocate_quick_gradient

  subroutine deallocate_quick_gradient()

    use quick_molspec_module, only : quick_molspec
    implicit none

    if(allocated(tmp_grad)) deallocate(tmp_grad) 
    
    if(quick_molspec%nextatom.gt.0) then
      if(allocated(tmp_ptchg_grad)) deallocate(tmp_ptchg_grad)
    endif
  
  end subroutine deallocate_quick_gradient

end module quick_gradient_module

