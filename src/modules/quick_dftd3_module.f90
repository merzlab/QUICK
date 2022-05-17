
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
    module procedure calculate_ene
    module procedure calculate_ene_grad
  end interface calculateDFTD3
contains

  subroutine calculate_ene

    

  end subroutine calculate_ene


  subroutine calculate_ene_grad


  end subroutine calculate_ene_grad

end module quick_dftd3_module

