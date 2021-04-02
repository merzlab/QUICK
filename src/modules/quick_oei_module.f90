#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 03/24/2021                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains all one electron integral (oei) & oei gradient ! 
! code.                                                               !
!---------------------------------------------------------------------!

module quick_oei_module

  implicit none
  private

  public :: get1eEnergy

contains

  !------------------------------------------------
  ! get1eEnergy
  !------------------------------------------------
  subroutine get1eEnergy()
     !------------------------------------------------
     ! This subroutine is to get 1e integral
     !------------------------------------------------
     use quick_calculated_module, only: quick_qm_struct
     use quick_method_module,only: quick_method
     use quick_timer_module, only:timer_begin, timer_end, timer_cumer
     use quick_basis_module

     implicit double precision(a-h,o-z)

     call cpu_time(timer_begin%tE)
  
     quick_qm_struct%Eel=0.d0
     call copySym(quick_qm_struct%o,nbasis)
     quick_qm_struct%Eel=quick_qm_struct%Eel+sum2mat(quick_qm_struct%dense,quick_qm_struct%o,nbasis)

     if (quick_method%unrst) then
       call copySym(quick_qm_struct%ob,nbasis)
       quick_qm_struct%Eel = quick_qm_struct%Eel+sum2mat(quick_qm_struct%denseb,quick_qm_struct%ob,nbasis)
     endif

     call cpu_time(timer_end%tE)
     timer_cumer%TE=timer_cumer%TE+timer_end%TE-timer_begin%TE
  
  end subroutine get1eEnergy
  
end module quick_oei_module
