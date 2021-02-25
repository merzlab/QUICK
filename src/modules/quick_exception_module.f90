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

!---------------------------------------------------------------------!
! This module performs exception handling in QUICK.                   !
!_____________________________________________________________________!

module quick_exception_module

  implicit none
  private
  public :: RaiseException

  interface RaiseException
    module procedure raise_exception
  end interface

contains

  ! raises exceptions
  subroutine raise_exception(ierr)

    use quick_files_module
    implicit none
    integer, intent(in) :: ierr

    if (ierr /= 0) then
      call print_exception(iOutFile, ierr)
      call quick_exit(iOutFile,1)
    endif

  end subroutine



end module quick_exception_module
