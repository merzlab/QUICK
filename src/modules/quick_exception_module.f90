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

#include "util.fh"

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
      call print_exception(ioutfile, ierr)
      call quick_exit(ioutfile,1)
    endif

  end subroutine raise_exception

  ! picks an error message and prints
  subroutine print_exception(ioutfile, ierr)

    implicit none
    integer, intent(in) :: ioutfile, ierr
    character(len=200) :: msg = ''

    select case(ierr)

    case(10)
      msg='f basis functions are currently not supported.'

    end select

    call PrtErr(ioutfile, trim(msg))

  end subroutine

end module quick_exception_module
