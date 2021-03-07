!---------------------------------------------------------------------!
! Created by Madu Manathunga on 02/25/2020                            !
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

    implicit none
    integer, intent(inout) :: ierr

    if (ierr /= 0) then
      call print_exception(ierr)
      call quick_exit(OUTFILEHANDLE,1)
    endif

  end subroutine raise_exception

  ! picks an error message and prints
  subroutine print_exception(ierr)

    implicit none
    integer, intent(in) :: ierr
    character(len=200) :: msg = ''

    write(*,*) ierr

    select case(ierr)

    case(10)
      msg='f basis functions are currently not supported.'

    case(11)
      msg='UHF/UDFT is currently not implemented.'

    case(12)
      msg='Failed to overwrite file.'

    case(13)
      msg='Failed to open input file.'

    case(14)
      msg='Failed to open output file.'

    case(15)
      msg='Failed to open basis file.'

    case(16)
      msg='Failed to open .dat file.'

    case(17)
      msg='Failed to open file.'

    case default
      msg='Unknown error.'

    end select

    call PrtErr(OUTFILEHANDLE, trim(msg))

  end subroutine

end module quick_exception_module
