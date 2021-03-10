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

    case(18)
      msg='Atom out of range for core.'

    case(19)
      msg='Systems with unpaired electrons require UHF/UDFT calculations which are currently not supported.'

    case(20)
      msg='Higher multiplicities require UHF/UDFT calculations which are currently not supported. '

    case(21)
      msg='Incorrect number of electrons for a doublet.'

    case(22)
      msg='Incorrect number of electrons for a triplet.'

    case(23)
      msg='Incorrect number of electrons for a quadruplet.'

    case(24)
      msg='No CUDA enabled devices found.'
 
    case(25)
      msg='Illegal device ID.'

    case(26)
      msg='CUDA version required for selected device is less than 1.3.'

    case(27)
      msg='CUDA 1.3 (or higher) enabled device not found.'

    case(28)
      msg='Device microarchitecture must be higher than 1.3.'

    case(29)
      msg='Microarchitecture of selected device is less than 3.0.'

    case(30)
      msg='Selected device is already in use.'

    case(31)
      msg='LIBXC keyword must be followed by a functional name.'

    case(32)
      msg='Requested LIBXC functional does not exist.'

    case default
      msg='Unknown error.'

    end select

    call PrtErr(OUTFILEHANDLE, trim(msg))

  end subroutine

end module quick_exception_module
