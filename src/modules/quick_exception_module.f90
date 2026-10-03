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
#if defined(RESTART_HDF5)
      msg='Failed to open .h5 file.'
#else
      msg='Failed to open .dat file.'
#endif

    case(17)
      msg='Failed to open file.'

    case(18)
      msg='Atom out of range for core.'

    case(19)
      msg='Systems with unpaired electrons require UHF/UDFT calculations.'

    case(20)
      msg='Higher multiplicities require UHF/UDFT calculations.'

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
      msg='Insufficient memory available on selected device.'

    case(31)
      msg='LIBXC keyword must be followed by a functional name.'

    case(32)
      msg='Requested LIBXC functional does not exist.'

    case(33)
      msg='SCF failed to converge. ALLOW_BAD_SCF keyword must be specified to proceed with geometry optimization.'

    case(34)
      msg='Dispersion correction unavailable for DFT functional being used.'

    case(35)
      msg='Requested export file format is not available.'

    case(36)
      msg='Support for F functions is disabled. Please recompile the code with support for F functions.'

    case(37)
      msg='G functions are not supported. Please choose a different basis set.'

    case(38)
      msg='Support for F functions is currently not available in cEW-enabled QM/MM.'      

    case(39)
      msg='Support for F functions is currently not available for GPU accelerated UHF/UDFT gradients.'

    case(40)
      msg='Matrix inversion failed during calculation of ESP charges.'

    case(41)
      msg='No of points out of range while creating vanderwaals surface for ESP charge calculation.'

    case(42)
      msg='Number of launched processes is greater than the available number of GPUs. Please &
              &relaunch with lower number of processes.'

    case(43)
      msg='Divide and conquer Fermi level search failed to converge. The assembled density &
              &matrix does not hold the correct number of electrons, so the energy is not &
              &meaningful. ALLOW_BAD_SCF keyword must be specified to proceed anyway.'

    case(44)
      msg='MFCC fragment setup failed. The fragment geometry produced by the MFCC &
              &fragmentation could not be turned into a valid closed shell molecule.'

    case(45)
      msg='MFCC could not determine or apply the atom reordering it needs. Every hydrogen &
              &must be adjacent to the heavy atom it is bonded to.'

    case(46)
      msg='Divide and conquer cannot run with a near-linearly-dependent basis. The canonical &
              &orthogonalisation that removes the dependency produces delocalised basis &
              &functions, which cannot be partitioned into atom-based subsystems. Use a basis &
              &without the dependency, or run without DIVCON.'

    case(47)
      msg='Divide and conquer does not compute the exchange-correlation term, so it cannot be &
              &combined with DFT. The call to get_xc is absent from the divide and conquer &
              &operator build, which leaves the Fock matrix without any XC contribution and the &
              &energy badly wrong rather than merely approximate. Use HF with DIVCON, or DFT &
              &without it.'

    case(48)
      msg='A file was opened with no name. The name was blank by the time it reached &
              &quick_open, which normally means set_quick_files never ran in that context.'

    case default
      msg='Unknown error.'

    end select

    call PrtErr(OUTFILEHANDLE, trim(msg))

  end subroutine

end module quick_exception_module
