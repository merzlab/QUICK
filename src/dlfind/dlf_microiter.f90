
! **********************************************************************
! **                                                                  **
! **               Microiterative optimisation routines               **
! **                                                                  **
! **                          Tom Keal, 2012                          **
! **                                                                  **
! **********************************************************************
!!****h* DL-FIND/microiter
!!
!! NAME
!! microiter
!!
!! FUNCTION
!! Support routines for microiterative optimisation
!!
!! COMMENTS
!! Implemented under the HECToR dCSE project
!! "Microiterative QM/MM Optimisation for Materials Chemistry"
!! T.W. Keal, P. Sherwood, A. Walsh, A.A. Sokol and C.R.A. Catlow
!!
!! www.hector.ac.uk/cse/distributedcse/reports/chemshell02
!!
!! The microiterative minimisation algorithm is based on the original 
!! implementation for HDLCOpt:
!!
!! Johannes Kaestner, Stephan Thiel, Hans Martin Senn, Paul Sherwood
!! and Walter Thiel, J. Chem. Theory Comput. v3 p1064 (2007)
!!
!! In the DL-FIND implementation microiterative minimisation, 
!! P-RFO, dimer and NEB are supported.
!!
!! DATA
!!  $Date$
!!  $Rev$
!!  $Author$
!!  $URL$
!!  $Id$
!!
!! COPYRIGHT
!!
!!  Copyright 2012 Tom Keal (thomas.keal@stfc.ac.uk) 
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as 
!!  published by the Free Software Foundation, either version 3 of the 
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public 
!!  License along with DL-FIND.  If not, see 
!!  <http://www.gnu.org/licenses/>.
!!
!! SOURCE
module dlf_microiter
  use dlf_parameter_module, only: rk
  type microiter_type
     integer :: varperimage ! total no. of internal coordinates per image
     integer :: coreperimage ! no. of inner region coordinates per image
     integer :: outerperimage ! no. of outer region coordinates per image
     integer :: nmicvar ! number of internal coordinates in microiterative region
     integer :: nmicimage ! number of images involved in microiterative opt
     real(rk), allocatable :: micicoords(:)
     real(rk), allocatable :: micigradient(:)
     real(rk), allocatable :: micstep(:)
     logical :: tmicoldenergy ! do we have an old energy from the current cycle
     real(rk) :: micoldenergy
     real(rk), allocatable :: micoldgradient(:)
     real(rk) :: macroenergy_conv ! save macro energy convergence test info
  end type microiter_type
  type(microiter_type), save :: microiter
end module dlf_microiter
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_init
!!
!! FUNCTION
!!
!! Initialise microiterative optimisation. Identify inner atoms, 
!! allocate arrays, etc.
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_init
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  use dlf_allocate, only: allocate,deallocate
  use dlf_stat, only: stat
  implicit none
  integer :: ivar

  ! TODO should we be using glob lbfgs_mem to init?? what does hdlcopt do?

  ! Get number of internal coords in each region (per image)
  call dlf_direct_get_nivar(0, microiter%varperimage)
  call dlf_direct_get_nivar(1, microiter%coreperimage)
  call dlf_direct_get_nivar(2, microiter%outerperimage)

  if (microiter%varperimage /= microiter%coreperimage + microiter%outerperimage) then
     call dlf_fail("dlf_microiter_init: inconsistent varperimage, coreperimage, outperimage")
  end if

  ! Size of microiterative region depends on optimisation method
  select case (glob%icoord)
  case (100:199)
     ! NEB
     microiter%nmicvar = microiter%outerperimage * glob%nimage
     microiter%nmicimage = glob%nimage
     if (glob%nivar /= microiter%nmicvar + glob%nicore) then
        call dlf_fail("dlf_microiter_init: inconsistent nivar, nmicvar, nicore for NEB")
     end if
  case (200:299)
     ! Dimer 
     ! Only one set of environment coords, which correspond to the 
     ! dimer midpoint (translation step).
     microiter%nmicvar = microiter%outerperimage
     microiter%nmicimage = 1
     if (glob%nivar /= 2 * microiter%nmicvar + glob%nicore) then
        call dlf_fail("dlf_microiter_init: inconsistent nivar, nmicvar, nicore for dimer")
     end if     
  case default
     ! Microiterative minimisation, P-RFO ...
     microiter%nmicvar = microiter%outerperimage
     microiter%nmicimage = 1
     if (glob%nivar /= microiter%nmicvar + glob%nicore) then
        call dlf_fail("dlf_microiter_init: inconsistent nivar, nmicvar, nicore")
     end if
  end select
  
  ! Initialise microiterative LBFGS
  call dlf_lbfgs_select("microiter",.true.)
  call dlf_lbfgs_init(microiter%nmicvar, glob%lbfgs_mem) 
  call dlf_lbfgs_deselect

  ! Allocate microiterative arrays
  call allocate(microiter%micicoords, microiter%nmicvar)
  call allocate(microiter%micigradient, microiter%nmicvar)
  call allocate(microiter%micstep, microiter%nmicvar)
  call allocate(microiter%micoldgradient, microiter%nmicvar)
  
  ! Init statistics
  stat%miccycle = 0
  stat%tmiccycle = 0
  stat%tmicaccepted = 0

end subroutine dlf_microiter_init
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_destroy
!!
!! FUNCTION
!!
!! Initialise microiterative optimisation. Identify inner atoms, 
!! allocate arrays, etc.
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_destroy
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  use dlf_allocate, only: allocate,deallocate
  implicit none
  logical :: exists

  call dlf_lbfgs_exists("microiter", exists)
  if (exists) then
     call dlf_lbfgs_select("microiter",.false.)
     call dlf_lbfgs_destroy
     call dlf_lbfgs_deselect
  end if

  ! Deallocate microiterative arrays
  if (allocated(microiter%micicoords)) call deallocate(microiter%micicoords)
  if (allocated(microiter%micigradient)) call deallocate(microiter%micigradient)
  if (allocated(microiter%micstep)) call deallocate(microiter%micstep)
  if (allocated(microiter%micoldgradient)) call deallocate(microiter%micoldgradient)

end subroutine dlf_microiter_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_check_macrostep
!!
!! FUNCTION
!!
!! For macroiterative steps, only move the inner region
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_check_macrostep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  implicit none
  integer :: ivar, iglobstart, iglobend
  
  do ivar = 1, microiter%nmicimage
     iglobstart =  ((ivar - 1) * microiter%varperimage) + microiter%coreperimage + 1
     iglobend = ivar * microiter%varperimage
     glob%step(iglobstart:iglobend) = 0.0d0
  end do

end subroutine dlf_microiter_check_macrostep
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_reset_macrostep
!!
!! FUNCTION
!!
!! If macroiterative step is rejected, inner region step should be halved,
!! while outer region should be set back to the original coordinates
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_reset_macrostep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  implicit none
  integer :: ivar, iglobstart, iglobend
  
  do ivar = 1, microiter%nmicimage
     ! Inner region: halve step
     iglobstart = ((ivar - 1) * microiter%varperimage) + 1
     iglobend = ((ivar - 1) * microiter%varperimage) + microiter%coreperimage
     glob%step(iglobstart:iglobend) = glob%step(iglobstart:iglobend) * 0.5d0
     glob%icoords(iglobstart:iglobend) = glob%icoords(iglobstart:iglobend) - &
          glob%step(iglobstart:iglobend)     
     ! Outer region: reset step completely
     iglobstart = iglobend + 1
     iglobend = ivar * microiter%varperimage
     glob%icoords(iglobstart:iglobend) = glob%icoords(iglobstart:iglobend) - &
          glob%step(iglobstart:iglobend) 
     glob%step(iglobstart:iglobend) = 0.0d0
  end do

end subroutine dlf_microiter_reset_macrostep
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_enter
!!
!! FUNCTION
!!
!! Enter the microiterative optimisation loop
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_enter
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  use dlf_stat, only: stat
  implicit none

  call dlf_lbfgs_select("microiter",.false.)
  ! Reset microiter memory
  call dlf_lbfgs_restart

  microiter%micstep(:) = 0.0d0

  ! Microiteration mode
  glob%imicroiter = 2

  ! Reset microiterative cycle counter
  stat%miccycle = 0

  ! We do not yet have an old energy from this cycle
  microiter%tmicoldenergy = .false.
  microiter%micoldenergy = 0.0d0
  ! Save macro energy convergence test information
  microiter%macroenergy_conv = glob%oldenergy_conv

end subroutine dlf_microiter_enter
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_itomic
!!
!! FUNCTION
!!
!! Set up microiterative arrays
!!
!! INPUTS
!!
!! glob%icoords
!! glob%igradient
!!
!! OUTPUTS
!! 
!! microiter%micicoords
!! microiter%micigradient
!!
!! SYNOPSIS
subroutine dlf_microiter_itomic
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  implicit none
  integer :: ivar, imicstart, imicend, iglobstart, iglobend

  ! Copy outer region data to microiterative arrays
  do ivar = 1, microiter%nmicimage
     imicstart = ((ivar - 1) * microiter%outerperimage) + 1
     imicend = ivar * microiter%outerperimage
     iglobstart = ((ivar - 1) * microiter%varperimage) + microiter%coreperimage + 1
     iglobend = ivar * microiter%varperimage
     microiter%micicoords(imicstart:imicend) = glob%icoords(iglobstart:iglobend)
     microiter%micigradient(imicstart:imicend) = glob%igradient(iglobstart:iglobend)
  end do
  
  !write(*,*) "MICICOORDS"
  !write(*,*) microiter%micicoords
  !write(*,*) "***"
  !write(*,*) "MICIGRADIENT"
  !write(*,*) microiter%micigradient
  !write(*,*) "***"

end subroutine dlf_microiter_itomic
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_formstep
!!
!! FUNCTION
!!
!! Microiterative formstep
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_formstep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  implicit none

  call dlf_lbfgs_step(microiter%micicoords, microiter%micigradient, &
       microiter%micstep)

end subroutine dlf_microiter_formstep
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_scalestep
!!
!! FUNCTION
!!
!! Scale microiterative step
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_scalestep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  implicit none
  integer :: ivar, icount

  call dlf_scalestep_microiter(microiter%nmicvar, microiter%micstep)

  ! Following dlf_scalestep
  microiter%micoldgradient = microiter%micigradient
  
end subroutine dlf_microiter_scalestep
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_step
!!
!! FUNCTION
!!
!! Carry out microiterative step
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_microiter_step
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  use dlf_stat, only: stat
  implicit none
  integer :: ivar, imicstart, imicend, iglobstart, iglobend

  ! Update global coordinates with microiterative step
  do ivar = 1, microiter%nmicimage
     imicstart = ((ivar - 1) * microiter%outerperimage) + 1
     imicend = ivar * microiter%outerperimage
     iglobstart =  ((ivar - 1) * microiter%varperimage) + microiter%coreperimage + 1
     iglobend = ivar * microiter%varperimage
     glob%icoords(iglobstart:iglobend) = glob%icoords(iglobstart:iglobend) + &
          microiter%micstep(imicstart:imicend)
     ! This is for eventual reset of macrostep if not accepted
     glob%step(iglobstart:iglobend) = glob%step(iglobstart:iglobend) + &
          microiter%micstep(imicstart:imicend)
  end do

  ! Store old energy
  microiter%micoldenergy = glob%energy
  microiter%tmicoldenergy = .true.
  
end subroutine dlf_microiter_step
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_test_acceptance
!!
!! FUNCTION
!!
!! Test acceptance of microiterative step
!!
!! INPUTS
!!
!! microiter%micstep
!!
!! OUTPUTS
!! 
!! microiter%micstep
!!
!! SYNOPSIS
subroutine dlf_microiter_test_acceptance(tswitch)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  implicit none
  logical, intent(out) :: tswitch
  integer :: ivar, imicstart, imicend, iglobstart, iglobend

  call test_acceptance_microiter(microiter%nmicvar, glob%energy, &
       microiter%micigradient, microiter%micoldgradient, &
       microiter%tmicoldenergy, &
       microiter%micoldenergy, microiter%micstep, microiter%micicoords, &
       glob%taccepted, tswitch)

  ! Update glob%icoords if rejected
  if (.not. glob%taccepted) then
     do ivar = 1, microiter%nmicimage
        imicstart = ((ivar - 1) * microiter%outerperimage) + 1
        imicend = ivar * microiter%outerperimage
        iglobstart =  ((ivar - 1) * microiter%varperimage) + microiter%coreperimage + 1
        iglobend = ivar * microiter%varperimage
        glob%icoords(iglobstart:iglobend) = glob%icoords(iglobstart:iglobend) - &
             microiter%micstep(imicstart:imicend)
        ! This is for eventual reset of macrostep if not accepted
        glob%step(iglobstart:iglobend) = glob%step(iglobstart:iglobend) - &
             microiter%micstep(imicstart:imicend)        
     end do
  end if

  ! Restore macro energy if switching
  if (tswitch) then
     glob%toldenergy_conv = .true.
     glob%oldenergy_conv = microiter%macroenergy_conv     
  end if

end subroutine dlf_microiter_test_acceptance
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* microiter/dlf_microiter_convergence
!!
!! FUNCTION
!!
!! Check if microiterations have converged
!!
!! INPUTS
!!
!! none
!!
!! OUTPUTS
!! 
!! tconv
!!
!! SYNOPSIS
subroutine dlf_microiter_convergence(tconv)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout
  use dlf_microiter
  use dlf_stat, only: stat
  implicit none
  logical, intent(out) :: tconv

  call convergence_set_info("of microiterations", microiter%nmicvar, &
       glob%energy, microiter%micigradient, microiter%micstep)
  
  ! No energy/step on first microiter cycle
  if (stat%miccycle == 1) glob%toldenergy_conv = .false.
  
  call convergence_test(stat%miccycle, .true., tconv)

  ! Reset to macro old energy for macro step
  if (tconv) then
     glob%toldenergy_conv = .true.
     glob%oldenergy_conv = microiter%macroenergy_conv
  end if
  
end subroutine dlf_microiter_convergence
!!****
