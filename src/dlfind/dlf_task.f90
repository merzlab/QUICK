! **********************************************************************
! **                                                                  **
! **                Task assignment and management                    **
! **                                                                  **
! **********************************************************************
!!****h* main/task
!!
!! NAME
!! task
!!
!! FUNCTION
!! Manage optimisation tasks, call dlf_run (which does one 
!! optimisation loop)
!!
!! DATA
!! $Date$                   
!! $Rev$                                                              
!! $Author$                                                         
!! $URL$   
!! $Id$                      
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
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
!!****
module dlf_task_module
  logical, save :: tconverged ! is the last calculation converged
end module dlf_task_module

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* task/dlf_task
!!
!! FUNCTION
!! Assign and manage dl-find tasks
!!
!! In case of a direct coordinate transformation (glob%icoord<10):
!! * Find the number of degrees of freedom
!! * (Initialise HDLCs)
!! * Allocate global arrays (glob%icoords, glob%igradient, glob%step)
!!
!! In multiple image methods (NEB, dimer) call their respective init routines
!!
!! SYNOPSIS
subroutine dlf_task( &
#ifdef GAMESS
    core&
#endif
    )
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,printf
  use dlf_stat, only: stat
  use dlf_allocate, only: allocate,deallocate
  use dlf_task_module, only: tconverged
  use dlf_store
  implicit none
#ifdef GAMESS
  real(rk) :: core(*) ! GAMESS memory, not used in DL-FIND
#endif
  integer  :: task ! will become global!
  logical  :: tok
  real(8)  :: distort_in
! **********************************************************************

! 0      normal, no modification
! 1xxx   input near TS (+ TS-mode if avail.) 
! 1001   TS with dimer, then vibration freqyency
! 1011   TS with dimer/P-RFO, then find both downhill structures
!
! 2xxx   input RS + PS

!  glob%task=1001

  select case (glob%task)

! ======================================================================
! task = 0: standard optimisation, input unmodified
! ======================================================================
  case (0)
print*,"dlf_task task=0"
    ! main optimisation cycle
    call dlf_run( &
#ifdef GAMESS
        core&
#endif
        )

! ======================================================================
! task = 1001: TS with dimer, then vibration frequency
! ======================================================================
  case (1001)
    
    ! switch off any NEB attempts
    if( glob%icoord>=100 .and. glob%icoord<200) then
      glob%icoord=mod(glob%icoord,10)
    end if

    ! make sure the dimer method is used
    if( glob%icoord<200 .or. glob%icoord>299) then
      glob%icoord=mod(glob%icoord,10)+220
    end if

    if(printl>=2) &
        write(stdout,1000) "Searching the transition state with the dimer method"


    ! main optimisation cycle
    call dlf_run( &
#ifdef GAMESS
        core&
#endif
        )

    ! catch the case of a not-converging initial calculation
    if(.not.tconverged.and. printl >=2 ) then
      write(stdout,1000) "Dimer optimisation not converged, skipping"
      write(stdout,1000) " the frequency calculation"
      return
    end if
    
    if(printl>=2) &
        write(stdout,1000) "Calculating the imaginary frequency with the dimer method"
    
    ! make sure direct coordinates are cartesians
    glob%icoord=glob%icoord-mod(glob%icoord,10)

    if(.not.allocated(glob%xcoords2)) call dlf_fail( &
        "Error: glob%xcoords2 should be allocated after dimer run")
    
    ! get the end coordinates after the last run
    call dlf_formstep_get_ra("TSCOORDS",glob%nat*3,glob%xcoords,tok)

    ! send the TS coords to the calling program to deal with (e.g. write to a file)
    call dlf_put_coords(glob%nvar,3,glob%energy,glob%xcoords,glob%iam)
    
    ! make sure TS-mode from the first run is used
    call dlf_formstep_get_ra("TSMODE_R",glob%nat*3,glob%xcoords2(:,:,1),tok)
    if(.not.tok) then
      if(printl>=2) &
          write(stdout,1000) "Warning: unable to use dimer direction from previous run"
      glob%tcoords2=.false.
      call deallocate(glob%xcoords2)
    end if

    glob%massweight=.true.
    glob%maxrot=40
    glob%tolrot=0.1D0
    glob%maxcycle=1
    ! make sure input is not converged
    glob%tolerance=glob%tolerance * 0.001D0 

    call dlf_stat_reset

    ! main optimisation cycle
    call dlf_run( &
#ifdef GAMESS
        core&
#endif
        )

    ! send the TS mode to the calling program to deal with (e.g. write to a file)
    call dlf_put_coords(glob%nvar,2,0.0D0,glob%xcoords2(:,:,1),glob%iam)

    if(printl>=2) &
        write(stdout,1000) "Finished: Transition mode and imag. frequency have been calculated."

! ======================================================================
! task = 1011: TS search (dimer or P-RFO), then find both downhill structures
! ======================================================================
  case (1011)

    ! switch off any NEB attempts
    if( glob%icoord>=100 .and. glob%icoord<200) then
      glob%icoord=mod(glob%icoord,10)
    end if

    if(glob%iopt/=10) then
      if( glob%icoord<200 .or. glob%icoord>299) then
        ! nither P-RFO nor dimer chosen in the input. Choosing P-RFO
        glob%icoord=mod(glob%icoord,10)+220
        glob%iopt=3
        glob%iline=0
        if(printl>=2) then
          write(stdout,1000) "A transition state search was requested using tasks,"
          write(stdout,1000) " however, no TS option was chosen. Using the dimer method"
        end if
      end if
    end if

    if(abs(glob%distort) > 0.D0 ) then
      distort_in=glob%distort
    else
      distort_in=0.1D0
    end if
    glob%distort=0.D0

    if(printl>=2) &
        write(stdout,1000) "Searching the transition state"

    ! main optimisation cycle
    call dlf_run( &
#ifdef GAMESS
        core&
#endif
        )

    ! catch the case of a not-converging initial calculation
    if(.not.tconverged.and. printl >=2 ) then
      write(stdout,1000) "Transition state optimisation not converged, skipping"
      write(stdout,1000) " the downhill runs"
      return
    end if

    ! store the transition structure
    call dlf_formstep_get_ra("TSCOORDS",glob%nat*3,glob%xcoords,tok)
    call store_allocate("tscoords",3*glob%nat)
    call store_set("tscoords",3*glob%nat,glob%xcoords)

    ! store the transition mode
    ! make sure TS-mode from the first run is used
    call dlf_formstep_get_ra("TSMODE_R",glob%nat*3,glob%xcoords2(:,:,1),tok)
    if(.not.tok) then
      write(stdout,1000) "Transition mode not available, skipping"
      write(stdout,1000) " the downhill runs"
      return
    end if

    call store_allocate("tsmode",3*glob%nat)
    call store_set("tsmode",3*glob%nat,glob%xcoords2(:,:,1))

    ! write an xyz file with the TS structure
    if(printf>=4 .and. glob%iam == 0) then
      open(unit=601,file="TS.xyz")
      call write_xyz(601,glob%nat,glob%znuc,glob%xcoords)
      close(601)
      call dlf_put_coords(glob%nvar,3,glob%energy,glob%xcoords,glob%iam)
    end if

    ! Now minimise the system downhill with positive distort
    glob%iopt=3
    glob%iline=1
    glob%toldenergy=.false. ! to initiate a new trust radius
    glob%distort=distort_in
    ! switch off dimer method
    if(glob%icoord>100) glob%icoord=mod(glob%icoord,10)

    call dlf_stat_reset

    if(printl>=2) &
        write(stdout,1000) "Transition state converged, now minimising &
        &downhill into the first direction"

    ! main optimisation cycle
    call dlf_run( &
#ifdef GAMESS
        core&
#endif
        )

    if(printl>=2) then
        write(stdout,1000) "Downhill minimsation in the first&
        & direction has been finished."
        write(stdout,1000) "Now minimising downhill into the second &
            &direction."
      end if

    ! write an xyz file with the minimum structure
    if(tconverged.and.printf>=4 .and. glob%iam == 0) then
      open(unit=601,file="minimum_+.xyz")
      call write_xyz(601,glob%nat,glob%znuc,glob%xcoords)
      close(601)
      call dlf_put_coords(glob%nvar,4,glob%energy,glob%xcoords,glob%iam)
    end if

    ! Now minimise the system downhill with negative distort

    call store_get("tscoords",3,glob%nat,glob%xcoords)
    call store_get("tsmode",3,glob%nat,glob%xcoords2(:,:,1))
    

    glob%toldenergy=.false. ! to initiate a new trust radius
    glob%distort=-distort_in

    call dlf_stat_reset

    ! main optimisation cycle
    call dlf_run( &
#ifdef GAMESS
        core&
#endif
        )

    ! write an xyz file with the minimum structure
    if(tconverged.and.printf>=4 .and. glob%iam == 0) then
      open(unit=601,file="minimum_-.xyz")
      call write_xyz(601,glob%nat,glob%znuc,glob%xcoords)
      close(601)
      call dlf_put_coords(glob%nvar,5,glob%energy,glob%xcoords,glob%iam)
    end if

    call store_delete("tscoords")
    call store_delete("tsmode")

    if(printl>=2) &
        write(stdout,1000) "Finished: Downhill minimsation in the second&
        & direction has been finished."

! ======================================================================
! Wrong task setting
! ======================================================================
  case default
    write(stderr,*) "Task number",glob%task," not implemented"
    call dlf_fail("Task number error")

  end select

1000 format ("TaskManager: ",a)

end subroutine dlf_task
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* task/dlf_task_set_l
!!
!! FUNCTION
!! Set logical entries in the task_module
!!
!! SYNOPSIS
subroutine dlf_task_set_l(label,value)
!! SOURCE
  use dlf_task_module
  implicit none
  character(*), intent(in) :: label
  logical     , intent(in) :: value
! **********************************************************************
  if (label=="CONVERGED") then
    tconverged=value
  else
    call dlf_fail("Wrong label in dlf_task_set_l")
  end if
end subroutine dlf_task_set_l
!!****
