! **********************************************************************
! **                Coordinate transformation: main unit              **
! **                                                                  **
! **   Subunits:                                                      **
! **     dlf_neb.f90                                                  **
! **     dlf_dimer.f90                                                **
! **     dlf_hdlc_interface.f90                                       **
! **                                                                  **
! **                                                                  **
! **                                                                  **
! **********************************************************************
!!****h* DL-FIND/coords
!!
!! NAME
!! coords
!!
!! FUNCTION
!! Coordinate transformation: main unit
!!
!! Weight transformation is calculated at coordinate initialisation
!!
!! DATA
!! $Date: 2010-11-29 17:09:49 +0100 (Mon, 29 Nov 2010) $                   
!! $Rev: 451 $                                                              
!! $Author: twk $                                                         
!! $URL: http://ccpforge.cse.rl.ac.uk/svn/dl-find/trunk/dlf_coords.f90 $   
!! $Id: dlf_coords.f90 451 2010-11-29 16:09:49Z twk $                      
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

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_coords_init
!!
!! FUNCTION
!! Initialise coordinate transformation
!!
!! In case of a direct coordinate transformation (glob%icoord<10):
!! * Find the number of degrees of freedom
!! * (Initialise HDLCs)
!! * Allocate global arrays (glob%icoords, glob%igradient, glob%step)
!!
!! In multiple image methods (NEB, dimer) call their respective init routines
!!
!! SYNOPSIS
subroutine dlf_coords_init
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer               :: iat,ivar
  real(rk)              :: svar
  real(rk), allocatable :: tmp_icoords(:)
  real(RK), external    :: ddot
  logical               :: tok
! **********************************************************************

  ! ====================================================================
  ! CHECK FOR A CONSISTENT RESIDUE SPECIFICATION
  ! ====================================================================
  if(mod(glob%icoord,10)==3.or.mod(glob%icoord,10)==4) then
    ! The only allowed residue specification is: all atoms are 
    ! optimised, and all are part of one residue
    if(minval(glob%spec(:)) < 0) call dlf_fail("Frozen atoms are not &
        &permitted when using pure internals, use HDLC instead")
    if(minval(glob%spec(:)) /= maxval(glob%spec(:))) &
        call dlf_fail("All atoms must belong to the same residue for &
        &pure internals")
    if(maxval(glob%spec(:))==0) glob%spec(:)=1
  end if

  select case (glob%icoord)

! ======================================================================
! Cartesians
! ======================================================================
  case (0, 10)
    call dlf_cartesian_get_nivar(0, glob%nivar)
    if(printl >=4) then
      do iat=1,glob%nat
        select case (glob%spec(iat))
        case (0)
          write(stdout,1000) iat,"free"
        case (-1)
          write(stdout,1000) iat,"frozen"
        case (-2)
          write(stdout,1000) iat,"x frozen"
        case (-3)
          write(stdout,1000) iat,"y frozen"
        case (-4)
          write(stdout,1000) iat,"z frozen"
        case (-23)
          write(stdout,1000) iat,"x and y frozen"
        case (-24)
          write(stdout,1000) iat,"x and z frozen"
        case (-34)
          write(stdout,1000) iat,"y and z frozen"
        case default
          write(stdout,1001) iat,"free. Spec. residue:",glob%spec(iat)
        end select
      end do
    end if

    ! Extra coordinates for Lagrange-Newton
    if (glob%icoord == 10) glob%nivar = glob%nivar + 2

    ! allocate arrays
    call allocate( glob%icoords,glob%nivar)
    call allocate( glob%igradient,glob%nivar)
    call allocate( glob%step,glob%nivar)
    call allocate( glob%iweight,glob%nivar)
    glob%icoords(:)=0.D0
    glob%igradient(:)=0.D0
    glob%step(:)=0.D0
    glob%iweight(:)=0.D0

    ! Allocate conint arrays
    ! and initialise extra Lagrange-Newton coordinates
    if (glob%icoord == 10) call dlf_ln_allocate

    ! set weight
    ivar=1
    do iat=1,glob%nat
      if(glob%spec(iat)>=0.or.(glob%iopt>=102.and.glob%iopt<=103) ) then
        glob%iweight(ivar:ivar+2)=glob%weight(iat)
        ivar=ivar+3
      else if(glob%spec(iat)==-1) then
      else if(glob%spec(iat)>=-4) then
        glob%iweight(ivar:ivar+1)=glob%weight(iat)
        ivar=ivar+2
      else
        glob%iweight(ivar)=glob%weight(iat)
        ivar=ivar+1
      end if
    end do
    ! Set some sensible weights for the extra coordinates
    if (glob%icoord == 10) then
       glob%iweight(glob%nivar - 1:glob%nivar) = 1.0d0
       ivar = ivar + 2
    endif
    if(ivar-1/=glob%nivar) then
      call dlf_fail("Error in cartesian iweight calculation")
    end if

    ! Number of variables in inner region for microiterative calculations
    call dlf_cartesian_get_nivar(1, glob%nicore)

! ======================================================================
! HDLC/DLC
! ======================================================================
  case (1:4, 11:14)

		call dlf_hdlc_init(glob%nat,glob%spec,mod(glob%icoord,10),glob%ncons, &
        glob%icons,glob%nconn,glob%iconn)

		call dlf_hdlc_get_nivar(0, glob%nivar)
		call dlf_hdlc_get_nivar(1, glob%nicore)
        if((glob%iopt ==100) .AND. (glob%icoord == 3 .OR. glob%icoord == 4)) then
          allocate(glob%b_hdlc(glob%nivar,3*glob%nat))
        endif
        if((glob%iopt ==101) .AND. (glob%icoord == 3 .OR. glob%icoord == 4)) then
          call dlf_fail("Ts search with GPR in internal coordiantes not yet implemented!")
        endif
    ! calculate weights here
		call dlf_hdlc_create(glob%nat,glob%nicore,glob%spec,glob%micspec, &
        glob%znuc,1,glob%xcoords,glob%weight,glob%mass)

    ! Extra coordinates for Lagrange-Newton
    if ((glob%icoord / 10) == 1) glob%nivar = glob%nivar + 2

    !allocate arrays
    
    if (glob%gpr_internal == 2) then
      call allocate( glob%igradient,glob%nvar)
      call allocate( glob%step,glob%nvar)
      call allocate(glob%icoords2,glob%nivar)
      call allocate( glob%icoords,glob%nvar)
      glob%icoords2(:)=0.0d0
    else
      call allocate( glob%igradient,glob%nivar)
      call allocate( glob%step,glob%nivar)
      call allocate( glob%icoords,glob%nivar)
    endif

    glob%icoords(:)=0.D0
    glob%igradient(:)=0.D0
    glob%step(:)=0.D0

    ! Allocate conint arrays
    ! and initialise extra Lagrange-Newton coordinates
    if ((glob%icoord / 10) == 1) call dlf_ln_allocate

    ! get weights
    call allocate( glob%iweight,glob%nivar)

    if ((glob%icoord / 10) == 1) then
       call dlf_hdlc_getweight(glob%nat, glob%nivar - 2, glob%nicore, &
            glob%micspec, glob%weight, glob%iweight(1:glob%nivar - 2))
       ! Set some sensible weights for the extra coordinates
       glob%iweight(glob%nivar - 1:glob%nivar) = 1.0d0
    else
       call dlf_hdlc_getweight(glob%nat,glob%nivar,glob%nicore,&
            glob%micspec,glob%weight,glob%iweight)
    endif

! ======================================================================
! NEB 
! ======================================================================
  case (100:199)
    call dlf_neb_init(glob%nimage,glob%icoord)

! ======================================================================
! Dimer Method
! ======================================================================
  case (200:299)
    call dlf_dimer_init(glob%icoord)

! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error")

  end select

  ! distort if requested and a single-image method is used
  if(glob%icoord<100 .and. glob%tcoords2 .and. abs(glob%distort)>0.D0) then
    ! make xcoords2 relative
    if(.not.glob%tsrelative) glob%xcoords2(:,:,1)=glob%xcoords2(:,:,1)-glob%xcoords
    ! guess at distance distort in xyz space
    svar=dsqrt(ddot(glob%nvar,glob%xcoords2(:,:,1),1,glob%xcoords2(:,:,1),1))
    glob%xcoords2(:,:,1)=glob%xcoords2(:,:,1) / svar * glob%distort ! SIGN ENTERS HERE
    ! make xcoords2 absolute
    glob%xcoords2(:,:,1)=glob%xcoords2(:,:,1)+glob%xcoords

    ! calculate internals for coords
    glob%xgradient(:,:)=0.D0
    call dlf_direct_xtoi(glob%nvar,glob%nivar,glob%nicore,glob%xcoords,glob%xgradient, &
        glob%icoords,glob%igradient)
    ! calculate internals for coords2 (stored in tmp_icoords)
    call allocate(tmp_icoords, glob%nivar)
    call dlf_direct_xtoi(glob%nvar,glob%nivar,glob%nicore,glob%xcoords2(:,:,1), &
        glob%xgradient,tmp_icoords,glob%igradient)
    ! make tmp_icoords relative
    tmp_icoords=tmp_icoords-glob%icoords
    svar=dsqrt(ddot(glob%nivar,tmp_icoords,1,tmp_icoords,1))
    ! now distort icoords
    if (.not.(glob%iopt/10==6)) then ! do not distort for IRC
      glob%icoords=glob%icoords+tmp_icoords/svar*abs(glob%distort)
    end if
    call deallocate(tmp_icoords)
    ! transform icoords back to xcoords 
    call dlf_direct_itox(glob%nvar,glob%nivar,glob%nicore, &
        glob%icoords,glob%xcoords,tok)
    if(.not.tok) then
      call dlf_fail("Back transformation after distort failed. Use a&
          & smaller value for distort.")
    end if
  end if

! formats
1000 format ("Atom ",i6,2x,a)
1001 format ("Atom ",i6,2x,a,2x,i6)

end subroutine dlf_coords_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_coords_destroy
!!
!! FUNCTION
!! Deallocate global arrays belonging to internal coordinates
!!
!! SYNOPSIS
subroutine dlf_coords_destroy
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_allocate, only: deallocate
  implicit none
! **********************************************************************
  select case (glob%icoord)

! ======================================================================
! Cartesians
! ======================================================================
  case (0, 10)

    ! deallocate arrays
    if (allocated(glob%icoords)) call deallocate( glob%icoords)
    if (allocated(glob%igradient)) call deallocate( glob%igradient)
    if (allocated(glob%step)) call deallocate( glob%step)
    if (allocated(glob%iweight)) call deallocate( glob%iweight)

! ======================================================================
! HDLC/DLC
! ======================================================================
  case (1:4, 11:14)

    ! deallocate arrays
    if (allocated(glob%icoords)) call deallocate( glob%icoords)
    if (allocated(glob%igradient)) call deallocate( glob%igradient)
    if (allocated(glob%step)) call deallocate( glob%step)
    if (allocated(glob%iweight)) call deallocate( glob%iweight)
    if (allocated(glob%b_hdlc)) call deallocate(glob%b_hdlc)
    if(allocated(glob%icoords2)) call deallocate(glob%icoords2)
    call dlf_hdlc_destroy

! ======================================================================
! NEB 
! ======================================================================
  case (100:199)
    call dlf_neb_destroy

! ======================================================================
! Dimer Method
! ======================================================================
  case (200:299)
    call dlf_dimer_destroy

! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error")

  end select

end subroutine dlf_coords_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_coords_xtoi
!!
!! FUNCTION
!! Transform Cartesian coordinates and gradient to internals
!!
!! In case of a direct coordinate transformation, call dlf_direct_xtoi
!!
!! In multiple image methods (NEB, dimer) call their respective xtoi routines
!!
!! INPUTS
!!
!! glob%nvar, glob%nivar, glob%xcoords, glob%xgradient
!!
!! OUTPUTS
!! 
!! glob%icoords, glob%igradient, trerun_energy, testconv
!!
!! SYNOPSIS
subroutine dlf_coords_xtoi(trerun_energy,testconv,iimage)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout, stderr, printl
  implicit none
  logical, intent(out) :: trerun_energy
  logical, intent(out) :: testconv ! is convergence tested here?
  integer, intent(out) :: iimage ! which image to be calculated next?
! **********************************************************************

  testconv=.false.
  iimage=1
  
  if(printl>=6) write(stdout,*) "Transforming X to I"
  trerun_energy=.false.

  select case (glob%icoord/10)

! ======================================================================
! Direct coordinate transform
! ======================================================================
  case (0)
    if ((glob%iopt ==100) .and. glob%icoord==3) then
      call dlf_gpr_xtoi(glob%nvar,glob%nivar,glob%nicore,glob%xcoords, &
          glob%xgradient,glob%icoords,glob%igradient,glob%icoords2)
    else
      call dlf_direct_xtoi(glob%nvar,glob%nivar,glob%nicore,glob%xcoords, &
          glob%xgradient,glob%icoords,glob%igradient)
    endif

! ======================================================================
! Lagrange-Newton coordinates
! ======================================================================
  case (1)

    call dlf_ln_xtoi

! ======================================================================
! NEB 
! ======================================================================
  case (10:19) ! 100-199

    if(glob%iopt==12) then
      if(glob%inithessian==7) then
        call dlf_neb_xtoi(trerun_energy,iimage)
        ! read QTS hessian after all images have been re-calculated:
        if(iimage==1) call dlf_qts_get_hessian(trerun_energy)
      else
        call dlf_qts_get_hessian(trerun_energy)
      end if
    else
      call dlf_neb_xtoi(trerun_energy,iimage)
    end if

! ======================================================================
! Dimer 
! ======================================================================
  case (20:29) ! 200-299

    call dlf_dimer_xtoi(trerun_energy,testconv)

! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error")

  end select

  ! call the test for different deltas here
  if(glob%iopt==9) &
      call dlf_test_delta(trerun_energy)

end subroutine dlf_coords_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_coords_itox
!!
!! FUNCTION
!! Transform internal coordinates to xyz coordinates, handle breakdowns.
!! Gradient is not back-transformed.
!!
!! In case of a direct coordinate transformation, call dlf_direct_itox.
!!
!! In multiple image methods (NEB, dimer) call their respective itox routines.
!!
!! INPUTS
!!
!! glob%nvar, glob%nivar, glob%icoords
!!
!! OUTPUTS
!! 
!! glob%xcoords
!!
!! SYNOPSIS
subroutine dlf_coords_itox(iimage)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  implicit none
  integer, intent(out) :: iimage ! which image to be calculated next?  
  logical :: tok
! **********************************************************************
  iimage=1

  if(printl>=6) write(stdout,'("Transforming I to X")')

  select case (glob%icoord/10)

! ======================================================================
! Direct coordinate transform
! ======================================================================
  case (0)

    
    if(glob%gpr_internal ==2) then

     call dlf_gpr_itox(glob%nvar,glob%nivar,glob%nicore,glob%icoords,glob%xcoords,tok)
   else 
      call dlf_direct_itox(glob%nvar,glob%nivar,glob%nicore,glob%icoords,glob%xcoords,tok)
    
    if(.not.tok.and. mod(glob%icoord,10)>0 .and. mod(glob%icoord,10)<=4 ) then
      if(printl>=4) write(stdout, &
          "('HDLC coordinate breakdown. Recalculating HDLCs and &
          &restarting optimiser.')")
      call dlf_hdlc_reset
      ! the arrays glob%spec,glob%znuc have to be changed when using 
      !  more instances of hdlc
      call dlf_hdlc_create(glob%nvar/3,glob%nicore,glob%spec,glob%micspec, &
          glob%znuc,1,glob%xcoords,glob%weight,glob%mass)
      ! recalculate iweight
      call dlf_hdlc_getweight(glob%nat,glob%nivar,glob%nicore,glob%micspec, &
           glob%weight,glob%iweight)
      call dlf_formstep_restart
      glob%havehessian=.false.
    end if
    endif

! ======================================================================
! Lagrange-Newton coordinates
! ======================================================================
  case (1)

    call dlf_direct_itox(glob%nvar,glob%nivar - 2,glob%nicore, &
         glob%icoords(1:glob%nivar - 2),glob%xcoords,tok)
    if(.not.tok.and. mod(glob%icoord,10)>0 .and. mod(glob%icoord,10)<=4 ) then
      if(printl>=4) write(stdout, &
          "('HDLC coordinate breakdown. Recalculating HDLCs and &
          &restarting optimiser.')")
      call dlf_hdlc_reset
      ! the arrays glob%spec,glob%znuc have to be changed when using 
      !  more instances of hdlc
      call dlf_hdlc_create(glob%nvar/3,glob%nicore,glob%spec,glob%micspec, &
          glob%znuc,1,glob%xcoords,glob%weight,glob%mass)
      ! recalculate iweight
      call dlf_hdlc_getweight(glob%nat,glob%nivar - 2, glob%nicore, &
           glob%micspec, glob%weight, glob%iweight(1:glob%nivar - 2))
      call dlf_formstep_restart
      glob%havehessian=.false.
    end if

! ======================================================================
! NEB 
! ======================================================================
  case (10:19) ! 100-199

    call dlf_neb_itox(iimage)

! ======================================================================
! Dimer 
! ======================================================================
  case (20:29) ! 200-299

    call dlf_dimer_itox

! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error")

  end select

end subroutine dlf_coords_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_direct_xtoi
!!
!! FUNCTION
!! Transform xyz coordinates to internal coordinates
!! Only one set of coordinates is transformed, not a path 
!! (as possible by dlf_coords_xtoi)
!!
!! x-weights are transformed to i-weights
!!
!! INPUTS
!!
!! only local variables
!!
!! OUTPUTS
!! 
!! only local variables except for glob%massweight
!!
!! SYNOPSIS
subroutine dlf_direct_xtoi(nvar,nivar,nicore,xcoords,xgradient,icoords,&
    igradient)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  implicit none
  integer ,intent(in)  :: nvar,nivar,nicore
  real(rk),intent(in)  :: xcoords(nvar)
  real(rk),intent(in)  :: xgradient(nvar)
  real(rk),intent(out) :: icoords(nivar)
  real(rk),intent(out) :: igradient(nivar)
! **********************************************************************
  select case (mod(glob%icoord,10))
! ======================================================================
! Cartesian coordinates
! ======================================================================
  case (0)
    if(glob%tatoms) then
      call dlf_cartesian_xtoi(nvar/3,nivar,nicore,glob%massweight, &
          xcoords,xgradient,icoords,igradient)
    else
      icoords(:)=xcoords(:)
      igradient(:)=xgradient(:)
    end if
! ======================================================================
! HDLC/DLC
! ======================================================================
  case (1:4)

    call dlf_hdlc_xtoi(nvar/3,nivar,nicore,glob%micspec, &
         xcoords,xgradient,icoords,igradient)
    
    
    if (glob%iopt == 100) then
		igradient(:) = xgradient(:)
	endif

! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error (direct)")

  end select
end subroutine dlf_direct_xtoi
!!****

subroutine dlf_gpr_xtoi(nvar,nivar,nicore,xcoords,xgradient,icoords,&
    igradient,icoords2)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  implicit none
  integer ,intent(in)  :: nvar,nivar,nicore
  real(rk),intent(in)  :: xcoords(nvar)
  real(rk),intent(in)  :: xgradient(nvar)
  real(rk),intent(out) :: icoords(nvar)
  real(rk),intent(out) :: igradient(nvar)
  real(rk),intent(out)  :: icoords2(nivar)

  if(glob%gpr_internal == 2) then
  call dlf_hdlc_xtoi(nvar/3,nivar,nicore,glob%micspec, &
         xcoords,xgradient,icoords2,igradient)      
  call dlf_cartesian_xtoi(nvar/3,nvar,nicore,glob%massweight, &
          xcoords,xgradient,icoords,igradient)  
  elseif(glob%gpr_internal == 1) then
    call dlf_hdlc_xtoi(nvar/3,nivar,nicore,glob%micspec, &
         xcoords,xgradient,icoords,igradient)
  else
    call dlf_error("This method for GPR internal is not implented")
  endif
  !igradient(:) = xgradient(:)
end subroutine dlf_gpr_xtoi

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_direct_itox
!!
!! FUNCTION
!! Transform internal coordinates to xyz coordinates, do not handle breakdowns
!! Only one set of coordinates is transformed, not a path 
!! (as possible by dlf_coords_itox)
!!
!! INPUTS
!!
!! only local variables
!!
!! OUTPUTS
!! 
!! only local variables
!!
!! SYNOPSIS
subroutine dlf_direct_itox(nvar,nivar,nicore,icoords,xcoords,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer ,intent(in)   :: nvar,nivar,nicore
  real(rk),intent(in)   :: icoords(nivar)
  real(rk),intent(inout):: xcoords(nvar)
  logical ,intent(out)  :: tok
! **********************************************************************
  select case (mod(glob%icoord,10))
! ======================================================================
! Cartesian coordinates
! ======================================================================
  case (0)
    if(glob%tatoms) then
      call dlf_cartesian_itox(nvar/3,nivar,nicore,glob%massweight,icoords,xcoords)
    else
      xcoords(:)=icoords(:)
    end if
    tok=.true.

! ======================================================================
! HDLC/DLC
! ======================================================================
  case (1:4)
    if(glob%gpr_internal == 2 ) then
      call dlf_cartesian_itox(nvar/3,nvar,nicore,glob%massweight,glob%step,xcoords)
    else
      call dlf_hdlc_itox(nvar/3,nivar,nicore,glob%micspec,icoords,xcoords,tok)
    endif
	
! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error (direct)")

  end select
end subroutine dlf_direct_itox
!!****

subroutine dlf_gpr_itox(nvar,nivar,nicore,icoords,xcoords)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer ,intent(in)   :: nvar,nivar,nicore
  real(rk),intent(in)   :: icoords(nvar)
  real(rk),intent(inout):: xcoords(nvar)
  call dlf_cartesian_itox(nvar/3,nvar,nicore,glob%massweight,icoords,xcoords)
  !xcoords(:)=icoords(:)
end subroutine dlf_gpr_itox

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_direct_get_nivar
!!
!! FUNCTION
!! return the number of internal degrees of freedom to optimise for 
!! one image
!!
!! region = 0 - full region
!! region = 1 - inner (macroiterative) region
!! region = 2 - outer (microiterative) region
!!
!! INPUTS
!!
!! glob%spec
!!
!! OUTPUTS
!! 
!! local
!!
!! SYNOPSIS
subroutine dlf_direct_get_nivar(region, nivar)
!! SOURCE
  use dlf_global, only: glob,stderr
  implicit none
  integer, intent(in) :: region ! microiterative region
  integer, intent(out) :: nivar ! number of internal variables
! **********************************************************************
  select case (mod(glob%icoord,10))
  ! Cartesian coordinates
  case (0)
    call dlf_cartesian_get_nivar(region, nivar)
  ! HDLC/DLC
  case (1:4)
    call dlf_hdlc_get_nivar(region, nivar)
! ======================================================================
! Wrong coordinate setting
! ======================================================================
  case default
    write(stderr,*) "Coordinate type",glob%icoord,"not implemented"
    call dlf_fail("Coordinate type error (direct_get_nivar)")

  end select
end subroutine dlf_direct_get_nivar
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_get_nivar
!!
!! FUNCTION
!! return the number of degrees of freedom to optimise in Cartesian 
!! coordinates
!!
!! region = 0 - full region
!! region = 1 - inner (macroiterative) region
!! region = 2 - outer (microiterative) region
!!
!! INPUTS
!!
!! glob%spec, glob%micspec
!!
!! OUTPUTS
!! 
!! local
!!
!! SYNOPSIS
subroutine dlf_cartesian_get_nivar(region, nivar)
!! SOURCE
  use dlf_global, only: glob,stderr
  implicit none
  integer, intent(in) :: region ! microiterative region
  integer, intent(out) :: nivar ! number of internal variables
  integer :: iat
  logical :: warned
  ! negative spec values:
  ! -1  x,y,z frozen
  ! -2  x frozen
  ! -3  y frozen
  ! -4  z frozen
  ! -23 x and y frozen
  ! -24 x and z frozen
  ! -34 y and z frozen
! **********************************************************************
  warned=.false.
  if(.not.glob%tatoms) then
    ! no Cartesian constraints if no atoms as input
    nivar=glob%nvar
    return
  end if
  nivar=0
  do iat=1,glob%nat
    ! Ignore inner/outer regions as appropriate
    if (region == 1 .and. glob%micspec(iat) == 0) cycle
    if (region == 2 .and. glob%micspec(iat) == 1) cycle
    if(glob%spec(iat)>0) then
      if(.not.warned) then
        !print*,"Warning: fragments not used when Cartesian coordinates are requested!"
        warned=.true.
      end if
      nivar=nivar+3
    else if(glob%spec(iat)==0.or.(glob%iopt>=102.and.glob%iopt<=103)) then
      ! not frozen
      nivar=nivar+3
    else if(glob%spec(iat)==-1) then
    else if(glob%spec(iat)>=-4) then
      nivar=nivar+2
    else
      nivar=nivar+1
    end if
    ! invalid setting will be recognised in the conversion
  end do
end subroutine dlf_cartesian_get_nivar
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_xtoi
!!
!! FUNCTION
!! Transform xyz coordinates to "internal" Cartesian coordinates.
!! Do mass weighting eventually
!!
!! Meaning of negative glob%spec values:
!!   -1  x,y,z frozen
!!   -2  x frozen
!!   -3  y frozen
!!   -4  z frozen
!!   -23 x and y frozen
!!   -24 x and z frozen
!!   -34 y and z frozen
!!
!! INPUTS
!!
!! glob%spec, (glob%mass)
!!
!! OUTPUTS
!! 
!! only local variables
!!
!! SYNOPSIS
subroutine dlf_cartesian_xtoi(nat,nivar,nicore,massweight,xcoords,xgradient,&
    icoords,igradient)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl  
  implicit none
  integer ,intent(in)  :: nat,nivar,nicore
  logical ,intent(in)  :: massweight
  real(rk),intent(in)  :: xcoords(3,nat)
  real(rk),intent(in)  :: xgradient(3,nat)
  real(rk),intent(out) :: icoords(nivar)
  real(rk),intent(out) :: igradient(nivar)
  integer              :: iat,iivar,iinner,iouter
  logical              :: warned
  real(rk)             :: massf
! **********************************************************************
  if(.not.glob%tatoms) then
    call dlf_fail("dlf_cartesian_xtoi can only be used for atom input")
  end if
  warned=.false.
  ! Separate counts to order icoords into inner and outer groups
  iinner = 1
  iouter = nicore + 1
  do iat=1,glob%nat
    if(massweight) then
      massf=dsqrt(glob%mass(iat))
    else
      massf=1.D0
    end if
    if (glob%micspec(iat) == 1) then
       iivar = iinner
    else
       iivar = iouter
    end if
    if(glob%spec(iat)>0) then
      if(.not.warned) then
        !print*,"Warning: fragments not used when Cartesian coordinates are requested!"
        warned=.true.
      end if
      icoords(iivar:iivar+2)=massf*xcoords(:,iat)
      igradient(iivar:iivar+2)=xgradient(:,iat)/massf
      iivar=iivar+3
    else if(glob%spec(iat)==0.or.(glob%iopt>=102.and.glob%iopt<=103)) then
      ! not frozen
      icoords(iivar:iivar+2)=massf*xcoords(:,iat)
      igradient(iivar:iivar+2)=xgradient(:,iat)/massf
      iivar=iivar+3
    else if(glob%spec(iat)==-1) then
    else if(glob%spec(iat)==-2) then
      icoords(iivar:iivar+1)=massf*xcoords(2:3,iat)
      igradient(iivar:iivar+1)=xgradient(2:3,iat)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-3) then
      icoords(iivar:iivar+1)=massf*xcoords(1:3:2,iat)
      igradient(iivar:iivar+1)=xgradient(1:3:2,iat)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-4) then
      icoords(iivar:iivar+1)=massf*xcoords(1:2,iat)
      igradient(iivar:iivar+1)=xgradient(1:2,iat)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-23) then
      icoords(iivar)=massf*xcoords(3,iat)
      igradient(iivar)=xgradient(3,iat)/massf
      iivar=iivar+1
    else if(glob%spec(iat)==-24) then
      icoords(iivar)=massf*xcoords(2,iat)
      igradient(iivar)=xgradient(2,iat)/massf
      iivar=iivar+1
    else if(glob%spec(iat)==-34) then
      icoords(iivar)=massf*xcoords(1,iat)
      igradient(iivar)=xgradient(1,iat)/massf
      iivar=iivar+1
    else
      write(stderr,"('Spec setting of atom',i5,' is wrong:',i5)") &
          iat,glob%spec(iat)
      call dlf_fail("Wrong spec setting")
    end if
    if (glob%micspec(iat) == 1) then
       iinner = iivar
    else
       iouter = iivar
    end if    
  end do
  if(glob%icoord /= 3 .AND. glob%icoord /=4) then
  if(iinner /= nicore + 1) then
    if (printl>=4) write(stdout,'(I10, I10)') iinner-1,nicore
    call dlf_fail("Error in the transformation cartesian_xtoi (inner)")
  end if
  if(iouter /= nivar + 1) then
    if (printl>=4) write(stdout,'(I10, I10)') iouter-1,nivar
    call dlf_fail("Error in the transformation cartesian_xtoi (outer)")
  end if
  endif
end subroutine dlf_cartesian_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_itox
!!
!! FUNCTION
!! Convert internal cartesian coordinats (frozen components missing)
!! to the full set of cartesians. Do mass re-weighting eventually.
!!
!! IMPORTANT: this routine relies on the fact that the frozen components 
!! of glob%xcoords are not modified by any other routine than this one.
!!
!! INPUTS
!!
!! glob%spec, (glob%mass)
!!
!! OUTPUTS
!! 
!! only local variables
!!
!! SYNOPSIS
subroutine dlf_cartesian_itox(nat,nivar,nicore,massweight,icoords,xcoords)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use quick_molspec_module, only: xyz, quick_molspec
  implicit none
  integer ,intent(in)   :: nat,nivar,nicore
  logical ,intent(in)   :: massweight
  real(rk),intent(in)   :: icoords(nivar)
  real(rk),intent(inout):: xcoords(3,nat)
  integer               :: iat,jat,iivar,iinner,iouter
  logical               :: warned
  real(rk)              :: massf
! **********************************************************************
  if(.not.glob%tatoms) then
    call dlf_fail("dlf_cartesian_itox can only be used for atom input")
  end if
  warned=.false.
  ! Separate counts as icoords is ordered into inner and outer groups
  iinner = 1
  iouter = nicore + 1
  do iat=1,glob%nat
    if(massweight) then
      massf=dsqrt(glob%mass(iat))
    else
      massf=1.D0
    end if
    if (glob%micspec(iat) == 1) then
       iivar = iinner
    else
       iivar = iouter
    end if
    if(glob%spec(iat)>0) then
      if(.not.warned) then
        !print*,"Warning: fragments not used when cartesian coordinates are requested!"
        warned=.true.
      end if
      xcoords(:,iat)=icoords(iivar:iivar+2)/massf
      iivar=iivar+3
    else if(glob%spec(iat)==0.or.(glob%iopt>=102.and.glob%iopt<=103)) then
      ! not frozen
      xcoords(:,iat)=icoords(iivar:iivar+2)/massf
      iivar=iivar+3
    else if(glob%spec(iat)==-1) then
    else if(glob%spec(iat)==-2) then
      xcoords(2:3,iat)=icoords(iivar:iivar+1)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-3) then
      xcoords(1:3:2,iat)=icoords(iivar:iivar+1)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-4) then
      xcoords(1:2,iat)=icoords(iivar:iivar+1)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-23) then
      xcoords(3,iat)=icoords(iivar)/massf
      iivar=iivar+1
    else if(glob%spec(iat)==-24) then
      xcoords(2,iat)=icoords(iivar)/massf
      iivar=iivar+1
    else if(glob%spec(iat)==-34) then
      xcoords(1,iat)=icoords(iivar)/massf
      iivar=iivar+1
    else
      write(stderr,"('Spec setting of atom',i5,' is wrong:',i5)") &
          iat,glob%spec(iat)
      call dlf_fail("Wrong spec setting")
    end if
    if (glob%micspec(iat) == 1) then
       iinner = iivar
    else
       iouter = iivar
    end if
  end do
  if(glob%icoord /=4 .AND. glob%icoord /=3) then
  if(iinner /= nicore + 1) then
    if (printl>=4) write(stdout,'(I10, I10)') iinner-1,nicore
    call dlf_fail("Error in the transformation cartesian_itox (inner)")
  end if
  if(iouter /= nivar + 1) then
    if (printl>=4) write(stdout,'(I10, I10)') iouter-1,nivar
    call dlf_fail("Error in the transformation cartesian_itox (outer)")
  end if
  endif
end subroutine dlf_cartesian_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_gradient_itox
!!
!! FUNCTION
!! Convert an internal cartesian gradient (frozen components missing)
!! to the full set of cartesian gradient. Do mass re-weighting eventually.
!!
!! IMPORTANT: the gradient on frozen atoms is zeroed in this routine
!!
!! INPUTS
!!
!! glob%spec, (glob%mass)
!!
!! OUTPUTS
!! 
!! only local variables
!!
!! SYNOPSIS
subroutine dlf_cartesian_gradient_itox(nat,nivar,nicore,massweight,igradient,xgradient)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  implicit none
  integer ,intent(in)   :: nat,nivar,nicore
  logical ,intent(in)   :: massweight
  real(rk),intent(in)   :: igradient(nivar)
  real(rk),intent(out)  :: xgradient(3,nat)
  integer               :: iat,iivar,iinner,iouter
  logical               :: warned
  real(rk)              :: massf
! **********************************************************************
  if(.not.glob%tatoms) then
    call dlf_fail("dlf_cartesian_itox can only be used for atom input")
  end if
  if(mod(glob%icoord,10) /= 0 ) call dlf_fail("dlf_cartesian_gradient_itox&
      & can only be used for cartesian coordinates")
  xgradient(:,:)=0.D0
  warned=.false.
  ! Separate counts as icoords is ordered into inner and outer groups
  iinner = 1
  iouter = nicore + 1
  do iat=1,glob%nat
    if(massweight) then
      massf=1.D0/dsqrt(glob%mass(iat))
    else
      massf=1.D0
    end if
    if (glob%micspec(iat) == 1) then
       iivar = iinner
    else
       iivar = iouter
    end if
    if(glob%spec(iat)>0) then
      if(.not.warned) then
        !print*,"Warning: fragments not used when cartesian coordinates are requested!"
        warned=.true.
      end if
      xgradient(:,iat)=igradient(iivar:iivar+2)/massf
      iivar=iivar+3
    else if(glob%spec(iat)==0.or.(glob%iopt>=102.and.glob%iopt<=103)) then
      ! not frozen
      xgradient(:,iat)=igradient(iivar:iivar+2)/massf
      iivar=iivar+3
    else if(glob%spec(iat)==-1) then
    else if(glob%spec(iat)==-2) then
      xgradient(2:3,iat)=igradient(iivar:iivar+1)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-3) then
      xgradient(1:3:2,iat)=igradient(iivar:iivar+1)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-4) then
      xgradient(1:2,iat)=igradient(iivar:iivar+1)/massf
      iivar=iivar+2
    else if(glob%spec(iat)==-23) then
      xgradient(3,iat)=igradient(iivar)/massf
      iivar=iivar+1
    else if(glob%spec(iat)==-24) then
      xgradient(2,iat)=igradient(iivar)/massf
      iivar=iivar+1
    else if(glob%spec(iat)==-34) then
      xgradient(1,iat)=igradient(iivar)/massf
      iivar=iivar+1
    else
      write(stderr,"('Spec setting of atom',i5,' is wrong:',i5)") &
          iat,glob%spec(iat)
      call dlf_fail("Wrong spec setting")
    end if
    if (glob%micspec(iat) == 1) then
       iinner = iivar
    else
       iouter = iivar
    end if
  end do
  if(iinner /= nicore + 1) then
    if (printl>=4) write(stdout,'(I10, I10)') iinner-1,nicore
    call dlf_fail("Error in the transformation cartesian_gradient_itox (inner)")
  end if
  if(iouter /= nivar + 1) then
    if (printl>=4) write(stdout,'(I10, I10)') iouter-1,nivar
    call dlf_fail("Error in the transformation cartesian_gradient_itox (outer)")
  end if
end subroutine dlf_cartesian_gradient_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_coords_hessian_xtoi
!!
!! FUNCTION
!! Transform Cartesian Hessian into internal Hessian
!!
!! INPUTS
!!
!! local vars (+ glob%spec, glob%mass)
!!
!! OUTPUTS
!! 
!! glob%ihessian
!!
!! SYNOPSIS
subroutine dlf_coords_hessian_xtoi(nvar,xhessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  implicit none
  integer, intent(in) :: nvar
  real(rk),intent(in) :: xhessian(nvar,nvar)
  integer :: i
! **********************************************************************
  select case (glob%icoord)
  ! Cartesians
  case (0)
    call dlf_cartesian_hessian_xtoi(glob%nat,nvar,glob%nivar,glob%massweight,xhessian,&
        glob%spec,glob%mass,glob%ihessian)
  ! HDLC/DLC
  case (1:4)
    call dlf_hdlc_hessian_xtoi(nvar/3,glob%nivar,glob%xcoords,xhessian,glob%ihessian)
  ! qTS in mass-weighted internal coordinates
  case (190)
    call qts_hessian_etos_halfpath 
    !call dlf_qts_get_hessian_external ! previous name of qts_hessian_etos_halfpath
  case default
    write(stderr,*) "Hessian transformation for coordinate type", &
        glob%icoord,"not implemented"
    call dlf_fail("Hessian transformation error")
  end select
end subroutine dlf_coords_hessian_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_hessian_xtoi
!!
!! FUNCTION
!! transform a Cartesian Hessian to an internal Cartesian (taking
!! frozen atoms into account. This routine may be improved for speed
!! and memory usage.
!!
!! INPUTS
!!
!! local vars (+ glob%spec, glob%mass)
!!
!! OUTPUTS
!! 
!! local vars
!!
!! SYNOPSIS
subroutine dlf_cartesian_hessian_xtoi(nat,nvar,nivar,massweight,xhessian,&
    spec,mass,ihessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  implicit none
  integer ,intent(in)  :: nat,nvar,nivar
  logical ,intent(in)  :: massweight
  real(rk),intent(in)  :: xhessian(nvar,nvar)
  integer ,intent(in)  :: spec(nat)
  real(rk),intent(in)  :: mass(nat)
  real(rk),intent(out) :: ihessian(nivar,nivar)
  integer              :: iat,iivar,ivar
  real(rk)             :: umat(nvar,nivar)
  real(rk)             :: umatt(nivar,nvar)
  real(rk)             :: tmpmat(nvar,nivar)
  real(rk)             :: massf
! **********************************************************************
  if(.not.glob%tatoms) then
    if(nvar/=nivar) call dlf_fail("Wrong number of DOF in dlf_cartesian_hessian_xtoi")
    ihessian=xhessian
    return
  end if
  ! set up umat
  umat=0.D0
  iivar=1
  do iat=1,nat
    if(massweight) then
      massf=1.D0/dsqrt(mass(iat))
    else
      massf=1.D0
    end if
    ivar=(iat-1)*3+1
    if(spec(iat)>=0.or.(glob%iopt>=102.and.glob%iopt<=103)) then
      ! not frozen
      umat(ivar  ,iivar  )=massf
      umat(ivar+1,iivar+1)=massf
      umat(ivar+2,iivar+2)=massf
      iivar=iivar+3
    else if(spec(iat)==-1) then
    else if(spec(iat)==-2) then
      umat(ivar+1,iivar  )=massf
      umat(ivar+2,iivar+1)=massf
      iivar=iivar+2
    else if(spec(iat)==-3) then
      umat(ivar  ,iivar  )=massf
      umat(ivar+2,iivar+1)=massf
      iivar=iivar+2
    else if(spec(iat)==-4) then
      umat(ivar  ,iivar  )=massf
      umat(ivar+1,iivar+1)=massf
      iivar=iivar+2
    else if(spec(iat)==-23) then
      umat(ivar+2,iivar  )=massf
      iivar=iivar+1
    else if(spec(iat)==-24) then
      umat(ivar+1,iivar  )=massf
      iivar=iivar+1
    else if(spec(iat)==-34) then
      umat(ivar  ,iivar  )=massf
      iivar=iivar+1
    else
      write(stderr,"('Spec setting of atom',i5,' is wrong:',i5)") &
          iat,spec(iat)
      call dlf_fail("Wrong spec setting")
    end if
  end do
  if(iivar/=nivar+1) then
    if (printl>=4) write(stdout,'(I10, I10)') iivar-1,nivar
    call dlf_fail("Error in the transformation cartesian_xtoi")
  end if
  ! now transform the hessian
  call dlf_matrix_multiply(nvar,nivar,nvar,1.D0,xhessian,umat,0.D0,tmpmat)
  umatt=transpose(umat)
  call dlf_matrix_multiply(nivar,nivar,nvar,1.D0,umatt,tmpmat,0.D0,ihessian)

end subroutine dlf_cartesian_hessian_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_hessian_itox
!!
!! FUNCTION
!! transform an internal Hessian to a Cartesian Hessian (taking
!! frozen atoms into account. This routine can be improved for speed
!! and memory usage.
!!
!! INPUTS
!!
!! local vars
!!
!! OUTPUTS
!! 
!! local vars
!!
!! SYNOPSIS
subroutine dlf_cartesian_hessian_itox(nat,nvar,nivar,massweight,ihessian,&
    spec,mass,xhessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr, glob,stdout,printl
  implicit none
  integer ,intent(in)  :: nat,nvar,nivar
  logical ,intent(in)  :: massweight
  real(rk),intent(in)  :: ihessian(nivar,nivar)
  integer ,intent(in)  :: spec(nat)
  real(rk),intent(in)  :: mass(nat)
  real(rk),intent(out) :: xhessian(nvar,nvar)
  integer              :: iat,iivar,ivar
  real(rk)             :: umat(nvar,nivar)
  real(rk)             :: umatt(nivar,nvar)
  real(rk)             :: tmpmat(nivar,nvar)
  real(rk)             :: massf
! **********************************************************************
  ! set up umat
  umat=0.D0
  iivar=1
  do iat=1,nat
    if(massweight) then
      massf=dsqrt(mass(iat))
    else
      massf=1.D0
    end if
    ivar=(iat-1)*3+1
    if(spec(iat)>=0.or.(glob%iopt>=102.and.glob%iopt<=103)) then
      ! not frozen
      umat(ivar  ,iivar  )=massf
      umat(ivar+1,iivar+1)=massf
      umat(ivar+2,iivar+2)=massf
      iivar=iivar+3
    else if(spec(iat)==-1) then
    else if(spec(iat)==-2) then
      umat(ivar+1,iivar  )=massf
      umat(ivar+2,iivar+1)=massf
      iivar=iivar+2
    else if(spec(iat)==-3) then
      umat(ivar  ,iivar  )=massf
      umat(ivar+2,iivar+1)=massf
      iivar=iivar+2
    else if(spec(iat)==-4) then
      umat(ivar  ,iivar  )=massf
      umat(ivar+1,iivar+1)=massf
      iivar=iivar+2
    else if(spec(iat)==-23) then
      umat(ivar+2,iivar  )=massf
      iivar=iivar+1
    else if(spec(iat)==-24) then
      umat(ivar+1,iivar  )=massf
      iivar=iivar+1
    else if(spec(iat)==-34) then
      umat(ivar  ,iivar  )=massf
      iivar=iivar+1
    else
      write(stderr,"('Spec setting of atom',i5,' is wrong:',i5)") &
          iat,spec(iat)
      call dlf_fail("Wrong spec setting")
    end if
  end do
  if(iivar/=nivar+1) then
    if (printl>=4) write(stdout,'(I10, I10)') iivar-1,nivar
    call dlf_fail("Error in the transformation cartesian_itox")
  end if
  ! now transform the hessian
  umatt=transpose(umat)
  call dlf_matrix_multiply(nivar,nvar,nivar,1.D0,ihessian,umatt,0.D0,tmpmat)
  call dlf_matrix_multiply(nvar,nvar,nivar,1.D0,umat,tmpmat,0.D0,xhessian)

!  call dlf_matrix_multiply(3*nat,nivar,3*nat,1.D0,xhessian,umat,0.D0,tmpmat)
!  call dlf_matrix_multiply(nivar,nivar,3*nat,1.D0,umatt,tmpmat,0.D0,ihessian)

end subroutine dlf_cartesian_hessian_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_coords_tranrot
!!
!! FUNCTION
!! Project out the translation and rotation component of a vector in the 
!! current internal coordinate system
!! at the moment, this is only done in Cartesian coordinates if no atom 
!! is frozen
!! vector may be a gradient (which should not contain rotation or 
!! translation anyway) or the dimer axis in the dimer method
!!
!! Rotation not yet implemented!
!!
!! INPUTS
!!
!! only local variables
!!
!! OUTPUTS
!! 
!! only local variables
!!
!! SYNOPSIS
subroutine dlf_coords_tranrot(nvar,vector)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,printl,stdout
  implicit none
  integer , intent(in)       :: nvar
  real(rk), intent(inout)    :: vector(nvar)
  real(rk)                   :: svar
  integer                    :: nat
! **********************************************************************
  if(mod(glob%icoord,10)/=0) return
  if(glob%massweight) return
  if(.not.glob%tatoms) return
  if(minval(glob%spec(:)) < 0) then
    if (printl>=4) write(stdout,'("Warning: removal of rotation and translation not possible&
        & for frozen atoms")')
    return
  end if
  nat=nvar/3
  svar=sum(vector(1:nvar:3))/dble(nat)
  if(printl>=6) write(stdout,'("Removing x-translation:",es12.4)') svar
  vector(1:nvar:3)=vector(1:nvar:3)-svar

  svar=sum(vector(2:nvar:3))/dble(nat)
  if(printl>=6) write(stdout,'("Removing y-translation:",es12.4)') svar
  vector(2:nvar:3)=vector(2:nvar:3)-svar

  svar=sum(vector(3:nvar:3))/dble(nat)
  if(printl>=6) write(stdout,'("Removing z-translation:",es12.4)') svar
  vector(3:nvar:3)=vector(3:nvar:3)-svar

  ! rotation not yet implemented - IMPROVE!
end subroutine dlf_coords_tranrot
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_cartesian_align
!!
!! FUNCTION
!! Translate and rotate coords2 to minimum square distance from coords1.
!! Both sets have to contain Cartesian coordinates.
!!
!! If one atom is frozen, rotation is done around this atom (no translation
!! is done).
!! If more than one atom is frozen, or other Cartesian constraints are
!! present, nothing is done.
!!
!! See W. Kabsch, Acta Cryst. A 32, p 922 (1976).
!! This follows an RMS best fit procedure.
!!
!! Attention: the input coordinates have to be Cartesians, not 
!! mass-weighted Cartesians!
!!
!! INPUTS
!!
!! only local variables
!!
!! OUTPUTS
!! 
!! only local variables
!!
!! SYNOPSIS
subroutine dlf_cartesian_align(nat,coords1,coords2)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,printl,stdout
  implicit none
  integer , intent(in)       :: nat
  real(rk), intent(inout)       :: coords1(3,nat)
  real(rk), intent(inout)    :: coords2(3,nat)
  !
  real(rk)                   :: weight(nat) ! now local, can be set on input
  real(rk)                   :: svar,detrot
  integer                    :: ivar,rotat,iat,i,j,itry
  real(rk)                   :: center(3),rmat(3,3),rsmat(3,3)
  real(rk)                   :: eigvec(3,3),eigval(3)
  real(rk)                   :: trans(3),rotmat(3,3)
! **********************************************************************
  if(.not.glob%tatoms) return
  if(minval(glob%spec(:)) < -1) return
  rotat=0
  if(minval(glob%spec(:)) < 0) then
    ivar=0
    do iat=1,nat
      if(glob%spec(iat)==-1) then
        ivar=ivar+1
        if(ivar>1) exit
        rotat=iat
      end if
    end do
    if(ivar>1) return
  end if
  weight(:)=1.D0
  ! remove translation
  ! get centre of coords1 and translational difference
  center(:)=0.D0
  trans(:)=0.D0
  if(rotat==0) then
    do iat=1,nat
      center(:)=center(:)+weight(iat)*coords1(:,iat)
      trans(:)=trans(:)+weight(iat)*(coords1(:,iat)-coords2(:,iat))
    end do
    center(:)=center(:)/sum(weight)
    trans(:)=trans(:)/sum(weight)
  else
    center(:)=coords1(:,rotat)
    trans(:)=coords1(:,rotat)-coords2(:,rotat)
  end if
  ! translate them to common centre
  do iat=1,nat
    coords2(:,iat)=coords2(:,iat)+trans(:)
  end do
  if(printl>=6) write(stdout,"('Translating by ',3f10.5)") trans

  ! now get rotation
  ! following W. Kabsch, Acta Cryst. A 32, p 922 (1976)
  rmat=0.D0
  do iat=1,nat
    do i=1,3
      do j=1,3
        rmat(i,j)=rmat(i,j)+weight(iat)*(coords1(i,iat)-center(i))* &
            (coords2(j,iat)-center(j))
      end do
    end do
  end do
  rmat=rmat/sum(weight)
  !write(*,"('R   ',3f10.3)") rmat
  rsmat=transpose(rmat)
  eigvec=matmul(rsmat,rmat)
  rsmat=eigvec

  !write(stdout,"('RtR ',3f10.3)") rsmat
  call dlf_matrix_diagonalise(3,rsmat,eigval,eigvec)
  !do i=1,3
  !  write(*,"('Eigval, vec a_k ',f10.3,5x,3f10.3)") eigval(i),eigvec(:,i)
  !end do

  ! It turns out that the rotation matrix may have a determinat of -1
  ! in the procedure used here, i.e. the system is mirrored - which is
  ! wrong chemically. This can be avoided by inserting a minus in the
  ! equation
  ! 1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))

  ! So, here we first calculate the rotation matrix, and if it is
  ! zero, the first eigenvalue is reversed

  do itry=1,2
    ! rsmat are the vectors b:
    j=-1
    do i=1,3
      if(eigval(i)<1.D-8) then
        if(i>1) then
          ! the system is linear - no rotation necessay.
          ! WHY ?! There should still be one necessary!
          return
          !print*,"Eigenval. zero",i,eigval(i)
          !call dlf_fail("Error in dlf_cartesian_align")
        end if
        j=1
      else
        if(i==1.and.itry==2) then
          rsmat(:,i)=-1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        else
          rsmat(:,i)=1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        end if
      end if
    end do
    if(j==1) then
      ! one eigenvalue was zero, the system is planar
      rsmat(1,1)=rsmat(2,2)*rsmat(3,3)-rsmat(3,2)*rsmat(2,3)
      rsmat(2,1)=rsmat(3,2)*rsmat(1,3)-rsmat(1,2)*rsmat(3,3)
      rsmat(3,1)=rsmat(1,2)*rsmat(2,3)-rsmat(2,2)*rsmat(1,3)
      ! deal with negative determinant
      if (itry==2) then
         rsmat(:,1) = -rsmat(:,1)
      end if
    end if

    do i=1,3
      do j=1,3
        rotmat(i,j)=sum(rsmat(i,:)*eigvec(j,:))
      end do
    end do
    !write(*,"('rotmat ',3f10.3)") rotmat
    detrot=   &
        rotmat(1,1)*(rotmat(2,2)*rotmat(3,3)-rotmat(2,3)*rotmat(3,2)) &
        -rotmat(2,1)*(rotmat(1,2)*rotmat(3,3)-rotmat(1,3)*rotmat(3,2)) &
        +rotmat(3,1)*(rotmat(1,2)*rotmat(2,3)-rotmat(1,3)*rotmat(2,2))
    !write(*,*) "Determinat of rotmat", detrot
    if(detrot > 0.D0) exit
    if(detrot < 0.D0 .and. itry==2) then
      call dlf_fail("Error in dlf_cartesian_align, obtained a mirroring instead of rotation.")
    end if

  end do

  do iat=1,nat
    coords2(:,iat)= coords2(:,iat)-center
    coords2(:,iat)=matmul(rotmat,coords2(:,iat))
    coords2(:,iat)= coords2(:,iat)+center
  end do
end subroutine dlf_cartesian_align
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_re_mass_weight_hessian
!!
!! FUNCTION 
!!
!! Get a hessian in mass-weighted coordinates and return one in
!! mass-weighted coordinates with different masses
!!
!! The hessian is in icoords. It will be converted back to xcoords
!! (partially: the frozen atoms will be ignored). Thus, the spec array
!! is needed.
!!
!! SYNOPSIS
subroutine dlf_re_mass_weight_hessian(nat,nivar,mass_in,mass_out,ihessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,printl,stdout
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer , intent(in)       :: nat ! global number of atoms
  integer , intent(in)       :: nivar ! number of DOF in ihessian
  real(rk), intent(in)       :: mass_in(nat)
  real(rk), intent(in)       :: mass_out(nat)
  real(rk), intent(inout)    :: ihessian(nivar,nivar)
  !
  integer, allocatable :: mapfrozen(:) ! nat
  integer :: iat
  integer :: natr ! number of non-frozen atoms
  integer, allocatable :: spec(:) ! natr
  real(rk),allocatable :: mass(:) ! natr
  real(rk),allocatable :: xhessian(:,:) ! (3*natr,3*natr)
  real(rk) :: svar

  if(nat/=glob%nat) call dlf_fail("Number of atoms inconsistent in dlf_re_mass_weight_hessian")

  call allocate(mapfrozen,nat)

  ! map to a smaller array of xcoords (only non-frozen atoms)
  natr=0
  mapfrozen(:)=0
  do iat=1,nat
    if(glob%spec(iat)/=-1) then
      natr=natr+1
      mapfrozen(iat)=natr ! mapfrozen runs from 1 to natr, its index from 1 to nat
    end if
  end do
  if(printl>=6) write(stdout,*) "Number of non-frozen atoms",natr
  
  call allocate(spec,natr)
  call allocate(mass,natr)
  spec=0
  mass=-1.D0

  do iat=1,nat
    if(mapfrozen(iat)>0) then
      mass(mapfrozen(iat))=mass_in(iat)
      spec(mapfrozen(iat))=glob%spec(iat)
    end if
  end do
  ! sanity check
  if(minval(mass)<0.D0) call dlf_fail("error in mass-array conversion")
  call allocate(xhessian,3*natr,3*natr)

  call dlf_cartesian_hessian_itox(natr,3*natr,nivar,.true.,ihessian,&
    spec,mass,xhessian)

  ! now modify masses
  do iat=1,nat
    if(mapfrozen(iat)>0) then
      ! only modify the masses if they differ by less than a factor of
      ! 4. This is to avoid problems with the wrong (surplus)
      ! conversion between atomic mass units and atomic units
      if(abs(mass_in(iat)-mass_out(iat))>1.D-5) then
        svar=abs(mass(mapfrozen(iat))/mass_out(iat))
        if(svar<10.D0.and.svar>0.1D0) then
          mass(mapfrozen(iat))=mass_out(iat)
          if(printl>=4) then
            write(stdout,'(a,i5,a,f10.6,a,f10.6)') "Mass change atom",iat,": old mass ",mass_in(iat), &
                " new mass ",mass_out(iat)
          end if
        else
          if(printl>=0) then
            write(stdout,'(a,i5,a,f10.6,a,f10.6,a)') "Warning: Mass of atom",iat,": old mass ",mass_in(iat), &
                " new mass ",mass_out(iat)," NOT CHANGED (unrealistic difference)"
          end if
        end if
      end if
    end if
  end do
  
  ! adjust args
  call dlf_cartesian_hessian_xtoi(natr,3*natr,nivar,.true.,xhessian,&
      spec,mass,ihessian)

  call deallocate(xhessian)
  call deallocate(mass)
  call deallocate(spec)
  call deallocate(mapfrozen)

end subroutine dlf_re_mass_weight_hessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_hessian_project
!!
!! FUNCTION
!! Project out rotation and translation from a hessian
!!
!! Projection of the translation works perfect. However, the rotation
!! apparently is a problem. Even with an analytic hessian
!! (LJ-particles) the rotational eigenvalues are not zero by
!! far. Using either of the impelemented rotation methods does not
!! work perfectly. Also using the projcted hessian for NR in qTS makes
!! things worse than ignoring soft modes
!!
!! SYNOPSIS
subroutine dlf_hessian_project(nvar,coords,hess)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,printl,stdout
  implicit none
  integer,  intent(in)   :: nvar
  real(rk), intent(in)   :: coords(nvar) !mass-weighted coordinates
  real(rk), intent(inout):: hess(nvar,nvar) ! hessian with respect to MW-coordinates
  !
  integer, parameter :: nzero=6
  integer   :: ivar,jvar,nimage
  integer   :: iat(nvar) ! iat(ivar)
  real(rk)  :: com(3) ! center of mass
  real(rk)  :: ccoords(nvar) ! coords centered with respect to com
  real(rk)  :: inert(3,3)
  real(rk)  :: minert(3),inertdir(3,3)
  real(rk)  :: zeromode(nvar,nzero) ! 3 translational modes and 3 rotational modes
  real(rk)  :: bmode(nvar,nzero) ! breathing mode
  real(rk)  :: tmpvec(nvar)
  real(rk)  :: svar
  real(rk)  :: eigvec(nvar,nvar)
  real(rk)  :: eigval(nvar)
  real(rk)  :: vibspace(nvar,nvar-nzero)
  real(rk)  :: testmode(nvar)
  real(rk)  :: evzero,evbreath,frac,offdiag,sumsvar
  ! new hessian
  real(rk)  :: vhess(nvar-nzero,nvar-nzero)
  real(rk)  :: veigval(nvar-nzero)
  real(rk)  :: veigvec(nvar-nzero,nvar-nzero)
  ! ********************************************************************
  nimage=nvar/3/glob%nat ! this may be dangerous

  if(nvar /= 3*glob%nat*nimage) call dlf_fail("Size of Hessian inconsistent in &
      &dlf_hessian_project")
  if(glob%nzero /= 6) call dlf_fail("dlf_hessian_project works only for &
      &free molecules (nzero=6)")

  ! set iat
  do ivar=1,nvar
    iat(ivar)=mod(int(dble(ivar-1)/3.D0),glob%nat)+1
  end do

  ! create the zero translational modes
  zeromode(:,:)=0.D0
  do ivar=1,nvar,3
    zeromode(ivar  ,1)=sqrt(glob%mass(iat(ivar)))
    zeromode(ivar+1,2)=sqrt(glob%mass(iat(ivar)))
    zeromode(ivar+2,3)=sqrt(glob%mass(iat(ivar)))
  end do
  ! normalize translational modes
  do ivar=1,3
    svar=sqrt(dot_product(zeromode(:,ivar),zeromode(:,ivar)))
    zeromode(:,ivar)=zeromode(:,ivar)/svar
    ! print for test
    tmpvec=matmul(hess,zeromode(:,ivar))
    svar=dot_product(zeromode(:,ivar),tmpvec)
    write(*,"('Expectation value of mode',i3,es15.7)") ivar,svar
  end do

!!$  ! expectation value of a random mode
!!$  call random_number(zeromode(:,nzero))
!!$  zeromode(:,nzero)=zeromode(:,nzero)-0.5D0
!!$  do ivar=nzero,nzero
!!$    svar=sqrt(dot_product(zeromode(:,ivar),zeromode(:,ivar)))
!!$    zeromode(:,ivar)=zeromode(:,ivar)/svar
!!$    ! print for test
!!$    tmpvec=matmul(hess,zeromode(:,ivar))
!!$    svar=dot_product(zeromode(:,ivar),tmpvec)
!!$    write(*,"('Expectation value of random mode',i3,es15.7)") ivar,svar
!!$  end do
!!$  zeromode(:,nzero)=0.D0

  ! calculate COM
  com=0.D0
  do ivar=1,nvar
    com(mod(ivar-1,3)+1)=com(mod(ivar-1,3)+1) + sqrt(glob%mass(iat(ivar)))*coords(ivar)
  end do
  com=com/sum(glob%mass(1:glob%nat))/dble(nimage) ! com in cartesians
  !print*,"center of mass",com

  ! calculate centered coords (MW)
  do ivar=1,nvar
    ccoords(ivar)=coords(ivar) - com(mod(ivar-1,3)+1)*sqrt(glob%mass(iat(ivar)))
  end do

  ! calculate inertia tensor
  inert=0.D0
  do ivar=1,nvar,3
    inert(1,1)=inert(1,1) + ccoords(ivar+1)**2+ccoords(ivar+2)**2 
    inert(1,2)=inert(1,2) - ccoords(ivar)*ccoords(ivar+1)
    inert(1,3)=inert(1,3) - ccoords(ivar)*ccoords(ivar+2)
    inert(2,2)=inert(2,2) + ccoords(ivar)**2 + ccoords(ivar+2)**2
    inert(2,3)=inert(2,3) - ccoords(ivar+1)*ccoords(ivar+2)
    inert(3,3)=inert(3,3) + ccoords(ivar)**2 + ccoords(ivar+1)**2
  end do
  inert(2,1)=inert(1,2)
  inert(3,1)=inert(1,3)
  inert(3,2)=inert(2,3)

!!$  write(*,'(3es15.7)') inert

  call dlf_matrix_diagonalise(3,inert,minert,inertdir)

!!$  print*,"Moments of inertia"
!!$  do ivar=1,3
!!$    write(stdout,"(es15.7,3x,3f10.5)") minert(ivar),inertdir(:,ivar)
!!$  end do

  testmode=0.D0
  bmode=0.D0
  ! create the zero translational modes
  if(nzero==6) then
  do ivar=1,nvar,3
    ! using just cartesian coordinates - should lead to the same space of rotations
    zeromode(ivar  ,4)=0.D0
    zeromode(ivar+1,4)=-ccoords(ivar+2)
    zeromode(ivar+2,4)=ccoords(ivar+1) 

    ! testmode: breathing mode orthogonal to x
    bmode(ivar  ,4)=0.D0
    bmode(ivar+1,4)=ccoords(ivar+1)
    bmode(ivar+2,4)=ccoords(ivar+2) 
    
    zeromode(ivar  ,5)=-ccoords(ivar+2)
    zeromode(ivar+1,5)=0.D0
    zeromode(ivar+2,5)=ccoords(ivar)
    !breathing mode orthogonal to y
    bmode(ivar  ,5)=ccoords(ivar)
    bmode(ivar+1,5)=0.D0
    bmode(ivar+2,5)=ccoords(ivar+2) 
    
    zeromode(ivar  ,6)=-ccoords(ivar+1)
    zeromode(ivar+1,6)=ccoords(ivar)
    zeromode(ivar+2,6)=0.D0
    !breathing mode orthogonal to z
    bmode(ivar  ,6)=ccoords(ivar)
    bmode(ivar+1,6)=ccoords(ivar+1)
    bmode(ivar+2,6)= 0.D0

!!$    ! using the principle directions of inertia
!!$    zeromode(ivar  ,4)=ccoords(ivar+1)*inertdir(3,1)-ccoords(ivar+2)*inertdir(2,1)
!!$    zeromode(ivar+1,4)=ccoords(ivar+2)*inertdir(1,1)-ccoords(ivar)*inertdir(3,1)
!!$    zeromode(ivar+2,4)=ccoords(ivar)*inertdir(2,1)-ccoords(ivar+1)*inertdir(1,1)
!!$
!!$    zeromode(ivar  ,5)=ccoords(ivar+1)*inertdir(3,2)-ccoords(ivar+2)*inertdir(2,2)
!!$    zeromode(ivar+1,5)=ccoords(ivar+2)*inertdir(1,2)-ccoords(ivar)*inertdir(3,2)
!!$    zeromode(ivar+2,5)=ccoords(ivar)*inertdir(2,2)-ccoords(ivar+1)*inertdir(1,2)
!!$
!!$    zeromode(ivar  ,6)=ccoords(ivar+1)*inertdir(3,3)-ccoords(ivar+2)*inertdir(2,3)
!!$    zeromode(ivar+1,6)=ccoords(ivar+2)*inertdir(1,3)-ccoords(ivar)*inertdir(3,3)
!!$    zeromode(ivar+2,6)=ccoords(ivar)*inertdir(2,3)-ccoords(ivar+1)*inertdir(1,3)
  end do
  end if

  ! check mode 4
  do jvar=4,nzero
    svar=sqrt(dot_product(zeromode(:,jvar),zeromode(:,jvar)))
    zeromode(:,jvar)=zeromode(:,jvar)/svar
    ! print for test
    tmpvec=matmul(hess,zeromode(:,jvar))
    evzero=dot_product(zeromode(:,jvar),tmpvec)
    write(*,"('Expectation value of mode    ',i3,es15.7)") jvar,evzero
    ! normalize bmode
    svar=sqrt(dot_product(bmode(:,jvar),bmode(:,jvar)))
    bmode(:,jvar)=bmode(:,jvar)/svar
    ! print for test
    tmpvec=matmul(hess,bmode(:,jvar))
    evbreath=dot_product(bmode(:,jvar),tmpvec)
    offdiag=dot_product(zeromode(:,jvar),tmpvec)
    write(*,"('Expectation value of br mode ',i3,es15.7)") jvar,evbreath
    write(*,"('br H zero                    ',i3,es15.7)") jvar,offdiag
    !write(*,"('scalar procuct rotate breathe',es15.7)") sum(bmode(:,jvar)*zeromode(:,jvar))
    !frac=sqrt(1.D0-evbreath/(evbreath-evzero))
    if(evzero/(evzero-evbreath) > 0.D0 ) then
      frac=sqrt(evzero/(evzero-evbreath))
      !print*,"frac 1",frac
      !print*,"diskriminate",(offdiag/evbreath)**2 - evzero/evbreath
      !frac=-offdiag/evbreath + sqrt( (offdiag/evbreath)**2 - evzero/evbreath)
      !print*,"frac 2",frac
      !print*,"supposed to be zero",evzero+2.D0*frac*offdiag+frac**2*evbreath
      if(frac>0.D0.and.frac<1.D0) then
        !frac=0.168D0
        !print*,"frac",frac
        zeromode(:,jvar)=sqrt(1.D0-frac**2)*zeromode(:,jvar)+frac*bmode(:,jvar)
        !! normalize bmode
        !svar=sqrt(dot_product(zeromode(:,jvar),zeromode(:,jvar)))
        !print*,"is normalization necessary?",svar
        !zeromode(:,jvar)=zeromode(:,jvar)/svar
        ! print for test
        tmpvec=matmul(hess,zeromode(:,jvar))
        evbreath=dot_product(zeromode(:,jvar),tmpvec)
        write(*,"('Expectation value of te mode ',i3,es15.7)") jvar,evbreath
      end if
    else
      write(*,"('No mixing of rotation and breathing mode possible for mode ',i2)"),jvar
    end if
  end do
  ! Observations: for an exact analytic hessian of a stationary point
   ! (vdW spheres), the projection works perfect. However, in this
  ! case, the rotational eigenvalues of the exact Hessian are also
  ! very close to zero but still distinct from the translational ones.

  ! z = zero vector, b=breathing vector, n=new most zero vector
  ! n=az+cb
  ! nHn=0
  ! (az+cb) H (az+cb) = 0
  ! a^2 zHz + 2ac zHb + c^1 bHb = 0
  ! a^2+c^2=1
  ! a^2 zHz + 2a sqrt(1-a^2) zHb + (1-a^2) bHb = 0
  ! cos^2 phi zHz + 2 sin(phi) cos(phi) zHb + sin^2 phi bHb = 0
  ! small phi: zHz + 2 phi zHb + phi^2 bHb = 0
  ! phi= -zHb/bHb +- sqrt( (zHb/bHb)^2 - 4*zHz*bHb )
  ! transzendent ...
  ! Annahme: xHb=0
  ! a^2 (zHz-bHb) = - bHb
  ! a= sqrt( bHb/(bHb-zHz) )
  ! c= sqrt(1- bHb/(bHb-zHz) ) = frac
  ! c= sqrt(-zHz/(bHb-zHz) ) = frac
  ! c= sqrt( zHz/(zHz-bHb) ) = frac

!!$  ! normalize rotational modes
!!$  do ivar=4,nzero
!!$    svar=sqrt(dot_product(zeromode(:,ivar),zeromode(:,ivar)))
!!$    !print*,"norm of mode",svar
!!$    zeromode(:,ivar)=zeromode(:,ivar)/svar
!!$    ! print for test
!!$    tmpvec=matmul(hess,zeromode(:,ivar))
!!$    svar=dot_product(zeromode(:,ivar),tmpvec)
!!$    write(*,"('Expectation value of mode',i3,es15.7)") ivar,svar
!!$    !print*,"Expectation value of mode",ivar,svar
!!$  end do

  ! now define the vector space of the remaining nvar-nzero modes
  ! randomize them and make them orthogonal to the zero modes
  call random_number(vibspace)
  vibspace=vibspace-0.5D0
  do ivar=1,nvar-nzero
    ! Gram-Schmidt orthogonalization
    do jvar=1,nzero
      svar=sum(zeromode(:,jvar)*vibspace(:,ivar))
      !print*,"overlap",svar
      vibspace(:,ivar)=vibspace(:,ivar)-svar*zeromode(:,jvar)
      !print*,"overlap after orthog",sum(zeromode(:,jvar)*vibspace(:,ivar))
    end do
    do jvar=1,ivar-1
      svar=sum(vibspace(:,jvar)*vibspace(:,ivar))
      !print*,"overlap",svar
      vibspace(:,ivar)=vibspace(:,ivar)-svar*vibspace(:,jvar)
      !print*,"overlap after orthog",sum(zeromode(:,jvar)*vibspace(:,ivar))
    end do
    ! normalize new vector
    svar=sum(vibspace(:,ivar)*vibspace(:,ivar))
    if(svar<1.D-10) call dlf_fail("Hessian projection not possible")
    vibspace(:,ivar)=vibspace(:,ivar)/sqrt(svar)
  end do

  vhess=matmul(transpose(vibspace),matmul(hess,vibspace))

  ! do diagonalization of vibrational hessian
  call dlf_matrix_diagonalise(nvar-nzero,vhess,veigval,veigvec)
  do ivar=1,10
    write(*,"('Eigenvalue no',i3,es15.7)") ivar+nzero,veigval(ivar)
  end do
  open(file='veigval',unit=10)
  do ivar=1,nvar-nzero
    write(10,"(i3,es15.7)") ivar+nzero,veigval(ivar)
  end do
  close(10)
  
  ! do full diagonalization for comparison
  call dlf_matrix_diagonalise(nvar,hess,eigval,eigvec)
  if (printl>=4) write(stdout,'("full hess")')
  sumsvar=0.D0
  do ivar=1,min(21,nvar)
    svar=0.D0
    do jvar=1,nzero
      svar=svar+sum(eigvec(:,ivar)*zeromode(:,jvar))**2
    end do
    sumsvar=sumsvar+svar
    if (printl>=4) write(*,"('Eigenvalue no',i3,es15.7,f10.5)") ivar,eigval(ivar),svar
  end do
  if (printl>=4) write(stdout,'("sum of projections", ES11.4)') sumsvar
  open(file='eigval',unit=10)
  do ivar=1,nvar
    if (printl>=4) write(10,"(i3,es15.7)") ivar,eigval(ivar)
  end do
  close(10)

!!$  ! try to reconstruct the Newton-raphson step
!!$  call dlf_matrix_invert(nvar-nzero,.false.,vhess,svar)
!!$  !if(printl>=4) write(stdout,"('Determinant of v-H in NR step: ',es10.3)") svar
!!$  hess=matmul(vibspace,matmul(vhess,transpose(vibspace)))

  !print*,"hessinv",hess(1:3,1)

  !call dlf_fail("JK stop")

end subroutine dlf_hessian_project
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_checkpoint_coords_write
!!
!! FUNCTION
!! Write checkpoint information
!!
!! SYNOPSIS
subroutine dlf_checkpoint_coords_write
!! SOURCE
  use dlf_global, only: glob,stderr
  implicit none
! **********************************************************************

  ! hdlc
  if(mod(glob%icoord,10)>=1 .and. mod(glob%icoord,10)<=4) then
    call dlf_checkpoint_hdlc_write
  end if

  ! NEB
  if(glob%icoord>=100 .and. glob%icoord<200) then
    call dlf_checkpoint_neb_write
  end if

  ! DIMER
  if(glob%icoord>=200 .and. glob%icoord<300) then
    call dlf_checkpoint_dimer_write
    ! make sure L-BFGS checkpoint is written if dimer is used but no
    ! global l-bfgs
    if(glob%iopt/=3) call DLF_checkpoint_LBFGS_write
  end if

end subroutine dlf_checkpoint_coords_write
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* coords/dlf_checkpoint_coords_read
!!
!! FUNCTION
!! Read checkpoint information
!!
!! SYNOPSIS
subroutine dlf_checkpoint_coords_read(tok)
!! SOURCE
  use dlf_global, only: glob,stderr
  implicit none
  logical,intent(out) :: tok
! **********************************************************************
  tok=.true.

  ! hdlc
  if(mod(glob%icoord,10)>=1 .and. mod(glob%icoord,10)<=4) then
    call dlf_checkpoint_hdlc_read(tok)
    if(.not.tok) return
  end if

  ! NEB
  if(glob%icoord>=100 .and. glob%icoord<200) then
    call dlf_checkpoint_neb_read(tok)
    if(.not.tok) return
  end if

  ! DIMER
  if(glob%icoord>=200 .and. glob%icoord<300) then
    call dlf_checkpoint_dimer_read(tok)
    if(.not.tok) return

    ! make sure L-BFGS checkpoint is written if dimer is used but no
    ! global l-bfgs
    if(glob%iopt/=3) then
      call DLF_checkpoint_LBFGS_read(tok)
    end if

  end if

end subroutine dlf_checkpoint_coords_read
!!****
