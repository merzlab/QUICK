! **********************************************************************
! **                      Improved Dimer method                       **
! **                 with L-BFGS rotation optimisation                **
! **                and L-BFGS translation optimisation               **
! **********************************************************************
!!****h* coords/dimer
!!
!! NAME
!! dimer
!!
!! FUNCTION
!! Apply the dimer method for transition state search
!!
!! COMMENTS
!! Cartesians of midpoint and tangent are stored
!!
!! icoords holds two images:
!! R_midpoint, Axis (= R_end-R_midpoint, length is dimer%delta)
!! igradient the corresponding gradient. 
!!
!! The dimer Distance is kept in i-space, the distance between 
!! R_midpoint and R_end is dimer%delta
!!
!! DATA
!!  $Date: 2006-12-13 15:55:10 +0000 (Wed, 13 Dec 2006) $
!!  $Rev: 29 $
!!  $Author: jk37 $
!!  $URL: http://ccpforge.cse.rl.ac.uk/svn/dl-find/trunk/dlf_dimer.f90 $
!!  $Id: dlf_dimer.f90 29 2006-12-13 15:55:10Z jk37 $
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
module dlf_dimer
  use dlf_parameter_module, only: rk
  type dimer_type
    integer               :: varperimage  ! no. of icoords per image
    integer               :: coreperimage ! no. of inner region icoords per image
    integer               :: status
    integer               :: mode !:
      ! 0: all rotations covered by the optimiser
      ! 1: rotations by linesearch here. No extrapolation of gradient
      ! 2: rotations by linesearch here. Extrapolation of gradient
    real(rk)              :: delta
    real(rk)              :: emid   ! energy of the dimer midpoint
    real(rk)              :: curve  ! curvature
    real(rk)              :: tolrot ! tolerance in rotation angle in order
                                    ! not to perform a new dimer rotation
    real(rk), allocatable :: xtangent(:,:) ! (3,nat) half dimer axis vector
    real(rk), allocatable :: xmidpoint(:,:) ! (3,nat)
    real(rk), allocatable :: rotgrad(:) ! (varperimage)
    real(rk), allocatable :: rotdir(:) ! (varperimage)
    real(rk), allocatable :: oldrotgrad(:) ! (varperimage)
    real(rk), allocatable :: oldrotdir(:) ! (varperimage)
    real(rk), allocatable :: vector(:) ! (varperimage)
    real(rk), allocatable :: theta(:) ! (varperimage)
    real(rk), allocatable :: grad1(:) ! (varperimage) gradient at 
                                      ! dimer end before rotation
    logical               :: toldrot
    logical               :: toptdir ! rotation optimised here rather than in formstep
    integer               :: cgstep
    logical               :: extrapolate_grad ! use extrapolated gradient
                             ! after rotation directly for next rotation?
    integer               :: nrot ! current number of rotations at fixed midpoint
    integer               :: maxrot ! max number of rotations at fixed midpoint
    real(rk)              :: phi,dcurvedphi
    ! test               
    logical               :: cdelta
  end type dimer_type
  type(dimer_type),save   :: dimer
  ! hard-coded at the moment:
  logical,parameter :: rot_lbfgs=.true.
end module dlf_dimer
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_init
!!
!! FUNCTION
!!
!! Initialise dimer calculations:
!! * Find the number of degrees of freedom
!! * (Initialise HDLCs)
!! * Allocate global and internal arrays
!! * Define the first dimer geometry
!! * Set default values for the calculation
!!
!! INPUTS
!!
!! glob%delta glob%tolrot glob%maxrot glob%tcoords2 glob%xcoords2
!! glob%nat / glob%nvar glob%spec glob%icoord glob%ncons glob%icons
!! glob%nconn glob%iconn glob%znuc glob%xcoords
!!
!! OUTPUTS
!! 
!! glob%nivar, glob%xcoords2 (if not set)
!! 
!! allocates: glob%icoords, glob%igradient, glob%step
!!
!! SYNOPSIS
subroutine dlf_dimer_init(icoord)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,pi
  use dlf_dimer
  use dlf_allocate, only: allocate,deallocate
  implicit none
  integer, intent(in)   :: icoord ! choice of dimer details 
  integer               :: ivar,iat
  real(rk)              :: svar
  real(RK) ,external    :: ddot
  logical               :: trandom,tplanar
  real(rk)              :: vnormal(3),v1(3),v2(3)
  real(rk) ,parameter   :: tolplanar=1.D-4
! **********************************************************************
  dimer%cdelta=.false. !<- this does not seem to make a difference

  ! set mode
  if( icoord<200 .or. icoord>299) call dlf_fail("Wrong icoord in Dimer")

  ! set parameters
  dimer%delta=glob%delta
  dimer%status=1
  dimer%nrot=0
  dimer%mode=mod(icoord/10,10)
  dimer%extrapolate_grad=(dimer%mode==2)
  dimer%tolrot=glob%tolrot/180.D0*pi
  dimer%maxrot=glob%maxrot

  ! check coords2
  trandom=.false.
  if(.not.glob%tcoords2) then
    ! if not provided by input, use random direction
    if(printl>=2) write(stdout,"('Dimer direction randomised')")
    glob%tcoords2=.true.
    if(glob%tatoms) then
      call allocate(glob%xcoords2,3,glob%nat,1)
    else
      call allocate(glob%xcoords2,1,glob%nvar,1)
    end if

    call random_number(glob%xcoords2)
    ! center coordinates
    glob%xcoords2=glob%xcoords2-0.5D0

    !check if system is planar
    if(glob%tatoms.and.glob%nat>=4) then
      v1=glob%xcoords(:,1)-glob%xcoords(:,2)
      v2=glob%xcoords(:,1)-glob%xcoords(:,3)
      vnormal(1)=v1(2)*v2(3)-v1(3)*v2(2)
      vnormal(2)=v1(3)*v2(1)-v1(1)*v2(3)
      vnormal(3)=v1(1)*v2(2)-v1(2)*v2(1)
      tplanar=.true.
      do iat=4,glob%nat
        if(abs(sum(vnormal*(glob%xcoords(:,1)-glob%xcoords(:,iat)))) &
            >= tolplanar) then
          tplanar=.false.
          exit
        end if
      end do
      if(tplanar)then
        svar=sqrt(sum(vnormal**2))
        vnormal=vnormal/svar
        if(printl>=2) write(stdout,*) "System planar, taking care of &
            &that when randomising dimer direction"
        do iat=1,glob%nat
          svar=sum(vnormal*glob%xcoords2(:,iat,1))
          glob%xcoords2(:,iat,1)=glob%xcoords2(:,iat,1)-svar*vnormal
        end do
        print*,"Random direction after planarising:"
        write(*,'(3f10.5)') glob%xcoords2
      end if
    end if

    if(dimer%mode==0) dimer%mode=1
    trandom=.true.
  end if

  if(dimer%mode >= 1) dimer%toptdir=.true.    

  ivar=mod(icoord,10)
  select case (ivar)
  ! Cartesian coordinates
  case (0,5)
    ! define the number of internal variables (covering all images)
    call dlf_direct_get_nivar(0, dimer%varperimage)
    call dlf_direct_get_nivar(1, dimer%coreperimage)

    ! calculate iweights
    call allocate( glob%iweight,dimer%varperimage*2)
    ! first image
    ivar=1
    do iat=1,glob%nat
      if(glob%spec(iat)>=0) then
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
    if(ivar-1/=dimer%varperimage) then
      call dlf_fail("Error in cartesian iweight calculation in dimer")
    end if
    ! copy weights to other image
    glob%iweight(dimer%varperimage+1:2*dimer%varperimage)=glob%iweight(1:dimer%varperimage)

  ! HDLC
  case(1:4)
    call dlf_hdlc_init(glob%nat,glob%spec,mod(glob%icoord,10),glob%ncons, &
        glob%icons,glob%nconn,glob%iconn)

    call dlf_hdlc_get_nivar(0, dimer%varperimage)
    call dlf_hdlc_get_nivar(1, dimer%coreperimage)

    ! create hdlc with midpoint as coordinates
    call dlf_hdlc_create(glob%nat,dimer%coreperimage,glob%spec,glob%micspec,glob%znuc,1, &
        glob%xcoords(:,:),glob%weight,glob%mass)

    ! calculate iweights
    call allocate( glob%iweight,dimer%varperimage*2)
    call dlf_hdlc_getweight(glob%nat,dimer%varperimage,dimer%coreperimage,glob%micspec,&
        glob%weight,glob%iweight(1:dimer%varperimage))
    ! copy weights to other image
    glob%iweight(dimer%varperimage+1:2*dimer%varperimage)=glob%iweight(1:dimer%varperimage)

  case default
    write(stderr,'("Coordinate type ",i3," not supported in Dimer")') icoord
    call dlf_fail("Wrong coordinate type in dimer")
  end select

  ! simple dimer method:
  glob%nivar = 2 * dimer%varperimage
  glob%nicore = 2 * dimer%coreperimage

  if(glob%tatoms) then
    call allocate( dimer%xtangent, 3, glob%nat)
    call allocate( dimer%xmidpoint, 3, glob%nat)
  else
    call allocate( dimer%xtangent, 1, glob%nvar)
    call allocate( dimer%xmidpoint, 1, glob%nvar)
  end if

  call allocate( glob%icoords,glob%nivar)
  call allocate( glob%igradient,glob%nivar)
  call allocate( glob%step,glob%nivar) 

  call allocate( dimer%rotgrad,dimer%varperimage)
  call allocate( dimer%rotdir,dimer%varperimage)
  call allocate( dimer%oldrotgrad,dimer%varperimage)
  call allocate( dimer%oldrotdir,dimer%varperimage)
  call allocate( dimer%vector,dimer%varperimage)
  call allocate( dimer%theta,dimer%varperimage)
  call allocate( dimer%grad1,dimer%varperimage)
  dimer%toldrot=.false.

  ! xcoords contains midpoint, xcoords2 contains endpoint 1, distance 
  !  not necessarily correct

  !Set icoords(1)
  glob%xgradient=0.D0

  call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%xcoords,glob%xgradient,glob%icoords(1:dimer%varperimage), & 
      glob%igradient(1:dimer%varperimage))
  dimer%xmidpoint(:,:)=glob%xcoords(:,:)

  ! Guess xcoords2 at distance delta from xcoords
  if(glob%tsrelative.or.trandom) then
    dimer%xtangent(:,:)=glob%xcoords2(:,:,1)
  else
    dimer%xtangent(:,:)=glob%xcoords2(:,:,1)-glob%xcoords(:,:)
  end if
  svar=dsqrt(ddot(glob%nvar,dimer%xtangent,1,dimer%xtangent,1))
  dimer%xtangent(:,:)=dimer%xtangent(:,:) / svar * dimer%delta
  glob%xcoords(:,:)=glob%xcoords(:,:) + dimer%xtangent(:,:)

  ! set icoords2 to the guessed endpoint
  call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%xcoords,glob%xgradient,glob%icoords(dimer%varperimage+1:), &
      glob%igradient(1:dimer%varperimage))

  ! set xcoords back for first step
  glob%xcoords(:,:)=glob%xcoords(:,:) - dimer%xtangent(:,:)

  ! set icoords2 to R_end-R_midpoint with the proper length
  glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:)- &
      glob%icoords(1:dimer%varperimage)

  ! weight icoords2
  glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:) * &
      glob%iweight(1:dimer%varperimage)

  ! remove translation and rotation from dimer axis
  call dlf_coords_tranrot(dimer%varperimage,glob%icoords(dimer%varperimage+1:))
  svar=ddot(dimer%varperimage,glob%icoords(dimer%varperimage+1:),1, &
      glob%icoords(dimer%varperimage+1:),1)
  if(svar>1.D-15) then
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:) &
        / dsqrt(svar) * dimer%delta
  else
    call dlf_fail("Input dimer distance too small!")
  end if

  ! rubbish for printing
  if(.not.glob%tatoms) then
    if (glob%iam == 0) then
       open(unit=40,file="axis.inc")
       open(unit=41,file="dimer.xy",position="append")
    end if
  end if

  if(rot_lbfgs) then
    call dlf_lbfgs_select("dimer rotation",.true.)
    call dlf_lbfgs_init(dimer%varperimage,min(dimer%maxrot,dimer%varperimage)) 
    call dlf_lbfgs_deselect
  end if

end subroutine dlf_dimer_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_destroy
!!
!! FUNCTION
!! Destroy dimer arrays and close files
!!
!! SYNOPSIS
subroutine dlf_dimer_destroy
!! SOURCE
  ! deallocate arrays concerning internal coordinates
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_dimer
  use dlf_allocate, only: deallocate
  implicit none
  logical :: exists
! **********************************************************************
  ! deallocate arrays
  if (allocated(glob%icoords)) call deallocate( glob%icoords)
  if (allocated(glob%igradient)) call deallocate( glob%igradient)
  if (allocated(glob%step)) call deallocate( glob%step)
  if (allocated(glob%iweight)) call deallocate( glob%iweight)

  if (allocated(dimer%rotgrad)) call deallocate( dimer%rotgrad)
  if (allocated(dimer%rotdir)) call deallocate( dimer%rotdir)
  if (allocated(dimer%oldrotgrad)) call deallocate( dimer%oldrotgrad)
  if (allocated(dimer%oldrotdir)) call deallocate( dimer%oldrotdir)
  if (allocated(dimer%vector)) call deallocate( dimer%vector)
  if (allocated(dimer%theta)) call deallocate( dimer%theta)
  if (allocated(dimer%grad1)) call deallocate( dimer%grad1)

  if(rot_lbfgs) then
    ! dimer%oldrotdir, dimer%oldrotgrad are not needed!
    call dlf_lbfgs_exists("dimer rotation", exists)
    if (exists) then
       call dlf_lbfgs_select("dimer rotation",.false.)
       call dlf_lbfgs_destroy
       call dlf_lbfgs_deselect
    end if
  end if

  select case (mod(glob%icoord,10))
  ! HDLC
  case(1:4)
    call dlf_hdlc_destroy
  end select

  if (allocated(dimer%xtangent)) call deallocate( dimer%xtangent)
  if (allocated(dimer%xmidpoint)) call deallocate( dimer%xmidpoint)

  ! rubbish for printing
  if(.not.glob%tatoms) then
    if (glob%iam == 0) then
       close(40)
       close(41)
    end if
  end if

end subroutine dlf_dimer_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_xtoi
!!
!! FUNCTION
!! * Convert individual energy/gradients into the dimer force expression
!! * Estimate a dimer rotation angle
!! * Optimise the dimer rotation eventually
!!
!! SYNOPSIS
subroutine dlf_dimer_xtoi(trerun_energy,testconv)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,pi
  use dlf_stat, only: stat
  use dlf_dimer
  use dlf_allocate, only: allocate,deallocate
  implicit none
  logical, intent(out)  :: trerun_energy
  logical, intent(out)  :: testconv ! is convergence tested here?
  real(rk)              :: svar
  logical               :: stoprot
  real(rk),external     :: ddot
  real(rk)              :: phi1
  logical               :: exists
! **********************************************************************

  ! If in a microiterative loop, just xtoi the midpoint
  if (glob%imicroiter == 2) then
     call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
          glob%xcoords,glob%xgradient,glob%icoords(1:dimer%varperimage), &
          glob%igradient(1:dimer%varperimage))
     return
  end if

  ! ====================================================================
  ! Optimise the rotation
  ! ====================================================================

  ! this routine has to be called several times from the main cycle 
  !  (after each E&G evaluation). Thus we have to use dimer%status to 
  !  keep track.
  
  stoprot=.false.
  trerun_energy=.true.

  if(dimer%status == 1) then
    ! check overall convergence, set xcoords to x1
    call dlf_dimer_was_midpoint(trerun_energy,testconv)
    if(.not.trerun_energy) return ! converged
    ! now calculate endpoint: g1
    ! dimer%status = 2 is set in dlf_dimer_was_midpoint
  else if (dimer%status == 2) then
    ! calculate theta,
    ! estimate rotation angle: phi1
    ! check convergence of rotation (based on phi1),
    ! rotate the dimer to x1'
    call dlf_dimer_was_g1(.false.,stoprot)
    ! now calculate g1'
    dimer%status = 3
  else if (dimer%status == 3) then
    ! calculate optimal rotation angle phi_opt,
    ! check convergence, 
    ! set dimer direction according to phi_opt
    call dlf_dimer_was_g1prime(stoprot,phi1)
    if(dimer%extrapolate_grad.and..not.stoprot) then
      ! extrapolate the gradient at g1
      !call dlf_dimer_extrapolate_gradient(phi1)
      call dlf_dimer_extrapolate_gradient(dimer%varperimage,phi1, &
          dimer%phi,glob%igradient(1:dimer%varperimage),dimer%grad1,&
          glob%igradient(dimer%varperimage+1:))

      call dlf_dimer_was_g1(.true.,stoprot)
      ! now calculate g1'
      dimer%status = 3
    else
      ! now calculate endpoint: g1
      dimer%status = 2
    end if
  else
    call dlf_fail("Wrong dimer%status")
  end if

  ! ====================================================================
  ! If the rotations are converged, transform the gradient
  ! ====================================================================
  if(stoprot) then
    ! rotation has converged, transform gradient and return
    if(printl>=2) then
      if(abs(dimer%phi) <= dimer%tolrot) then
        write(stdout,"('Dimer rotation converged after ',i4,' steps')") &
            dimer%nrot
      else
        write(stdout,"('Dimer rotation not converged after ',i4,' steps')") &
            dimer%nrot
        write(stdout,"('Dimer rotation terminated, maximum number of&
            & rotations reached')")
      end if
    end if
    ! dimer is not rotated any more, send tsmode to set_tsmode
    ! TS mode in internals
    call dlf_formstep_set_tsmode(dimer%varperimage,11,dimer%vector) 
    
    if(printl>=2) write(stdout,'("Curvature after dimer rotation:     &
        &      ",f12.5)') &
        dimer%curve
    
    ! print frequency in case of mass-weighted coordinates
    if(glob%massweight .and. printl>=2) then

      call dlf_print_wavenumber(dimer%curve,.true.)

!!$      ! sqrt(H/u)/a_B/2/pi/c / 100
!!$      svar=sqrt( 4.35974417D-18/ 1.66053886D-27 ) / ( 2.D0 * pi * &
!!$          0.5291772108D-10 * 299792458.D0) / 100.D0
!!$      svar=sqrt(abs(dimer%curve)) * svar
!!$      if(dimer%curve<0.D0) svar=-svar
!!$      write(stdout,"('Frequency of transition mode',f10.3,' cm^-1 &
!!$          &(negative value denotes imaginary frequency)')") &
!!$          svar
    end if
    
    
    ! transform gradient:
    if(dimer%toptdir) then
      ! set rotational force to zero if rotation is optimised here 
      ! rather than by formstep
      glob%igradient(dimer%varperimage+1:)=0.d0
    else
      glob%igradient(dimer%varperimage+1:)=dimer%rotgrad*1.0D0 
    end if
    
    ! translational gradient:
    svar=sum(dimer%vector(:) * glob%igradient(1:dimer%varperimage))
    if(dimer%curve > 0.D0) then
      glob%igradient(1:dimer%varperimage)=-svar*dimer%vector(:)
    else
      glob%igradient(1:dimer%varperimage)=glob%igradient(1:dimer%varperimage) - &
          2.D0 * svar * dimer%vector(:)
    end if

    dimer%nrot=0
    dimer%toldrot=.false. ! no CG for next rotation
    if(rot_lbfgs) then
      call dlf_lbfgs_select("dimer rotation",.false.)
      call dlf_lbfgs_restart
      dimer%rotdir(:)=0.D0
      call dlf_lbfgs_deselect
    end if
    dimer%status=1
    trerun_energy=.false.
    testconv=.true.

  end if !(stoprot)

end subroutine dlf_dimer_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_was_midpoint
!!
!! FUNCTION
!! * Called after the dimer midpoint was calculated
!! * Check overall convergence of the dimer optimisation
!! * Set xcoords to dimer endpoint before rotation (x1)
!!
!! SYNOPSIS
subroutine dlf_dimer_was_midpoint(trerun_energy,testconv)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use dlf_stat, only: stat
  use dlf_dimer
  use dlf_allocate, only: allocate,deallocate
  implicit none
  logical, intent(out)  :: trerun_energy
  logical, intent(out)  :: testconv ! is convergence tested here?
  real(rk)              :: dlength,svar
  logical               :: tok,tconv
  real(rk),external     :: ddot
! **********************************************************************
  testconv=.true.
  ! ==================================================================
  ! This was the midpoint calculation, xcoords contains midpoint
  ! ==================================================================

  ! send information to set_tsmode
  call dlf_formstep_set_tsmode(1,-1,glob%energy) ! send energy
  call dlf_formstep_set_tsmode(glob%nvar,0,glob%xcoords) ! TS-geometry

  call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%xcoords,glob%xgradient,glob%icoords(1:dimer%varperimage), &
      glob%igradient(1:dimer%varperimage))

  ! set xcoords to endpoint 1:

  !Set icoords(1) to endpoint1:
  if(.not.dimer%cdelta) then
    dlength=dsqrt(ddot(dimer%varperimage,glob%icoords(dimer%varperimage+1:),1, &
        glob%icoords(dimer%varperimage+1:),1))
    !print*,"Dimer length scaling",dlength/dimer%delta
    glob%icoords(1:dimer%varperimage)=glob%icoords(1:dimer%varperimage)+ &
        glob%icoords(dimer%varperimage+1:)*dimer%delta/dlength
  else
    glob%icoords(1:dimer%varperimage)=glob%icoords(1:dimer%varperimage)+ &
        glob%icoords(dimer%varperimage+1:)
  end if

  !Transform to x-coords
  call dlf_direct_itox(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%icoords(1:dimer%varperimage),glob%xcoords(:,:),tok)
  if(.not.tok) then

    if(printl>=4) write(stdout, &
        "('HDLC coordinate breakdown at dimer endpoint. Recalculating HDLCs and &
        &restarting optimiser and dimer.')")
    call dlf_hdlc_reset
    call dlf_hdlc_create(glob%nat,dimer%coreperimage,glob%spec,glob%micspec,&
        glob%znuc,1,glob%xcoords,glob%weight,glob%mass)
    ! recalculate iweights
    call dlf_hdlc_getweight(glob%nat,dimer%varperimage,dimer%coreperimage,glob%micspec,&
        glob%weight,glob%iweight(1:dimer%varperimage))
    ! copy weights to other image
    glob%iweight(dimer%varperimage+1:2*dimer%varperimage)= &
        glob%iweight(1:dimer%varperimage)
    
    call dlf_formstep_restart
    ! calculate internal coordinates of midpoint
    call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
        glob%xcoords,glob%xgradient,glob%icoords(1:dimer%varperimage), &
        glob%igradient(1:dimer%varperimage))
    !calculate internal coordinates of endpoint 1
    glob%xcoords=glob%xcoords+dimer%xtangent
    call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
        glob%xcoords,glob%xgradient,glob%icoords(dimer%varperimage+1:), &
        glob%igradient(dimer%varperimage+1:))
    ! now first half of internal gradient is correct, second half is
    ! not, but will be recalculated anyway.
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:) -&
        glob%icoords(1:dimer%varperimage)
    ! normalise (they are not necessarily normal in the new HDLCs)
    svar=ddot(dimer%varperimage,glob%icoords(dimer%varperimage+1:),1, &
        glob%icoords(dimer%varperimage+1:),1)
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:) &
        / dsqrt(svar) * dimer%delta
    ! set icoords(1) to endpoint 1, will be reset to midpoint below
    glob%icoords(1:dimer%varperimage)=glob%icoords(1:dimer%varperimage)+ &
        glob%icoords(dimer%varperimage+1:)
    
  else
    ! store cartesian dimer vector
    dimer%xtangent=glob%xcoords-dimer%xmidpoint
  end if
  !Set icoords(1) back to midpoint:
  if(.not.dimer%cdelta) then
    glob%icoords(1:dimer%varperimage)=glob%icoords(1:dimer%varperimage)- &
        glob%icoords(dimer%varperimage+1:)*dimer%delta/dlength
  else
    glob%icoords(1:dimer%varperimage)=glob%icoords(1:dimer%varperimage)- &
        glob%icoords(dimer%varperimage+1:)
  end if
  
  dimer%emid=glob%energy

  ! ==================================================================
  ! Test convergence
  ! ==================================================================
  call convergence_set_info("of dimer midpoint",dimer%varperimage,&
      dimer%emid,glob%igradient(1:dimer%varperimage),glob%step(1:dimer%varperimage))
  ! test convergence
  call convergence_test(stat%ccycle,.true.,tconv)
  trerun_energy=.true.
  if(tconv) then
    ! send again, so that the same valued will be tested
    call convergence_set_info("of dimer midpoint",dimer%varperimage,&
        dimer%emid,glob%igradient(1:dimer%varperimage),glob%step(1:dimer%varperimage))
    ! If converged, make sure that the latest TSmode is also set!
    call dlf_formstep_set_tsmode(dimer%varperimage,11,glob%icoords(dimer%varperimage+1:))
    ! TS mode in internals
    testconv=.false. ! should be tested again in the main cycle, which causes it to stop
    trerun_energy=.false.
    dimer%status=1
    return
  end if

  if(printl>=4) write(stdout,"('Next calculation will be the dimer &
      &endpoint before rotation')")
  trerun_energy=.true.
  dimer%status=2
end subroutine dlf_dimer_was_midpoint
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_was_g1
!!
!! FUNCTION
!! * Called after the dimer endpoint before rotation was calculated
!! * calculate theta (rotation direction)
!! * estimate rotation angle: phi1
!! * check convergence of rotation (based on phi1)
!! * rotate the dimer to x1' (xcoords) if rotation continues
!!
!! SYNOPSIS
subroutine dlf_dimer_was_g1(internal,stoprot)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,pi
  use dlf_stat, only: stat
  use dlf_dimer
  use dlf_allocate, only: allocate,deallocate
  implicit none
  logical,   intent(in)  :: internal ! internal gradient is already known
  logical,   intent(out) :: stoprot
  real(rk)               :: svar,gamma
  real(rk),  external    :: ddot
  logical                :: tok
! **********************************************************************

  ! ==================================================================
  ! This was one endpoint calculation, xcoords contains endpoint 1
  ! ==================================================================
  
  if(.not.internal) then
    call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
        glob%xcoords,glob%xgradient,glob%icoords(dimer%varperimage+1:), &
        glob%igradient(dimer%varperimage+1:))
    ! transform coords:
    dimer%vector(:)=glob%icoords(dimer%varperimage+1:)-glob%icoords(1:dimer%varperimage)
    glob%icoords(dimer%varperimage+1:)=dimer%vector(:)
    
    ! normalise dimer%vector:
    svar=dsqrt(ddot(dimer%varperimage,dimer%vector,1,dimer%vector,1))
    if(dimer%cdelta.and.abs(svar-dimer%delta) > 1.D-6 ) then
      print*,"Error: dimer distance wrong:"
      print*,"Distance after step:",svar
      print*,"Required distance:",dimer%delta
      call dlf_fail("Wrong dimer distance!")
    end if
    dimer%vector=dimer%vector/svar
  end if

  ! calculate curvature
  dimer%curve=(ddot(dimer%varperimage,glob%igradient(dimer%varperimage+1:),1,&
      dimer%vector,1)-ddot(dimer%varperimage,glob%igradient(1:dimer%varperimage),1,&
      dimer%vector,1))/dimer%delta
  if(printl>=2.and..not.internal) then
    write(stdout,"('Curvature at dimer midpoint               ',f12.5)") dimer%curve
  end if

  ! direction for rotation:
  dimer%rotgrad(:)=glob%igradient(dimer%varperimage+1:)- &
      glob%igradient(1:dimer%varperimage)
  ! weight direction for rotation
  dimer%rotgrad(:)=dimer%rotgrad(:) * glob%iweight(1:dimer%varperimage)
  ! orthogonalise to vector
  svar=ddot(dimer%varperimage,dimer%vector,1, dimer%rotgrad,1)
  dimer%rotgrad(:)=dimer%rotgrad(:) - svar * dimer%vector(:)

  if(rot_lbfgs) then
    call dlf_lbfgs_select("dimer rotation",.false.)
    call dlf_lbfgs_step(dimer%vector,dimer%rotgrad,dimer%rotdir)
    call dlf_lbfgs_deselect
  else
    ! CG rotation
    ! find rotation direction by conjugate gradient
    if((.not.dimer%toldrot).or.mod(dimer%cgstep,10)==0) then
      dimer%cgstep=1
      gamma =0.D0
      dimer%oldrotdir(:)=0.D0
    else
      gamma=ddot(dimer%varperimage, dimer%rotgrad-dimer%oldrotgrad,1 &
          ,dimer%rotgrad,1) / &
          ddot(dimer%varperimage,dimer%oldrotgrad,1,dimer%oldrotgrad,1)
      dimer%cgstep=dimer%cgstep+1
    end if
    gamma=max(gamma,0.d0)
    dimer%rotdir(:)= -dimer%rotgrad(:) + gamma * dimer%oldrotdir(:)
  end if

  dimer%oldrotgrad(:) = dimer%rotgrad(:)
  dimer%toldrot=.true. ! used: CG, away: SD

  dimer%theta(:)=dimer%rotdir(:)
  ! orthogonalise to dimer%vector
  svar=ddot(dimer%varperimage,dimer%vector,1,dimer%theta,1)
  dimer%theta(:)=dimer%theta(:) - svar * dimer%vector(:)
  ! normalise
  svar=dsqrt(sum(dimer%theta**2))
  dimer%theta(:)=dimer%theta(:)/svar
    
  ! estimate the rotation angle:
  dimer%dcurvedphi=2.D0 * sum((glob%igradient(dimer%varperimage+1:) &
      -glob%igradient(1:dimer%varperimage)) &
      * dimer%theta(:))/dimer%delta
  
  ! estimate the angle from curve and dcurvedphi. c0 and ca are ignored
  ! at the moment
  !call guess_phimin(dimer%curve,dimer%dcurvedphi,dimer%c0,dimer%ca,dimer%phi)
  dimer%phi=abs(0.5D0*atan(1.d0/( 2.d0*dimer%curve/dimer%dcurvedphi)))

  if(printl>=4) then
    write(stdout,"('Predicted dimer rotation angle:    ',f7.3,' deg')") &
        dimer%phi*180.D0/pi
  end if

  stoprot=.true.

  if(dimer%toptdir) then
    ! rotate dimer:
    if(abs(dimer%phi) > dimer%tolrot .and. dimer%nrot < dimer%maxrot) then 
      glob%icoords(dimer%varperimage+1:)=glob%icoords(1:dimer%varperimage) + &
          cos(dimer%phi) * dimer%delta * dimer%vector(:) + &
          sin(dimer%phi) * dimer%delta * dimer%theta(:)
      !Transform to x-coords
      call dlf_direct_itox(glob%nvar,dimer%varperimage,dimer%coreperimage, &
          glob%icoords(dimer%varperimage+1:),glob%xcoords(:,:),tok)
      if(.not.tok) call dlf_fail("Dimer internal->x conversion failed C")

      ! store cartesian dimer vector
      dimer%xtangent=glob%xcoords-dimer%xmidpoint

      dimer%status=3
      if(printl>=4) write(stdout,"('Next calculation will be the dimer &
          &endpoint after rotation')")

      ! rotation not converged
      stoprot=.false.

    ! if rotation finished, icoords(dimer%varperimage+1:) are relative coordinates
    ! of the non-rotated dimer.
    end if ! abs(dimer%phi) > dimer%tolrot

  end if ! dimer%toptdir

end subroutine dlf_dimer_was_g1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_was_g1prime
!!
!! FUNCTION
!! * Called after the dimer endpoint after rotation was calculated
!! * calculate optimal rotation angle phi_opt
!! * check convergence based on phi_opt
!! * set dimer direction according to phi_opt (xcoords)
!!
!! SYNOPSIS
subroutine dlf_dimer_was_g1prime(stoprot,phi1)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,pi
  use dlf_stat, only: stat
  use dlf_dimer
  use dlf_allocate, only: allocate,deallocate
  implicit none
  logical,   intent(out) :: stoprot
  real(rk),  intent(out) :: phi1
  real(rk),  allocatable :: vector2(:)
  real(rk)               :: curve2,a1,a0,b1
  logical                :: tok
  real(rk), external     :: ddot
! **********************************************************************

  ! ==================================================================
  ! This was the second endpoint calculation
  ! ==================================================================

  ! Dimer has already been rotated once, now determine optimal rotational position.
  ! Vector and theta are allocated and point to the old dimer direction and
  ! old rotational direction.
  ! xcoords contains endpoint 1 after rotation.

  dimer%nrot=dimer%nrot+1

  ! store endpoint gradient before rotation
  dimer%grad1=glob%igradient(dimer%varperimage+1:)

  call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%xcoords,glob%xgradient,glob%icoords(dimer%varperimage+1:), &
      glob%igradient(dimer%varperimage+1:))

  call allocate(vector2,dimer%varperimage)
  vector2=(cos(dimer%phi) * dimer%vector(:) + sin(dimer%phi) * dimer%theta(:))

  ! calculate curvature
  curve2=(ddot(dimer%varperimage,glob%igradient(dimer%varperimage+1:),1,&
      vector2,1)-ddot(dimer%varperimage,glob%igradient(1:dimer%varperimage),1,&
      vector2,1))/dimer%delta
  
  if(printl>=4) then
    write(stdout,"('Curvature at dimer midpoint after rotation',f12.5)") curve2
  end if

  ! we use the first version of Eq 30 in hey05 for a1:
  a1= ( dimer%curve - curve2 + 0.5D0 * dimer%dcurvedphi * &
      sin(2.D0*dimer%phi) ) / ( 1.D0 - cos(2.D0 * dimer%phi) )
  b1= 0.5D0 * dimer%dcurvedphi
  a0= 2.d0 * ( dimer%curve - a1 )
  
  phi1=dimer%phi ! angle of first rotation
  dimer%phi=0.5d0 * datan(b1/a1) ! phi_opt, rotational minimum

  ! Make sure the angle is a minimum, not a maximum
  curve2=0.5D0 * a0 + a1 * cos(2.D0 * dimer%phi) + b1 * sin(2.D0 * dimer%phi)
  if(printl>=6) then
    write(stdout,"('Expected curvature at dimer minimum       ',f12.5)") curve2
  end if
  if( curve2 > dimer%curve ) then
    if(printl>=6) print*,"Obtained angle seems to be a maximum: ",curve2
    dimer%phi=dimer%phi-0.5d0 * pi
    curve2=0.5D0 * a0 + a1 * cos(2.D0 * dimer%phi) + b1 * sin(2.D0 * dimer%phi)
    if(printl>=6) print*,"New curvature, now hopefully a minimum",curve2
  end if
  if(abs(dimer%phi-phi1) > pi*0.5D0) then
    if(dimer%phi<phi1) then
      dimer%phi=dimer%phi+pi
    else
      dimer%phi=dimer%phi-pi
    end if
  end if
  
  if(printl>=4) then
    write(stdout,"('Curvature at dimer minimum                ',f12.5)") curve2
    write(stdout,"('Actual dimer rotation angle:       ',f7.3,' deg')") &
        dimer%phi*180.D0/pi
  end if

  ! if abs(dimer%phi)< dimer%tolrot, do this rotation but exit afterwards!
  stoprot=(abs(dimer%phi) < dimer%tolrot)

  glob%icoords(dimer%varperimage+1:)= &
      cos(dimer%phi) * dimer%delta * dimer%vector(:) + &
      sin(dimer%phi) * dimer%delta * dimer%theta(:)
  
  vector2= (cos(dimer%phi) * dimer%vector(:) + sin(dimer%phi) * dimer%theta(:))

  dimer%vector=vector2
  dimer%curve=curve2

  ! Set old rotation direction (for CG) to the normal vector. This was shown to be 
  !  better than the rotated one.
  dimer%oldrotdir(:)  = dimer%theta(:) * &
      sqrt(ddot(dimer%varperimage,dimer%rotdir(:),1,dimer%rotdir(:),1))

  glob%icoords(dimer%varperimage+1:)=glob%icoords(1:dimer%varperimage) + &
      dimer%delta * vector2(:)
  !Transform to x-coords
  call dlf_direct_itox(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%icoords(dimer%varperimage+1:),glob%xcoords(:,:),tok)
  if(.not.tok) call dlf_fail("Dimer internal->x conversion failed D")

  ! store cartesian dimer vector
  dimer%xtangent=glob%xcoords-dimer%xmidpoint

  ! set icoords back to relative rotated coords
  glob%icoords(dimer%varperimage+1:)= dimer%delta * vector2(:)

  if(.not.dimer%extrapolate_grad.and..not.stoprot) then
    if(printl>=4) write(stdout,"('Next calculation will be the dimer &
        &endpoint before rotation')")
  end if
  call deallocate(vector2)

end subroutine dlf_dimer_was_g1prime
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_extrapolate_gradient
!!
!! FUNCTION
!! * Extrapolate the i-gradient at the new dimer endpoint g1
!!
!! SYNOPSIS
subroutine dlf_dimer_extrapolate_gradient(nvar,phi1,phim,grad0,grad1,grad1prime)
!! SOURCE
  ! linear in r between r_0, r_1, and r_1'
  use dlf_parameter_module, only: rk
  use dlf_global, only: pi
  implicit none
  integer  ,intent(in)    :: nvar
  real(rk) ,intent(in)    :: phi1
  real(rk) ,intent(in)    :: phim
  real(rk) ,intent(in)    :: grad0(nvar)
  real(rk) ,intent(in)    :: grad1(nvar)
  real(rk) ,intent(inout) :: grad1prime(nvar)
  real(rk)                :: par1,par1prime,par0
  ! ******************************************************************
  par1=dsin(phi1-phim)/dsin(phi1)
  par1prime=dsin(phim)/dsin(phi1)
  par0=1.D0-dcos(phim)-dsin(phim)*dtan(0.5D0*phi1)
  grad1prime(:) = par1 * grad1(:) + par1prime * grad1prime(:) &
      + par0 * grad0(:)
end subroutine dlf_dimer_extrapolate_gradient
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_checkstep
!!
!! FUNCTION
!! If rotation is optimised in itox, just set rotational gradient to zero
!!
!! If rotation is optimised by the optimiser, scale rotational step,
!! the adjust the step to preserve the dimer direction.
!!
!! SYNOPSIS
subroutine dlf_dimer_checkstep
!! SOURCE
  ! Make sure the distance between the images is constant 
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,pi,printl
  use dlf_dimer, only: dimer
  implicit none
  real(rk)              :: n0s, ss, svar
  real(RK) ,external    :: ddot
! **********************************************************************

  ! Versions for determining the step in the dimer direction
  if(dimer%toptdir) glob%step(dimer%varperimage+1:)=0.D0

  ! scale rotational step back ...
  ! Scaled L-BFGS rotation
  glob%step(dimer%varperimage+1:)=glob%step(dimer%varperimage+1:)*0.10D0  

  if(dimer%cdelta) then
    ! we have to scale the N-step back
    glob%step(dimer%varperimage+1:) = glob%step(dimer%varperimage+1:) + &
        glob%icoords(dimer%varperimage+1:)
    ss= ddot(dimer%varperimage,glob%step(dimer%varperimage+1:), 1, &
        glob%step(dimer%varperimage+1:), 1)

    write(*,'("Distance scaled from",F10.5," to",F10.5)') SQRT(ss),dimer%delta
    glob%step(dimer%varperimage+1:) = glob%step(dimer%varperimage+1:) &
        / dsqrt(ss) * dimer%delta
    ! make sure N does not change by more than 90 deg (as then -step is the relevant step)
    if(ddot(dimer%varperimage,glob%step(dimer%varperimage+1:), 1, &
        glob%icoords(dimer%varperimage+1:),1) < 0.D0 ) &
        glob%step(dimer%varperimage+1:)= - glob%step(dimer%varperimage+1:)
    glob%step(dimer%varperimage+1:) = glob%step(dimer%varperimage+1:) - &
        glob%icoords(dimer%varperimage+1:)

  end if
  ! rubbish for printing >>>>>>
  if(dimer%curve>0.D0) then
    svar=5.D-2/dimer%delta
  else
    svar=5.D-2/dimer%delta
  end if
  n0s=(glob%energy-dimer%emid)*svar
  if(.not.glob%tatoms) then
    if (glob%iam == 0) then
       write(40,"('cylinder{<',f12.4,',',f12.4,',',f12.4,'>,"//&
           &"<',f12.4,',',f12.4,',',f12.4,'> 0.005} ')") &
           glob%icoords(1)-glob%icoords(3)*svar,min(dimer%emid,1.D3)-n0s,&
           glob%icoords(2)-glob%icoords(4)*svar, &
           glob%icoords(1)+glob%icoords(3)*svar, min(dimer%emid,1.D3)+n0s,&
           glob%icoords(2)+glob%icoords(4)*svar
!!$       write(40,"('sphere{<',f12.4,',',f12.4,',',f12.4,'> 0.02} ')") &
!!$           glob%icoords(1)-glob%icoords(3)*svar,min(dimer%emid,1.D3)-n0s,&
!!$           glob%icoords(2)-glob%icoords(4)*svar
!!$       write(40,"('sphere{<',f12.4,',',f12.4,',',f12.4,'> 0.02} ')") &
!!$           glob%icoords(1)+glob%icoords(3)*svar,min(dimer%emid,1.D3)+n0s,&
!!$           glob%icoords(2)+glob%icoords(4)*svar
!!$       write(40,"('sphere{<',f12.4,',',f12.4,',',f12.4,'> 0.021} ')") &
!!$           glob%icoords(1),min(dimer%emid,1.D3),&
!!$           glob%icoords(2)
       write(41,*) glob%icoords(1:2)
    end if
  end if
  ! <<<< end of rubbish for printing

  ! calculate the angle between the new normal and the old normal:
  ! Maximum change is 90 deg, as only the direction, not the orientation is relevant
  if((.not.dimer%toptdir) .and. printl>=2) then
    n0s= ddot(dimer%varperimage,glob%icoords(dimer%varperimage+1:), 1, &
        glob%step(dimer%varperimage+1:), 1)
    svar=1.D0+n0s/dimer%delta**2
    if(svar>=1.D0) then
      svar=0.D0
    else if (svar<=-1.D0) then
      svar=180.D0
    else
      svar=acos(svar)*180.D0/pi
    end if
    write(stdout,"('Dimer axis rotated by ',f5.1,' degrees')") svar
  end if

  if(dimer%cdelta) then
    ! check:
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:)+ &
        glob%step(dimer%varperimage+1:)
    ss= sqrt(ddot(dimer%varperimage,glob%icoords(dimer%varperimage+1:), 1, &
        glob%icoords(dimer%varperimage+1:), 1))
    if(abs(ss-dimer%delta) .gt. 1.D-6) then
      print*,"Error: dimer distance wrong:"
      print*,"Distance after step:",ss
      print*,"Required distance:",dimer%delta
      call dlf_fail("Wrong dimer distance")
    end if
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:)- &
        glob%step(dimer%varperimage+1:)
  end if
end subroutine dlf_dimer_checkstep
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_dimer_itox
!!
!! FUNCTION
!! Set xcoords to dimer midpoint
!!
!! SYNOPSIS
subroutine dlf_dimer_itox
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_dimer, only: dimer,rot_lbfgs
  implicit none
  logical               :: tok
  real(rk)              :: svar
  real(RK) ,external    :: ddot
! **********************************************************************
  call dlf_direct_itox(glob%nvar,dimer%varperimage,dimer%coreperimage, &
      glob%icoords(1:dimer%varperimage),glob%xcoords(:,:),tok)
  if(.not.tok) then
    ! we have to cancel that step ...
    if(printl>=4) write(stdout, &
        "('HDLC coordinate breakdown at dimer midpoint. Recalculating HDLCs and &
        &restarting optimiser and dimer.')")
    call dlf_hdlc_reset
    glob%xcoords=dimer%xmidpoint
    call dlf_hdlc_create(glob%nat,dimer%coreperimage,glob%spec,glob%micspec,&
        glob%znuc,1,glob%xcoords,glob%weight,glob%mass)

    ! recalculate iweights
    call dlf_hdlc_getweight(glob%nat,dimer%varperimage,dimer%coreperimage,glob%micspec,&
        glob%weight,glob%iweight(1:dimer%varperimage))
    ! copy weights to other image
    glob%iweight(dimer%varperimage+1:2*dimer%varperimage)= &
        glob%iweight(1:dimer%varperimage)

    call dlf_formstep_restart
    ! reset internal rotation optimisation 
    dimer%nrot=0
    dimer%toldrot=.false. ! no CG for next rotation
    if(rot_lbfgs) then
      call dlf_lbfgs_select("dimer rotation",.false.)
      call dlf_lbfgs_restart
      dimer%rotdir(:)=0.D0
      call dlf_lbfgs_deselect
    end if


    ! calculate internal coordinates of midpoint
    call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
        glob%xcoords,glob%xgradient,glob%icoords(1:dimer%varperimage), &
        glob%igradient(1:dimer%varperimage))

    !calculate internal coordinates of endpoint 1
    glob%xcoords=glob%xcoords+dimer%xtangent
    call dlf_direct_xtoi(glob%nvar,dimer%varperimage,dimer%coreperimage, &
        glob%xcoords,glob%xgradient,glob%icoords(dimer%varperimage+1:), &
        glob%igradient(dimer%varperimage+1:))
    ! all the gradients are rubbish, but will be recalculated anyway

    ! make second set of icoords relative
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:) -&
        glob%icoords(1:dimer%varperimage)
    ! normalise second set of icoords (they are not necessarily normal in the new HDLCs)
    svar=ddot(dimer%varperimage,glob%icoords(dimer%varperimage+1:),1, &
        glob%icoords(dimer%varperimage+1:),1)
    glob%icoords(dimer%varperimage+1:)=glob%icoords(dimer%varperimage+1:) &
        / dsqrt(svar) * dimer%delta
  else
    ! store cartesian dimer midpoint
    dimer%xmidpoint=glob%xcoords
  end if
  if(printl>=4) write(stdout,"('Next calculation will be the dimer &
        & midpoint')")
end subroutine dlf_dimer_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_checkpoint_dimer_write
!!
!! FUNCTION
!! Write checkpoint information
!!
!! SYNOPSIS
subroutine dlf_checkpoint_dimer_write
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr
  use dlf_dimer, only: dimer
  use dlf_checkpoint, only: tchkform,write_separator
  implicit none
! **********************************************************************
  if(tchkform) then

    open(unit=100,file="dlf_dimer.chk",form="formatted")

    call write_separator(100,"Dimer Sizes")
    write(100,*) dimer%varperimage
    call write_separator(100,"Dimer Parameters")
    write(100,*) dimer%status, dimer%mode, dimer%delta, dimer%emid, &
        dimer%curve, dimer%tolrot, dimer%toldrot, dimer%toptdir, &
        dimer%cgstep, dimer%extrapolate_grad, dimer%nrot, dimer%maxrot, &
        dimer%phi, dimer%dcurvedphi, dimer%cdelta
    call write_separator(100,"Dimer Arrays")
    write(100,*) dimer%xtangent, dimer%xmidpoint, dimer%rotgrad, &
        dimer%rotdir, dimer%oldrotgrad, dimer%oldrotdir, dimer%vector, &
        dimer%theta, dimer%grad1
    call write_separator(100,"END")


  else

    open(unit=100,file="dlf_dimer.chk",form="unformatted")

    call write_separator(100,"Dimer Sizes")
    write(100) dimer%varperimage
    call write_separator(100,"Dimer Parameters")
    write(100) dimer%status, dimer%mode, dimer%delta, dimer%emid, &
        dimer%curve, dimer%tolrot, dimer%toldrot, dimer%toptdir, &
        dimer%cgstep, dimer%extrapolate_grad, dimer%nrot, dimer%maxrot, &
        dimer%phi, dimer%dcurvedphi, dimer%cdelta
    call write_separator(100,"Dimer Arrays")
    write(100) dimer%xtangent, dimer%xmidpoint, dimer%rotgrad, &
        dimer%rotdir, dimer%oldrotgrad, dimer%oldrotdir, dimer%vector, &
        dimer%theta, dimer%grad1
    call write_separator(100,"END")

  end if

  close(100)
    
end subroutine dlf_checkpoint_dimer_write
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dimer/dlf_checkpoint_dimer_read
!!
!! FUNCTION
!! Read checkpoint information
!!
!! SYNOPSIS
subroutine dlf_checkpoint_dimer_read(tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr,stdout,printl
  use dlf_dimer, only: dimer
  use dlf_checkpoint, only: tchkform,read_separator
  implicit none
  logical,intent(out) :: tok
  logical             :: tchk
  integer             :: varperimage
! **********************************************************************
  tok=.false.

  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_dimer.chk",EXIST=tchk)
  if(.not.tchk) then
    write(stdout,10) "File dlf_dimer.chk not found"
    return
  end if

  if(tchkform) then
    open(unit=100,file="dlf_dimer.chk",form="formatted")
  else
    open(unit=100,file="dlf_dimer.chk",form="unformatted")
  end if


  call read_separator(100,"Dimer Sizes",tchk)
  if(.not.tchk) return  

  if(tchkform) then
    read(100,*,end=201,err=200) varperimage
  else
    read(100,end=201,err=200) varperimage
  end if

  if(dimer%varperimage/=varperimage) then
    write(stdout,10) "Different numbers of variables per dimer image"
    close(100)
    return
  end if

  call read_separator(100,"Dimer Parameters",tchk)
  if(.not.tchk) return    

  if(tchkform) then
    read(100,*,end=201,err=200) &
        dimer%status, dimer%mode, dimer%delta, dimer%emid, &
        dimer%curve, dimer%tolrot, dimer%toldrot, dimer%toptdir, &
        dimer%cgstep, dimer%extrapolate_grad, dimer%nrot, dimer%maxrot, &
        dimer%phi, dimer%dcurvedphi, dimer%cdelta
  else
    read(100,end=201,err=200) &
        dimer%status, dimer%mode, dimer%delta, dimer%emid, &
        dimer%curve, dimer%tolrot, dimer%toldrot, dimer%toptdir, &
        dimer%cgstep, dimer%extrapolate_grad, dimer%nrot, dimer%maxrot, &
        dimer%phi, dimer%dcurvedphi, dimer%cdelta
  end if

  call read_separator(100,"Dimer Arrays",tchk)
  if(.not.tchk) return

  if(tchkform) then
    read(100,*,end=201,err=200) &
        dimer%xtangent, dimer%xmidpoint, dimer%rotgrad, &
        dimer%rotdir, dimer%oldrotgrad, dimer%oldrotdir, dimer%vector, &
        dimer%theta, dimer%grad1
  else
    read(100,end=201,err=200) &
        dimer%xtangent, dimer%xmidpoint, dimer%rotgrad, &
        dimer%rotdir, dimer%oldrotgrad, dimer%oldrotdir, dimer%vector, &
        dimer%theta, dimer%grad1
  end if

  call read_separator(100,"END",tchk)
  if(.not.tchk) return

  close(100)
  tok=.true.

  if(printl >= 6) write(stdout,"('Dimer checkpoint file successfully read')")

  return

  ! return on error
200 continue
  write(stdout,10) "Error reading file"
  return
201 continue
  write(stdout,10) "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a)
    
end subroutine dlf_checkpoint_dimer_read
!!****

