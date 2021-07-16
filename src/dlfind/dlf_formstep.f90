! **********************************************************************
! **                Optimisation Algorithms: main unit                **
! **********************************************************************
!!****h* DL-FIND/formstep
!!
!! NAME
!! formstep
!!
!! FUNCTION
!! Optimisation algorithms: determine a search direction. 
!!
!! The variable
!!              IOPT
!! determines which algorithm is to be used. The routines called by this file are
!! supposed to do unconstrained optimisation in the corresponding internal
!! coordinates.
!! The file also contains routines for calculating and updating a Hessian.
!!
!! Inputs
!!    glob%icoords
!!    glob%igradient
!!    glob%toldenergy
!!     Initialisation routines may require additional parameters.
!! 
!! Outputs
!!    glob%step
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
!!****
module dlf_formstep_module
  use dlf_parameter_module, only: rk
  ! conjugate gradient data
  real(rk), allocatable, save :: oldg1(:)     ! glob%nivar
  real(rk), allocatable, save :: g1(:)        ! glob%nivar
  real(rk), allocatable, save :: oldcoords(:) ! glob%nivar
  integer, save               :: cgstep       ! number of CG steps taken
  integer, parameter          :: maxcgstep=10
  ! damped dynamics data
  real(rk)                    :: fricm        ! decreasing friction in damped dynamics
  ! data for dlf_formstep_set_tsmode
  real(rk), allocatable, save :: tscoords(:) ! x-coords
  real(rk), allocatable, save :: tsmode_r(:) ! x-coords, relative
  real(rk), save              :: energy      ! ts energy
  logical , save              :: tenergy     ! is energy set?
  logical , save              :: tsc_ok      ! are TS coords ok to use?
  logical , save              :: tsm_ok      ! is TS-mode ok to use?
  ! communication to outside
  logical , save              :: needhessian 
end module dlf_formstep_module

! module for Optimisers using the Hessian
module dlf_hessian
  use dlf_parameter_module, only: rk
  logical ,save        :: fd_hess_running ! true if FD Hessian is currently running
                                   ! initially set F in dlf_formstep_init
  integer ,save        :: nihvar   ! size of Hessian (nihvar, nihvar)
  integer, save        :: numfd    ! number of FD Hessian eigenmodes to calculate
  integer ,save        :: iivar,direction
  real(rk),save        :: soft ! Hessian eigenvalues absolutely smaller 
                               ! than "soft" are ignored in P-RFO
  integer ,save        :: follow ! Type of mode following:
                          ! 0: no mode following: TS mode has the lowest eigenvalue
                          ! 1: specify direction by input - not yet implemented
                          ! 2: determine direction at first P-RFO step
                          ! 3: update direction at each P-RFO step
  logical ,save        :: tsvectorset ! is tsverctor defined?
  integer ,save        :: tsmode ! number of mode to maximise
  logical ,save        :: twopoint ! type of finite difference Hessian
  real(rk),save        :: storeenergy
  logical ,save        :: carthessian ! should the Hessian be updated in 
                                      ! Cartesian coordinates (T) or internals (F)
  real(rk),allocatable,save :: eigvec(:,:)  ! (nihvar,nihvar) Hessian eigenmodes
  real(rk),allocatable,save :: eigval(:)    ! (nihvar) Hessian eigenvalues
  real(rk),allocatable,save :: storegrad(:) ! (nihvar) old gradient in fdhessian
  integer             ,save :: iupd ! actual number of Hessian updates
  real(rk),allocatable,save :: tsvector(:) ! (nihvar) Vector to follow in P-RFO
  ! The old arrays are set in formstep. Used there and in hessian_update
  real(rk),allocatable,save :: oldc(:)      ! (nihvar) old i-coords
  real(rk),allocatable,save :: oldgrad(:)   ! (nihvar) old i-coords
  real(rk)            ,save :: minstep      ! minimum step length for Hessian update to be performed
                                       ! set in formstep_init
  logical,save              :: fracrec ! recalculate a fraction of the hessian?
end module dlf_hessian

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep
!!
!! FUNCTION
!!
!! The optimisation algorithm: calculate a step vector from the gradient
!! and possibly some gradient history information or hessian.
!!
!! Some optimisers are implemented here, some in their own files (L-BFGS,
!! P-RFO).
!!
!! INPUTS
!!
!! glob%igradient
!!
!!
!! OUTPUTS
!! 
!! glob%step
!!
!! SYNOPSIS
subroutine dlf_formstep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,pi
  use dlf_formstep_module
  use dlf_hessian
  use dlf_allocate
  use dlf_constants, only : dlf_constants_get
  implicit none
  !
  real(rk)       :: oldstep(glob%nivar)
  real(rk)       :: svar,gamma
  logical        :: trestart
  integer        :: ivar,jvar
  real(rk),allocatable :: hess(:,:) ! for Newton-Raphson
  real(rk),allocatable :: ev2(:) ! for Newton-Raphson
  !real(rk),allocatable :: eigval(:) ! for Newton-Raphson
  real(rk),external :: ddot
  real(rk)       :: friction
  real(rk)       :: eigvalThreshold = 1.0d-10
  real(rk)       :: projGradient
  ! for NR-QTS
  real(rk)       :: beta_hbar,dtau,CM_INV_FOR_AMU,amu
! **********************************************************************
  select case (glob%iopt)

! ======================================================================
! Steepest descent
! ======================================================================
  case (0)

    glob%step(:)= -1.D0 * glob%igradient(:)
    if(glob%iline/=0)  glob%step(:)= glob%step(:)* glob%scalestep

! ======================================================================
! Conjugate gradient following Polak and Ribiere
! ======================================================================
  case (1)
    
    if(glob%toldenergy) then

      !Powell-Beale Restarts
      trestart= (&
          dot_product( oldg1(:) , glob%igradient(:)) .gt. &
          0.2D0 * dot_product( glob%igradient(:), glob%igradient(:)) )

      ! get old step from coordinate differences
      oldstep(:)=glob%icoords(:)-oldcoords(:)
      ! normalise?

      if(trestart) then
        gamma = 0.D0
        if(printl >= 2) print*,"Restarting CG algorithm"
      else

        ! should old gradient and old step be stored in the global module
        ! or internally here?
        gamma=dot_product( &
            glob%igradient(:)-oldg1(:) , glob%igradient(:) ) / &
            dot_product( oldg1(:) , oldg1(:) )
        if(printl>=6) print*,"Continuing CG algorithm"
      end if

      glob%step(:)= -1.D0 * glob%igradient(:) + gamma * oldstep(:)

    else
      ! first step: do an SD step
      glob%step(:)= -1.D0 * glob%igradient(:)
    end if
    oldg1(:)=glob%igradient(:)
    oldcoords(:)=glob%icoords(:)

! ======================================================================
! Conjugate gradient following Polak and Ribiere - better implementation
! ======================================================================
  case (2)
    ! oldcoords is the old search direction here!
    if(glob%toldenergy) then

      if(cgstep < maxcgstep) then

        gamma=ddot(glob%nivar,glob%igradient(:)-oldg1(:),1,&
            glob%igradient(:),1) / &
            ddot(glob%nivar,oldg1(:),1,oldg1(:),1)
        cgstep=cgstep+1
        if(gamma < 0.D0) gamma=0.D0
      else
        gamma = 0.D0
        oldcoords(:)=0.D0
        cgstep=0
        if(printl >= 4) write(stdout,'("Restarting CG algorithm")')

      end if

      glob%step(:)= -1.D0 * glob%igradient(:) + gamma * oldcoords(:)

    else
      ! first step: do an SD step
      glob%step(:)= -1.D0 * glob%igradient(:)
      cgstep=0
    end if
    
    oldg1(:)=glob%igradient(:)
    oldcoords(:)=glob%step(:)

    glob%step(:)=glob%step(:)*glob%scalestep

! ======================================================================
! L-BFGS
! ======================================================================
  CASE (3)

    CALL DLF_LBFGS_STEP(GLOB%ICOORDS,GLOB%IGRADIENT,GLOB%STEP)

! ======================================================================
! P-RFO
! ======================================================================
  CASE (10)

    ! Note in the microiterative case (where nihvar /= glob%nivar), we
    ! rely on icoords etc. being ordered with inner region first.

    ! store old coords and grad
    if(sum((oldc(:)-glob%icoords(1:nihvar))**2) > minstep ) then
      oldc(:)=glob%icoords(1:nihvar)
      oldgrad(:)=glob%igradient(1:nihvar)
    end if
    ! send information to set_tsmode
    call dlf_formstep_set_tsmode(1,-1,glob%energy) ! send energy
    call dlf_formstep_set_tsmode(glob%nvar,0,glob%xcoords) ! TS-geometry
    call dlf_prfo_step(nihvar, glob%icoords(1:nihvar), &
        glob%igradient(1:nihvar), glob%ihessian, glob%step(1:nihvar))

! ======================================================================
! Hessian and thermal analysis only
! ======================================================================
  CASE (9, 11, 12)
    ! do nothing

! ======================================================================
! Newton-Raphson
! ======================================================================
  CASE (20)
    if(.not.glob%havehessian) call dlf_fail("No Hessian present in Newton-Raphson")

    ! store Hessian
    call allocate(hess,glob%nivar,glob%nivar)
    call allocate(ev2,glob%nivar)
    hess(:,:)=glob%ihessian(:,:)

!    ivar = array_invert(hess,svar,.true.,glob%nivar)
!!$    ! full inverstion - only works without soft modes
!!$    call dlf_matrix_invert(glob%nivar,.false.,hess,svar)
!!$!    if(printl>=4) write(stdout,"('Determinant of H in NR step: ',es10.2)") svar

    ! Alternative: skip eigenvectors which belong to the 6 softest modes
    if(printl>=4) write(stdout,"('Diagonalizing the Hessian...')")
    call dlf_matrix_diagonalise(glob%nivar,glob%ihessian,eigval,hess)
    ev2=abs(eigval)
    soft=-0.1D0
    do ivar=1,glob%nzero
      soft=minval(ev2)
      ev2(minloc(ev2))=huge(1.D0)
    end do
    if(printl>=4) write(stdout,"('Criterion for soft modes in NR: ',es10.3)") soft

    if(glob%icoord==190) then
      call dlf_constants_get("AMU",amu)
      call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMu)
    END if

    glob%step(:)=0.D0
    do ivar=1,glob%nivar
      if(abs(eigval(ivar))>soft) then
        if(ivar<=12.and.printl>=4.and.glob%icoord==190) then
          ! I am not sure how realistic these are with varying dtau
          svar=sqrt(abs(eigval(ivar)*dble(glob%nimage)*amu))*CM_INV_FOR_AMU
          if(eigval(ivar)<0.D0) svar=-svar
          write(stdout,'(" Eigval ",i2,1x,es10.3,2x,f10.3," cm^-1, used")') &
              ivar,eigval(ivar),svar
        end if
        eigval(ivar)=1.D0/eigval(ivar)
        ! for finding first-order saddle points:
        if(glob%icoord==190 .and. ivar>1 .and. eigval(ivar)<0.D0 ) then
          eigval(ivar)=-eigval(ivar)
          if(printl>=4) write(stdout,'("Inverted mode",i3)') ivar
        end if

        ! contribution to NR step
        svar=ddot(glob%nivar,hess(:,ivar),1,glob%igradient,1)
        glob%step=glob%step - hess(:,ivar) * eigval(ivar) * svar 

      else
        if(ivar<=12.and.printl>=4.and.glob%icoord==190) then
          svar=sqrt(abs(eigval(ivar)*dble(glob%nimage)*amu))*CM_INV_FOR_AMU
          if(eigval(ivar)<0.D0) svar=-svar
          write(stdout,'(" Eigval ",i2,1x,es10.3,2x,f10.3," cm^-1, ignored")') &
              ivar,eigval(ivar),svar
        end if
        eigval(ivar)=0.D0
      end if
    end do

    call deallocate(hess)
    call deallocate(ev2)
    
    ! store old coords and grad
    if(sum((oldc(:)-glob%icoords(:))**2) > minstep ) then
      oldc(:)=glob%icoords(:)
      oldgrad(:)=glob%igradient(:)
    end if

! ======================================================================
! Damped dynamics
! ======================================================================
  CASE (30)
    if(glob%toldenergy) then
      if(glob%energy<glob%oldenergy) then
        friction=fricm*glob%fricfac
        fricm=friction
      else
        if(printl>=4) write(stdout,"('Energy increasing, using high &
            &friction')")
        friction=glob%fricp
      end if
    else
      friction=glob%fric0
      fricm=friction
      oldcoords(:)=glob%icoords(:)
    end if
    glob%step(:)=1.D0/(1.D0+friction)*( &
        (1.D0-friction)*glob%icoords(:) - &
        (1.D0-friction)*oldcoords(:) - &
        glob%igradient(:)*glob%timestep**2 )

    !svar=sum(glob%step**2)*0.5D0/timestep**2+glob%energy
    !print*,"Total energy:",svar,"friction",friction

    oldcoords(:)=glob%icoords(:)

! ======================================================================
! Lagrange-Newton
! ======================================================================
  CASE (40)
    if(.not.glob%havehessian) call dlf_fail("No Hessian present in Newton-Raphson")

    ! Solving the equation for the step,
    !
    !    H s = -g
    ! =>   s = - H(-1) g
    !
    ! But matrix inversion is prone to numerical instabilities when the
    ! interstate coupling gradient is very small.
    !
    ! Therefore instead we diagonalise the Hessian matrix (following MNDO).
    !
    !      D = X(-1) H X
    !
    ! The step is then,
    !
    !      s = - X D(-1) X(-1) g
    !
    ! X, the matrix of eigenvectors, is simple to invert as X(-1) = X(t)
    ! D, the matrix of eigenvalues, is also straightforward as it is diagonal
    !
    call dlf_matrix_diagonalise(glob%nivar, glob%ihessian, eigval, eigvec)
    glob%step = 0.0d0
    do ivar = 1, glob%nivar
       if (abs(eigval(ivar)) > eigvalThreshold) then 
          projGradient = ddot(glob%nivar, eigvec(1, ivar), 1, glob%igradient, 1)
          do jvar = 1, glob%nivar
             glob%step(jvar) = glob%step(jvar) - &
                  (projGradient / eigval(ivar)) * eigvec(jvar, ivar)
          enddo
       endif
    enddo

    if (printl >= 6) then
       write(stdout, '(a)') "Hessian eigenvalues:"
       write(stdout, '(12f10.5)') eigval(:)
       write(stdout, '(a)') "Hessian eigenvectors:"
       do ivar = 1, glob%nivar
          write(stdout, '(12f10.5)') eigvec(ivar, :)
       end do
    end if

    ! store old coords and grad
    if(sum((oldc(:)-glob%icoords(:))**2) > minstep ) then
      oldc(:)=glob%icoords(:)
      call dlf_ln_savegrads
      ! Note oldgrad(:) is not used for LN, as the 'old' gradient
      ! for updating the Hessian has to be rebuilt with knowledge
      ! of the current Lagrange multipliers     
    end if

! ======================================================================
! Wrong optimisation type setting
! ======================================================================
  case default
    write(stderr,*) "Optimisation algorithm",glob%iopt,"not implemented"
    call dlf_fail("Optimisation algorithm error")

  end select
end subroutine dlf_formstep
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep_restart
!!
!! FUNCTION
!!
!! Restart the optimisation algorithm.
!!
!! SYNOPSIS
subroutine dlf_formstep_restart
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_formstep_module, only: cgstep,maxcgstep,fricm,oldcoords
  implicit none
! **********************************************************************
  select case (glob%iopt)
  case(2)
    cgstep = maxcgstep + 1 ! make sure CG is restarted
! L-BFGS
  case (3)
    call dlf_lbfgs_restart
  case (30)
    fricm=glob%fric0
    oldcoords(:)=glob%icoords(:)
  end select
  glob%toldenergy=.false.

  ! call an external routine (currently only used by ChemShell)
  call dlf_update()

end subroutine dlf_formstep_restart
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep_init
!!
!! FUNCTION
!!
!! Allocate arrays for the optimisation algorithm.
!!
!!
!! SYNOPSIS
subroutine dlf_formstep_init(needhessian_)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_formstep_module
  use dlf_hessian
  use dlf_allocate, only: allocate
  implicit none
  logical,intent(out)     :: needhessian_
! **********************************************************************
  fd_hess_running=.false.
  needhessian=.false.

  ! get Hessian values from global module
  soft=glob%soft
  ! To reproduce the old default "twopoint" behaviour, we default to 
  ! a two point FD Hessian if an external Hessian is selected but not
  ! available
  twopoint = (glob%inithessian == 2 .or. glob%inithessian == 0)
  carthessian=(glob%carthessian==1)
  follow=0 ! get from global eventually - IMPROVE
  tsmode=1
  minstep=glob%minstep

  ! energy not set in dlf_formstep_set_tsmode
  tenergy=.false.

  select case (glob%iopt)
! steepest descent
  case (1,2,30)
    call allocate(oldg1,glob%nivar)
    call allocate(g1,glob%nivar)
    call allocate(oldcoords,glob%nivar)
! L-BFGS
  case (3)
    call dlf_lbfgs_init(glob%nivar,glob%lbfgs_mem)
  case (9)
! fd-test
    needhessian=.true.
  case (10)
! P-RFO
    needhessian=.true.
    if (glob%imicroiter > 0) then
       call allocate(tsvector,glob%nicore)
    else
       call allocate(tsvector,glob%nivar)
    end if
    tsvector(:)=0.D0
    tsvectorset=.false.
  case (11,12)
! Hessian and thermal analysis only
    needhessian=.true.
! Newton-Raphson
  case (20, 40)
    needhessian=.true.
  end select

  ! allocate the global Hessian if required
  if(needhessian) then
    ! part of module dlf_hessian
    if (glob%imicroiter > 0) then
       ! In microiterative P-RFO only the inner region has a Hessian
       nihvar = glob%nicore
    else
       nihvar = glob%nivar
    end if
    call allocate(glob%ihessian,nihvar,nihvar)
    glob%ihessian=0.D0
    glob%havehessian=.false.
    ! set numfd just to init it, reset in fdhessian 
    numfd = nihvar

    ! update and PRFO
    call allocate(oldc,nihvar)
    oldc(:)=0.D0 ! initialise, as the question of update may depend on them
    call allocate(oldgrad,nihvar)
    iupd=0
    ! FD HESSIAN calculation
    call allocate(storegrad,nihvar)

    call allocate(eigval,nihvar)
    call allocate(eigvec,nihvar,nihvar)

  end if

  needhessian_=needhessian

  ! Initialise microiterative optimisation if required
  if (glob%imicroiter > 0) call dlf_microiter_init

end subroutine dlf_formstep_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep_destroy
!!
!! FUNCTION
!!
!! Deallocate arrays for the optimisation algorithm.
!!
!!
!! SYNOPSIS
subroutine dlf_formstep_destroy
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_formstep_module
  use dlf_hessian
  use dlf_allocate, only: deallocate
  implicit none
! **********************************************************************

  select case (glob%iopt)
! steepest descent
  case (1,2,30)
    if (allocated(oldg1)) call deallocate(oldg1)
    if (allocated(g1)) call deallocate(g1)
    if (allocated(oldcoords)) call deallocate(oldcoords)

! L-BFGS
  case (3)
    call dlf_lbfgs_destroy
! P-RFO
  case (10)
    if (allocated(tsvector)) call deallocate(tsvector)
  end select

  if(allocated(glob%ihessian)) then
    call deallocate(glob%ihessian)
    glob%havehessian=.false.

    if (allocated(oldc)) call deallocate(oldc)
    if (allocated(oldgrad)) call deallocate(oldgrad)

    ! FD HESSIAN calculation
    if (allocated(storegrad)) call deallocate(storegrad)

    if (allocated(eigval)) call deallocate(eigval)
    if (allocated(eigvec)) call deallocate(eigvec)

  end if

  if (glob%imicroiter > 0) call dlf_microiter_destroy

end subroutine dlf_formstep_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_formstep_write
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_formstep_module
  use dlf_hessian
  use dlf_checkpoint, only: tchkform,write_separator
  implicit none
! **********************************************************************

! Open the checkpoint file (it may result in an empty file)
  if(tchkform) then
    open(unit=100,file="dlf_formstep.chk",form="formatted")
  else
    open(unit=100,file="dlf_formstep.chk",form="unformatted")
  end if

! Data for dlf_formstep_set_tsmode
  if(allocated(tscoords).or.allocated(tsmode_r).or.tenergy) then
    call write_separator(100,"TSMODE")

    ! write allocation status
    if(tchkform) then
      write(100,*) allocated(tscoords), allocated(tsmode_r),tenergy
    else
      write(100) allocated(tscoords), allocated(tsmode_r),tenergy
    end if

    if(allocated(tscoords)) then
      if(tchkform) then
        write(100,*) tscoords(:)
      else
        write(100) tscoords(:)
      end if
    end if

    if(allocated(tsmode_r)) then
      if(tchkform) then
        write(100,*) tsmode_r(:)
      else
        write(100) tsmode_r(:)
      end if
    end if

    if(tenergy) then
      if(tchkform) then
        write(100,*) energy
      else
        write(100) energy
      end if
    end if

    call write_separator(100,"END TSM")
  end if

  select case (glob%iopt)

! Algorithms with no checkpoint:
  case (0, 11, 12, 20, 40, 51, 52)

! Conjugate gradient, damped dyn
  case (1:2,30)

    if(tchkform) then
      call write_separator(100,"CG-Arrays")
      write(100,*) cgstep,fricm
      write(100,*) oldg1,g1,oldcoords
      call write_separator(100,"END")
    else
      call write_separator(100,"CG-Arrays")
      write(100) cgstep,fricm
      write(100) oldg1,g1,oldcoords
      call write_separator(100,"END")
    end if

! L-BFGS
  CASE (3)

    CALL DLF_checkpoint_LBFGS_write

! P-RFO
  case (10)
    call write_separator(100,"TS-vectorset")
    if(tchkform) then
      write(100,*) tsvectorset
    else
      write(100) tsvectorset
    end if
    if(tsvectorset) then
      call write_separator(100,"TS-vector")
      if(tchkform) then
        write(100,*) tsmode,tsvector
      else
        write(100) tsmode,tsvector
      end if
    end if
    call write_separator(100,"END")

! ======================================================================
! Wrong optimisation type setting
! ======================================================================
  case default
    write(stderr,'(a,i4,a)') "Optimisation algorithm",glob%iopt,"not implemented"
    call dlf_fail("Optimisation algorithm error")

  end select
  
  ! close dlf_formstep.chk
  close(100)

! ======================================================================
! Write Hessian Data
! ======================================================================
  if(allocated(glob%ihessian)) then
    if(tchkform) then
      open(unit=100,file="dlf_hessian.chk",form="formatted")
      call write_separator(100,"Hessian size")
      write(100,*) nihvar
      call write_separator(100,"Hessian data")
      write(100,*) glob%havehessian,fd_hess_running,iivar,direction,storeenergy,iupd,fracrec,numfd
      call write_separator(100,"Hessian arrays")
      write(100,*) glob%ihessian,oldc,oldgrad
      if(allocated(storegrad)) write(100,*) storegrad
      call write_separator(100,"END")
      close(100)
    else
      open(unit=100,file="dlf_hessian.chk",form="unformatted")
      call write_separator(100,"Hessian size")
      write(100) nihvar
      call write_separator(100,"Hessian data")
      write(100) glob%havehessian,fd_hess_running,iivar,direction,storeenergy,iupd,fracrec,numfd
      call write_separator(100,"Hessian arrays")
      write(100) glob%ihessian,oldc,oldgrad
      if(allocated(storegrad)) write(100) storegrad
      call write_separator(100,"END")
      close(100)
    end if
  end if

end subroutine dlf_checkpoint_formstep_write

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_formstep_read(tok)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,stderr
  use dlf_formstep_module
  use dlf_hessian
  use dlf_allocate, only: allocate, deallocate
  use dlf_checkpoint, only: tchkform, read_separator
  implicit none
  logical,intent(out) :: tok
  logical             :: tchk
  integer             :: var
  logical             :: alloc_tscoords,alloc_tsmode_r
! **********************************************************************
  tok=.false.

  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_formstep.chk",EXIST=tchk)
  if(.not.tchk) then
    write(stdout,10) "File dlf_formstep.chk not found"
    return
  end if
  
  ! open the checkpoint file
  if(tchkform) then
    open(unit=100,file="dlf_formstep.chk",form="formatted")
  else
    open(unit=100,file="dlf_formstep.chk",form="unformatted")
  end if

! Data for dlf_formstep_set_tsmode
  call read_separator(100,"TSMODE",tchk)
  if(tchk) then

    ! read allocation status and data
    if(tchkform) then
      read(100,*,end=201,err=200) alloc_tscoords,alloc_tsmode_r,tenergy
    else
      read(100,end=201,err=200) alloc_tscoords,alloc_tsmode_r,tenergy
    end if

    if(alloc_tscoords) then
      if(.not.allocated(tscoords)) call allocate(tscoords,glob%nvar)
      if(tchkform) then
        read(100,*,end=201,err=200) tscoords(:)
      else
        read(100,end=201,err=200) tscoords(:)
      end if
    else
      if(allocated(tscoords)) call deallocate(tscoords)
    end if

    if(alloc_tsmode_r) then
      if(.not.allocated(tsmode_r)) call allocate(tsmode_r,glob%nvar)
      if(tchkform) then
        read(100,*,end=201,err=200) tsmode_r(:)
      else
        read(100,end=201,err=200) tsmode_r(:)
      end if
    else
      if (allocated(tsmode_r)) call deallocate(tsmode_r)
    end if

    if(tenergy) then
      if(tchkform) then
        read(100,*,end=201,err=200) energy
      else
        read(100,end=201,err=200) energy
      end if
    end if

    call read_separator(100,"END TSM",tchk)
    if(.not.tchk) return

  else
    ! the checkpoint file has to be reopened
    if(tchkform) then
      open(unit=100,file="dlf_formstep.chk",form="formatted")
    else
      open(unit=100,file="dlf_formstep.chk",form="unformatted")
    end if
  end if

  select case (glob%iopt)

! ======================================================================
! Algorithms with no checkpoint
! ======================================================================
  case (0,11,12,20, 40, 51, 52)

    tok=.true.

! ======================================================================
! Algorithms with checkpoint handled here: Conjugate gradient, damped
! dynamics
! ======================================================================
  case (1:2,30)

    call read_separator(100,"CG-Arrays",tchk)
    if(.not.tchk) return 
    
    if(tchkform) then
      read(100,*,end=201,err=200) cgstep,fricm
      read(100,*,end=201,err=200) oldg1,g1,oldcoords
    else
      read(100,end=201,err=200) cgstep,fricm
      read(100,end=201,err=200) oldg1,g1,oldcoords
    end if

    call read_separator(100,"END",tchk)
    if(.not.tchk) return

! ======================================================================
! L-BFGS
! ======================================================================
  CASE (3)

    CALL DLF_checkpoint_LBFGS_read(tok)
    return

! P-RFO
  case (10)

    call read_separator(100,"TS-vectorset",tchk)
    if(.not.tchk) return 

    if(tchkform) then
      read(100,*,end=201,err=200) tsvectorset
    else
      read(100,end=201,err=200) tsvectorset
    end if

    if(tsvectorset) then
      call read_separator(100,"TS-vector",tchk)
      if(.not.tchk) return 

      if(tchkform) then
        read(100,*,end=201,err=200) tsmode,tsvector
      else
        read(100,end=201,err=200) tsmode,tsvector
      end if
    end if
    call read_separator(100,"END",tchk)
    if(.not.tchk) return

! ======================================================================
! Wrong optimisation type setting
! ======================================================================
  case default
    write(stderr,'(a,i4,a)') "Optimisation algorithm",glob%iopt,"not implemented"
    call dlf_fail("Optimisation algorithm error")

  end select

  ! close dlf_formstep.chk
  close(100)

! ======================================================================
! Read Hessian Data
! ======================================================================
  if(allocated(glob%ihessian)) then
    tok=.false.
    ! check if checkpoint file exists
    INQUIRE(FILE="dlf_hessian.chk",EXIST=tchk)
    if(.not.tchk) then
      write(stdout,10) "File dlf_hessian.chk not found"
      return
    end if
    if(tchkform) then
      open(unit=100,file="dlf_hessian.chk",form="formatted")
    else
      open(unit=100,file="dlf_hessian.chk",form="unformatted")
    end if

    call read_separator(100,"Hessian size",tchk)
    if(.not.tchk) return

    if(tchkform) then
      read(100,*,end=201,err=200) var
    else
      read(100,end=201,err=200) var
    end if
    if(var/=nihvar) then
      print*,var,nihvar
      write(stdout,10) "Inconsistent Hessian size"
      close(100)
      return
    end if

    call read_separator(100,"Hessian data",tchk)
    if(.not.tchk) return 

    if(tchkform) then
      read(100,*,end=201,err=200) glob%havehessian,fd_hess_running,iivar, &
          direction,storeenergy,iupd,fracrec,numfd
    else
      read(100,end=201,err=200) glob%havehessian,fd_hess_running,iivar, &
          direction,storeenergy,iupd,fracrec,numfd
    end if

    call read_separator(100,"Hessian arrays",tchk)
    if(.not.tchk) return 

    if(tchkform) then
      read(100,*,end=201,err=200) glob%ihessian,oldc,oldgrad
    else
      read(100,end=201,err=200) glob%ihessian,oldc,oldgrad
    end if
    if(allocated(storegrad)) then
      if(tchkform) then
        read(100,*,end=201,err=200) storegrad
      else
        read(100,end=201,err=200) storegrad
      end if
    end if
    call read_separator(100,"END",tchk)
    if(.not.tchk) return 

    close(100)
  end if

  tok=.true.
  return

  ! return on error
200 continue
  write(stdout,10) "Error reading CG/Hessian checkpoint file"
  return
201 continue
  write(stdout,10) "Error (EOF) reading CG/Hessian checkpoint file"
  return

10 format("Checkpoint reading WARNING: ",a)


end subroutine dlf_checkpoint_formstep_read

! **********************************************************************
! **                                                                  **
! **        DL-FIND Hessian Routines (including P-RFO)                **
! **                                                                  **
! **                                                                  **
! **                                                                  **
! **                                                                  **
! **                                                                  **
! **********************************************************************

!!****h* formstep/hessian
!!
!! NAME
!! hessian
!!
!! FUNCTION
!! Calculate (and use) the hessian. Includes P-RFO
!!
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_makehessian
!!
!! FUNCTION
!!
!! called from the main cycle in dl-find
!! Build up a Hessian by either:
!! * Reading it in from external sources (dlf_get_hessian)
!! * Updating an existing Hessian
!! * Building it from scratch by finite-difference
!! * Setting it to the identity matrix
!! * Improving an existing Hessian by finite-difference in its eigenmodes
!!
!! SYNOPSIS
subroutine dlf_makehessian(trerun_energy,tconv)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_stat, only: stat
  use dlf_constants, only: dlf_constants_get
  use dlf_allocate
  use dlf_hessian
  implicit none
  logical, intent(inout) :: trerun_energy
  logical, intent(inout) :: tconv
  integer                :: status
  integer                :: iimage ! not used here
  logical                :: was_updated,tok
  ! this should be improved ...
  ! real(rk) :: hess_loc(glob%nvar,glob%nvar)
  real(rk),allocatable :: hess_loc(:,:) !glob%nvar,glob%nvar)
  real(rk),allocatable :: mass(:) 
  real(rk) :: svar,arr2(2)
  integer :: i,ivar
! **********************************************************************
  ! Don't update the Hessian on a microiterative step
  ! as the Hessian only applies to the inner (P-RFO) region
  if (glob%imicroiter == 2) return

  if (glob%icoord >= 10 .and. glob%icoord <= 19) then
     if (glob%imicroiter > 0) call dlf_fail("microiterative LN not implemented")
     call dlf_conint_make_ln_hess(trerun_energy,tconv)
     return
  endif

  if(glob%icoord==190) then
    if (glob%imicroiter > 0) call dlf_fail("microiterative qts not implemented") 
    call dlf_qts_get_hessian(trerun_energy)
    call qts_hessian_etos_halfpath 
    !
    return
    !
  end if

  ! read hessian if inithessian=5
  if (glob%inithessian == 5) then
    if (glob%imicroiter > 0) call dlf_fail("microiterative qts not implemented")
    ivar=1
    call allocate(mass,glob%nat)
    call read_qts_hessian(glob%nat,ivar,glob%nivar,glob%temperature,&
        svar,glob%xcoords,glob%ihessian,svar,arr2,mass,"",tok)
    if(.not.tok) call dlf_fail("Reading Hessian from file failed")
    if(ivar/=1) call dlf_fail("Hessian must contain one image")
    call deallocate(mass)
    call dlf_constants_get("AMU",svar)
    glob%ihessian=glob%ihessian*svar !*(1.66054D-27/9.10939D-31)
    call dlf_matrix_diagonalise(glob%nivar,glob%ihessian,eigval,eigvec)
    if (printl>=4 .or. (glob%iopt==11 .and. printl>=2)) then      
      write(stdout,"(/,'Hessian eigenvalues:')")
      write(stdout,"(12f9.5)") eigval
    end if
    glob%havehessian=.true.
    return
  end if

  call dlf_hessian_update(nihvar, glob%icoords(1:nihvar), oldc, &
       glob%igradient(1:nihvar), oldgrad, glob%ihessian, &
       glob%havehessian, fracrec, was_updated)
  ! issue: since was_updated is not used any more, oldgrad will be wrong for too large minstep
  
  if(.not.glob%havehessian) then

    if(.not. fd_hess_running) then
      ! test for convergence before calculating the Hessian
      ! skip that if Hessian calculation is the main task
      if(glob%iopt/=11.and.glob%iopt/=12) then
        call convergence_test(stat%ccycle,.true.,tconv)
        if(tconv) return
      end if

      if (glob%inithessian == 4) then
         ! Initial Hessian is the identity matrix
         glob%ihessian = 0.0d0
         do i = 1, nihvar
            glob%ihessian(i, i) = 1.0d0
         end do
         glob%havehessian = .true.
      end if

      if (glob%inithessian == 0) then
         if (glob%imicroiter > 0) call dlf_fail(&
              "inithessian = 0 with microiterative opt not yet implemented")
         ! try to get an analytic Hessian - care about allocation business
         ! call to an external routine ...
         call allocate(hess_loc,glob%nvar,glob%nvar)
         call dlf_get_hessian(glob%nvar,glob%xcoords,hess_loc,status)
         if(status==0.and.printl>=4) write(stdout,"('Analytic hessian calculated')")
         !switch:
         !status=1
         
         if(status==0) then
            !        write(*,"('HESS',2F10.2)") HESS_LOC
            ! convert it into internals
            call clock_start("COORDS")
            !     write(*,"('xHESS',12F10.4)") HESS_LOC
            call dlf_coords_hessian_xtoi(glob%nvar,hess_loc)
            !     write(*,"('iHESS',8F10.4)") glob%ihessian
            call clock_stop("COORDS")
            glob%havehessian=.true.
         else
            if(printl>=2) write(stdout,'(a)') &
                 "External Hessian not available, using two point FD."
         end if
         call deallocate(hess_loc)
      end if
    end if

    if(.not.glob%havehessian) then

      if (glob%inithessian == 3) then
         ! Simple diagonal Hessian a la MNDO
         call dlf_diaghessian(nihvar, glob%energy, glob%icoords(1:nihvar), &
              glob%igradient(1:nihvar), glob%ihessian, glob%havehessian)
      else
         ! Finite Difference Hessian calculation in internal coordinates
         ! dlf_fdhessian writes fd_hess_running
         call dlf_fdhessian(nihvar, fracrec, glob%energy, glob%icoords(1:nihvar), &
              glob%igradient(1:nihvar), glob%ihessian, glob%havehessian)
      end if

      ! check if FD-Hessian calculation currently running
      trerun_energy=(fd_hess_running) 
      if(trerun_energy) then
        call clock_start("COORDS")
        call dlf_coords_itox(iimage)
        call clock_stop("COORDS")
      end if

    end if

    if(glob%havehessian) then
      ! Print out Hessian if running Hessian evaluation only or if debugging
      if (printl>=6 .or. (glob%iopt==11 .and. printl>=2)) then
        write(stdout,"(/,'Symmetrised Hessian matrix (DL-FIND coordinates):')")
        call dlf_matrix_print(nihvar, nihvar, glob%ihessian)
      end if
      ! Determine eigenvalues and eigenvectors
      call dlf_matrix_diagonalise(nihvar,glob%ihessian,eigval,eigvec)
      if (printl>=4 .or. (glob%iopt==11 .and. printl>=2)) then      
         write(stdout,"(/,'Hessian eigenvalues:')")
         write(stdout,"(12f9.5)") eigval
      end if
   end if

 end if

end subroutine dlf_makehessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_diaghessian
!!
!! FUNCTION
!!
!! Calculate a simple diagonal Hessian by one-point finite difference,
!! much like the default behaviour of the standard MNDO optimiser.
!!
!! Fracrecalc is not applicable here as it makes no sense to partially update
!! an already available Hessian with a diagonal Hessian
!!
!! SYNOPSIS
subroutine dlf_diaghessian(nvar_,energy,coords,gradient,hess,havehessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_hessian
  implicit none
  !
  integer,intent(in)     :: nvar_ ! Number of variables 
  real(rk),intent(inout) :: energy          ! at the 2nd step out
  real(rk),intent(inout) :: coords(nvar_)   ! always changed
  real(rk),intent(inout) :: gradient(nvar_) ! at the 2nd step out
  real(rk),intent(inout) :: hess(nvar_,nvar_)
  logical,intent(out)    :: havehessian
  integer                :: i
  real(rk)               :: dx
  real(rk)               :: svar
! **********************************************************************

  if(.not.fd_hess_running) then
    ! First step - initialise
     if(printl >= 4) then
        write(stdout,'(A)') &
             "Finite-difference calculation for a diagonal Hessian"
     end if

     ! backup energy and gradient so it can be restored after Hessian calc
     storeenergy = energy
     storegrad(1:nvar_) = gradient(:)

     hess = 0.0d0
     ! move to the 2nd point, backwards along the gradient
     do i = 1, nvar_
        coords(i) = coords(i) - sign(glob%delta, gradient(i))
     end do

     fd_hess_running = .true.
     havehessian = .false.
   
     if(printl >= 4) then
        write(stdout,'("Delta : ",es10.2)') glob%delta
     end if

     return
  end if ! End of initialisation step


  ! Second step - calculate diagonal Hessian
  if(printl>=4) then
    write(stdout,"('Finite-difference diagonal Hessian: 2nd point')")
    write(stdout,"('Energy difference to 1st point:        ',es10.2,' H')") energy-storeenergy
    svar=sqrt(sum((gradient(:)-storegrad(1:nvar_))**2))
    write(stdout,"('Abs. Gradient difference to 1st point: ',es10.2)") svar
  end if

  ! calculate Hessian and restore coordinates
  do i = 1, nvar_
     dx = sign(glob%delta, storegrad(i))
     hess(i, i) = (storegrad(i) - gradient(i)) / dx
     ! Hessian should be positive
     if (hess(i, i) < 0.0d0) hess(i, i) = abs(storegrad(i)) / glob%delta
     ! Minimum threshold set to identity matrix element
     hess(i, i) = max(hess(i, i), 1.0d0)
     coords(i) = coords(i) + dx   
  end do

  ! restore energy and gradient
  energy = storeenergy
  gradient(:) = storegrad(1:nvar_)
  
  fd_hess_running=.false.
  havehessian=.true.

end subroutine dlf_diaghessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_fdhessian
!!
!! FUNCTION
!!
!! Calculate the Hessian by finite differencing.
!!
!! One-point and Two-point formulas are available.
!!
!! If Fracrecalc is true, only part of the Hessian is recalculated:
!! The Hessian is expressed in its eigenmodes. Finite-difference elongations
!! along the lowest eigenmodes are calculated. The gradient is transformed
!! back into the eigenmode-space. Finally, the Hessian is transformed back
!! into coordinate space. Frac_recalc is only used if a Hessian is
!! already available, and if do_partial_fd (in _update) is true.
!!
!! SYNOPSIS
subroutine dlf_fdhessian(nvar_,fracrecalc,energy,coords,gradient,hess,havehessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_allocate, only: allocate, deallocate
  use dlf_hessian
  implicit none
  !
  integer,intent(in)     :: nvar_ ! Number of variables 
  logical,intent(inout)  :: fracrecalc ! Recalculate only fraction
  real(rk),intent(inout) :: energy          ! at the last step out
  real(rk),intent(inout) :: coords(nvar_)   ! always changed
  real(rk),intent(inout) :: gradient(nvar_) ! at the last step out
  real(rk),intent(inout) :: hess(nvar_,nvar_)
  logical,intent(out)    :: havehessian
  integer                :: status ! parallel FD check
  integer                :: fail,ivar,jvar,taskfarm_mode
  real(rk)               :: svar
  real(rk),allocatable   :: tmpmat(:,:)
! **********************************************************************
  if(.not.fd_hess_running) then
    ! First step - initialise
    if(fracrecalc) then

      ! all negative, "zero", and the first positive (i.e. > soft) modes
      ! will be recalculated.

      ! Diagonalise the present Hessian
      call dlf_matrix_diagonalise(nvar_,hess,eigval(1:nvar_),eigvec(1:nvar_,1:nvar_))
      !write(stdout,"('eigval-pa',12f10.4)") eigval

      do ivar=1,nvar_
        if(eigval(ivar)>max(soft,0.D0)) exit
      end do

      if(printl >= 4) then
        write(stdout,'(A,i3,a)') "Finite-difference calculation of the", &
            ivar," lowest Hessian eigenmodes"
      end if
      numfd = ivar
    else
      if(printl >= 4) then
        write(stdout,'(A)') &
            "Finite-difference calculation of the whole Hessian"
      end if
      numfd = nvar_
    end if
    iivar=1
    direction=1 ! do first 1, then -1

    ! Task-farming allocation based on iivar
    ! For full Hessian calculations only 
    ! (standalone Hessian calcs in mind rather than optimisations)
    ! To extend to fracrecalc case, would need to consider how to deal
    ! with non-zero initial Hessian when sharing data
    call dlf_qts_get_int("TASKFARM_MODE",taskfarm_mode)
    if (.not. fracrecalc .and. glob%ntasks > 1.and.taskfarm_mode==1) then
       glob%dotask = (mod(iivar,glob%ntasks) == glob%mytask)
    end if

    storeenergy=energy
    ! keep gradient and restore it when Hessian has finished
    storegrad(1:nvar_)=gradient(:)

    fd_hess_running=.true.

    if(fracrecalc) then
      hess=0.D0
      do ivar=1,nvar_
        hess(ivar,ivar)=eigval(ivar)
      end do

      ! do the step
      coords(:)=coords(:) + glob%delta * eigvec(1:nvar_,1)

    else
      hess=0.D0

      ! do the step
      coords(1)=coords(1)+glob%delta

    end if

    

    havehessian=.false.

    if(printl >= 4) then
      write(stdout,'("Delta : ",es10.2)') glob%delta
    end if

    return
    
  end if !(.not.fd_hess_running)

  if(printl>=4 .and. glob%dotask) then
    write(stdout,"('Finite-difference Hessian: variable ',i4,'/',i4,&
        &' direction=',i2)") iivar,numfd,direction
  end if
  if(printl>=3 .and. glob%dotask) then
    write(stdout,"('Energy difference to midpoint:        ',es10.2,' H')") energy-storeenergy
    svar=sqrt(sum((gradient(:)-storegrad(1:nvar_))**2))
    write(stdout,"('Abs. Gradient difference to midpoint: ',es10.2)") svar
  end if

  ! The values of iivar and direction are those for which the gradient is currently available

  if(iivar<numfd.or. (twopoint.and.direction==1) ) then
    ! general step in the course of Hessian calculation

    if(direction==1.and.twopoint) then
      if (glob%dotask) then
        hess(iivar,:)=gradient(:)
      end if
      direction=-1
      if(fracrecalc) then
        coords(:)=coords(:) - 2.D0 * glob%delta * eigvec(1:nvar_,iivar)
      else
        coords(iivar)=coords(iivar)-2.D0*glob%delta
      end if
    else
      if(twopoint) then
        !set back coordinates
        if(fracrecalc) then
          gradient(:)=(hess(iivar,:)-gradient(:))/(2.D0*glob%delta)
          call dlf_matrix_multiply(1,nvar_,nvar_,1.D0,gradient,eigvec(1:nvar_,1:nvar_), &
               0.D0,hess(iivar,:))
          coords(:)=coords(:) + glob%delta * eigvec(1:nvar_,iivar)
        else
          if (glob%dotask) then
            hess(iivar,:)=(hess(iivar,:)-gradient(:))/(2.D0*glob%delta)
          end if
          coords(iivar)=coords(iivar)+glob%delta
        end if
        direction=1
      else
        if(fracrecalc) then
          gradient(:)=(gradient(:)-storegrad(1:nvar_)) / glob%delta
          call dlf_matrix_multiply(1,nvar_,nvar_,1.D0,gradient,eigvec(1:nvar_,1:nvar_), &
               0.D0,hess(iivar,:))
          coords(:)=coords(:) - glob%delta * eigvec(1:nvar_,iivar)
        else
          if (glob%dotask) then 
            hess(iivar,:)=(gradient(:)-storegrad(1:nvar_)) / glob%delta
          end if
          coords(iivar)=coords(iivar)-glob%delta
        end if
      end if
      iivar=iivar+1
      ! Task-farming: next allocation
      call dlf_qts_get_int("TASKFARM_MODE",taskfarm_mode)
      if (.not. fracrecalc .and. glob%ntasks > 1.and.taskfarm_mode==1) then
        glob%dotask = (mod(iivar,glob%ntasks) == glob%mytask)
      end if
      if(fracrecalc) then
        coords(:)=coords(:) + glob%delta * eigvec(1:nvar_,iivar)
      else
        coords(iivar)=coords(iivar)+glob%delta
      end if
    end if
      
    havehessian=.false.
      
  else !(iivar<numfd.or. (twopoint.and.direction==1) )
    ! final step: calculate Hessian and deallocate
    !set back coordinates
    if(twopoint) then
      if(fracrecalc) then
        gradient(:)=(hess(iivar,:)-gradient(:))/(2.D0*glob%delta)
        call dlf_matrix_multiply(1,nvar_,nvar_,1.D0,gradient,eigvec(1:nvar_,1:nvar_), &
             0.D0,hess(iivar,:))
        coords(:)=coords(:) + glob%delta * eigvec(1:nvar_,iivar)
      else
        if (glob%dotask) then 
           hess(iivar,:)=(hess(iivar,:)-gradient(:))/(2.D0*glob%delta)
        end if
        coords(iivar)=coords(iivar)+glob%delta
      end if
    else
      if(fracrecalc) then
        gradient(:)=(gradient(:)-storegrad(1:nvar_)) / glob%delta
        call dlf_matrix_multiply(1,nvar_,nvar_,1.D0,gradient,eigvec(1:nvar_,1:nvar_), &
             0.D0,hess(iivar,:))
        coords(:)=coords(:) - glob%delta * eigvec(1:nvar_,iivar)
      else
        if (glob%dotask) then 
          hess(iivar,:)=(gradient(:)-storegrad(1:nvar_)) / glob%delta
        end if
        coords(iivar)=coords(iivar)-glob%delta
      end if
    end if

    ! Task-farming: check no gradient evaluations in other workgroups failed
    ! then share Hessian data
    call dlf_qts_get_int("TASKFARM_MODE",taskfarm_mode)
    if (.not. fracrecalc .and. glob%ntasks > 1.and.taskfarm_mode==1) then
       ! If it has got here all calcs on this workgroup have succeeded
       status = 0
       call dlf_tasks_int_sum(status, 1)
       if (status > 0) then
          call dlf_fail("Task-farmed gradient evaluations failed")
       end if
       call dlf_tasks_real_sum(hess, nvar_*nvar_)
    end if

    ! Symmetrise the Hessian
    if(fracrecalc) then

      ! build up the Hessian

      ! symmetrise
      do ivar=1,numfd
        do jvar=ivar+1,nvar_
          if(jvar<=numfd) then
            hess(ivar,jvar)=0.5D0*(hess(ivar,jvar)+hess(jvar,ivar))
            hess(jvar,ivar)=hess(ivar,jvar)
          else
            hess(jvar,ivar)=hess(ivar,jvar)
          end if
        end do
      end do

      ! Multiply with the eigenvectors to restore the real Hessian
      call allocate(tmpmat,nvar_,nvar_)
      eigvec(1:nvar_,1:nvar_) = transpose(eigvec(1:nvar_,1:nvar_))
      call dlf_matrix_multiply(nvar_,nvar_,nvar_,1.D0,hess,eigvec(1:nvar_,1:nvar_),0.D0,tmpmat)
      eigvec(1:nvar_,1:nvar_) = transpose(eigvec(1:nvar_,1:nvar_))
      call dlf_matrix_multiply(nvar_,nvar_,nvar_,1.D0,eigvec(1:nvar_,1:nvar_),tmpmat,0.D0,hess)
      call deallocate(tmpmat)
      
    else
      do ivar=1,numfd
        do jvar=ivar+1,numfd
          hess(ivar,jvar)=0.5D0*(hess(ivar,jvar)+hess(jvar,ivar))
          hess(jvar,ivar)=hess(ivar,jvar)
        end do
      end do
    end if

    ! restore energy and gradient
    energy=storeenergy
    gradient(:)=storegrad(1:nvar_)

    fd_hess_running=.false.
    havehessian=.true.
    call dlf_qts_get_int("TASKFARM_MODE",taskfarm_mode)
    if(taskfarm_mode==1) glob%dotask = .true.

  end if ! (iivar<numfd.or. (twopoint.and.direction==1) )

end subroutine dlf_fdhessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_test_delta
!!
!! FUNCTION
!!
!! Test various different values of delta for calculating a Hessian by
!! a two-point finite-difference of the gradients.
!!
!! For numdelta=9 (hardcoded), 19 Energy and gradient calculations are
!! performed. The elongations are always calculated for the first
!! element in the coordinates array. The different values for the
!! first diagonal element of the resulting Hessian are plotted at the
!! end.
!!
!! This routine is called if iopt=9
!!
!! SYNOPSIS
subroutine dlf_test_delta(trerun_energy)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_hessian
  implicit none
  logical,intent(out):: trerun_energy
  integer,parameter :: numdelta=9 ! hardcoded for the time being
  real(rk),save     :: dder(numdelta)
  real(rk)          :: delta(numdelta)
  integer           :: ivar
  real(rk)          :: svar
  logical           :: tok
! **********************************************************************

  if(.not.allocated(storegrad)) then
    call dlf_fail("Hessian module must be initiated when dlf_test_delta is called")
  end if

  if (glob%imicroiter > 0) then
     call dlf_fail('dlf_test_delta not yet compatible with microiterative PRFO')
  end if

  trerun_energy=.true.
  ! set the values of delta to try out
  delta(1)=0.0001D0
  delta(2)=0.0002D0
  delta(3)=0.0005D0
  do ivar=4,numdelta
    delta(ivar)=delta(ivar-3)*10.D0
  end do

  if(.not.fd_hess_running) then
    ! First step - initialise

    iivar=1
    direction=1 ! do first 1, then -1
    storeenergy=glob%energy
    ! keep gradient and restore it when Hessian has finished
    storegrad(:)=glob%igradient(:)

    fd_hess_running=.true.

    dder=0.D0
    ! do first step
    glob%icoords(1)=glob%icoords(1)+delta(1)

  else !(.not.fd_hess_running)

    ! intermediate steps

    if(printl>=4) then
      write(stdout,"('Finite-difference Hessian test calculation ',i4,'/',i4,&
          &' direction=',i2)") iivar,numdelta,direction
    end if
    if(printl>=3) then
      write(stdout,'("Delta :                               ",es10.2)') delta(iivar)
      write(stdout,"('Energy difference to midpoint:        ',es10.2,' H')") glob%energy-storeenergy
      svar=sqrt(sum((glob%igradient(:)-storegrad(:))**2))
      write(stdout,"('Abs. Gradient difference to midpoint: ',es10.2)") svar
    end if

    ! The values of iivar and direction are those for which the gradient is currently available

    if(iivar<numdelta.or. direction==1 ) then
      ! general step in the course of Hessian calculation

      if(direction==1) then
        dder(iivar)=glob%igradient(1)
        direction=-1
        glob%icoords(1)=glob%icoords(1)-2.D0*delta(iivar)
      else
        !set back coordinates
        dder(iivar)=(dder(iivar)-glob%igradient(1))/(2.D0*delta(iivar))
        if(printl>=4) then
          write(stdout,'("Delta: ",es10.3," First diagonal element of the &
              &Hessian: ",es20.13)') delta(iivar),dder(iivar)
        end if
        glob%icoords(1)=glob%icoords(1)+delta(iivar)

        direction=1
        iivar=iivar+1

        glob%icoords(1)=glob%icoords(1)+delta(iivar)
      end if
    else ! (iivar<numdelta.or. direction==1 )
      ! final step: calculate Hessian and deallocate
      !set back coordinates
      dder(iivar)=(dder(iivar)-glob%igradient(1))/(2.D0*delta(iivar))
      glob%icoords(1)=glob%icoords(1)+delta(iivar)

      ! restore energy and gradient
      glob%energy=storeenergy
      glob%igradient(:)=storegrad(:)

      fd_hess_running=.false.

      !havehessian=.true.
      trerun_energy=.false.

      ! print report!
      if(printl>=2) then
        write(stdout,*) "Delta    First diagonal element of the Hessian"
        do ivar=1,numdelta
          write(stdout,'(es10.3,es20.13)') delta(ivar),dder(ivar)
        end do
      end if
      
    end if ! (iivar<numdelta.or. direction==1 )
      
  end if ! (.not.fd_hess_running)

  ! transform coordinates back to xcoords
  call dlf_direct_itox(glob%nvar,glob%nivar,glob%nicore,glob%icoords,glob%xcoords,tok)
  if(.not.tok.and. mod(glob%icoord,10)>0 .and. mod(glob%icoord,10)<=4 ) then
    call dlf_fail('HDLC coordinate breakdown')
  end if

end subroutine dlf_test_delta
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_hessian_update
!!
!! FUNCTION
!!
!! Update the global Hessian
!!
!! If we do not have a Hessian, there is nothing to update. Return.
!!
!! If we have one, and are allowed to update it, update it. If too many
!!  updates have been made, destroy and recalculate it.
!!
!! COMMENTS
!!
!! The gradient, coordinates and hessian are passed to and from the
!! routine through the arguments and are therefore independent of the
!! data stored in glob. This is useful for the Lagrange-Newton method
!! where the optimiser gradient is not necessarily the same as the 
!! gradient used for the update of the Hessian.
!!
!! INPUTS
!!
!! nvar - dimensions of gradient, coordinates, Hessian
!! coords(nvar) - current coordinates
!! oldcoords(nvar) - coordinates of previous step
!! gradient(nvar) - current gradient
!! oldgradient(nvar) - gradient of previous step
!! havehessian - if false return as no update is possible
!! fracrecalc
!! glob%update - specifies which update algorithm to use
!! glob%maxupd - check that max no. of updates is not exceeded
!! glob%nivar - check that nvar is consistent
!! glob%icoord - for check of consistency of nvar
!! (dlf_hessian) fd_hess_running
!! (dlf_hessian) iupd - no. of updates since last reset
!!
!! OUTPUTS
!! 
!! hess(nvar, nvar) - the updated Hessian in internal coordinates
!! havehessian - false if no update requested or max updates reached
!! fracrecalc
!! (dlf_hessian) iupd - no. of updates since last reset
!!
!! SYNOPSIS
subroutine dlf_hessian_update(nvar, coords, oldcoords, gradient, &
     oldgradient, hess, havehessian, fracrecalc, was_updated)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout, stderr, printl
  use dlf_hessian
  implicit none
  integer,intent(in)    :: nvar ! used for temporary storage arrays
  real(rk),intent(in) :: coords(nvar)   
  real(rk),intent(in) :: oldcoords(nvar)
  real(rk),intent(in) :: gradient(nvar)
  real(rk),intent(in) :: oldgradient(nvar)
  real(rk),intent(inout) :: hess(nvar,nvar)
  logical,intent(inout) :: havehessian
  logical,intent(inout) :: fracrecalc ! recalculate fraction of Hessian
  logical,intent(out)   :: was_updated ! at return: was the Hessian updated here?
  ! temporary arrays
  real(rk) :: fvec(nvar),step(nvar), tvec(nvar)
  real(rk) :: fx_xx,xx,svar,bof, dds, ddtd
  real(RK) ,external :: ddot
  integer  :: ivar,jvar
  logical,parameter :: do_partial_fd=.false. ! Current main switch!
! **********************************************************************

  was_updated=.false.

  if(.not.fd_hess_running) fracrecalc=.false.
  if(.not.havehessian.or.glob%update==0) then
    havehessian=.false.
    return
  end if

  ! Check for maximum number of updates reached
  ! In case of partial finite-difference, do an update first, then return and 
  ! Recalculate the lower modes
  if(iupd>=glob%maxupd .and. .not. do_partial_fd) then
    havehessian=.false.
    iupd=0
    hess = -1.D0
    fracrecalc=.false.
    return
  end if

  if (glob%icoord >= 10 .and. glob%icoord <= 19) then
    if (nvar /= glob%nivar - 2) &
        call dlf_fail("Inconsistent Lagrange-Newton nvar in dlf_hessian_update")
  elseif(glob%icoord == 190 ) then
    continue
  elseif(glob%imicroiter > 0) then
     if (nvar /= glob%nicore) &
        call dlf_fail("Inconsistent microiterative nvar in dlf_hessian_update")
  elseif(nvar/=glob%nivar) then
      call dlf_fail("Inconsistent nvar in dlf_hessian_update")
  endif

  ! Useful variables for updating
  fvec(:) = gradient(:) - oldgradient(:)
  step(:) = coords(:) - oldcoords(:)

  xx=ddot(nvar,step,1,step,1)
  if(xx <= minstep ) then
    if(printl>=2) write(stdout,"('Step too small. Skipping hessian update')")
    return
  end if

  iupd=iupd+1

  if(printl>=3) then
     select case (glob%update)
     case(1)
        write(stdout,"('Updating Hessian with the Powell update, No ',i5)") iupd
     case(2)
        write(stdout,"('Updating Hessian with the Bofill update, No ',i5)") iupd
     case(3)
        write(stdout,"('Updating Hessian with the BFGS update, No ',i5)") iupd
     end select
  end if
    
  select case (glob%update)
  case(1,2)
     ! Powell/Bofill updates

     ! fvec = fvec - hessian x step
     call dlf_matrix_multiply(nvar,1,nvar,-1.D0, hess,step,1.D0,fvec)

     fx_xx=ddot(nvar,fvec,1,step,1) / xx

     if(glob%update==2) then
        svar=ddot(nvar,fvec,1,fvec,1)
        bof=fx_xx**2 * xx / svar
        if(printl>=6) write(stdout,'("Bof=",es10.3)') bof
     end if

     do ivar=1,nvar
        do jvar=ivar,nvar

           ! Powell
           svar=fvec(ivar)*step(jvar) + step(ivar)*fvec(jvar) - fx_xx* step(ivar)*step(jvar)

           if(glob%update==2) then
              ! Bofill
              svar=svar*(1.D0-bof) + bof/fx_xx * fvec(ivar)*fvec(jvar)
           end if

           hess(ivar,jvar) = hess(ivar,jvar) + svar/xx
           hess(jvar,ivar) = hess(ivar,jvar)
        end do
     end do
     
  case(3)
     ! BFGS update

     ! tvec is hessian x step
     tvec = 0.0d0
     call dlf_matrix_multiply(nvar, 1, nvar, 1.0d0, hess, step, 0.0d0, tvec)

     dds = ddot(nvar, fvec, 1, step, 1)
     ddtd = ddot(nvar, step, 1, tvec, 1)

     do ivar = 1, nvar
        do jvar = ivar, nvar
           svar = (fvec(ivar) * fvec(jvar)) / dds - &
                  (tvec(ivar) * tvec(jvar)) / ddtd
           hess(ivar, jvar) = hess(ivar, jvar) + svar
           hess(jvar, ivar) = hess(ivar, jvar)
        end do
     end do

  case default
     ! Update mechanism not recognised
     write(stderr,*) "Hessian update", glob%update, "not implemented"
     call dlf_fail("Hessian update error")

  end select

  was_updated=.true.

  ! Check for maximum number of updates reached
  ! In case of partial finite-difference, do an update first, then return and 
  ! Recalculate the lower modes
  if(iupd>=glob%maxupd .and. do_partial_fd) then
    havehessian=.false.
    iupd=0
    fracrecalc=.true.

  end if

end subroutine dlf_hessian_update
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_prfo_step
!!
!! FUNCTION
!!
!! Calculate a P-RFO step
!! This routine does only require the current Hessian and gradient,
!! not an old one.
!!
!! Notes to the Hessian update:
!! Paul updates some modes of the Hessian, whenever the number of positive 
!! eigenvalues (non-positive is below "soft") changes. He recalculates all
!! negative and soft modes plus one.
!!
!! SYNOPSIS
subroutine dlf_prfo_step(nvar,coords,gradient,hessian,step)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl,stderr,glob,pi
  use dlf_hessian
  implicit none
  integer ,intent(in)   :: nvar 
  real(rk),intent(in)   :: coords(nvar)
  real(rk),intent(in)   :: gradient(nvar)
  real(rk),intent(in)   :: hessian(nvar,nvar)
  real(rk),intent(out)  :: step(nvar)
  !
  ! these may be made variable in the future:
  integer   :: maxit=100 ! maximum number of iterations to find lambda
!  real(rk)  :: tol=1.D-5 ! tolerance in iterations
  real(rk)  :: tol=1.D-6 ! tolerance in iterations
  real(rk)  :: bracket=1.D-4  ! threshold for using bracket scheme - why not 0.D0?
  real(rk)  :: delta=0.05D0, big=1000.D0 ! for bracketing 
  !
  real(rk)              :: lamts,lamu,laml,lam
  real(rk)              :: ug(nvar) ! eigvec * gradient
  real(rk)              :: tmpvec(nvar)
  integer               :: ivar,iter,nmode,maxsoft
  real(rk)              :: svar,lowev,maxov
  real(rk)              :: soft_tmp,ev2(nvar)
  real(rk)              :: CM_INV_FOR_AMU
  real(RK) ,external    :: ddot
  logical               :: conv=.false.,err=.false.
  logical               :: skipmode(nvar)
  logical :: dbg=.true.
  real(8)               :: lam_thresh
! **********************************************************************
  
  maxsoft=glob%nzero
  ! try:
  if(glob%icoord==190) then
    tol=1.D-13
    bracket=1.D-6
    delta=1.D-3
    maxit=1000
  end if


  if(.not.glob%havehessian) call dlf_fail("No Hessian present in P-RFO") !??

  call dlf_matrix_diagonalise(nvar,hessian,eigval,eigvec)

  if(printl >= 2) then
    write(stdout,"('Eigenvalues of the Hessian:')") 
    write(stdout,"(9f10.4)") eigval
  end if

  ! Determine mode to follow
  if(follow==0) then
    tsmode=1
  else if(follow==1) then
    call dlf_fail("Hessian mode following 1 not implemented")
  else if(follow==2.or.follow==3) then
    if(tsvectorset) then
      maxov= dabs(ddot(nvar,eigvec(:,tsmode),1,tsvector,1))
      nmode=tsmode
      if(printl>=4) write(stdout,"('Overlap of current TS mode with &
          &previous one ',f6.3)") maxov
      do ivar=1,nvar
        if(ivar==tsmode) cycle
        svar= dabs(ddot(nvar,eigvec(:,ivar),1,tsvector,1))
        if(svar > maxov) then
          if(printl>=6) write(stdout,"('Overlap of mode',i4,' with &
              & TS-vector is ',f6.3,', larger than TS-mode',f6.3)") &
              ivar,svar,maxov
          maxov=svar
          nmode=ivar
        end if
      end do
      if(nmode /= tsmode) then
        !mode switching!
        if(printl>=2) write(stdout,"('Switching TS mode from mode',i4,&
            &' to mode',i4)") tsmode,nmode
        tsmode=nmode
        if(printl>=4) write(stdout,"('Overlap of current TS mode with &
            &previous one ',f6.3)") maxov
      end if
      if(follow==3) tsvector=eigvec(:,tsmode)
    else
      ! first step: use vector 1
      tsvector(:)=eigvec(:,1)
      tsvectorset=.true.
    end if
  else
    write(stderr,"('Wrong setting of follow:',i5)") follow
    call dlf_fail("Hessian mode following wrong")
  end if

  ! print frequency in case of mass-weighted coordinates
  if(glob%massweight  .and.(.not.glob%icoord==190) .and.printl>=2) then
    ! sqrt(H/u)/a_B/2/pi/c / 100
    !svar=sqrt( 4.35974417D-18/ 1.66053886D-27 ) / ( 2.D0 * pi * &
    !    0.5291772108D-10 * 299792458.D0) / 100.D0
    !call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMU)
    !svar=sqrt(abs(eigval(tsmode))) * CM_INV_FOR_AMU
    !if(eigval(tsmode)<0.D0) svar=-svar
    !write(stdout,"('Frequency of transition mode',f10.3,' cm^-1 &
    !    &(negative value denotes imaginary frequency)')") &
    !    svar
    call dlf_print_wavenumber(eigval(tsmode),.true.)
  end if

  call dlf_formstep_set_tsmode(nvar,11,eigvec(:,tsmode))

  ! calculate eigvec*gradient
  do ivar=1,nvar
    ug(ivar) = ddot(nvar,eigvec(:,ivar),1,gradient(:),1)
  end do

  ! calculate Lambda that minimises along the TS-mode:
  lamts=0.5D0 * ( eigval(tsmode) + dsqrt( eigval(tsmode)**2 + 4.D0 * ug(tsmode)**2) )

  if(printl >= 2 .or. dbg) then
    write(stdout,'("Lambda for maximising TS mode:     ",es12.4," Eigenvalue:",es12.4)') lamts,eigval(tsmode)
  end if

  ! Calculate the number of modes considered "soft"
  if (glob%imicroiter > 0 .and. maxsoft > 0 .and. printl >= 2) then
     write(stdout,'("Warning: nzero > 0 is not appropriate for microiterative optimisation")')
     write(stdout,'("nzero=0 recommended to allow core region to rotate/translate in environment.")')
  end if
  nmode=0
  soft_tmp=soft
  ev2=0.D0
  do ivar=1,nvar
    if(ivar==tsmode) cycle
    if(abs(eigval(ivar)) < soft ) then
      nmode=nmode+1
      ev2(ivar)=dabs(eigval(ivar))
    end if
  end do

  ! Check that at most 6 modes are considered "soft"
  if(nmode>maxsoft) then
    do ivar=nmode-1,maxsoft,-1
      soft_tmp=maxval(ev2)
      ev2(maxloc(ev2))=0.D0
    end do
    if(printl>=4) write(stdout,'("Criterion for soft modes tightened to &
        &",es12.4)') soft_tmp
    ! recalculate nmode
    nmode=0
    do ivar=1,nvar
      if(ivar==tsmode) cycle
      if(abs(eigval(ivar)) < soft_tmp ) then
        nmode=nmode+1
        if(printl>=4) write(stdout,'("Mode ",i4," considered soft")') ivar
      end if
    end do
  end if

  if(nmode>0.and.printl>=2) &
      write(stdout,'("Ignoring ",i3," soft modes")') nmode

  ! find lowest eigenvalue that is not TS-mode and not soft
  !   i.e. the lowest eigenmode that is minimised
  do ivar=1,nvar
    if(ivar==tsmode) cycle
    if(abs(eigval(ivar)) < soft_tmp ) cycle
    lowev=eigval(ivar)
    exit
  end do

  ! define skipmode
  skipmode(:)=.true.
  do ivar=1,nvar
    if(ivar==tsmode) cycle
    if(abs(eigval(ivar)) < soft_tmp ) cycle
    ! instead of the above line: modes 2-7 soft
    !  if(ivar>=2.and.ivar<=7) cycle
    !  print*,"Modes 2 to 7 soft"
    !<<<<
    skipmode(ivar)=.false.
  end do

  lamu=0.D0
  laml=0.D0
  lam=0.D0

  if(lowev < bracket) then
    lam=lowev-delta
    lamu=lowev
    laml=-big
  end if

  do iter=1,maxit

    svar=0.D0

    do ivar=1,nvar
      if(ivar==tsmode) cycle
      !if(abs(eigval(ivar)) < soft_tmp ) cycle
      if(skipmode(ivar)) cycle
      if(abs(lam - eigval(ivar)) > 1.D-14) then
        svar=svar+ ug(ivar)**2 / (lam - eigval(ivar) )
      end if
    end do

    if(abs(svar-lam) < tol) then
      ! we are converged

      if(lam>lowev) then
        print*,"Lambda > lowest non-TS eigenvalue, bad Hessian?"
        err=.true.
      end if

      if(lam>0.D0 .and. lowev>0.D0) then
        print*,"Lambda and lowest non-TS eigenvalue >0. Bad Hessian?"
        print*,"Lambda:",lam
        !err=.true.
      end if

      if(dbg.and..not.err) then
        print*,"Lambda converged in",iter,"iterations"
      end if
      
      conv=.true.

      exit

    end if

    ! we are not converged. Next iteration:

    !write(*,'("A",4f15.8)') svar,lam,lamu,laml
    if(lowev < bracket ) then
      if(svar < lam) lamu=lam
      if(svar > lam) laml=lam
      if(laml > -big) then
        lam=0.5D0 * (lamu + laml)
      else
        lam = lam-delta
      end if
    else
      lam=svar
    end if
    !write(*,'("B",4f15.8)') svar,lam,lamu,laml

  end do

  !if(.not.conv) call dlf_fail("P-RFO loop not converged")
  if(err) call dlf_fail("P-RFO error")

  if(printl >= 2 .or. dbg) &
      write(stdout,'("Lambda for minimising other modes: ",es12.4)') lam

  ! calculate step:
  step=0.D0
  do ivar=1,nvar
    if(ivar==tsmode) then
      if( abs(lamts-eigval(ivar)) < 1.D-5 ) then
        ug(ivar)=1.D0
      else
        ug(ivar)=ug(ivar) / (eigval(ivar) - lamts)
      end if
    else
      !if(abs(eigval(ivar)) < soft_tmp ) then
      if(skipmode(ivar) ) then
        if(printl>=4) write(stdout,'("Mode ",i4," ignored, as &
            &|eigenvalue| ",es10.3," < soft =",es10.3)') &
            ivar,eigval(ivar),soft_tmp
        cycle
      end if
      if(glob%icoord==190) then
        lam_thresh=1.D-10
      else
        lam_thresh=1.D-5
      end if
      if( abs(lam-eigval(ivar)) < lam_thresh ) then 
        print*,"WARNING: lam-eigval(ivar) small for non-TS mode",ivar,"!"
        ug(ivar)= -1.D0 / eigval(ivar) ! take a newton-raphson step for this one
      else
        ug(ivar)=ug(ivar) / (eigval(ivar) - lam)
      end if
    end if
    if(printl>=6) write(stdout,'("Mode ",i4," Length ",es10.3)') ivar,ug(ivar)
    step(:) = step(:) - ug(ivar) * eigvec(:,ivar)
  end do

  if(printl >= 2 .or. dbg) &
      write(stdout,'("P-RFO step length:                 ",es12.4)') sqrt(sum(step(:)**2))

end subroutine dlf_prfo_step
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep_set_tsmode
!!
!! FUNCTION
!!
!! Write the transition mode to a file
!!
!! mode:
!!   -2   deallocate all arrays
!!   -1   coords contains only energy (nvar=1)
!!   00   TS structure in x-coordinates
!!   01   TS structure in i-coordinates (similar x-coords have to be provided priorly)
!!   10   TS mode relative to TS structure in x-coordinates
!!   11   TS mode relative to TS structure in i-coordinates
!!   20   TS mode absolute in x-coordinates (not yet implemented)
!!   21   TS mode absolute in i-coordinates (not yet implemented)
!!
!! This routine does not produce errors in case of wrong input,
!! it just does nothing in this case.
!!
!! SYNOPSIS
subroutine dlf_formstep_set_tsmode(nvar,mode,coords)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,printf
  use dlf_formstep_module, only: tscoords, tsmode_r, energy, tenergy, &
      tsc_ok, tsm_ok
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer ,intent(in) :: nvar
  integer ,intent(in) :: mode
  real(rk),intent(in) :: coords(nvar)
  real(rk),allocatable :: store1(:),store2(:),store3(:)
  logical              :: tok
  real(rk)             :: svar
  real(rk),external    :: ddot
  integer              :: nivarf, nicoref
! **********************************************************************
  if(glob%icoord==190) return
  if(mode==-1) then
    if(nvar/=1) return
    tenergy=.true.
    energy=coords(1)
  else if(mode==0) then
    if(allocated(tscoords)) call deallocate(tscoords)
    if(nvar/=glob%nvar) return
    call allocate(tscoords,nvar)
    tscoords=coords
    tsc_ok=.true.
  else if(mode==1) then
    if(.not.(allocated(tscoords).and.tsc_ok)) return
    ! This mode does not appear to ever be called, so I do not know
    ! how it should be modified for the microiterative case
    call dlf_direct_itox(glob%nvar,nvar,glob%nicore,coords,tscoords,tok)
    if(.not.tok) then
      call deallocate(tscoords)
      write(stdout,'(a)') "HDLC breakdown in set_tsmode ignored in mode 1"
      tsc_ok=.false.
      return
    end if
  else if(mode==10) then
    if(allocated(tsmode_r)) call deallocate(tsmode_r)
    if(nvar/=glob%nvar) return
    call allocate(tsmode_r,nvar)
    tsmode_r=coords
    tsm_ok=.true.
  else if(mode==11) then
    ! transform tscoords to icoords
    if(.not.(allocated(tscoords).and.tsc_ok)) return
    if(allocated(tsmode_r)) call deallocate(tsmode_r)
    ! nvar= number of internal coordinates
    call allocate(tsmode_r,glob%nvar)
    tsmode_r=tscoords
    call allocate(store1,glob%nvar) ! xgradient
    store1=0.D0
    ! get full region nivar (in a microiterative calc, nvar is only inner region)
    call dlf_direct_get_nivar(0, nivarf)
    call dlf_direct_get_nivar(1, nicoref)
    call allocate(store2,nivarf) ! igradient
    store2=0.D0
    call allocate(store3,nivarf) ! icoords
    store3=0.D0
    call dlf_direct_xtoi(glob%nvar,nivarf,nicoref,tscoords,store1,store3,store2)
    ! add coords
    ! make sure the relative coords are short:
    svar=ddot(nvar,coords,1,coords,1)
    store3(1:nvar)=store3(1:nvar)+coords/sqrt(svar)*0.05D0
    ! transform to x-coords
    call dlf_direct_itox(glob%nvar,nivarf,nicoref,store3,tsmode_r,tok)
    call deallocate(store1)
    call deallocate(store2)
    call deallocate(store3)
    if(.not.tok) then
      write(stdout,'(a)') "HDLC breakdown in set_tsmode ignored in mode 11"
      call deallocate(tsmode_r)
      tsm_ok=.false.
      return
    end if
    ! subtract tscoords
    tsmode_r=tsmode_r-tscoords
    tsm_ok=.true.
  else if(mode==-2) then
    if(allocated(tsmode_r)) call deallocate(tsmode_r)
    if(allocated(tscoords)) call deallocate(tscoords)
    tsc_ok=.false.
    tsm_ok=.false.
  end if
  if(tsc_ok.and.tsm_ok.and. tenergy) then
    if(printf>=2) then
      if(printl>=4) write(stdout,"('Writing TS-mode')")
      ! JMC call dlf_put_coords(glob%nvar,1,energy,tscoords,glob%iam) JMC
      ! changed the mode in this call from 1 to 3 for convenience with the
      ! castep interface; this means that for CRYSTAL and CASTEP the ts coords
      ! will not go to the general coords file (at least one of which in each
      ! case is written in append mode) but instead will go to a specific file
      ! for TS coords only, currently overwritten each time (see the routines
      ! in the interface files for more details)
      call dlf_put_coords(glob%nvar,3,energy,tscoords,glob%iam)
      if(glob%tsrelative) then
        call dlf_put_coords(glob%nvar,2,energy,tsmode_r,glob%iam) 
      else
        call dlf_put_coords(glob%nvar,2,energy,tscoords+tsmode_r,glob%iam) 
      end if
      if(printf>=3.and.glob%iam == 0) call write_xyz(32,glob%nat,glob%znuc,tsmode_r)
    end if
    !call deallocate(tscoords)
    !call deallocate(tsmode_r)
    tsc_ok=.false.
    tsm_ok=.false.
    tenergy=.false.
  end if

end subroutine dlf_formstep_set_tsmode
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep_get_ra
!!
!! FUNCTION
!!
!! Get a real-number array from the formstep module
!!
!! SYNOPSIS
subroutine dlf_formstep_get_ra(label,array_size,array,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_formstep_module, only: tscoords, tsmode_r
  implicit none
  character(*), intent(in) :: label
  integer     , intent(in) :: array_size
  real(rk)    , intent(out):: array(array_size)
  logical     , intent(out):: tok
! **********************************************************************
  tok=.false.
  if (label=="TSCOORDS") then
    if(.not.allocated(tscoords)) return
    if(size(tscoords) /= array_size) return
    array(:)=tscoords(:)
    tok=.true.
  else if (label=="TSMODE_R") then
    if(.not.allocated(tsmode_r)) return
    if(size(tsmode_r) /= array_size) return
    array(:)=tsmode_r(:)
    tok=.true.
  else
    call dlf_fail("Wrong label in dlf_formstep_get_ra")
  end if
end subroutine dlf_formstep_get_ra
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* formstep/dlf_formstep_get_logical
!!
!! FUNCTION
!!
!! Get a logical value the formstep module
!!
!! SYNOPSIS
subroutine dlf_formstep_get_logical(label,tval)
!! SOURCE
  use dlf_formstep_module, only: needhessian
  use dlf_hessian, only: fd_hess_running
  implicit none
  character(*), intent(in) :: label
  logical     , intent(out):: tval
! **********************************************************************
  if (label=="NEEDHESSIAN") then
    tval=needhessian
  else if (label=="FD_HESS_RUNNING") then
    tval=fd_hess_running
  else
    call dlf_fail("Wrong label in dlf_formstep_get_logical")
  end if
end subroutine dlf_formstep_get_logical
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_thermal
!!
!! FUNCTION
!!
!! * diagonalise the (mass-weighted) hessian
!! * calculate vibration frequencies
!! * calculate thermal and entropic contributions
!!
!! SYNOPSIS
subroutine dlf_thermal()
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl,pi,printf
  use dlf_stat, only: stat
  use dlf_constants, only : dlf_constants_get
  use dlf_hessian
  implicit none
  real(rk)        :: frequency_factor,svar
  real(rk)        :: temperature,planck,boltz,wavenumber
  real(rk)        :: speed_of_light, angstrom,hartree,avog
  real(rk)        :: evib, frequ, svib, vibtemp, zpe 
  real(rk)        :: zpetot,evibtot,svibtot,kboltz_au,svar2
  integer         :: ival, npmodes
  real(rk)        :: peigval(glob%nivar)
  real(rk)        :: ev2(glob%nivar),soft_local
  real(rk)        :: arr2(2)
  logical         :: tok
! **********************************************************************
  if (.not.glob%massweight .or. glob%icoord /= 0) then
     call dlf_fail("Thermal analysis only possible in mass-weighted &
         &cartesian coordinates")
  endif
  if (.not.glob%havehessian) then
    call dlf_fail("No hessian present for thermal analysis")
  end if
  if (glob%imicroiter > 0) then
     call dlf_fail("dlf_thermal not compatible with microiter")
  end if
  ! only act on first node, but print analysis even for printl=0
  if(printl<0) return

  ! Project out translation and rotational modes
  ! (if no atoms are frozen)
  call dlf_thermal_project(npmodes,peigval,tok)

  ! Vibrational analysis

  ! constants from NIST (2008)
  !speed_of_light=299792458.D0
  !planck=6.62606896D-34
  !boltz=1.3806504D-23
  !avog=6.02214179D23

  ! derived values
  !angstrom=0.5291772108D-10
  !hartree=4.35974417D-18

  ! sqrt(H/u)/a_B/2/pi/c / 100
  !frequency_factor=dsqrt( 4.35974417D-18/ 1.66053886D-27 ) / ( 2.D0 * pi * &
  !    angstrom * speed_of_light) / 100.D0
  !print*,"Frequ-fac",frequency_factor

  call dlf_constants_get("SOL",speed_of_light)
  call dlf_constants_get("HARTREE",hartree)
  call dlf_constants_get("PLANCK",planck)
  call dlf_constants_get("CM_INV_FOR_AMU",svar)
  call dlf_constants_get("KBOLTZ",boltz)
  call dlf_constants_get("AVOGADRO",avog)
  frequency_factor=svar
  !print*,"Frequ-fac",frequency_factor

  temperature=glob%temperature

  write(stdout,*)
  write(stdout,"('Thermochemical analysis')")
  write(stdout,"('Temperature: ',f10.2,' Kelvin')") temperature
  if (.not. tok) then
     write(stdout,"('Modes assumed to have zero vibrational frequency:',i3)") glob%nzero
  end if
  write(stdout,"(' Mode     Eigenvalue Frequency Vib.T.(K)      &
      &  ZPE (H)   Vib. Ene.(H)      - T*S (H)')")
  zpetot=0.D0
  evibtot=0.D0
  svibtot=0.D0

  soft_local = -0.1d0
  if (.not. tok) then
     ! If frozen atoms were found, trans/rot modes were not projected out
     ! Instead fall back to a user-specified number of modes to ignore (nzero)
     ev2=abs(eigval)
     do ival=1,glob%nzero
        soft_local=minval(ev2)
        ev2(minloc(ev2))=huge(1.D0)
     end do
  end if

  ! print out frequencies:
  do ival=1, npmodes
    wavenumber = sqrt(abs(peigval(ival))) * frequency_factor

    if (peigval(ival)<0.D0) then
       ! Imaginary modes
       write(stdout,"(i5,f15.10,f10.3,'i')") ival,peigval(ival),wavenumber
    else if (.not. tok .and. abs(peigval(ival))<=soft_local) then
       ! User-specified soft modes
       write(stdout,"(i5,f15.10,f10.3)") ival,peigval(ival),wavenumber
    else

      ! convert eig from cm-1 to Hz 
      frequ = wavenumber * 1.D2 * speed_of_light
      ! vibrational temperature in K
      vibtemp = frequ*planck/boltz
      ! zero point vib. (J)
      zpe = planck*frequ*0.5D0
      ! vibrational energy
      svar = dexp(vibtemp/temperature)-1.D0
      evib = planck*frequ / svar
      ! vibrational entropy
      svar = dexp(-vibtemp/temperature)
      svib = vibtemp/temperature / (1.D0/svar -1.D0)
      svib = svib - dlog(1.D0 - svar)
      svib = svib * boltz
      
      write(stdout,"(i5,f15.10,2f10.3,3f15.10)") ival,peigval(ival),wavenumber, &
          vibtemp,zpe/hartree,evib/hartree, -svib/hartree*temperature
      
      zpetot=zpetot+zpe
      evibtot=evibtot+evib
      svibtot=svibtot+svib
    end if
  end do
  write(*,"('total',35x,3f15.10)") &
      zpetot/hartree,evibtot/hartree,-svibtot/hartree*temperature
  write(*,"('total vibrational energy correction to E_electronic',&
      &f15.10,' H')") (zpetot+evibtot-svibtot*temperature)/hartree
  write(*,"('total ZPE  ',f15.5,' J/mol')") zpetot*avog
  write(*,"('total E vib',f15.5,' J/mol')") evibtot*avog
  write(*,"('total S vib',f15.5,' J/mol/K')") svibtot*avog
  
  ! write out the hessian and the energy of the "Midpoint" in qts format
  call dlf_constants_get("AMU",svar2)
  if(abs(peigval(1))>soft_local.and.peigval(1)<0.D0) then
    !
    ! we have a transition state
    !
    ! write crossover temperature for tunnelling
    call dlf_print_wavenumber(eigval(1),.false.)
    ! format as reactant (only hessian eigenvalues)
    call write_qts_reactant(glob%nat,glob%nivar,glob%energy,&
        glob%xcoords,eigval/svar2,"ts")
    arr2=-1.D0
    call write_qts_hessian(glob%nat,1,glob%nivar,-1.D0,&
        glob%energy,glob%xcoords,glob%ihessian/svar2,0.D0,arr2,"ts")
    if(printf>=4) then

      ! write an xyz file of the transition mode
      open(unit=55,file="tsmode_mov.xyz")
      call write_xyz(55,glob%nat,glob%znuc,glob%xcoords)
      svar=glob%distort
      call dlf_cartesian_xtoi(glob%nat,glob%nivar,glob%nicore,glob%massweight,glob%xcoords, &
          glob%xgradient,glob%icoords,glob%igradient)
      ! this is an abuse of glob%xcoords and glob%icoords. They now contain the moved coords...
      if(abs(svar)<1.D-7) svar=0.5D0/maxval(abs(eigvec(:,1)))
      glob%icoords=glob%icoords+svar*eigvec(:,1)
      call dlf_cartesian_itox(glob%nat,glob%nivar,glob%nicore, &
           glob%massweight,glob%icoords,glob%xcoords)

      call write_xyz(55,glob%nat,glob%znuc,glob%xcoords)
      close(55)

      ! also write to result2
      if(glob%tsrelative) then
        glob%icoords=svar*eigvec(:,1)
        call dlf_cartesian_itox(glob%nat,glob%nivar,glob%nicore,glob%massweight,glob%icoords,glob%xcoords)
        call dlf_put_coords(glob%nvar,2,glob%energy,glob%xcoords,glob%iam)         
      else
        call dlf_put_coords(glob%nvar,2,glob%energy,glob%xcoords,glob%iam) 
      end if
    end if
  else
    ! we probably have a reactant
    ! format as reactant (only hessian eigenvalues)
    call write_qts_reactant(glob%nat,glob%nivar,glob%energy,&
        glob%xcoords,eigval/svar2,"")
    arr2=-1.D0
    call write_qts_hessian(glob%nat,1,glob%nivar,-1.D0,&
        glob%energy,glob%xcoords,glob%ihessian/svar2,0.D0,arr2,"rs")
  end if

end subroutine dlf_thermal
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/dlf_thermal_project
!!
!! FUNCTION
!!
!! * project out non-zero translational and rotational modes from Hessian
!!
!! References: 
!!   1 "Vibrational Analysis in Gaussian", Joseph W. Ochterski
!!      http://www.gaussian.com/g_whitepap/vib/vib.pdf
!!   2 The code in OPTvibfrq in opt/vibfrq.f
!!
!! Notes
!!   - Both of the above start from non-mass-weighted Cartesians whereas
!!     in DL-FIND the Hessian is already mass-weighted 
!!   - There appears to be an error in Ref 1 eqn 5 as the expression 
!!     should be multiplied by sqrt(mass), not divided by it (cf ref 2)
!!     If the uncorrected eqn5 is used then the rotational modes are not 
!!     fully orthogonal to the translational ones.
!!
!! SYNOPSIS
subroutine dlf_thermal_project(npmodes,peigval,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_hessian
  implicit none
  real(rk), external :: ddot
  integer, intent(out)  :: npmodes ! number of vibrational modes
  real(rk), intent(out) :: peigval(glob%nivar) ! eigenvalues after projection
  logical               :: tok
  real(rk)              :: comcoords(3,glob%nat) ! centre of mass coordinates
  real(rk)              :: com(3) ! centre of mass
  real(rk)              :: totmass ! total mass
  real(rk)              :: moi(3,3) ! moment of inertia tensor
  real(rk)              :: moivec(3,3) ! MOI eigenvectors
  real(rk)              :: moival(3) ! MOI eigenvalues
  real(rk)              :: transmat(glob%nivar,glob%nivar) ! transformation matrix
  real(rk)              :: px(3), py(3), pz(3)
  real(rk)              :: smass
  real(rk), parameter   :: mcutoff = 1.0d-12
  integer               :: ntrro ! number of trans/rot modes
  real(rk)              :: test, norm
  real(rk)              :: trialv(glob%nivar)
  real(rk)              :: phess(glob%nivar,glob%nivar) ! projected Hessian
  real(rk)              :: peigvec(glob%nivar, glob%nivar) ! eigenvectors after proj.
  real(rk)              :: pmodes(glob%nivar, glob%nivar) ! vib modes after proj.
  integer               :: pstart
  integer               :: ival, jval, kval, lval, icount
! **********************************************************************
  tok=.false.
  ! Do not continue if any coordinates are frozen
  if (glob%nivar /= glob%nat * 3 .or. glob%nat==1) then
     write(stdout,*)
     write(stdout,"('Frozen atoms found: no modes will be projected out')")
     npmodes = glob%nivar
     peigval = eigval
     return
  end if

  write(stdout,*)
  write(stdout,"('Projecting out translational and rotational modes')")

  ! Calculate centre of mass and moment of inertia tensor

  ! xcoords is not fully up to date so convert icoords instead
  call dlf_cartesian_itox(glob%nat, glob%nivar, glob%nicore, &
       glob%massweight, glob%icoords, comcoords)

  com(:) = 0.0d0
  totmass = 0.0d0
  do ival = 1, glob%nat
     com(1:3) = com(1:3) + glob%mass(ival) * comcoords(1:3, ival)
     totmass = totmass + glob%mass(ival)
  end do
  com(1:3) = com(1:3) / totmass

  do ival = 1, glob%nat
     comcoords(1:3, ival) = comcoords(1:3, ival) - com(1:3)
  end do

  moi(:,:) = 0.0d0
  do ival = 1, glob%nat
     moi(1,1) = moi(1,1) + glob%mass(ival) * &
          (comcoords(2,ival) * comcoords(2,ival) + comcoords(3,ival) * comcoords(3,ival))
     moi(2,2) = moi(2,2) + glob%mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) + comcoords(3,ival) * comcoords(3,ival))
     moi(3,3) = moi(3,3) + glob%mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) + comcoords(2,ival) * comcoords(2,ival))
     moi(1,2) = moi(1,2) - glob%mass(ival) * comcoords(1, ival) * comcoords(2, ival)
     moi(1,3) = moi(1,3) - glob%mass(ival) * comcoords(1, ival) * comcoords(3, ival)
     moi(2,3) = moi(2,3) - glob%mass(ival) * comcoords(2, ival) * comcoords(3, ival)
  end do
  moi(2,1) = moi(1,2)
  moi(3,1) = moi(1,3)
  moi(3,2) = moi(2,3)

  call dlf_matrix_diagonalise(3, moi, moival, moivec)

  if (printl >= 6) then
     write(stdout,"(/,'Centre of mass'/3f15.5)") com(1:3)
     write(stdout,"('Moment of inertia tensor')")
     write(stdout,"(3f15.5)") moi(1:3, 1:3)
     write(stdout,"('Principal moments of inertia')")
     write(stdout,"(3f15.5)") moival(1:3)
     write(stdout,"('Principal axes')")
     write(stdout,"(3f15.5)") moivec(1:3, 1:3)
  end if

  ! Construct transformation matrix to internal coordinates
  ntrro = 6
  transmat(:, :) = 0.0d0
  do ival = 1, glob%nat
     smass = sqrt(glob%mass(ival))
     kval = 3 * (ival - 1)
     ! Translational vectors
     transmat(kval+1, 1) = smass
     transmat(kval+2, 2) = smass
     transmat(kval+3, 3) = smass
     ! Rotational vectors
     px = sum(comcoords(1:3,ival) * moivec(1:3,1))
     py = sum(comcoords(1:3,ival) * moivec(1:3,2))
     pz = sum(comcoords(1:3,ival) * moivec(1:3,3))
     transmat(kval+1:kval+3, 4) = (py*moivec(1:3,3) - pz*moivec(1:3,2))*smass
     transmat(kval+1:kval+3, 5) = (pz*moivec(1:3,1) - px*moivec(1:3,3))*smass
     transmat(kval+1:kval+3, 6) = (px*moivec(1:3,2) - py*moivec(1:3,1))*smass
  end do
  ! Normalise vectors and check for linear molecules (one less mode)
  do ival = 1, 6
     test = ddot(glob%nivar, transmat(1,ival), 1, transmat(1,ival), 1)
     if (test < mcutoff) then
        kval = ival
        ntrro = ntrro - 1
        if (ntrro < 5) then
           write(stdout,"('Error: too few rotational/translation modes')")
           npmodes = glob%nivar
           peigval = eigval
           return
        end if
     else
        norm = 1.0d0/sqrt(test)
        call dscal(glob%nivar, norm, transmat(1,ival), 1)
     end if
  end do
  if (ntrro == 5 .and. kval /= 6) then
     transmat(:, kval) = transmat(:, 6)
     transmat(:, 6) = 0.0d0
  end if
  write(stdout,"(/,'Number of translational/rotational modes:',i4)") ntrro

  ! Generate 3N-ntrro other orthogonal vectors 
  ! Following the method in OPTvibfrq
  icount = ntrro
  do ival = 1, glob%nivar
     trialv(:) = 0.0d0
     trialv(ival) = 1.0d0
     do jval = 1, icount
        ! Test if trial vector is linearly independent of previous set
        test = -ddot(glob%nivar, transmat(1,jval), 1, trialv, 1)
        call daxpy(glob%nivar, test, transmat(1,jval), 1, trialv, 1)
     end do
     test = ddot(glob%nivar, trialv, 1, trialv, 1)
     if (test > mcutoff) then
        icount = icount + 1
        norm = 1.0d0/sqrt(test)
        transmat(1:glob%nivar, icount) = norm * trialv(1:glob%nivar)
     end if
     if (icount == glob%nivar) exit
  end do
  if (icount /= glob%nivar) then
     write(stdout,"('Error: unable to generate transformation matrix')")
     npmodes = glob%nivar
     peigval = eigval
     return
  end if
  if (printl >= 6) then
     write(stdout,"(/,'Transformation matrix')")
     call dlf_matrix_print(glob%nivar, glob%nivar, transmat)
  end if

  ! Apply transformation matrix: D(T) H D
  ! Use peigvec as scratch to store intermediate
  phess(:,:) = 0.0d0
  peigvec(:,:) = 0.0d0
  call dlf_matrix_multiply(glob%nivar, glob%nivar, glob%nivar, &
       1.0d0, glob%ihessian, transmat, 0.0d0, peigvec)
  ! Should alter dlf_matrix_multiply to allow transpose option to be set...
  transmat = transpose(transmat)
  call dlf_matrix_multiply(glob%nivar, glob%nivar, glob%nivar, &
       1.0d0, transmat, peigvec, 0.0d0, phess)
  transmat = transpose(transmat)

  if (printl >= 6) then
     write(stdout,"(/,'Hessian matrix after projection:')")
     call dlf_matrix_print(glob%nivar, glob%nivar, phess)
  end if

  ! Find eigenvalues of Nvib x Nvib submatrix
  peigval(:) = 0.0d0
  peigvec(:,:) = 0.0d0
  npmodes = glob%nivar - ntrro
  pstart = ntrro + 1
  call dlf_matrix_diagonalise(npmodes, phess(pstart:glob%nivar, pstart:glob%nivar), &
       peigval(1:npmodes), peigvec(1:npmodes,1:npmodes))

  if (printl >= 6) then
     write(stdout,"('Vibrational submatrix eigenvalues:')")
     write(stdout,"(12f9.5)") peigval(1:npmodes)
     write(stdout,"('Vibrational submatrix eigenvectors:')")
     call dlf_matrix_print(npmodes, npmodes, peigvec(1:npmodes, 1:npmodes))
  end if

  ! Print out normalised normal modes
  ! These are in non-mass-weighted Cartesians (division by smass)
  pmodes(:,:) = 0.0d0
  do kval = 1, glob%nivar
     do ival = 1, npmodes
        do jval = 1, npmodes
           pmodes(kval, ival) = pmodes(kval, ival) + &
                transmat(kval, ntrro + jval) * peigvec(jval, ival)
        end do
        lval = (kval - 1) / 3 + 1
        smass = sqrt(glob%mass(lval))
        pmodes(kval, ival) = pmodes(kval, ival) / smass
     end do
  end do
  do ival = 1, npmodes
     test = ddot(glob%nivar, pmodes(1,ival), 1, pmodes(1,ival), 1)
     norm = 1.0d0 / sqrt(test)
     call dscal(glob%nivar, norm, pmodes(1,ival), 1)
  end do

  if (printl >= 4) then
     write(stdout,"(/,'Normalised normal modes (Cartesian coordinates):')")
     call dlf_matrix_print(glob%nivar, npmodes, pmodes(1:glob%nivar, 1:npmodes))
  end if
  
  tok=.true.

end subroutine dlf_thermal_project
!!****
