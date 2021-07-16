!!****h* formstep/lbfgs
!!
!! NAME
!! L-BFGS
!!
!! FUNCTION
!! Optimisation algorithms: determine a search direction. 
!!
!!
!!       LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION
!!                         JORGE NOCEDAL
!!                       *** July 1990 ***
!!
!!
!!    This subroutine solves the unconstrained minimization problem
!!
!!                     min F(x),    x= (x1,x2,...,xN),
!!
!!     using the limited memory BFGS method. The routine is especially
!!     effective on problems involving a large number of variables. In
!!     a typical iteration of this method an approximation Hk to the
!!     inverse of the Hessian is obtained by applying M BFGS updates to
!!     a diagonal matrix Hk0, using information from the previous M steps.
!!     The user specifies the number M, which determines the amount of
!!     storage required by the routine. 
!!     The algorithm is described in "On the limited memory BFGS method
!!     for large scale optimization", by D. Liu and J. Nocedal,
!!     Mathematical Programming B 45 (1989) 503-528.
!!
!!    M (Nmem) is an INTEGER variable that must be set by the user to
!!            the number of corrections used in the BFGS update. It
!!            is not altered by the routine. Values of M less than 3 are
!!            not recommended; large values of M will result in excessive
!!            computing time. 3<= M <=7 is recommended. Restriction: M>0.
!!
!! An f77 version of this file was originally obtained from
!! http://www.ece.northwestern.edu/~nocedal/lbfgs.html
!! Condition for Use: This software is freely available for educational
!! or commercial purposes. We expect that all publications describing 
!! work using this software quote at least one of the references:
!!
!! J. Nocedal. Updating Quasi-Newton Matrices with Limited Storage (1980),
!! Mathematics of Computation 35, pp. 773-782.
!!
!! D.C. Liu and J. Nocedal. On the Limited Memory Method for Large Scale
!! Optimization (1989), Mathematical Programming B, 45, 3, pp. 503-528.
!!
!! For normal use in one instance, use dlf_lbfgs_init, _step, and _destroy. 
!! To restart the optimiser, use dlf_lbfgs_restart.
!!
!! If two overlapping optimisations should use L-BFGS, the module can be started
!! in more instances: use dlf_lbfgs_select to select an instance. Always the current
!! instance will be affected by dlf_lbfgs_step, dlf_lbfgs_restart and dlf_lbfgs_destroy. Use 
!! dlf_lbfgs_deselect to select the main instance (first instance). To invoke a new instance, 
!! use dlf_lbfgs_deselect("newname",.true.) and then dlf_lbfgs_init.
!!
!!**********************************************************************

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

MODULE LBFGS_MODULE
  USE dlf_parameter_module, only: rk 
  type lbfgs_type
    integer                 :: N ! number of variables
    integer                 :: M ! number steps to remember
    REAL(RK), ALLOCATABLE   :: store(:)  ! N WORK 1 - N
    REAL(RK), ALLOCATABLE   :: store2(:) ! N old coords
    REAL(RK), ALLOCATABLE   :: rho(:)    ! M WORK N+1 - N+M
    REAL(RK), ALLOCATABLE   :: alpha(:)  ! M WORK N+M+1 - N+2M
    REAL(RK), ALLOCATABLE   :: step(:,:) ! M,N WORK N+2M+1 - N+2M+NM
    REAL(RK), ALLOCATABLE   :: dgrad(:,:)! M,N WORK N+2M+NM+1 - N+2M+2NM
    logical                 :: tprecon ! is precon set (and allocated)?
    real(rk), allocatable   :: precon(:,:) ! (N,N) guess for the initial inverse hessian
                                ! full matrix, not just the diagonal. If this is used,
                                ! the algorithm gets order N^2 !
    
    integer                 :: point ! CURRENT POSITION IN THE WORK ARRAY
    INTEGER                 :: iter ! number of iteration
    logical                 :: tinit
    character(40)           :: tag
    type(lbfgs_type),pointer  :: next
  end type lbfgs_type
  type(lbfgs_type),pointer,save :: lbfgs
  type(lbfgs_type),pointer,save :: lbfgs_first
  logical, save               :: tinit=.false.
  LOGICAL,PARAMETER :: dbg=.false. ! write debug info 
  character(40),save     :: newtag="none"
END MODULE LBFGS_MODULE

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_step
!!
!! FUNCTION
!!
!! Form an L-BFGS step from the input geometry, gradient, and the history
!!
!! SYNOPSIS
SUBROUTINE dlf_lbfgs_step(x,g,step_)
!! SOURCE
  USE dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl
  USE lbfgs_module
  IMPLICIT NONE
  !
  ! Dummy arguments
  !
  REAL(RK), intent(in)    :: x(lbfgs%n) ! position (coords)
  REAL(RK), intent(in)    :: g(lbfgs%n) ! gradient
  REAL(RK), intent(out)   :: step_(lbfgs%n) ! Step
  !
  ! Local variables
  !
  REAL(RK) :: diag(lbfgs%n)
  REAL(RK) :: beta , gnorm , sq , stp , yr , ys , yy 
  INTEGER  :: bound , cp , i 
  INTEGER  :: oldpoint,ivar
  real(RK) ,external :: ddot
! **********************************************************************
  if(.not.tinit) call dlf_fail("LBFGS not initialised!")
  if(.not.lbfgs%tinit) then
    print*,"Instance of L-BFGS:",trim(lbfgs%tag)
    call dlf_fail("This instance of LBFGS not initialised!")
  end if

  if(lbfgs%iter==0) then ! first iteration, steepest descent!
    lbfgs%point = 1
    oldpoint = 1

    lbfgs%iter=1
    ! if the fist step should be smaller, include a factor here 
    step_(:) = -g(:)*0.02D0/dsqrt(sum(g**2))
    
    ! Store old gradient and coordinates
    lbfgs%store(:) = g(:)
    lbfgs%store2(:)= x(:)

    return

  end if

  ! ====================================================================
  ! All steps but first: calculate L-BFGS Step
  ! ====================================================================

  ! COMPUTE THE NEW STEP AND GRADIENT CHANGE
  lbfgs%step(lbfgs%point,:) = x(:) - lbfgs%store2(:) 
  lbfgs%dgrad(lbfgs%point,:) = g(:) - lbfgs%store(:)

  oldpoint=lbfgs%point

  ! take next point
  lbfgs%point = lbfgs%point + 1
  IF ( lbfgs%point>lbfgs%m ) lbfgs%point = 1

  lbfgs%iter = lbfgs%iter + 1

  if(dbg) print*,"@100 lbfgs%point=",lbfgs%point
  if(dbg) print*,"lbfgs%dgrad",lbfgs%dgrad
  if(dbg) print*,"lbfgs%step",lbfgs%step

  bound = lbfgs%iter - 1
  IF ( lbfgs%iter>lbfgs%m ) bound = lbfgs%m
  ys = DDOT(lbfgs%n,lbfgs%dgrad(oldpoint,:),1,lbfgs%step(oldpoint,:),1)
  yy = DDOT(lbfgs%n,lbfgs%dgrad(oldpoint,:),1,lbfgs%dgrad(oldpoint,:),1)
  diag(:) = ys/yy ! default guess for diag
  !print*,"JK precon scale factor:",ys/yy

  if(dbg) print*,"Before 200: ys,yy",ys,yy

  ! ====================================================================
  ! COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
  ! "Updating quasi-Newton matrices with limited storage",
  ! Mathematics of Computation, Vol.24, No.151, pp. 773-782.
  ! ====================================================================
  !
  cp = lbfgs%point
  lbfgs%rho(oldpoint) = 1.D0/ys
  if(dbg) print*,"lbfgs%rho",lbfgs%rho
  if(dbg) print*,"bound",bound

  lbfgs%store(:) = -g(:)
  cp = lbfgs%point
  DO i = 1 , bound
    cp = cp - 1
    IF ( cp==0 ) cp = lbfgs%m 
    sq = DDOT(lbfgs%n,lbfgs%step(cp,:),1,lbfgs%store(:),1)
    lbfgs%alpha(cp) = lbfgs%rho(cp)*sq
    !CALL DAXPY(lbfgs%n,-lbfgs%alpha(cp),lbfgs%dgrad(cp,:),1,lbfgs%store(:),1) 
    lbfgs%store(:)=lbfgs%store(:)-lbfgs%alpha(cp)*lbfgs%dgrad(cp,:)
  END DO
  if(dbg) print*,"removed DAXPY"
  if(dbg) print*,"AFTER CALL TO DAXPY"
  !
  if(lbfgs%tprecon) then
    ! diag= lbfgs%precon * lbfgs%store manually :-(
    do ivar=1,lbfgs%n
      diag(ivar)=sum(lbfgs%precon(:,ivar)*lbfgs%store(:))
    end do
    !diag = matmul(lbfgs%precon,lbfgs%store)
    lbfgs%store(:) = diag !* ys/yy
    !lbfgs%store(:) = lbfgs%store(:)*ys/yy
  else
    lbfgs%store(:) = diag(:)*lbfgs%store(:) 
  end if
  !
  DO i = 1 , bound
    if(dbg) print*,"cp",cp
    yr = DDOT(lbfgs%n,lbfgs%dgrad(cp,:),1,lbfgs%store(:),1)
    beta = lbfgs%rho(cp)*yr
    beta = lbfgs%alpha(cp) - beta
    !CALL DAXPY(lbfgs%n,beta,lbfgs%step(cp,:),1,lbfgs%store(:),1)
    lbfgs%store(:)=lbfgs%store(:)+beta*lbfgs%step(cp,:)
    cp = cp + 1
    IF ( cp>lbfgs%m ) cp = 1
  END DO
  !
  ! STORE THE NEW SEARCH DIRECTION
  lbfgs%step(lbfgs%point,:) = lbfgs%store(:)

  DO i = 1 , lbfgs%n
    lbfgs%store(i) = g(i)
  END DO

  stp = 1.D0

  ! ====================================================================
  ! check that function is descending
  ! ====================================================================
  yr= ddot (lbfgs%n,lbfgs%step(lbfgs%point,:),1,g(:),1)
  if(yr>0.D0) then
    if(printl>=4) write(stdout,*) "Inverting BFGS direction"
    stp=-stp
  end if

  if(dbg) then
    print*,"stp",stp
    print*,"lbfgs%step(:,:)",lbfgs%step(:,:)
    print*,"lbfgs%point",lbfgs%point
    print*,"lbfgs%step(lbfgs%point,:)",lbfgs%step(lbfgs%point,:)
  end if

  ! set step (return value)
  step_(:)=stp*lbfgs%step(lbfgs%point,:)
  
  ! Store old gradient and coordinates
  lbfgs%store(:) = g(:)
  lbfgs%store2(:)= x(:)

END SUBROUTINE DLF_LBFGS_STEP
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_restart
!!
!! FUNCTION
!!
!! Restart the L-BFGS algorithm, reset the memory
!!
!! SYNOPSIS
subroutine dlf_lbfgs_restart
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout, printl
  USE lbfgs_module
  implicit none
  ! **********************************************************************
  if(.not.tinit) call dlf_fail("LBFGS not initialised!")
  if(.not.lbfgs%tinit) then
    print*,"Instance of L-BFGS:",trim(lbfgs%tag)
    call dlf_fail("This instance of LBFGS not initialised!")
  end if

  if(printl>=4) then
    if(trim(lbfgs%tag)=="main") then
      write(stdout,'("Restarting L-BFGS optimiser")')
    else
      write(stdout,'("Restarting L-BFGS optimiser, instance: ",a)') &
          trim(lbfgs%tag)
    end if
  end if
  lbfgs%iter=0
end subroutine dlf_lbfgs_restart
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_select
!!
!! FUNCTION
!!
!! Select an instance of L-BFGS 
!!
!! To be called before initialisation of new instances, but not before
!! the first instance, which is automaticall called "main".
!!
!! SYNOPSIS
subroutine dlf_lbfgs_select(tag,newinstance)
!! SOURCE
  USE lbfgs_module
  implicit none
  character(*), intent(in)  :: tag
  logical     , intent(in)  :: newinstance
  ! **********************************************************************
  if(.not.tinit) then
    !call dlf_fail("LBFGS not initialised in lbfgs_select!")
    call dlf_lbfgs_init(1,1) ! initialise a dummy first instance
    lbfgs%tinit=.false.
  end if
    

  ! try to selcet instance with %tag=tag
  lbfgs=>lbfgs_first
  do while (associated(lbfgs%next))
    if(trim(lbfgs%tag)==trim(tag)) exit
    lbfgs=>lbfgs%next
  end do

  if(newinstance) then
    if(trim(lbfgs%tag)==trim(tag)) then 
      print*,"Error, instance ",trim(tag)," already exists and selected &
          &with flag 'new'"
      call dlf_fail("Error selecting new hdlcopt instance")
    end if
    newtag=tag
    ! last instance selected, newtag contains name of new instance
  else
    if(lbfgs%tag/=tag) then
      print*,"Error, instance ",trim(tag)," does not exist"
      print*,"Existing inctances:"
      lbfgs=>lbfgs_first
      do while (associated(lbfgs))
        print*,"--",trim(lbfgs%tag),"--"
        lbfgs=>lbfgs%next
      end do
      call dlf_fail("Error selecting new hdlcopt instance")

    end if
    ! instance with %tag = tag selected
  end if
  if(dbg) PRINT*,"SELECTED -",trim(tag),"-"
end subroutine dlf_lbfgs_select
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_deselect
!!
!! FUNCTION
!!
!! Select first instance of L-BFGS 
!!
!! SYNOPSIS
subroutine dlf_lbfgs_deselect
!! SOURCE
  USE lbfgs_module
  implicit none
  ! **********************************************************************
  ! do nothing in case lbfgs does not exist
  if(.not.tinit) return !call dlf_fail("LBFGS not initialised!")
  call dlf_lbfgs_select("main",.false.)
  newtag="main"
end subroutine dlf_lbfgs_deselect
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_exists
!!
!! FUNCTION
!!
!! Find out if an instance with a specified tag is initialised
!!
!! SYNOPSIS
subroutine dlf_lbfgs_exists(tag,exists)
!! SOURCE
  USE lbfgs_module
  implicit none
  character(*),   intent(in)  :: tag
  logical     ,   intent(out) :: exists
  type(lbfgs_type),pointer    :: lbfgs_search
  ! **********************************************************************
  if(.not.tinit) then
    exists=.false.
    return
  end if

  lbfgs_search=>lbfgs_first
  exists=.false.
  do while (associated(lbfgs_search))
    if(trim(lbfgs_search%tag)==trim(tag)) then
      if(lbfgs_search%tinit) then
        exists=.true.
        return
      end if
    end if
    lbfgs_search => lbfgs_search%next
  end do
end subroutine dlf_lbfgs_exists
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_init
!!
!! FUNCTION
!!
!! Initialise the L-BFGS routines, allocate memory
!!
!! SYNOPSIS
subroutine dlf_lbfgs_init(nvar,nmem)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr
  USE lbfgs_module
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer,  intent(in)    :: Nvar ! number of variables
  integer,  intent(in)    :: Nmem ! number steps to remember
  ! **********************************************************************
  if(tinit) then
    ! this is initialisation of a new (not first) instance

    ! trap initialisation of "main" if another instance is allready there
    if(trim(newtag)=="main") then
      if(trim(lbfgs%tag)/="main".or.lbfgs%n/=1) then
        call dlf_fail("L-BFGS main instance is allready initialised")
      end if
      ! deallocate 
      call deallocate(lbfgs%store)
      call deallocate(lbfgs%store2)
      call deallocate(lbfgs%rho)
      call deallocate(lbfgs%alpha)
      call deallocate(lbfgs%step)
      call deallocate(lbfgs%dgrad)
      
    else

      ! lbfgs should now point to the last existing instance
      if(dbg) print*,"Current lbfgs instance: ",trim(lbfgs%tag)
      
      ! check that no instance with tag=newtag exists
      lbfgs=>lbfgs_first
      do while (associated(lbfgs%next))
        lbfgs=>lbfgs%next
        if(trim(lbfgs%tag)==trim(newtag)) then
          print*,"Instance with name ",trim(newtag)," already initialised"
          call dlf_fail("Instance with name already initialised")
        end if
      end do
      
      allocate(lbfgs%next)
      lbfgs=>lbfgs%next
      nullify(lbfgs%next)
      
    end if ! (trim(newtag)=="main")

  else
    ! this is the initialisation of the first instance
    tinit=.true.
    newtag="main"
    if(associated(lbfgs)) call dlf_fail("This instance of LBFGS has already been initialised")
    ! allocate the lbfgs pointer
    allocate(lbfgs)

    nullify(lbfgs%next)
    lbfgs_first => lbfgs

  end if


  lbfgs%tag=newtag
  lbfgs%tinit=.true.

  if(dbg) print*,"Allocating ",trim(lbfgs%tag)
  lbfgs%n=nvar
  lbfgs%m=nmem
  lbfgs%tprecon=.false.
  if(lbfgs%n<=0) call dlf_fail("nvar in L-BFGS has to be > 0")
  if(lbfgs%m<=0) call dlf_fail("Nmem in L-BFGS has to be > 0")

  ! allocate memory
  call allocate(lbfgs%store,LBFGS%N)
  call allocate(lbfgs%store2,LBFGS%N)
  call allocate(lbfgs%rho,LBFGS%M)
  call allocate(lbfgs%alpha,LBFGS%M)
  call allocate(lbfgs%step,LBFGS%M,LBFGS%N)
  call allocate(lbfgs%dgrad,LBFGS%M,LBFGS%N)

  ! variables to set at the beginning
  lbfgs%iter = 0

  ! initialise (mainly to avoid NaNs in checkpointing)
  lbfgs%store(:)=0.D0
  lbfgs%store2(:)=0.D0
  lbfgs%rho(:)=0.D0
  lbfgs%alpha(:)=0.D0
  lbfgs%step(:,:)=0.D0
  lbfgs%dgrad(:,:)=0.D0

end subroutine dlf_lbfgs_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_destroy
!!
!! FUNCTION
!!
!! Deallocate memory
!!
!! SYNOPSIS
subroutine dlf_lbfgs_destroy
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr
  USE lbfgs_module
  use dlf_allocate, only: deallocate
  implicit none
  logical         :: allgone
  ! **********************************************************************
  ! Only need to deallocate if LBFGS has been initialised
  if(.not.tinit) return

  !deallocate memory of the present instance
  call deallocate(lbfgs%store)
  call deallocate(lbfgs%store2)
  call deallocate(lbfgs%rho)
  call deallocate(lbfgs%alpha)
  call deallocate(lbfgs%step)
  call deallocate(lbfgs%dgrad)
  if(lbfgs%tprecon) then
    deallocate(lbfgs%precon)
  end if
  lbfgs%tinit=.false.
!print*,"Destroying ",trim(lbfgs%tag)

  ! check if all instances are deleted
  allgone=.true.
  lbfgs=>lbfgs_first
  do while (associated(lbfgs))
    if(lbfgs%tinit) then
      allgone=.false.
      exit
    end if
    lbfgs => lbfgs%next
  end do
  ! this may leave lbfgs pointing nowhere. This is fine as one cannot expect
  ! it to point somewhere usefull after dlf_lbfgs_destroy


  if(allgone) then

    ! if only a dummy has been initialised for the instance MAIN, it may still be
    ! allocated, even though %tinit would be false. Deallocate in this case
    lbfgs=>lbfgs_first
    if(allocated(lbfgs%store)) call deallocate(lbfgs%store)
    if(allocated(lbfgs%store2)) call deallocate(lbfgs%store2)
    if(allocated(lbfgs%rho)) call deallocate(lbfgs%rho)
    if(allocated(lbfgs%alpha)) call deallocate(lbfgs%alpha)
    if(allocated(lbfgs%step)) call deallocate(lbfgs%step)
    if(allocated(lbfgs%dgrad)) call deallocate(lbfgs%dgrad)

    ! deallocate everything
    tinit=.false.
    lbfgs=>lbfgs_first
    do while (associated(lbfgs%next))
      lbfgs_first => lbfgs
      lbfgs => lbfgs%next
!print*,"Finally deallocating ",trim(lbfgs_first%tag)
      deallocate(lbfgs_first)
    end do
!print*,"Finally deallocating ",trim(lbfgs%tag)
    deallocate(lbfgs)
    nullify(lbfgs)
    nullify(lbfgs_first)
  end if

end subroutine dlf_lbfgs_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* lbfgs/dlf_lbfgs_precon
!!
!! FUNCTION
!!
!! Supply a precondition matrix, an estimate of the inverse Hessian
!! to the module
!! The space for the matrix is only allocated here.
!!
!! SYNOPSIS
subroutine dlf_lbfgs_precon(precon)
!! SOURCE
  use dlf_parameter_module, only: rk
  !use dlf_global, only: glob,stderr
  USE lbfgs_module
  use dlf_allocate, only: allocate
  implicit none
  real(rk)  ,intent(in):: precon(lbfgs%N,lbfgs%N)
  ! **********************************************************************
  if(.not.tinit) call dlf_fail("LBFGS not initialised in lbfgs_precon!")
  if(.not.lbfgs%tinit) then
    print*,"Instance of L-BFGS:",trim(lbfgs%tag)
    call dlf_fail("This instance of LBFGS not initialised!")
  end if
  if(.not.lbfgs%tprecon) then
    lbfgs%tprecon=.true.
    call allocate(lbfgs%precon, LBFGS%N, LBFGS%N)
  end if
  lbfgs%precon=precon
end subroutine dlf_lbfgs_precon
!!****

!   ----------------------------------------------------------
!   local routine, only to be used if no external ddot is
!   available (which is not recommended!)
FUNCTION DDOT_internal(n,dx,incx,dy,incy)
  USE dlf_parameter_module, only: rk                        
  IMPLICIT NONE
  !
  ! Dummy arguments
  !
  INTEGER :: incx , incy , n
  REAL(RK) :: DDOT_internal
  REAL(RK) , DIMENSION(n) :: dx , dy
  INTENT (IN) dx , dy , incx , incy , n
  !
  ! Local variables
  !
  REAL(RK) :: dtemp
  INTEGER :: i , ix , iy , m , mp1
  !
  !     forms the dot product of two vectors.
  !     uses unrolled loops for increments equal to one.
  !     jack dongarra, linpack, 3/11/78.
  !
  !
  DDOT_internal = 0.0D0
  dtemp = 0.0D0
  IF ( n<=0 ) RETURN
  IF ( incx==1 .AND. incy==1 ) THEN
    !
    !        code for both increments equal to 1
    !
    !
    !        clean-up loop
    !
    m = MOD(n,5)
    IF ( m/=0 ) THEN
      DO i = 1 , m
        dtemp = dtemp + dx(i)*dy(i)
      END DO
      IF ( n<5 ) THEN
        DDOT_internal = dtemp
        return
      END IF
    END IF
    mp1 = m + 1
    DO i = mp1 , n , 5
      dtemp = dtemp + dx(i)*dy(i) + dx(i+1)*dy(i+1) + dx(i+2)     &
          & *dy(i+2) + dx(i+3)*dy(i+3) + dx(i+4)*dy(i+4)
    END DO
    DDOT_internal = dtemp
  ELSE
    !
    !        code for unequal increments or equal increments
    !          not equal to 1
    !
    ix = 1
    iy = 1
    IF ( incx<0 ) ix = (-n+1)*incx + 1
    IF ( incy<0 ) iy = (-n+1)*incy + 1
    DO i = 1 , n
      dtemp = dtemp + dx(ix)*dy(iy)
      ix = ix + incx
      iy = iy + incy
    END DO
    DDOT_internal = dtemp
    RETURN
  END IF
END FUNCTION DDOT_INTERNAL

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_lbfgs_write
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr
  USE lbfgs_module
  use dlf_checkpoint, only: tchkform,write_separator
  implicit none
  type(lbfgs_type),pointer :: lbfgs_current
  ! **********************************************************************
  if(.not.tinit) call dlf_fail("LBFGS not initialised! (in checkpoint write)")

  lbfgs_current => lbfgs
  lbfgs => lbfgs_first

  if(tchkform) then
    open(unit=100,file="dlf_lbfgs.chk",form="formatted")
    call write_separator(100,"current")
    write(100,*) lbfgs_current%tag
    do while (associated(lbfgs))
      call write_separator(100,"NM")
      write(100,*) lbfgs%n,lbfgs%m
      call write_separator(100,"Arrays")
      write(100,*) lbfgs%store,lbfgs%store2,lbfgs%rho,lbfgs%alpha,lbfgs%step,lbfgs%dgrad
      call write_separator(100,"Position")
      write(100,*) lbfgs%point,lbfgs%iter
      lbfgs=>lbfgs%next
    end do
    call write_separator(100,"END")
  else
    open(unit=100,file="dlf_lbfgs.chk",form="unformatted")
    call write_separator(100,"current")
    write(100) lbfgs_current%tag
    do while (associated(lbfgs))
      call write_separator(100,"NM")
      write(100) lbfgs%n,lbfgs%m
      call write_separator(100,"Arrays")
      write(100) lbfgs%store,lbfgs%store2,lbfgs%rho,lbfgs%alpha,lbfgs%step,lbfgs%dgrad
      call write_separator(100,"Position")
      write(100) lbfgs%point,lbfgs%iter
      lbfgs=>lbfgs%next
    end do
    call write_separator(100,"END")
  end if
  close(100)
  lbfgs => lbfgs_current

end subroutine dlf_checkpoint_lbfgs_write

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_lbfgs_read(tok)
  use dlf_global, only: stdout,printl
  USE lbfgs_module
  use dlf_checkpoint, only: tchkform, read_separator
  implicit none
  logical,intent(out) :: tok
  logical             :: tchk
  integer             :: n_f,m_f
  character(40)       :: tag_read
  ! **********************************************************************
  tok=.false.
  if(.not.tinit) call dlf_fail("LBFGS not initialised! (in checkpoint read)")

  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_lbfgs.chk",EXIST=tchk)
  if(.not.tchk) then
    write(stdout,10) "File dlf_lbfgs.chk not found"
    return
  end if

  if(tchkform) then
    open(unit=100,file="dlf_lbfgs.chk",form="formatted")
  else
    open(unit=100,file="dlf_lbfgs.chk",form="unformatted")
  end if

  lbfgs => lbfgs_first

  call read_separator(100,"current",tchk)
  if(.not.tchk) return    

  if(tchkform) then
    read(100,*,end=201,err=200) tag_read
  else
    read(100,end=201,err=200) tag_read
  end if

  do while (associated(lbfgs))


    call read_separator(100,"NM",tchk)
    if(.not.tchk) return    

    if(tchkform) then
      read(100,*,end=201,err=200) n_f,m_f
    else
      read(100,end=201,err=200) n_f,m_f
    end if
    
    if(n_f/=lbfgs%n) then
      write(stdout,10) "Different L-BFGS system size"
      close(100)
      return
    end if
    if(m_f/=lbfgs%m) then
      write(stdout,10) "Different L-BFGS memory size"
      close(100)
      return
    end if
    
    call read_separator(100,"Arrays",tchk)
    if(.not.tchk) return 
    
    if(tchkform) then
      read(100,*,end=201,err=200) lbfgs%store,lbfgs%store2,lbfgs%rho,lbfgs%alpha,lbfgs%step,lbfgs%dgrad
    else
      read(100,end=201,err=200) lbfgs%store,lbfgs%store2,lbfgs%rho,lbfgs%alpha,lbfgs%step,lbfgs%dgrad
    end if
    
    call read_separator(100,"Position",tchk)
    if(.not.tchk) return 

    if(tchkform) then
      read(100,*,end=201,err=200) lbfgs%point,lbfgs%iter
    else
      read(100,end=201,err=200) lbfgs%point,lbfgs%iter
    end if

    lbfgs=>lbfgs%next

  end do ! while (associated(lbfgs))

  call read_separator(100,"END",tchk)
  if(.not.tchk) return 
    
  ! now make sure that the instance of the checkpoint is selected
  call dlf_lbfgs_select(trim(tag_read),.false.)

  if(printl >= 6) write(stdout,"('LBFGS checkpoint file sucessfully read')")
  close(100)
  tok=.true.

  return

  ! return on error
  close(100)
200 continue
  write(stdout,10) "Error reading LBFGS checkpoint file"
  return
201 continue
  write(stdout,10) "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a)

end subroutine dlf_checkpoint_lbfgs_read
