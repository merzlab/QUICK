!***************************************************************************
!***************************************************************************
! The code below is also included in the lbfgs.f90 file of the 
! neuralnetwork program. I just included it here as well to 
! guarantee very easy copy/paste procedure to integrate gpr.f90
! in dl-find. Also I transformed it to a oop-style programming.

!!****h* gpr/oop_lbfgs
!!
!! NAME
!! OOP L-BFGS
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
!!An f77 version of this file was originally obtained from
!!http://www.ece.northwestern.edu/~nocedal/lbfgs.html
!!Condition for Use: This software is freely available for educational
!!or commercial purposes. We expect that all publications describing 
!!work using this software quote at least one of the references:
!!
!!J. Nocedal. Updating Quasi-Newton Matrices with Limited Storage (1980),
!!   Mathematics of Computation 35, pp. 773-782.
!!D.C. Liu and J. Nocedal. On the Limited Memory Method for Large Scale
!!   Optimization (1989), Mathematical Programming B, 45, 3, pp. 503-528.
!!
!!**********************************************************************

module oop_lbfgs
  use dlf_parameter_module, only: rk
  implicit none
!   integer,parameter :: rk=kind(1.D0) 
  type testtype
    integer                 :: testint
  end type testtype
  type oop_lbfgs_type
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
                                ! the algorithm gets order N**2 !
    
    integer                 :: point ! CURRENT POSITION IN THE WORK ARRAY
    INTEGER                 :: iter ! number of iteration
    logical                 :: tinit = .false.
    character(40)           :: tag
    type(oop_lbfgs_type),pointer  :: next
    type(oop_lbfgs_type),pointer  :: prev
    contains
      procedure             ::  next_step => oop_lbfgs_step    
      procedure             ::  restart => oop_lbfgs_restart
      procedure             ::  init => oop_lbfgs_init
      procedure             ::  do_precon => oop_lbfgs_precon
      procedure             ::  destroy  => oop_lbfgs_destroy
  end type oop_lbfgs_type
  integer , parameter :: lbfgs_stdout=2021
!   type(lbfgs_type),pointer,save :: lbfgs
  LOGICAL,PARAMETER :: dbg=.false. ! write debug info 

  contains
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/lbfgs/oop_lbfgs_step
!!
!! FUNCTION
!!
!! Form an L-BFGS step from the input geometry, gradient, and the history
!!
!! SYNOPSIS
SUBROUTINE oop_lbfgs_step(lbfgs, x,g,step_, inverted)
!! SOURCE
  !use dlf_global, only: lbfgs_stdout
!   USE GPR_LBFGS_MODULE_PAN
  IMPLICIT NONE
  !
  ! Dummy arguments
  !
  class(oop_lbfgs_type)       :: lbfgs
  REAL(RK), intent(in)    :: x(lbfgs%n) ! position (coords)
  REAL(RK), intent(in)    :: g(lbfgs%n) ! gradient
  REAL(RK), intent(out)   :: step_(lbfgs%n) ! Step
  logical, intent(out),optional :: inverted
  real(rk)                :: tmpVec(lbfgs%n)
  !
  ! Local variables
  !
  REAL(RK) :: diag(lbfgs%n)
  REAL(RK) :: beta , sq , stp , yr , ys , yy 
  INTEGER  :: bound , cp , i 
  INTEGER  :: oldpoint,ivar
  real(rk),external :: ddot
  if(present(inverted)) inverted = .false.
!   real(RK) :: oop_DDOT_internal
! **********************************************************************
  if(.not.lbfgs%tinit) stop "LBFGS not initialised!"

  if(lbfgs%iter==0) then ! first iteration, steepest descent!
    lbfgs%point = 1
    oldpoint = 1

    lbfgs%iter=1
    ! if the fist step should be smaller, include a factor here 
    if (sum(g**2)>1d-16) then
      step_(:) = -g(:)*1.e-4_rk/dsqrt(sum(g**2))
    else
      step_(:) = 0d0
    end if
    
    ! Store old gradient and coordinates
    lbfgs%store(:) = g(:)
    lbfgs%store2(:)= x(:)

    return

  end if

  ! ====================================================================
  ! All steps but first: calculate L-BFGS Step
  ! ====================================================================

  ! COMPUTE THE NEW STEP AND GRADIENT CHANGE
  lbfgs%step(:,lbfgs%point) = x(:) - lbfgs%store2(:) 
  lbfgs%dgrad(:,lbfgs%point) = g(:) - lbfgs%store(:)

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
  ys = ddot(lbfgs%n,lbfgs%dgrad(:,oldpoint),1,lbfgs%step(:,oldpoint),1)
  yy = ddot(lbfgs%n,lbfgs%dgrad(:,oldpoint),1,lbfgs%dgrad(:,oldpoint),1)
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
    sq = ddot(lbfgs%n,lbfgs%step(:,cp),1,lbfgs%store(:),1)
    lbfgs%alpha(cp) = lbfgs%rho(cp)*sq
    !CALL DAXPY(lbfgs%n,-lbfgs%alpha(cp),lbfgs%dgrad(:,cp),1,lbfgs%store(:),1) 
    lbfgs%store(:)=lbfgs%store(:)-lbfgs%alpha(cp)*lbfgs%dgrad(:,cp)
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
    lbfgs%store(:) = diag(:)*lbfgs%store(:) ! here, the pre-conditioner might enter
  end if
  !
  DO i = 1 , bound
    if(dbg) print*,"cp",cp
    yr = ddot(lbfgs%n,lbfgs%dgrad(:,cp),1,lbfgs%store(:),1)
    beta = lbfgs%rho(cp)*yr
    beta = lbfgs%alpha(cp) - beta
    !CALL DAXPY(lbfgs%n,beta,lbfgs%step(:,cp),1,lbfgs%store(:),1)
    lbfgs%store(:)=lbfgs%store(:)+beta*lbfgs%step(:,cp)
    cp = cp + 1
    IF ( cp>lbfgs%m ) cp = 1
  END DO
  !
  ! STORE THE NEW SEARCH DIRECTION
  lbfgs%step(:,lbfgs%point) = lbfgs%store(:)

  DO i = 1 , lbfgs%n
    lbfgs%store(i) = g(i)
  END DO

  stp = 1.D0

  ! ====================================================================
  ! check that function is descending
  ! ====================================================================
  !yr= ddot (lbfgs%n,lbfgs%step(:,lbfgs%point),1,g(:),1)
  yr= sum(lbfgs%step(:,lbfgs%point)*g(:))
  if(yr>0.D0) then
    if(present(inverted)) then
      inverted = .true.
    else
      !print*,"Inverting BFGS direction"
    end if
    stp=-stp
  end if

  if(dbg) then
    print*,"stp",stp
    print*,"lbfgs%step(:,:)",lbfgs%step(:,:)
    print*,"lbfgs%point",lbfgs%point
    print*,"lbfgs%step(lbfgs%point,:)",lbfgs%step(:,lbfgs%point)
  end if

  ! set step (return value)
  step_(:)=stp*lbfgs%step(:,lbfgs%point)
  
  ! Store old gradient and coordinates
  lbfgs%store(:) = g(:)
  lbfgs%store2(:)= x(:)

END SUBROUTINE oop_lbfgs_step
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/lbfgs/oop_lbfgs_restart
!!
!! FUNCTION
!!
!! Restart the L-BFGS algorithm, reset the memory
!!
!! SYNOPSIS
subroutine oop_lbfgs_restart(lbfgs)
!! SOURCE
!  use parameter_module, only: rk
!  use global, only: glob,lbfgs_stdout
!   USE GPR_LBFGS_MODULE_PAN
  implicit none
  class(oop_lbfgs_type)       :: lbfgs
  ! **********************************************************************
  write(lbfgs_stdout,'("Restarting L-BFGS optimiser")')
  lbfgs%iter=0
end subroutine oop_lbfgs_restart
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/lbfgs/oop_lbfgs_init
!!
!! FUNCTION
!!
!! Initialise the L-BFGS routines, allocate memory
!!
!! SYNOPSIS
subroutine oop_lbfgs_init(lbfgs,nvar,nmem,tprecon_in)
!! SOURCE
  !use dlf_parameter_module, only: rk
  !use dlf_global, only: glob,stderr
!   USE GPR_LBFGS_MODULE_PAN
  !use dlf_allocate, only: allocate
  implicit none
  class(oop_lbfgs_type)       :: lbfgs
  integer,  intent(in)    :: Nvar ! number of variables
  integer,  intent(in)    :: Nmem ! number steps to remember
  logical,  intent(in)    :: tprecon_in ! will a preconditioner be provided?
  ! **********************************************************************
  if(lbfgs%tinit) call oop_fail_local("LBFGS already initialised!")
!   allocate(lbfgs)
  lbfgs%tinit=.true.
  lbfgs%n=nvar
  lbfgs%m=nmem
  lbfgs%tprecon=tprecon_in
  if(lbfgs%n<=0) call oop_fail_local("nvar in L-BFGS has to be > 0")
  if(lbfgs%m<=0) call oop_fail_local("Nmem in L-BFGS has to be > 0")
  ! allocate memory
  allocate(lbfgs%store(LBFGS%N))
  allocate(lbfgs%store2(LBFGS%N))
  allocate(lbfgs%rho(LBFGS%M))
  allocate(lbfgs%alpha(LBFGS%M))
  allocate(lbfgs%step(LBFGS%N,LBFGS%M))
  allocate(lbfgs%dgrad(LBFGS%N,LBFGS%M))
  if(lbfgs%tprecon) allocate(lbfgs%precon(LBFGS%N,LBFGS%N))
  ! variables to set at the beginning
  lbfgs%iter = 0
end subroutine oop_lbfgs_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/lbfgs/oop_lbfgs_precon
!!
!! FUNCTION
!!
!! Supply a precondition matrix, an estimate of the inverse Hessian
!! to the module
!!
!! SYNOPSIS
subroutine oop_lbfgs_precon(lbfgs,precon)
!! SOURCE
  !use dlf_parameter_module, only: rk
  !use dlf_global, only: glob,stderr
!   USE GPR_LBFGS_MODULE_PAN
  !use dlf_allocate, only: allocate
  implicit none
  class(oop_lbfgs_type)       :: lbfgs
  real(rk)  ,intent(in):: precon(lbfgs%N,lbfgs%N)
  ! **********************************************************************
  if(.not.lbfgs%tinit) call oop_fail_local("LBFGS not initialised in oop_lbfgs_precon!")
  if(.not.lbfgs%tprecon) call oop_fail_local("tprecon must be set in oop_lbfgs_precon!")
  lbfgs%precon=precon
end subroutine oop_lbfgs_precon
!!****
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/lbfgs/oop_lbfgs_destroy
!!
!! FUNCTION
!!
!! Deallocate memory
!!
!! SYNOPSIS
subroutine oop_lbfgs_destroy(lbfgs)
!! SOURCE
  !use dlf_parameter_module, only: rk
  !use dlf_global, only: glob,stderr
!   USE GPR_LBFGS_MODULE_PAN
  !use dlf_allocate, only: deallocate
  implicit none
  class(oop_lbfgs_type)       :: lbfgs
  ! **********************************************************************
  if(.not.lbfgs%tinit) call oop_fail_local("LBFGS not initialised!")
  lbfgs%tinit=.false.
  ! check for positivity

  !deallocate memory
  deallocate(lbfgs%store)
  deallocate(lbfgs%store2)
  deallocate(lbfgs%rho)
  deallocate(lbfgs%alpha)
  deallocate(lbfgs%step)
  deallocate(lbfgs%dgrad)
  if(lbfgs%tprecon) then
    deallocate(lbfgs%precon)
  end if

!   deallocate(lbfgs)


end subroutine oop_lbfgs_destroy
!!****

!   ----------------------------------------------------------
!   local routine, only to be used if no external ddot is
!   available (which is not recommended!)
FUNCTION oop_DDOT_internal(n,dx,incx,dy,incy)
  !USE dlf_parameter_module, only: rk                        
!   USE GPR_LBFGS_MODULE_PAN
  IMPLICIT NONE
  !
  ! Dummy arguments
  !
  INTEGER :: incx , incy , n
  REAL(RK) :: oop_DDOT_internal
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
  oop_DDOT_internal = 0.0D0
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
        oop_DDOT_internal = dtemp
        return
      END IF
    END IF
    mp1 = m + 1
    DO i = mp1 , n , 5
      dtemp = dtemp + dx(i)*dy(i) + dx(i+1)*dy(i+1) + dx(i+2)     &
          & *dy(i+2) + dx(i+3)*dy(i+3) + dx(i+4)*dy(i+4)
    END DO
    oop_DDOT_internal = dtemp
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
    oop_DDOT_internal = dtemp
    RETURN
  END IF
END FUNCTION oop_DDOT_internal

subroutine oop_fail_local(str)
  character(*) :: str
  print*,str
  !CALL PANDORA_FAIL()
  stop "OOP_LBFGS FAIL LOCAL"
end subroutine oop_fail_local

end module oop_lbfgs
