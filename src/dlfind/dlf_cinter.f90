! **********************************************************************
! **                       Curve Interpolation                        **
! **********************************************************************
!!****h* DL-FIND/cinter
!!
!! NAME
!! cinter
!!
!! FUNCTION
!! It interpolates curves using splines or GPR.
!! It also has some additional functionality like distributing points
!! equidistantly along the curve, some example curves, ...
!! This module is purely object oriented.
!!
!! Inputs
!!    points to interpolate
!! 
!! Outputs
!!    a curve-type that can be evaluated arbitrarily
!!
!! COMMENTS
!!    Use either Spline_Curve type or GPR_curve type to determine
!!    which method to use
!!
!! COPYRIGHT
!!
!!  Copyright 2018 , Alexander Denzel (denzel@theochem.uni-stuttgart.de)
!!  Johannes Kaestner (kaestner@theochem.uni-stuttgart.de)
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
! #define TestOnMB

module cinter_module
use gpr_module
use gpr_types_module
use gpr_opt_module
use dlf_parameter_module, only: rk
use oop_lbfgs
implicit none
type curve_type
  logical                       ::  interpolated
  logical                       ::  parameter_init = .false.
  integer                       ::  nDim
  integer                       ::  nControlPoints
  real(rk), allocatable         ::  pts_in_time(:)
  real(rk), allocatable         ::  controlPoints(:,:)
contains
  ! interpolates points to a curve
  procedure                     ::  initialize      
  ! evaluates the curve at a certain value of the parametrization
  procedure                     ::  evaluate        
  ! evaluating the gradient of the curve at one control point
  procedure                     ::  eval_grad_at_cPt
  ! evaluating the gradient of the curve at all the control points
  procedure                     ::  eval_grad_at_cPts
  ! evaluates the derivative along the curve at a certain value of the parametrization
  procedure                     ::  eval_grad
  ! set equidistant points on the curve as control points
  procedure                     ::  setEqDistPts
  ! distribute points on the curve equidistantly (in the nDim-D space)
  procedure                     ::  eqDistPts
  ! adding some points to make it approximately equidistantly
  procedure                     ::  addPtsToEqDist
  ! calculate the loss function for the eqDistPts subroutine
  procedure                     ::  calc_L_and_dL
  ! Simple 1D curve
  procedure                     ::  build1DExample
  ! Test function that builds a 3D curve resembling a heart
  procedure                     ::  build3Dheartshape 
  ! destroy all allocated objects
  procedure                     ::  destroy_cinter  
  ! give the time in the parametrization space of a certain control point
  procedure                     ::  timeOfCPoint
  ! setting the end points of the curve to specific values
  ! that can be used to fix the endpoints in path optimization
  procedure                     ::  setEndPoints
  ! write data for plotting the curve
  procedure                     ::  writePlotData
  ! write data for plotting the curve and some extra points
  procedure                     ::  writePlotData_extra
  ! calculates the difference between this and a different curve
  procedure                     ::  calcDiffToCurve
  ! smoothens the curve by changing the s_n parameter
  procedure                     ::  smooth
  ! plotting vectors on all points given
  procedure                     ::  writeVectorPlots
  ! backup the interpolation
  procedure                     ::  backupCurve 
  ! read in the backup
  procedure                     ::  readInCurve
  ! when two cPts are to close redistribute
  procedure                     ::  avoidSmallDistance
  ! check if the endpts are too far away from the original ones
  procedure                     ::  avoidDeviationEndPts
  ! add a point between the two points given by the indices
  procedure                     ::  addPtBetweenPts
end type curve_type

type, extends(curve_type) :: GPR_curve
!   type(gpr_type)                ::  gpr_instance
  integer                       ::  gpr_nat
  integer                       ::  gpr_OffsetType
  integer                       ::  gpr_kernel_type
  type(gpr_type), allocatable   ::  gprs(:)
  real(rk)                      ::  gamma, s_f, s_n(3)
contains
  procedure                     ::  setGPRParameters
end type GPR_curve

type, extends(curve_type) :: Spline_Curve
  ! these type of splines are sometimes called
  ! "tensor product splines" or simply "multi-dimensional splines"
  ! each dimension is a seperate spline (they are independent)
  
  real(rk), allocatable         ::  params(:,:,:) 
                    ! dimensions of this vector are:
                    ! dimension of the problem, 4(parameters for cubic splines),
                    ! number of control points
  real(rk), allocatable         ::  distances(:)
end type Spline_Curve

contains

subroutine setGPRParameters(curve, gamma_in, s_f_in, s_n_in,&
                            OffsetType_in, kernel_type_in, nat_in)
  class(GPR_curve)     ::  curve
  real(rk), intent(in) , optional :: gamma_in, s_f_in, s_n_in(3)
  integer, intent(in), optional ::  OffsetType_in, kernel_type_in
  integer, intent(in), optional ::  nat_in
  
  if(present(gamma_in)) then
    curve%gamma = gamma_in
  else
    curve%gamma = 1d0/(5d-1)**2 ! MB: 1/0.3^2
  end if
  
  if(present(s_f_in)) then
    curve%s_f = s_f_in
  else
    curve%s_f = 1d0
  end if
  
  if (present(s_n_in)) then
    curve%s_n = s_n_in
  else
    curve%s_n = (/2d-4,1d-4,3d-4/)
  end if
  
  if (present(OffsetType_in)) then
    curve%gpr_OffsetType = OffsetType_in
  else
    curve%gpr_OffsetType = 7
  end if
  
  if (present(kernel_type_in)) then
    curve%gpr_kernel_type = kernel_type_in
  else
    curve%gpr_kernel_type = 0
  end if
  
  if(present(nat_in)) then
    curve%gpr_nat = nat_in
  else
    curve%gpr_nat = -1
  end if  
  curve%parameter_init = .true.
end subroutine setGPRParameters

subroutine initialize(curve, nDim_in, nControlPoints_in, coords)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  nDim_in
  integer, intent(in)   ::  nControlPoints_in
  real(rk),intent(in)   ::  coords(nDim_in,nControlPoints_in)
  real(rk)              ::  coord_tmp(nControlPoints_in)
  integer               ::  dimiter, titer, i, j
  curve%nDim = nDim_in
  curve%nControlPoints = nControlPoints_in
  if (allocated(curve%pts_in_time)) deallocate(curve%pts_in_time)
  if (allocated(curve%controlPoints)) deallocate(curve%controlPoints)
  allocate(curve%controlPoints(curve%nDim,curve%nControlPoints))
  allocate(curve%pts_in_time(curve%nControlPoints))
  curve%controlPoints(:,:) = coords(:,:)
  select type (curve)
  type is (curve_type)
  class is (GPR_curve)
    ! ****************************************************
    ! GPR curve
    if (printl>=6.and.mod(nDim_in,3)/=0.and.curve%gpr_nat==-1) &
        write(stdout,'(&
            "Warning: natoms is not set correctly. ", &
            "Basic spline interpolation should still work.")')
    if (.not.curve%parameter_init) call curve%setGPRParameters() ! set to default if not set
!   subroutine GPR_construct(this, nt, nat, sdgf, OffsetType, kernel_type, order)
    if (allocated(curve%gprs)) deallocate(curve%gprs)
    
    allocate(curve%gprs(curve%nDim))
    do titer = 1, curve%nControlPoints
      curve%pts_in_time(titer) = REAL(titer-1,kind=rk)*&
                                 1d0/REAL(curve%nControlPoints-1, kind=rk)
    end do
    do dimiter = 1, curve%nDim
      if (curve%gprs(dimiter)%constructed) call GPR_destroy(curve%gprs(dimiter))
      ! NOTE: EXPLICIT INVERSE IS REQUIRED HERE, SINCE I USE IT IN GPRMEP.F90
      ! FOR THE TRANSFORMATION OF THE LOSS FUNCTION INTO THE SPACE OF 
      ! THE CONTROL POINTS
      curve%gprs(dimiter)%iChol = .false.
      if (glob%spec((dimiter-1)/3+1)>=0) then
        ! free atom
        call GPR_construct(curve%gprs(dimiter), curve%nControlPoints, &
                         curve%gpr_nat, 1, &
                         curve%gpr_OffsetType, curve%gpr_kernel_type, 0) !1D GP
        coord_tmp(:) = coords(dimiter,:)
        call GPR_init(curve%gprs(dimiter), curve%pts_in_time, &
                      curve%gamma, curve%s_f, curve%s_n, coord_tmp(:))
        ! EFFICIENCY!! One could avoid recalculating+Inverting the covariance 
        ! matrix when there are no new training points
        ! BUT NOTE: EXPLICIT INVERSE IS REQUIRED HERE, SINCE I USE IT IN GPRMEP.F90
        ! FOR THE TRANSFORMATION OF THE LOSS FUNCTION INTO THE SPACE OF 
        ! THE CONTROL POINTS
        call GPR_interpolation(curve%gprs(dimiter),8)
      else
        ! frozen atom
        ! do nothing... evaluation must simply choose the constant value
      end if

    end do
!   subroutine GPR_init(this, x, gamma, s_f, s_n, es, gs, hs)
  class is (Spline_Curve)
    ! ****************************************************
    ! Spline curve
    if(printl>=4) write(stdout,'("Spline interpolation...")')
    if (allocated(curve%params)) deallocate(curve%params)
    if (allocated(curve%distances)) deallocate(curve%distances)
    
    allocate(curve%params(curve%nDim, 4, curve%nControlPoints))
    allocate(curve%distances(curve%nControlPoints-1))
    
    curve%distances(:) = 1d0/REAL(curve%nControlPoints-1, kind=rk)
    do titer = 1, curve%nControlPoints
      curve%pts_in_time(titer) = REAL(titer-1,kind=rk)*&
                                 1d0/REAL(curve%nControlPoints-1, kind=rk)
    end do
    do dimiter=1,curve%nDim
      ! Spline in dimension dimiter shall be independent
      ! of Splines in the other dimensions
      call dlf_init_csplines(curve%nControlPoints, coords(dimiter,:),&
                             curve%distances,curve%params(dimiter,:,:))
    end do
    if(printl>=4) write(stdout,'("Splines constructed")')
  class default
    call dlf_fail("Unexpected type! (initialize)")
  end select
end subroutine initialize

! time means "value of the variable that parametrizes the curve"
subroutine evaluate(curve, ntimesteps, times, point)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  ntimesteps
  real(rk), intent(in)  ::  times(ntimesteps)
  real(rk), intent(out) ::  point(curve%ndim,ntimesteps)
  integer               ::  dimiter
  integer               ::  titer
  select type(curve)
  type is (curve_type)
  class is (GPR_curve)
    do titer=1,ntimesteps
      do dimiter=1, curve%nDim
        if (glob%spec((dimiter-1)/3+1)>=0) then
          ! free atom
          call gpr_eval(curve%gprs(dimiter), times(titer),point(dimiter,titer))
        else
          ! frozen atom
          point(dimiter,titer) = curve%controlPoints(dimiter, 1)
        end if
      end do
    end do
  class is (Spline_Curve)
    do dimiter=1, curve%nDim
      call dlf_eval_cspline(curve%nControlPoints, curve%params(dimiter,:,:), &
                            curve%distances, ntimesteps, times, point(dimiter,:))
    end do    
  class default
    call dlf_fail("Unexpected type! (evaluate)")
  end select
end subroutine evaluate

subroutine eval_grad_at_cPts(curve, grads)
  class(curve_type)     ::  curve
  real(rk), intent(out) ::  grads(curve%nDim,curve%nControlPoints)
  call curve%eval_grad(curve%nControlPoints, curve%pts_in_time, grads)  
end subroutine

subroutine eval_grad_at_cPt(curve, pointnr, grad)
  class(curve_type)             ::  curve
  integer, intent(in)           ::  pointnr
  real(rk), intent(out)         ::  grad(curve%nDim)
  call curve%eval_grad(1, curve%pts_in_time(pointnr), grad)  
end subroutine

! subroutine to evaluate the derivative along the curve at certain times
! time means "value of the variable that parametrizes the curve"
subroutine eval_grad(curve, ntimesteps, times, grad)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  ntimesteps
  real(rk), intent(in)  ::  times(ntimesteps)
  real(rk), intent(out) ::  grad(curve%ndim,ntimesteps)
  integer               ::  titer, dimiter
  
  select type(curve)
  type is (curve_type)
  class is (GPR_curve)
    do titer=1,ntimesteps
      do dimiter=1, curve%nDim
        if (glob%spec((dimiter-1)/3+1)>=0) then
          ! free atom
          call gpr_eval_grad(curve%gprs(dimiter), times(titer),&
                             grad(dimiter,titer))
        else
          ! frozen atom
          grad(dimiter, titer) = 0d0
        end if
      end do
    end do
  class is (Spline_Curve)
    do dimiter=1, curve%nDim
      call dlf_eval_csplines_d(curve%nControlPoints, curve%params(dimiter,:,:), &
                            curve%distances, ntimesteps, times, grad(dimiter,:))
    end do    
  class default
    call dlf_fail("Unexpected type! (evaluate)")
  end select
end subroutine eval_grad

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* cinter/setEqDistPts
!!
!! FUNCTION
!!   Builds a NEW curve with equidistantly distributed points 
!!   along the old curve
!!
!! COMMENTS
!!   It is not the same curve after this!
!!
!! INPUTS
!!   nPoints - number of points to be distributed
!!   reuseExisting - use the existing points and start from them
!!                   (these are probably anyway the same as when I start by a
!!                   equidistant distribution in parametrization space,
!!                   like I do here)
!!
!! OUTPUTS 
!!   points 
!!     - The new control Points
!!
!! SYNOPSIS
subroutine setEqDistPts(curve, nPoints, points, eps)
  class(curve_type)     :: curve
  integer, intent(in)   :: nPoints
  real(rk)              :: init_timesOfPts(nPoints), &
                           timesOfPts(nPoints)
  real(rk),intent(out)  :: points(curve%nDim,nPoints)
  real(rk),intent(in)   :: eps
#ifdef DebugInfo  
  write(*,fmt='(A,ES10.3)') &
   "Making the points of the curve equidistant with convergence criterion", eps
#endif
  call curve%eqDistPts(nPoints, timesOfPts, points, eps)
  
  ! this rewrites the "timesOfPts" again to the standard 1/nPoints distribution
  ! in the parametrization space
  call curve%initialize(curve%nDim, nPoints, points)
  
end subroutine setEqDistPts

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* cinter/eqDistPts
!!
!! FUNCTION
!! Distibutes nPoints points equidistantly along the curve
!!
!! COMMENTS
!!
!! INPUTS
!! nPoints - number of points to be distributed
!!
!! OUTPUTS 
!! timesOfPts - curve-parameter-values of the equidistantly distributed points
!!
!! SYNOPSIS
subroutine eqDistPts(curve, nPoints, timesOfPts, points, eps, startingTimes)
  class(curve_type)     :: curve
  ! number of points that should be distributed equidistantly
  integer, intent(in)   :: nPoints
  ! the curve-parameter-values of the equidistantly distributed points
  real(rk), intent(out) :: timesOfPts(nPoints)
  real(rk)              :: timesOfPts_old(nPoints)
  real(rk)              :: timesOfPts_original(nPoints)
  ! the equidistantly distributed points on the curve
  real(rk), intent(out) :: points(curve%nDim,nPoints)
  real(rk),intent(in)   :: eps
  ! you can give some starting guess by giving the points on the curve
  ! in terms of the curve's parametrization
  real(rk), intent(in), optional &
                        :: startingTimes(nPoints)
  real(rk)              :: points_old(curve%nDim,nPoints)
  ! iteration variable for dimension
  integer               :: dimiter
  ! iteration variable for the points
  integer               :: piter
  ! Loss function for optimization
  real(rk)              :: L, L_v(1)
  ! Gradient of the loss function with respect to all points except the 
  ! fixed end-points
  real(rk)              :: dL(nPoints-2)
  ! step in the optimization
  real(rk)              :: step(nPoints)
  type(gpr_type)        :: gprL
  type(optimizer_type)  :: gprOptL
  logical               :: converged
  integer               :: optCounter, i, j
  ! maximum distance between two points at the beginning
  real(rk)              :: maxDist
  real(rk)              :: tolerance_addingPts
  real(rk)              :: damping
  real(rk)              :: Lmax, dLmax ! convergence criteria for the
                                       ! redistribution
  type(oop_lbfgs_type)  :: lbfgs_cinter
  logical               :: inverted
  integer               :: inverted_counter
  tolerance_addingPts = 20d-2 ! in percent
  Lmax = eps/2d0!1d-4/2d0
  dLmax = eps*5d0!1d-3/2d0
  damping = 1d1
  ! initialize with point distribution that is equidistant in the
  ! parametrization variable t, when no initial times are given
  if (present(startingTimes)) then
    timesOfPts(:) = startingTimes(:)
  else
    do piter = 1, nPoints
      timesOfPts(piter) = real(piter-1, kind=rk)*1d0/real(nPoints-1,kind=rk)
    end do
  end if
  points(:,1) = curve%controlPoints(:,1)
  do piter = 2, nPoints-1
    call curve%evaluate(1, timesOfPts(piter), points(:,piter))
  end do
  points(:,nPoints) = curve%controlPoints(:,curve%nControlPoints)
  ! Now optimize... it's a N-2 dimensional optimization problem
  
                         !nt,nat,      sdgf,meanOffset?,kernel,order
                         
                         
!   call GPR_construct(gprL, 0,  0, nPoints-2, 3         ,     1,1)
!   call manual_offset(gprL,1d1)
!                               ! gamma,s_f,s_n(3)
!   call GPR_init_without_tp(gprL,1d0/(1d0/real(nPoints,kind=rk))**2,1d0,(/1d-6,1d-6,3d-4/)) ! gamma=1/20^2 im paper
!   gprL%iChol = .true.
  ! not that using the new Cholesky decomposition (iterative),
  ! the multi-level scheme is completely irrelevant and all data will
  ! be ignored.
!   call GPR_Optimizer_define('MIN', gprOptL,gprL,9999,10,1d0/real(nPoints,kind=rk) )
  converged = .false.
  call curve%calc_L_and_dL(nPoints, timesOfPts, points, L, dL)
  optCounter = 1
  ! initialize lbfgs
  call lbfgs_cinter%init(nPoints-2, MAX(nPoints/2,2), .false.)
  step(:) = 0d0
  converged = (dsqrt(L)/nPoints<Lmax.and.MAXVAL(ABS(dL))<dLmax)
  timesOfPts_original(:) = timesOfPts(:)
  inverted_counter = 0
  do while (.not.converged)
    if(inverted_counter>10) then
      ! reset to original point, 
      ! add some new points on the curve and then try again, 
      ! also restart lbfgs
      if(printl>=6) write(stdout,&
        '("Inverted lbfgs step several times...", &
          "adding more points and try again")')
      call curve%addPtsToEqDist(tolerance_addingPts)
      timesOfPts(:) = timesOfPts_original(:)
      call lbfgs_cinter%destroy()
      call lbfgs_cinter%init(nPoints-2, MAX(nPoints/2,2), .false.)
      inverted_counter = 0
      ! if that also does not work, reset and add even more points -> decrease 
      ! the tolerance for adding points
      tolerance_addingPts = tolerance_addingPts*0.95d0
      optCounter = 1
      cycle
    end if
    L_v(1) = L
    timesOfPts_old(:) = timesOfPts(:)
    ! GPR opt
!     call GPR_Optimizer_step(gprL,gprOptL,timesOfPts_old(2:nPoints-1),&
!                             timesOfPts(2:nPoints-1),&
!                             timesOfPts_old(2:nPoints-1),L_v(1),dL)
!     step(:) = timesOfPts(:) - timesOfPts_old(:)
    
    ! LBFGS opt
    call lbfgs_cinter%next_step(timesOfPts_old(2:nPoints-1), &
                                dL(1:nPoints-2), step(2:nPoints-1),inverted)
    if (inverted) inverted_counter = inverted_counter + 1
    ! gradient descent
!     step(2:nPoints-1) = - 0.0000001d0 * dL(:)

      do i = 2, nPoints-1
        if (step(i)>0d0) then
          step(i) = MIN(step(i),(timesOfPts_old(i+1)-timesOfPts_old(i))/(damping))
        else 
          step(i) = MAX(step(i),(timesOfPts_old(i-1)-timesOfPts_old(i))/(damping))
        end if
    end do
    timesOfPts(:) = timesOfPts_old(:) + step(:)
    ! set new points at the new times
    points_old(:,:) = points(:,:)
    points(:,1) = curve%controlPoints(:,1)
    do piter = 2, nPoints-1
      call curve%evaluate(1, timesOfPts(piter), points(:,piter))
    end do
    points(:,nPoints) = curve%controlPoints(:,curve%nControlPoints)
    call curve%calc_L_and_dL(nPoints, timesOfPts, points, L, dL)
    ! check for convergence
    converged = (MAXVAL(ABS(dL))<dLmax)!.and.dsqrt(L)/nPoints<Lmax)
    optCounter = optCounter + 1
    if (optCounter>100000) then
      if(printl>=4) write(stdout,'("Could not make points equidistant.")')
      exit
    end if
  end do
  call lbfgs_cinter%destroy()
!   call GPR_destroy(gprL)
!   call GPR_Optimizer_destroy(gprOptL)
end subroutine eqDistPts

subroutine addPtsToEqDist(curve, allowedDeviation)
  class(curve_type)     :: curve
  real(rk), allocatable :: newPts(:,:)
  logical, allocatable  :: indicesAdd(:)
  real(rk)              :: avgDist
  real(rk),intent(in)   :: allowedDeviation
  integer               :: i, newPtIter, oldPtIter, newNrPts
  integer               :: nToAdd, nDim
  if(printl>=4) write(stdout, &
                '("Adding pts to make the set of points more equidistant.")')
  avgDist = 0d0
  do i = 1, curve%nControlPoints-1
    avgDist = avgDist + norm2(curve%controlPoints(:,i+1)-curve%controlPoints(:,i))
  end do
  avgDist = avgDist / (curve%nControlPoints-1)
  nToAdd = 1
!   do while(nToAdd>=1)
    nToAdd = 0
    ! check how many and where you should add points
    allocate(indicesAdd(curve%nControlPoints-1))
    indicesAdd(:) = .false.
    do i = 1, curve%nControlPoints-1
      if (norm2(curve%controlPoints(:,i+1)-curve%controlPoints(:,i))>&
          (1d0+allowedDeviation)*avgDist) then
        ! must add a point between point i and i+1
        indicesAdd(i) = .true.
        nToAdd = nToAdd + 1
      end if
    end do
    ! add points
    newNrPts = curve%nControlPoints+nToAdd
    if(printl>=4) write(stdout,'("New number of points", I10)') newNrPts
    allocate(newPts(curve%nDim,newNrPts))
    newPtIter = 1
    do oldPtIter = 1, curve%nControlPoints-1
      newPts(:,newPtIter) = curve%controlPoints(:,oldPtIter)
      newPtIter = newPtIter + 1
      if (indicesAdd(oldPtIter)) then
        newPts(:,newPtIter) = (curve%controlPoints(:,oldPtIter+1)+&
                               curve%controlPoints(:,oldPtIter))/2d0
        newPtIter = newPtIter + 1
      end if
    end do
    newPts(:,newPtIter) = curve%controlPoints(:,curve%nControlPoints)
    ! copy new points to the curve
    nDim = curve%nDim
    call curve%initialize(nDim, newNrPts,newPts)
    deallocate(newPts)
    deallocate(indicesAdd)
!   end do  
end subroutine addPtsToEqDist

subroutine addPtBetweenPts(curve, lPtInd,rPtInd,point)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  lPtInd, rPtInd
  real(rk), intent(in)  ::  point(curve%nDim)
  integer               ::  nDim, newNrPts
  real(rk)              ::  newPts(curve%nDim,&
                                   curve%nControlPoints+1)
  newPts(:,1:lPtInd) = curve%controlPoints(:,1:lPtInd)
  newPts(:,lPtInd+1) = point
  newPts(:,rPtInd+1:curve%nControlPoints+1) = &
        curve%controlPoints(:,rPtInd:curve%nControlPoints)
  ! copy new points to the curve
  nDim = curve%nDim
  newNrPts = curve%nControlPoints+1
  call curve%initialize(nDim, newNrPts,newPts)
end subroutine addPtBetweenPts

subroutine calc_L_and_dL(curve, nPoints, timesOfPts, points, L, dL)
  class(curve_type)     :: curve
  ! number of points that should be distributed equidistantly
  integer, intent(in)   :: nPoints
  ! the curve-parameter-values of the equidistantly distributed points
  real(rk), intent(in)  :: timesOfPts(nPoints)
  ! the equidistantly distributed points on the curve
  real(rk), intent(in)  :: points(curve%nDim,nPoints)
  ! Loss function for optimization
  real(rk), intent(out) :: L
  ! Gradient of the loss function with respect to all points except the 
  ! fixed end-points
  real(rk), intent(out) :: dL(nPoints-2)
  ! Delta between the points that are currently chosen along the curve
  real(rk)              :: D(nPoints-1)
  ! Change of Delta nr j, with respect to curve-parameter-value t nr j
  real(rk)              :: dDdt(nPoints-1)
  ! Change of Delta nr j+1, with respect to curve-parameter-value t nr j
  real(rk)              :: dDp1dt(nPoints-1)
  ! auxiliary vector
  real(rk)              :: tmpVec(curve%nDim), grad1d(curve%nDim,1)
  ! auxiliary scalar
  real(rk)              :: tmp, tmp2, dist
  ! auxialiary for absolute value
  real(rk)              ::  absv
  ! iteration variable for dimension
  integer               :: dimiter, dimiter2
  ! iteration variable for the points
  integer               :: piter
  integer               :: i,j
  ! calculate the distances
  do piter = 1, nPoints-1
    tmpVec(:) = points(:,piter+1)-points(:,piter)
    D(piter) = norm2(tmpVec)
  end do
  dDdt(:) = 0d0
  dDp1dt(:) = 0d0
  ! calculate distance derivatives
  do piter = 1, nPoints-1  
    call curve%eval_grad(1,timesOfPts(piter),grad1d(:,1))
    do dimiter = 1, curve%nDim      
        ! first all the dDdt
        if(piter<nPoints) then
          ! x_piter+1 - x_piter
          tmpVec(:) = points(:,piter+1)-points(:,piter)
          ! abs(x_piter+1 - x_piter)
          absv = dsqrt(dot_product(tmpVec,tmpVec))
      !     ! the prefactor
          tmp = 1d0 / absv
          dDdt(piter) = dDdt(piter) - tmp*tmpVec(dimiter)*grad1d(dimiter,1)
        end if
        
        ! then all the dDp1dt
        if(piter>1) then
          ! x_piter - x_piter-1
          tmpVec(:) = points(:,piter)-points(:,piter-1)
          ! abs(x_piter+1 - x_piter)
          absv = dsqrt(dot_product(tmpVec,tmpVec))
      !     ! the prefactor
          tmp = 1d0 / absv
          dDp1dt(piter) = dDp1dt(piter) + tmp*tmpVec(dimiter)*grad1d(dimiter,1)
        end if
    end do
  end do
    
  ! Now calculate L (the loss function for the optimization)
  !   L is basically just the sum over all differences between the
  !   neighbouring point distances. If it is zero, all points are 
  !   equidistantly distributed
  L = 0d0
  do piter = 1, nPoints-2
    L = L + (D(piter+1)-D(piter))**2
  end do
  
  ! Now calculate the derivatives/gradient of L
  dL(:) = 0d0
  do j = 2, nPoints-1
    i = j-1 ! index shift to match fortran form of 1,...,npoints-2
    
!     dL(i) = dL(i) + 2d0*abs(D(j)-D(j-1)) * dDdt(j)*(D(j)-D(j-1))/abs(D(j)-D(j-1))!*SIGN(1d0,D(j)-D(j-1))
!     dL(i) = dL(i) - 2d0*abs(D(j)-D(j-1)) * dDp1dt(j)*(D(j)-D(j-1))/abs(D(j)-D(j-1))!*SIGN(1d0,D(j)-D(j-1))
    dL(i) = dL(i) + 2d0 * dDdt(j)*(D(j)-D(j-1))!*SIGN(1d0,D(j)-D(j-1))
    dL(i) = dL(i) - 2d0 * dDp1dt(j)*(D(j)-D(j-1))!*SIGN(1d0,D(j)-D(j-1))
    
!     if(j>=3) dL(i) = dL(i) + 2d0*abs(D(j-1)-D(j-2)) * dDp1dt(j)*SIGN(1d0,D(j-1)-D(j-2))
!     
!     if (j<=nPoints-2) dL(i) = dL(i) - 2d0*abs(D(j+1)-D(j)) *  dDdt(j)*SIGN(1d0,D(j+1)-D(j))
    
    if(j>=3) dL(i) = dL(i) + 2d0*(D(j-1)-D(j-2)) * dDp1dt(j)
    
    if (j<=nPoints-2) dL(i) = dL(i) - 2d0*(D(j+1)-D(j)) *  dDdt(j)
  end do
end subroutine calc_L_and_dL

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* cinter/dlf_init_csplines
!!
!! FUNCTION
!! Generates nInput-1 cubic splines from nInput datapoints
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Uses external functino dgtsv to solve tridiagonal system
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! y(nInput) - datapoints for interpolation
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!!
!! OUTPUTS 
!! param(4,nInput-1) - array with spline parameters
!!
!! SYNOPSIS
subroutine dlf_init_csplines(nInput, y, h, param)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput ! = (n+1) -> nInput is number of datapoints
  real(rk), intent(in)   :: y(nInput) ! y axis (values at the points)
  real(rk), intent(in)   :: h(nInput-1) ! distances between the points
  real(rk), intent(out)  :: param(4,nInput-1)
  real(rk)               :: fpp(nInput) ! temporary variables to
                                        ! simplify the calculation
                                        ! fpp stands for f''                                        
  real(rk)               :: diag(nInput-2)
  real(rk)               :: subdiag(nInput-3)
  real(rk)               :: superdiag(nInput-3)
  real(rk)               :: rhs(nInput-2)
  integer                :: n ! Number of Splines that will be created
  integer                :: i, info
  ! one spline less then datapoints
  n = nInput - 1
! ======================================================================
! Calculate temporary variables fpp
! ======================================================================
  ! Natural boundary conditions:
  fpp(:) = 0d0
  fpp(n+1) = 0d0
  fpp(1) = 0d0
  diag(:) = 0d0
  subdiag(:) = 0d0
  superdiag(:) = 0d0
  ! Now calculate all other fpp entries
  ! For this purpose, calculate the matrix that will be inverted

  ! diagonal elements
  do i = 1, n-1
    diag(i) =  2D0*(h(i)+h(i+1))
  end do
  ! off-diagonal elements
  do i = 1, n-2
    subdiag(i) = h(i+1)
    superdiag(i) = h(i+1)
  end do   
  ! Right hand side
  rhs(:) = 0d0
  do i = 1, n-1
    rhs (i) = 6D0*((y(i+2)-y(i+1))/h(i+1)-(y(i+1)-y(i))/h(i)) 
  end do  
  call dgtsv(n-1, 1, subdiag, diag, superdiag, rhs, n-1, info)
! ======================================================================
! Calculate the spline parameters
! ======================================================================
  ! Constants of the spline
  fpp(2:n) = rhs(1:n-1)
  param(1,1:n) = y(1:n)
  do i = 1, n
    param(2,i) = (y(i+1)-y(i))/h(i)-h(i)/6D0*(fpp(i+1)+2*fpp(i))
  end do
  do i = 1, n
    param(3,i) = fpp(i)/2D0
  end do
  do i = 1, n
    param(4,i) = (fpp(i+1)-fpp(i))/(6D0*h(i))
  end do  
end subroutine dlf_init_csplines
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* cinter/dlf_eval_cspline
!!
!! FUNCTION
!! Evaluates splines
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Must be given same value of nInput
!! and the output of dlf_sct_init_csplines (parameters of the splines)
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! param(4,nInput-1) - array with spline parameters
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!! nevals - number of points that shall be evaluated
!! x(nevals) - points on which the spline should be evaluated
!!
!! OUTPUTS 
!! eval(nevals) - Result of the evaluation
!!
!! SYNOPSIS
subroutine dlf_eval_cspline(nInput, param, h, nevals, x, eval)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput, nevals ! = (n+1)
  real(rk), intent(in)   :: param(4,nInput-1)
  real(rk), intent(in)   :: h(nInput) ! distances between the points
                                      ! (last entry ignored)
  real(rk), intent(in)   :: x(nevals)
  real(rk), intent(out)  :: eval(nevals)
  real(rk)               :: pos
  integer                :: n ! Number of Splines
  integer                :: i,k
  ! splines are of the form 
  ! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
  n = nInput - 1
  do k = 1, nevals 
    ! evaluate all elements of x, d.h. evaluate at x(k)
    ! Throw error, if evaluation point is too small
    ! Area of definition begins with 0d0 but a tolerance
    ! of 1d-12 seems acceptable.
    if (x(k)<-1d-12) then
      call dlf_fail("Error: evaluation of spline not &
                      & in the area of definition! (too small)")
    end if  
    ! evaluate at position x(k)
    pos = 0d0 ! position of search
    i = 1
    ! n is number of splines
    do while(i<=n-1)
      if(x(k)<=pos+h(i)) then 
        ! evaluate Spline_i at x(k)
        eval(k) = param(1,i) + param(2,i)*(x(k)-pos) + &
                  param(3,i)*(x(k)-pos)**2 + param(4,i)*(x(k)-pos)**3
        exit
      else
        pos = pos + h(i)
        i = i + 1
      end if
    end do 
    ! Tolerance of 1d-12 for the last spline (right border value)
    if (i==n) then
      if (x(k)<= pos+h(n)+1d-12) then
        eval(k) = param(1,n) + param(2,n)*(x(k)-pos) + &
                  param(3,n)*(x(k)-pos)**2 + param(4,n)*(x(k)-pos)**3  
      else
        ! Throw error, if evaluation point is too high
        call dlf_fail("Error: evaluation of spline not &
                         in the area of definition (too high)!")
      end if
    end if
  end do
end subroutine dlf_eval_cspline
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_eval_csplines_d
!!
!! FUNCTION
!! Evaluates the derivative of splines
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Must be given same value of nInput
!! and the output of dlf_sct_init_csplines (parameters of the splines)
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! param(4,nInput-1) - array with spline parameters
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!! nevals - number of points that shall be evaluated
!! x(nevals) - points on which the spline should be evaluated
!!
!! OUTPUTS 
!! eval(nevals) - Result of the evaluation
!!
!! SYNOPSIS
subroutine dlf_eval_csplines_d(nInput, param, h, nevals, x, eval)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput, nevals ! = (n+1)
  real(rk), intent(in)   :: param(4,nInput-1)
  real(rk), intent(in)   :: h(nInput) ! distances between the points
  real(rk), intent(in)   :: x(nevals)
  real(rk), intent(out)  :: eval(nevals)
  real(rk)               :: pos
  integer                :: n ! Number of Splines 
  integer                :: i,k
  ! splines are of the form 
  ! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
  n = nInput - 1
  do k = 1, nevals 
    ! evaluate all elements of x, d.h. evaluate at x(k)
    ! Throw error, if evaluation point is too small
    if (x(k)<0d0) then
      call dlf_fail("Error: evaluation of spline not &
                     & in the area of definition! (too small)")
    end if  
    ! evaluate at position x(k)
    pos = 0d0 ! position of search
    i = 1
    do while(i<=n)
      if(x(k)<=pos+h(i)) then 
        ! evaluate Spline_i at x(k)
        eval(k) = param(2,i) + 2D0*param(3,i)*(x(k)-pos) &
                             + 3D0*param(4,i)*(x(k)-pos)**2
        exit
      else
        pos = pos + h(i)
      end if
      i = i + 1
    end do    
    ! Throw error, if evaluation point is too high
    if(i==n+1) then
      call dlf_fail("Error: evaluation of spline not &
                      & in the area of definition (too high)!")
    end if
  end do
end subroutine dlf_eval_csplines_d
!!****


subroutine build1DExample(curve)
  class(curve_type) ::  curve
  integer           ::  nDim_in = 1
  integer           ::  nControlPoints_in = 20
  real(rk)          ::  coords(1,20)
  integer           ::  titer
  
  do titer = 1, nControlPoints_in
    coords(1,titer) = (real(titer-1,kind=rk))**2d0
  end do
  call initialize(curve, nDim_in, nControlPoints_in, coords)
end subroutine build1DExample

subroutine build3Dheartshape(curve)
  class(curve_type) ::  curve
  integer           ::  nDim_in = 3
  integer           ::  nControlPoints_in = 200
  real(rk)          ::  coords(3,200)
  integer           ::  titer
  
  do titer = 1, nControlPoints_in
    !3,14*2=6,28
    call heartshape(real(titer-1,kind=rk)*6.28d0/real(nControlPoints_in,kind=rk),&
                    coords(:,titer))
  end do
  
  call initialize(curve, nDim_in, nControlPoints_in, coords)
end subroutine build3Dheartshape

subroutine heartshape(u,x)
  real(rk), intent(in)  ::  u
  real(rk), intent(out) ::  x(3)
   x(1) =4d0*(sin(u))**3d0;
   x(2) = 0.25d0*(13d0*cos(u)-5d0*cos(2d0*u)-2d0*cos(3d0*u)-cos(4d0*u));
   x(3) = sin(u);
end subroutine heartshape

subroutine destroy_cinter(curve)
  class(curve_type)     :: curve
  
  if (allocated(curve%controlPoints)) deallocate(curve%controlPoints)
  if (allocated(curve%pts_in_time)) deallocate(curve%pts_in_time)
  
  select type(curve)
  type is (curve_type)
  class is (GPR_curve)
    if(allocated(curve%gprs)) deallocate(curve%gprs)  
  class is (Spline_Curve)
    if(allocated(curve%params)) deallocate(curve%params)  
    if(allocated(curve%distances)) deallocate(curve%distances)
  class default
    STOP "Unexpected type! (evaluate)"
  end select
end subroutine destroy_cinter

subroutine timeOfCPoint(curve, ptNumber, time)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  ptNumber
  real(rk), intent(out) ::  time
  time = curve%pts_in_time(ptNumber)
end subroutine timeOfCPoint

subroutine setEndPoints(curve, startPt, endPt, nExclPts, exclPts)
!   use mod_gprmep, only: gprmep_type
  class(curve_type)             ::  curve
  real(rk), intent(in)          ::  startPt(curve%nDim), endPt(curve%nDim)
  integer, intent(out)          ::  nExclPts
  real(rk), allocatable, intent(out)::  exclPts(:,:)
!   class(gprmep_type),optional   ::  gprmep
  real(rk)                      ::  points(curve%nDim,curve%nControlPoints)
  integer                       ::  i,j, s_near, e_near
  real(rk)                      ::  t_NewStart, t_NewEnd, dt
  real(rk)                      ::  s_dist, e_dist
  if (allocated(exclPts)) deallocate(exclPts)
  ! Find the control points that are closest to the start and the end point
  ! evaluate all points
  call curve%evaluate(curve%nControlPoints, curve%pts_in_time, points)
  s_dist = norm2(points(:,1)-startPt(:))
  e_dist = norm2(points(:,curve%nControlPoints)-endPt(:))
  s_near = 1
  e_near = curve%nControlPoints
  do i = 2, MAX(curve%nControlPoints,2)
    if (norm2(points(:,i)-startPt(:))<s_dist) then
      s_near = i
      s_dist = norm2(points(:,i)-startPt(:))
    end if
  end do
  do i = 2, MAX(curve%nControlPoints,2)
    if (norm2(points(:,curve%nControlPoints+1-i)-endPt(:))<e_dist) then
      e_near = curve%nControlPoints+1-i
      e_dist = norm2(points(:,curve%nControlPoints+1-i)-endPt(:))
    end if
  end do
  nExclPts = (curve%nControlPoints-e_near)+(s_near-1)
  if (nExclPts>0) then
    allocate(exclPts(curve%nDim,nExclPts))
    j = 1
    do i = 1, s_near-1
      exclPts(:,j) = points(:,i)
      j = j + 1
    end do
    do i = e_near, curve%nControlPoints-1
      exclPts(:,j) = points(:,i)
      j = j + 1
    end do
  end if
  ! s_near and e_near are the points which are closest to the intended
  ! start and end point
  ! now distribute points between these 
  t_NewStart = curve%pts_in_time(s_near)
  t_NewEnd = curve%pts_in_time(e_near)
  dt = (t_NewEnd-t_NewStart)/(curve%nControlPoints-1)
  do i = 1, curve%nControlPoints
    curve%pts_in_time(i) = t_NewStart + (i-1)*dt
  end do  
  ! evaluate all points on the new times again
  call curve%evaluate(curve%nControlPoints, curve%pts_in_time, points)
  ! overwrite the endpoints with the respective start/endpoint
  points(:,1)                    = startPt(:)
  points(:,curve%nControlPoints) = endPt(:)
  ! re-initialize the curve
  call curve%initialize(curve%nDim, curve%nControlPoints,points)
end subroutine

subroutine writePlotData(curve, nPlottingPoints,plotnr)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  nPlottingPoints
  integer, intent(in)   ::  plotnr
  integer               ::  i, j, k
  real(rk)              ::  val, times(nPlottingPoints)
  real(rk)              ::  points(curve%nDim,nPlottingPoints)
  character*14          ::  filename
  character*15          ::  filename2
  character*16          ::  filename3
  if (plotnr>9999) STOP "Too much plots."
  WRITE(filename,'(a,i4.4,a)') "cPlot",plotnr,".plot"
  open(unit=101,file=filename,status='UNKNOWN', action="write")
#ifdef TestOnMB  
  WRITE(filename2,'(a,i4.4,a)') "c2Plot",plotnr,".plot"
  WRITE(filename3,'(a,i4.4,a)') "pc2Plot",plotnr,".plot"
  open(unit=102,file=filename2,status='UNKNOWN', action="write")
  open(unit=103,file=filename3,status='UNKNOWN', action="write")
#endif  
  
  select type (curve)
  type is (curve_type)
  class is (GPR_curve)
  if(printl>=6) write(stdout,'("Plotinfo: writing to file ", A)') filename
  do j = 1, curve%nDim
!   do j = curve%nDim, curve%nDim
    do i = 1, nPlottingPoints      
      times(i) = REAL(i-1,kind=rk)/REAL(nPlottingPoints-1,kind=rk)
      call GPR_eval(curve%gprs(j), times(i),val)
      write(101, '(i4.4,1x,i4.4,1x,ES10.3)') j, i, val
    end do
  end do
#ifdef TestOnMB
  call curve%evaluate(nPlottingPoints, times, points)
  do i = 1, nPlottingPoints
    write(102, '(i4.4,1x,ES10.3,1x,ES10.3)') i, points(1:2,i)
  end do
  do i = 1, curve%nControlPoints
    times(i) = real(i-1,kind=rk)/real(curve%nControlPoints-1,kind=rk)
    call curve%evaluate(1, times(i), points(:,i))  
    write(103, '(i4.4,1x,ES10.3,1x,ES10.3)') i, points(1:2,i)
  end do
#endif  
  class is (Spline_Curve)
    STOP "Not implemented for splines at the moment. (writePlotData)"
  class default
    STOP "This class is not specified. (writePlotData)"
  end select
  close(101)
  close(102)
  close(103)
end subroutine writePlotData

! markThesePts is a vector of points that 
! should be written in a special file (helps to mark them
! in gnuplot for example)
subroutine writePlotData_extra(curve, nPlottingPoints,plotnr, &
                nMarkPts, markThesePts)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  nPlottingPoints
  integer, intent(in)   ::  plotnr
  integer, intent(in)   ::  nMarkPts
  real(rk),intent(in)   ::  markThesePts(curve%nDim,nMarkPts)
  integer               ::  i, j, k
  real(rk)              ::  val, times(nPlottingPoints)
  real(rk)              ::  points(curve%nDim,nPlottingPoints)
  character*14          ::  filename
  character*15          ::  filename2
  character*16          ::  filename3
  character*16          ::  filename4
  if (plotnr>9999) STOP "Too much plots."
  WRITE(filename,'(a,i4.4,a)') "cPlot",plotnr,".plot"
  open(unit=101,file=filename,status='UNKNOWN', action="write")
#ifdef TestOnMB  
  WRITE(filename2,'(a,i4.4,a)') "c2Plot",plotnr,".plot"
  WRITE(filename3,'(a,i4.4,a)') "pc2Plot",plotnr,".plot"
  open(unit=102,file=filename2,status='UNKNOWN', action="write")
  open(unit=103,file=filename3,status='UNKNOWN', action="write")
  WRITE(filename4,'(a,i4.4,a)') "ec2Plot",plotnr,".plot"
  open(unit=104,file=filename4,status='UNKNOWN', action="write")
#endif  
  
  select type (curve)
  type is (curve_type)
  class is (GPR_curve)
  if(printl>=6) write(stdout,'("Plotinfo: writing to file ", A)') filename
  do j = 1, curve%nDim
!   do j = curve%nDim, curve%nDim
    do i = 1, nPlottingPoints      
      times(i) = REAL(i-1,kind=rk)/REAL(nPlottingPoints-1,kind=rk)
      call GPR_eval(curve%gprs(j), times(i),val)
      write(101, '(i4.4,1x,i4.4,1x,ES10.3)') j, i, val
    end do
  end do
#ifdef TestOnMB
  call curve%evaluate(nPlottingPoints, times, points)
  do i = 1, nPlottingPoints
    write(102, '(i4.4,1x,ES10.3,1x,ES10.3)') i, points(1:2,i)
  end do
  do i = 1, curve%nControlPoints
    times(i) = real(i-1,kind=rk)/real(curve%nControlPoints-1,kind=rk)
    call curve%evaluate(1, times(i), points(:,i))  
    write(103, '(i4.4,1x,ES10.3,1x,ES10.3)') i, points(1:2,i)
  end do
  do i = 1, nMarkPts 
    write(104, '(i4.4,1x,ES10.3,1x,ES10.3)') i, markThesePts(1:2,i)
  end do  
#endif  
  class is (Spline_Curve)
    STOP "Not implemented for splines at the moment. (writePlotData)"
  class default
    STOP "This class is not specified. (writePlotData)"
  end select
  close(101)
  close(102)
  close(103)
  close(104)
end subroutine writePlotData_extra

subroutine writeVectorPlots(curve, nVecs, plotnr, vecs, maxVeclength)
  class(curve_type)     ::  curve
  integer, intent(in)   ::  nVecs
  integer, intent(in)   ::  plotnr
  real(rk), intent(in)  ::  vecs(3,nVecs)
  real(rk), intent(in)  ::  maxVeclength  
  real(rk)              ::  val, times(nVecs)
  real(rk)              ::  points(curve%nDim,nVecs)
  real(rk)              ::  maxLength
  character*16          ::  filename
  integer               ::  i,j
  if (plotnr>9999) call dlf_fail("Too much plots.") 
  if(printl>=4) write(stdout,'("writing plot data nr. ", I10)') plotnr
  select type (curve)
  type is (curve_type)
  class is (GPR_curve)
  
  do j = 1, curve%nDim
!   do j = curve%nDim, curve%nDim
    do i = 1, nVecs      
      times(i) = REAL(i-1,kind=rk)/REAL(nVecs-1,kind=rk)
      call GPR_eval(curve%gprs(j), times(i),val)
    end do
  end do
  call curve%evaluate(nVecs, times, points)
  WRITE(filename,'(a,i4.4,a)') "vc2Plot",plotnr,".plot"
  open(unit=101,file=filename,status='UNKNOWN', action="write")
  if(printl>=4) write(stdout,'( "VectorPlot: writing to file ", A)') filename
!   do j = 1, curve%nDim
  maxLength=0d0
  do i = 1, nVecs
    if (maxLength<norm2(vecs(:,i))) maxLength = norm2(vecs(:,i))
  end do
  do i = 1, nVecs
    write(101, '(ES10.3,1x,ES10.3,1x,ES10.3,1x,ES10.3,1x)') &
      points(1:2,i), (vecs(1:2,i)/maxLength)*maxVeclength
  end do
  close(101)
  class is (Spline_Curve)
    call dlf_fail("Not implemented for splines at the moment. (writePlotData)")
  class default
    call dlf_fail("This class is not specified. (writePlotData)")
  end select
end subroutine writeVectorPlots

subroutine calcDiffToCurve(curve, curve2, diff, maxdiff)
  class(curve_type)             ::  curve
  type(GPR_curve), intent(in)   ::  curve2
  real(rk), intent(out)         ::  diff    ! difference between two curves
  real(rk), intent(out)         ::  maxdiff ! maximum difference between two curves
  real(rk)                      ::  tmpvec(curve%nDim), tmpvec2(curve%nDim)
  real(rk)                      ::  tmpdiff
  integer                       ::  nMsrPts ! # points to measure the difference
  integer                       ::  i, j
  real(rk)                      ::  time(1)
  select type (curve)
  type is (curve_type)
  class is (GPR_curve)
  ! ********* based on spatial differences between points along the curve *****
  diff = 0d0
  maxdiff = 0d0
  nMsrPts = 100
  do i = 1, nMsrPts
    time(1) = real(i-1,kind=rk)/real(nMsrPts-1,kind=rk)
    call curve2%evaluate(1, time(1), tmpvec2)
    call curve%evaluate(1, time(1), tmpvec)
    tmpdiff = norm2(tmpvec2 - tmpvec)!sum(tmpvec2 - tmpvec)
    diff = diff + tmpdiff
    if (tmpdiff>maxdiff) maxdiff = tmpdiff
  end do
  diff = diff / nMsrPts!dsqrt(real(nMsrPts,kind=rk))

  class is (Spline_Curve)
    ! ****************************************************
    ! Spline curve
    call dlf_fail("calcDiffToCurve not implemented with splines!")
  class default
    call dlf_fail("Unexpected type! (initialize)")
  end select
end subroutine calcDiffToCurve

! Smoothes the curve by choosing a higher s_n (tolerance for noise in the data)
! The higher s_n the smoother the curve, but the more meaningless it becomes
subroutine smooth(curve, s_n, points)
  class(curve_type)     ::  curve
  real(rk), intent(in)  ::  s_n
  ! control points will be used by the program calling this function
  real(rk), intent(out) ::  points(curve%nDim,curve%nControlPoints) 
  real(rk)              ::  old_s_n
  select type (curve)
  type is (curve_type)
  class is (GPR_curve)
    old_s_n = curve%s_n(1)
    call curve%setGPRParameters(curve%gamma, curve%s_f, &
                                (/ s_n, curve%s_n(2), curve%s_n(3) /))
    call curve%initialize(curve%nDim, curve%nControlPoints, curve%controlPoints)
    call curve%setGPRParameters(curve%gamma, curve%s_f, &
                                (/ old_s_n, curve%s_n(2), curve%s_n(3)/) )
!     call curve%initialize(curve%nDim, curve%nControlPoints, curve%controlPoints)
    call curve%setEqDistPts(curve%nControlPoints, points,1d-4)
  class is (Spline_Curve)
    call dlf_fail("smooth not implemented for splines.")
  class default
    call dlf_fail("This curve type is not implemented in smooth.")
  end select
end subroutine smooth

subroutine backupCurve(curve)
  class(curve_type)     ::  curve
  character*12          ::  filename
  integer               ::  i,j
  if(printl>=4) write(stdout,'("Saving the curve...")')
  WRITE(filename,'(a)') "Curve.backup"
  open(unit=101,file=filename,status='UNKNOWN', action="write")
  write(101, '(2i5.5)') curve%nDim, curve%nControlPoints
  do i = 1, curve%nControlPoints
    do j = 1, curve%nDim
      write(101,'(ES16.9)') curve%controlPoints(j,i)
    end do
  end do
  close(101)
end subroutine backupCurve

subroutine readInCurve(curve)
  class(curve_type)     ::  curve
  character*12          ::  filename
  integer               ::  i,j
  integer               ::  nDim_local, nCPts_local
  real(rk), allocatable ::  controlPoints_local(:,:)
  if(printl>=4) write(stdout,'("Reading the curve.")')
  WRITE(filename,'(a)') "Curve.backup"
  open(unit=101,file=filename,status='UNKNOWN', action="read")
  read(101, '(2i5.5)') nDim_local, nCPts_local
  allocate(controlPoints_local(nDim_local,nCPts_local))
  do i = 1, nCPts_local
    do j = 1, nDim_local
      read(101,'(ES16.9)') controlPoints_local(j,i)
    end do
  end do
  close(101)  
  call curve%initialize(nDim_local, nCPts_local, controlPoints_local)
  deallocate(controlPoints_local)
end subroutine readInCurve

subroutine avoidSmallDistance(curve, limit, points)
  class(curve_type)       ::  curve
  real(rk), intent(in)    ::  limit ! when the distance somewhere is 
                                  ! smaller than limit redist
  real(rk), intent(inout) ::  points(curve%nDim,curve%nControlPoints)
  logical                 ::  redist
  integer                 ::  i
  redist = .false.
  do i = 1, curve%nControlPoints-1
    if (norm2(curve%controlPoints(:,i+1)-curve%controlPoints(:,i))<limit) then
      redist = .true.
      exit
    end if
  end do
  if (redist) then
    if(printl>=4) write(stdout,'("avoidSmallDistance: ",&
            "Some points are too close, redistribution of points necessary.")')
    call curve%setEqDistPts(curve%nControlPoints, points,1d-4)    
  else
    if(printl>=6) &
        write(stdout,'("avoidSmallDistance: No redistribution necessary.")')
  end if
end subroutine avoidSmallDistance

subroutine avoidDeviationEndPts(curve, startPt, endPt, limit, nExclPts, exclPts)
  class(curve_type)             ::  curve
  real(rk), intent(in)          ::  startPt(curve%nDim), endPt(curve%nDim)
  real(rk), intent(in)          ::  limit
  integer, intent(out)          ::  nExclPts
  real(rk), allocatable, intent(out)::  exclPts(:,:)
  nExclPts = 0
  if (norm2(startPt(:)-curve%controlPoints(:,1))>limit.or.&
      norm2(endPt(:)  -curve%controlPoints(:,curve%nControlPoints))>limit) then
      if(printl>=6) write(stdout,&
        '("Deviation of end points is too large, fix them...")')
      call curve%setEndPoints(startPt, endPt, nExclPts, exclPts)
  else
  if(printl>=6) write(stdout,'("Deviation of end points are in the limit.")')
  end if
end subroutine avoidDeviationEndPts

end module cinter_module
