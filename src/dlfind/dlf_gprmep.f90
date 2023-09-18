! **********************************************************************
! **                       MEP search with GPR                        **
! **********************************************************************
!!****h* DL-FIND/gprmep
!!
!! NAME
!! gprmep
!!
!! FUNCTION
!! Performs an MEP search based on GPR
!!
!! Inputs
!!    starting guess for the MEP
!! 
!! Outputs
!!    MEP
!!
!! COMMENTS
!!    -
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

! #define TIMING

! Also set this flag in dlf_cinter.f90 to see plots
! #define TestOnMB

! #define setPlotOn
! #define plotNEBs

#define optPtsInsteadOfWeights
! only with optPtsInsteadOfWeights
#define projectOut
#define checkAndCorrectDistances


module gprmep_module
use gpr_module
use gpr_types_module
use cinter_module
use dlf_parameter_module, only: rk
use dlf_global, only: glob
use geodesic_module
use dlf_global, only: stdout, stderr, printl
#ifdef TIMING
use mod_oop_clock
#endif
implicit none

type pt_list
  real(rk), allocatable         ::  coords(:,:) ! coords of the point
  integer                       ::  np = 0      ! # points remaining in list
  integer                       ::  ptIter = 0
end type pt_list

type gprmep_type
  integer                       ::  nOpts ! number of optimization steps
  logical                       ::  interpolated
  logical                       ::  parameter_init = .false.
  integer                       ::  nDim
  integer                       ::  nPoints
  real(rk), allocatable         ::  points(:,:)
  real(rk), allocatable         ::  oldPoints(:,:)
  integer                       ::  nat, varperimage
  real(rk)                      ::  S_0,S_pot,S_ins
  real(rk)                      ::  temperature
  real(rk), allocatable         ::  ene(:) !nimage
  real(rk), allocatable         ::  xcoords_readin(:,:) !3*nat,nimage
  real(rk), allocatable         ::  dtau(:)  !nimage+1
  real(rk)                      ::  etunnel
  real(rk), allocatable         ::  dist(:)  !nimage+1
  integer                       ::  status ! 0: nothing done
                                           ! 1: pts distributed
  logical                       ::  endPtsCalculated
  integer                       ::  ptCalc ! iterator for pts
  real(rk)                      ::  maxstep ! maximally allowed step size
  type(GPR_curve)               ::  curve
  type(GPR_curve)               ::  old_curve
  type(GPR_type)                ::  gpPES
  integer                       ::  n_discrete = 100
                                               ! nr of points to 
                                               ! discretize the path
                                               ! for calculating the
                                               ! loss function
  real(rk), allocatable         ::  d_times(:) ! discrete times for 
                                               ! calculating the 
                                               ! loss function
  real(rk), allocatable         ::  vecsAlongCurve(:,:)
                                               ! vectors that are tangents
                                               ! on the curve at the
                                               ! points discretizing 
                                               ! the curve
  real(rk), allocatable         ::  vecsAlongCurve_n(:,:)
                                               ! same but normalized
  real(rk), allocatable         ::  coords_discrete(:,:)
  real(rk), allocatable         ::  esAtDiscretePts(:)
  real(rk), allocatable         ::  gradsAtDiscretePts(:,:)
                                               ! gradients (GP-PES) at the 
                                               ! discretization points
                                               ! (projected out tangents)
  real(rk), allocatable         ::  hessiansAtDiscretePts(:,:,:)
                                               ! Hessians (GP-PES) at the 
                                               ! discretization points
                                               ! (projected out tangents)
  integer                       ::  plotnr
  type(pt_list)                 ::  ptToCalcList
  real(rk)                      ::  variance_limit
  real(rk), allocatable         ::  react(:)
  real(rk), allocatable         ::  prod(:)
  integer, allocatable          ::  znuc(:)
  integer                       ::  minNrPts, maxNrPts, increaseNrPts
  integer                       ::  opt_state
                                    ! 0: standard GPRMEP
                                    ! 1: TS search
                                    ! 2: GPRMEP after TS search
  real(rk)                      ::  maxcChangeConv
  real(rk)                      ::  cChangeConv
  real(rk)                      ::  maxPathUpdate
  real(rk)                      ::  maxStepSize
  real(rk)                      ::  varLimit_start
  real(rk)                      ::  varLimit_end
  real(rk)                      ::  var_allow_cConv
  real(rk)                      ::  tolg
  real(rk)                      ::  tole
  real(rk)                      ::  tolrmsg,tols,tolrmss
  logical                       ::  unitsOpen
  integer                       ::  unitp
  logical                       ::  mepConverged
  real(rk), allocatable         ::  TS(:)
  real(rk), allocatable         ::  oldTSGuess(:)
  type(optimizer_type)          ::  opt
  integer                       ::  maxNrPtsInLevel
  integer                       ::  shiftToLowerLevel
  real(rk)                      ::  globtol
  integer                       ::  lOfTS, rOfTS
  logical                       ::  tsSearchJustStarted
  integer                       ::  gprmep_mode ! = 0 : just one path opt
                                    ! = 1: additional precise gprts search
                                    ! = 2: after gprts search path opt again
#ifdef TIMING
  type(clock_type)              ::  overallClock
#endif
  real(rk)                      ::  overallTime2, overallTime1
#ifdef TIMING
  type(clock_type)              ::  clock, clock2
#endif
  ! stuff needed for gprmep_loss_parsAndGrad
  real(rk), allocatable         ::  pForceNorm2_discretePts(:),&
                                    tNorms_at_dPts(:), &
                                    projMat(:,:,:), &
                                    tmpMat(:,:,:)
  logical                       ::  parasInit = .false.
  integer                       ::  nOpts_eBased
  integer                       ::  fpp_nT
  integer                       ::  maxPtsToCalc
!   real(rk)                      ::  manual_e_g_loss = -1d0
!   logical                       ::  smoothing
  ! For calculating the energies of the final path
  logical                       ::  finalEnergiesToCalc
  integer                       ::  nFinalEnergies
  real(rk), allocatable         ::  energiesAtFinalPts(:)
contains
  ! read the initial path, for example from NEB in IDPP
  procedure                     ::  readQtsCoords => gprmep_readQtsCoords
  ! start an initial path via the minima -> linear starting path
  procedure                     ::  initWithMinima => gprmep_initWithMinima
  ! init with a given Path from .xyz file
  procedure                     ::  initFromPath => gprmep_initFromPath
  ! initialize the standard paras for the optimization
  procedure                     ::  initStandardParas => gprmep_initStandardParas
  ! initialize and allocate parameters for the path and GPR-PES
  procedure                     ::  init_and_alloc => gprmep_init_and_alloc
  ! give the next necessary energy evaluation to DL-FIND
  procedure                     ::  giveNextPt => gprmep_giveNextPt
  ! gprmep_destroy all allocated objects
  procedure                     ::  destroy => gprmep_destroy
  ! check convergence criteria
  procedure                     ::  checkConvergence => gprmep_checkConvergence
  ! signal that convergence criteria are satisfied
  procedure                     ::  signalConverged => gprmep_signalConverged
  ! calculate necessary stuff for grad based loss function
  procedure                     ::  loss_parsAndGrad => gprmep_loss_parsAndGrad
  ! add points to the ptToCalcList
  procedure                     ::  add_PtsToCalc => gprmep_add_PtsToCalc
  ! add a point to the ptToCalcList if its variance is high enough
  procedure                     ::  add_PtWithVarianceCheck => &
                                    gprmep_add_PtWithVarianceCheck
  ! get the next point to calculate from ptToCalcList
  procedure                     ::  checkout_PtToCalc => gprmep_checkout_PtToCalc
  ! change the number of points
  procedure                     ::  change_nInterPts => gprmep_change_nInterPts
  ! fill the list of "to be calculated" points with points of high variance
  procedure                     ::  fillListAccordingToVariance => gprmep_fillListAccordingToVariance
  ! change the optimization to more tight convergence criteria (post-opt)
  procedure                     ::  changeToPostOpt => gprmep_changeToPostOpt
  ! check if the list of points to calculate still satisfies the variance
  ! criterion
  procedure                     ::  checkPtsToCalcForVarChanges => gprmep_checkPtsToCalcForVarChanges
  ! gives the balancing factor between energy and gradient based Loss function
  procedure                     ::  weight_e_g_Loss => gprmep_weight_e_g_Loss
  ! writes files to track the development of the MEP
  procedure                     ::  saveImage => gprmep_saveImage
  ! rotating the images along the path to minimize the rotational
  ! changes along the MEP
  procedure                     ::  rotateBack => gprmep_rotateBack
  ! TS search in GPRMEP
  procedure                     ::  ts_search_init => gprmep_ts_search_init
  procedure                     ::  ts_search_wrapUp => gprmep_ts_search_wrapUp
  procedure                     ::  randomizeUntilVarLow => gprmep_randomizeUntilVarLow
  procedure                     ::  testTSconv => gprmep_testTSconv
  ! calculate the final path energies after optimization for example
  procedure                     ::  calcFinalPathEnergies => gprmep_calcFinalPathEnergies
end type gprmep_type
  ! this instance of gprmep_type is used in dl-find
  type(gprmep_type),save        ::  gprmep_instance
  
contains
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_readQtsCoords
!!
!! FUNCTION
!! read a starting guess for gprmep optimization from qts_coords.txt
!!
!! SYNOPSIS
subroutine gprmep_readQtsCoords(gprmep, filename, maxstep_in, globtol_in,gprmep_mode)
!! SOURCE  
  class(gprmep_type)        ::  gprmep
  character(*), intent(in)  ::  filename
  real(rk), intent(in)      ::  maxstep_in !maxstep is not used at the moment
  real(rk), intent(in)      ::  globtol_in
  integer, intent(in)       ::  gprmep_mode
  logical                   ::  there
  character(128)            ::  line
  integer                   ::  ios, i
  real(rk)                  ::  readBuffScalar
  real(rk), allocatable     ::  readBuffVec(:)
#ifdef TestOnMB
  real(rk)                  ::  rsMB(3),psMB(3), gMB(3),lastMB(3),stepMB(3),&
                                newMB(3), lastEMB, hMB(3,3)
  integer                   ::  mbstepcounter
#endif
  gprmep%maxstep = maxstep_in
  gprmep%nOpts = 0
  gprmep%globtol = globtol_in
  gprmep%gprmep_mode = gprmep_mode
  if (printl>=6) write(stdout,'("global tolerance value", ES11.4)') globtol_in
#ifndef TestOnMB
!   gprmep%smoothing = .true.
  if (printl>=4) write(stdout,'("Reading initial MEP guess from file ",A)') filename
  inquire(file=filename,exist=there)
  if(.not.there) call dlf_fail("file for initial MEP guess does not exist!")
#endif
#ifdef TestOnMB
  gprmep%nat = 1
  gprmep%nPoints = 10 !MB
#endif
#ifndef TestOnMB
  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200)
  read(555,*,end=201,err=200) gprmep%nat, gprmep%nPoints, gprmep%varperimage
#endif
  call gprmep%initStandardParas()
!   gprmep%nPoints = gprmep%nPoints - 1

  gprmep%nDim = gprmep%nat * 3 ! this is true for cartesians
  
  allocate(readBuffVec(gprmep%nDim))
#ifndef TestOnMB
  read(555,*,end=201,err=200) gprmep%temperature
  !read(555,*,end=201,err=200) S_0 
  read(555,fmt="(a)") line
  read(line,*,iostat=ios) gprmep%S_0,gprmep%S_pot
  if(ios/=0) then
    read(line,*) gprmep%S_0
  end if
  read(555,*,end=201,err=200) gprmep%S_ins
  if(ios/=0) then
    gprmep%S_pot=gprmep%S_ins-0.5D0*gprmep%S_0
    if (printl>=6) write(stdout,'("Warning: could not read S_pot from qts_coords.txt")')
  end if
#endif

  if (allocated(gprmep%ene)) deallocate(gprmep%ene)
  if (allocated(gprmep%xcoords_readin)) deallocate(gprmep%xcoords_readin)
  if (allocated(gprmep%dtau)) deallocate(gprmep%dtau)
  if (allocated(gprmep%dist)) deallocate(gprmep%dist)
  
  allocate(gprmep%ene(gprmep%nPoints))
  allocate(gprmep%xcoords_readin(gprmep%nDim,gprmep%nPoints))
  allocate(gprmep%dtau(gprmep%nPoints+1))
  allocate(gprmep%dist(gprmep%nPoints+1))
  
  call gprmep%init_and_alloc()

#ifdef TestOnMB
  rsMB=(/ -0.050010822944531706D0, 0.4666941048659066D0 , 0.D0 /)
  psMB=(/-0.5582236346340204D0,   1.4417258418038705D0,   0.0D0 /)
#endif

#ifndef TestOnMB
  read(555,*,end=201,err=200) gprmep%ene(1:gprmep%nPoints), readBuffScalar
  read(555,*,end=201,err=200) gprmep%xcoords_readin(1:gprmep%nDim,1:gprmep%nPoints), readBuffVec
  ! Necessary because the first point seems to be the last point again in qts_path.txt
  gprmep%nPoints = gprmep%nPoints - 1
#endif
  open(unit=104,file='initialPath.plot',status='UNKNOWN', action="write")
  do i = 1, gprmep%nPoints
#ifdef TestOnMB
    gprmep%xcoords_readin(:,i) = rsMB(:)+&
        (psMB(:)-rsMB(:))& !MB
        *real(i-1,kind=rk)/real(gprmep%nPoints-1,kind=rk)
#endif
    write(104,'(ES10.3,1x,ES10.3)') gprmep%xcoords_readin(1:gprmep%nDim,i)
  end do
  close(104)
#ifndef TestOnMB  
  ! Necessary because the first point seems to be the last point again in qts_path.txt
  gprmep%nPoints = gprmep%nPoints + 1
  
  ! try and read dtau (not here in old version, and we have to stay consistent)
  read(555,fmt="(a)",iostat=ios) line
  if(ios==0) then
    read(555,*,end=201,err=200) gprmep%dtau(1:1+gprmep%nPoints), readBuffScalar
    read(555,*,end=201,err=200) gprmep%etunnel
    read(555,*,end=201,err=200) gprmep%dist(1:1+gprmep%nPoints), readBuffScalar
  else
    if (printl>=6) write(stdout,'("Warning, dtau not read from qts_coords.txt, using constant dtau")')
    gprmep%dtau=-1.D0
    gprmep%etunnel=-1.D0  
    gprmep%dist(:)=-1.D0 ! set to sueless value to flag that it was not read
  end if

! Necessary because the first point seems to be the last point again in qts_path.txt
  gprmep%nPoints = gprmep%nPoints - 1
  close(555)
#endif    
  ! rewrite xcoords_readin to match the convention in cinter_module
  do i = 1, gprmep%nPoints
    gprmep%points(:,i) = gprmep%xcoords_readin(:,i)
  end do
  gprmep%react(:)=gprmep%points(:,1)
  gprmep%prod(:) =gprmep%points(:,gprmep%nPoints)
  deallocate(gprmep%xcoords_readin)
  call gprmep%curve%initialize(gprmep%nDim,gprmep%nPoints,gprmep%points)

  if (printl>=4) write(stdout,'(A," successfully read.")') filename
  
#ifdef TestOnMB
  lastMB = 0d0
  call init_MB()
  mbstepcounter = 0
  do while (.not.gprmep%mepConverged)
    call MB(lastMB,lastEMB,gMB,hMB)
    mbstepcounter = mbstepcounter + 1
    call gprmep%giveNextPt(lastMB,lastEMB,gMB,newMB)
    lastMB = newMB
    print*, "nr of energy evaluations on MB:", mbstepcounter
  end do
  print*, "nr of energy evaluations on MB:", mbstepcounter
  call dlf_fail("MB converged")
#endif
  
  return  
200 continue
  if (printl>=2) write(stdout,'("Error reading file")')
  call dlf_fail("Error reading qts_coords.txt file")
201 continue
  if (printl>=2) write(stdout,'("Error (EOF) reading file")')
  call dlf_fail("Error (EOF) reading qts_coords.txt file")
end subroutine gprmep_readQtsCoords

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_initWithMinima
!!
!! FUNCTION
!! Initialize a gprmep optimization by only giving the two minima
!!
!! SYNOPSIS
subroutine gprmep_initWithMinima(gprmep, nDim_in, react, prod, maxstep_in, &
    globtol_in, znuc_in, nPoints_in, initWithGeodesic, gprmep_mode)
!! SOURCE
  class(gprmep_type)        ::  gprmep
  integer, intent(in)       ::  nDim_in
  real(rk), intent(in)      ::  react(nDim_in), prod(nDim_in)
  real(rk), intent(in)      ::  maxstep_in !maxstep is not used at the moment
  real(rk), intent(in)      ::  globtol_in
  integer, intent(in)       ::  znuc_in(nDim_in/3)
  integer, intent(in)       ::  nPoints_in
  logical, intent(in)       ::  initWithGeodesic
  integer, intent(in)       ::  gprmep_mode
  real(rk), allocatable     ::  geodesic_pts(:,:)
  integer                   ::  i
  real(rk)                  ::  eps
  gprmep%gprmep_mode = gprmep_mode
!   integer, external         ::  mkl_get_max_threads,omp_get_max_threads
!   call mkl_set_num_threads(4)
#ifdef TIMING
  call gprmep%overallClock%start("GPRMEP plus GPRTS clock ",1d-3)
#endif
#ifdef plotNEBs
  call plotNEBwithCurve(9,nDim_in/3)
#endif
#ifdef TestOnMB
  call gprmep%readQtsCoords("qts_coords.txt", maxstep_in, &
                            globtol_in, gprmep_mode)
  return
#endif
  gprmep%maxstep = maxstep_in
  gprmep%nOpts = 0
  gprmep%nPoints = nPoints_in
  gprmep%nDim = nDim_in
  gprmep%globtol = globtol_in
#ifndef TestOnMB
  gprmep%nat = nDim_in/3
  if (MOD(gprmep%nDim,3)/=0) &
    call dlf_fail("Only 3*nat coords allowed! (gprmep_initWithMinima)")
#endif  
  call gprmep%initStandardParas()
  call gprmep%init_and_alloc()
  gprmep%znuc(:) = znuc_in(:)
  gprmep%react(:)=react(:)
  gprmep%prod(:) =prod(:)  
  if (dot_product(gprmep%prod(:)-gprmep%react(:),&
                  gprmep%prod(:)-gprmep%react(:))<1d-14) &
    call dlf_fail("Reactant and product seems to be the same structure.")
#ifndef TestOnMB  
  call dlf_cartesian_align(gprmep%nat, gprmep%react, gprmep%prod)
#endif
  if (dot_product(gprmep%prod(:)-gprmep%react(:),&
                  gprmep%prod(:)-gprmep%react(:))<1d-14) &
    call dlf_fail(&
        "After rotation reactant and product seems to be the same structure.")
  if (initWithGeodesic) then
    eps = 1d-5
#ifdef TIMING    
    call gprmep%clock%beat("Starting geodesic")
#endif
    call geo_inst%geodesicFromMinima(gprmep%nat,gprmep%znuc,&
              gprmep%react,gprmep%prod,gprmep%nPoints,.false.,eps)
#ifdef TIMING
    call gprmep%clock%beat("Geodesic done")
#endif
!     call dlf_fail("GEODESIC CHECK")
!     call geo_inst%geodesicFromMinima(gprmep%nat,gprmep%znuc,&
!               gprmep%react,gprmep%prod,gprmep%nPoints,.true.,eps)
    allocate(geodesic_pts(geo_inst%nDim,geo_inst%nPts))
    do i = 1, geo_inst%nPts
      geodesic_pts(:,i) = geo_inst%xCoords(:,i)
    end do
    
    call gprmep%curve%initialize(gprmep%nDim,geo_inst%nPts,geodesic_pts)
!     call gprmep%curve%addPtsToEqDist(1d-1)
!     call write_path_xyz(geo_inst%nAtoms,geo_inst%znuc,"eqDist1",&
!                         gprmep%curve%nControlPoints,&
!                         gprmep%curve%controlPoints,.false.)

!     call gprmep%curve%setEqDistPts(geo_inst%nPts,geodesic_pts, 1d-1)
!     call gprmep%curve%setEqDistPts(geo_inst%nPts,geodesic_pts, 1d-2)
    call gprmep%curve%setEqDistPts(gprmep%nPoints,gprmep%points, 1d-4)
    call write_path_xyz(gprmep%nat,gprmep%znuc,"eqDist2",&
                        gprmep%nPoints,&
                        gprmep%points,.false.)
    deallocate(geodesic_pts)
  else
    do i = 1, gprmep%nPoints
      gprmep%points(:,i) = gprmep%react(:) + &
          (gprmep%prod(:)-gprmep%react(:)) * real(i-1,kind=rk) / real(gprmep%nPoints-1,kind=rk)
    end do  
  end if
  call gprmep%curve%initialize(gprmep%nDim,gprmep%nPoints, gprmep%points)
!   call dlf_fail("Tested geodesic on MB")
end subroutine gprmep_initWithMinima

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_initFromPath
!!
!! FUNCTION
!! Initialize a gprmep optimization by giving a starting path
!!
!! SYNOPSIS
subroutine gprmep_initFromPath(gprmep, filename, &
                               maxstep_in, globtol_in,nPointsToBe,gprmep_mode)
!! SOURCE
  use dlf_constants, only: dlf_constants_get
  class(gprmep_type)        ::  gprmep
  character(*), intent(in)  ::  filename
  real(rk), intent(in)      ::  maxstep_in !maxstep is not used at the moment
  real(rk), intent(in)      ::  globtol_in
  integer, intent(in)       ::  nPointsToBe
  integer, intent(in)       ::  gprmep_mode
  integer                   ::  i
!   real(rk), allocatable     ::  coords(:,:,:), coords2(:,:,:)
  real(rk), allocatable     ::  pointsRev(:,:)
  if(allocated(gprmep%znuc)) deallocate(gprmep%znuc)
  if(allocated(gprmep%points)) deallocate(gprmep%points)
  gprmep%maxstep = maxstep_in
  gprmep%globtol = globtol_in
  gprmep%gprmep_mode = gprmep_mode
  SELECT CASE (gprmep_mode)
    CASE(0)
      if (printl>=4) then
        write(stdout,'("Optimizing only a path. No subsequent TS optimization is done.")')
        write(stdout,'("To perform also TS optimization choose tsopt=true in chemshell")')
        write(stdout,'("Or select glob%gprmep_mode=1 for dl-find standalone.")')
      end if
    CASE(1)
      if (printl>=4) &
        write(stdout,'("Optimizing a path. Then performing a TS optimization.")')
    CASE(2)
      if (printl>=4) &
        write(stdout,'("Optimizing a path. Then performing a TS optimization. Then reoptimize the path")')
    CASE DEFAULT
      call dlf_fail("This value for gprmep_mode is not known.")
  END SELECT
  gprmep%nOpts = 0
  call get_path_properties(filename,gprmep%nat, gprmep%nPoints)
  gprmep%nDim = gprmep%nat*3
  call gprmep%initStandardParas()
  call gprmep%init_and_alloc()
!   call dlf_fail("INIT DONE")
  allocate(pointsRev(gprmep%nDim,gprmep%nPoints))
  call read_path_xyz(filename,gprmep%nat,gprmep%nPoints,gprmep%znuc,pointsRev)
  call write_path_xyz(gprmep%nat,gprmep%znuc,'initialpath_copy',&
                      gprmep%nPoints,pointsRev, .false.)
  do i = 1, gprmep%nPoints
    gprmep%points(:,i) = pointsRev(:,i)
  end do
  deallocate(pointsRev)
  gprmep%react(:)=gprmep%points(:,1)
  gprmep%prod(:) =gprmep%points(:,gprmep%nPoints)
! #ifndef TestOnMB  
!   call dlf_cartesian_align(gprmep%nat, gprmep%react, gprmep%prod)
! #endif
  ! nothing done until here
  call gprmep%curve%initialize(gprmep%nDim,gprmep%nPoints,gprmep%points)
  call gprmep%change_nInterPts(nPointsToBe)
end subroutine gprmep_initFromPath

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_destroy
!!
!! FUNCTION
!! Destroy (deallocate etc.) everything needed for gprmep
!!
!! SYNOPSIS
subroutine gprmep_destroy(gprmep)
!! SOURCE
  class(gprmep_type)        ::  gprmep
  integer                   ::  iimage
  if (allocated(gprmep%ene)) deallocate(gprmep%ene)
  if (allocated(gprmep%xcoords_readin)) deallocate(gprmep%xcoords_readin)
  if (allocated(gprmep%dtau)) deallocate(gprmep%dtau)
  if (allocated(gprmep%dist)) deallocate(gprmep%dist)
  if (allocated(gprmep%points)) deallocate(gprmep%points)
  if (allocated(gprmep%oldPoints)) deallocate(gprmep%oldPoints)
  if (allocated(gprmep%d_times)) deallocate(gprmep%d_times)
  if (allocated(gprmep%vecsAlongCurve)) deallocate(gprmep%vecsAlongCurve)
  if (allocated(gprmep%vecsAlongCurve_n)) deallocate(gprmep%vecsAlongCurve_n)
  if (allocated(gprmep%coords_discrete)) deallocate(gprmep%coords_discrete)
  if (allocated(gprmep%esAtDiscretePts)) &
                                    deallocate(gprmep%esAtDiscretePts)
  if (allocated(gprmep%gradsAtDiscretePts)) &
                deallocate(gprmep%gradsAtDiscretePts)
  if (allocated(gprmep%esAtDiscretePts)) deallocate(gprmep%esAtDiscretePts)
  if (allocated(gprmep%hessiansAtDiscretePts)) &
                deallocate(gprmep%hessiansAtDiscretePts)
  if (allocated(gprmep%react)) deallocate(gprmep%react)
  if (allocated(gprmep%prod)) deallocate(gprmep%prod)
  if (gprmep%unitsOpen) then
    do iimage=1,gprmep%nPoints
      close(unit=gprmep%unitp+iimage)
    end do
  end if
  ! stuff for gprmep_loss_parsAndGrad
  if(allocated(gprmep%projMat)) &
    deallocate(gprmep%projMat)
  if(allocated(gprmep%tmpMat)) &
    deallocate(gprmep%tmpMat)
  if(allocated(gprmep%pForceNorm2_discretePts)) &
    deallocate(gprmep%pForceNorm2_discretePts)
  if(allocated(gprmep%tNorms_at_dPts))&
    deallocate(gprmep%tNorms_at_dPts)
  call GPR_destroy(gprmep%gpPES)
  gprmep%parasInit = .false.
end subroutine gprmep_destroy

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_giveNextPt
!!
!! FUNCTION
!! Core function of gprmep optimization that puts out the point that
!! should be calculated (calculation managed by the calling dl-find functions)
!!
!! SYNOPSIS
subroutine gprmep_giveNextPt(gprmep, lastPt_in, lastE_in, lastG_in, newPt)
!! SOURCE
  use oop_lbfgs
  use dlf_time
  class(gprmep_type)        ::  gprmep
  real(rk), intent(in)      ::  lastPt_in(gprmep%nDim)
  real(rk), intent(in)      ::  lastE_in
  real(rk)                  ::  lastE(1)
  real(rk), intent(in)      ::  lastG_in((gprmep%nDim))
  real(rk), intent(out)     ::  newPt(gprmep%nDim)
  integer                   ::  i, j
  real(rk),allocatable      ::  vecs_alongCurve(:, :), dotProds(:)
  integer                   ::  stepCounter
  ! convergence of overall path
  type(oop_lbfgs_type)      ::  lbfgs
  integer                   ::  opt_d, arr_size
  real(rk), allocatable     ::  opt_pars(:)  ! parameters to be 
                                             ! optimized
  real(rk), allocatable     ::  opt_grad(:)  ! gradient in
                                             ! parameter space
  real(rk), allocatable     ::  opt_step(:)  ! step in
                                             ! parameter space  
  real(rk), allocatable     ::  eb_opt_grad(:)  ! energy based gradient in
                                                ! parameter space
  real(rk), allocatable     ::  gb_opt_grad(:)  ! gradient based gradient in
                                                ! parameter space    
  real(rk)                  ::  max_eb_opt_grad
  real(rk), allocatable     ::  orig_pars(:)
  logical                   ::  converged
  logical                   ::  valid_nextPt
  
  real(rk)                  ::  rmsstep, maxstep, rmsgrad, maxgrad
  real(rk)                  ::  cChange ! change of the curve
                                        ! compared to last opt step
  real(rk)                  ::  maxcChange ! maximum change
  logical                   ::  var_cConv
  integer                   ::  opt_nPoints
  real(rk), allocatable     ::  opt_points(:,:)
#ifdef setPlotOn
#ifdef TestOnMB
  real(rk), allocatable     ::  plotPoints(:,:), vecPoints(:,:)
#endif
#endif
  integer                   ::  start_opt_d, end_opt_d
  integer                   ::  maxgradpointpos
  logical                   ::  TSconverged
  real(rk)                  ::  E1(1)
  logical                   ::  inverted
  integer                   ::  nInverted
#ifdef TIMING
  call gprmep%clock%restart("gprmep")
  call gprmep%clock%setCutOff(1d-1)
#endif
  if (gprmep%mepConverged) return
  if (printl>=4) then
        write(stdout,'(" ")')
        write(stdout,'(" ")')
  end if
  opt_nPoints = gprmep%nPoints
  ! check for unrealistically tight convergence criteria
  if (gprmep%curve%s_n(1)*5d0>gprmep%maxcChangeConv.or.&
    gprmep%curve%s_n(1)*5d0>gprmep%cChangeConv) then
    if (printl>=6) &
        write(stdout,'("s_n(1), maxcChangeConv, cChangeConv", 3ES11.4)') &
            gprmep%curve%s_n(1), &
            gprmep%maxcChangeConv, gprmep%cChangeConv
    call dlf_fail("Convergence criteria are only 5 times lower than max precision.")
  end if
  ! Catch the case of a TS search
  if (gprmep%opt_state==1) then
    ! first: check for convergence:
    if ((.not.gprmep%tsSearchJustStarted).and.&
              norm2(gprmep%TS-lastPt_in)>1d-15) &
        call dlf_fail("Strange... should be the same...")
    gprmep%tsSearchJustStarted = .false.
    gprmep%oldTSGuess = lastPt_in
    call gprmep%testTSconv(lastPt_in,lastE_in,lastG_in,TSconverged)
    if (.not.TSconverged) then
      if (printl>=4) &
        write(stdout,'("Transition state search did not converge until now.")')
      ! If not converged -> 
      E1(1) = lastE_in
      call GPR_Optimizer_step(gprmep%gpPES,gprmep%opt,lastPt_in,newPt,&
                              lastPt_in,E1(1),lastG_in)
      gprmep%TS = newPt
      return
    else
      if (printl>=4) write(stdout,'("Transition state search did converge.")')
      ! Embed that point into the path at the correct position 
      ! For example between the pts lOfTS and rOfTS
      ! If the point is too close to either of these, we do nothing
!       call gprmep%ts_search_wrapUp()
      call gprmep%signalConverged()
      if (gprmep%mepConverged) return
!       ! continue with usual GPRMEP search, if wanted...
!       if (gprmep%gprmep_mode==1) then
!         ! abort here!
!       end if
    end if
  end if
  
  if (gprmep%opt_state == 3) then
    call gprmep%calcFinalPathEnergies(newPt, lastE_in)
    return
  end if
  
  allocate(opt_points(gprmep%nDim,opt_nPoints))
#ifdef projectOut  
  allocate(vecs_alongCurve(gprmep%nDim,opt_nPoints))
  allocate(dotProds(opt_nPoints))
#endif
  ! minNrPts+maxNrPts are determined when initializing the path
  gprmep%increaseNrPts = 0

  if (gprmep%status==0) then
#ifdef setPlotOn
    do j = 1, gprmep%curve%nDim
      do i = 1, gprmep%nPoints      
        times(i) = REAL(i-1,kind=rk)/REAL(gprmep%nPoints      -1,kind=rk)
        call GPR_eval(gprmep%curve%gprs(j), times(i),val)
      end do
    end do
    gprmep%plotnr=0
    call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif
    call gprmep%curve%setEqDistPts(gprmep%curve%nControlPoints,&
                                   gprmep%points,1d-4)
#ifdef setPlotOn
    gprmep%plotnr=1
    call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif
    if (printl>=6) &
        write(stdout,'("Adding points to the list of points to be calculated")')
    call gprmep%add_PtsToCalc(gprmep%nPoints-2,gprmep%points(:,2:gprmep%nPoints-1))
    ! do end points last (they might not be necessary since they
    ! are not changed anyway)
    call gprmep%add_PtsToCalc(1,gprmep%points(:,1))
    call gprmep%add_PtsToCalc(1,gprmep%points(:,gprmep%nPoints))
    if (printl>=4) &
        write(stdout,'("Initialization of gprmep complete!")')
    gprmep%status = 1
  else
    ! a point was already calculated and lastPt_in, lastE, lastG_in 
    ! have some relevent information in it
    lastE(1)=lastE_in
!     if (gprmep%ptCalc<6) then
    call GPR_add_tp(gprmep%gpPES, 1, lastPt_in, lastE(1), lastG_in)
    if (gprmep%gpPES%nt>gprmep%maxNrPtsInLevel) &
        call GPR_newLevel(gprmep%gpPES,gprmep%shiftToLowerLevel)
    call GPR_interpolation(gprmep%gpPES)   
    call gprmep%checkPtsToCalcForVarChanges()
  end if
  call gprmep%checkout_PtToCalc(newPt(:), valid_nextPt)
  ! only if no point is left in ptToCalcList optimize the path
  if (.not.valid_nextPt) then
    if (printl>=4) &
        write(stdout,'("Curve-Optimization on GPR-PES...")')

    ! we have nTP energy weights and nDim*nTP gradient weights
    call gprmep%old_curve%initialize(gprmep%curve%nDim,&
        gprmep%curve%nControlPoints, gprmep%curve%controlPoints)
    ! increase number of points to improve the optimization on the 
    ! GPR-PES surface (also opt_points must be initialized)
#ifdef TIMING    
    call gprmep%clock%beat("Curve init done")
#endif
!     call gprmep%curve%setEqDistPts(opt_nPoints,opt_points, 1d-3)
    call gprmep%curve%setEqDistPts(opt_nPoints,opt_points, 1d-4)
#ifdef TIMING    
    call gprmep%clock%beat("Set EQDist done")
#endif
    ! Optimize the path
    gprmep%nOpts = gprmep%nOpts + 1
    ! ********************************************************************
    ! The actual optimization takes place here
    ! ********************************************************************
    ! nr of points on the path to calculate the loss function  
!     opt_d = gprmep%nDim*(gprmep%curve%nControlPoints)
    opt_d = gprmep%nDim*(gprmep%curve%nControlPoints-2)
    start_opt_d = gprmep%nDim+1
    end_opt_d = gprmep%nDim*(gprmep%curve%nControlPoints-1)
    arr_size = gprmep%nDim*gprmep%curve%nControlPoints
      allocate(opt_pars(arr_size))
      allocate(opt_grad(arr_size))
      allocate(eb_opt_grad(arr_size))
      allocate(gb_opt_grad(arr_size))
      allocate(opt_step(arr_size))
      allocate(orig_pars(arr_size))
      eb_opt_grad(:) = 0d0
      opt_step(:) = 0d0
      opt_pars(:) = 0d0
      opt_grad(:) = 0d0
      ! The following does not really make sense since the points are not
      ! all really in the tolerance range, just the ones from the former
      ! ToCalcList are... (in earlier versions this was included).
      ! In the paper version this is already commented out.
!       ! check if the path is already converged (with respect to the 
!       ! loss function)
!       call gprmep%checkConvergence(arr_size,start_opt_d,end_opt_d,opt_d, &
!              opt_pars,opt_grad,&
!              completeMaxgrad,completeRMSGrad, converged)
      ! writing the current positions to file
      call gprmep%saveImage()
!       if (converged.and.gprmep%nOpts>1) then
!         call gprmep%signalConverged()
!         if (gprmep%mepConverged) return
!         ! just switched to gprts?
!         if (gprmep%opt_state==1) then 
!           newPt = gprmep%TS
!           call gprmep%randomizeUntilVarLow(newPt)
!           ! distort the newPt if it lies very close to a training point
!           ! -> avoid crashing of GPR-surface building          
!           return
!         end if        
!       end if
      call lbfgs%init(opt_d, MAX(opt_d/2,50), .false.)
      converged = .false.
      stepCounter = 0
      nInverted = 0
      do while (.not.converged)
        if(stepCounter>1000) then
          if (printl>=6) &
            write(stdout,'("Optimization on GPR-PES aborted after 1000 steps.")')
          exit
        end if

        stepCounter = stepCounter + 1
        call gprmep%loss_parsAndGrad(&
                        arr_size,opt_pars,opt_grad,.true.)

        ! Squash the gradients (seems not to make sense)
!         call squashGrad(gprmep%curve%nControlPoints,gprmep%nDim, arr_size, opt_grad)

#ifdef setPlotOn
#ifdef TestOnMB
        allocate(plotPoints(gprmep%nDim,gprmep%curve%nControlPoints))
        allocate(vecPoints(gprmep%nDim,gprmep%curve%nControlPoints))
        do i = 1, gprmep%curve%nControlPoints
          ! walking over all GPRs (in every dimension of the PES there is one)
          do j = 1, gprmep%nDim
            plotPoints(j,i) = &
                opt_pars(gprmep%nDim*(i-1)+j)
            vecPoints(j,i) = &
                -opt_grad(gprmep%nDim*(i-1)+j)
          end do
        end do
        gprmep%plotnr = gprmep%plotnr + 1
        call gprmep%curve%writePlotData(100,gprmep%plotnr)
        call gprmep%curve%writeVectorPlots(gprmep%curve%nControlPoints, &
                gprmep%plotnr, vecPoints, 1d-1)
!         call gprmep%curve%writeVectorPlots(gprmep%curve%nControlPoints, &
!                 gprmep%plotnr+1, plotPoints, vecPoints, 1d-1)
        deallocate(plotPoints)
        deallocate(vecPoints)
#endif
#ifdef TIMING
        call gprmep%clock%beat("Plottingstuff done")
#endif
#endif
        max_eb_opt_grad = MAXVAL(eb_opt_grad(start_opt_d:end_opt_d))
        if (stepCounter==1) orig_pars(:) = opt_pars(:)
!         opt_step(:) = 9d99
        call lbfgs%next_step(opt_pars(start_opt_d:end_opt_d),&
               opt_grad(start_opt_d:end_opt_d), &
               opt_step(start_opt_d:end_opt_d),inverted)
        if(inverted) nInverted = nInverted + 1
#ifdef projectOut
        call gprmep%curve%eval_grad_at_cPts(vecs_alongCurve)
       ! normalize the tangents
        do i = 1, gprmep%curve%nControlPoints
          vecs_alongCurve(:,i) = vecs_alongCurve(:,i)/&
                               norm2(vecs_alongCurve(:,i))
        end do
        ! ***********************************
        ! project out of steps
        dotProds(:) = 0d0
        do i = 1, gprmep%nDim
          do j = 1, gprmep%curve%nControlPoints
            dotProds(j) = dotProds(j) + vecs_alongCurve(i,j)*&
                          opt_step((j-1)*gprmep%nDim+i)                   
            if (printl>=6) &
                write(stdout,'("Step components along the path (projected out):", ES11.4)')&
                    dotProds(j)
          end do
        end do
        do i = 1, gprmep%nDim
          do j = 1, gprmep%curve%nControlPoints
            opt_step((gprmep%nDim)*(j-1)+i) = &
            opt_step((gprmep%nDim)*(j-1)+i) - &
              dotProds(j)*vecs_alongCurve(i,j)
          end do
        end do
        ! ***********************************
        ! project out of gradients (for convergence check)
        dotProds(:) = 0d0
        do i = 1, gprmep%nDim
          do j = 1, gprmep%curve%nControlPoints
            dotProds(j) = dotProds(j) + vecs_alongCurve(i,j)*&
                          opt_grad((j-1)*gprmep%nDim+i)                           
            if (printl>=6) &
                write(stdout,'("Step components along the path (projected out):",&
                    ES11.4)') dotProds(j)
          end do
        end do
        do i = 1, gprmep%nDim
          do j = 1, gprmep%curve%nControlPoints
            opt_grad((gprmep%nDim)*(j-1)+i) = &
            opt_grad((gprmep%nDim)*(j-1)+i) - &
              dotProds(j)*vecs_alongCurve(i,j)
          end do
        end do
#endif          
        if (norm2(opt_step(start_opt_d:end_opt_d))>gprmep%maxStepSize) then
          opt_step(start_opt_d:end_opt_d) = &
            opt_step(start_opt_d:end_opt_d)/&
            (norm2(opt_step(start_opt_d:end_opt_d)))*gprmep%maxStepSize
        end if
        rmsstep = norm2(opt_step(start_opt_d:end_opt_d))/&
                    dsqrt(REAL(opt_d,kind=rk))
        maxstep = maxval(abs(opt_step(start_opt_d:end_opt_d)))
        rmsgrad = norm2(opt_grad(start_opt_d:end_opt_d))/&
                    dsqrt(REAL(opt_d,kind=rk))
        maxgrad = maxval(abs(opt_grad(start_opt_d:end_opt_d)))
        maxgradpointpos = &
          (SUM(MAXLOC(abs(opt_grad(start_opt_d:end_opt_d))))-1)/3 + 1
        opt_pars(start_opt_d:end_opt_d) = opt_pars(start_opt_d:end_opt_d) + &
                                   opt_step(start_opt_d:end_opt_d)
        do i = 1, gprmep%nDim
          ! walking over all GPRs (in every dimension of the PES there is one)
          ! The endpoints are not changed
          do j = 2, gprmep%curve%nControlPoints-1
            opt_points(i,j) = &
            opt_pars((gprmep%nDim)*(j-1)+i)
          end do     
        end do     
        call gprmep%curve%initialize(gprmep%nDim,opt_nPoints,opt_points)
#ifdef checkAndCorrectDistances
        ! prevent points from getting to close together
        call gprmep%curve%avoidSmallDistance(&
            norm2(gprmep%prod(:)-gprmep%react(:))/&
            gprmep%curve%nControlPoints/&
            2d0, opt_points)

#endif
        converged=(gprmep%nOpts>1.and.&
            rmsstep<gprmep%tolrmss .and. &
            maxstep<gprmep%tols .and. &
            rmsgrad<gprmep%tolrmsg .and. &
            maxgrad<gprmep%tolg.and.&
            stepCounter>1) ! the first lbfgs step is always too small
! 78456 FORMAT (A,ES10.3,3X,ES10.3,3X,ES10.3,3X,ES10.3,3X,I4,3X,I4)
78457 FORMAT (A,ES10.3,2X,ES10.3,2X,ES10.3,2X,ES10.3,2X,ES10.3,2X,&
              ES10.3,2X,ES10.3,2X,ES10.3,2X,I4,2X,I4)
        if(MOD(stepCounter,500)==0.and.stepCounter>500) then
          if (printl>=6) then
            write(stdout,'(" ")')
            write(stdout,78457) "Opt on GPR-PES: convcrits", rmsstep, gprmep%tolrmss, &
              maxstep, gprmep%tols, rmsgrad, gprmep%tolrmsg, maxgrad, &
              gprmep%tolg, gprmep%gpPES%nt, stepCounter
          end if
        end if

        if (converged.and.printl>=4) then
          write(stdout,'("Curve converged on the GPR-PES.")')
        end if

        if (printl>=6) &
            write(stdout,'("Opt on GPR-PES: RMS of pathchange is ", ES11.4)') &
                norm2(orig_pars(start_opt_d:end_opt_d)-&
                opt_pars(start_opt_d:end_opt_d))

        if (norm2(orig_pars(start_opt_d:end_opt_d)-opt_pars(start_opt_d:end_opt_d))>gprmep%maxPathUpdate) then
          converged=.true.
     
          if (printl>=6) then
            write(stdout,'("Opt on GPR-PES: maximally allowed change to the path reached")')
            write(stdout,'("Opt on GPR-PES: stopping path update.")')
          end if
        end if
        if (.not.converged.and.printl>=6) &
            write(stdout,'("Opt on GPR-PES: Not converged. ",&
                           "Continue optimization on GPR-PES.")')
        if (isnan(rmsstep)) call dlf_fail("rmsstep is NAN")
#ifdef setPlotOn     
        gprmep%plotnr = gprmep%plotnr+1
        call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif
        if(nInverted>10) then
          if(printl>=4) &
            write(stdout,'("L-BFGS inverted too often. Aborting Optimization on GPR-PES.")')
          exit
        end if
      end do ! while (.not.converged)
#ifdef TIMING
      call gprmep%clock%beat("OptOnGPRconverged")
#endif
      if(printl>=4) write(stdout,'(" ")')
      
      deallocate(opt_step)
      deallocate(opt_grad)
      deallocate(eb_opt_grad)
      deallocate(gb_opt_grad)
      deallocate(opt_pars)
      deallocate(orig_pars)    
      call lbfgs%destroy()
#ifdef setPlotOn   
      gprmep%plotnr = gprmep%plotnr+1
      call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif
#ifdef setPlotOn   
      gprmep%plotnr = gprmep%plotnr+1
      call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif

      ! decrease number of points to calc less points on the real PES
      ! NOTE: This has to be here anyway 
      ! (also without changing nr of ctrlPts)
!       call gprmep%curve%addPtsToEqDist(5d-2)
!       call gprmep%curve%setEqDistPts(gprmep%nPoints,&
!         gprmep%points, 1d-2)
!       call gprmep%curve%setEqDistPts(gprmep%nPoints,&
!         gprmep%points, 1d-3)
      call gprmep%curve%setEqDistPts(gprmep%nPoints,&
        gprmep%points, 1d-4)
#ifdef TIMING
      call gprmep%clock%beat("setEqDistPts done")
#endif
      ! recalc the next points
      gprmep%ptCalc=1
      gprmep%status=1
      var_cConv = .true.
      call gprmep%fillListAccordingToVariance(var_cConv)      
#ifdef setPlotOn
      gprmep%plotnr = gprmep%plotnr+1
      if (gprmep%ptToCalcList%np>0.and.allocated(gprmep%ptToCalcList%coords)) then
        call gprmep%curve%writePlotData_extra(100,gprmep%plotnr, &
                    gprmep%ptToCalcList%np,gprmep%ptToCalcList%coords)
      else
        call gprmep%curve%writePlotData(100,gprmep%plotnr)
      end if
#endif
    ! **************************************************************
    ! calculate the difference between the last and the current curve
    if (gprmep%old_curve%nControlPoints==gprmep%curve%nControlPoints) then
      if(printl>=6) &
            write(stdout,'( "Calculating the difference to the old curve...")')
      call gprmep%curve%calcDiffToCurve(gprmep%old_curve, cChange, maxcChange)
      if(printl>=6) &
            write(stdout,'(A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,20x,A,2x,I4)') &
                "Curve Convergence:",cChange," of ", &
                gprmep%cChangeConv," and ",maxcChange," of ", &
                gprmep%maxcChangeConv," stepnr:", stepCounter
      if (((maxcChange < gprmep%maxcChangeConv).and.&
           (   cChange < gprmep%cChangeConv   ).and.var_cConv)&
              .or.gprmep%ptToCalcList%np==0 )then
        if (gprmep%ptToCalcList%np==0) then
          if(printl>=4) &
            write(stdout,'("No points with high variance left. ",&
                          "All have been calculated.")')
        else
          if(printl>=4) &
            write(stdout,'("Convergence criteria for curve change satisfied.")')
        end if
        call gprmep%signalConverged()
        if (gprmep%mepConverged) return
        ! just switched to gprts?
        if (gprmep%opt_state==1) then 
          newPt = gprmep%TS
          call gprmep%randomizeUntilVarLow(newPt)
          if(printl>=4) &
            write(stdout,'("Initialized the TS search.")')
          ! distort the newPt if it lies very close to a training point
          ! -> avoid crashing of GPR-surface building          
          return
        end if 
        ! just switched to calculating energies of final path
        if(gprmep%opt_state == 3) then
          call gprmep%calcFinalPathEnergies(newPt, lastE_in)
          return
        end if
      else
        ! get the next point to calculate the energies and gradients at
        if(printl>=4) then
            write(stdout,'("Convergence criteria not met yet. Need new energies...")')
            write(stdout,'(" ")')
        end if
        call gprmep%checkout_PtToCalc(newPt(:), valid_nextPt)
        if (.not.valid_nextPt) call dlf_fail("This should not happen! (valid_nextPt)")
      end if
    end if
    ! **************************************************************
  end if ! curve optimization
  deallocate(opt_points)
#ifdef projectOut  
  deallocate(vecs_alongCurve)
  deallocate(dotProds)
#endif
end subroutine gprmep_giveNextPt

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_loss_parsAndGrad
!!
!! FUNCTION
!! calculates different loss functions I defined for the gprmep optimization
!! (energy and gradient based). The subroutine also outputs
!! the parameters that need to be optimized (the control points of the
!! interpolation for example). This routine is called in every
!! optimization step of the gprmep-path on the GPR-PES surface.
!!
!! SYNOPSIS
subroutine gprmep_loss_parsAndGrad(gprmep,arr_size,opt_pars,opt_grad,&
                                   recalcProjection)
!! SOURCE                            
  class(gprmep_type)        ::  gprmep
  integer, intent(in)       ::  arr_size
  real(rk), intent(out)     ::  opt_pars(arr_size),&
                                opt_grad(arr_size)
!   real(rk),intent(out)      ::  eb_opt_grad(arr_size)
!   real(rk),intent(out)      ::  gb_opt_grad(arr_size)
  logical, intent(in)       ::  recalcProjection
  real(rk)                  ::  gb_tmp,&
                                opt_grad_tmp(arr_size)
  real(rk)                  ::  eb_tmp
  ! to understand all this stuff following you should have a look at my 
  ! theses/notes to understand the notations
  ! dx^l_i / w_k
  real(rk)                  ::  dxli_wk
  ! d^2x^j_i / (dw_k dt)
  real(rk)                  ::  d2xji_dwkdt
  real(rk)                  ::  diff(1)                            
  integer                   ::  i,j,k,l,m, alpha, thread_nr
  real(rk)                  ::  time(1)
!   integer, external         ::  mkl_get_max_threads
#ifdef TIMING
  real(rk)                  ::  time1,time2
#endif
  ! parameter values
    if (gprmep%curve%nDim/=gprmep%nDim) call dlf_fail("curve and gprmep have different nDim")
      do i = 1, gprmep%nDim
        ! walking over all GPs (in every dimension of the PES there is one)
        do j = 1, gprmep%curve%nControlPoints
          opt_pars((gprmep%nDim)*(j-1)+i) = gprmep%curve%controlPoints(i,j)
        end do
      end do
      ! determine the tangent vectors at the discretization points
      call gprmep%curve%eval_grad(gprmep%n_discrete, gprmep%d_times, gprmep%vecsAlongCurve)
      ! normalize the tangents
      do i = 1, gprmep%n_discrete
        gprmep%tNorms_at_dPts(i) = norm2(gprmep%vecsAlongCurve(:,i))
        gprmep%vecsAlongCurve_n(:,i) = &
            gprmep%vecsAlongCurve(:,i)/gprmep%tNorms_at_dPts(i)
      end do
      ! determine the coordinates of the discretization points
      call gprmep%curve%evaluate(gprmep%n_discrete, gprmep%d_times, gprmep%coords_discrete)
      ! determine the gradients on the GP-PES at the discretization points
      ! and project out the directions along the curve (tangents)
      do i = 1, gprmep%n_discrete
        call GPR_eval(gprmep%gpPES, gprmep%coords_discrete(:,i), &
                      gprmep%esAtDiscretePts(i))
        call GPR_eval_grad(gprmep%gpPES, gprmep%coords_discrete(:,i), &
                                    gprmep%gradsAtDiscretePts(:,i))
        
        ! projecting out tangents
        call projectOut_nVector(gprmep%nDim, &
                                gprmep%gradsAtDiscretePts(:,i),&
                                gprmep%vecsAlongCurve_n(:,i))
        ! gradsAtDiscretePts now contains the perpendicular forces
        gprmep%pForceNorm2_discretePts(i) = norm2(gprmep%gradsAtDiscretePts(:,i))**2
      end do
      ! calculate the GP-PES Hessians at the discretization points
      ! and project out tangents
#ifdef withopenmp
    call omp_set_num_threads( omp_get_max_threads() )
#ifdef TIMING
    time1 = omp_get_wtime()
#endif
#endif
    thread_nr = 1
      do i = 1, gprmep%n_discrete
#ifdef withopenmp
        thread_nr = OMP_GET_THREAD_NUM()+1
#endif
        call GPR_eval_hess(gprmep%gpPES, gprmep%coords_discrete(:,i),&
                           gprmep%hessiansAtDiscretePts(:,:,i))
        ! projecting out tangential components
        ! projection matrix is identity matrix minus 
        ! outer product of tangential vector with itself
        if(recalcProjection) then
          gprmep%projMat(:,:,thread_nr) = -spread(gprmep%vecsAlongCurve_n(:,i),2,gprmep%nDim)*&
                         spread(gprmep%vecsAlongCurve_n(:,i),1,gprmep%nDim)
          do j = 1, gprmep%nDim
            gprmep%projMat(j,j,thread_nr) = gprmep%projMat(j,j,thread_nr) + 1d0
          end do
        end if

        ! actual projection
!         CALL DGEMM('N','N',M,N,K,ALPHA,A,M,B,K,BETA,C,M)
!         print*, "MKL threads", MKL_GET_MAX_THREADS()
!         call DGEMM('N','N',gprmep%nDim,gprmep%nDim,gprmep%nDim,1d0,&
!                     gprmep%hessiansAtDiscretePts(:,:,i),gprmep%nDim,&
!                     gprmep%projMat(:,:,thread_nr),gprmep%nDim,0d0,&
!                     gprmep%tmpMat(:,:,thread_nr),gprmep%nDim)
        gprmep%tmpMat(:,:,thread_nr) = matmul(gprmep%hessiansAtDiscretePts(:,:,i),gprmep%projMat(:,:,thread_nr))
!         call DGEMM('N','N',gprmep%nDim,gprmep%nDim,gprmep%nDim,1d0,&
!                     gprmep%projMat(:,:,thread_nr),gprmep%nDim,&
!                     gprmep%tmpMat(:,:,thread_nr),gprmep%nDim,0d0,&
!                     gprmep%hessiansAtDiscretePts(:,:,i),gprmep%nDim)
        gprmep%hessiansAtDiscretePts(:,:,i) = matmul(&
            gprmep%projMat(:,:,thread_nr),gprmep%tmpMat(:,:,thread_nr))
        ! hessiansAtDiscretePts now contains the "perpendicular Hessians"
      end do
      opt_grad_tmp(:) = 0d0
      ! this will be the gradient of the loss function in the parameter space
      ! using a gradient based loss function
!       gb_opt_grad(:) = 0d0
      ! this will be the gradient of the loss function in the parameter space
      ! using an energy based loss function
!       eb_opt_grad(:) = 0d0

      ! run over all parameters w_k 
      ! (every loop calc 1 energy- and one grad-component)
      ! run over all gprs (one for every dimension l)
      do l = 1, gprmep%nDim
        ! run over all training points in each gpr
       if (glob%spec((l-1)/3+1)<0) then
        ! frozen atom
       else
        do k = 1, gprmep%curve%nControlPoints
          alpha = (gprmep%curve%nControlPoints)*(l-1)+k ! position in opt_grad_tmp
          ! run over all discretization points i            
          do i = 1, gprmep%n_discrete          
            ! fixed k&i-> calc dxli_wk
            time(1) = gprmep%d_times(i)
            dxli_wk = kernel(gprmep%curve%gprs(l), time, &
                             gprmep%curve%gprs(l)%xs(:,k)) !gprs(l) is a 1D GP
            ! in my notation this l should be a j
            diff(1) = time(1)-gprmep%curve%gprs(l)%xs(1,k)
            d2xji_dwkdt = kernel_d1_exp1(gprmep%curve%gprs(l), &
                             diff(1),1,abs(diff(1)))
            ! in the first step the hessian is not approximated in a 
            ! meaningful way -> only energy based loss function makes sense.
            ! Later we only choose gradient based
            if (gprmep%nOpts > 1) then
            gb_tmp =  &
                2d0*(dot_product(gprmep%hessiansAtDiscretePts(l,:,i),&
                                  gprmep%gradsAtDiscretePts(:,i)))*dxli_wk*&
                  gprmep%tNorms_at_dPts(i)+ &
                  gprmep%pForceNorm2_discretePts(i)*&
                         gprmep%vecsAlongCurve(l,i)*d2xji_dwkdt/&
                         gprmep%tNorms_at_dPts(i)     
            end if
            if (gprmep%nOpts <= gprmep%nOpts_eBased) then
              eb_tmp= &
                  2d0*gprmep%esAtDiscretePts(i)*& 
                  gprmep%gradsAtDiscretePts(l,i)*&
                  dxli_wk*gprmep%tNorms_at_dPts(i)+ &
                  gprmep%esAtDiscretePts(i)**2*&
                  gprmep%vecsAlongCurve(l,i)*d2xji_dwkdt/&
                  gprmep%tNorms_at_dPts(i)
            end if
            if (gprmep%nOpts == 1) then
              opt_grad_tmp(alpha) = opt_grad_tmp(alpha) + eb_tmp
            else if (gprmep%nOpts <= gprmep%nOpts_eBased) then
              ! weighting between e and g based
              opt_grad_tmp(alpha) = opt_grad_tmp(alpha) + gb_tmp * &
                                  gprmep%weight_e_g_Loss(time(1))
              opt_grad_tmp(alpha) = opt_grad_tmp(alpha) + eb_tmp * &
                                  (1d0 - gprmep%weight_e_g_Loss(time(1)))
            else
              opt_grad_tmp(alpha) = opt_grad_tmp(alpha) + gb_tmp
            end if
          end do 
        end do  
       end if
      end do    
      ! multiplication by \Delta t
!       gb_opt_grad(:) = gb_opt_grad(:)/REAL(gprmep%n_discrete,kind=rk)
!       eb_opt_grad(:) = eb_opt_grad(:)/REAL(gprmep%n_discrete,kind=rk)   
      opt_grad_tmp(:) = opt_grad_tmp(:)/REAL(gprmep%n_discrete,kind=rk)
      if (gprmep%n_discrete<gprmep%curve%nControlPoints-2) &
        call dlf_fail("n_discrete must be >=  gprmep%curve%nControlPoints-2!")
            
          
      
!       ! Now transforming in the space of the control points
!       if (.not.(gprmep%curve%GPRs(1)%K_stat==8)) &
!           call dlf_fail("KM is not the complete inverse!")
      ! assert the length of arr_size
      if (gprmep%curve%nControlPoints/=(arr_size)/gprmep%nDim) call dlf_fail("Sth wrong?")

      opt_grad(:) = 0d0
      do l = 1, gprmep%nDim
        if (glob%spec((l-1)/3+1)<0) then
          ! frozen atom
        else
          alpha = (gprmep%curve%nControlPoints)*(l-1)
          do m = 1, gprmep%curve%nControlPoints
            opt_grad((gprmep%nDim)*(m-1)+l) = &
                                dot_product(&
                                gprmep%curve%GPRs(l)%KM(m,:),&
                                opt_grad_tmp(alpha+1:alpha+gprmep%curve%nControlPoints))
          end do
        end if
      end do
!       ! energy based gradient
!       opt_grad_tmp(:)=eb_opt_grad(:)
! 
!       do l = 1, gprmep%nDim
!         alpha = (gprmep%curve%nControlPoints)*(l-1)
!         do m = 1, gprmep%curve%nControlPoints
! !           print*, "cIndicesE", (gprmep%curve%nControlPoints)*(l-1)+m
!           eb_opt_grad((gprmep%nDim)*(m-1)+l) = dot_product(&
!                               gprmep%curve%GPRs(l)%KM(m,:),&
!                               opt_grad_tmp(alpha+1:alpha+gprmep%curve%nControlPoints))!*&
! !                               weightGradient(m,&
! !                                     gprmep%curve%nControlPoints)
!         end do
!       end do
!       
!       ! gradient based gradient
!       opt_grad_tmp(:)=gb_opt_grad(:)
!       do l = 1, gprmep%nDim
!         do m = 1, gprmep%curve%nControlPoints
!           gb_opt_grad((gprmep%nDim)*(m-1)+l) = dot_product(&
!                               gprmep%curve%GPRs(l)%KM(m,:),&
!                               opt_grad_tmp(alpha+1:alpha+gprmep%curve%nControlPoints))!*&
! !                               weightGradient(m,&
! !                                     gprmep%curve%nControlPoints)
!         end do
!       end do

end subroutine gprmep_loss_parsAndGrad

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_weight_e_g_Loss
!!
!! FUNCTION
!! Weights the gradient and energy based loss function in a linear way. 
!! The right mixture can sometimes be beneficial
!! A value close to 0 means energy based (0 completely energy based), 
!! a value close to 1 means gradient based (1 completely gradient based).
!! SYNOPSIS
function gprmep_weight_e_g_Loss(gprmep,time)
!! SOURCE
  class(gprmep_type)    ::  gprmep
  real(rk)              ::  gprmep_weight_e_g_Loss
  real(rk), intent(in)  ::  time ! must be between 0 and 1
!   real(rk)              ::  h=0.6d0
!   real(rk)              ::  t=0.2d0
!   real(rk)              ::  pi = 4.D0*DATAN(1.D0)
!   gprmep_weight_e_g_Loss = h-(h-t)*SIN(time*pi) ! pretty OK
!   gprmep_weight_e_g_Loss = (h+t)/2d0+(h-t)/2d0*COS(time*2d0*pi)
!   gprmep_weight_e_g_Loss = t+(h-t)*(2d0*(time-0.5d0))**2
  ! Endpoints are minimized in energy
!   if (time<1d-2) gprmep_weight_e_g_Loss = 0d0
!   if (time>1d0-1d-2) gprmep_weight_e_g_Loss = 0d0
!   print*, "egweights", time, gprmep_weight_e_g_Loss
   
!   gprmep_weight_e_g_Loss=MIN((REAL((gprmep%nOpts-1),kind=rk)/10d0)**2,1d0)
  gprmep_weight_e_g_Loss=MIN((REAL((gprmep%nOpts-1),kind=rk)/&
                             REAL(gprmep%nOpts_eBased,kind=rk)),1d0)
!   if (gprmep%opt_state) gprmep_weight_e_g_Loss=1d0
!   if (gprmep%nOpts>8) then 
!     gprmep_weight_e_g_Loss = 0d0
!     print*, "SwitchToE"
!   end if
!     gprmep_weight_e_g_Loss=1d0
!     gprmep_weight_e_g_Loss=0d0
!   if (gprmep%manual_e_g_loss>=-1d-16) &
!     gprmep_weight_e_g_Loss = gprmep%manual_e_g_loss
end function gprmep_weight_e_g_Loss

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/weightGradient
!!
!! FUNCTION
!! Was used to put some weight on the gradient, dependent where it is on the 
!! path.
!! This function is not used anymore...
function weightGradient(nr,maxNr)
!! SOURCE
  real(rk)              ::  weightGradient
  integer, intent(in)   ::  nr, maxNr
  real(rk)              ::  s2pi = dsqrt(2d0*(4.D0*DATAN(1.D0)))
  real(rk)              ::  sigma
  real(rk)              ::  mu
  ! gaussian
  sigma = 1d0
  mu = 0.5d0
  weightGradient = dexp(-((Real(nr-1,kind=rk)/Real(maxNr-1,kind=rk)-mu)/sigma)**2/2d0)/(dsqrt(sigma)*s2pi)/0.382925
  weightGradient = 1d0
!   weightGradient = 0d0
!   print*, "weightGradient", nr, maxNr, weightGradient
end function weightGradient

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/squashGrad
!!
!! FUNCTION
!! Can be used to squash the gradients a bit, if some are extremely large.
!! This subroutine is not used anymore...
subroutine squashGrad(nPts, nDim, arr_size, vecs)
!! SOURCE
  integer, intent(in)       ::  nPts, nDim, arr_size
  real(rk), intent(inout)   ::  vecs(arr_size)
  real(rk)                  ::  avgLength, maxLength
  real(rk)                  ::  lengths(nPts), oldLengths(nPts)
  real(rk)                  ::  cap
  integer                   ::  i, j
  lengths = 0d0
  do j = 1, nDim
    do i = 1, nPts
      ! gradient length for gradient at point i
      lengths(i) = lengths(i) + (vecs((j-1)*nPts+i))**2
    end do
  end do  
  lengths(:) = dsqrt(lengths(:))
  avgLength = sum(lengths(:))/nPts
  maxLength = MAXVAL(lengths(:))
  cap = 0.7*maxLength
  oldLengths(:) = lengths(:)
  ! Squash the lengths of the gradients with softmax
  ! then adapt the lengths of the gradients
  call softMax(nPts, lengths)
  do j = 1, nDim
    do i = 1, nPts
!       vecs((j-1)*nPts+i) = vecs((j-1)*nPts+i)/oldLengths(i)*lengths(i)*avgLength/2d0
!       if (oldLengths(i)>cap) then
!         vecs((j-1)*nPts+i) = vecs((j-1)*nPts+i)/oldLengths(i)*cap
!       end if
!       vecs((j-1)*nPts+i) = vecs((j-1)*nPts+i) *0.5d0
    end do
  end do  
end subroutine squashGrad

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/softMax
!!
!! FUNCTION
!! Implementing softMax. Not used anymore.
subroutine softMax(nDim, vec)
!! SOURCE
  integer, intent(in)       ::  nDim
  real(rk), intent(inout)   ::  vec(nDim)
  real(rk)                  ::  denominator
  denominator = SUM(DEXP(vec(:)))
  vec(:) = dexp(vec(:))/denominator
end subroutine softMax

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_add_PtsToCalc
!!
!! FUNCTION
!! add some points to the ToCalcList (these are then calculated via dl-find)
subroutine gprmep_add_PtsToCalc(gprmep, nPoints, newCoords)
!! SOURCE
  class(gprmep_type)    ::  gprmep
  integer, intent(in)   ::  nPoints
  real(rk), intent(in)  ::  newCoords(gprmep%nDim,nPoints)
  integer               ::  oldnPoints
  real(rk), allocatable ::  oldCoords(:,:)
  oldnPoints = gprmep%ptToCalcList%np
  gprmep%ptToCalcList%np = gprmep%ptToCalcList%np + nPoints
  if (allocated(gprmep%ptToCalcList%coords)) then
    allocate(oldCoords(gprmep%nDim,oldnPoints))
    oldCoords = gprmep%ptToCalcList%coords
    deallocate(gprmep%ptToCalcList%coords)
    allocate(gprmep%ptToCalcList%coords(gprmep%nDim,gprmep%ptToCalcList%np))
    gprmep%ptToCalcList%coords(:,1:oldnPoints) = oldCoords(:,1:oldnPoints)
    deallocate(oldCoords)
  else 
    allocate(gprmep%ptToCalcList%coords(gprmep%nDim,gprmep%ptToCalcList%np))
  end if
  gprmep%ptToCalcList%coords(:,oldnPoints+1:gprmep%ptToCalcList%np) = &
                                              newCoords(:,1:nPoints) 
end subroutine gprmep_add_PtsToCalc

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_add_PtWithVarianceCheck
!!
!! FUNCTION
!! add some points to the ToCalcList (these are then calculated via dl-find)
!! but first make a check whether the variance of this point is high enough
subroutine gprmep_add_PtWithVarianceCheck(gprmep, newCoords, added, variance)
!! SOURCE
  class(gprmep_type)                ::  gprmep
  real(rk), intent(in)              ::  newCoords(gprmep%nDim)
  logical, intent(out)              ::  added
  real(rk), intent(out)             ::  variance
  call GPR_variance(gprmep%gpPES,newCoords,variance)
  if(printl>=6) write(stdout,'("variance at suggested point", ES11.4)') variance
  if (variance>=gprmep%variance_limit) then
    ! add this point
    call gprmep%add_PtsToCalc(1,newCoords)
    added = .true.
  else
    ! do not add this point
    added = .false.
  end if
end subroutine gprmep_add_PtWithVarianceCheck

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_checkout_PtToCalc
!!
!! FUNCTION
!! Checkout the new point of the ToCalcList
!! If no valid point is given -> must perform optimization
subroutine gprmep_checkout_PtToCalc(gprmep, coords, valid)
!! SOURCE
  class(gprmep_type)    ::  gprmep
  real(rk), intent(out) ::  coords(gprmep%nDim)
  logical, intent(out)  ::  valid
  gprmep%ptToCalcList%ptIter = gprmep%ptToCalcList%ptIter + 1
  if(printl>=6) write(stdout,'("trying to check out point", 2ES11.4)') &
    gprmep%ptToCalcList%ptIter, gprmep%ptToCalcList%np
  if (gprmep%ptToCalcList%ptIter>gprmep%ptToCalcList%np) then
    ! no point remaining
    valid = .false.
    ! remove the old points
    if (allocated(gprmep%ptToCalcList%coords)) then
      deallocate(gprmep%ptToCalcList%coords)
    end if
    gprmep%ptToCalcList%ptIter = 0
    gprmep%ptToCalcList%np = 0
  else
    ! give out next point
    coords(:) = gprmep%ptToCalcList%coords(:,gprmep%ptToCalcList%ptIter)
    valid = .true.
  end if
end subroutine gprmep_checkout_PtToCalc

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_rotateBack
!!
!! FUNCTION
!! Rotate back some points to match reactants rotation state
subroutine gprmep_rotateBack(gprmep)
!! SOURCE
  class(gprmep_type)        ::  gprmep
  integer                   ::  i
  do i = 1, gprmep%curve%nControlPoints
    call dlf_cartesian_align(gprmep%nat, gprmep%react, gprmep%points(:,i))
  end do
  ! re-initialize with new points
  call gprmep%curve%initialize(gprmep%nDim, gprmep%curve%nControlPoints, gprmep%points)
end subroutine gprmep_rotateBack

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_checkConvergence
!!
!! FUNCTION
!! Check if gprmep has converged wrt. the loss function
subroutine gprmep_checkConvergence(gprmep,arr_size,start_opt_d,&
                        end_opt_d, opt_d,&
                        opt_pars,opt_grad,&
                        maxgrad,rmsgrad,&
                        converged)
!! SOURCE                
  class(gprmep_type)        ::  gprmep
  integer, intent(in)       ::  arr_size, start_opt_d, end_opt_d, opt_d
  real(rk), intent(out)     ::  opt_pars(arr_size),&
                                opt_grad(arr_size)
  real(rk), intent(out)     ::  maxgrad, rmsgrad
  real(rk)                  ::  conv_maxgrad, conv_rmsgrad
  logical, intent(out)      ::  converged
  call gprmep%loss_parsAndGrad(&
                        arr_size,opt_pars,opt_grad,.true.)
  maxgrad = MAXVAL(opt_grad(start_opt_d:opt_d))
  rmsgrad = dsqrt(SUM((opt_grad(start_opt_d:opt_d))**2)/(opt_d))
  conv_maxgrad = 4d-3
  conv_rmsgrad = 1d-3
  converged = (maxgrad<conv_maxgrad.and.rmsgrad<conv_rmsgrad)
end subroutine gprmep_checkConvergence

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_signalConverged
!!
!! FUNCTION
!! Send a signal that the algorithm is converged. 
!! There is also the possibility to decide to go to a finer optimization
!! via "final
subroutine gprmep_signalConverged(gprmep)
!! SOURCE
  use dlf_constants, only: dlf_constants_get
  use dlf_global, only: glob
  class(gprmep_type)        ::  gprmep
  ! if .true. don't change to postopt but converge directly
  character(20)                         ::  filename
  real(rk)                              ::  xGuess(gprmep%nDim), time(1), &
                                            tStep, energy, tmpE
  real(rk)                              ::  times(100), &
                                            positions(gprmep%nDim,100)
#ifdef TestOnMB                                            
  real(rk)                              ::  ang_au
#endif
  integer                               ::  i
!   real(rk)                              ::  vecAlong(gprmep%nDim)
!   real(rk)                              ::  grad(gprmep%nDim), grad1d
#ifdef TestOnMB
  call dlf_constants_get("ANG_AU",ang_au)
#endif
  ! opt_state specifies whether we are in the process of doing
  ! 0 : standard GPRMEP
  ! 1 : TS search
  ! 2 : GPRMEP after TS search
  
  ! gprmep_mode specifies whether we want to do
  ! 0 : just one path optimization
  ! 1 : additional precise gprts search
  ! 2 : after gprts search path optimization again
  ! 3 : calculating real energies on the final path
  
  if (gprmep%opt_state==0.or.gprmep%opt_state==2) then
    write(filename,'(A)') "gprmep_ts_guess.xyz"
    if(printl>=4) write(stdout,'("Writing transition state guess to file ", A)') filename
    open(unit=556, file=filename,status='UNKNOWN', action="write")
    do i = 1, 100
      times(i) = REAL(i-1,kind=rk)/REAL(100-1,kind=rk)
    end do
    call gprmep%curve%evaluate(100, times, positions)
    xGuess(:) = positions(:,1)
    time = 0d0
    call GPR_eval(gprmep%gpPES,xGuess,energy)
    do i = 2, 100
      call GPR_eval(gprmep%gpPES,positions(:,i),tmpE)
      if (tmpE>energy) then
        energy = tmpE
        xGuess = positions(:,i)
        time(1) = REAL(i-1,kind=rk)/REAL(100-1,kind=rk)
      end if
    end do
    ! 1-D Optimization to find the maximum (energy) on the path
    ! starting guess is t=1/2
    tStep = 1d-3
    call gprmep%curve%evaluate(1, time, xGuess)
    call GPR_eval(gprmep%gpPES,xGuess,energy)
    do while (.true.)
      time(1) = time(1) + tStep
      call gprmep%curve%evaluate(1, time, xGuess)
      call GPR_eval(gprmep%gpPES,xGuess,tmpE)
      if(tmpE>energy) then
        ! correct direction
        energy = tmpE
        if (tStep<1d-6) exit
      else
        ! wrong direction
        tStep = -tStep/2d0
        time(1) = time(1) + tStep
        if (tStep<1d-9) exit
      end if
    end do
#ifdef TestOnMB    
    xGuess = xGuess/ang_au
#endif
    call write_xyz(556,gprmep%nat,glob%znuc,xGuess)
    close(556)
  else if (gprmep%opt_state==1) then
    write(filename,'(A)') "gprmep_ts.xyz"
    if(printl>=4) write(stdout,'(&
        "Writing transition state (from GPRTS) to file ", A)') filename
    open(unit=556, file=filename,status='UNKNOWN', action="write")
#ifdef TestOnMB    
    gprmep%TS = gprmep%TS/ang_au
#endif
    call write_xyz(556,gprmep%nat,glob%znuc,gprmep%TS)
    close(556)
  end if
    
  ! Is it the final "signalConverged" call?
  ! (we do not want to converge something else)
  if (gprmep%opt_state==2.or.gprmep%gprmep_mode==0.or.&
      (gprmep%opt_state==1.and.gprmep%gprmep_mode==1)) then
#ifdef setPlotOn      
    gprmep%plotnr = gprmep%plotnr+1
    call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif
    ! this writes out tsmode.xyz and coords.xyz 
    ! (which is turned to result in chemshell)
    if (gprmep%opt_state==0.and.gprmep%gprmep_mode==0) then
      call dlf_put_coords(gprmep%nDim,1,0d0,xGuess,glob%iam)
    else if (gprmep%opt_state==1.and.gprmep%gprmep_mode==1) then
      call dlf_put_coords(gprmep%nDim,1,0d0,gprmep%TS,glob%iam)
    else if (gprmep%opt_state==2.and.gprmep%gprmep_mode==2) then
      call dlf_put_coords(gprmep%nDim,1,0d0,gprmep%TS,glob%iam)
    else if (gprmep%opt_state==3) then
      ! just ignore this assertion here
    else
      call dlf_fail("This case of opt_state/=gprmep_mode should not occur here...)")
    end if
    if (allocated(gprmep%TS)) deallocate(gprmep%TS)
    ! Calculate the real energies of the path here
    if (glob%calc_final_energies) then 
      gprmep%opt_state = 3
      if (gprmep%finalEnergiesToCalc) then
        return
      else
        ! all energies are calculated
        call write_energies(gprmep%nPoints, gprmep%nDim, &
                            gprmep%energiesAtFinalPts, &
                            gprmep%points, &
                            "path_energies")
        deallocate(gprmep%energiesAtFinalPts)
      end if
    end if
    call write_path_xyz(gprmep%nat,gprmep%znuc,"finalpath",&
                        gprmep%nPoints,&
                        gprmep%points,.false.)
!     call gprmep%destroy()
!     call gprmep%curve%backupCurve()
    gprmep%mepConverged = .true.
    if (gprmep%gprmep_mode==1) then 
#ifdef TIMING
        call gprmep%overallClock%beat("GPRTS converged as well")
#endif
#ifdef withopenmp
        gprmep%overallTime2 = omp_get_wtime()
        if(printl>=4) write(stdout,'("Clock GPRTS converged:", ES11.4)') &
            gprmep%overallTime2-gprmep%overallTime1
        gprmep%overallTime1 = omp_get_wtime()
#endif
    end if
  else if(gprmep%opt_state==0.and.gprmep%gprmep_mode>0) then
    call gprmep%changeToPostOpt()  
  else if (gprmep%opt_state==1.and.gprmep%gprmep_mode>1) then
    call gprmep%ts_search_wrapUp()
  else
    call dlf_fail("Invalid opt_state in signalConverged")
  end if
end subroutine gprmep_signalConverged

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_change_nInterPts
!!
!! FUNCTION
!! Change the number of points that are used to build the path
subroutine gprmep_change_nInterPts(gprmep, new_nPts)
!! SOURCE
  class(gprmep_type)    ::  gprmep
  integer, intent(in)   ::  new_nPts
  integer               ::  i
  if (new_nPts==gprmep%nPoints) return
  gprmep%nPoints = new_nPts
  ! # points to calc loss function
  gprmep%n_discrete = MAX(new_nPts,gprmep%n_discrete) 
  ! # points to check for need of energy calc (variance)
  gprmep%fpp_nT = MAX(new_nPts,gprmep%fpp_nT)
!   print*, "gprmep%n_discrete",gprmep%n_discrete 
!   print*, "gprmep%n_discrete", gprmep%n_discrete
  if (allocated(gprmep%points)) deallocate(gprmep%points)
  allocate(gprmep%points(gprmep%nDim,gprmep%nPoints))
  if (allocated(gprmep%oldPoints)) deallocate(gprmep%oldPoints)
  allocate(gprmep%oldPoints(gprmep%nDim,gprmep%nPoints))
  if (allocated(gprmep%d_times)) deallocate(gprmep%d_times)
  if (allocated(gprmep%vecsAlongCurve)) deallocate(gprmep%vecsAlongCurve)
  if (allocated(gprmep%vecsAlongCurve_n)) deallocate(gprmep%vecsAlongCurve_n)
  if (allocated(gprmep%coords_discrete)) deallocate(gprmep%coords_discrete)
  if (allocated(gprmep%esAtDiscretePts)) &
                                    deallocate(gprmep%esAtDiscretePts)
  if (allocated(gprmep%gradsAtDiscretePts)) &
                                    deallocate(gprmep%gradsAtDiscretePts)
  if (allocated(gprmep%hessiansAtDiscretePts)) &
                                    deallocate(gprmep%hessiansAtDiscretePts)
  allocate(gprmep%d_times(gprmep%n_discrete))
  allocate(gprmep%vecsAlongCurve(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%vecsAlongCurve_n(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%coords_discrete(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%gradsAtDiscretePts(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%esAtDiscretePts(gprmep%n_discrete))
  allocate(gprmep%hessiansAtDiscretePts(gprmep%nDim,gprmep%nDim,&
                                        gprmep%n_discrete))
  if(allocated(gprmep%pForceNorm2_discretePts)) &
    deallocate(gprmep%pForceNorm2_discretePts)
  allocate(gprmep%pForceNorm2_discretePts(gprmep%n_discrete))
  if(allocated(gprmep%tNorms_at_dPts))&
    deallocate(gprmep%tNorms_at_dPts)
  allocate(gprmep%tNorms_at_dPts(gprmep%n_discrete))
!   call gprmep%curve%setEqDistPts(gprmep%nPoints,gprmep%points, 1d-3)
  call gprmep%curve%setEqDistPts(new_nPts,gprmep%points, 1d-4)  
  do i = 1, gprmep%n_discrete
    ! I chose not to use the endpoints 
    gprmep%d_times(i) = REAL(i,kind=rk) / &
                        REAL(gprmep%n_discrete+1,kind=rk)
  end do
end subroutine

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_fillListAccordingToVariance
!!
!! FUNCTION
!! Finding all local maxima of the variance along the curve
!! adding those of the maxima that do have high enough variance
subroutine gprmep_fillListAccordingToVariance(gprmep,var_cConv)
!! SOURCE
  class(gprmep_type)                ::  gprmep
  logical, intent(out)              ::  var_cConv ! determines whether the variance
                                                  ! criteria for the curve were satisfied
  integer                           ::  i, pos
  real(rk)                          ::  times(gprmep%fpp_nT)
  real(rk)                          ::  variances(gprmep%fpp_nT)
  real(rk)                          ::  sort_variances(gprmep%fpp_nT)
  real(rk)                          ::  points(gprmep%nDim,gprmep%fpp_nT)
  integer                           ::  nLocMaxima
  real(rk)                          ::  lastVal
  ! the fpp_nV points with highest variance
!   real(rk)                          ::  hi_vars(fpp_nV)
  integer                           ::  hi_pos(gprmep%fpp_nT)
  logical                           ::  goingUp
  var_cConv = .true.
  do i = 1, gprmep%fpp_nT
    times(i) = (REAL(i-1,kind=rk))/(REAL(gprmep%fpp_nT-1,kind=rk))
  end do
  call gprmep%curve%evaluate(gprmep%fpp_nT,times,points)
  ! first one must be extra becaus of parallelization (first one might solve
  ! the linear system for the next cycles to use)
  call GPR_variance(gprmep%gpPES,points(:,1),variances(1))
  !$omp parallel DEFAULT(NONE), private( i ), shared ( gprmep,points, variances)
  !$omp do schedule(static, 1)
  do i = 2, gprmep%fpp_nT
    call GPR_variance(gprmep%gpPES,points(:,i),variances(i))
  end do
  !$omp end parallel
  pos = 1
  lastVal = variances(1)
  nLocMaxima = 0
  goingUp = .true.
  do while (pos<gprmep%fpp_nT)
    if (goingUp) then
      if (variances(pos+1)>=lastVal) then
        pos = pos + 1
        lastVal = variances(pos)
      else
        nLocMaxima = nLocMaxima + 1
        hi_pos(nLocMaxima) = pos
        goingUp = .false.
        pos = pos + 1
      end if
    else
      ! going down
      pos = pos + 1
      lastVal = variances(pos)
      if (variances(pos)>=lastVal) goingUp = .true.
    end if
  end do

  call bubbleSort_dec(nLocMaxima, hi_pos)

  ! sort the variances at the beginning of the variance array
  do i = 1, nLocMaxima
    sort_variances(i) = variances(hi_pos(i))
  end do  
  
  call sort2ArraysUnisonRev(nLocMaxima,sort_variances(1:nLocMaxima),&
                                    hi_pos(1:nLocMaxima))
  nLocMaxima = MIN(nLocMaxima,gprmep%nPoints)
  ! add points with variance check!
  if(printl>=4) write(stdout,'("Variances of suggested points ", &
          "(local maxima of the variance along the path)")')
  do i = 1, MIN(nLocMaxima,gprmep%maxPtsToCalc) ! MIN(nLocMaxima, fpp_nTmax)
    if (variances(hi_pos(i))>gprmep%var_allow_cConv) var_cConv = .false.
    
    if (variances(hi_pos(i))>gprmep%variance_limit) then
      ! add the point
      if(printl>=4) write(stdout,'(A,I4,A,ES10.3,A,ES10.3,A)') &
              "      Pt.nr.", i, " : ", &
              variances(hi_pos(i)), "  >? ",&
              gprmep%variance_limit, &
              " (Need energy)"
      call gprmep%add_PtsToCalc(1,points(:,hi_pos(i)))
    else
      if(printl>=4) write(stdout,'(A,I4,A,ES10.3,A,ES10.3)') &
              "      Pt.nr.", i, " : ", &
              variances(hi_pos(i)), "  >? ",&
              gprmep%variance_limit
    end if
  end do
end subroutine gprmep_fillListAccordingToVariance

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/sort2ArraysUnison
!!
!! FUNCTION
!! sorting (increasing) after values in first array (a)
subroutine sort2ArraysUnison(n,a,b)
!! SOURCE
  integer, intent(in)   ::  n    ! array size
  real(rk),intent(inout)::  a(n) ! first array (reals)
  integer,intent(inout) ::  b(n) ! second array (ints)
  integer               ::  i,j,pos
  real(rk)              ::  aVal
  integer               ::  bVal
  ! bubble sort
  do i = 1, n
    do j = i, n
      pos = SUM(MINLOC(a(i:n))) + i - 1
      ! exchange entry i with entry pos
      aVal    = a(pos)
      a(pos)  = a(i)
      a(i)    = aVal     
      
      bVal    = b(pos)
      b(pos)  = b(i)
      b(i)    = bVal
    end do
  end do
end subroutine sort2ArraysUnison

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/sort2ArraysUnisonRev
!!
!! FUNCTION
!! sorting (decreasing) after values in first array (a)
subroutine sort2ArraysUnisonRev(n,a,b)
!! SOURCE
  integer, intent(in)   ::  n    ! array size
  real(rk),intent(inout)::  a(n) ! first array (reals)
  integer,intent(inout) ::  b(n) ! second array (ints)
  integer               ::  i,j,pos
  real(rk)              ::  aVal
  integer               ::  bVal
  ! bubble sort
  do i = 1, n
    do j = i, n
      pos = SUM(MAXLOC(a(i:n))) + i - 1
      ! exchange entry i with entry pos
      aVal    = a(pos)
      a(pos)  = a(i)
      a(i)    = aVal     
      
      bVal    = b(pos)
      b(pos)  = b(i)
      b(i)    = bVal
    end do
  end do
end subroutine sort2ArraysUnisonRev

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/bubbleSort_dec
!!
!! FUNCTION
!! sorting (decreasing) after values in first array (a)
subroutine bubbleSort_dec(n,b)
!! SOURCE
  integer, intent(in)   ::  n    ! array size
  integer,intent(inout) ::  b(n) ! array (ints)
  integer               ::  i,j,pos
  integer               ::  bVal
  ! bubble sort
  do i = 1, n
    do j = i, n
      pos = SUM(MAXLOC(b(i:n))) + i - 1
      bVal    = b(pos)
      b(pos)  = b(i)
      b(i)    = bVal
    end do
  end do
end subroutine bubbleSort_dec

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_initStandardParas
!!
!! FUNCTION
subroutine gprmep_initStandardParas(gprmep)
  class(gprmep_type)        ::  gprmep
  real(rk)                  ::  maxGBsForKM ! maximum nr of GBs for KM allowed
  gprmep%parasInit = .true.
  gprmep%n_discrete = MAX(60,2*gprmep%nPoints)!MAX(100,2*gprmep%nPoints)!gprmep%nPoints!MAX(100,2*gprmep%nPoints) ! # points to calc loss function
  gprmep%fpp_nT = gprmep%nPoints!gprmep%nPoints!100!2*gprmep%nPoints!MAX(20,2*gprmep%nPoints) ! # points to check for need of energy calc (variance)
  gprmep%maxPtsToCalc = gprmep%nPoints
  gprmep%nOpts_eBased = 4
  if(printl>=6) then
    write(stdout,'("GPRMEP parameters:")')
    write(stdout,'("number of points to evaluate loss functions:                     ",&
          I8)') gprmep%n_discrete
    write(stdout,'("number of points to consider for energy calculation:             ",&
          I8)') gprmep%fpp_nT
    write(stdout,'("Number of optimization runs still considering energy based loss: ",&
          I8)') gprmep%nOpts_eBased
  end if
  ! Parameters for the optimization of the path on the GPR-PES
!   gprmep%manual_e_g_loss=-1d0
  ! Convergence criteria for comparing the new (optimized) curve
  ! to the last (optimized) curve
  gprmep%maxcChangeConv = 1d-3!5d-3!5d-3!1d-2
  gprmep%cChangeConv = 1d-3!5d-3!5d-3!1d-2
  if(printl>=6) then
    write(stdout,'("Limits for curve convergence on GPR-PES:")')
    write(stdout,'("max Change of points on Curve:                                   ",&
            ES11.4)') gprmep%maxcChangeConv
    write(stdout,'("max Change of overall curve:                                     ",&
            ES11.4)') gprmep%cChangeConv
  end if
  ! Maximal change of the parameters during one optimization run on the
  ! GPR-PES (the optimization stops, if a change is larger)
  gprmep%maxPathUpdate = 5d-1 !1d-1
  if(printl>=6) &
    write(stdout,'("max parameter change for complete optimization on GPR-PES:       ",&
            ES11.4)') gprmep%maxPathUpdate
  ! maximally allowed step size in one step of the steps 
  ! of the optimization on the GPR-PES.
  gprmep%maxStepSize = 2d-2
  if(printl>=6) &
    write(stdout,'("max parameter change for one step of the optimization on GPR-PES:",&
            ES11.4)') gprmep%maxStepSize
  gprmep%tolg = gprmep%globtol*1d1  ! 1d-4
  !! For the paper benchmarks I choose gprmep%tolg =   gprmep%globtol*1d1
  !! normally one would to extra calculation for gprmep and then gprts.
  gprmep%tole =    1d-3
  gprmep%tolrmsg = gprmep%tolg / 1.5D0 !* 2d1
  gprmep%tols =    gprmep%tolg * 4.D0
  gprmep%tolrmss = gprmep%tolg * 8.D0/3.D0
!   gprmep%tolg =   gprmep%tolg * 2d1 ! 1d-4
  if(printl>=6) then
    write(stdout,'("Convergence criteria for optimization on the GPR-PES")')
    write(stdout,'("tolg", ES11.4)') gprmep%tolg
!     write(stdout,'("tole", ES11.4)') gprmep%tole
    write(stdout,'("tolrmsg", ES11.4)') gprmep%tolrmsg
    write(stdout,'("tols", ES11.4)') gprmep%tols
    write(stdout,'("tolrmss", ES11.4)') gprmep%tolrmss
    write(stdout,'("tolg", ES11.4)') gprmep%tolg
  end if
  
  ! under which variance the point's energy is not recalculated
  gprmep%variance_limit = 1d-11!*1d-2! 1d-12 
  gprmep%var_allow_cConv = 2.5d-10!*1d-2!5d-11*5d-1
  if(printl>=6) then
    write(stdout,'("Variance parameters")')
    write(stdout,'("Variance criterion                                               ",&
            ES11.4)')gprmep%variance_limit
    write(stdout,'("Variance criterion to allow convergence when the curve ",&
          "on GPR-PES fullfilled convergence criteria",&
            ES11.4)')gprmep%var_allow_cConv
  end if
    ! Multi-level stuff
    maxGBsForKM = 3d0 ! set the maximally allowed nr of GBs for the covariance matrix
                    ! after that, multi-level will minimize that size
    gprmep%maxNrPtsInLevel=dsqrt(maxGBsForKM*1d9/8d0)/gprmep%nDim
    gprmep%shiftToLowerLevel=gprmep%maxNrPtsInLevel/6
    
    gprmep%nOpts = 0
    gprmep%opt_state = 0

    ! Just if the energies of the final path should be calculated on the 
    ! real PES this is relevant. It is just the starting value
    ! that says "There are energies to calculate".
    gprmep%finalEnergiesToCalc = .true.
    ! the number of energies is not known yet
    gprmep%nFinalEnergies = -1
end subroutine gprmep_initStandardParas

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_changeToPostOpt
!!
!! FUNCTION
!! Changing to a finer optimization run with tighter convergence criteria.
subroutine gprmep_changeToPostOpt(gprmep)
!! SOURCE
  class(gprmep_type)                    ::  gprmep
  real(rk)                              ::  times(100), &
                                            positions(gprmep%nDim,100)
  real(rk)                              ::  xGuess(gprmep%nDim), time(1), &
                                            tStep, energy, tmpE
  integer                               ::  i
#ifdef TIMING
  call gprmep%overallClock%beat("Starting GPRTS from the guessed GPRMEP point")
#ifdef withopenmp
  gprmep%overallTime2 = omp_get_wtime()
  write(stdout,'("Clock GPRMEP (without GPRTS) converged:", ES11.4)')&
    gprmep%overallTime2-gprmep%overallTime1
  gprmep%overallTime1 = omp_get_wtime()
#endif
#endif
  ! Optimize the transition state from the highest (energy) point
#ifdef setPlotOn      
    gprmep%plotnr = gprmep%plotnr+1
    call gprmep%curve%writePlotData(100,gprmep%plotnr)
#endif
    do i = 1, 100
      times(i) = REAL(i-1,kind=rk)/REAL(100-1,kind=rk)
    end do
    call gprmep%curve%evaluate(100, times, positions)
    xGuess(:) = positions(:,1)
    time = 0d0
    call GPR_eval(gprmep%gpPES,xGuess,energy)
    do i = 2, 100
      call GPR_eval(gprmep%gpPES,positions(:,i),tmpE)
      if (tmpE>energy) then
        energy = tmpE
        xGuess = positions(:,i)
        time(1) = REAL(i-1,kind=rk)/REAL(100-1,kind=rk)
      end if
    end do

    ! 1-D Optimization to find the maximum (energy) on the path
    ! starting guess is t=1/2
    tStep = 1d-3
    call gprmep%curve%evaluate(1, time, xGuess)
    call GPR_eval(gprmep%gpPES,xGuess,energy)
    do while (.true.)
      time(1) = time(1) + tStep
      call gprmep%curve%evaluate(1, time, xGuess)
      call GPR_eval(gprmep%gpPES,xGuess,tmpE)
      if(tmpE>energy) then
        ! correct direction
        energy = tmpE
        if (tStep<1d-6) exit
      else
        ! wrong direction
        tStep = -tStep/2d0
        time(1) = time(1) + tStep
        if (tStep<1d-9) exit
      end if
    end do
    
    ! save index of pt left and right of the guessed pt
    gprmep%lOfTS = 0
    do i = 1, gprmep%curve%nControlPoints-1
      if (gprmep%curve%pts_in_time(i)>time(1)) then
        !continue
      else
        gprmep%lOfTS=i
        gprmep%rOfTS=i+1
      end if
    end do
  ! xguess is now the highest (energy) point on the curve
  ! start a TS search from there
  if (gprmep%lOfTS==0) call dlf_fail("no lOfTS found...")
  call gprmep%ts_search_init(xguess)
end subroutine gprmep_changeToPostOpt

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_ts_search_init
!!
!! FUNCTION
!! Initialize a transition state search
subroutine gprmep_ts_search_init(gprmep, xguess)
!! SOURCE
  class(gprmep_type)                    ::  gprmep
  real(rk)                              ::  xguess(gprmep%nDim)
  integer                               ::  trConv_onlyTSsearch
  if(printl>=4) then
    write(stdout,'("Starting TS search")')
    write(stdout,'("Energy calculations before TS search:", I8)')&
                    gprmep%gpPES%nt
  end if
  allocate(gprmep%TS(gprmep%nDim))
  allocate(gprmep%oldTSGuess(gprmep%nDim))
  call GPR_Optimizer_define('TS', gprmep%opt,gprmep%gpPES,&
                            gprmep%maxNrPtsInLevel,gprmep%shiftToLowerLevel, &
                            1d-1,MIN(1d-7,gprmep%globtol/5d1),&
                            1d-2,gprmep%globtol)
  
  gprmep%oldTSGuess(:)=xguess(:)
  ! Find a saddle point on this surface as the initial guess
  gprmep%TS = 0d0
  trConv_onlyTSsearch=-1
  call GPR_find_Saddle(gprmep%gpPES, gprmep%opt,0, xguess,&
                       gprmep%TS, trConv_onlyTSsearch, .true.)
  if (trConv_onlyTSsearch==-1) call dlf_fail("ERROR IN GPR_find_Saddle")
  if (trConv_onlyTSsearch==3) then          
    if(printl>=4) then
      write(stdout,'("P-RFO did not find a TS on the GP surface!")')
      write(stdout,'("Use simple dimer translation on GP surface.")')
    end if
    call GPR_find_Saddle(gprmep%gpPES, gprmep%opt,2, xguess,&
                         gprmep%TS,trConv_onlyTSsearch,.true.)
  end if
  gprmep%TS = xguess
  gprmep%opt_state=1
  gprmep%tsSearchJustStarted=.true.
end subroutine gprmep_ts_search_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_ts_search_wrapUp
!!
!! FUNCTION
!! Destroy a transition state search
subroutine gprmep_ts_search_wrapUp(gprmep)
!! SOURCE
  class(gprmep_type)                    ::  gprmep
  if (norm2(gprmep%TS(:)-gprmep%curve%controlPoints(:,gprmep%lOfTS))<&
                      gprmep%globtol.or.&
        norm2(gprmep%TS(:)-gprmep%curve%controlPoints(:,gprmep%rOfTS))<&
                      gprmep%globtol.or.&
      gprmep%gprmep_mode<2) then
  ! do nothing, point is accurately included
    if (gprmep%gprmep_mode<2) then
      if (printl>=4)&
        write(stdout,'("TS is not included in the curve, but it was determined.")')
    else
      if (printl>=4)&
        write(stdout,'("TS is one of the control points.")')
    end if
  else
    
    if (printl>=4) &
      write(stdout,'("TS will be included in the interpolated curve.")')
    call gprmep%curve%addPtBetweenPts(gprmep%lOfTS, gprmep%rOfTS,gprmep%TS)
    call gprmep%curve%setEqDistPts(gprmep%nPoints,gprmep%points, 1d-4)
  end if
  open(unit=556, file="gprmep_ts.xyz",status='UNKNOWN', action="write")
#ifndef TestOnMB
  call write_xyz(556,gprmep%nat,glob%znuc,gprmep%TS)
#endif
#ifdef TestOnMB
  call write_xyz_noConv(556,gprmep%nat,glob%znuc,gprmep%TS)
#endif
  close(556)
  ! Maybe Fix that point, or maybe not...
      
  
  
  ! be aware that the TS is now set. It doesn't have to be found again
  ! by finding the highest (energy) point again like done in 
  ! signalConverged.
!   print*, "A flag to determine whether one wants the exact TS or just the",&
!     " approximation on the curve would be nice."
!   print*, "Up to now it simply gives the latter."
  ! cannot be deallocated here... is still needed
!   deallocate(gprmep%TS)  
    if (allocated(gprmep%oldTSGuess)) deallocate(gprmep%oldTSGuess)
  call GPR_Optimizer_destroy(gprmep%opt)
  gprmep%opt_state=2
end subroutine gprmep_ts_search_wrapUp

subroutine gprmep_testTSconv(gprmep, lastPt_in,lastE_in,lastG_in,TSconverged)
!! SOURCE
  class(gprmep_type)                    ::  gprmep
  real(rk)                  ::  stepTSsearch(gprmep%nDim)
  real(rk),intent(in)       ::  lastPt_in(gprmep%nDim), lastE_in,&
                                lastG_in(gprmep%nDim)
  logical,intent(out)      ::  TSconverged
  TSconverged = .false.
  stepTSsearch = (lastPt_in-gprmep%oldTSGuess)
  if (MAXVAL(ABS(lastG_in))<gprmep%globtol.and.&
      norm2(lastG_in)/dsqrt(REAL(gprmep%nDim,kind=rk))<gprmep%globtol/1.5d0.and.&
      MAXVAL(ABS(stepTSsearch))<gprmep%globtol*4d0.and.&
      norm2(stepTSsearch)/dsqrt(REAL(gprmep%nDim,kind=rk))<gprmep%globtol*8.d0/3.d0)&
      then
    TSconverged = .true.
  end if
end subroutine gprmep_testTSconv

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_calcFinalPathEnergies
!!
!! FUNCTION
!! Determine next pt to calculate etc. when calculating energies
!! of the final path
subroutine gprmep_calcFinalPathEnergies(gprmep, newPt, lastE_in)
!! SOURCE
  class(gprmep_type)                    ::  gprmep
  real(rk), intent(inout)               ::  newPt(gprmep%nDim)
  real(rk), intent(in)                  ::  lastE_in
  ! only re-calculate the energies of the final path
  ! pay attention that one can use GPR-PES energies for points which
  ! are calculated already
  if(gprmep%nFinalEnergies==-1) then
    ! First energy must be calculated. Also initialize everything needed.
    gprmep%nFinalEnergies = gprmep%nPoints
    allocate(gprmep%energiesAtFinalPts(gprmep%nFinalEnergies))
    ! Calculate starting from the last
    newPt(:) = gprmep%points(:,gprmep%nFinalEnergies)
    return
  end if
  ! here the incoming energy has to be saved
  gprmep%energiesAtFinalPts(gprmep%nFinalEnergies) = lastE_in
  gprmep%nFinalEnergies = gprmep%nFinalEnergies - 1
  if(gprmep%nFinalEnergies==0) then
    ! very last point was calculated
    gprmep%finalEnergiesToCalc = .false.
    call gprmep%signalConverged()
    if (gprmep%mepConverged) return
  else
    ! new energy must be calculated at point nr gprmep%nFinalEnergies
    newPt(:) = gprmep%points(:,gprmep%nFinalEnergies)
    return
  end if    
end subroutine gprmep_calcFinalPathEnergies

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_checkPtsToCalcForVarChanges
!!
!! FUNCTION
!! Check which points must be added to the ToCalcList according to variance
subroutine gprmep_checkPtsToCalcForVarChanges(gprmep)
!! SOURCE
  class(gprmep_type)        ::  gprmep
!   integer, intent(in)       ::  nPtsToCheck_in
  integer                   ::  nPtsToCheck
  real(rk)                  ::  variance
  integer                   ::  i, j
  integer                   ::  nToStay=0
  logical                   ::  stay(gprmep%ptToCalcList%np-&
                                     gprmep%ptToCalcList%ptIter)
  real(rk), allocatable     ::  oldList(:,:)
  integer                   ::  start
  ! the next point that will be checked out is (start + 1)
  start = gprmep%ptToCalcList%ptIter
  ! cannot check more points than the number of pts that are in the list
  nPtsToCheck = gprmep%ptToCalcList%np-start!MIN(nPtsToCheck_in,gprmep%ptToCalcList%np-start)
  stay(:) = .true.
  nToStay = nPtsToCheck
  
  do i = 1, nPtsToCheck
    call GPR_variance(gprmep%gpPES,&
                      gprmep%ptToCalcList%coords(:,start+i),variance)
    if (variance<gprmep%variance_limit) then
      ! delete that element
      stay(i) = .false.
      nToStay = nToStay - 1
      if (printl>=6)&
        write(stdout,'("One element does not have to be calculated anymore.",&
          "Its variance is low enough.")')
    end if
  end do
  if (nToStay<nPtsToCheck) then
    ! delete some pts
    if (printl>=4)&
        write(stdout,'("eliminating ", I8,&
            " points in the ToCalcList. They do not satisfy",&
            " the variance criterion anymore:")') nPtsToCheck-nToStay
    allocate(oldList(gprmep%nDim,gprmep%ptToCalcList%np))
    oldList(:,:) = gprmep%ptToCalcList%coords
    deallocate(gprmep%ptToCalcList%coords)
    allocate(gprmep%ptToCalcList%coords(&
        gprmep%nDim,gprmep%ptToCalcList%np-start-(nPtsToCheck-nToStay)))
    
    j = 1
    do i = 1, nPtsToCheck
      if (stay(i)) then
        ! copy it
        gprmep%ptToCalcList%coords(:,j) = oldList(:,start+i)
        j = j + 1
      else
        ! do not copy it
        if (printl>=4)&
          write(stdout,'("Point nr ", I8)') start + i
      end if
    end do
    ! copy unchecked points
    do i = start+nPtsToCheck+1, gprmep%ptToCalcList%np
      gprmep%ptToCalcList%coords(:,j) = oldList(:,i)
      j = j + 1
    end do
    ! deleted the old points below ptIter
    gprmep%ptToCalcList%ptIter = 0
    gprmep%ptToCalcList%np = gprmep%ptToCalcList%np-start-(nPtsToCheck-nToStay)
    deallocate(oldList)
  else
    if (printl>=6)&
        write(stdout,'("All points to be calculated ",&
            "satisfied the variance check again.")')
  end if
end subroutine gprmep_checkPtsToCalcForVarChanges

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/projectOut_nVector
!!
!! FUNCTION
!! projecting out a normalized! Vector "projOut" from the vector "vec"
subroutine projectOut_nVector(nDim, vec, projOut)
!! SOURCE
  integer, intent(in)       :: nDim
  real(rk), intent(inout)   :: vec(nDim)
  real(rk), intent(in)      :: projOut(nDim)
  vec(:) = vec(:) - dot_product(vec(:),projOut(:)) * projOut(:)
end subroutine projectOut_nVector

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_init_and_alloc
!!
!! FUNCTION
!! Initialize (allocating arrays etc.) for gprmep optimization
subroutine gprmep_init_and_alloc(gprmep)
!! SOURCE
  class(gprmep_type)    ::  gprmep
  integer               ::  i, tmpI
  if (.not.gprmep%parasInit) call dlf_fail("You must first initialize the parameters.")
  if (allocated(gprmep%d_times)) deallocate(gprmep%d_times)
  if (allocated(gprmep%vecsAlongCurve)) deallocate(gprmep%vecsAlongCurve)
  if (allocated(gprmep%vecsAlongCurve_n)) deallocate(gprmep%vecsAlongCurve_n)
  if (allocated(gprmep%coords_discrete)) deallocate(gprmep%coords_discrete)
  if (allocated(gprmep%esAtDiscretePts)) &
                                    deallocate(gprmep%esAtDiscretePts)
  if (allocated(gprmep%gradsAtDiscretePts)) &
                                    deallocate(gprmep%gradsAtDiscretePts)
  if (allocated(gprmep%hessiansAtDiscretePts)) &
                                    deallocate(gprmep%hessiansAtDiscretePts)
  if (allocated(gprmep%react)) deallocate(gprmep%react)
  if (allocated(gprmep%prod)) deallocate(gprmep%prod)
  allocate(gprmep%d_times(gprmep%n_discrete))
  allocate(gprmep%vecsAlongCurve(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%vecsAlongCurve_n(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%coords_discrete(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%gradsAtDiscretePts(gprmep%nDim,gprmep%n_discrete))
  allocate(gprmep%esAtDiscretePts(gprmep%n_discrete))
  allocate(gprmep%hessiansAtDiscretePts(gprmep%nDim,gprmep%nDim,&
                                        gprmep%n_discrete))
  allocate(gprmep%react(gprmep%nDim))
  allocate(gprmep%prod(gprmep%nDim))
  
  if (allocated(gprmep%points)) deallocate(gprmep%points)
  allocate(gprmep%points(gprmep%nDim,gprmep%nPoints))
  if (allocated(gprmep%oldPoints)) deallocate(gprmep%oldPoints)
  allocate(gprmep%oldPoints(gprmep%nDim,gprmep%nPoints))
  if (allocated(gprmep%znuc)) deallocate(gprmep%znuc)
  allocate(gprmep%znuc(gprmep%nat))
  ! stuff for gprmep_loss_parsAndGrad
  ! USE OPENMP NOT IN THIS BUT IN SMALLER SUBROUTINES
! #ifdef withopenmp
!   tmpI = omp_get_max_threads()
! #endif
! #ifndef withopenmp   
  tmpI = 1
! #endif
  if(allocated(gprmep%projMat)) &
    deallocate(gprmep%projMat)
  allocate(gprmep%projMat(gprmep%nDim, gprmep%nDim,tmpI))
  if(allocated(gprmep%tmpMat)) &
    deallocate(gprmep%tmpMat)
  allocate(gprmep%tmpMat(gprmep%nDim, gprmep%nDim,tmpI))
  if(allocated(gprmep%pForceNorm2_discretePts)) &
    deallocate(gprmep%pForceNorm2_discretePts)
  allocate(gprmep%pForceNorm2_discretePts(gprmep%n_discrete))
  if(allocated(gprmep%tNorms_at_dPts))&
    deallocate(gprmep%tNorms_at_dPts)
  allocate(gprmep%tNorms_at_dPts(gprmep%n_discrete))
  
  gprmep%status = 0
  gprmep%endPtsCalculated = .false.
  gprmep%unitsOpen = .false.
  gprmep%unitp = 1741
  gprmep%gpPES%iChol = .true.
  gprmep%mepConverged = .false.
  ! initialize the GPR surface
  call GPR_construct(gprmep%gpPES, 0, gprmep%nat, gprmep%nDim, 3,1,1)
  call manual_offset(gprmep%gpPES,10d0)
  call GPR_init_without_tp(gprmep%gpPES,1d0/(20d0)**2,1d0,&
                           (/1d-7,1d-7,3d-4/))
  do i = 1, gprmep%n_discrete
    ! I chose not to use the endpoints 
    gprmep%d_times(i) = REAL(i,kind=rk) / &
                        REAL(gprmep%n_discrete+1,kind=rk)
  end do
end subroutine gprmep_init_and_alloc

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dl-find/gprmep_saveImage
!!
!! FUNCTION
!! current coordinates are written to .xyz files
!! (for every image one)
subroutine gprmep_saveImage(gprmep)
!! SOURCE
  use dlf_global, only: glob
  class(gprmep_type)        ::  gprmep
  integer                   ::  iimage
  integer                   ::  maxxyzfile=100
  character(30)             ::  filename
  if(.not.gprmep%unitsOpen) then
    do iimage=1,gprmep%curve%nControlPoints
      if(iimage>maxxyzfile) exit
      if(iimage<10) then
        write(filename,'("000",i1)') iimage
      else if(iimage<100) then
        write(filename,'("00",i2)') iimage
      else if(iimage<1000) then
        write(filename,'("0",i3)') iimage
      else
        write(filename,'(i4)') iimage
      end if
      filename="gprmep_"//trim(adjustl(filename))//".xyz"
!       if (glob%iam == 0) &
      open(unit=gprmep%unitp+iimage,file=filename)
    end do

    ! write initial xyz coordinates
!     if (glob%iam == 0) then
      do iimage=1,gprmep%nPoints !gprmep%curve%nControlPoints
        if(iimage>maxxyzfile) exit
!         if(xyzall) then ! all atoms not only "active" ones
          call write_xyz(gprmep%unitp+iimage,gprmep%nat,glob%znuc,gprmep%points(:,iimage))
      end do
!     end if
  end if
end subroutine gprmep_saveImage

subroutine gprmep_randomizeUntilVarLow(gprmep, point)
  class(gprmep_type)        ::  gprmep
  real(rk),intent(inout)    ::  point(gprmep%nDim)
  real(rk)                  ::  distortion(gprmep%nDim), variance
  call GPR_variance(gprmep%gpPES,point,variance)
  call init_random_seed()
  do while (variance<gprmep%variance_limit) 
    call random_number(distortion)
    distortion = distortion*1d-6
    point = point + distortion
    call GPR_variance(gprmep%gpPES,point,variance)
  end do
end subroutine gprmep_randomizeUntilVarLow

subroutine reversed_write_path_xyz(nat,znuc,filename,nPts,points, multFiles)
  use dlf_parameter_module, only: rk
  implicit none
  
  character(*), intent(in)  ::  filename
  integer, intent(in)       ::  nat
  integer, intent(in)       ::  znuc(nat)
  integer, intent(in)       ::  nPts
  real(rk), intent(in)      ::  points(nPts,3*nat)
  real(rk)                  ::  points_ordered(3*nat,nPts)
  logical, intent(in)       ::  multFiles
  integer                   ::  i
  do i = 1, nPts
    points_ordered(:,i) = points(i,:)
  end do
  call write_path_xyz(nat,znuc,filename,nPts,points_ordered, multFiles)
end subroutine reversed_write_path_xyz

end module gprmep_module

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!! Müller Brown Stuff
module MB_mod
  use dlf_parameter_module, only: rk
  real(rk) :: acappar(4),apar(4),bpar(4),cpar(4),x0par(4),y0par(4),pi
  integer   ::  eval_counter
  logical   ::  initialized = .false.
end module MB_mod

subroutine init_MB()
  use MB_mod
  implicit none
  real(rk) :: ebarr_=0.5D0
 if (.not.initialized) then
  acappar(1)=-200.D0*ebarr_/106.D0
  acappar(2)=-100.D0*ebarr_/106.D0
  acappar(3)=-170.D0*ebarr_/106.D0
  acappar(4)=  15.D0*ebarr_/106.D0
  apar(1)=-1.D0
  apar(2)=-1.D0
  apar(3)=-6.5D0
  apar(4)=0.7D0
  bpar(1)=0.D0
  bpar(2)=0.D0
  bpar(3)=11.D0
  bpar(4)=0.6D0
  cpar(1)=-10.D0
  cpar(2)=-10.D0
  cpar(3)=-6.5D0
  cpar(4)=0.7D0
  x0par(1)=1.D0
  x0par(2)=0.D0
  x0par(3)=-0.5D0
  x0par(4)=-1.D0
  y0par(1)=0.D0
  y0par(2)=0.5D0
  y0par(3)=1.5D0
  y0par(4)=1.D0
  pi=4.D0*atan(1.D0)
  eval_counter = 0
  initialized = .true.
 end if
end subroutine init_MB

subroutine MB(coords,energy,gradient,hess)
  use MB_mod
  !  Mueller-Brown Potential
  !  see K Mueller and L. D. Brown, Theor. Chem. Acta 53, 75 (1979)
  !  taken from JCP 111, 9475 (1999)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk)  ,intent(in)             :: coords(3)
  real(rk)  ,intent(out)            :: energy
  real(rk)  ,intent(out)            :: gradient(3)
  real(rk)  ,optional,intent(out)   :: hess(3,3)
  !
  ! variables for Mueller-Brown potential
  real(rk) :: x,y,svar,svar2, dxvar, dyvar
  integer  :: icount
  eval_counter = eval_counter + 1
! **********************************************************************
!  call test_update

  !print*,"coords in energy eval",coords
  x =  coords(1)
  y =  coords(2)
      
  energy=0.D0
  gradient=0.D0
  hess(:,:) = 0d0
  do icount=1,4
    svar= apar(icount)*(x-x0par(icount))**2 + &
      bpar(icount)*(x-x0par(icount))*(y-y0par(icount)) + &
      cpar(icount)*(y-y0par(icount))**2 
    svar2= acappar(icount) * dexp(svar)
    energy=energy+ svar2
    dxvar = (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))
    dyvar = (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))
    gradient(1)=gradient(1) + svar2 * dxvar
    gradient(2)=gradient(2) + svar2 * dyvar
    hess(1,1)=hess(1,1)+(2d0*apar(icount)+dxvar**2)*svar2
    hess(1,2)=hess(1,2)+(bpar(icount)+dxvar*dyvar)*svar2
    hess(2,2)=hess(2,2)+(2d0*cpar(icount)+dyvar**2)*svar2
  end do
  hess(2,1)=hess(1,2)
  energy=energy+0.692D0
end subroutine MB

subroutine plotNEBwithCurve(nImages,nAtoms)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout, stderr, printl
  integer, intent(in)       ::  nImages
  integer, intent(in)       ::  nAtoms
  character(12)             ::  filename
  real(rk)                  ::  points(3*nAtoms,nImages)
  integer                   ::  status,i, j
  logical                   ::  eof
  character(4)              ::  buffer
  real(rk)                  ::  time
  
  
  do i = 1, nImages
    write(filename,fmt='(A4,I4.4,A4)') "neb_",i,".xyz"
    if(printl>=4) write(stdout,'(A)') filename
    open(unit=555,file=filename, action='read')
    eof=.false.
    do while (.not.eof)
      read(555,*, iostat=status)
      if (status<0) eof=.true.
    end do
    if (eof) then
      do j = 1, nAtoms+1
        backspace 555
      end do
    else 
      call dlf_fail("should not happen (plotNEBwithCurve)")
    end if
    do j = 1, nAtoms
      read(555,FMT='(A4,F10.7,2X,F10.7,2X,F10.7)',end=201,err=200) &
        buffer, points((j-1)*3+1,i), &
        points((j-1)*3+2,i), points((j-1)*3+3,i)
    end do
!     if(printl>=6) write(stdout,'(A)') "Coords", points(:,i)
    close (555)    
  end do
  
  do i = 1, nAtoms*3
    write(filename,FMT='(A4,I4.4,A4)') "pNEB",i,".plt"
    open(unit=555,file=filename, action='write')  
    do j = 1, nImages
      time = REAL(j-1,kind=rk)/REAL(nImages-1,kind=rk)*100
      write(555,*) time, points(i,j)/0.529177249d0
    end do
    close(555)
  end do
  

  return  
200 continue
  if(printl>=6) write(stderr,'("Error reading NEB file")') 
  call dlf_fail("Error reading NEB file")
201 continue
  if(printl>=6) write(stderr,'("Error (EOF) reading NEB file")') 
  call dlf_fail("Error (EOF) reading NEB file")
end subroutine plotNEBwithCurve

SUBROUTINE init_random_seed()
!! SOURCE
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
          
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
          
    CALL SYSTEM_CLOCK(COUNT=clock)
          
    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
          
    DEALLOCATE(seed)
END SUBROUTINE init_random_seed
