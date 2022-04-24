!!****h* main/dlf_global
!!
!! FUNCTION
!! Module to store global data and arrays
!!
!! Most input parameters and global options are documented here
!!
!! The following routines have to be changed if this module is changed:
!! * dlf_default_init
!! * dlf_default_set
!! * dlf_get_params (external !)
!! * dlf_checkpoint_write
!! * dlf_checkpoint_read
!! * and others, depending on the change
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
!! SYNOPSIS
module dlf_global
  use dlf_parameter_module, only: rk
  implicit none

  ! Type declaration
  type glob_type
    integer  :: nvar        ! number of xyz variables (3*nat)
    integer  :: nat         ! number of xyz atoms
    real(rk) :: tolerance   ! main convergence criterion (Max grad comp.)
    real(rk) :: tolerance_e ! convergence criterion on energy change
    real(rk) :: energy
    real(rk) :: oldenergy
    logical  :: toldenergy  ! is oldenergy defined?
    logical  :: tinit=.false. ! true if this instance is initialised
    logical  :: tatoms      ! atoms or arbitrary DOF
    integer  :: maxcycle    ! maximum number of cycles
    integer  :: maxene      ! maximum number of E&G evaluations
    real(rk) :: distort     ! shift start structure along coords2 (+ or -)
    logical  :: massweight  ! use mass-weighted coordinates
    integer  :: maxdump     ! do only dump restart file after the at most
                            !   maxdump E&G evaluations
    logical  :: toldenergy_conv ! does the convergence module know of 
                            !     an old energy?
    real(rk) :: oldenergy_conv ! value of the old energy in convergence
    integer  :: task        ! number of taks for the task manager

    ! Types of calculations
    integer  :: iopt         ! type of optimisation algorithm
    integer  :: iline        ! type of line search or trust radius
    integer  :: imultistate  ! type of multistate calculation (0 = none)
    integer  :: needcoupling ! true if interstate coupling gradients 
                             !  should be calculated
    integer  :: imicroiter   ! flag for microiterative calculations 
                             ! =0 : standard, non-microiterative calculation
                             ! >0 : microiterative calculation
                             !  [=1 : inside macroiterative loop]
                             !  [=2 : inside microiterative loop]

    ! Line search properties
    real(rk) :: maxstep     ! maximum length of the step in internals
    real(rk) :: scalestep   ! constant factor with which to scale the step
    logical  :: taccepted   ! is this step accepted

    ! Optimiser parameters
    integer  :: lbfgs_mem   ! number of steps in LBFGS memory
    real(rk) :: temperature ! temperature for thermal analysis

    ! qTS parameters
    integer  :: nzero       ! number of zero vibrational modes in system 
    integer  :: qtsflag     ! additional info, like if tunnelig splittings are
                            ! to be calculated (see dlf_qts.f90)

    ! Damped dynamics
    real(rk) :: timestep 
    real(rk) :: fric0       ! start friction
    real(rk) :: fricfac     ! factor to reduce friction (<1) whenever
                            !   the energy is decreasing
    real(rk) :: fricp       ! friction to use whenever energy increasing

    ! Conical intersection search
    integer  :: state_i     ! lower state
    integer  :: state_j     ! upper state
    real(rk) :: pf_c1       ! penalty function parameter (aka alpha)
    real(rk) :: pf_c2       ! penalty function parameter (aka beta)
    real(rk) :: gp_c3       ! gradient projection parameter (aka alpha0)
    real(rk) :: gp_c4       ! gradient projection parameter (aka alpha1)
    real(rk) :: ln_t1       ! Lagrange-Newton orthogonalisation on 
                            !   threshold
    real(rk) :: ln_t2       ! Lagrange-Newton orthogonalisation off 
                            !   threshold

    ! hessian parameters
    integer  :: update      ! Hessian update scheme
    integer  :: maxupd      ! Maximum number of Hessian updates
    real(rk) :: delta       ! Delta-x in finite-difference Hessian
    real(rk) :: soft        ! Abs(eigval(hess)) < soft -> ignored in P-RFO
    integer  :: inithessian ! Option for method of calculating the initial 
                            !   Hessian
    integer  :: carthessian ! Hessian update in cartesians?
    logical  :: tsrelative  ! Transition vector I/O absolute or relative?
    real(rk) :: minstep     ! Hessian is not updated if step < minstep

    ! internal coordinates
    integer  :: icoord      ! type of internal coordinates
    integer  :: icoordinner ! type of internal coordinates for inner region
    integer  :: nivar       ! number of variables in internals
    logical  :: tcoords2    ! are coords2 used?
    integer  :: nimage      ! Number of images (e.g. in NEB)
    real(rk) :: nebk        ! force constant for NEB
    real(rk) :: neb_climb_test  ! threshold scale factor for spawning climbing image
    real(rk) :: neb_freeze_test ! threshold scale factor for freezing NEB images
    integer  :: maxrot      ! maximum number of rotations in each DIMER step
    real(rk) :: tolrot      ! angle tolerance for rotation (deg) in DIMER

    integer  :: ncons       ! number of constraints
    integer  :: nconn       ! number of user provided connections

    logical  :: havehessian ! do we have a valid hessian?

    ! restarting
    integer  :: dump        ! after how many E&G calculations to dump a 
                            !   checkpoint file?
    integer  :: restart     ! restart mode: 0 new, 1 read dump file ...

    ! cleanup
    integer :: cleanup      ! check that dlf_fail does not enter an infinite loop
                            ! when cleaning up allocatable arrays

    ! Parallel optimisation
    integer  :: po_pop_size      ! sample population size
    real(rk) :: po_radius_base   ! base value for the sample radii
    real(rk) :: po_contraction   ! factor by which the search radius decreases between 
                                 ! search cycles.  Cycle = 1 energy eval for 
                                 ! each member of the sample population.
    real(rk) :: po_tolerance_g   ! convergence criterion: max abs component of g
    real(rk) :: po_tol_r_base    ! base value for the radii tolerances
    integer  :: po_distribution  ! type of distribn of sample points in space
    integer  :: po_maxcycle      ! maximum number of cycles
    integer  :: po_init_pop_size ! size of initial population
    integer  :: po_reset         ! number of cycles before population resetting
    real(rk) :: po_mutation_rate ! Fraction of the total number of coordinates 
                                 ! in the population to be mutated (randomly 
                                 ! shifted) per cycle
    real(rk) :: po_death_rate    ! Fraction of the population to be replaced by
                                 ! offspring per cycle 
    real(rk) :: po_scalefac      ! Multiplying factor for the absolute gradient 
                                 ! vector in the force_bias stoch. search scheme
    integer  :: po_nsave         ! number of low-energy minima to store

    ! MPI-related variables that need to be known throughout the code, with 
    ! initialization for the case of a serial buils without a call to the dummy 
    ! routine dlf_mpi_initialize
    integer  :: nprocs=1         ! Total number of processors involved
    integer  :: ntasks=1         ! number of taskfarms (workgroups)
    integer  :: nprocs_per_task=1 ! number of processors per taskfarm
    integer  :: iam=0            ! rank (or index, starting from zero) of this 
                                 ! processor within global_comm
    integer  :: iam_in_task=0    ! rank (or index, starting from zero) of this
                                 ! processor within the most local communicator
    integer  :: mytask=0         ! index, starting from zero, of the task farm 
                                 ! to which this processor belongs
    !integer  :: master_in_task   ! rank of the spokesperson proc in each farm
    !                             ! within the communicator of a taskfarm

    integer  :: serial_cycle     ! Parallel NEB - calculate the first e/g array
                                 ! on all processors to ensure each image 
                                 ! calculation starts from a good guess on 
                                 ! the first cycle
    logical  :: dotask=.true.    ! Flag set if this processor should do the
                                 ! task in question

   ! ARRAYS
    real(rk),allocatable :: xcoords(:,:) ! xyz coordinates 
                                         !   (3,nat) / (1,nvar)
    real(rk),allocatable :: xcoords2(:,:,:)! xyz coordinates second set
                                         ! (3,nat,nframe) / (1,nvar,nframe)
                                         ! or different, depending 
                                         ! on the purpose!
    real(rk),allocatable :: xgradient(:,:)! xyz gradient (3,nat) / (1,nvar)
    integer ,allocatable :: znuc(:)      ! nuclear charges (nat)
    real(rk),allocatable :: weight(:)    ! weight for tsmodes etc (nat)
    real(rk),allocatable :: iweight(:)   ! weight of internal coordinates
                                         !   (nivar)
    real(rk),allocatable :: mass(:)      ! atomic mass (nat)

    ! arrays concerning internals
    real(rk),allocatable :: icoords(:)   ! internal coords (nivar)
    real(rk),allocatable :: igradient(:) ! gradient in internals (nivar)
    real(rk),allocatable :: ihessian(:,:)! Hessian in internals (nivar,nivar)
                                         !  allocation handled in formstep
    real(rk),allocatable :: step(:)      ! step in internals (nivar)
    integer ,allocatable :: spec(:)      ! (nat) fragment number, or
                                         ! -1: frozen,
                                         ! -2: x frozen, see dlf_coords.f90
    integer ,allocatable :: icons(:,:)   ! (5,ncons) constraints
    integer ,allocatable :: iconn(:,:)   ! (2,ncons) user provided 
                                         !  connections

    ! multiple state arrays for conical intersection searches
    real(rk),allocatable :: msenergy(:)       ! multistate energy
    real(rk),allocatable :: msgradient(:,:,:) ! xyz multistate gradients
    real(rk),allocatable :: mscoupling(:,:)   ! xyz interstate coupling 
                                              !   gradient

    ! arrays for parallel optimisation
    real(rk), allocatable :: po_radius(:) ! per-atom search radii (the prevailing units)
    real(rk), allocatable :: po_tolerance_r(:) ! tolerances on po_radius: stop if any component 
             ! of po_radius shrinks to less than the corresponding component of po_tolerance_r
    real(rk), allocatable :: po_radius_scaling(:) ! vector of scaling factors to convert 
             ! the base values (po_radius_base and po_tol_r_base) into the above 2 vectors.

    ! microiterative optimisation
    integer ,allocatable :: micspec(:)  ! (nat) inner region flag from spec array
    integer :: nicore ! number of internal coordinates in inner region 
                      ! (=0 for non-microiterative calculation)
    integer :: maxmicrocycle ! max number of microiterative cycles 
                             ! before switching back to macro
    logical :: micro_esp_fit ! fit ESP charges to inner region during microiterations
    ! ESP fit correction arrays
    ! final dimension in each is for multiple images
    real(rk),allocatable :: macrocoords(:,:,:) ! xyz coordinates for last macro step
    real(rk),allocatable :: g0corr(:,:,:)      ! xyz gradient correction for ESP fit
    real(rk),allocatable :: e0corr(:)                      ! E0 - E0(esp)

  end type glob_type

  ! Variables in the module
  type(glob_type) ,save :: glob
  integer           :: stdout=2021
  integer           :: stderr=0
  real(rk)          :: pi     ! 3.1415...
  integer           :: printl ! how verbosely to write info to stdout
  integer           :: printf ! how verbosely files should be written
  logical           :: keep_alloutput=.true. ! Can be used to provide some 
  ! control over I/O in a parallel run, in conjunction with the interface 
  ! routine dlf_output.  For example: if true, write the output from 
  ! procs with rank > 0 to output.proc<proc number>, instead of standard 
  ! output.  Otherwise the output goes to /dev/null, instead of standard out. 

end module dlf_global
!!****

