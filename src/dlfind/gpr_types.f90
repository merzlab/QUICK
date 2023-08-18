module gpr_types_module
    use dlf_parameter_module, only: rk
    implicit none
    
    ! some global timing variables
    real(rk)    ::  glob_time_1_
    real(rk)    ::  glob_time_2_
#ifdef withopenmp
    real(rk)    ::  glob_time_1_omp_
    real(rk)    ::  glob_time_2_omp_
#endif
    
    type :: taylor_type
        logical                 ::  initialized = .false.
        real(rk), allocatable   ::  ptOfExpansion(:)
        real(rk)                ::  energy
        real(rk), allocatable   ::  grad(:)
        real(rk), allocatable   ::  hess(:,:)
        real(rk), allocatable   ::  tmp_vec(:)
        real(rk), allocatable   ::  tmp_vec2(:)
    end type taylor_type
    
    type :: gpr_type
        integer                 ::  nat     ! number of atoms
        integer                 ::  sdgf    ! Spatial degrees of freedom
        integer					::  idgf	! Internal degrees or freedom
        integer					::  internal! Needed to differntiate between normal and internal coords
        integer                 ::  kernel_type
                                            ! choosing the kernel
                                            ! = 0 > squared exponential
        logical                 ::  ext_data=.false.
                                            ! is external gpr data used?
                                            ! (mainly interesting for setting 
                                            ! of the mean value)
        logical                 ::  providingKM = .false. 
                                            ! Is the covariance matrix provided
                                            ! when data is read
        logical                 ::  provideTrafoInFile
                                            ! Writes align_refcoords,
                                            ! align_modes, refmass in file
        logical                 ::  massweight ! Using massweighted coordinates?
        integer                 ::  K_stat  ! -1 not valid/allocated
                                            ! 0 Covariance matrix
                                            ! 1 -deleted- LU decomposition of Cov Mat.
                                            ! 2 -deleted- Inverse by LU
                                            ! 3 Cholesky decomposition
                                            ! 4 Inverse by Cholesky (only upper)
                                            ! 5 myOwnCholesky(linear) solved
                                            ! 6 myOwnCholesky(linear) solved, 
                                            !   last CholNotIncluded points 
                                            !   not included
                                            ! 7 myOwnCholesky(linear) solved, 
                                            !   last CholNotIncluded points 
                                            !   not solved, but included
                                            ! 8 Inverse by Cholesky (complete)
        logical                 ::  iChol=.true.
                                            ! use my own iterative cholesky
                                            ! decomposition
        integer                 ::  solveMethod = 3
                                            ! which method should be used to
                                            ! solve the linear system
                                            ! 1 -deleted- LU decomposition
                                            ! 2 -deleted- inverse by LU
                                            ! 3 Cholesky
                                            ! 4 Inverse by Cholesky
        logical                 ::  w_is_solution
                                            ! The weight vector this%w is 
                                            ! currently the solution to the
                                            ! equation KM*w=y
        real(rk),dimension(:,:),allocatable ::  KM ! Covariance matrix
        real(rk), dimension(:),pointer      ::  KM_lin ! Covariance matrix
        real(rk), dimension(:),pointer      ::  KM_lin_chol ! Covariance matrix
    
        integer                             ::  nk_lin
        integer                             ::  nk_lin_old
        ! the following amount of training points (+corresponding array sizes)
        ! are allocated additional to the currently required one
        ! whentraining points are added 
        ! -> better memory management
        ! it is chosen dependent on the order of the GP
        ! padding is only activated as soon as the first training
        ! point is added through "GPR_add_tp"
        integer                             ::  max_pad
        ! for how many hessians should be padded when using arbitrary training points
        integer                             ::  max_pad_hess42
        integer                             ::  nt_pad = 0
        integer                             ::  ene_pad = 0
        integer                             ::  grad_pad = 0
        integer                             ::  hess_pad = 0        
        real(rk), dimension(:), pointer     ::  w   ! weights
        real(rk), dimension(:,:),pointer    ::  xs ! coordinates
        real(rk), dimension(:,:),pointer    ::  ixs !internal coordinates
        real(rk), dimension(:), pointer     ::  es ! energies
        real(rk), dimension(:,:),pointer    ::  gs ! gradients
        real(rk), dimension(:,:,:),pointer	::  b_matrix! B matrix needed for transformating the KM      
        real(rk), dimension(:,:), pointer   ::  igs ! gradient, when using internal coordinates
        real(rk), dimension(:,:,:),pointer  ::  hs ! hessians
        integer, dimension(:),pointer       ::  order_tps(:)
        integer                         ::  active_targets = -1 ! which of the
                                            ! targets below is actively used
        ! First set of pointers: These are just referenced by the pointers above
        ! only interesting for development/debugging
        real(rk), dimension(:), pointer     ::  work_1
        real(rk), dimension(:),pointer      ::  KM_lin_1
        real(rk), dimension(:),pointer      ::  KM_lin_chol_1 ! Covariance matrix
        real(rk), dimension(:), pointer     ::  w_1  
        real(rk), dimension(:,:),pointer    ::  xs_1 
        real(rk), dimension(:,:),pointer    ::  ixs_1
        real(rk), dimension(:), pointer     ::  es_1
        real(rk), dimension(:,:),pointer    ::  gs_1
        real(rk), dimension(:,:,:),pointer	::  b_matrix_1
        real(rk), dimension(:,:), pointer   ::  igs_1
        real(rk), dimension(:,:,:),pointer  ::  hs_1
        integer, dimension(:),pointer       ::  order_tps_1(:)
        ! Second set of pointers: These are just referenced by the pointers above
        ! only interesting for development/debugging
        real(rk), dimension(:), pointer     ::  work_2
        real(rk), dimension(:),pointer      ::  KM_lin_2
        real(rk), dimension(:),pointer      ::  KM_lin_chol_2 ! Covariance matrix
        real(rk), dimension(:), pointer     ::  w_2
        real(rk), dimension(:,:),pointer    ::  xs_2
        real(rk), dimension(:,:),pointer    ::  ixs_2
        real(rk), dimension(:), pointer     ::  es_2
        real(rk), dimension(:,:,:),pointer	::  b_matrix_2
        real(rk), dimension(:,:),pointer    ::  igs_2
        real(rk), dimension(:,:),pointer    ::  gs_2
        real(rk), dimension(:,:,:),pointer  ::  hs_2
        integer, dimension(:),pointer       ::  order_tps_2(:)
        ! Needed for dl-find:
        real(rk)                            ::  tmpEnergy(1)
        ! Needed for coordinate transformation, simply copied from our 
        ! neuralnetwork code (see there for further information)
        real(rk), allocatable    ::  align_refcoords(:)
        real(rk), allocatable    ::  align_modes(:,:)
        real(rk), allocatable    ::  refmass(:)
        ! Some additional parameters for GPR
        integer                  ::  nk      ! size of covariance matrix
        integer                  ::  nk_old
        integer                  ::  nt      ! # training sets
        integer                  ::  nt_old      ! # training sets
        integer                  ::  ntg, ntg_old, nth, nth_old
        real(rk)                 ::  mean    ! prior mean value
        integer                  ::  order      ! which kind of data is provided
                                        ! and used by GPR (energy, grad, hess)
                                        ! = 0 : energies
                                        ! = 1 : gradients
                                        ! = 2 : hessians
                                        ! = 42: arbitrary
        integer                  ::  old_order
                                        ! saves the last set order
        ! Squared exponential things:
        real(rk)                 ::  s_f    ! parameter in covariance functions
                                    ! (simply a factor, that can improve 
                                    !  numerical stability for the solving
                                    !  of the linear system in rare cases)
        real(rk)                 ::  gamma
                                    ! =1/l**2 with l being the "lengthscale"
                                    ! SE is defined via gamma, Matern via l
        real(rk)                 ::  s_n(3)
                                    ! The standard deviation of the assumed
                                    ! normally distributed noises
                                    ! on energies (s_n(1)), 
                                    ! gradients (s_n(2)) and hessians (s_n(3))
                                    ! only s_n**2 is used -> the variance
        ! Matern things:
        real(rk)                 ::  l ! can be interpreted as a characteristic
                                    ! lengthscale and "corresponds" to
                                    ! 1/sqrt(gamma) in the SE kernel
        ! to speed up:
        real(rk)                 ::  l4 ! l^4
        real(rk)                 ::  l2 ! l^2  
        real(rk)                 ::  s_f2
        ! program flow things:    
        logical                  ::  initialized ! is the GPR initialized?
        
        ! For parameter optimization the parameters are scaled up
        logical                  ::  scaling_set =.false.
        real(rk), allocatable    ::  scaling(:)
        
        logical                  ::  constructed=.false. ! is the GPR constructed?
        real(rk)                ::  sKM(2) ! Scaling for the linear system
        integer                 ::  OffsetType !defines the kind of (prior mean) offset:
                                    ! 0: no offset, just take the mean
                                    ! 1: Take a standard offset (higher)
                                    ! 2: Take the last trainingpoint in es 
                                    !    as an offset
                                    ! 3: Choose an offset manually though
                                    !    subroutine manual_offset (init with 0) that can be added to the 
                                    ! max of all energy values
                                    ! 4: take standard offset (lower)
                                    ! 5: GPR as offset (multilevelGPR)
                                    ! 6: -deleted- SaddlePoint Bias
                                    ! 7: linear interpolation between first
                                    !    and last es
                                    ! 8: Taylor-Expansion at a certain point
        integer                 ::  oldOffsetType
        real(rk)                ::  manualOffset=0d0
        type(taylor_type)       ::  taylor
        ! choosing if the manual offset should be added to maxval
        ! of es, or be simply a constant independent of the es
        logical                 ::  addToMax
        real(rk),  allocatable  ::  directSmallEigval(:)
        logical                 ::  directSmallEigvalSet=.false.
        integer, allocatable    ::  wait_vBoldScaling(:) ! counts down to allow
                                    ! very bold scaling to happen only every
                                    ! 20 steps or so
        type(gpr_type), POINTER ::  lowerLevelGPR => null() ! For multi-level
                                    ! can point to another GP from which the 
                                    ! current GP learns the error
        type(gpr_type), pointer ::  GPR_int2 => null() ! Learning from a Method 2 GPR                            
        integer                 ::  level=0 ! Level of the current GP
        integer                 ::  nSubLevels=0
        ! A tolerance that allows to check if the noise parameters are still
        ! sensible for the task
        real(rk)                ::  tolerance
        real(rk)                ::  tol_noise_factor
    end type gpr_type
    
    type :: gpr_type_pointerlist
      type(gpr_type), pointer  ::  gp
    end type
    
    ! rotation index type
    type :: rot_index_type
        integer                 ::  start
        integer                 ::  end
    end type rot_index_type
    
    ! define a Dimer type
    type :: gpr_dimer_type
        real(rk), allocatable   ::  midPt(:)  ! midpoint of the dimer
        real(rk), allocatable   ::  endPt(:)  ! direction of the dimer
                                                    ! from the midpoint
        real(rk)                ::  length    ! the length of the dimer
                                    ! this should be adapted close to
                                    ! convergence according to 
                                    ! http://epubs.siam.org/doi/pdf/10.1137/110843149
                                    ! the dimer_length should always be
                                    ! 2*(dimer_endPt-dimer_midPt) when
                                    ! a dimer is placed somewhere
        logical                 ::  constructed = .false.
    end type
      
    ! Manager for different iterations in Multi-level GPR
    type :: gpr_iteration_manager
        ! The number of iterations used at the moment
        integer                                 ::  nIterations
        ! The links to the GP_0s in every iteration
        ! The first entry will be the original GP_0
        ! Starting from the second the iterations are referenced
        type(gpr_type_pointerlist),allocatable  ::  level0Links(:)
        ! DeltaEnergy that must be achieved in the last iteration
        real(rk)                                ::  eDelta
        ! Same for the gradient entries
        real(rk)                                ::  gDelta
        ! Same for the Hessian entries
        real(rk)                                ::  hDelta
    end type
end module gpr_types_module
