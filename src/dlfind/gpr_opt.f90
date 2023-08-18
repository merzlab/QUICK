! **********************************************************************
! **                   Gaussian Process Regression                    **
! **                      for interpolation and                       **
! **                      geometry optimisation                       **
! **********************************************************************
!!****h* gpr
!!
!! NAME
!! gpr
!!
!! FUNCTION
!! Several methods to use GPR in geometry optimization and interpolation
!!
!! GPR_Optimizer_define - Defines which kind of optimization should be performed
!!                        and specifies some parameters for that
!! GPR_Optimizer_step - Perform a step of the defined optimizer
!! GPR_Optimizer_destroy - Deletes allocated memory of the optimizer

module gpr_opt_module
  use gpr_types_module
  use gpr_module
  implicit none
    type :: optimizer_type
        integer                  ::  opt_type_int=0 ! Defines the type of 
                                                ! optimization (GPR_Optimizer)
                                                ! -1 will be Max search
                                                ! 1 will be Min search
                                                ! 0 TS search
        real(rk)                ::  maxstep ! The maximum step that the optimizer
                                    ! is allowed to take. Very important! to
                                    ! limit the overshooting processes.
                                    ! could be between 0.5 and 1.0 a.u. 
                                    ! in Cartesians
        real(rk)                ::  tolerance ! Tolerance value, corresponding
                                    ! to the one in dl-find
                                    ! Convergence criteria are calculated 
                                    ! from that
        real(rk)                ::  tolOnGPR ! the tolerance criterion
                                    ! for searches on the GP itself
        real(rk)                ::  tol_g ! Convergence criterion for max entry
                                    ! of the gradient (like in dl-find)
        real(rk)                ::  tol_rmsg ! Convergence criterion for norm
                                    ! of the gradient (like in dl-find)
        real(rk)                ::  tol_s ! Convergence criterion for max entry
                                    ! of the step vector (like in dl-find)
        real(rk)                ::  tol_rmss ! Convergence criterion for norm
                                    ! of the step vector (like in dl-find)
        real(rk)                ::  maxStepOnGPR ! Step size limitation for
                                    ! steps taken on the GP itself
        real(rk)                ::  oldgradsq ! saves the squared norm of the
                                    ! gradient obtained in the last step
                                    ! to compare it to the current step
                                    ! (improved or not?)
        real(rk)                ::  gradsq ! squared norm in the current step
        
        real(rk)                ::  tol_variance
        
        ! Convergence criteria
        logical                 ::  sConverged=.false. ! max step entry converged?
        logical                 ::  rmssconv=.false. ! RMS(step) converged?
        logical                 ::  gConverged=.false. ! max grad entry converged?
        logical                 ::  rmsgconv = .false. !RMS(grad) converged?
        logical                 ::  lastStepWasOvershot=.false. ! was the last
                                    ! step already overshot
         
        real(rk)                ::  aktuelle_spur
        logical                 ::  sepDimOs = .false.
        logical                 ::  step_os = .false.
        ! In the following borders the standard overshooting will take place
        real(rk)                ::  max_bold_scaling, min_bold_scaling
        ! This parameter limits the upper borde of the standard overshooting
        real(rk)                ::  limit_bold_scaling
        ! Starting with this cos(angular_deviation), the standard overshooting
        ! will take place
        real(rk)                ::  angle_start_bold_scaling
        ! After how many steps the mean offset type should be changed during
        ! optimization (is not used in the first paper version from 
        ! end of 2017)
        integer                 ::  changeMeanOffset
        ! Determines how many data points should be in GPR_0, the last GP
        ! (relevant for multi-level)
        integer                 ::  MaxnPointsinLevel
        ! Determines how many data points should be put into a "higher" level
        ! (to GP_1 from GP_0 for example) after MaxnPointsinLevel is reached
        integer                 ::  nPointsToLowerLevel
        ! Save the original values in the following variables
        ! (the variables above must be changed along the TS search)
        integer                 ::  MaxnPtsLevel_save
        integer                 ::  nPtsinLowLevel_save
        
        ! For seperate dimension overshooting
        real(rk) ::  dimMaxScaling
        integer  ::  nStepsVBScaling
        real(rk) ::  dimStepSize
        real(rk) ::  dimMinValue
        integer  ::  nPause        
        
        ! For transition state searches
        
        ! Shows whether the returned "position" as one that
        ! should be considered to be a guess for the TS or is just
        ! needed in the rotational optimization.
        logical                 ::  posIsResultOfRot ! Used in DL-FIND!
        ! dimer
        type(gpr_dimer_type)    ::  dimer
        ! Number of translational steps performed
        integer                 ::  nTransSteps
        ! Number of translational steps before 
        ! a new rotational optimization is performed
        integer                 ::  nTransBeforeRotOpt
        ! Counter for the number of translations without optimization
        integer                 ::  nTransSinceRot
        ! Tolerance value for translation (usually not needed)
        real(rk)                ::  transTolerance
        ! Tolerance value of rotation
        real(rk)                ::  rotTolerance
        ! Tolerance value of rotation in the first rotation procedure
        real(rk)                ::  rotToleranceFirstStep
        ! Length of the dimer
        real(rk)                ::  dimer_length
        ! Is this the first rotation procedure? 
        logical                 ::  firstRotation
        ! Remembers the indices of the rotation in level 0
        ! (necessary to prevent in multi-level GPR from 
        ! splitting points that were within one rotation)
        type(rot_index_type),&
        allocatable             ::  rot_indices(:)
        ! number of these
        integer                 ::  nrot_indices
        ! Damping factor for classical dimer translation
        real(rk)                ::  dimerTransDamping
        ! number of classical dimer translations before saddle
        ! point search on the GP is performed
        integer                 ::  nDimerTransBegin
        ! How many saddle point searches shall be performed !
        ! (starting from nDimerTransBegin) until a classical dimer trans
        ! is performed again
        integer                 ::  nTSSearchesUntilDimerTrans
        ! number of max(gradient) changes considered to to decide
        ! whether the search of Saddle point on the GP
        ! is converging or not
        integer                 ::  nDelGs
        ! The changes of the 
        real(rk),allocatable    ::  delGs(:)
        real(rk)                ::  oldGMax
        ! If the last #nDelgs max(gradient) changes summed up
        ! are smaller than this value, the search for a saddle
        ! point on the GP surfae is aborted
        real(rk)                ::  minGchange
        ! #training points that were endpoint of the dimer at the current position
        ! (# rotations since last translation + 1)
        integer                 ::  nEndPts
        ! the last dimer midpt/the last position before a translation
        real(rk),allocatable    ::  lastMidPt(:)
        ! Projection matrix for translational modes
        real(rk),allocatable    ::  projTransMat(:,:)
        ! TempMatrix for projection
        real(rk),allocatable    ::  tmpProjTransMat(:,:)
        ! translational vectors
        real(rk),allocatable    ::  transModes(:,:)
        
        ! NEB path approx thingy
        integer                 :: nat
        integer                 :: nimage
        integer                 :: varperimage
        real(rk)                :: temperature
        real(rk)                :: S_0,S_pot,S_ins
        real(rk),allocatable    :: ene(:)
        real(rk),allocatable    :: xcoords(:,:)
        real(rk),allocatable    :: dtau(:)
        real(rk)                :: etunnel
        real(rk),allocatable    :: dist(:)
        integer                 :: position
        integer                 :: direction
        logical                 :: limitStepExternally=.true.
        logical                 :: initDimer=.true.
        integer                 :: stateNEBap
                                   !0: chose first point and read in NEB approx
                                   !1: search for optimal point on NEB approx
                                   !2: search for TS with NEB approx points
                                   !   next time (include another point)
                                   !3: search for TS with NEB approx points
                                   !   immediately
                                   !4: start with usual rot./transl. TS search
    end type optimizer_type
  private   ::  GPR_opt_step_overshooting, GPR_opt_sepDim_overshooting, &
            GPR_Optimizer_step_TS, GPR_Optimizer_TS_dimerDirect, &
            angle_along, GPR_initProjectOutTrans, GPR_projectOutTrans        
  public    ::  GPR_Optimizer_define, GPR_Optimizer_destroy, &
            GPR_Opt_GlobGPRExtreme, &
            GPR_Optimizer_step, &
            GPR_construct_dimer, GPR_place_dimer, &
            GPR_destroy_dimer, GPR_move_dimer, GPR_move_dimer_to, &
            GPR_newLengthOf_dimer, GPR_rotateToDirection_dimer, &
            gpr_searchglobal_flag, GPR_find_Saddle, &
            GPR_Find_Extreme
  logical, save               ::  gpr_searchglobal_flag = .false.
  integer                     ::  gpr_sConvergedCounter = 0
  contains 
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Optimizer_define
!!
!! FUNCTION
!! Initializes the optimization algorithm that makes use of GPR
!! Here it is specified, if one wants to search for a maximum or a minimum.
!! Also the tolerance of the including master-program (e.g. dl-find) 
!! should be given. Giving the same tolerance as dl-find will result in the 
!! same convergence criteria as in dl-find.
!! tolOnGPR and maxStepOnGPR define the parameters for the internal
!! LBFGS. If not set to a specific value, default values will be defined.
!!
!! SYNOPSIS
subroutine GPR_Optimizer_define(opt_type,&
    opt, this, MaxnPointsinLevel, nPointsToLowerLevel, &
    maxstep, tolOnGPR, maxStepOnGPR, tolerance)!, tolerance_e)
!! SOURCE    
    implicit none
    character(len=*), intent(in)        ::  opt_type
    type(optimizer_type), intent(out)   ::  opt
    type(gpr_type), intent(inout)       ::  this
    integer, intent(in)                 ::  MaxnPointsinLevel
    integer, intent(in)                 ::  nPointsToLowerLevel
    real(rk), optional, intent(in)      ::  maxstep
    real(rk), optional, intent(in)      ::  tolOnGPR
    real(rk), optional, intent(in)      ::  maxStepOnGPR
    real(rk), optional, intent(in)      ::  tolerance
    real(rk)                            ::  maxGBsForKM
    
    if(this%sdgf>300.and.printl>=6) then
      write(stdout,'("WARNING: The number of dimensions for this ",&
      "System is over 300. On some machines this may cause sudden stop/",&
      "other problems because memory can be depleted.")')
      write(stdout,'("The program automatically",& 
      " limits the number of allowed training points per level (multi-level",&
      "-scheme. But even if it does run through, ",&
      "the number of training points in the last level", &
      " can be small. This might lead to bad convergence of the optimizer.")')
      write(stdout,'("Much larger system sizes (maybe near to 1000 dimensions) ", &
      " are only possible on very large machines.")')
    end if
    opt%initDimer=.true.
    if (this%nt==0) opt%stateNEBap=0
    opt%limitStepExternally=.true.
    ! Some standard starting values
    opt%max_bold_scaling=5d0!9d0 !9d0
    opt%min_bold_scaling=1d0
    opt%angle_start_bold_scaling=0.9d0  !0.9d0  ! Should be between 0 and 1 at least,
    opt%limit_bold_scaling=10d0
    !*******************
    ! separate dimension overshooting
    opt%dimMaxScaling = 4d0 !4d0
    opt%nStepsVBScaling = 20 !20
    opt%dimStepSize = opt%tol_s
    opt%dimMinValue=1d-5
    opt%nPause=20 !20

    !*******************
    ! Multi-level stuff
    if (MaxnPointsinLevel==0) then
      ! set default
      maxGBsForKM = 3d0 ! set the maximally allowed nr of GBs for the covariance matrix
                    ! after that, multi-level will minimize that size
      opt%MaxnPointsinLevel=dsqrt(maxGBsForKM*1d9/8d0)/this%sdgf
    else
      opt%MaxnPointsinLevel=MaxnPointsinLevel!60/70
    end if
    if (nPointsToLowerLevel==0) then
      ! set default
      opt%nPointsToLowerLevel=opt%MaxnPointsinLevel/6
    else
      opt%nPointsToLowerLevel=nPointsToLowerLevel!10/20
    end if
    opt%MaxnPtsLevel_save=opt%MaxnPointsinLevel
    opt%nPtsinLowLevel_save=opt%nPointsToLowerLevel
    if(printl>=6) then
      write(stdout,'(A,i10)') &
        "Multilevel - Maximum number of pts: ", &
        opt%MaxnPointsinLevel
      write(stdout,'(A,i10)') &
        "Multilevel - Number of points moved to lower level: ", &
        opt%nPointsToLowerLevel
    end if
    if (.not.this%iChol.and.printl>=4) then
      write(stdout,'(" ")')
      write(stdout,'("Some information on GPR multi-level settings:")')
      write(stdout,'("Limit number of points in level zero to ", I10)')&
              opt%MaxnPointsinLevel
      write(stdout,&
        '("Number of points in the lower Levels should be at least", I10)') &
        opt%nPointsToLowerLevel
    else
      if (printl>=6) &
        write(stdout,&
          '("Using an iterative Cholesky decomposition method to solve ", &
          "the linear system for GPR after each inclusion of a new point.")')
    end if
    
    !*******************
    ! Values for TS search
    if (present(tolOnGPR)) then 
        opt%transTolerance=tolOnGPR
    else
        opt%transTolerance=1d-7!1d-10
    end if
    if (present(tolerance)) then
      if (tolerance<5d1*tolOnGPR) then
        if (printl>=2) then
          write(stdout,'("The tolerance on the GP surface should be ",&
            "considerably smaller than the one on the PES!")')
          write(stdout,'("tolerance:", ES11.4)') tolerance
          write(stdout,'("tolOnGPR :", ES11.4)') tolOnGPR
        end if
      end if
    end if
    allocate(opt%lastMidPt(this%idgf))
    opt%lastMidPt = 0d0
    opt%rotTolerance=1d-3
    opt%rotToleranceFirstStep=1d-4
    opt%dimer_length=1d-1 ! This is only the length of the dimer from midpoint to endpoint (not from end to end)
    ! number of translational steps between rotation optimizations
    ! in transition state search
    opt%nTransBeforeRotOpt= opt%MaxnPtsLevel_save-&
                            opt%nPtsinLowLevel_save
    ! number of classical dimer translations before saddle
    ! point search on the GP is performed
    opt%nDimerTransBegin=0
    ! How many saddle point searches shall be performed
    ! (starting from nDimerTransBegin) until a classical dimer trans
    ! is performed again
    opt%nTSSearchesUntilDimerTrans=1024
    ! number of max(gradient) changes considered to to decide
    ! whether the search of Saddle point on the GP
    ! is converging or not
    opt%nDelGs=512
    allocate(opt%delGs(opt%nDelGs))
    ! If the last #nDelgs max(gradient) changes summed up
    ! are smaller than this value, the search for a saddle
    ! point on the GP surfae is aborted
    opt%minGchange=1d-6
    ! Damping factor for classic dimer translation
    opt%dimerTransDamping=1d-1
    
    if (trim(opt_type)=='MAX') then
        opt%opt_type_int= -1
    else if (trim(opt_type)=='MIN') then
        opt%opt_type_int= 1
    else if (trim(opt_type)=='TS') then
        opt%opt_type_int=0        
        allocate (opt%rot_indices(opt%MaxnPointsinLevel))
        opt%nrot_indices=0
        if(.not.allocated(opt%transModes))&
          allocate(opt%transModes(this%idgf,3))
        if(.not.allocated(opt%projTransMat))&
          allocate(opt%projTransMat(this%idgf,this%idgf))  
        if(.not.allocated(opt%tmpProjTransMat))&
          allocate(opt%tmpProjTransMat(this%idgf,this%idgf))  
        call GPR_initProjectOutTrans(this,opt)
    else
        call dlf_fail("Optimizer type is not known! Choose 'MAX', 'MIN' or 'TS'!")
    end if    
    if (present(maxstep)) then
        opt%maxstep=maxstep
    else
        opt%maxstep=0.5d0
    end if
    
    if (present(tolerance)) then
        opt%tolerance=tolerance
    else
        opt%tolerance=1d-4
    end if
    
    this%tolerance = opt%tolerance

    opt%tol_g=opt%tolerance
    opt%tol_rmsg= opt%tol_g / 1.5D0
    opt%tol_s=    opt%tolerance * 4.D0             
    opt%tol_rmss= opt%tolerance * 8.D0/3.D0
    
    if (present(tolOnGPR).and.present(maxStepOnGPR)) then
        opt%tolOnGPR = tolOnGPR
        opt%maxStepOnGPR = maxStepOnGPR
    else if ((.not.present(tolOnGPR)).and.(.not.present(maxStepOnGPR)))then
      if (printl>=4.and.this%sdgf>1) &
        write(stdout,'("tolerance and maxstep on gpr are set to default!")')
        opt%tolOnGPR = 1d-7
        opt%maxStepOnGPR = 1d-1
    else
        call dlf_fail("Provide both tolOnGPR and maxStepOnGPR for LBFGS or none.")
    end if
    opt%oldgradsq = 0d0
    opt%gradsq    = 0d0
    gpr_searchglobal_flag=.false.
    opt%posIsResultOfRot=.false.
     opt%dimStepSize = opt%tol_s
   opt%tol_variance = 1d-2
end subroutine GPR_Optimizer_define
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Optimizer_destroy
!!
!! FUNCTION
!! Destroy all the variables connected with the optimizer structure
!!
!! SYNOPSIS
subroutine GPR_Optimizer_destroy(opt)
!! SOURCE
  type(optimizer_type), intent(inout)   ::  opt
  if(allocated(opt%delGs))deallocate(opt%delGs)
  if(allocated(opt%rot_indices))deallocate(opt%rot_indices)
  if(allocated(opt%lastMidPt)) deallocate(opt%lastMidPt)
  if(allocated(opt%transModes)) deallocate(opt%transModes)
  if(allocated(opt%projTransMat)) deallocate(opt%projTransMat)  
  if(allocated(opt%tmpProjTransMat)) deallocate(opt%tmpProjTransMat)  
  if (opt%dimer%constructed) call GPR_destroy_dimer(opt%dimer)
end subroutine GPR_Optimizer_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Opt_GlobGPRExtreme
!!
!! FUNCTION
!! This finds a global extremum on the GP. 
!! A search for extrema is carried out starting from every possible training 
!! point in the GP. The lowest/highest of those is chosen as the
!! "global minimum/maximum"
!! - In the case of only one data point you can directly start with 
!!   GPR_Optimizer_step with the one single
!!   data point as your initial guess.
!!
!! SYNOPSIS
subroutine GPR_Opt_GlobGPRExtreme(this,opt, extremum,gpr_hdlc)
!! SOURCE
    use oop_hdlc
    implicit none
    type(gpr_type),intent(inout)        ::  this
    type(optimizer_type), intent(inout)    ::  opt
    real(rk), intent(out)               ::  extremum(this%idgf)
    type(hdlc_ctrl),intent(inout),optional :: gpr_hdlc
    real(rk)                            ::  best_extr_val, tmp
    real(rk)                            ::  tmp_pos(this%sdgf),&
                                            tmp_extr(this%idgf)
    real(rk)                            ::  itmp_extr(this%sdgf)                                        
    real(rk)                            ::  itmp_pos(this%idgf)
    integer                             ::  i,j
    integer                             ::  i_search
    logical                             ::  mask(this%nt)
    ! Search the "global" GPR max/min (starting LBFGS from every single
    ! point in "this". Memorize the highest value and output it.
    if (printl>=6) &
        write(stdout,'("Searching the global extremum on the GP surface.")')
    mask(:)=.true.
    if (opt%opt_type_int==0) &
        call dlf_fail("Must be implemented (GPR_Opt_GlobGPRExtreme)!")
 
    do i = 1, MAX(this%nt/10,MIN(3,this%nt))    ! only search in the lowest/highest 10% of trainingpoints
        if(opt%opt_type_int==1) then
            i_search=sum(minloc(this%es(1:this%nt),mask(1:this%nt)))
        else if(opt%opt_type_int==-1) then
            i_search=sum(maxloc(this%es(1:this%nt),mask(1:this%nt)))
        else
            call dlf_fail("WRONG Optimization type (GPR_Opt_GlobGPRExtreme)")
        end if
               
        mask(i_search)=.false.
        if(this%internal == 2) then
          itmp_pos(:)=this%ixs(:,i_search)
        else
          tmp_pos(:)=this%xs(:,i_search)
        endif
        if(this%internal == 2) then
          call GPR_Find_Extreme(this,opt,itmp_pos,tmp_extr,itmp_extr,gpr_hdlc) 
        else
          call GPR_Find_Extreme(this,opt,tmp_pos,tmp_extr)
        endif 
        if(this%internal == 2) then  
        if (i==1) then
            extremum=tmp_extr
            call GPR_eval(this,itmp_extr,tmp)
            best_extr_val=tmp
        else
            call GPR_eval(this,itmp_extr,tmp)
            if (tmp*opt%opt_type_int < best_extr_val*opt%opt_type_int) then
                extremum=tmp_extr
                best_extr_val=tmp
            end if
        end if
        else
        if (i==1) then
            extremum=tmp_extr
            call GPR_eval(this,tmp_extr,tmp)
            best_extr_val=tmp
        else
            call GPR_eval(this,tmp_extr,tmp)
            if (tmp*opt%opt_type_int < best_extr_val*opt%opt_type_int) then
                extremum=tmp_extr
                best_extr_val=tmp
            end if
        end if 
        endif       
        if (printl>=6) &
          write(stdout,'("Extreme energy values", I8, I8, ES11.4)') &
            i_search,this%nt,tmp
    end do
end subroutine GPR_Opt_GlobGPRExtreme
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Optimizer_step
!!
!! FUNCTION
!! This is the core function of the GPR optimizer.
!! The usual procedure will be to give this function the last guess
!! of the extremum to be searched. This subroutine includes this guess, 
!! with all the given data (energy, gradient, hessian) in the GP
!! and tries to estimate the extremum on the real function (PES) the GP tries
!! to represent.
!! This method does all the overshooting, extimation of extrema,
!! parameter adaption, ...
!!
!! SYNOPSIS
subroutine GPR_Optimizer_step(this,opt,prior_estimate,extremum,&
    est_xs,est_es,est_gs,est_ixs,est_igs,bmatrix,est_hs)
!! SOURCE
    use oop_hdlc
    use dlf_global
    implicit none
    type(gpr_type),intent(inout),target ::  this
    type(optimizer_type),intent(inout)  ::  opt 
    real(rk), intent(in)                ::  prior_estimate(this%idgf)    ! Estimate of the maximum
    real(rk), intent(out)               ::  extremum(this%idgf)    ! Extremum on GPR
    real(rk),optional,intent(in)        ::  est_xs(this%sdgf)  
                                                ! Position of newly added
                                                ! (most likely estimated) point
    real(rk),optional,intent(in)        ::  est_es(1) ! Changed to 1 instead of this%nt
                                                ! Energy...
    real(rk),optional,intent(in)        ::  est_gs(this%sdgf)  
                                                ! Gradient...
    real(rk), optional,intent(in)       ::  est_ixs(this%idgf)
    real(rk),optional, intent(in)       ::  est_igs(this%idgf)
    real(rk),optional, intent(in)       ::  bmatrix(this%sdgf,this%idgf)
    real(rk),optional,intent(in)        ::  est_hs(this%sdgf,this%sdgf)  
                                                ! Hessian...
    real(rk)                            ::  RMS_grad
    real(rk)                            ::  angle
    real(rk)                            ::  extremum_internal(this%sdgf)
    real(rk)                            ::  step(this%idgf)
    real(rk)                            ::  istep(this%idgf)
    logical                             ::  doVeryBoldStep
    integer                             ::  stepsForBolderScaling
    real(rk)                            ::  maxBolderScaling
    real(rk)                            ::  angleStartBolderScaling
    type(gpr_type), pointer             ::  pointGPR
    real(rk)                            ::  mindist
    real(rk)                            ::  scaleFactor=1d0
    integer                             ::  i,k,j
    real(rk)                            ::  pdist_int,pdist_cart,prev_conv
    integer                             ::  paras(1) ,istat 
    real(rk)                            ::  s_n(3),result
    real(rk)                            ::  result_grad(this%idgf),grad,like(3),gh(1),coord(1)
    real(rk)                            ::  start,end,tmp,y(3)
    real(rk)                            ::  var
    real(rk)                            :: xgradient(this%idgf),igradient(this%sdgf),tmp_step(this%idgf)
    type(hdlc_ctrl)                     :: gpr_hdlc
    integer                             ::  max_mid
    logical                             ::  nans_occured
    real(rk)                            ::  est_grad(this%idgf)
    real(rk)                            ::  p(1),new_params(1)
    real(rk)                            ::  new_point(this%sdgf,1),new_energy(1)
    real(rk)                            ::  new_l
    logical                             ::  opt_hyper
    if (opt%opt_type_int==0.and.present(est_gs)) then
      if(this%internal == 2 .or. this%internal == 1) then
        call dlf_error("TS search with GPR in internal coordinates not yet implemented!")
      else 
        call GPR_Optimizer_step_TS(this,opt,extremum,&
          est_xs,est_es,est_gs)
      endif
      return
    end if
    if (printl>=4) &
        write(stdout, '("Opt_step OffsetType",I3)')  this%offsetType
    if(this%internal == 2) then
      opt%gConverged = (MAXVAL(abs(est_igs(:)))<opt%tol_g)  
      opt%rmsgconv = (dsqrt(sum(est_igs(:)**2)/real(this%idgf,KIND=rk)) < opt%tol_rmsg)
    else
      opt%gConverged = (MAXVAL(abs(est_gs(:)))<opt%tol_g)
      opt%rmsgconv = (dsqrt(sum(est_gs(:)**2)/real(this%sdgf,KIND=rk)) < opt%tol_rmsg)
    endif
    stepsForBolderScaling = 50
    maxBolderScaling = 2d0
    angleStartBolderScaling = 0.99d0    
    ! Add the new 
    doVeryBoldStep = .false.

    
  if (this%nt>=3) then  
    if (this%internal == 2) then
      angle=angle_along(this%idgf,this%ixs(:,this%nt-2),&
                        this%ixs(:,this%nt-1),this%ixs(:,this%nt))
    else 
      angle=angle_along(this%sdgf,this%xs(:,this%nt-2),&
                        this%xs(:,this%nt-1),this%xs(:,this%nt))    
    endif 
    if (angle<0d0) gpr_searchglobal_flag=.true.
  end if
  if (this%internal == 2) then
    gpr_hdlc = hdlc_ctrl()
    call gpr_hdlc%dlf_hdlc_init(glob%nat,glob%spec,mod(glob%icoord,10),glob%ncons, &
         glob%icons,glob%nconn,glob%iconn)              
  endif  

  if (present(est_gs)) then
    if(this%internal == 2) then
      opt%gradsq = dot_product(est_igs(:),est_igs(:))
    else
      opt%gradsq = dot_product(est_gs(:),est_gs(:))
    endif
    RMS_grad=dsqrt(opt%gradsq/this%idgf)
!     if ((.not.this%iChol.or.((this%sdgf+1)*this%nt<=5000)) .AND. opt%oldgradsq <= opt%gradsq) then
      if (opt%oldgradsq <= opt%gradsq) then ! CHANGED
        if (printl>=6) &
          write(stdout,'("Old gradient was better, increasing gamma.")')
        gpr_searchglobal_flag=.true.
          call GPR_changeparameters(this,this%gamma*1.1d0,this%s_f,this%s_n)   
          if (printl>=4) &
            write(stdout,'("Length scale parameter decreased to",&
                ES11.4)') 1.D0/sqrt(this%gamma)
          if (this%iChol) then
            this%K_stat = -1
          end if
      end if
  end if

 ! print *,'length;', dsqrt(1/this%gamma)
  opt%oldgradsq = opt%gradsq

  if (present(est_xs)) then
    if (this%order==0) then
      call GPR_add_tp(this,1, est_xs,est_es)
    else if (this%order==1) then
      if(this%internal == 2) then
        call GPR_add_tp(this,1, est_xs,est_es,est_gs,est_ixs,est_igs,bmatrix)
        if(this%nt==1) then
          call gpr_hdlc%dlf_hdlc_create(glob%nat,glob%nicore,glob%spec,glob%micspec, &
             glob%znuc,1,this%ixs(:,1),glob%weight,glob%mass)   
          else 
            call gpr_hdlc%dlf_hdlc_create(glob%nat,glob%nicore,glob%spec,glob%micspec, &
               glob%znuc,1,this%ixs(:,1),glob%weight,glob%mass) 
        endif          
      else
        call GPR_add_tp(this,1, est_xs,est_es,est_gs)
      endif
    else if (this%order==2) then
      call GPR_add_tp(this,1, est_xs,est_es,est_gs,est_hs)
    end if
    ! introduce new level when there are too many points
!    if (this%nt>opt%MaxnPointsinLevel) then! .and.&
  !         .not.((opt%sConverged).or.(opt%gConverged))) then
!      call GPR_newLevel(this,opt%nPointsToLowerLevel)
 !   end if
  end if
  !! if hypeperparameter needs to be adapted.
  !if( this%nt == 5 .or. mod(this%nt,25)==0) then
  !  call GPR_get_hyperparameters(this,1,(/2/),(/1/),(/15/),4,new_params(1)) 
  !  new_l =    new_params(1)
  !  call GPR_changeparameters(this,1d0/this%l**2,this%s_f,(/this%s_n(1),new_l,this%s_n(3)/))
  !  if (this%iChol) then
  !   this%K_stat = -1
  !  end if       
 ! endif

  call GPR_interpolation(this)
   
  if(gpr_searchglobal_flag) then
    if(this%internal ==2) then
    call GPR_Opt_GlobGPRExtreme(this,opt,extremum,gpr_hdlc)
    else
      call GPR_Opt_GlobGPRExtreme(this,opt,extremum)
    endif
    gpr_searchglobal_flag=.false.
  else
    if(this%internal ==2) then
      call GPR_Find_Extreme(this,opt,prior_estimate,extremum,extremum_internal,gpr_hdlc)
    else
      call GPR_Find_Extreme(this,opt,prior_estimate,extremum)
    endif
  end if
  
  
! ***********************************************************************

    step(:)= extremum(:)-prior_estimate(:)
    if (this%nt>=2) then
       call GPR_opt_step_overshooting(this,opt,step, extremum,scaleFactor)
!     if (this%gamma/=1d0/(20d0)**2) then
!       call dlf_fail("GAMMA changed?")
!     end if
       call GPR_opt_sepDim_overshooting(this,opt,scaleFactor,step, extremum)
    end if


    if(this%internal == 2) then 
      xgradient(:) = 0.0d0
      igradient(:) = 0.0d0 
      call gpr_hdlc%dlf_hdlc_xtoi(glob%nvar/3,glob%nivar,glob%nicore,glob%micspec, &
           prior_estimate(:)+step(:) ,xgradient,extremum_internal,igradient)
          call GPR_variance(this,extremum_internal,var)  
          var = dsqrt(var) 
        else
          call GPR_variance(this,prior_estimate(:)+step(:),var)
          var = dsqrt(var)
        endif

! ************************************************************************
! Limit step to maxstep
  if(var>opt%tol_variance) then
    if (dsqrt(dot_product(step(:),step(:)))>opt%maxstep) then
      step(:) = step(:)/dsqrt(dot_product(step(:),step(:)))*opt%maxstep
    end if
  endif

 
  extremum(:)=prior_estimate(:)+step(:)  
!     call GPR_write(this, .false.,  "nn.dat")   
!     call writeParameterFile(this)
    if(printl>=6) then
      write(stdout,'("Extreme_Dim", I9)') this%idgf
      do i = 1, this%idgf
        write(stdout,'("New_extreme", I9, ES11.4)') i, extremum(i)
      end do
    end if
    if(this%internal == 2) call gpr_hdlc%dlf_hdlc_destroy() 
end subroutine GPR_Optimizer_step
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_get_hyperparameters
!!
!! FUNCTION
!! Adapts the hyperparameter by calculating four log p values and 
!! optimize it with a GPR. Currently only the length scale is working.
!!
!! SYNOPSIS
subroutine GPR_get_hyperparameters(this,nr_param,which_params,start,endp,nr_tp,new_params)
!! SOURCE

   type(gpr_type), intent(inout)  ::  this
   integer, intent(in)            ::  nr_param
   integer, intent(in)            ::  which_params(nr_param)
   real(rk),intent(in)            ::  start(nr_param)
   real(rk),intent(in)            ::  endp(nr_param)
   integer,intent(in)             ::  nr_tp
   real(rk),intent(out)           ::  new_params(nr_param)

   integer                        :: i
   real(rk)                       :: like(nr_tp)
   real(rk)                       :: length
   real(rk)                       :: p(nr_tp,nr_param)
   type(gpr_type)                 ::  hyperGPR
   type(optimizer_type)           ::  hyperOpt
   real(rk)                       ::  extremum(nr_param)
   real(rk)                       ::  likel,var
   
   length = start(1)

   do i=1,nr_tp
     length = length+(endp(1)-start(1))/nr_tp
     call GPR_changeparameters(this,1/length**2,this%s_f,this%s_n)
     if (this%iChol) then
      this%K_stat = -1
     end if
     call calc_p_like(this, nr_param, (/which_params/), like(i), p(i,:))
   enddo

   call GPR_Optimizer_define('MAX',hyperOpt,this,200,20)
   hyperGPR%iChol = .false.
   call GPR_construct(hyperGPR, nr_tp,1 ,1   ,4  ,1   ,0)
   call GPR_init(hyperGPR,p(:,1),1/40d0**2,1d0,(/0d0,1d0,1d0/),like)
   call LBFGS_Max_Like(hyperGPR, 1, (/2/))
   call GPR_interpolation(hyperGPR)
   call GPR_Opt_GlobGPRExtreme(hyperGPR,hyperOpt, extremum)
   call GPR_eval(hyperGPR, extremum, likel)
   call GPR_variance(hyperGPR,extremum,var)
   new_params(:) = extremum(:)
   call GPR_destroy(hyperGPR)
end subroutine GPR_get_hyperparameters
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Opt_restart
!!
!! FUNCTION
!! If the backtransformation in method 1 fails, introduce a new level
!!
!! SYNOPSIS
subroutine GPR_Opt_restart(this)
!! SOURCE
  type(gpr_type),intent(inout) :: this
  call GPR_newLevel(this,this%nt)
  if (this%iChol) then
     this%K_stat = -1
  end if
end subroutine GPR_Opt_restart
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_opt_step_overshooting
!!
!! FUNCTION
!! Overshooting procedure for speeding up gpr-optimizations
!!
!! SYNOPSIS
subroutine GPR_opt_step_overshooting(this,opt,step, extremum,scaleFactor)
!! SOURCE
  type(gpr_type), intent(in)          ::  this
  type(optimizer_type), intent(inout) ::  opt
  ! The step vector before the overshooting. will be changed to
  ! respective new, overshot step vector
  real(rk), intent(inout)             ::  step(this%idgf)
  real(rk), intent(in)                ::  extremum(this%idgf)
  real(rk), intent(out)               ::  scaleFactor
  real(rk)                            ::  maxScaleLoc, angle
  logical                             ::  pts_are_theSame = .false.
  if (this%nt<2.or.(opt%opt_type_int==0.and.opt%nTransSteps<1)) then
    ! no overshooting possible    
    call dlf_fail("Called GPR_opt_step_overshooting with too few points.")
  end if
  opt%rmssconv=(dsqrt(sum(step(:)**2)/real(this%idgf,KIND=rk)) < opt%tol_rmss)
  opt%sConverged=(maxval(abs(step(:))) < opt%tol_s)
  
  if (opt%rmssconv.or.opt%sConverged) gpr_sConvergedCounter = 0
  scaleFactor=1d0
  maxScaleLoc = tanh((maxval(abs(step(:)))/&
        opt%tol_s)**(2d0)-1d0)*(opt%max_bold_scaling-1d0)/2d0+&
        (opt%max_bold_scaling-1d0)/2d0+1d0
  if(this%internal == 2) then      
    if (opt%opt_type_int==0) then
      angle=angle_along(this%idgf,opt%lastMidPt,&
                        this%ixs(1:this%idgf,this%nt),extremum,pts_are_theSame)
    else
      angle=angle_along(this%idgf,this%ixs(:,this%nt-1),&
            this%ixs(:,this%nt),extremum,pts_are_theSame)
    end if
  else 
    if (opt%opt_type_int==0) then
      angle=angle_along(this%sdgf,opt%lastMidPt,&
                        this%xs(1:this%sdgf,this%nt),extremum,pts_are_theSame)
    else
      angle=angle_along(this%sdgf,this%xs(:,this%nt-1),&
            this%xs(:,this%nt),extremum,pts_are_theSame)
    end if  
  endif  
  if (pts_are_theSame) then
    call dlf_fail("The newly found point is too close to an already Existing &
    training point. Convergence criteria are too tight for GPR or  &
    the GPR-based optimizer is not able to converge")
  end if
  if(opt%lastStepWasOvershot) then
    if (angle>opt%angle_start_bold_scaling) then
      opt%max_bold_scaling=MIN(opt%max_bold_scaling*1.05d0,opt%limit_bold_scaling)
      if (printl>=4)&
        write(stdout,'("Scaling up max overshooting factor to:  ",ES11.4)') &
            opt%max_bold_scaling
    end if
    opt%lastStepWasOvershot=.false.
  end if
  
  if (this%nt>2.and.((.not.opt%sConverged))) then
!      gpr_sConvergedCounter = gpr_sConvergedCounter - 1 ! (changed since the first paper) but this should not do anything...
!      if (gpr_sConvergedCounter<=0) then ! (changed since the paper)
      ! Reduce the max scaling, if close to convergence
      if(opt%sConverged) maxScaleLoc=dsqrt(maxScaleLoc)
      if (angle>=opt%angle_start_bold_scaling) then
        opt%lastStepWasOvershot=.true.
        scaleFactor=opt%min_bold_scaling+&
          (maxScaleLoc-opt%min_bold_scaling)*&
          ((angle-opt%angle_start_bold_scaling)/&
           (1d0-opt%angle_start_bold_scaling))**4
      
      if (printl>=4) &
        write(stdout,'("Overshooting: angle,factor:",ES11.4,ES11.4)') &
          angle, scaleFactor
        opt%step_os = .true.
      else
        if (printl>=4) &
          write(stdout,'("No overshooting: angle was:", ES11.4)') angle
        end if  
!      end if ! (changed since the paper)
    end if
    
    step(:)=scaleFactor*step(:)
end subroutine GPR_opt_step_overshooting
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_opt_sepDim_overshooting
!!
!! FUNCTION
!! Overshooting procedure that overshoots in some dimensions separately
!! in order to speed up gpr-optimizations
!!
!! SYNOPSIS
subroutine GPR_opt_sepDim_overshooting(this,opt,previous_scaleFactor,step,&
                                        extremum)
!! SOURCE
  type(gpr_type), intent(inout)         ::  this
  type(optimizer_type), intent(inout)   ::  opt
  real(rk), intent(in)                  ::  previous_scaleFactor
  real(rk), intent(inout)               ::  step(this%idgf)
  real(rk), intent(in)                  ::  extremum(this%idgf)
  type(gpr_type)                        ::  dimGPR
  type(optimizer_type)                  ::  dimOpt
  integer                               ::  nStepsPosOrNeg
  integer                               ::  posOrNeg
  integer                               ::  i,j
  logical                               ::  doVeryBoldStep
!   logical, allocatable                  ::  overshootThisDim(:)
  real(rk), allocatable                 ::  equiDist(:), dimValues(:)
  real(rk)                              ::  meanJump
  real(rk)                              ::  dimExtrapolatedPos(1)
  real(rk)                              ::  dimExtrapolated,&
                                            dimExtrapolatedStep
  doVeryBoldStep=.false.
!   allocate(overshootThisDim(this%nt))
!   overshootThisDim(:) = .false.
  if (this%nt>=opt%nStepsVBScaling.and.&
      (previous_scaleFactor<opt%dimMaxScaling)&
      .and.(this%wait_vBoldScaling(1)<=0).and.(.not.opt%sConverged)) then
    do i = 1, this%idgf    
    ! Only for those dimensions that are not "converged"
    if (step(i)>opt%tol_s) then 
      ! Test in every dimension, if overshooting is possible
      nStepsPosOrNeg = 0
      if (step(i)>opt%dimStepSize) then
        ! Are the last 5 steps also positive
        nStepsPosOrNeg = 1
        posOrNeg=1
        do j = 1, this%nt-1
          if(this%internal == 2) then
          if ((this%ixs(i,this%nt-j+1)-this%ixs(i,this%nt-j))>opt%dimStepSize) then
            ! posOrNeg stays 1
            nStepsPosOrNeg = nStepsPosOrNeg + 1
          else if (nStepsPosOrNeg>=opt%nStepsVBScaling) then
            ! posOrNeg stays 1
            EXIT
          else
            ! No scaling to be done
            posOrNeg=0
            EXIT
          end if
          else
          if ((this%xs(i,this%nt-j+1)-this%xs(i,this%nt-j))>opt%dimStepSize) then
            ! posOrNeg stays 1
            nStepsPosOrNeg = nStepsPosOrNeg + 1
          else if (nStepsPosOrNeg>=opt%nStepsVBScaling) then
            ! posOrNeg stays 1
            EXIT
          else
            ! No scaling to be done
            posOrNeg=0
            EXIT
          end if
          endif          
        end do
      else if (step(i)<-opt%dimStepSize) then
        ! Are the last 5 steps also negative
        nStepsPosOrNeg = 1
        posOrNeg=-1
        do j = 1, this%nt-1
          if(this%internal == 2) then
            if ((this%ixs(i,this%nt-j+1)-this%ixs(i,this%nt-j))<-opt%dimStepSize) then
              ! posOrNeg stays -1
              nStepsPosOrNeg = nStepsPosOrNeg + 1
            else if (nStepsPosOrNeg>=opt%nStepsVBScaling) then
              ! posOrNeg stays -1
              EXIT
            else
              ! No scaling to be done
              posOrNeg=0
              EXIT
            end if
          else 
            if ((this%xs(i,this%nt-j+1)-this%xs(i,this%nt-j))<-opt%dimStepSize) then
              ! posOrNeg stays -1
              nStepsPosOrNeg = nStepsPosOrNeg + 1
            else if (nStepsPosOrNeg>=opt%nStepsVBScaling) then
              ! posOrNeg stays -1
              EXIT
            else
              ! No scaling to be done
              posOrNeg=0
              EXIT
            end if
          endif
        end do
      else 
        ! No scaling to be done
        posOrNeg=0        
      end if
      if (posOrNeg/=0) then
        doVeryBoldStep=.true.
        EXIT ! CHANGED
!         overshootThisDim(i) = .true.
      end if
    end if
    end do
  end if
  
  if (doVeryBoldStep) then
    do i = 1, this%idgf
    ! only for those dimensions for which the criterion is fulfilled
!     if (.not.overshootThisDim(i)) cycle ! CHANGED
    ! Only for those dimensions that are not "converged"
    if (step(i)>opt%tol_s) then 
        posOrNeg=1        
    else if (step(i)<-opt%dimStepSize) then
        posOrNeg=-1
    else 
        ! No scaling to be done
      posOrNeg=0        
    end if

      ! Which kind of scaling should be done (if at all)
      if (posOrneg/=0) then
        ! Do Optimization
        if (posOrNeg==1) then
          call GPR_Optimizer_define('MAX',dimOpt,this,200,20)
                                        !nat,sdgf,OST,kern,order
          dimGPR%iChol = .false.
          call GPR_construct(dimGPR, this%nt,1 ,1   ,4  ,1   ,0)           
        else if (posOrNeg==-1) then
          call GPR_Optimizer_define('MIN',dimOpt,this,200,20)
                                                !nat,sdgf,OST,kern,order
          dimGPR%iChol = .false.
          call GPR_construct(dimGPR, this%nt,1 ,1   ,1  ,1   ,0)          
        end if

        allocate(equiDist(this%nt))
        allocate(dimValues(this%nt))
        if(this%internal == 2) then
          equiDist(this%nt) = &
              dot_product(extremum(:)-this%ixs(:,this%nt),&
                          extremum(:)-this%ixs(:,this%nt))
        else
          equiDist(this%nt) = &
              dot_product(extremum(:)-this%xs(:,this%nt),&
                          extremum(:)-this%xs(:,this%nt))        
        endif                
        meanJump = equiDist(this%nt)
        do j = 1, this%nt-1
            if(this%internal == 2) then
              equiDist(j)=dot_product(&
              this%ixs(:,j+1)-this%ixs(:,j),&
              this%ixs(:,j+1)-this%ixs(:,j))
            else
              equiDist(j)=dot_product(&
              this%xs(:,j+1)-this%xs(:,j),&
              this%xs(:,j+1)-this%xs(:,j))            
            endif
            meanJump = meanJump + equiDist(j)
        end do
        meanJump = meanJump/this%nt
        do j = 2, this%nt
            equiDist(j) = equiDist(j)+equiDist(j-1)
        end do

        ! Fill up with all positions in this dimension that show the 
        ! monotonicity (all pos or neg step)
        dimValues(this%nt) = extremum(i)
        if(this%internal == 2)then
          do j = 1, this%nt-1
            dimValues(j) = this%ixs(i,j+1)
          end do                        !g    s_f,s_n  
        else
          do j = 1, this%nt-1
            dimValues(j) = this%xs(i,j+1)
          end do                        !g    s_f,s_n          
        endif
        dimGPR%iChol = .false.
        call GPR_init(dimGPR,equiDist,0.1d0,1d0,(/meanJump*1d-2,1d0,1d0/),dimValues)
        call GPR_interpolation(dimGPR)
        call GPR_Find_Extreme(dimGPR, dimOpt, equiDist(this%nt), dimExtrapolatedPos)
        call GPR_eval(dimGPR, dimExtrapolatedPos, dimExtrapolated)
        if(this%internal == 2) then
        if (ABS(step(i)) < ABS((dimExtrapolated-this%ixs(i,this%nt)))) then
            dimExtrapolatedStep =  dimExtrapolated-this%ixs(i,this%nt)
            dimExtrapolatedStep = &
            SIGN(MIN(ABS(dimExtrapolatedStep),&
                ABS(extremum(i)-this%ixs(i,this%nt))*opt%dimMaxScaling),&
              dimExtrapolatedStep)
              
          if (ABS(step(i))<ABS(dimExtrapolatedStep)) then
            if (printl>=6) then
1827 FORMAT(A, I3, 2x, ES11.4,A3,ES10.3,A4,ES10.3)
              write(stdout,1827) "very bold step", i, &
                extremum(i)-this%ixs(i,this%nt)," / ",step(i), " -> ", &
                dimExtrapolatedStep
            end if
            step(i) = dimExtrapolatedStep
            this%wait_vBoldScaling(1)=opt%nPause
          end if
        end if
        else
                if (ABS(step(i)) < ABS((dimExtrapolated-this%xs(i,this%nt)))) then
            dimExtrapolatedStep =  dimExtrapolated-this%xs(i,this%nt)
            dimExtrapolatedStep = &
            SIGN(MIN(ABS(dimExtrapolatedStep),&
                ABS(extremum(i)-this%xs(i,this%nt))*opt%dimMaxScaling),&
              dimExtrapolatedStep)
              
          if (ABS(step(i))<ABS(dimExtrapolatedStep)) then
            if (printl>=6) then
1828 FORMAT(A, I3, 2x, ES11.4,A3,ES10.3,A4,ES10.3)
              write(stdout,1827) "very bold step", i, &
                extremum(i)-this%xs(i,this%nt)," / ",step(i), " -> ", &
                dimExtrapolatedStep
            end if
            step(i) = dimExtrapolatedStep
            this%wait_vBoldScaling(1)=opt%nPause
          end if
        end if
        endif
        call GPR_destroy(dimGPR)
        deallocate(equiDist)
        deallocate(dimValues)
        gpr_searchglobal_flag=.true.
        opt%sepDimOs = .true.
      end if
    end do
  end if
  this%wait_vBoldScaling(1)=this%wait_vBoldScaling(1)-1
!   deallocate(overshootThisDim) ! CHANGED
end subroutine GPR_opt_sepDim_overshooting
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_find_Saddle
!!
!! FUNCTION
!! Finds a saddle point on the GP-surface via 
!! P-RFO (method==0), smallest eigval follow (method==1) or
!! smalles eigval follow with small steps (method==2)
!! Can also be restricted to "maxstep".
!!
!! SYNOPSIS
subroutine GPR_find_Saddle(this, opt, method,&
        startingPoint,saddlePos,translationConverged, restrict)
!! SOURCE        
    implicit none
    type(gpr_type),intent(inout)::  this
    type(optimizer_type),&
                intent(inout)   ::  opt
    integer, intent(in)         ::  method 
                                    ! 0 : P-RFO 
                                    ! 1 : smallest eigVal Follow 
                                    ! 2 : smallest eigVal Follow
                                    !     with small steps until
                                    !     maxstep is reached.
    real(rk), intent(in)        ::  startingPoint(this%sdgf)
    real(rk), intent(out)       ::  saddlePos(this%sdgf)
    logical, intent(in)         ::  restrict ! determines whether
                                    ! the saddle point search 
                                    ! shall be restricted to
                                    ! maxstep etc...
    real(rk)                    ::  eigVal
    real(rk)                    ::  newDirection(this%sdgf)
    real(rk)                    ::  newPos(this%sdgf),oldPos(this%sdgf)
    real(rk)                    ::  midPtGrad(this%sdgf)
    real(rk)                    ::  oldMidPtGrad(this%sdgf)
    real(rk)                    ::  tmpHess(this%sdgf,this%sdgf),&
                                    tmpHess2(this%sdgf,this%sdgf)
    real(rk)                    ::  tmpEnergy
    real(rk)                    ::  DNRM2
    ! Coded state of the result
    ! -1: error
    ! 0 : step converged
    ! 1 : grad converged and eigval < 0
    ! 2 : stopped at 2*maxstep
    ! 3 : stopped at max number of steps
    integer, intent(out)        ::  translationConverged
    logical                     ::  stopTranslation
    real(rk)                    ::  transForce(this%sdgf)
    real(rk)                    ::  transForceDamper
    ! Not all convergence criteria must be fulfilled for the
    ! saddle point search to be considered converged!
    ! Either the step size is converged,
    ! or the smallestEigval is negativ and gradient small.
    real(rk)                    ::  gradConvergence
    integer                     ::  someCounter, stepCounter
    integer                     ::  gIndex, nzero
    integer                     ::  maxNumberOfSteps
    real(rk)                    ::  delGAv, maxGrad
    integer                     ::  updateCounter
    integer                     ::  updateMax
!******************************************************************************

    if (.not.this%w_is_solution) call dlf_fail("Solve the linear system first.")
    call gpr_driver_init
if (restrict) then
  if (method==0) then
    transForceDamper=0d0
    maxNumberOfSteps=MAX(INT(opt%maxstep/opt%maxStepOnGPR*10d0),250)
  else if (method==1) then
    transForceDamper=2d-1
    maxNumberOfSteps=MAX(INT(opt%maxstep/opt%maxStepOnGPR*10d0),100)
  else if (method==2) then
    transForceDamper=2d-1
    maxNumberOfSteps=MAX(INT(opt%maxstep/opt%maxStepOnGPR/&
                         transForceDamper*10d0),100)
  end if
else
  maxNumberOfSteps=1000 ! That "restriction" should at least be there
end if
    gradConvergence=opt%tol_g
    opt%delGs(:)=0d0
    delGAv=0d0
    translationConverged=-1
    stopTranslation=.false.
    someCounter = 0
    stepCounter = 0
    call GPR_eval(this,startingPoint,tmpEnergy)
    newPos=startingPoint
    call gpr_prfo_init(this)
#ifndef withopenmp
    call GPR_eval_hess(this,newPos,tmpHess)
#endif
#ifdef withopenmp
    call GPR_eval_hess_parallel(this,newPos,tmpHess)
#endif
    
    
#ifndef TestOnMB
    !call GPR_projectOutTrans(this,opt,tmpHess)
#endif
    call GPR_eval_grad(this,newPos,midPtGrad)
#ifdef TestOnMB
    print*, "Warning: in gpr.f90 TestOnMB is activated."
    tmpHess(3,:) = 0d0
    tmpHess(:,3) = 0d0
    midPtGrad(3) = 0d0
#endif
    
!*********************************************************
! Optimization Loop
!*********************************************************
updateMax=0 ! Turned every update off.
            ! It seemed to worsen results.
updateCounter = updateMax ! first step: always evaluate the Hess
do while(.not.stopTranslation)
  SELECT CASE(method)
    CASE (0)
!**********************************    
! P-RFO Saddle search
    someCounter = someCounter +1
      stepCounter = stepCounter + 1
      if (this%nat>2) then
        nzero=6
      else if (this%nat==2) then
        nzero=5
      else if (this%nat==0) then
        nzero=0
      else if (this%nat==1) then
        nzero=3
      end if
      nzero = 0
      newDirection = 0d0
#ifndef TestOnMB
      call gpr_prfo_step(this%sdgf,nzero,.false.,&
           midPtGrad,tmpHess,newDirection) ! here the Hess is intent(in)
#endif
#ifdef TestOnMB
      print*, "Warning: in gpr.f90 TestOnMB is activated."
      nzero=1
      call gpr_prfo_step(this%sdgf,nzero,.false.,&
           midPtGrad,tmpHess,newDirection) ! here the Hess is intent(in)
#endif
      if(DNRM2(this%sdgf, newDirection,1)>opt%maxStepOnGPR)&
        newDirection(:)=newDirection(:)/&
                        dsqrt(sum(newDirection(:)**2))*&
                        opt%maxStepOnGPR
      oldPos=newPos
      newPos=oldPos+newDirection
      oldMidPtGrad = midPtGrad
      call GPR_eval_grad(this,newPos,midPtGrad)
#ifdef TestOnMB
    midPtGrad(3) = 0d0
#endif
      if (updateCounter>=updateMax) then
#ifndef withopenmp        
        call GPR_eval_hess(this,newPos,tmpHess)
#endif
#ifdef withopenmp
        call GPR_eval_hess_parallel(this,newPos,tmpHess)
#endif
#ifndef TestOnMB
        !call GPR_projectOutTrans(this,opt,tmpHess) ! here the Hess is intent(inout)
#endif
#ifdef TestOnMB
    print*, "Warning: in gpr.f90 TestOnMB is activated."
    tmpHess(3,:) = 0d0
    tmpHess(:,3) = 0d0
    midPtGrad(3) = 0d0
#endif
        tmpHess2(:,:) = tmpHess(:,:)
        
        updateCounter = 0
      else
        !update
        ! use smaller step size then
!         if(DNRM2(this%sdgf, newDirection,1)>opt%maxStepOnGPR/5d0)&
!         newDirection(:)=newDirection(:)/&
!                         dsqrt(sum(newDirection(:)**2))*&
!                         opt%maxStepOnGPR/5d0
!         newPos=oldPos+newDirection
!         call GPR_eval_grad(this,newPos,midPtGrad) 
        call gpr_bofill_update(this%sdgf,midPtGrad,oldMidPtGrad,&
                               newPos,oldPos,tmpHess,1d-4) ! here the Hess is intent(inout)
!         call GPR_projectOutTrans(this,opt,tmpHess) ! here the Hess is intent(inout)
        updateCounter = updateCounter + 1
#ifdef TestOnMB
    print*, "Warning: in gpr.f90 TestOnMB is activated."
    tmpHess(3,:) = 0d0
    tmpHess(:,3) = 0d0
    midPtGrad(3) = 0d0
#endif
        tmpHess2(:,:) = tmpHess(:,:)
      end if
      ! tmpHess and tmpHess2 have same entries
      ! saddlePos only used as temporary variable
      call eigVec_to_smallestEigVal(this%sdgf,tmpHess2,saddlePos,eigVal) ! here the Hess is intent(inout)

      maxGrad=MAXVAL(abs(midPtGrad))
      if (printl>=6) &
        write(stdout,'("Performed P-RFO step nr ", I9)') someCounter
!**********************************   
    CASE (1)
!**********************************    
      ! My own saddle search
      stepCounter = stepCounter + 1
        ! Evaluate the Hessian matrix at the midpoint
        call GPR_eval_hess(this,newPos,tmpHess)
        ! Diagonalize the Hessian to find the mode corresponding to the
        ! smallest eigenvalue (There are some parallel algorithms to find the mode
        ! corresponding to the smallest eigenvalue,
        ! maybe I should have a look at them in the future)
        
        call eigVec_to_smallestEigVal(this%sdgf,tmpHess,newDirection,eigVal)

      ! Translate dimer in the new direction
      ! Calculate the force 
      oldMidPtGrad=midPtGrad
      call GPR_eval_grad(this,newPos,midPtGrad)
      
      maxGrad=MAXVAL(abs(midPtGrad))
      gIndex=MOD(stepCounter-1,opt%nDelGs)+1
      delGAv=delGAv-opt%delGs(gIndex)+maxGrad-opt%oldGMax
      opt%delGs(gIndex)=maxGrad-opt%oldGMax
      opt%oldGMax=maxGrad
      
      transForce = -midPtGrad + &
                   2d0*dot_product(midPtGrad,(newDirection)/&
                   DNRM2(this%sdgf, newDirection,1))*&
                   ((newDirection)/&
                   DNRM2(this%sdgf, newDirection,1))
      oldPos=newPos
      newDirection=transForce*transForceDamper
      if(DNRM2(this%sdgf, newDirection,1)>opt%maxStepOnGPR)&
        newDirection(:)=newDirection(:)/&
                        DNRM2(this%sdgf, newDirection,1)*&
                        opt%maxStepOnGPR
      newPos = newPos+newDirection
      call GPR_eval(this,newPos,tmpEnergy)
!**********************************   
    CASE (2)
!**********************************    
    ! My own saddle search
      stepCounter = stepCounter + 1
        ! Evaluate the Hessian matrix at the midpoint
        call GPR_eval_hess(this,newPos,tmpHess)
        ! Diagonalize the Hessian to find the mode corresponding to the
        ! smallest eigenvalue (There are some parallel algorithms to find the mode
        ! corresponding to the smallest eigenvalue,
        ! maybe I should have a look at them in the future)
        
        call eigVec_to_smallestEigVal(this%sdgf,tmpHess,newDirection,eigVal)

      ! Translate dimer in the new direction
      ! Calculate the force 
      oldMidPtGrad=midPtGrad
      call GPR_eval_grad(this,newPos,midPtGrad)
      
      maxGrad=MAXVAL(abs(midPtGrad))
      gIndex=MOD(stepCounter-1,opt%nDelGs)+1
      delGAv=delGAv-opt%delGs(gIndex)+maxGrad-opt%oldGMax
      opt%delGs(gIndex)=maxGrad-opt%oldGMax
      opt%oldGMax=maxGrad
      newDirection = newDirection/DNRM2(this%sdgf, newDirection,1)
      
      transForce = -midPtGrad + &
                   2d0*dot_product(midPtGrad,(newDirection))*&
                   (newDirection)
      oldPos=newPos
      newDirection=transForce*transForceDamper
      if(DNRM2(this%sdgf, newDirection,1)>opt%maxStepOnGPR)&
        newDirection(:)=newDirection(:)/&
                        DNRM2(this%sdgf, newDirection,1)*&
                        opt%maxStepOnGPR
      newPos = newPos+newDirection
      call GPR_eval(this,newPos,tmpEnergy)
!**********************************   
    CASE DEFAULT
      call dlf_fail("Non-existing method chosen for TS search on GP.")
  END SELECT
      ! Stop, if step converged:
      if (DNRM2(this%sdgf, newDirection,1)<opt%transTolerance) then
        if (printl>=4) &
          write(stdout,'("TS search on GPR-PES: step size converged.")')
        translationConverged=0
        stopTranslation=.true.
      ! Stop, if gradient converged:
      else if (eigVal<-1d-10.and.&
          maxGrad<opt%tol_g*1d-2) then
        translationConverged=1
        stopTranslation=.true.
        if (printl>=4) &
          write(stdout,'(&
          "TS search on GPR-PES: gradient converged and smallest eigval<0")')
      ! Stop, if distance to startingPoint is too large (3*maxstep):
      ! (only in method 0+1) -> translation not converged
      else if (restrict.and.method/=2.and.&
               NORM2(newPos-startingPoint)>opt%maxstep*2d0) then
        stopTranslation=.true.
        translationConverged=2
        if (printl>=4) &
          write(stdout,'("TS search on GPR-PES: stopped at 2*maxstep")')
      ! Stop, if distance to startingPoint is too large (1*maxstep):
      ! (only in method 2) -> translation converged
      else if (restrict.and.method==2.and.&
            NORM2(newPos-startingPoint)>opt%maxstep) then
        stopTranslation=.true.
        translationConverged=2
        if (printl>=4) &
          write(stdout,'("TS search on GPR-PES: stopped at maxstep")')
      else if (stepCounter>maxNumberOfSteps) then
        stopTranslation=.true.
        translationConverged=3
        if (method==2) translationConverged=3
        if (printl>=4) &
          write(stdout,'("TS search on GPR-PES: step not converged")')
      end if
!*********************************************************
! End of Optimization Loop
!*********************************************************
end do
    saddlePos = newPos
end subroutine GPR_find_Saddle
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Optimizer_step_TS
!!
!! FUNCTION
!! Does an optimization step to find a transition state via gpr-opt
!!
!! SYNOPSIS
subroutine GPR_Optimizer_step_TS(this,opt,nextPoint,&
    est_xs,est_es,est_gs)
!! SOURCE    
    use dlf_global
    implicit none
    type(gpr_type),intent(inout),target ::  this
    type(optimizer_type),intent(inout)  ::  opt 
    real(rk), intent(out)               ::  nextPoint(this%sdgf)    ! post_estimate or needed training point on GPR
    real(rk),intent(in)        ::  est_xs(this%sdgf)  
                                                ! Position of newly added
                                                ! (most likely estimated) point
    real(rk),intent(in)        ::  est_es(1) ! Changed to 1 instead of this%nt
                                                ! Energy...
    real(rk),intent(in)        ::  est_gs(this%sdgf)  
                                                ! Gradient...
    real(rk)                ::  prior_estimate(this%sdgf)    ! prior_estimate of the maximum
    integer                             ::  DoRotOrTrans=-1
                                            ! 0 do nothing
                                            ! 1 do rotation
                                            ! 2 do translation
    logical                             ::  rotationConverged
    logical                             ::  doTranslation
    real(rk)                            ::  direction(this%sdgf)
    real(rk)                            ::  oldEndPt(this%sdgf)
    real(rk)                            ::  tmpEndPt(this%sdgf)
    real(rk)                            ::  transForce(this%sdgf)
    real(rk)                            ::  gradMidPt(this%sdgf)
    integer                             ::  translationConverged
    integer                             ::  rotsCounter=0
    real(rk)                    ::  tmpHess(this%sdgf,this%sdgf)
    real(rk)    ::  eigval, rotVal, scaleFactor,&
                    steplength
    integer     ::  i
    logical     ::  TS_search_failed
    logical     ::  splitsConvergence
    real(rk)    :: dotP_dir_grad
    character(size(glob%gprmep_initPathName))      :: CharArrayToString
    prior_estimate(:)=est_xs(:)
    
  ! Queries for using the guess for paths as initial starting sets
  if (glob%useGeodesic.and.opt%stateNEBap==0) then
    if (glob%usePath) call dlf_fail("UseGeodesic and usePath are set!")
    ! Geodesic as start
    call gpr_geo_inst%geodesicFromMinima(glob%nat,glob%znuc,&
           glob%xcoords, glob%xcoords2,glob%nimage,.false.,1d-5)
    opt%nimage = glob%nimage
    opt%nat = glob%nat
    allocate(opt%ene(opt%nimage))
    allocate(opt%xcoords(3*opt%nat,opt%nimage))
    allocate(opt%dtau(opt%nimage+1))
    allocate(opt%dist(opt%nimage+1))
    opt%xcoords(:,:) = gpr_geo_inst%xCoords(:,:)
    glob%usePath = .true.
    glob%useGeodesic = .false.
  else if (glob%usePath.and.opt%stateNEBap==0) then
    ! using an initial path
    if (glob%gprmep_pathNameLength>0) then
      ! path file is given
      do i = 1, size(glob%gprmep_initPathName)
        CharArrayToString(i:i) = glob%gprmep_initPathName(i)
      end do
      call get_path_properties(CharArrayToString,opt%nat, opt%nimage)
      allocate(opt%ene(opt%nimage))
      allocate(opt%xcoords(3*opt%nat,opt%nimage))
      allocate(opt%dtau(opt%nimage+1))
      allocate(opt%dist(opt%nimage+1))
      call read_path_xyz(CharArrayToString,opt%nat,opt%nimage,glob%znuc,&
                         opt%xcoords)
    else
      ! no path file is given -> using nebpath.xyz
      if (printl>=2) &
        write(stdout,'("No pathfile given.")')
      if (printl>=2) &
        write(stdout,'("Using an NEB approximation to start TS search.")')
      call gpr_read_qts_coords_init(opt%nat,opt%nimage,opt%varperimage)
      allocate(opt%ene(opt%nimage))
      allocate(opt%xcoords(3*opt%nat,opt%nimage))
      allocate(opt%dtau(opt%nimage+1))
      allocate(opt%dist(opt%nimage+1))
      call gpr_read_qts_coords(opt%nat,opt%nimage,opt%varperimage,&
                               opt%temperature,opt%S_0, opt%S_pot, opt%S_ins,&
                               opt%ene,opt%xcoords,opt%dtau,opt%etunnel,opt%dist)
    end if
  end if  
  if (glob%usePath) then
    SELECT CASE(opt%stateNEBap)
      CASE(0) ! First point on MEP approx
        ! first guess
        if (this%sdgf/=3*opt%nat) call dlf_fail("sdgf not equal to 3*nat")
        opt%position=opt%nimage/2
        if (opt%position==0) then
          call dlf_fail("Not enough images in file.")
        end if
        nextPoint(:) = opt%xcoords(:,opt%position)
        opt%direction=0
        opt%limitStepExternally=.false.
        opt%stateNEBap=1
        if (printl>=4) &
            write(stdout,'("Initial position on the MEP approx path:", &
                I9, " of ", I9, " images")') opt%position,opt%nimage
        RETURN
      
      CASE(1) ! Have not found optimal point on MEP
              ! approximation path
        call GPR_add_tp(this, 1, est_xs, est_es, est_gs)
        call GPR_interpolation(this)
        dotP_dir_grad=dot_product(est_gs,&
                       (opt%xcoords(:,opt%position+1)&
                        -opt%xcoords(:,opt%position)))
        if(dotP_dir_grad>0) then
          ! gradient points in the same direction (angle pi/2 tolerance)
          ! as the vector to the next picture on the MEP approximation
          opt%position=opt%position+1
          if(opt%position<=opt%nimage) then
            if (printl>=4) &
              write(stdout,'("Next position on the MEP approx path   :", &
                I9, " of ", I9, " images")') opt%position, opt%nimage
            ! the direction is 0 (first time) or > 0 -> continue with search
            ! it is negative -> stop after including the next point
            if(opt%direction >= 0)then
              opt%direction=1
            else
              opt%stateNEBap=3 ! SHOULD BE 2 BUT 3 WORKS BETTER !CHANGED
            end if !(opt%direction >= 0)
          else
            opt%stateNEBap=3
          end if !(opt%position+1<=opt%nimage)
          
        else !(dotP_dir_grad>0)
          ! gradient points in the opposite direction (angle pi/2 tolerance)
          ! as the vector to the next picture on the MEP approximation
          opt%position=opt%position-1
          if(opt%position>0) then
            if (printl>=4) &
              write(stdout,'("Next position on the MEP approx path   :", &
                I9, " of ", I9, " images")') opt%position, opt%nimage
            
            ! the direction is 0 (first time) or < 0 -> continue with search
            ! it is positive -> stop after including the next point
            if(opt%direction <= 0)then
              opt%direction=-1
            else
              opt%stateNEBap=3 ! SHOULD BE 2 BUT 3 WORKS BETTER !CHANGED
            end if !(opt%direction <= 0)
          else 
            opt%stateNEBap=3
          end if !(opt%position-1>0)
        end if !(dotP_dir_grad>0)
        if(opt%stateNEBap==1.or.opt%stateNEBap==2) then
          nextPoint(:) = opt%xcoords(:,opt%position)
          ! for stateNEBap==2 next time directly go to TS search
          if (opt%stateNEBap==2) opt%stateNEBap=3 
          RETURN
        end if !(opt%stateNEBap==2)
        
      CASE(2)
        if (printl>=4) &
          write(stdout,'("Converging rotation.")')
        opt%stateNEBap=4
        deallocate(opt%ene)
        deallocate(opt%xcoords)
        deallocate(opt%dtau)
        deallocate(opt%dist)
        
      CASE(4)
        CONTINUE
        
      CASE DEFAULT
        call dlf_fail("This case should not exist. (opt%stateNEBap in gpr.f90)")
      END SELECT !opt%stateNEBap
      
      ! only stateNEBap==3 survives until here (all others "returned")
      if (opt%stateNEBap==3) then
        opt%initDimer=.false.
        if (printl>=4) &
          write(stdout,'("Constructing dimer for a first real TS search.")')
        call GPR_construct_dimer(this,opt%dimer)
        ! Do the first step      
        opt%rot_indices(1)%start=1
        opt%nTransSinceRot=0
        opt%nTransSteps=0
        do i = 1, this%sdgf
          direction(i)=1d0 ! this somehow works best
        end do
        opt%position = SUM(MAXLOC(this%es(1:this%nt)))
#ifdef writeBestGuessInGPRPP        
        !*************************
        if (printl>=4) then
          write(stdout,'(" ")')
          write(stdout,'(&
          "Writing the point of the MEP which is closest to the TS: ", &
          "Initpath_pt_closeTS.xyz")')
        end if
        open(unit=298, file = "Initpath_pt_closeTS.xyz", status="replace")
        call write_xyz(298,glob%nat,glob%znuc,this%xs(:,opt%position))
        close(298)
        if (printl>=4) &
          write(stdout,'(" ")')
        !*************************
#endif
        
        prior_estimate(:) = this%xs(:,opt%position)
        tmpEndPt(:) = prior_estimate(:)+direction(:)
        call GPR_place_dimer(this,opt%dimer,prior_estimate,&
                                    tmpEndPt)
        call GPR_newLengthOf_dimer(opt%dimer,opt%dimer_length)
        nextPoint=opt%dimer%endPt
        opt%nEndPts=1
        DoRotOrTrans=2
        opt%posIsResultOfRot=.false.
      end if !(stateNEBap==3)      
  end if !(glob%usePath)
    rotationConverged=.false.
    TS_search_failed=.false.
    splitsConvergence=.false.
    doTranslation=.false.
    opt%gConverged = (MAXVAL(abs(est_gs(:)))<opt%tol_g)
    opt%gradsq = dot_product(est_gs(:),est_gs(:))
    opt%oldgradsq = opt%gradsq
    ! Adding the new point to the GP surface has to be done in any case
    !call GPR_distance_nearest_tp(this,est_xs,distTP,.true.)
    call GPR_add_tp(this, 1, est_xs, est_es, est_gs)
    
    !*********************************************
    ! Optimizing hyperparameters with LBFGS in max likelihood method
 !    if (MOD(this%nt,1)==0) call LBFGS_Max_Like(this, 3, (/2,3,4/)) ! gamma + s_n for energies and grads
!     if (MOD(totalTPnumber(this),5)==0) call LBFGS_Max_Like(this, 1, (/2/)) ! gamma
    if(this%nt==7) call LBFGS_Max_Like(this,2,(/2,4/))   
    !**********************************************
    ! multilevel GPR system
    if (this%nt>opt%MaxnPointsinLevel.and.&
        (.not.opt%posIsResultOfRot)) then
      if (opt%nrot_indices==0) then
        ! No multi-level can be done
        if (printl>=2) &
          write(stdout,'("Warning: The number of steps in level 0 ",&
                "seems not enough to converge one rotation.")')
      else
        do i=1,opt%nrot_indices
          ! would the splitting of the data points across levels
          ! lead to a splitting of data points that belong to one rotational
          ! convergence?
          if (opt%rot_indices(i)%end>opt%nPointsToLowerLevel.and.&
              opt%rot_indices(i)%start<=opt%nPointsToLowerLevel) then
            splitsConvergence=.true.
            ! increase the number of points that are moved
            ! in the next level to allow multi-level for one of the
            ! next steps.
            opt%nPointsToLowerLevel=opt%nPointsToLowerLevel+1
          end if
          if (opt%rot_indices(i)%start>opt%nPointsToLowerLevel)&
            exit ! search can stop here
        end do
        if (.not.splitsConvergence) then
          ! now we can introduce a new level
          i = 1
          ! counting the number of rotational optimizations that are
          ! put into the next level
          rotsCounter=0
          do while (opt%rot_indices(i)%end<=&
                    opt%nPointsToLowerLevel.and.&
                    i<=opt%nrot_indices)
            i = i+1
            rotsCounter=rotsCounter+1
          end do
          ! reduce the number of considered rotations by that number
          opt%nrot_indices=opt%nrot_indices-rotsCounter
          ! shift to the rotations that are still in level 0
          opt%rot_indices=cshift(opt%rot_indices,rotsCounter)
          ! shift there indices to the values they now have in level 0
          opt%rot_indices(:)%end=opt%rot_indices(:)%end-&
                                opt%nPointsToLowerLevel
          opt%rot_indices(:)%start=opt%rot_indices(:)%start-&
                                opt%nPointsToLowerLevel
          ! introduce a new level with nPointsToLowerLevel points
          call GPR_newLevel(this,opt%nPointsToLowerLevel)
          ! set everyting back to original values
          opt%MaxnPointsinLevel=opt%MaxnPtsLevel_save
          opt%nPointsToLowerLevel=opt%nPtsinLowLevel_save
        else
          if (printl>=4) &
            write(stdout,'("A new level would split rotational convergence.",&
                " Postponing introduction of a new Level.")')
        end if
      end if
    end if
    !*********************************************
    ! Build the GP surface
    if (this%sdgf*this%nt>30000) then
      if (printl>=4) &
        write(stdout,'(&
        "To prevent the program from using too much memory a new level",&
        " is introduced here.")')
      call GPR_newLevel(this,this%nt/3)
    end if
    call GPR_interpolation(this)
    !**********************************************
    ! First step is always rotation, also if the last position given to
    ! the calling program was a result of a rotation
    ! then go in rotational procedure
    if (opt%initDimer) then
      opt%initDimer=.false.
      if (printl>=6) &
        write(stdout,'("Constructing dimer for first rotational step,")')
      call GPR_construct_dimer(this,opt%dimer)
      ! Do the first step
      opt%rot_indices(1)%start=1
      opt%firstRotation=.true.
      opt%nTransSinceRot=0
      opt%nTransSteps=0
      call random_seed()
      do i = 1, this%sdgf
        direction(i)=1d0 ! this somehow works best
      end do
      
      tmpEndPt(:) = prior_estimate(:)+direction(:)
      call GPR_place_dimer(this,opt%dimer,prior_estimate,&
                                    tmpEndPt)
      call GPR_newLengthOf_dimer(opt%dimer,opt%dimer_length)
      nextPoint=opt%dimer%endPt
      opt%nEndPts=1
      DoRotOrTrans=0
      opt%posIsResultOfRot=.true.
      !call GPR_add_tp(this, 1, nextPoint, est_es, est_gs)
      opt%nEndPts=2
      DoRotOrTrans=1
      !call GPR_interpolation(this)
    else if (opt%posIsResultOfRot) then ! Did a rotation as last step
      ! Continue Rotation procedure
      DoRotOrTrans=1
    else if (.not.opt%posIsResultOfRot) then ! Did a translation as last step
      ! Check whether another translation should be done
      ! for example via max number of translations -> set doTranslation
      ! Otherwise switch to rotation
      doTranslation = (opt%nTransSinceRot<opt%nTransBeforeRotOpt)
      !***********************************************
      if (doTranslation) then
        DoRotOrTrans=2
      else
        DoRotOrTrans=1
        opt%rot_indices(opt%nrot_indices+1)%start=this%nt
      end if
    end if
    !**************************************************
    ! DoRotOrTrans==1 means Rotation
    ! DoRotOrTrans==2 means translation
    ! DoRotOrTrans==0 means Do nothing
    if (DoRotOrTrans==1) then
      ! Start rotation
      ! Check rotational convergence -> set rotationConverged
#ifndef withopenmp
    call GPR_eval_hess(this,opt%dimer%midPt,tmpHess)
#endif
#ifdef withopenmp
    call GPR_eval_hess_parallel(this,opt%dimer%midPt,tmpHess)
#endif
      call eigVec_to_smallestEigVal(this%sdgf,tmpHess,direction,eigval)
      oldEndPt=opt%dimer%endPt
      call GPR_rotateToDirection_dimer(this,opt%dimer,direction)
      rotVal = abs(dot_product(opt%dimer%endPt-opt%dimer%midPt,&
                              oldEndPt-opt%dimer%midPt)/&
                              (opt%dimer%length)**2)
      ! Determine whether the rotation converged
      if (opt%firstRotation) then
        rotationConverged=(rotVal>1d0-opt%rotToleranceFirstStep)
      else
        rotationConverged=(rotVal>1d0-opt%rotTolerance)
      end if
      
      ! Only if the last step was a rotation the rotation can be
      ! considered to be "converged", otherwise it is just the 
      ! (possibly) bad guess on the GP-surface far away from any rotation
      ! optimizations.
      if (opt%posIsResultOfRot.and.rotationConverged) then
        opt%rot_indices(opt%nrot_indices+1)%end=this%nt
        opt%nrot_indices=opt%nrot_indices+1
        if (printl>=4) &
            write(stdout,'("Eigenmode Converged! Starting translation.")')
        opt%nEndPts=0
        opt%firstRotation=.false.
        opt%nTransSinceRot=0
        DoRotOrTrans=2
      else
        ! Nothing to do anymore, need new data
        if (printl>=4) &
          write(stdout,'( "Eigenmode not converged yet.",&
                " Need new gradient.")')
        if (printl>=6) &
        write(stdout,'("rotation value (cos):",ES11.4)') rotVal
        nextPoint=opt%dimer%endPt
        if(opt%nTransSteps>=1.and.&
         dot_product(direction,opt%dimer%midPt-opt%lastMidPt)<0d0) then
          direction(:)=-direction(:)
          call GPR_rotateToDirection_dimer(this,opt%dimer,direction)
          nextPoint=opt%dimer%endPt
        end if
        DoRotOrTrans=0
        opt%posIsResultOfRot=.true.
        opt%nEndPts=opt%nEndPts+1
      end if
    end if
    if (DoRotOrTrans==2) then

    if (printl>=6) &
        write(stdout,'("Doing translational step.")')
    if (printl>=4) &
        write(stdout,'("Searching for TS on GP via P-RFO")')
        
    call GPR_find_Saddle(this, opt,0, opt%dimer%midPt,&
                nextPoint, translationConverged, .true.)
    if (translationConverged==-1) call dlf_fail("ERROR IN GPR_find_Saddle")
    if (translationConverged==3) then          
      if (printl>=4) then
        write(stdout,'("P-RFO did not find a TS on the GP surface!")')
        write(stdout,'("Use simple dimer translation on GP surface.")')
      end if
      call GPR_find_Saddle(this, opt,2, opt%dimer%midPt,&
                nextPoint,translationConverged,.true.)
    end if
        do i = 1, this%sdgf
          if (isnan(nextPoint(i))) then!.or.(.not.translationConverged)) then
            if (printl>=4) &
                write(stdout,'(&
                "TS search on GP failed. Using dimer translation on GP.")')
            call GPR_Optimizer_TS_dimerDirect(this,opt,gradMidPt,&
                tmpHess,direction,eigval,transForce,.not.rotationConverged)
            steplength = dsqrt(dot_product(direction,direction))
            direction  = direction / steplength &
                         * MIN(opt%maxstep/2d0,steplength)
            nextPoint=opt%dimer%midPt+direction
            TS_search_failed=.true.
            exit
          end if
        end do
        
        if(.not.TS_search_failed) then
          if (printl>=4) &
            write(stdout,'(&
                "Saddle point search on GPR converged.")')
          direction(:) = nextPoint-opt%dimer%midPt
        end if

        ! Scales up direction by calculated "scaleFactor" but
        ! does not change "nextPoint" -> only direction is the
        ! valid description of the next point after this
        if (opt%nTransSteps>=1) then
          call GPR_opt_step_overshooting(&
                this,opt,direction,nextPoint,scaleFactor)
        end if

      ! Limit to max step length
      steplength = dsqrt(dot_product(direction,direction))
      if (steplength > opt%maxstep) then
        direction(:) = direction(:)/steplength*opt%maxstep
        steplength = dsqrt(dot_product(direction,direction))
      end if
      nextPoint=opt%dimer%midPt+direction
      opt%nTransSinceRot=opt%nTransSinceRot+1
      opt%posIsResultOfRot=.false.
      opt%nTransSteps=opt%nTransSteps+1

      nextPoint=opt%dimer%midPt+direction
      opt%lastMidPt=opt%dimer%midPt
      call GPR_move_dimer_to(this,opt%dimer,nextPoint)   
    end if
    if(DoRotOrTrans==-1) call dlf_fail("Error in GPR_Optimizer_step_TS!")
    if (printl>=4) &
        write(stdout,'("Nr of translational steps:", I8)') opt%nTransSteps
    if (glob%usePath.and.opt%stateNEBap==3) then
      opt%stateNEBap=4
      call GPR_destroy_dimer(opt%dimer)
      opt%initDimer=.true.
      if (printl>=4) &
        write(stdout,'(&
            "Finished initialisation via NEB approx.",& 
            "Switching to normal TS search.")')
      deallocate(opt%ene)
      deallocate(opt%xcoords)
      deallocate(opt%dtau)
      deallocate(opt%dist)
    else
      opt%limitStepExternally=.true.
    end if
end subroutine GPR_Optimizer_step_TS     
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Optimizer_TS_dimerDirect
!!
!! FUNCTION
!! Find the direction in which a dimer should be translated
!! when using standard dimer translation.
!!
!! SYNOPSIS
subroutine GPR_Optimizer_TS_dimerDirect(this,opt,gradMidPt,&
        tmpHess,direction,eigval,transForce,doRot)
!! SOURCE        
  type(gpr_type), intent(inout)         ::  this
  type(optimizer_type),intent(inout)    ::  opt
  real(rk), intent(inout)               ::  gradMidPt(this%sdgf)
  real(rk), intent(inout)               ::  tmpHess(this%sdgf,this%sdgf)
  real(rk), intent(inout)               ::  direction(this%sdgf)
  real(rk), intent(inout)               ::  eigval
  real(rk), intent(inout)               ::  transForce(this%sdgf)
  logical, intent(in)                   ::  doRot
  call GPR_eval_grad(this,opt%dimer%midPt, gradMidPt)
  if (doRot) then
    call GPR_eval_hess(this,opt%dimer%midPt,tmpHess)
    call eigVec_to_smallestEigVal(this%sdgf,tmpHess,direction,eigval)
    call GPR_rotateToDirection_dimer(this,opt%dimer,direction)
  end if
  transForce = -gradMidPt + 2d0*&
    dot_product(gradMidPt,&
    (opt%dimer%endPt-opt%dimer%midPt)/opt%dimer%length)*&
    ((opt%dimer%endPt-opt%dimer%midPt)/opt%dimer%length)
  direction  = transForce*opt%dimerTransDamping  
end subroutine GPR_Optimizer_TS_dimerDirect
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_Find_Extreme
!!
!! FUNCTION
!! Finds an extremum on a GP. This can be a maximum or a minimum, depending
!! on the specifications of the optimizer_type
!! The search for the extremum starts at the "guess".
!! Ideally, it should find the nearest minumum/maximum
!!
!! SYNOPSIS
subroutine GPR_Find_Extreme(this, opt, guess, extremum,extremum_internal,gpr_hdlc)
!! SOURCE
    use oop_lbfgs
    use oop_hdlc
    !use dlf_hdlc_interface
    implicit none
    type(gpr_type), intent(inout)   ::  this
    real(rk), intent(in)            ::  guess(this%idgf)
    type(optimizer_type),intent(inout) ::  opt
    real(rk), intent(out)           ::  extremum(this%idgf)
    real(rk), intent(out),optional   ::  extremum_internal(this%sdgf)
    type(hdlc_ctrl),intent(inout),optional  :: gpr_hdlc
    real(rk)                        ::  grad(this%idgf), oldgrad(this%idgf)
    real(rk)                        ::  step(this%idgf)
    real(rk)                        ::  gradlength
    real(rk)                        ::  stepl
    integer                         ::  member !nr of steps to remember by lbfgs
    integer                         ::  nrsteps,i
    real(rk)                        ::  tmp    
    real(rk)                        ::  bmatrix(this%sdgf,this%idgf)
    type(oop_lbfgs_type)            ::  lbfgs_inst
    real(rk)                        :: xgradient(this%idgf),igradient(this%sdgf)
    real(rk)                        :: var

    if(opt%opt_type_int==0) then
        call dlf_fail("Do not call GPR_Find_Extreme for TS-optimization!")
    end if
    extremum(:)=guess(:)
    if(this%internal ==2) then
      member = MAX(this%idgf/2,2)
    else
      member = MAX(this%sdgf/2,2)
    endif
!     call gpr_lbfgs_init(this%sdgf, member, .false.)
    call lbfgs_inst%init(this%idgf, member, .false.)
    
    oldgrad = -4d0
    grad = 4d0
    stepl=opt%tolOnGPR*2
    gradlength=opt%tolOnGPR*2
    nrsteps=1
    if(this%internal == 2) then
      xgradient(:) = 0.0d0
      igradient(:) = 0.0d0      
    endif
    ! gradlength is needed for the search in very bold step
    do while (stepl>opt%tolOnGPR .AND. gradlength>opt%tolOnGPR .and.(nrsteps<1000)) ! if the nrsteps limit is not used, that can lead to infinite loops
        nrsteps=nrsteps+1
        oldgrad=grad
        bmatrix(:,:)=0.0d0
        if (this%internal == 2) then            
         call gpr_hdlc%dlf_hdlc_xtoi(glob%nvar/3,glob%nivar,glob%nicore,glob%micspec, &
           extremum,xgradient,extremum_internal,igradient)
          bmatrix(:,:) = gpr_hdlc%bhdlc_matrix(:,:)
          call GPR_eval_grad(this, extremum_internal, grad,bmatrix) 
          call GPR_eval(this,extremum_internal,tmp)  
        else 
          call GPR_eval_grad(this, extremum, grad)          
          call GPR_eval(this,extremum,tmp)
        endif 
        grad(:) = opt%opt_type_int * grad(:)
        gradlength=dsqrt(dot_product(grad,grad))
!         call gpr_lbfgs_step(extremum, grad, step)
        call lbfgs_inst%next_step(extremum, grad, step)
        if(this%internal ==2 ) then
          call gpr_hdlc%dlf_hdlc_xtoi(glob%nvar/3,glob%nivar,glob%nicore,glob%micspec, &
           extremum(:)+step(:),xgradient,extremum_internal,igradient)        
          call GPR_variance(this,extremum_internal,var)         
        else
          call GPR_variance(this,extremum(:)+step(:),var)
        endif
        if (var <0d0) var = 0d0
        var =dsqrt(var)
        stepl=dsqrt(dot_product(step,step))
        if(var>opt%tol_variance) then
          if (stepl>opt%maxStepOnGPR) step(:)=step(:)/stepl*opt%maxStepOnGPR
        endif
        extremum(:) = extremum(:) + step(:)

        if (ISNAN(extremum(1))) call dlf_fail(&
        "GPR extremum search failed. Consider higher noise/regularization?")
      
        !if(printl>=4) &
        !  write(stdout,'("Steplength and Gradlength",ES11.4,ES11.4)')&
        !    stepl,gradlength
    end do
        if(printl>=6) &
          write(stdout,'("O! converged on GPR")')                
    call lbfgs_inst%destroy()
    if(printl>=6) then
      print *,'nr steps, stpl, gradlength', nrsteps,stepl,gradlength
    endif
end subroutine GPR_Find_Extreme 
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_initProjectOutTrans
!!
!! FUNCTION
!! Projecting out the translational modes of the Hessian
!!
!! SYNOPSIS
subroutine GPR_initProjectOutTrans(this,opt)
!! SOURCE
    type(gpr_type), intent(in)            ::  this
    type(optimizer_type),intent(inout)    ::  opt
    integer                               ::  i,j,k
    real(rk)                              ::  DNRM2
    real(rk)                              ::  tmpMat(this%sdgf,this%sdgf)
  opt%transModes(:,:)=0d0
  do i=1,this%sdgf,3
    opt%transModes(i,1)=1d0
    opt%transModes(i+1,2)=1d0
    opt%transModes(i+2,3)=1d0
  end do
  opt%transModes(:,1)=opt%transModes(:,1)/&
                        DNRM2(this%sdgf,opt%transModes(:,1),1) 
  opt%transModes(:,2)=opt%transModes(:,2)/&
                        DNRM2(this%sdgf,opt%transModes(:,2),1)
  opt%transModes(:,3)=opt%transModes(:,3)/&
                        DNRM2(this%sdgf,opt%transModes(:,3),1)

  do k = 1, 3
    tmpMat(:,:)=0d0
    do i =1, this%sdgf
      tmpMat(i,i)=1d0
    end do
    do i = 1,this%sdgf
      do j = 1,this%sdgf
        tmpMat(i,j)=tmpMat(i,j)-&
            opt%transModes(i,k)*opt%transModes(j,k)
      end do
    end do
    if (k==1) then
      opt%projTransMat(:,:)=tmpMat(:,:)
    else 
      opt%tmpProjTransMat = opt%projTransMat
      call dlf_matrix_matrix_mult(this%sdgf,opt%tmpProjTransMat,'N',&
            tmpMat,'N',opt%projTransMat)
    end if
  end do
end subroutine GPR_initProjectOutTrans
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/angle_along
!!
!! FUNCTION
!! gives the angle between the vector x1->x2 and x2->x3
!! in units of pi
!!
!! SYNOPSIS
real(rk) function angle_along(d,x1,x2,x3,pts_are_theSame)
!! SOURCE
    implicit none
    integer, intent(in)     ::  d !dimension of the vectors
    real(rk), intent(in)    ::  x1(d),x2(d),x3(d)
    logical, intent(out), optional :: pts_are_theSame
    real(rk)                ::  x3minusx2(d), x2minusx1(d),&
                                x3mx2_norm, x2mx1_norm
    if (present(pts_are_theSame)) &
      pts_are_theSame = .false.
    x3minusx2(:) = x3(:)-x2(:)
    x2minusx1(:) = x2(:)-x1(:)
    x3mx2_norm = NORM2(x3minusx2)
    x2mx1_norm = NORM2(x2minusx1)
    if (x3mx2_norm<=0d0.or.x2mx1_norm<=0d0) then
      if (present(pts_are_theSame))then
        pts_are_theSame = .true.
        return
      else
        call dlf_fail("The points are too close together (angle_along).")
      end if
    end if
    angle_along=dot_product(x3minusx2,x2minusx1)/&
                        (NORM2(x3minusx2)*NORM2(x2minusx1))
end function angle_along
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_projectOutTrans
!!
!! FUNCTION
!! Projecting out the translational modes of the Hessian
!!
!! SYNOPSIS
subroutine GPR_projectOutTrans(this,opt,Hess)
!! SOURCE
    type(gpr_type), intent(in)            ::  this
    type(optimizer_type),intent(inout)    ::  opt
    real(rk),intent(inout)                ::  Hess(this%sdgf,this%sdgf)
    call dlf_matrix_matrix_mult(this%sdgf,Hess,'N',&
            opt%projTransMat,'N',opt%tmpProjTransMat)
    call dlf_matrix_matrix_mult(this%sdgf,opt%projTransMat,'T',&
            opt%tmpProjTransMat,'N',Hess)    
end subroutine GPR_projectOutTrans
!!****

!!****************************************************************************
!! Dimer methods
!!****************************************************************************

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_construct_dimer
!!
!! FUNCTION
!! This function constructs a dimer on the GP surface
!! Before calling this routine you must construct a GPR (even if it has
!! no trainingpoints in it)
!!
!! SYNOPSIS
subroutine GPR_construct_dimer(this,dimer)
!! SOURCE
  type(gpr_type), intent(in)            ::  this
  type(gpr_dimer_type), intent(inout)   ::  dimer
  if(.not.this%constructed) &
    call dlf_fail("Construct the GPR before you construct a dimertype.")
  if(dimer%constructed) &
    call dlf_fail("Destroy the dimertype before constructing a new one")
  if (.not.allocated(dimer%midPt)) allocate(dimer%midPt(this%sdgf))
  if (.not.allocated(dimer%endPt)) allocate(dimer%endPt(this%sdgf))
  dimer%length=0d0  
  dimer%constructed=.true.    
end subroutine GPR_construct_dimer
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_place_dimer
!!
!! FUNCTION
!! This method places the dimer at a specific position (midPoint) in
!! a specific direction (endPoint)
!!
!! SYNOPSIS
subroutine GPR_place_dimer(this,dimer,midPoint,endPoint)
!! SOURCE
  type(gpr_type), intent(in)            ::  this
  type(gpr_dimer_type), intent(inout)   ::  dimer
  real(rk), intent(in)                  ::  midPoint(this%sdgf)
  real(rk), intent(in)                  ::  endPoint(this%sdgf)
  if(.not.dimer%constructed) &
    call dlf_fail("Create the dimertype before trying to place one!")
   dimer%midPt(:) = midPoint(:)
   dimer%endPt(:) = endPoint(:)
   dimer%length = dsqrt(dot_product(endPoint(:)-midPoint(:),&
                                        endPoint(:)-midPoint(:)))
end subroutine GPR_place_dimer
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_destroy_dimer
!!
!! FUNCTION
!! Destroy a dimer, delete the allocated memory
!!
!! SYNOPSIS
subroutine GPR_destroy_dimer(dimer)
!! SOURCE
  type(gpr_dimer_type), intent(inout)   ::  dimer
  if(.not.dimer%constructed) &
    call dlf_fail("Create the dimertype before trying to destroy one!")
  if(allocated(dimer%midPt)) deallocate(dimer%midPt)
  if(allocated(dimer%endPt)) deallocate(dimer%endPt)
  dimer%length=0d0  
  dimer%constructed=.false.    
end subroutine GPR_destroy_dimer
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_move_dimer
!!
!! FUNCTION
!! Move a dimer by a certain displacement
!!
!! SYNOPSIS
subroutine GPR_move_dimer(this, dimer, displacement)
!! SOURCE
  type(gpr_type)        ::  this
  type(gpr_dimer_type)  ::  dimer
  real(rk), intent(in)  ::  displacement(this%sdgf)
  dimer%endPt(:) =  dimer%endPt(:)+displacement(:)
  dimer%midPt(:) =  dimer%midPt(:)+displacement(:)
end subroutine GPR_move_dimer
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_move_dimer_to
!!
!! FUNCTION
!! Move a dimer to a specific position
!!
!! SYNOPSIS
subroutine GPR_move_dimer_to(this, dimer, newPos)
!! SOURCE
  type(gpr_type)        ::  this
  type(gpr_dimer_type)  ::  dimer
  real(rk), intent(in)  ::  newPos(this%sdgf)
  dimer%endPt(:) =  (dimer%endPt(:)-dimer%midPt(:))+newPos(:)
  dimer%midPt(:) =  newPos(:)
end subroutine GPR_move_dimer_to
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_newLengthOf_dimer
!!
!! FUNCTION
!! Change the length of the dimer
!!
!! SYNOPSIS
subroutine GPR_newLengthOf_dimer(dimer, new_length)
!! SOURCE
  type(gpr_dimer_type)  ::  dimer
  real(rk), intent(in)  ::  new_length
  dimer%endPt(:) =  dimer%midPt(:) + &
                    (dimer%endPt(:)-dimer%midPt(:))/&
                    dimer%length*new_length
  dimer%length   =  new_length
end subroutine GPR_newLengthOf_dimer
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_rotateToDirection_dimer
!!
!! FUNCTION
!! Rotating the endpoint in a certain direction from the midpoint
!!
!! SYNOPSIS
subroutine GPR_rotateToDirection_dimer(this, dimer, direction)
!! SOURCE
  type(gpr_type)        ::  this
  type(gpr_dimer_type)  ::  dimer
  real(rk), intent(in)  ::  direction(this%sdgf)
  dimer%endPt(:) =  dimer%midPt(:) + &
                    direction(:)/dsqrt(sum(direction**2))*(dimer%length)
end subroutine GPR_rotateToDirection_dimer
!!****
end module gpr_opt_module
