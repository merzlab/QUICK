! **********************************************************************
! **                                                                  **
! **                      DL-FIND main module                         **
! **                                                                  **
! **                    Johannes Kaestner 2006                        **
! **                                                                  **
! **                                                                  **
! **********************************************************************
!!****J* DL-FIND/main
!!
!! NAME
!! DL-FIND
!!
!! FUNCTION
!! Main unit of the optimiser DL-FIND
!!
!!
!! DATA
!! $Date$
!! $Revision$
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
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!! Main layout:
!! external program calls dl_find
!!
!! dl-find
!!   |
!!   dlf_read_in
!!   |
!!   dlf_task (calls dlf_run once or several times)
!!       |
!!       dlf_run 
!!       (restart is at the moment handeled there. This does not work really IMPROVE)
!!          |
!!          dlf_init
!!          main optimisation cycle
!!   |<--   dlf_destroy
!!   |
!!   shut down
!!   |
!! return to calling program
!!
subroutine dl_find (ierr2, master &
#ifdef GAMESS
    ,core&
#endif
    )
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout
  use dlf_stat, only: stat
  use dlf_allocate, only: allocate_report,allocate,deallocate
  use dlf_store, only: store_delete_all
  use quick_molspec_module, only: natom, quick_molspec

  implicit none
  integer      :: nvarin ! number of variables to read in
                         !  3*nat
  integer      :: nvarin2! number of variables to read in
                         !  in the second array (coords2)
  integer      :: nspec  ! number of values in the integer
                                     !  array spec
  integer   ,intent(inout) :: ierr2
  integer   ,intent(in)   :: master ! 1 if this task is the master of
                                     ! a parallel run, 0 otherwise
#ifdef GAMESS
  real(rk) :: core(*) ! GAMESS memory, not used in DL-FIND
#endif
  ! 
! **********************************************************************

  ! Flag for dlf_fail - needs to be set before anything else
  glob%cleanup = 0

  nspec = 3*quick_molspec%natom
  nvarin = 3*quick_molspec%natom
  nvarin2 = quick_molspec%natom

  ! read input parameters, set defaults
  call dlf_read_in(nvarin,nvarin2,nspec,master)

  ! task manager, main optimisation cycle
  call dlf_task(ierr2 &
#ifdef GAMESS
    ,core&
#endif
    )
  ! shut down finally
  call dlf_deallocate_glob()

  ! deallocate arrays in formstep_set_tsmode
  call dlf_formstep_set_tsmode(1,-2,1.d0)

  ! delete dlf_store
  call store_delete_all

  call clock_stop("TOTAL")
!  call time_report

!  call allocate_report

end subroutine dl_find

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_read_in
!!
!! FUNCTION
!!
!! * Init clock
!! * Get input parameters from calling code
!! * set defaults
!! * call dlf_allocate_glob to allocate arrays globally to the optimiser
!!
!! SYNOPSIS
subroutine dlf_read_in(nvarin,nvarin2,nspec,master)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,pi,stdout,printl,printf
  use dlf_stat, only: stat
  use dlf_allocate, only: allocate,deallocate
  use dlf_store, only: store_initialise
  use dlf_constants, only: dlf_constants_init,dlf_constants_get
  implicit none
  integer   ,intent(in)    :: nvarin ! number of variables to read in
                                     !  3*nat
  integer   ,intent(in)    :: nvarin2! number of variables to read in
                                     !  in the second array (coords2)
  integer   ,intent(in)    :: nspec  ! number of values in the integer
                                     !  array spec
  integer   ,intent(in)    :: master ! 1 if this task is the master of
                                     ! a parallel run, 0 otherwise
  real(rk),allocatable :: tmpcoords(:),tmpcoords2(:)
  integer, allocatable :: spec(:)
  integer              :: ivar,nat,nframe,nmass,nweight,nz,tsrel,iat, jat
  integer              :: massweight,ierr
  real(rk)             :: svar
  integer              :: tdlf_farm
  integer              :: n_po_scaling
  integer              :: coupled_states
  integer              :: micro_esp_fit
! **********************************************************************



  pi=4.D0*datan(1.D0)

  call time_init
  call dlf_constants_init
    !call dlf_constants_get("NONE",svar) ! to cause printing of output
  call clock_start("TOTAL")

  if(nvarin<=0) call dlf_fail("A positive number of variables is needed")

  call allocate(tmpcoords,nvarin)
  call allocate(tmpcoords2,max(nvarin2,1))
  call allocate(spec,max(nspec,1))

  ! ====================================================================
  ! READ EXTERNAL PARAMETERS, SET DEFAULTS
  ! ====================================================================

  ! set parameters to something useless to recognise user input
  call dlf_default_init(nspec,spec)
  nz=0
  nweight=0

  ! ====================================================================
  ! The input arrays and their meaning:
  ! tmpcoords(nvarin) : one set of coordinates (main point)
  ! tmpcoords2(nvarin2) : a multi-purpose real array
  !    nframe*nat*3    entries of coordinates of nframe structures
  !    nweight         entries of weights (nat or 0)
  !    nmass           entries of atomic masses (nat or 0)
  !    n_po_scaling    entries of radii scaling factors in the parallel 
  !                    optimization (0 [meaning all radii set to the base 
  !                    value], or a pre-known nivar)
  !    i.e. nvarin2= nframe*nat*3 + nweight + nmass + n_po_scaling
  ! spec(nspec) : integer specification array
  !    nat     entries of freezing/residue number
  !    nz      entries of nuclear charges (same order as coords)
  !    5*ncons entries of constraints (typ, atom1,atom2, atom3, atom4)
  !    2*nconn entries of connections (atom1 atom2)
  !    nat     entries of microiterative region specification
  !    i.e. nspec= nat + nz + 5*ncons + 2*nconn + nat
  ! ====================================================================
  ! get input parameters
  ivar=1
  massweight=0
  tdlf_farm=1 ! set default value
  n_po_scaling=0 ! set default value
  coupled_states=1 ! set default value
  micro_esp_fit=0 ! set default value
  call dlf_get_params(nvarin,max(nvarin2,1),max(nspec,1), &
      tmpcoords,tmpcoords2,spec, ierr, &
      glob%tolerance,printl,glob%maxcycle,glob%maxene,&
      ivar,glob%icoord,glob%iopt,glob%iline,glob%maxstep, &
      glob%scalestep,glob%lbfgs_mem,glob%nimage,glob%nebk, &
      glob%dump,glob%restart,nz,glob%ncons,glob%nconn,&
      glob%update,glob%maxupd,glob%delta,glob%soft,glob%inithessian, &
      glob%carthessian,tsrel,glob%maxrot,glob%tolrot,nframe,nmass,nweight,&
      glob%timestep,glob%fric0,glob%fricfac,glob%fricp, &
      glob%imultistate, glob%state_i, glob%state_j, &
      glob%pf_c1, glob%pf_c2, glob%gp_c3, glob%gp_c4, glob%ln_t1, glob%ln_t2, &
      printf, glob%tolerance_e, glob%distort, massweight, glob%minstep, &
      glob%maxdump, glob%task, glob%temperature, &
      glob%po_pop_size, glob%po_radius_base, glob%po_contraction, &
      glob%po_tol_r_base, glob%po_tolerance_g, glob%po_distribution, &
      glob%po_maxcycle, glob%po_init_pop_size, glob%po_reset, &
      glob%po_mutation_rate, glob%po_death_rate, glob%po_scalefac, &
      glob%po_nsave,glob%ntasks,tdlf_farm,n_po_scaling, &
      glob%neb_climb_test, glob%neb_freeze_test, glob%nzero, &
      coupled_states, glob%qtsflag, &
      glob%imicroiter, glob%maxmicrocycle, micro_esp_fit)
  if(ierr/=0) call dlf_fail("Failed to read parameters")

  if (glob%ntasks <= 0) then
    write(stdout,'("glob%ntasks = ",i6)') glob%ntasks
    call dlf_fail("Number of task farms must be positive")
  end if

  ! call this subroutine even if glob%ntasks == 1 to sort out the 
  ! writing to files from each processor and the communicators
  call dlf_make_taskfarm(tdlf_farm)

  ! get logical variables (communication with c-code requires integers)
  glob%tatoms=(ivar==1)
  glob%tsrelative=(tsrel==1)
  glob%massweight=(massweight==1)
  glob%micro_esp_fit = (micro_esp_fit == 1)

  ! Do we need to calculate the interstate coupling gradient?
  ! If coupled_states is false, coupling = zero.
  if (glob%imultistate > 1 .and. coupled_states == 1) glob%needcoupling = 1

  ! set parameters that have not been set by the user to the default
  ! this routine defines the default values
  call dlf_default_set(nvarin)

  ! check consistency of the array sizes:
  nat=nvarin/3
  if(nspec/=nat+nz+5*glob%ncons+2*glob%nconn+nat) then
    write(stdout,'("nspec ",i6)') nspec
    write(stdout,'("nat   ",i6)') nat
    write(stdout,'("nz    ",i6)') nz
    write(stdout,'("ncons ",i6)') glob%ncons
    write(stdout,'("nconn ",i6)') glob%nconn
    write(stdout,'("nspec should be: nat + nz + 5*ncons + 2*nconn + nat")')
    call dlf_fail("Inconsistent size of array spec - interface error")
  end if

  if(glob%tatoms.and.(nvarin2/=nframe*nat*3+nweight+nmass &
      .or. (nweight/=0.and.nweight/=nat) &
      .or. (nmass/=0.and.nmass/=nat) &
      .or. (n_po_scaling < 0) ) ) then
    write(stdout,'("nvarin2      ",i6)') nvarin2
    write(stdout,'("nframe       ",i6)') nframe
    write(stdout,'("nat          ",i6)') nat
    write(stdout,'("nweight      ",i6)') nweight
    write(stdout,'("nmass        ",i6)') nmass
    write(stdout,'("n_po_scaling ",i6)') n_po_scaling
    write(stdout,'("varin2 should be: nframe*nat*3 + nweight + nmass + n_po_scaling")')
    write(stdout,'("nweight should be either 0 or nat")')
    write(stdout,'("nmass should be either 0 or nat")')
    call dlf_fail("Inconsistent size of array coords2 - interface error")
  end if

  ! check consistency of multistate calculations
  if (glob%imultistate > 0) call dlf_conint_check_consistency

  ! initialise printout
  if(master==0) then
    ! this is a slave, do not print anything!
    printl=-2
    printf=-2
  end if

  if(printl>=2) call dlf_printheader

  ! ====================================================================
  ! INITIALISE VARIOUS INSTANCES
  ! ====================================================================

  ! allocate storage
  call dlf_allocate_glob(nvarin,nvarin2,nspec,tmpcoords,tmpcoords2,spec,&
      nz,nframe,nmass,nweight,n_po_scaling)
  call deallocate(tmpcoords)
  call deallocate(tmpcoords2)
  call deallocate(spec)

  ! initialise (reset) all statistics counters
  stat%sene=0
  stat%pene=0
  call dlf_stat_reset

  ! initialise dlf_store
  call store_initialise

end subroutine dlf_read_in
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_run
!!
!! FUNCTION
!!
!! * Initialise coordinates, formstep, linesearch
!! * do main optimisation cycle
!! * Destroy coordinates, formstep, linesearch
!!
!! SYNOPSIS
subroutine dlf_run(ierr2 &
#ifdef GAMESS
    ,core&
#endif
    )
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl,printf
  use dlf_stat, only: stat
  use dlf_allocate, only: allocate,deallocate
  use quick_molspec_module, only: xyz, quick_molspec
  use quick_method_module,only: quick_method
  implicit none
#ifdef GAMESS
  real(rk) :: core(*) ! GAMESS memory, not used in DL-FIND
#endif
  integer  :: iimage,ivar,status,image_status,taskfarm_mode
  integer  :: icount, kiter, iat, jat
  real(rk) :: svar
  logical  :: tconv,trestarted,trerun_energy
  logical  :: needhessian ! do we need a Hessian?
  logical  :: testconv ! is convergence checked in coords_xtoi ?
  logical  :: trestarted_report,noeandg,fd_hess_running
  logical  :: tswitch
  integer   ,intent(inout) :: ierr2
! **********************************************************************
  
  ! classical rate (i.e. rate w/o tunneling) should be calculated
  if(glob%iopt == 13) then
    call dlf_htst_rate()
    return ! nothing else to do
  end if

  iimage=1
  glob%toldenergy_conv=.false.

  ! if thermal analysis (hessian only) is called, make sure that mass-weighted
  ! cartesians are used!
  if(glob%iopt==11) then
    glob%massweight=.true.
    glob%icoord=0
    glob%distort=0.D0
  end if

  ! if qTST thermal analysis (hessian only) is called, make sure that mass-weighted
  ! cartesians are used!
  if(glob%iopt==12) then
    glob%massweight=.true.
    glob%icoord=190
    glob%distort=0.D0
  end if

  ! initialise coordinate transform, allocate memory for it
  call dlf_coords_init
  
  ! initialise search algorithm
  call dlf_formstep_init(needhessian)

  ! initialise line search
  call linesearch_init

  ! prepare for main optimisation cycle

  glob%taccepted=.true.
  tconv=.false.
  trerun_energy=.false.
  glob%dotask=.true.
  status=0

  ! define, if Energy and gradient calculations are to be performed
  noeandg=.false.
  if(glob%inithessian==5) noeandg=.true.
  if(glob%iopt==13) noeandg=.true.

  ! read checkpoint file 
  if(glob%restart==1) then
    call clock_start("CHECKPOINT")
    call dlf_checkpoint_read(status,trestarted)
    call clock_stop("CHECKPOINT")
    if(.not.trestarted) call dlf_fail("Restart attempt failed")
    trestarted_report=.true.
    ! Parallel NEB - assume that image restart info for external
    ! program is also available
    glob%serial_cycle = 0 
  else
    trestarted=.false.
    trestarted_report=.false.
    ! Parallel NEB - each workgroup should calculate all images
    ! on the first cycle, so that each image calc restarts from the
    ! result of the last (as would normally happen in serial)
    ! For some external programs it may not be important to do this, 
    ! so this could be made user-defined.
    glob%serial_cycle = 1
    if (glob%icoord==190) glob%serial_cycle = 0
  end if

  ! report at the start
  call dlf_report(trestarted_report)
  

  ! open trajectory file
  if(glob%tatoms.or.glob%nimage.gt.1) then
    call clock_start("XYZ")
    if(printf>=3) then
      if(stat%sene>0) THEN
        if (glob%iam == 0) then
!          open(unit=30,file="path.xyz",position="APPEND")
!          open(unit=40,file="new_coords.xyz",position="APPEND")
!          open(unit=31,file="path_active.xyz",position="APPEND")
          ! Only open for TS search methods
          if (glob%iopt == 10 .or. (glob%icoord >= 100 .and. glob%icoord < 300)) then
             open(unit=32,file="path_tsmode.xyz",position="APPEND")
          end if
          ! Only for microiterative optimisations
          if (glob%imicroiter > 0) then
             open(unit=33,file="path_micro.xyz",position="APPEND")
          end if
        end if
      ELSE
        if (glob%iam == 0) then
!          open(unit=30,file="path.xyz")
!          open(unit=40,file="new_coords.xyz")
!          open(unit=31,file="path_active.xyz")
          ! Only open for TS search methods
          if (glob%iopt == 10 .or. (glob%icoord >= 100 .and. glob%icoord < 300)) then
             open(unit=32,file="path_tsmode.xyz")
          end if
          ! Only for microiterative optimisations
          if (glob%imicroiter > 0) then
             open(unit=33,file="path_micro.xyz")
          end if
        end if
      end if
    end if
    if(printf>=4.and.glob%iam == 0) then
      if(stat%sene>0) THEN
!        open(unit=300,file="path_force.xyz",position="APPEND")
        !open(unit=301,file="paths.xyz",position="APPEND")
      ELSE
!        open(unit=300,file="path_force.xyz")
        !open(unit=301,file="paths.xyz")
      end if
    end if
    call clock_stop("XYZ")
  else
    if (glob%iam == 0) then
       open(unit=30,file="path.inc")
       !open(unit=30,file="path.inc",position="append")
    end if
  end if

  ! ====================================================================
  ! MAIN OPTIMISATION CYCLE
  ! ====================================================================
  do WHILE (stat%ccycle.lt.quick_method%iopt)! exit conditions implemented via exit statements

    if (glob%iopt/10 == 5) then ! parallel optimisation
       call dlf_parallel_opt(trestarted_report, tconv &
#ifdef GAMESS
       ,core&
#endif
       )
       exit ! the MAIN OPTIMISATION CYCLE
    end if

    if(trestarted) then
       ! Task-farming: (at present) only workgroup 0 saves a checkpoint 
       ! file, and only partial checkpoint information is known to wg0
       ! until the end of the energy loop. Therefore
       ! restarts can only happen after the whole array is calculated.
       ! (see Q in dlf_checkpoint_write)
       if (glob%ntasks > 1) goto 1010
       ! Serial runs can potentially restart after 
       ! any energy evaluation
       goto 1000
       ! NB: serial and parallel checkpoint files are not compatible
    end if

    if(.not.trerun_energy) then
       if (glob%imicroiter < 2) then
          stat%ccycle = stat%ccycle+1
          if (printl > 0) then
            write(stdout,*)
            write(stdout,*) "@ Optimize for New Step"
            write(stdout,*)
            write(stdout,*)
            write(stdout,'(12("="))',advance="no")
            write(stdout,'(2x,"GEOMETRY FOR OPTIMIZATION STEP",I4,2x)',advance="no")stat%ccycle
            write(stdout,'(12("="))')
            write(stdout,*)
            write(stdout,'("GEOMETRY INPUT")')
            write(stdout,'("ELEMENT",6x,"X",14x,"Y",14x,"Z")')
            call write_xyz(stdout,glob%nat,glob%znuc,glob%xcoords)
          endif
       else 
          stat%miccycle = stat%miccycle + 1
          stat%tmiccycle = stat%tmiccycle + 1
       end if
    end if
    stat%sene=stat%sene+1
    if(stat%ccycle > glob%maxcycle) then
      stat%ccycle= stat%ccycle-1
      stat%sene=stat%sene-1
      if(printl>0) write(stdout,"(&
          &'Stopping: maximum number of cycles reached')")
      exit
    end if

    ! ==================================================================
    ! EVALUATE THE ENERGY
    ! ==================================================================

    ! get out of main cycle if energy evaluated more often than glob%maxene
    if(stat%sene > glob%maxene) then
      stat%ccycle= stat%ccycle-1
      stat%sene=stat%sene-1
      if(printl>=2) write(stdout,"(&
          &'Stopping: maximum number of energy evaluations reached')")
      exit
    end if

    ! Parallel NEB: determine whether this workgroup should calculate a gradient
    !if (glob%icoord/100 == 1 .and. glob%ntasks > 1 .and. glob%iopt/=12 ) then
    call dlf_qts_get_int("TASKFARM_MODE",taskfarm_mode)
    call dlf_formstep_get_logical("FD_HESS_RUNNING",fd_hess_running)
    if (glob%icoord/100 == 1 .and. glob%ntasks > 1 .and. &
        (.not.fd_hess_running .or. taskfarm_mode==2)) then
       !taskfarm_mode: 1 for parallelization of QTS-FD-Hessians within each image
       !taskfarm_mode: 2 for parallelization of the images in QTS-Hessian calculations
       ! Parallel NEB
      call dlf_qts_get_int("IMAGE_STATUS",image_status)
      if(glob%icoord==190.and.taskfarm_mode==2) iimage=image_status
       glob%dotask = (mod(iimage-1,glob%ntasks) == glob%mytask)
       !  Serial first cycle
       if (glob%serial_cycle == 1) glob%dotask = .true.
    end if
    ! Note for parallel FD Hessian, dotask is set in dlf_fdhessian
    ! Other methods: always calculate gradient

    if (glob%dotask) then

       if(printl>=6) write(stdout,"('Calculating the energy',i5)") stat%sene
       stat%pene = stat%pene + 1
       call clock_start("EANDG")

       if (glob%imultistate == 0) then
         ! An ordinary single state gradient

         ! don't do any optimizations or energy evaluations in case only
         ! the rate has to be calculated from an existing qts hessian
         if(.not.noeandg) then

            ! ESP fit setting
            kiter = -1
            if (glob%micro_esp_fit) then
               ! TODO - at certain times kiter should still be -1, e.g.
               ! during initial Hessian calc in P-RFO
               if (glob%imicroiter == 1) kiter = 0
               if (glob%imicroiter == 2) kiter = 1
            end if

            call dlf_get_gradient(glob%nvar,glob%xcoords,glob%energy, &
                 glob%xgradient,iimage,kiter,&
#ifdef GAMESS
                 core,&
#endif
                 status,ierr2)

            ! ESP fit corrections
            if (glob%micro_esp_fit .and. status == 0) then
               if (glob%imicroiter == 1) then
                  ! Macro step - calculate corrections to energy/gradient
                  ! Get ESP eandg
                  glob%macrocoords(:,:,iimage) = glob%xcoords
                  kiter = 1
                  call dlf_get_gradient(glob%nvar,glob%xcoords,glob%e0corr(iimage), &
                       glob%g0corr(:,:,iimage),iimage,kiter,&
#ifdef GAMESS
                       core,&
#endif
                       status,ierr2)


                  ! e0corr = E0(full) - E0(esp fit)
                  glob%e0corr(iimage) = glob%energy - glob%e0corr(iimage)
                  ! g0corr = G0(full) - G0(esp fit)
                  ! Eq 5 in HDLCOpt paper
                  ! External program should ensure that g0corr for inner regions = 0
                  glob%g0corr(:,:,iimage) = glob%xgradient - glob%g0corr(:,:,iimage)
               else if (glob%imicroiter == 2) then
                  ! Micro step - apply corrections
                  ! Eq 6 in HDLCOpt paper
                  svar = glob%e0corr(iimage) + &
                       sum( glob%g0corr(:,:,iimage) &
                       * (glob%xcoords - glob%macrocoords(:,:,iimage)) )
                  glob%energy = glob%energy + svar
                  if (printl >= 4) write(stdout,'(a, 22x, f20.6)') &
                       "Microiterative energy correction: ", svar
                  ! Eq 4 in HDLCOpt paper
                  glob%xgradient = glob%xgradient + glob%g0corr(:,:,iimage)


               else
                  call dlf_fail("ESP fit only appropriate for microiterative opts")
               end if
            end if

         end if
       else
          ! Multiple state gradient calculation
          if (printl >= 6) write(stdout, '(a)') "Calculating multistate energies"
          call dlf_get_multistate_gradients(glob%nvar,glob%xcoords,glob%msenergy, &
               glob%msgradient,glob%mscoupling,glob%needcoupling,iimage,status)
          ! Make sure coupling is zero if it is required by algorithm but 
          ! coupled_states was false
          if (glob%needcoupling == 0) glob%mscoupling = 0.0d0
          ! Form the objective function and gradient from the individual
          ! state gradients
          if (printl >= 4) then
             write(stdout, '(a, f20.10)') "Lower state energy: ", glob%msenergy(1)
             write(stdout, '(a, f20.10)') "Upper state energy: ", glob%msenergy(2)
             write(stdout, '(a, f20.10)') "Energy difference:  ", &
                  abs(glob%msenergy(1) - glob%msenergy(2))
          endif
          if (printl >= 6) write(stdout, '(a)') "Forming objective function"
          call dlf_make_conint_gradient
       endif

       call clock_stop("EANDG")

       ! check of NaN in the energy (comparison of NaN with any number 
       ! returns .false. , pgf90 does not understand isnan() )
       if( abs(glob%energy) > huge(1.D0) ) then
          status=1
       else
          if (.not. abs(glob%energy) < huge(1.D0) ) status=1
       end if

       ! Parallel runs must share status
       ! so that all workgroups are given notice of failure
       ! (status should be shared before attempting to share other data)
       if (status/=0) then
          if (glob%ntasks > 1) then
             call dlf_tasks_int_sum(status, 1)
          else
             call dlf_report(trestarted_report)
          end if
          call dlf_fail("Energy evaluation failed")
       end if
 
       if(iimage==1) then
          !if(printl>=2) !write(stdout,'(1x,a,es16.9)') &
          !     "Energy calculation finished, energy: ", &
          !     glob%energy
       else
          !if(printl>=2) write(stdout,'(1x,a,i4,a,es16.9)') &
          !     "Energy calculation of image ",iimage,&
          !     " finished, energy: ",glob%energy
       end if
       
    else ! .not. glob%dotask
       ! Multistate calculations should not reach this point
       if (glob%imultistate /= 0) then
          call dlf_fail("Multistate logic error")
       end if
       glob%energy = 0.d0
       glob%xgradient = 0.d0


       status = 0       
    end if

    ! send coordinates to the calling program
!    if(printf>=1) then
!      call dlf_put_coords(glob%nvar,1,glob%energy,glob%xcoords,glob%iam)
!    end if

    ! write restart information (serial runs)
    if(stat%sene<=glob%maxdump .and. glob%ntasks == 1) then
      if(glob%dump>0) then
        if(mod(stat%sene,glob%dump)==0) then
          call clock_start("CHECKPOINT")
          if(printl>=6) write(stdout,"('Writing restart information')")
          call dlf_checkpoint_write(status)
          call clock_stop("CHECKPOINT")
        end if
      end if
    end if
    ! come here if (serial) checkpoint file successfully read
1000 trestarted=.false.

    ! ==================================================================
    ! TRANSFORM CARTESIANS TO INTERNALS (COORDS AND GRAD)
    ! ==================================================================
    call clock_start("COORDS")
    call dlf_coords_xtoi(trerun_energy,testconv,iimage,status)

    ! Microiterative opt - set up mic arrays (after macro step as well)
    if (glob%imicroiter > 0) then
       call dlf_microiter_itomic
    end if

    call clock_stop("COORDS")

    ! copied here to allow for qTS Hessian calculations.
    ! ==================================================================
    if(trerun_energy) cycle ! main optimisation cycle
    ! ==================================================================

    ! ==================================================================
    ! Multistate gradient: post-transformation
    ! ==================================================================
    if (glob%imultistate == 3) then
       call clock_start("EANDG")
       call dlf_make_ln_gradient_posttrans
       call clock_stop("EANDG")
    endif

    ! ==================================================================
    ! Calculate the Hessian if necessary
    ! ==================================================================
    ! glob%ihessian is allocated and deallocated in formstep_init and
    !  formstep_destroy
    if(needhessian) then
      if(glob%iopt/=12.and.glob%iopt/=9) then
        tconv = .false.
        ! Make sure the Hessian exists. This can imply energy cycles (trerun_energy)
        ! or a call to dlf_get_hessian
        call dlf_makehessian(trerun_energy,tconv &
#ifdef GAMESS
          ,core&
#endif
            )
        if (tconv) exit
      else
!if glob%iop=12
!        call dlf_qts_makehessian(trerun_energy)
      end if
    end if

    ! ==================================================================
    if(trerun_energy) cycle ! main optimisation cycle
    ! ==================================================================

    ! Parallel NEB: revert to parallel
    glob%serial_cycle = 0

    ! Restart information for task-farmed jobs
    if(stat%ccycle <= glob%maxdump .and. glob%ntasks > 1) then
      if(glob%dump>0) then
        if(mod(stat%ccycle,glob%dump)==0) then
          call clock_start("CHECKPOINT")
          if(printl>=6) write(stdout,"('Writing restart information')")
          call dlf_checkpoint_write(status)
          call clock_stop("CHECKPOINT")
        end if
      end if
    end if    
    ! come here if task-farmed checkpoint file read
1010 trestarted=.false.

    ! exit if only the hessian and thermal analysis should be calculated
    if(glob%iopt == 11) then
      call dlf_thermal
      exit ! no optimisation cycles
    end if

    ! exit if only the instanton rate should be calculated
    if(glob%iopt == 12) then
      call dlf_qts_rate()
      exit ! no optimisation cycles
    end if

    ! exit if test_delta is running
    if(glob%iopt == 9) then
      exit ! no optimisation cycles
    end if
    

    ! write trajectory
    if(printf>=3 .and. glob%iam == 0) then
      if (glob%imicroiter < 2) then
         ! Write out standard optimisation cycles
         ! and macroiterative steps
!         call write_xyz(30,glob%nat,glob%znuc,glob%xcoords)
!         call write_xyz_active(31,glob%nat,glob%znuc,glob%spec,glob%xcoords)
      end if
      if (glob%imicroiter > 0) then
         ! Full microiterative opt path including microiterative steps
         call write_xyz(33,glob%nat,glob%znuc,glob%xcoords)
      endif
    end if

    ! write forcetrajectory
    if(printf>=4.and.glob%iam == 0) then
      if (glob%imicroiter < 2) then
         call write_xyz_active(stdout,glob%nat,glob%znuc,glob%spec,glob%xgradient)
      end if
    end if

    ! if trust-radius, test for step acceptance. 
    ! If rejected, do not form a new step and keep old energy.
    if (glob%imicroiter == 2) then
       call dlf_microiter_test_acceptance(tswitch)
       if (tswitch) then
          ! Reached minimum trust radius, switch to macro
          glob%imicroiter = 1
          call dlf_lbfgs_deselect
          ! Cycle to perform a new full eandg calculation
          ! TODO refactor to reach normal itox below
          call dlf_coords_itox(iimage)
          cycle
       end if
    else
       if (glob%iline==1) then
          call test_acceptance
       end if
       if (glob%iline==2) then
          call test_acceptance_g
       end if
       if (glob%iline==3) then
          call linesearch
       end if
    end if

  !  if(glob%icoord==190) then
  !    ! set convergence test to cartesian coordinates and gradients
  !    call dlf_qts_convergence
  !  end if

    ! ==================================================================
    ! TEST FOR CONVERGENCE
    ! ==================================================================
    if(glob%taccepted) then
       if (glob%imicroiter < 2) then
          stat%caccepted=stat%caccepted+1
       else 
          stat%tmicaccepted = stat%tmicaccepted + 1
       end if
      tconv=.false.
      if(.not.testconv) then
         if (glob%imicroiter < 2) then
            ! Standard convergence test
            call convergence_test(stat%ccycle,.true.,tconv)
            if (tconv) then
              if (printl > 0) then
                write(stdout,'(/" GEOMETRY OPTIMIZED AFTER",i5," CYCLES")') stat%ccycle
                call clock_start("COORDS")
                call dlf_coords_itox(iimage)
                call clock_stop("COORDS")
                write(stdout,*)
                write(stdout,*)"@ Finish Optimization for This Step "
                write(stdout,*)
                exit 
              endif
            endif
         else
            ! Test convergence of microiterations
            call dlf_microiter_convergence(tconv)
            if (tconv) then
               if (printl>=2) then
                  write(stdout,"('Microiterations converged')")
                  write(stdout,"('Switching to macroiterations.')")
               end if
               glob%imicroiter = 1
               call dlf_lbfgs_deselect
               ! Cycle to perform a new full eandg calculation
               ! TODO refactor to reach normal itox below
               call dlf_coords_itox(iimage)
               cycle
            end if
         end if
      end if
    end if
    
    ! Exit microiterations if maxmicrocycle reached
    if (glob%imicroiter == 2 .and. stat%miccycle == glob%maxmicrocycle) then
       if(printl>0) then
          write(stdout,"('Maximum number of microiterative cycles reached')")
          write(stdout,"('Switching to macroiterations.')")
       end if
       glob%imicroiter = 1
       call dlf_lbfgs_deselect
       ! Cycle to perform a new full eandg calculation
       ! TODO refactor to reach normal itox below
       call dlf_coords_itox(iimage)
       cycle
    end if

    ! ==================================================================
    ! FORM AN OPTIMISATION STEP
    ! ==================================================================
    if(glob%taccepted .and. glob%imicroiter < 2) then
      call clock_start("FORMSTEP")
      call dlf_formstep
      call clock_stop("FORMSTEP")
    end if

    ! ==================================================================
    ! DO A LINE SEARCH OR A TRUST RADIUS STEP
    ! ==================================================================

    if(glob%taccepted .and. glob%imicroiter < 2) then
      
      ! scale the step 
      call dlf_scalestep

      call clock_start("COORDS")

      ! check the step in special cases
      ! Set step of frozen NEB images to zero
      if(glob%icoord/100==1) call dlf_neb_checkstep

      ! Check step in case of dimer
      if(glob%icoord/100==2) call dlf_dimer_checkstep
      
      ! For macroiterations only move the inner region
      if (glob%imicroiter == 1) call dlf_microiter_check_macrostep

      call clock_stop("COORDS")

      ! do the step
      glob%icoords(:)=glob%icoords(:) + glob%step(:)

    end if

    ! write steptrajectory
  !  if(printf>=4) then
  !    ! these are icoords - it will only work for cartesians without frozen atoms
       ! therefore commented out
  !    call write_xyz(301,glob%nivar/3,glob%znuc(1:glob%nivar/3),glob%step)
  !  end if

    if(glob%taccepted .and. glob%imicroiter < 2) then
       ! store old values
       glob%oldenergy=glob%energy
       glob%toldenergy=.true.
    end if

    ! ==================================================================
    ! MICROITERATIVE STEP
    ! ==================================================================

    ! Microiterative step
    if (glob%imicroiter == 2 .and. glob%taccepted) then

       ! Form the microiterative step
       call clock_start("FORMSTEP")
       call dlf_microiter_formstep
       call clock_stop("FORMSTEP")

       ! Scale the microiterative step
       call dlf_microiter_scalestep

       ! Set microiterative step of frozen NEB images to zero
       if(glob%icoord/100==1) call dlf_neb_checkstep

       ! Do the step and store old microiterative energy
       call dlf_microiter_step

    end if

    ! Switch from macro to microiterations if necessary
    ! Do not take a micro step yet - need to get a reference energy
    ! to test acceptance of first micro step.
    if (glob%imicroiter == 1 .and. glob%taccepted) call dlf_microiter_enter

    ! ==================================================================
    ! write(stdout,*)" @TRANSFORM INTERNAL COORDINATES TO CARTESIANS (COORDS ONLY)
    ! ==================================================================

    call clock_start("COORDS")
    call dlf_coords_itox(iimage)
    call clock_stop("COORDS")
    if (printl > 0) then
      write(stdout,*)
      write(stdout,*)"@ Finish Optimization for This Step"
    endif

 do iat=1,glob%nat
    do jat=1,3
!      xyz(jat,iat)=glob%xcoords((iat-1)*3+jat)
       xyz(jat,iat)=glob%xcoords(jat,iat)
    enddo
 enddo

  end do ! main simulation cycle


  ! Job finished, prepare for shutdown
  if (glob%iopt /= 11 .and. glob%iopt /= 12 .and. glob%iopt /= 9 ) then
    if(tconv) then
      if (printl > 0) then
         write(stdout,'("================ OPTIMIZED GEOMETRY INFORMATION ==============")')
      endif
    else
      if (printl > 0) then
         write(stdout,*)                                                  
         write(stdout,*) "WARNING: REACHED MAX OPT CYCLES. THE GEOMETRY IS NOT OPTIMIZED."
         write(stdout,*) "         PRINTING THE GEOMETRY FROM LAST STEP."
         write(stdout,'("============= GEOMETRY INFORMATION (NOT OPTIMIZED) ===========")') 
      end if                                                                
    end if

    if (printl > 0) then     
      write(stdout,*)
      write(stdout,'(" OPTIMIZED GEOMETRY IN CARTESIAN")')
      write(stdout,'(" ELEMENT",6x,"X",14x,"Y",14x,"Z")')
      call write_xyz(stdout,glob%nat,glob%znuc,glob%xcoords)

      write(stdout,*)
      write(stdout,'(" FORCE")')
      write(stdout,'(" ELEMENT",6x,"X",14x,"Y",14x,"Z")')
      call write_xyz(stdout,glob%nat,glob%znuc,-glob%xgradient)      

      if (glob%imultistate == 0) then
        call convergence_get("VALE", svar)
        write(stdout,*) 
        write(stdout,'(" MINIMIZED ENERGY=",F15.10)') svar
      else
        write(stdout,*)
        write(stdout,'(1x,a,f20.12)') &
             "FINAL LOWER STATE ENERGY= ", glob%msenergy(1)
        write(stdout,'(1x,a,f20.12)') &
             "FINAL UPPER STATE ENERGY= ", glob%msenergy(2)
        write(stdout,'(1x,a,f20.12,/)') "FINAL ENERGY DIFFERENCE=  ", &
             abs(glob%msenergy(1) - glob%msenergy(2))
      end if
      write(stdout,'("===============================================================")')
      write(stdout,*) 
      write(stdout,*)"@ Finish Optimization Job"
      write(stdout,*)
    end if
  end if

  ! calculate the qts rate if converged
  if(glob%icoord==190.and.tconv.and.glob%iopt/=12.and.glob%havehessian) then
    if(printl>=2) then
      write(stdout,'(a)') "Calculating the Rate. WARNING: the rate information below was obtained"
      write(stdout,'(a)') " from updated Hessians. It is less accurate than a full rate calculation"
    end if
    call dlf_qts_rate()
  end if

  ! close trajectory files
  if(printf>=3 .and. glob%iam == 0) then
!    close(30)
!    close(31)
    if (glob%iopt == 10 .or. (glob%icoord >= 100 .and. glob%icoord < 300)) then
       close(32)
    end if
    if (glob%imicroiter > 0) close(33)
  end if
  ! close force trajectory files
  if(printf>=4.and.glob%iam == 0) then
!    close(300)
    !close(301)
  end if
  ! again some rubbish for printing ...
  if(glob%icoord/100==1.and..not.glob%tatoms) then
    if (glob%iam == 0) then
       open(unit=30,file="path.inc")
       do ivar=1,glob%nimage
         write(30,"('sphere{<',f12.4,',',f12.4,',',f12.4,'> 0.04}')") &
             glob%icoords(ivar*2-1),1.D0,glob%icoords(ivar*2)
         if(ivar>1) then
           write(30,"('cylinder{<',f12.4,',',f12.4,',',f12.4,'>,"//&
               &"<',f12.4,',',f12.4,',',f12.4,'> 0.01} ')") &
               glob%icoords(ivar*2-1),1.D0,glob%icoords(ivar*2), &
               glob%icoords((ivar-1)*2-1),1.D0,glob%icoords((ivar-1)*2)
         end if
       end do

!       close(30)
    end if
  end if

  ! ====================================================================
  ! CLOSE DOWN
  ! ====================================================================

  ! delete memory for line search 
  call linesearch_destroy

  ! delete memory for search algorithm
  call dlf_formstep_destroy

  ! delete memory for internal coordinates
  call dlf_coords_destroy

  !write report on this optimisation
  call dlf_report(trestarted_report)

end subroutine dlf_run
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_allocate_glob
!!
!! FUNCTION
!!
!! Allocate the arrays globally available to the optimiser,
!! set a few starting values.
!!
!! SYNOPSIS
subroutine dlf_allocate_glob(nvarin,nvarin2,nvarspec, &
    tmpcoords,tmpcoords2,spec,nz,nframe,nmass,nweight,n_po_scaling)
!! SOURCE
  use dlf_parameter_module, only: rk,ik
  use dlf_global, only: glob,stdout,printl
  use dlf_allocate, only: allocate
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer, intent(in) :: nvarin
  integer, intent(in) :: nvarin2
  integer, intent(in) :: nvarspec
  real(rk),intent(in) :: tmpcoords(nvarin)
  real(rk),intent(in) :: tmpcoords2(nvarin2)
  integer, intent(in) :: spec(nvarspec)
  integer, intent(in) :: nz
  integer, intent(in) :: nframe
  integer, intent(in) :: nmass
  integer, intent(in) :: nweight
  integer, intent(in) :: n_po_scaling
  integer :: nat,ivar
  real(rk):: svar
! **********************************************************************
  if(glob%tinit) return ! this instance has been initialised
  glob%tcoords2=(nframe>0)
  if(glob%tatoms) then
    ! input contains atoms
    if(mod(nvarin,3)/=0) call dlf_fail("nvarin has to be 3*nat")
    nat=nvarin/3
    glob%nvar=nvarin
    if(printl>4) write(stdout,'(a,i5,a)') &
      "Input contains ",nat," atoms"
    glob%nat=nat

    call allocate( glob%xcoords,3,nat)

    if(glob%tcoords2) then
      call allocate( glob%xcoords2,3,nat,nframe)
      glob%xcoords2 = reshape(tmpcoords2(1:nat*3*nframe),(/3,nat,nframe/))
    end if

    call allocate( glob%xgradient,3,nat)
    call allocate( glob%spec,nat)
    !nuclear charges
    call allocate( glob%znuc,nat)
    if(nz==nat) then
      glob%znuc(:)=spec(nat+1:nat+nz)
    else
      glob%znuc(:)=1
    end if
    
    glob%xcoords = reshape(tmpcoords,(/3,nat/))
    glob%spec(:) = spec(1:nat)

    if(glob%ncons>0) then
      call allocate( glob%icons,5,glob%ncons)
      ivar=nat+nz
      glob%icons=reshape(spec(ivar+1:ivar+5*glob%ncons),(/5,glob%ncons/))
    else
      call allocate( glob%icons,5,1)
      glob%icons(:,:)=0
    end if

    ! user input connections - additional to the hdlc-primitive created
    if(glob%nconn>0) then
      call allocate( glob%iconn,2,glob%nconn)
      ivar=nat+nz+glob%ncons
      glob%iconn=reshape(spec(ivar+1:ivar+2*glob%nconn),(/2,glob%nconn/))
    else
      call allocate( glob%iconn,2,1)
      glob%iconn(:,:)=0
    end if

    ! microiterative specification
    call allocate(glob%micspec, nat)
    if (glob%imicroiter > 0) then
       ivar = nat + nz + 5*glob%ncons + 2*glob%nconn
       glob%micspec(:) = spec(ivar+1:ivar+nat)
       do ivar = 1, nat
          if (glob%micspec(ivar) < 0 .or. glob%micspec(ivar) > 1) then
             write(stdout,'(a,2i5)') 'Atom, Micspec: ', ivar, glob%micspec(ivar)
             call dlf_fail("Incorrect microiterative specification")
          end if
       end do
    else
       ! override inner atoms list for non-microiterative opt
       ! (logically, all atoms are in the core)
       glob%micspec(:) = 1
    end if
    ! ESP fit
    call allocate(glob%macrocoords,3,nat,glob%nimage)
    call allocate(glob%g0corr,3,nat,glob%nimage)
    call allocate(glob%e0corr,glob%nimage)
    glob%macrocoords = 0.0d0
    glob%g0corr = 0.0d0
    glob%e0corr = 0.0d0

    call allocate( glob%weight,nat)
    if(nweight>0) then
      glob%weight=tmpcoords2(nat*3*nframe+1:nat*3*nframe+nat)
    else
      glob%weight(:)=1.D0
    end if
    ! Set weight to zero for outer microiterative region 
    ! so it takes no part in dimer rotation, NEB spring force etc.
    do ivar = 1, nat
       if (glob%micspec(ivar) == 0) glob%weight(ivar) = 0.0d0
    end do

    call allocate( glob%mass,nat)
    if(nmass>0) then
      glob%mass=tmpcoords2(nat*3*nframe+nweight+1:nat*3*nframe+nweight+nat)
      ! use atomic mass unit (mass of the electron) only in case of quantum TS search
      if((glob%icoord==190.or.glob%icoord==390).and.&
          glob%iopt/=11.and.glob%iopt/=13) then
        glob%massweight=.true.
        
        call dlf_constants_get("AMU",svar)
        glob%mass=glob%mass*svar !*(1.66054D-27/9.10939D-31)
        
      end if
    else
      glob%mass(:)=1.D0
    end if

    if (glob%iopt/10==5) then ! parallel optimizers
       call allocate(glob%po_radius_scaling,max(1,n_po_scaling))
       if (n_po_scaling == 0) then 
          ! this is a shorthand for setting all elements of the glob%po_radius(:) and 
          ! glob%po_tolerance_r(:) arrays to their respective base values.
          glob%po_radius_scaling(:) = 1.0D0 ! there should only be one component anyway
       else
          ! n_po_scaling < 0 will be caught in dlf_read_in
          glob%po_radius_scaling(:) = tmpcoords2(nat*3*nframe + nweight + nmass + 1 : &
                                      nat*3*nframe + nweight + nmass + n_po_scaling)
       end if 
    end if

    ! multistate arrays for conical intersection search
    if (glob%imultistate > 0) then
       call allocate(glob%msenergy, 2)
       call allocate(glob%msgradient, 3, nat, 2)
       if (glob%imultistate > 1) then
          ! Set up coupling array for GP/LN regardless of whether
          ! it needs to be calculated. Will be set to zero if
          ! needcoupling is false.
          call allocate(glob%mscoupling, 3, nat)
       else
          ! dummy array just to avoid problems in argument list
          ! of dlf_get_multistate_gradients - I assume you can't
          ! send unallocated arrays to a C subroutine
          call allocate(glob%mscoupling, 1, 1)
       endif
    endif

  else
    ! input contains sequential variables (as the 2D-potentials)
    glob%nat=-1
    glob%nvar=nvarin
    if(printl>4) write(stdout,'(a,i5,a)') &
      "Input contains ",nvarin," degrees of freedom (no atoms)"
    call allocate( glob%xcoords,1,nvarin)
    if(glob%tcoords2) then
      call allocate( glob%xcoords2,1,nvarin,nframe)
      glob%xcoords2 = reshape(tmpcoords2,(/1,nvarin,nframe/))
    end if
    call allocate( glob%xgradient,1,nvarin)
    ! NOT TO BE USED
    call allocate( glob%spec,1)
    call allocate( glob%icons,5,1)
    glob%icons(:,:)=0
    call allocate( glob%znuc,1)
    call allocate( glob%iconn,2,1)
    glob%iconn(:,:)=0
    call allocate( glob%weight,1)
    call allocate( glob%mass,1)

  !  glob%xcoords = reshape(tmpcoords,(/1,nvarin/))

  end if
  glob%toldenergy=.false.
  glob%tinit=.true.
end subroutine dlf_allocate_glob
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_deallocate_glob
!!
!! FUNCTION
!!
!! Deallocate global arrays
!!
!! SYNOPSIS
subroutine dlf_deallocate_glob
!! SOURCE
  use dlf_parameter_module, only: rk,ik
  use dlf_global, only: glob
  use dlf_allocate, only: deallocate
  implicit none
! **********************************************************************
  if(.not.glob%tinit) return ! this instance has not been initialised
  call deallocate( glob%xcoords )
  if(glob%tcoords2) then
    call deallocate( glob%xcoords2 )
  end if
  call deallocate( glob%xgradient )

  ! arrays only used for atoms
  call deallocate( glob%spec  )
  call deallocate( glob%icons )
  call deallocate( glob%znuc  )
  call deallocate( glob%iconn )
  call deallocate(glob%micspec)

  call deallocate( glob%mass )
  call deallocate( glob%weight )

  glob%tinit=.false.

  ! conical intersection search
    if (glob%imultistate > 0) then
       call deallocate(glob%msenergy)
       call deallocate(glob%msgradient)
       call deallocate(glob%mscoupling)
    endif

  ! parallel optimisation algorithms
  if (glob%iopt/10==5) call deallocate(glob%po_radius_scaling)

  call deallocate(glob%macrocoords)
  call deallocate(glob%g0corr)
  call deallocate(glob%e0corr)

end subroutine dlf_deallocate_glob
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_fail
!!
!! FUNCTION
!!
!! Shut down DL-FIND after a severe error. Calls dlf_error, which has
!! to be provided by the calling code.
!!
!! SYNOPSIS
subroutine dlf_fail(msg)
!! SOURCE
  use dlf_global, only: glob, stdout, stderr
  use dlf_store, only: store_delete_all
  implicit none
  character(*),intent(in) :: msg
  call flush(stdout)
  call flush(stderr)
  write(stderr,"(/,a,/,a,/)") "DL-FIND ERROR:",msg
  write(stdout,"(/,a,/,a,/)") "DL-FIND ERROR:",msg
  call flush(stdout)
  call flush(stderr)
  ! Clean up allocatable arrays.
  ! Otherwise they will stay allocated and DL-FIND
  ! cannot be called again by the external code.
  ! glob%cleanup prevents an infinite loop if one of the 
  ! destroy functions calls dlf_fail...
  glob%cleanup = glob%cleanup + 1
  if (glob%cleanup <= 1) then
     call linesearch_destroy
     call dlf_formstep_destroy
     call dlf_coords_destroy
     call dlf_deallocate_glob
     call dlf_formstep_set_tsmode(1,-2,1.0d0)
     call store_delete_all
  else
     call flush(stdout)
     call flush(stderr)
       write(stderr,"(/,a,/)") "dlf_fail: clean up failed"
       write(stdout,"(/,a,/)") "dlf_fail: clean up failed"
     call flush(stdout)
     call flush(stderr)
  end if
  ! Exit
  call dlf_error()
  ! this should not be reached, as dlf_error should not return
  stop
end subroutine dlf_fail
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_report
!!
!! FUNCTION
!!
!! Report information about the run to stdout. Called at the beginning 
!! and at the end of a calculations.
!!
!! SYNOPSIS
subroutine dlf_report(trestarted)
!! SOURCE
  use dlf_global, only: glob,stdout,printl
  use dlf_stat, only: stat
  implicit none
  logical ,intent(in) :: trestarted
  integer             :: ia3(3)
  integer             :: kk

! **********************************************************************


  ! Can't return as dlf_mpi_counters requires communication
  ! if(printl<=0) return
  if (printl <= 0) goto 111 ! to skip the report printing up to the 
                            ! call to dlf_mpi_counters

  write(stdout,'(/,a)') "DL-FIND Report:"
  write(stdout,'(a)')   "==============="

  ! ====================================================================
  ! Multistate calculations
  ! ====================================================================
  if (glob%imultistate == 1) then
     write(stdout,'(a)') &
          "Conical intersection algorithm: penalty function"
     write(stdout,1000) "Penalty function constant c1", &
          glob%pf_c1
     write(stdout,1000) "Penalty function constant c2", &
          glob%pf_c2
  end if
  if (glob%imultistate == 2) then
     write(stdout,'(a)') &
       "Conical intersection algorithm: gradient projection"
     write(stdout,1000) "Gradient projection constant c3", &
          glob%gp_c3
     write(stdout,1000) "Gradient projection constant c4", &
          glob%gp_c4
  end if
  if (glob%imultistate == 3) then
     write(stdout,'(a)') &
       "Conical intersection algorithm: Lagrange-Newton"
     write(stdout,1000) "Lagrange-Newton threshold t1", &
          glob%ln_t1
     write(stdout,1000) "Lagrange-Newton threshold t2", &
          glob%ln_t2
  end if
  if (glob%imultistate > 0) then
     write(stdout,2000) "Lower electronic state", &
          glob%state_i
     write(stdout,2000) "Upper electronic state", &
          glob%state_j
  end if

  ! ====================================================================
  ! Optimiser
  ! ====================================================================
  if(glob%iopt==0) write(stdout,'(a)') &
      "Optimisation algorithm: Steepest descent"
  if(glob%iopt==1) write(stdout,'(a)') &
      "Optimisation algorithm: Conjugate gradient"
  if(glob%iopt==2) write(stdout,'(a)') &
      "Optimisation algorithm: Conjugate gradient"
  if(glob%iopt==3) then
    write(stdout,'(a)') "Optimisation algorithm: L-BFGS"
    write(stdout,2000) "Number of steps in L-BFGS memory", &
         glob%lbfgs_mem
  end if
  if(glob%iopt==9) then 
    write(stdout,'(a)') &
      "Optimisation algorithm: no optimization. Test different deltas"
    write(stdout,'(a)') "    for the finite-difference calculation of Hessians."
  end if
  if (glob%iopt==51) write(stdout,'(a)') &
      "Optimisation algorithm: Stochastic search"
  if (glob%iopt==52) write(stdout,'(a)') &
      "Optimisation algorithm: Genetic algorithm"

  if(glob%iopt==10) write(stdout,'(a)') &
      "Optimisation algorithm: P-RFO"
  if(glob%iopt==11) then 
    write(stdout,'(a)') &
        "Optimisation algorithm: no optimisation, just calculate the"
    write(stdout,'(a)') "    Hessian and thermal corrections"
    if (glob%inithessian == 5) write(stdout,'(a)') &
        "Initial Hessian read from file"
    write(stdout,2000) "Modes assumed to have zero vibrational frequency",glob%nzero
    write(stdout,1000) "Temperature",glob%temperature,"K"
  end if
  if(glob%iopt==12) then 
    write(stdout,'(a)') &
        "Optimisation algorithm: no optimisation, just calculate the"
    write(stdout,'(a)') "    instanton reaction rate"
    write(stdout,2000) "Number of modes assumed to have zero vibrational frequency",glob%nzero
    if (glob%inithessian == 5) write(stdout,'(a)') &
        "Initial Hessian read from file"
  end if
  if(glob%iopt==13) then 
    write(stdout,'(a)') &
        "Calculation of the reaction rate (no tunnelling)"
  end if
  if(glob%iopt==20) write(stdout,'(a)') &
      "Optimisation algorithm: Newton-Raphson"
  if(glob%iopt==30) write(stdout,'(a)') &
      "Optimisation algorithm: Damped dynamics"
  if(glob%iopt==40) write(stdout,'(a)') &
      "Optimisation algorithm: Lagrange-Newton"

  ! Hessian options
  if(glob%iopt==10.or.glob%iopt==20.or.glob%iopt==40) then
    if (glob%inithessian == 0) write(stdout,'(a)') &
         "Initial Hessian from external program"
    if (glob%inithessian == 1) write(stdout,'(a)') &
         "Initial Hessian by one-point finite difference"
    if (glob%inithessian == 2) write(stdout,'(a)') &
         "Initial Hessian by two-point finite difference"
    if (glob%inithessian == 3) write(stdout,'(a)') &
         "Initial Hessian by diagonal one-point finite difference"
    if (glob%inithessian == 4) write(stdout,'(a)') &
         "Initial Hessian is identity matrix"
     if (glob%inithessian == 5) write(stdout,'(a)') &
         "Initial Hessian read from file"
    if(glob%update==0) write(stdout,'(a)') &
      "No Hessian updates"
    if(glob%update==1) write(stdout,'(a)') &
      "Hessian update mechanism: Powell"
    if(glob%update==2) write(stdout,'(a)') &
      "Hessian update mechanism: Bofill"
    if(glob%update==3) write(stdout,'(a)') &
      "Hessian update mechanism: BFGS"
    if(glob%icoord/=190) &
        write(stdout,2000) "Maximum Number of Hessian updates before recalc.", &
        glob%maxupd
    write(stdout,1000) "Finite difference for Hessian calculation", &
         glob%delta
    if(glob%soft>0.D0) then
      write(stdout,1000) "Eigenmodes below this value are considered soft", &
          glob%soft
    else
      write(stdout,'(a)') "No eigenmodes are considered soft"
    end if
    write(stdout, 1000) "Minimum step size for Hessian update", &
         glob%minstep
     
  end if
  if(glob%iopt==11 .or. glob%iopt==12) then
    write(stdout,1000) "Finite difference for Hessian calculation", &
         glob%delta
  end if

  ! Damped dynamics options
  if(glob%iopt==30) then
    write(stdout,1000) "Time step (a.u.)", &
          glob%timestep
    write(stdout,1000) "Start friction", &
          glob%fric0
    write(stdout,1000) "Friction decreasing factor if energy decreases", &
          glob%fricfac
    write(stdout,1000) "Friction to apply if energy increases", &
          glob%fricp
  end if

  ! Parallel optimisation options
  if(glob%iopt/10==5) then
    write(stdout,2000) "Working population size", &
          glob%po_pop_size
    write(stdout,1000) "Base sample radius", &
          glob%po_radius_base
    write(stdout,1000) "Tolerance on max component of mod g", &
          glob%po_tolerance_g
    write(stdout,2000) "Maximum number of cycles", &
          glob%po_maxcycle
    do kk = 1, SIZE(glob%po_radius_scaling,1)
       write(stdout,1000) "Scaling factor for radius component", &
       glob%po_radius_scaling(kk)
    end do
    if (glob%iopt==51) then
       write(stdout,'(a)') "Stochastic-search-specific options:"
       write(stdout,1000) "Radius contraction factor", &
          glob%po_contraction
       write(stdout,1000) "Base tolerance on radius", &
          glob%po_tol_r_base
       if (glob%po_distribution == 1) write(stdout,'(a)') &
          "Search strategy: uniform"
       if (glob%po_distribution == 2) write(stdout,'(a)') &
          "Search strategy: force_direction_bias"
       if (glob%po_distribution == 3) then 
          write(stdout,'(a)') "Search strategy: force_bias"
          write(stdout,1000) "Scaling factor for absolute gradient vector", &
             glob%po_scalefac
       end if
    else if (glob%iopt==52) then
       write(stdout,'(a)') "Genetic-algorithm-specific options:"
       write(stdout,2000) "Initial population size", &
          glob%po_init_pop_size
       write(stdout,2000) "Number of cycles before resetting population", &
          glob%po_reset
       write(stdout,1000) "Mutation rate", &
          glob%po_mutation_rate
       write(stdout,1000) "Death rate", &
          glob%po_death_rate
       write(stdout,2000) "Number of low-energy minima to store", &
          glob%po_nsave
    end if
  end if

  write(stdout,*)

  ! ====================================================================
  ! Line search / trust radius
  ! ====================================================================
  if (glob%iopt/10 /= 5) then ! we're not running a parallel optimisation 
     if(glob%iline==0) write(stdout,'(a)') &
         "Step length: simple scaled"
     if(glob%iline==1) write(stdout,'(a)') &
         "Trust radius based on energy"
     if(glob%iline==2) write(stdout,'(a)') &
         "Trust radius based on the gradient"
     if(glob%iline==3) write(stdout,'(a)') &
         "Line search"
     write(stdout,1000) "Maximum step length", glob%maxstep
     if(glob%iline==0 .or. glob%iopt == 0) write(stdout,1000) &
          "Scaling step by", glob%scalestep

     write(stdout,*)
  end if 

  ! ====================================================================
  ! Coordinate system
  ! ====================================================================

  ! direct coordinate system
  if(mod(glob%icoord,10)==0) then
    if(glob%massweight) then
      write(stdout,'(a)') "Coordinate system: Mass-weighted Cartesian coordinates"
    else
      write(stdout,'(a)') "Coordinate system: Cartesian coordinates"
    end if
  end if
  if(mod(glob%icoord,10)==1) write(stdout,'(a)') &
      "Coordinate system: Hybrid delocalised internal coordinates (HDLC)"
  if(mod(glob%icoord,10)==2) write(stdout,'(a)') &
      "Coordinate system: Hybrid delocalised total connection scheme (HDLC-TC)"
  if(mod(glob%icoord,10)==3) write(stdout,'(a)') &
      "Coordinate system: Delocalised internal coordinates (DLC)"
  if(mod(glob%icoord,10)==4) write(stdout,'(a)') &
      "Coordinate system: Delocalised total connection scheme (DLC-TC)"

  ! multi-image approaches
  if(glob%icoord/10==10) write(stdout,'(a)') &
        "Nudged elastic band with minimised start and endpoint"
  if(glob%icoord/10==11) write(stdout,'(a)') &
        "Nudged elastic band with start and endpoint minimised perpendicular to path"
  if(glob%icoord/10==12) write(stdout,'(a)') &
        "Nudged elastic band with frozen start and endpoint"
  if(glob%icoord>=100.and.glob%icoord<190) then
    write(stdout,2000) "Number of images",glob%nimage
    write(stdout,1000) "NEB spring constant",glob%nebk
    ! == 0.D0 for reals is dangerous. Maybe replace by <= 0.D0 or something?
    if (glob%neb_climb_test == 0.0d0) then
      write(stdout,'(a)') "No climbing image"
    else 
      write(stdout,1000) "Tolerance factor to spawn climbing image",glob%neb_climb_test
    endif
    if (glob%neb_freeze_test == 0.0d0) then
      write(stdout,'(a)') "No freezing of images during optimisation"
    else
      write(stdout,1000) "Tolerance factor to freeze images", glob%neb_freeze_test
    endif
  end if
  ! qTS search
  if(glob%icoord==190) then
    write(stdout,'(a)') &
        "Instanton (Quantum transition state) search"
    write(stdout,2000) "Number of images",glob%nimage
    write(stdout,1000) "Temperature",glob%temperature,"K"
    write(stdout,1000) "Fraction of adapted tau",glob%nebk
    if(glob%iopt==3) then
      if(glob%tolrot>90.D0) then
        write(stdout,'(a)') &
            "Mode following, mode estimated from tangent of the path"
      else
        write(stdout,'(a)') &
            "Dimer method, rotation by line search without extrapolation"
        write(stdout,1000) "Dimer distance (mid- to endpoint)",glob%delta
        ! glob%maxrot is not used in QTS dimer at the moment
        !write(stdout,2000) "Maximum number of dimer rotations",glob%maxrot
        write(stdout,1000) "Tolerance in rotation",glob%tolrot,"degrees"
      end if
    end if
    if(glob%tcoords2.and.abs(glob%distort)>0.D0) then
      write(stdout,'(a)') "Start configuration: distributing images along coords2"
      write(stdout,1000) "Spread along coords2: ",glob%distort
    end if
    write(stdout,2000) "Number of modes assumed to have zero vibrational frequency",glob%nzero

  end if

  ! Dimer method
  if(glob%icoord/100==2) then
    if(glob%icoord/10==20) write(stdout,'(a)') &
        "Dimer method, rotation by optimiser"
    if(glob%icoord/10==21) write(stdout,'(a)') &
        "Dimer method, rotation by line search without extrapolation"
    if(glob%icoord/10==22) write(stdout,'(a)') &
        "Dimer method, rotation by line search with extrapolation"
    write(stdout,1000) "Dimer distance (mid- to endpoint)",glob%delta
    write(stdout,2000) "Maximum number of dimer rotations",glob%maxrot
    write(stdout,1000) "Tolerance in rotation",glob%tolrot,"degrees"
    if(.not.glob%tcoords2) write(stdout,'(a)') &
        "Initial dimer direction randomised"

  end if

  ! Chain Method
  if(glob%icoord/100==3) then
    write(stdout,'(a)') "Chain method with augmented-Lagrange optimizer"
    write(stdout,1000) "Delta for finite-difference",glob%delta
    write(stdout,2000) "Number of images",glob%nimage
    write(stdout,1000) "Spring constant",glob%nebk
!    write(stdout,2000) "Functional (may change)",chain%functional
!    write(stdout,2000) "Lagrange parameters are updated every",&
!        lambda_intervall,"steps"
  end if

  write(stdout,*)

  ! ====================================================================
  ! System size
  ! ====================================================================

  if(glob%tatoms) &
      write(stdout,2000) "Number of atoms",glob%nat 
  if(glob%tcoords2) then
    ia3=shape(glob%xcoords2)
  else
    ia3=(/1,1,0/)
  end if
  ! input geometries are coords and coords2
  write(stdout,2000) "Number of input geometries",ia3(3)+1
  if(glob%tcoords2.and.abs(glob%distort)>0.D0) then
    write(stdout,1000) "Distorting start coordinates along coords2 by",&
        glob%distort
  end if

  write(stdout,2000) "Variables to be optimised",glob%nivar
  if (glob%imicroiter /= 0) then
     write(stdout, 2000) "Variables in inner (macroiterative) region",glob%nicore
  end if
  if(glob%dump>0) then
    write(stdout,2000) "Restart information is written every", &
        glob%dump,"steps"
  else
    write(stdout,'(a)') "No restart information is written"
  end if
  if(trestarted) then
    write(stdout,'(a)') "This run has been restarted from files."
  else
    write(stdout,'(a)') "This run has not been restarted."
  end if

  if(stat%ccycle > 0) then
    write(stdout,2000) "Number of energy evaluations on this processor",stat%pene
    if (glob%imicroiter == 0) then
       write(stdout,2000) "Number of steps",stat%ccycle
       write(stdout,2000) "Number of accepted steps / line searches",stat%caccepted
    else
       write(stdout,2000) "Number of macroiterative steps",stat%ccycle
       write(stdout,2000) "Number of accepted macroiterative steps",stat%caccepted
       write(stdout,2000) "Number of microiterative steps",stat%tmiccycle
       write(stdout,2000) "Number of accepted microiterative steps", stat%tmicaccepted
    end if
  else
    write(stdout,2000) "Maximum number of steps",glob%maxcycle
    write(stdout,2000) "Maximum number of energy evaluations",glob%maxene
    if (glob%imicroiter > 0) then
       write(stdout,2000) "Max no. of microiterative steps per macro step", &
            glob%maxmicrocycle
    end if
  end if

111 continue ! skip to here if printl is less than or equal to zero

  if (glob%ntasks > 1) call dlf_mpi_counters()

  if (printl > 0) write(stdout,*)
  call flush(stdout)

  ! real number
  1000 format (t1,'................................................', &
           t1,a,' ',t50,es10.3,1x,a)
  ! integer
  2000 format (t1,'................................................', &
           t1,a,' ',t50,i10,1x,a)
end subroutine dlf_report
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_printheader
!!
!! FUNCTION
!!
!! Print header to stdout
!!
!! SYNOPSIS
subroutine dlf_printheader
!! SOURCE
  use dlf_global, only: stdout
  implicit none
  character(50)   :: svn_v
  call dlf_svnversion(svn_v)
  if(svn_v=="") then
    ! svn not available, use the revision number of this file
    svn_v="$Revision$"
    svn_v= svn_v(11:)
  end if

  write(stdout,'(/a)') &
      "***********************************************************************"      
  write(stdout,'(a)') &
      "**                                                                   **"      
  write(stdout,'(a)') &
      "**                       ---------------------                       **"
  write(stdout,'(a)') &
      "**                              DL-FIND                              **"      
  write(stdout,'(a)') &
      "**                       Geometry Optimisation                       **"      
  write(stdout,'(a)') &
      "**                       ---------------------                       **"
  write(stdout,'(a)') &
      "**                                                                   **"      
  write(stdout,'(a)') &
      "**                 J. Kaestner, J.M. Carr, T.W. Keal,                **"
  write(stdout,'(a)') &
      "**                W. Thiel, A. Wander and P. Sherwood                **"
  write(stdout,'(a)') &
      "**                                                                   **" 
  write(stdout,'(a)') &
      "**              J. Phys. Chem. A, 2009, 113 (43), 11856.             **"
  write(stdout,'(a)') &
      "**                                                                   **" 
  write(stdout,'(a)') &
      "**   Please include this reference in published work using DL-FIND.  **"
  write(stdout,'(a)') &
      "**                                                                   **" 
  write(stdout,'(a)') &
      "**               Copyright:  STFC Daresbury Laboratory               **"      
!                    g  f  e  d  c  b  a  C  a  b  c  d  e  f  g
!  write(stdout,'(a)') &
!      "**                          $Revision$                         **"      
  write(stdout,'("**",27x,"Revision: ",a30,"**")') svn_v
  write(stdout,'(a)') &
      "**                                                                   **"      
  write(stdout,'(a/)') &
      "***********************************************************************"      
  call flush(stdout)
end subroutine dlf_printheader
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_default_init
!!
!! FUNCTION
!!
!! set input parameters to something useless to recognise user input
!! a few parameters are set here, see comments in the code below
!!
!! SYNOPSIS
subroutine dlf_default_init(nspec,spec)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,printf
  implicit none
  integer,intent(in) :: nspec
  integer,intent(out):: spec(nspec)
! **********************************************************************
  printl=-1
  printf=-1
  glob%maxcycle=-1
  glob%maxene=-1
  glob%tolerance=-1.D0
  glob%tolerance_e=-1.D0  
  glob%toldenergy=.false.
  glob%tinit=.false.
  glob%tatoms=.false.
  glob%iopt=-1
  glob%iline=-1
  glob%maxstep=-1.D0
  glob%scalestep=-1.D0
  glob%maxdump=-1
  
  glob%lbfgs_mem=-1
  glob%update=-1
  glob%maxupd=-1
  glob%havehessian=.false.
  glob%delta=-1.D0
  glob%soft=1.D20
  glob%inithessian = -1
  glob%carthessian=-1
  glob%tsrelative=.false.
  glob%minstep=-1.D0
  
  glob%icoord=-1
  glob%nimage=-1
  glob%nebk=-1.D0
  glob%neb_climb_test=-1.D0
  glob%neb_freeze_test=-1.D0
  glob%maxrot=-1
  glob%tolrot=-1.D20
  spec(:)=0 ! for spec, the default is set here (as it may contain
            ! positive and negative values)
  glob%ncons=-1
  glob%nconn=-1
  glob%dump=-1
  glob%restart=-1

  glob%timestep=-1.D0
  glob%fric0=-1.D0
  glob%fricfac=-1.D0
  glob%fricp=-1.D0

  glob%imultistate = 0
  glob%needcoupling = 0
  glob%state_i = 1
  glob%state_j = 2
  glob%pf_c1 = 5.0d0
  glob%pf_c2 = 5.0d0
  glob%gp_c3 = 1.0d0
  glob%gp_c4 = 0.9d0
  glob%ln_t1 = 1.0d-4
  glob%ln_t2 = 1.0d0

  glob%distort = 0.D0 ! default given here

  glob%task = -1
  
  glob%temperature = -1.D0

  glob%po_pop_size= -1
  glob%po_radius_base= -1.0D0
  glob%po_contraction= -1.0D0
  glob%po_tol_r_base= -1.0D0
  glob%po_tolerance_g= -1.0D0
  glob%po_distribution= -1
  glob%po_maxcycle= -1
  glob%po_init_pop_size= -1
  glob%po_reset= -1
  glob%po_mutation_rate= -1.0D0
  glob%po_death_rate= -1.0D0
  glob%po_scalefac= -1.0D0
  glob%po_nsave= -1

  glob%ntasks= -1 ! why???  no real need for this not to be 1...

  glob%nzero=-1

  glob%qtsflag=-1

  glob%maxmicrocycle = -1
  glob%micro_esp_fit = .false.

end subroutine dlf_default_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* main/dlf_default_set
!!
!! FUNCTION
!!
!! set parameters that have not been set by the user to the default
!! this routine defines the default values
!!
!! SYNOPSIS
subroutine dlf_default_set(nvarin)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,printf,stdout
  implicit none
  integer,   intent(in)  :: nvarin
! **********************************************************************
  if(printl < 0) printl=2
  if(printf<0) printf=2
  if(glob%maxcycle < 0) glob%maxcycle=100
  if(glob%maxene < 0) glob%maxene=100000

  if(glob%tolerance < 0.D0) glob%tolerance=4.5D-4
  if(glob%tolerance_e < 0.D0) glob%tolerance_e= glob%tolerance/ 450.D0
  if(glob%iopt < 0) glob%iopt=3 ! L-BFGS

  if(glob%iline < 0) then
    if(glob%iopt < 3 .or. glob%imultistate == 2) then
      glob%iline=2
    else
      glob%iline=0
      if(glob%iopt==3.and.glob%icoord<10) glob%iline=1
    end if
  end if
  if(glob%maxstep < 0.D0) glob%maxstep=0.5D0
  if(glob%scalestep < 0.D0) then
    !if(glob%iopt==3) then
      glob%scalestep=1.D0
    !else
    !  glob%scalestep=0.2D0
    !end if
  end if
  if(glob%lbfgs_mem<0) then
    glob%lbfgs_mem=max(nvarin,5)
    glob%lbfgs_mem=min(nvarin,50)
  end if
  if(glob%update<0) glob%update=2
  if(glob%maxupd<0) glob%maxupd=50
  if(glob%delta<0.D0) then
    glob%delta=0.01D0
    if(glob%icoord==190) then
      glob%delta=0.4D0
    end if
  end if
  if(glob%inithessian == -1) glob%inithessian = 0
  if(glob%carthessian==-1) glob%carthessian=0
  if(glob%minstep < 0.D0) glob%minstep = 1.D-5
  if(glob%maxdump < 0 ) glob%maxdump=100000

  if(glob%nimage<0) then
    if(glob%icoord>=100.and.glob%icoord<200) then
      !NEB
      glob%nimage=10
    else
      glob%nimage=1
    end if
  end if
  if(glob%icoord<0) then
    if(glob%nimage==1) then
      glob%icoord=0
    else
      glob%icoord=110 ! NEB with endpoints perpendicular to tau
    end if
  end if
  if(abs(glob%soft)>1.D19) then
    if ((glob%icoord >= 3 .and. glob%icoord <= 4) .or. &
        (glob%icoord >= 13 .and. glob%icoord <= 14)) then
      ! in case of internals only, there is nothing like soft modes
      glob%soft=-1.D0
    else
      glob%soft=5.D-3
    end if
  end if
  if(glob%nebk<0.D0) then
    glob%nebk=0.01D0
    if(glob%icoord==190) then
      glob%nebk=0.D0
    end if
  end if
  if(glob%neb_climb_test < 0.D0) glob%neb_climb_test = 3.0D0
  if(glob%neb_freeze_test < 0.D0) glob%neb_freeze_test = 1.0D0  

  if(glob%maxrot<0) glob%maxrot=10
  if(glob%tolrot<-1.D19) glob%tolrot=5.D0

  ! for "spec" the default is set in dlf_default_init!
  if(glob%ncons<0) glob%ncons=0
  if(glob%nconn<0) glob%nconn=0
  if(glob%dump<0) glob%dump=0
  if(glob%restart<0) glob%restart=0
  ! glob%weight and mass defaults are set in dlf_allocate_glob

  if(glob%timestep <=0.D0) glob%timestep= 1.D0
  if(glob%fric0 <=0.D0) glob%fric0= 0.3D0
  if(glob%fricfac <=0.D0) glob%fricfac=0.95D0
  if(glob%fricp <=0.D0) glob%fricp=0.3D0

  ! Conical intersection search parameters
  ! Note that <= 0 is allowed for ln_t1/ln_t2
  if (glob%pf_c1 .le. 0.D0) glob%pf_c1 = 5.0d0
  if (glob%pf_c2 .le. 0.D0) glob%pf_c2 = 5.0d0
  if (glob%gp_c3 .le. 0.D0) glob%gp_c3 = 1.0d0
  if (glob%gp_c4 .le. 0.D0) glob%gp_c4 = 0.9d0

  ! the default for glob%distort is in dlf_default_init

  if (glob%task < 0) glob%task = 0

  if(glob%temperature < 0.D0) glob%temperature=300.D0

  if (glob%po_pop_size < 1) glob%po_pop_size = 25
  if (glob%po_radius_base <= 0.0D0) glob%po_radius_base = 1.0D0
  if (glob%po_contraction <= 0.0D0) glob%po_contraction = 0.9D0
  if (glob%po_tol_r_base <= 0.0D0) glob%po_tol_r_base = 1.0D-8
  if (glob%po_tolerance_g <= 0.0D0) glob%po_tolerance_g = 1.0D-3
  if (glob%po_distribution < 0) glob%po_distribution = 3 ! force_bias
  if (glob%po_maxcycle <= 0) glob%po_maxcycle = 10000
  if (glob%po_init_pop_size < 1) glob%po_init_pop_size = 2*glob%po_pop_size
  if (glob%po_reset < 1) glob%po_reset = 500
  if (glob%po_mutation_rate < 0.0D0) glob%po_mutation_rate = 0.15D0
  if (glob%po_death_rate < 0.0D0) glob%po_death_rate = 0.5D0
  if (glob%po_scalefac <= 0.0D0) glob%po_scalefac = 10.0D0
  if (glob%po_nsave < 0) glob%po_nsave = 10

  if (glob%ntasks <= 0) glob%ntasks = 1

  ! Change any parameters that are incompatible with parallel optimisation
  if (glob%iopt/10==5) then
     !if (glob%icoord/=0) then ! change this in future!!!
     !   write(stdout,'(1x,a)') "Warning: parallel optimisation incompatible &
     !   &with icoord /= 0 currently; switching to Cartesians"
     !   glob%icoord=0
     !end if
     if (glob%iline/=0) then
        write(stdout,'(1x,a)') "Warning: parallel optimisation incompatible &
        &with iline /= 0; setting to zero" 
        glob%iline=0
     end if
  end if

  ! In principle could calculate the correct value for nzero
  ! based on molecular geometry and frozen atom information.
  ! For now safest to let the user switch it on manually.
  if(glob%nzero<0) glob%nzero = 0

  if(glob%qtsflag==-1) glob%qtsflag=0

  if(glob%maxmicrocycle < 0) glob%maxmicrocycle = 100

end subroutine dlf_default_set
!!****
