!
!---------------------------------------------------------------------!
!                           QUICK                                     !
!                                                                     !
!                      Copyright (c) 2022                             !
!       Regents of the University of California San Diego             !
!                  & Michigan State University                        !
!                      All Rights Reserved.                           !
!                                                                     ! 
! Copyright (C) 2022-2023 Merz lab                                    !
! Copyright (C) 2022-2023 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains all one electron integral (oei) & oei gradient ! 
! code.                                                               !
!---------------------------------------------------------------------!

#include "util.fh"

    program quick

    use allMod
    use divPB_Private, only: initialize_DivPBVars
    use quick_cutoff_module, only: schwarzoff
    use quick_exception_module
    use quick_cshell_eri_module, only: getEriPrecomputables
    use quick_cshell_gradient_module, only: cshell_gradient
    use quick_oshell_gradient_module, only: oshell_gradient
    use quick_optimizer_module
    use quick_sad_guess_module, only: getSadGuess
    use quick_molden_module, only : quick_molden, initializeExport, exportCoordinates, exportBasis, &
         exportMO, exportSCF, exportOPT
#ifdef MPIV
    use mpi
#endif

    implicit none

#if defined CUDA || defined HIP
    integer :: gpu_device_id = -1
#endif

    integer*4 :: iarg
    character(80) :: arg
    logical :: failed = .false.         ! flag to indicates SCF fail or OPT fail
    integer :: ierr                     ! return error info
    integer :: i,j,k
    double precision t1_t, t2_t
    common /timer/ t1_t, t2_t
    !------------------------------------------------------------------
    ! 1. The first thing that must be done is to initialize and prepare files
    !------------------------------------------------------------------
    ! Initial neccessary variables
    ierr=0
    SAFE_CALL(initialize1(ierr))
    !-------------------MPI/MASTER---------------------------------------
    masterwork_readInput: if (master) then

      ! read input argument
      call set_quick_files(.false.,ierr)    ! from quick_file_module
      CHECK_ERROR(ierr)

      ! open output file
      call quick_open(iOutFile,outFileName,'U','F','R',.false.,ierr)
      CHECK_ERROR(ierr)

      ! At the beginning of output file, copyright information will be output first
      SAFE_CALL(outputCopyright(iOutFile,ierr))

      ! Then output file information
      SAFE_CALL(PrtDate(iOutFile,'TASK STARTS ON:',ierr))
      call print_quick_io_file(iOutFile,ierr) ! from quick_file_module

      ! check MPI setup and output info
      !call check_quick_mpi(iOutFile,ierr)   ! from quick_mpi_module

#ifdef MPIV
      if (bMPI) call print_quick_mpi(iOutFile,ierr)   ! from quick_mpi_module
#endif

    endif masterwork_readInput
    !--------------------End MPI/MASTER----------------------------------

#if defined CUDA || defined HIP

    !------------------- CUDA -------------------------------------------
    ! startup cuda device
    SAFE_CALL(gpu_startup(ierr))
#ifdef __PGI
    iarg = COMMAND_ARGUMENT_COUNT()
#else
    iarg = iargc()
#endif

    SAFE_CALL(gpu_set_device(-1, ierr))

    ! Handles an old mechanism where the user can specify GPU id from CLI
    if (iarg .ne. 0) then
        do i = 1, iarg
            call getarg(int(i,4), arg)
            if (arg.eq."-gpu") then
                call getarg (int(i+1,4), arg)
                read(arg, '(I2)') gpu_device_id
                SAFE_CALL(gpu_set_device(gpu_device_id, ierr))
                write(*,*) "read -gpu from argument=",gpu_device_id
                exit
            endif
        enddo
    endif

    SAFE_CALL(gpu_init(ierr))

    ! write cuda information
    SAFE_CALL(gpu_write_info(iOutFile, ierr))
    !------------------- END CUDA ---------------------------------------
#endif

#if defined CUDA_MPIV || defined HIP_MPIV

    SAFE_CALL(mgpu_query(mpisize ,mpirank, mgpu_id, ierr))

    SAFE_CALL(mgpu_setup(ierr))

    if(master) SAFE_CALL(mgpu_write_info(iOutFile, mpisize, mgpu_ids, ierr))
    
    SAFE_CALL(mgpu_init(mpirank, mpisize, mgpu_id, ierr))

#endif


    !------------------------------------------------------------------
    ! 2. Next step is to read job and initial guess
    !------------------------------------------------------------------

    !read job spec and mol spec
    call read_Job_and_Atom(ierr)
    !allocate essential variables
    call alloc(quick_molspec,ierr)
    !if (quick_method%MFCC) call allocate_MFCC()
   
    RECORD_TIME(timer_end%TInitialize)
    timer_cumer%TInitialize = timer_cumer%TInitialize + timer_end%TInitialize - timer_begin%TInitialize
  
    ! Then do inital guess
    RECORD_TIME(timer_begin%TIniGuess)

    ! a. SAD intial guess
    if (quick_method%SAD) SAFE_CALL(getSadGuess(ierr))
    if (quick_method%writeSAD) then
       call quick_exit(iOutFile,ierr)
    end if

    ! b. MFCC initial guess
    !if (quick_method%MFCC) then
    !    call mfcc
    !    call getmolmfcc
    !endif

    !------------------------------------------------------------------
    ! 3. Read Molecule Structure
    !-----------------------------------------------------------------
    SAFE_CALL(getMol(ierr))

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

    call gpu_allocate_scratch(quick_method%grad .or. quick_method%opt)

    call upload(quick_method, ierr)

    if(.not.quick_method%opt)then
      call gpu_setup(natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &
                     quick_molspec%molchg, quick_molspec%iAtomType)
      call gpu_upload_xyz(xyz)
      call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
    endif
#endif
    
    ! Molden export
    ! initialize exporting
    if(write_molden .and. master) then
       call initializeExport(quick_molden, ierr)
    endif

    
    !------------------------------------------------------------------
    ! 4. SCF single point calculation. DFT if wanted. If it is OPT job
    !    ignore this part and go to opt part. We will get variationally determined Energy.
    !-----------------------------------------------------------------

    ! if it is div&con method, begin fragmetation step, initial and setup
    ! div&con varibles
    !if (quick_method%DIVCON) call inidivcon(quick_molspec%natom)

    ! if it is not opt job, begin single point calculation
    if(.not.quick_method%opt)then
!      if(.NOT.PBSOL)then
!        call getEnergy(failed)
!      else
!        HF=.true.
!        DFT=.false.
!        call getEnergy(failed)
!      endif
!   else
        call getEriPrecomputables   ! pre-calculate 2 indices coeffecient to save time
        call schwarzoff ! pre-calculate schwarz cutoff criteria
    endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
    if(.not.quick_method%opt)then
      call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
      ncontract, itype, aexp, dcoeff, &
      quick_basis%first_basis_function, quick_basis%last_basis_function, &
      quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
      quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
      quick_basis%ktype, quick_basis%kprim, quick_basis%kshell,quick_basis%Ksumtype, &
      quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, quick_basis%Qsbasis, quick_basis%Qfbasis, &
      quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)
 
      call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)

      call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, quick_molspec%extchg, ierr)

    endif
#endif

#if defined CUDA_MPIV || defined HIP_MPIV
    timer_begin%T2elb = timer_end%T2elb
    call mgpu_get_2elb_time(timer_end%T2elb)
    timer_cumer%T2elb = timer_cumer%T2elb+timer_end%T2elb-timer_begin%T2elb
#endif

    RECORD_TIME(timer_end%TIniGuess)
    timer_cumer%TIniGuess=timer_cumer%TIniGuess+timer_end%TIniGuess-timer_begin%TIniGuess &
                          -(timer_end%T2elb-timer_begin%T2elb)

    if (.not.quick_method%opt .and. .not.quick_method%grad) then
        SAFE_CALL(getEnergy(.false.,ierr))
        
    endif

    !------------------------------------------------------------------
    ! 5. OPT Geometry if wanted
    !-----------------------------------------------------------------

    ! Geometry optimization. Currently, only cartesian version is
    ! available. A improvement is in optimzenew, which is based on
    ! internal coordinates, but is under coding.
    if (quick_method%opt) then
        if (quick_method%usedlfind) then

#ifdef MPIV
            SAFE_CALL(dl_find(ierr, master))   ! DLC
#else
            SAFE_CALL(dl_find(ierr, .true.)) 
#endif

        else
            SAFE_CALL(lopt(ierr))         ! Cartesian
        endif
    endif
    
    if (.not.quick_method%opt .and. quick_method%grad) then
        if (quick_method%UNRST) then
            SAFE_CALL(oshell_gradient(ierr))
        else
            SAFE_CALL(cshell_gradient(ierr))
        endif
    endif

    ! Now at this point we have an energy and a geometry.  If this is
    ! an optimization job, we now have the optimized geometry.

    ! Molden output
    if(write_molden .and. master) then
       call exportCoordinates(quick_molden, ierr)
       call exportBasis(quick_molden, ierr)
       call exportMO(quick_molden, ierr)
       if (quick_method%opt) then
          call exportSCF(quick_molden, ierr)
          call exportOPT(quick_molden, ierr)
       end if
    endif



    !------------------------------------------------------------------
    ! 6. Other job options
    !-----------------------------------------------------------------

    ! 6.a PB Solvent Model
    ! 11/03/2010 Blocked by Yiao Miao
!   if (PBSOL) then
!       call initialize_DivPBVars()
!       call pPBDriver(ierror)
!   endif

    ! 6.b MP2,2nd order Møller–Plesset perturbation theory
    if(quick_method%MP2) then
    !    if(.not. quick_method%DIVCON) then
#ifdef MPIV
           if (master) then
!             call mpi_calmp2    ! MPI-MP2
!           else
#endif
             call calmp2()      ! none-MPI MP2
#ifdef MPIV
           endif
#endif
    !    else
    !        call calmp2divcon   ! DIV&CON MP2
    !    endif
    endif   !(quick_method%MP2)

    ! 6.c Freqency calculation and mode analysis
    ! note the analytical calculation is broken and needs to be fixed
    if (quick_method%freq) then
        call calcHessian(failed)
        if (failed) call quick_exit(iOutFile,1)     ! If Hessian matrix fails
        call frequency
    endif

    ! 6.d clean spin for unrestricted calculation
    ! If this is an unrestricted calculation, check out the S^2 value to
    ! see if this is a reasonable wave function.  If not, modify it.

!    if (quick_method%unrst) then
!        if (quick_method%debug) call debugCleanSpin
!        if (quick_method%unrst) call spinclean
!        if (quick_method%debug) call debugCleanSpin
!    endif

    if (master) then

        ! Convert Cartesian coordinator to internal coordinator
        if (quick_method%zmat) call zmake

        ! Calculate Dipole Moment
        if (quick_method%dipole) call dipole

    endif

    ! Now at this point we have an energy and a geometry.  If this is
    ! an optimization job, we now have the optimized geometry.

    !-----------------------------------------------------------------
    ! 7.The final job is to output energy and many other infos
    !-----------------------------------------------------------------
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
    call delete(quick_method,ierr)
#endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  call gpu_deallocate_scratch(quick_method%grad .or. quick_method%opt)
#endif


#if defined CUDA || defined HIP
    if (master) then
       SAFE_CALL(gpu_shutdown(ierr))
    endif
#endif

#if defined CUDA_MPIV || defined HIP_MPIV
    SAFE_CALL(delete_mgpu_setup(ierr))
    SAFE_CALL(mgpu_shutdown(ierr))
#endif

    call finalize(iOutFile,ierr,0)

    end program quick
