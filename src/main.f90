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

    use divPB_Private, only: initialize_DivPBVars
    use quick_cutoff_module, only: schwarzoff
    use quick_exception_module
    use quick_eri_cshell_module, only: getEriPrecomputables
    use quick_grad_cshell_module, only: cshell_gradient
    use quick_grad_oshell_module, only: oshell_gradient
    use quick_oeproperties_module, only: compute_oeprop
    use quick_optimizer_module
    use quick_sad_guess_module, only: getSadGuess
    use quick_molden_module, only : quick_molden, initializeExport, exportCoordinates, exportBasis, &
         exportMO, exportSCF, exportOPT
#if defined(RESTART_HDF5)
    use quick_restart_module, only: data_write_info, write_integer_array, write_double_array
#endif
    use quick_timer_module, only : timer_end, timer_cumer, timer_begin
    use quick_method_module, only : quick_method
    use quick_files_module, only: ioutfile, outFileName, iDataFile, dataFileName
    use quick_mpi_module, only: master, bMPI, print_quick_mpi, mpirank
    use quick_molspec_module, only: quick_molspec, natom, alloc
    use quick_files_module, only: write_molden, set_quick_files, print_quick_io_file
    use quick_molsurface_module, only: generate_MKS_surfaces
#ifdef MPIV
    use mpi
#endif
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
    use quick_basis_module, only: quick_basis, aexp, cutprim, dcoeff, itype
    use quick_basis_module, only: jbasis, jshell, maxcontract, nbasis, ncontract
    use quick_basis_module, only: nprim, nshell, Ycutoff
    use quick_molspec_module, only : xyz
    use quick_method_module, only: delete, upload
#endif

#if defined CUDA_MPIV || defined HIP_MPIV
    use quick_mpi_module, only: mpisize, mgpu_id, mgpu_ids
#endif

    implicit none

    logical :: failed = .false.         ! flag to indicates SCF fail or OPT fail
    integer :: ierr                     ! return error info
    integer :: i,j,k
    double precision :: t1_t, t2_t
    common /timer/ t1_t, t2_t

    !------------------------------------------------------------------
    ! 1. The first thing that must be done is to initialize and prepare files
    !------------------------------------------------------------------
    ierr = 0
    SAFE_CALL(initialize1(ierr))

    masterwork_readInput: if (master) then
      call set_quick_files(.false., ierr)
      CHECK_ERROR(ierr)

      call quick_open(iOutFile, outFileName, 'U', 'F', 'R', .false., ierr)
      CHECK_ERROR(ierr)

      SAFE_CALL(outputCopyright(iOutFile,ierr))

      SAFE_CALL(PrtDate(iOutFile, 'TASK STARTS ON:', ierr))
      call print_quick_io_file(iOutFile, ierr)

      !call check_quick_mpi(iOutFile, ierr)

#ifdef MPIV
      if (bMPI) call print_quick_mpi(iOutFile, ierr)
#endif
    endif masterwork_readInput

#if defined(GPU)
    SAFE_CALL(gpu_new(ierr))
    SAFE_CALL(gpu_init_device(ierr))
    SAFE_CALL(gpu_write_info(iOutFile, ierr))
#elif defined(MPIV_GPU)
    SAFE_CALL(gpu_new(mpirank, ierr))
    SAFE_CALL(mgpu_query(mpisize, mpirank, mgpu_id, ierr))
    SAFE_CALL(mgpu_setup(ierr))
    if (master) SAFE_CALL(mgpu_write_info(iOutFile, mpisize, mgpu_ids, ierr))
    SAFE_CALL(mgpu_init_device(mpirank, mpisize, mgpu_id, ierr))
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

#if defined(RESTART_HDF5)
    !write the required info to data file
    if(master .and. (quick_method%writeden .or. quick_method%writexyz)) call data_write_info(natom, quick_molspec%nbasis)
#endif

#if defined(GPU) || defined(MPIV_GPU)
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
    ! div&con variables
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

#if defined(GPU) || defined(MPIV_GPU)
    if (.not.quick_method%opt) then
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

#if defined(MPIV_GPU)
    timer_begin%T2elb = timer_end%T2elb
    call mgpu_get_2elb_time(timer_end%T2elb)
    timer_cumer%T2elb = timer_cumer%T2elb+timer_end%T2elb-timer_begin%T2elb
#endif

    RECORD_TIME(timer_end%TIniGuess)
    timer_cumer%TIniGuess=timer_cumer%TIniGuess+timer_end%TIniGuess-timer_begin%TIniGuess &
                          -(timer_end%T2elb-timer_begin%T2elb)

    if (.not.quick_method%opt .and. .not.quick_method%grad) then
        SAFE_CALL(getEnergy(.false.,ierr))
        
        ! One electron properties (ESP, EField)
        call compute_oeprop()

#if defined(RESTART_HDF5)
        if(master) then
          if(quick_method%writexyz)then
             call write_integer_array(quick_molspec%iattype, natom, 'iattype')
             call write_double_array(quick_molspec%xyz, 3, natom, 'xyz')
          endif 
        endif
#endif
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
#if defined(RESTART_HDF5)
        if(master) then
          if(quick_method%writexyz)then
             call write_integer_array(quick_molspec%iattype, natom, 'iattype')
             call write_double_array(quick_molspec%xyz, 3, natom, 'xyz')
          endif 
        endif
#endif
    endif
    
    if (.not.quick_method%opt .and. quick_method%grad) then
        if (quick_method%UNRST) then
            SAFE_CALL(oshell_gradient(ierr))
        else
            SAFE_CALL(cshell_gradient(ierr))
        endif

        ! One electron properties (ESP, EField) 
        call compute_oeprop()

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
#if defined(GPU) || defined(MPIV_GPU)
    call delete(quick_method, ierr)
    call gpu_deallocate_scratch(quick_method%grad .or. quick_method%opt)
#if defined(MPIV_GPU)
    SAFE_CALL(delete_mgpu_setup(ierr))
#endif
    SAFE_CALL(gpu_delete(ierr))
#endif

    call finalize(iOutFile,ierr,0)

    end program quick
