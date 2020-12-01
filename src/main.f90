! 
!************************************************************************
!                              QUICK                                   **
!                                                                      **
!                        Copyright (c) 2010                            **
!                Regents of the University of Florida                  **
!                       All Rights Reserved.                           **
!                                                                      **
!  This software provided pursuant to a license agreement containing   **
!  restrictions on its disclosure, duplication, and use. This software **
!  contains confidential and proprietary information, and may not be   **
!  extracted or distributed, in whole or in part, for any purpose      **
!  whatsoever, without the express written permission of the authors.  **
!  This notice, and the associated author list, must be attached to    **
!  all copies, or extracts, of this software. Any additional           **
!  restrictions set forth in the license agreement also apply to this  **
!  software.                                                           **
!************************************************************************
!
!  Cite this work as:
!  Miao,Y.: He, X.: Ayers,K; Brothers, E.: Merz,K. M. QUICK
!  University of Florida, Gainesville, FL, 2010
!************************************************************************

    program quick
    
    use allMod
    use divPB_Private, only: initialize_DivPBVars

    implicit none

#ifdef MPIV
    include 'mpif.h'
#endif

#ifdef CUDA 
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
    call initialize1(ierr)    
    !-------------------MPI/MASTER---------------------------------------
    masterwork_readInput: if (master) then

      ! read input argument
      call set_quick_files(ierr)    ! from quick_file_module

      ! open output file
      call quick_open(iOutFile,outFileName,'U','F','R',.false.)

      ! At the beginning of output file, copyright information will be output first 
      call outputCopyright(iOutFile,ierr)
      
      ! Then output file information
      call PrtDate(iOutFile,'TASK STARTS ON:')
      call print_quick_io_file(iOutFile,ierr) ! from quick_file_module

      ! check MPI setup and output info
      call check_quick_mpi(iOutFile,ierr)   ! from quick_mpi_module
      
#ifdef MPIV      
      if (bMPI) call print_quick_mpi(iOutFile,ierr)   ! from quick_mpi_module
#endif
    
    endif masterwork_readInput
    !--------------------End MPI/MASTER----------------------------------

#ifdef CUDA 

    !------------------- CUDA -------------------------------------------
    ! startup cuda device
    call gpu_startup()
#ifdef __PGI
    iarg = COMMAND_ARGUMENT_COUNT()
#else
    iarg = iargc()
#endif
    
    call gpu_set_device(-1)

    ! Handles an old mechanism where the user can specify GPU id from CLI
    if (iarg .ne. 0) then
        do i = 1, iarg
            call getarg(int(i,4), arg)
            if (arg.eq."-gpu") then
                call getarg (int(i+1,4), arg)
                read(arg, '(I2)') gpu_device_id
                call gpu_set_device(gpu_device_id)
                write(*,*) "read -gpu from argument=",gpu_device_id
                exit
            endif
        enddo
    endif

    call gpu_init()
 
    ! write cuda information
    call gpu_write_info(iOutFile)
    !------------------- END CUDA ---------------------------------------
#endif

#ifdef CUDA_MPIV

    call mgpu_query(mpirank)

    call mgpu_setup()

    if(master) call mgpu_write_info(iOutFile)
    
    call mgpu_init(mpirank, mpisize, mgpu_id)

#endif



    !------------------------------------------------------------------
    ! 2. Next step is to read job and initial guess
    !------------------------------------------------------------------

    !read job spec and mol spec
    call read_Job_and_Atom()
    !allocate essential variables
    call alloc(quick_molspec)
    if (quick_method%MFCC) call allocate_MFCC()
    
    ! Then do inital guess
    call cpu_time(timer_begin%TIniGuess)
    
    ! a. SAD intial guess
    if (quick_method%SAD) call getMolSad()

    ! b. MFCC initial guess
    if (quick_method%MFCC) then
!       call mfcc
!       call getmolmfcc
    endif
    
    !------------------------------------------------------------------
    ! 3. Read Molecule Structure
    !-----------------------------------------------------------------
    call getMol()

#if defined CUDA || defined CUDA_MPIV
    call gpu_setup(natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &
                   quick_molspec%molchg, quick_molspec%iAtomType)
    call gpu_upload_xyz(xyz)
    call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
#endif

    !------------------------------------------------------------------
    ! 4. SCF single point calculation. DFT if wanted. If it is OPT job
    !    ignore this part and go to opt part. We will get variationally determined Energy.
    !-----------------------------------------------------------------

    ! if it is div&con method, begin fragmetation step, initial and setup
    ! div&con varibles
    if (quick_method%DIVCON) call inidivcon(quick_molspec%natom)

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
        call g2eshell   ! pre-calculate 2 indices coeffecient to save time
        call schwarzoff ! pre-calculate schwarz cutoff criteria
    endif

#if defined CUDA || defined CUDA_MPIV    
    call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
    ncontract, itype, aexp, dcoeff, &
    quick_basis%first_basis_function, quick_basis%last_basis_function, & 
    quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
    quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
    quick_basis%ktype, quick_basis%kprim, quick_basis%kshell,quick_basis%Ksumtype, &
    quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, quick_basis%Qsbasis, quick_basis%Qfbasis, &
    quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)
   
    call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)
#endif

    call cpu_time(timer_end%TIniGuess)
    timer_cumer%TIniGuess=timer_cumer%TIniGuess+timer_end%TIniGuess-timer_begin%TIniGuess

    if (.not.quick_method%opt .and. .not.quick_method%grad) then
        call getEnergy(failed, .false.)
    endif

    if (failed) call quick_exit(iOutFile,1)


    !------------------------------------------------------------------
    ! 5. OPT Geometry if wanted
    !-----------------------------------------------------------------

    ! Geometry optimization. Currently, only cartesian version is 
    ! available. A improvement is in optimzenew, which is based on 
    ! internal coordinates, but is under coding.    
    if (quick_method%opt)  call optimize(failed)     ! Cartesian 
    if (.not.quick_method%opt .and. quick_method%grad) call gradient(failed)                             
    if (failed) call quick_exit(iOutFile,1)          ! If geometry optimization fails

    ! Now at this point we have an energy and a geometry.  If this is
    ! an optimization job, we now have the optimized geometry.


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
        if(.not. quick_method%DIVCON) then
#ifdef MPIV
           if (bMPI) then
             call mpi_calmp2    ! MPI-MP2
           else
#endif
             call calmp2()      ! none-MPI MP2
#ifdef MPIV
           endif
#endif
        else
            call calmp2divcon   ! DIV&CON MP2
        endif
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
#ifdef CUDA 
    if (master) then
       call gpu_shutdown()
    endif
#endif

#ifdef CUDA_MPIV
    call delete_mgpu_setup()
    call mgpu_shutdown()
#endif

    call finalize(iOutFile,0)
    
    end program quick
