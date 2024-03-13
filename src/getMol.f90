!
!	getMol.f90
!	new_quick
!        
!       Updated by Madu Manathunga on 04/20/2019
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

subroutine getMol(ierr)
   ! This subroutine is to get molecule information
   ! and assign basis function.
   use allmod
   use quick_gridpoints_module
   use quick_exception_module
#ifdef MPIV
   use mpi
#endif

   implicit none

   logical :: present
   integer :: i,j,k,itemp
   integer, intent(inout) :: ierr

   !-----------MPI/MASTER------------------------
   if (master) then

      ! Read info from AMBER
      call PrtAct(iOutfile,"Begin Reading Molecular Information")

      ! read xyz coordinates from the .in file 
      if(.not. isTemplate) then
       call quick_open(infile,inFileName,'O','F','W',.true.,ierr)
       CHECK_ERROR(ierr)
        ! read molecule coordinates
        call read2(quick_molspec,inFile,ierr)
        close(inFile)
      endif

      quick_molspec%nbasis   => nbasis
      quick_qm_struct%nbasis => nbasis
      call set(quick_molspec,ierr)

      ! quick forward coordinates stored in namelist to instant variables
      xyz(1:3,1:natom)=quick_molspec%xyz(1:3,1:natom)

      ! check the correctness between molecular specification and method used
      call check_quick_method_and_molspec(iOutFile,quick_molspec,quick_method,ierr)
      CHECK_ERROR(ierr)

   endif
   !-----------END MPI/MASTER-----------------------
#ifdef MPIV
   !-----------MPI/ALL NODES------------------------
   if (bMPI)  call mpi_setup_mol1()
   !-----------END MPI/ALL NODES--------------------
#endif

   ! At this point we have the positions and identities of the atoms. We also
   ! have the number of electrons. Now we must assign basis functions. This
   ! is done in a subroutine.
   call readbasis(natom,0,0,0,0,ierr)
   ! F implementation of GPU ERI code currently doesnt support open shell
   ! gradient calculations
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
   if(quick_method%hasF .and. quick_method%unrst .and. quick_method%grad) then
       ierr=39
       return
   endif
#endif
   CHECK_ERROR(ierr)

   quick_molspec%nbasis   => nbasis
   quick_qm_struct%nbasis => nbasis

   call alloc(quick_basis)
   call alloc(quick_qm_struct)
   call init(quick_qm_struct)



   !-----------MPI/MASTER------------------------
   if (master) then
      ! now print molecule specification to output file
      call print(quick_molspec,iOutFile,ierr)
      call print(quick_basis,iOutFile)

      ! the following some step are setup for basis and for ECP or DFT calculation
      ! and see comments for details
      ! Include the normalization constant in the coefficient.
      call normalize_basis()

      ! Store the normalized primitives coefficients for ECP calculations
      if (quick_method%ecp) call store_basis_to_ecp()      ! Alessandro GENONI 03/21/2007

   endif !Madu

#ifdef MPIV
   if (bMPI) call mpi_setup_mol2()
#endif
!--------------Madu----------------

   if (master) then
      ! Read params for semi-emipeircal DFT
      ! if (quick_method%SEDFT) call read_sedft_parm

      !  initialize density matrix
      SAFE_CALL(initialGuess(ierr))

      call PrtAct(iOutfile,"End Reading Molecular Information ")
   endif
   !-----------END MPI/MASTER------------------------

   return

end subroutine getmol


!--------------------
! check mol spec and method
!--------------------
subroutine check_quick_method_and_molspec(io,quick_molspec_arg,quick_method_arg,ierr)
   use quick_method_module
   use quick_molspec_module
   use quick_exception_module

   implicit none

   ! Argument
   integer, intent(inout) :: ierr
   integer io      ! io file to write warning
   type(quick_molspec_type) quick_molspec_arg  ! molspec type
   type(quick_method_type) quick_method_arg    ! method type

   ! inner variables
   integer i

   ! Check for errors.
   if (natom == 1 .and. quick_method_arg%opt) then
      quick_method_arg%opt=.false.
   endif

   ! Define the number of optimization cycles if not set. default is 9*natom
   if (quick_method_arg%iopt == 0 .and. quick_method_arg%opt) then
      quick_method_arg%iopt=3*3*quick_molspec_arg%natom
   endif

   ! if this is a core approximatation calculation, modify the atomic
   ! charges and the number of electrons.
   if (quick_method_arg%core) then
      do i=1,quick_molspec_arg%natom
         if (quick_molspec_arg%iattype(i) >= 3) then
            quick_molspec_arg%chg(i)=quick_molspec_arg%chg(i)-2.d0
            quick_molspec_arg%nelec=quick_molspec_arg%nelec-2
         endif
         if (quick_molspec_arg%iattype(i) >= 11) then
            quick_molspec_arg%chg(i)=quick_molspec_arg%chg(i)-8.d0
            quick_molspec_arg%nelec=quick_molspec_arg%nelec-8
         endif
         if (quick_molspec_arg%iattype(i) >= 18) &
            ierr=18
      enddo
   endif


   ! check the correctness between elections, multiplicity and unrestricted calculation
   ! request. And try to correct possible error
   i = mod(dble(quick_molspec_arg%nelec),2.d0)
   if (i.ne.0 .and. .not. quick_method_arg%unrst) then
      ierr=19
      quick_method_arg%UNRST=.true.
      return
   endif

   if (quick_molspec_arg%imult.ne.1 .and. .not. quick_method_arg%unrst) then
      ierr=20
      return
   endif

   if (quick_method_arg%unrst) then
      quick_molspec_arg%nelecb = quick_molspec_arg%nelec
      quick_molspec_arg%nelec = 0

      do WHILE (quick_molspec_arg%nelec.lt.quick_molspec_arg%nelecb)
         quick_molspec_arg%nelecb = quick_molspec_arg%nelecb-1
         quick_molspec_arg%nelec = quick_molspec_arg%nelec +1
      enddo

      if (quick_molspec_arg%imult .eq. 2 .and. quick_molspec_arg%nelec-1 .ne. quick_molspec_arg%nelecb) then
         ierr=21
         return
      endif

      if (quick_molspec_arg%imult .eq. 3) then
         quick_molspec_arg%nelec = quick_molspec_arg%nelec+1
         quick_molspec_arg%nelecb = quick_molspec_arg%nelecb - 1
         if (quick_molspec_arg%nelec-2 .ne. quick_molspec_arg%nelecb) then
           ierr=22
           return
         endif
      endif

      if (quick_molspec_arg%imult .eq. 4) then
         quick_molspec_arg%nelec = quick_molspec_arg%nelec+1
         quick_molspec_arg%nelecb = quick_molspec_arg%nelecb - 1
         if (quick_molspec_arg%nelec-3 .ne. quick_molspec_arg%nelecb) then
           ierr=23
           return
         endif
      endif
   endif

end subroutine check_quick_method_and_molspec


!--------------------------------------
! Initial Densitry Matrix
!--------------------------------------
subroutine initialGuess(ierr)
   use allmod
   use quick_sad_guess_module, only: getSadDense 
   implicit none
   logical :: present
   integer :: failed
   character(len=80) :: keyWD
   integer n,sadAtom
   integer Iatm,i,j
   double precision temp
   integer, intent(inout) :: ierr



   ! Initialize Density arrays. Create initial density matrix guess.
   call zeroMatrix(quick_qm_struct%dense,nbasis)
   if (quick_method%unrst) call zeroMatrix(quick_qm_struct%denseb,nbasis)

   present = .false.

#ifdef USEDAT
   ! if read matrix is requested, begin to read dmx file
   if (quick_method%readdmx) inquire (file=dataFileName,exist=present)
#endif
   if (present) then
      call quick_open(iDataFile, dataFileName, 'O', 'U', 'W',.true.,ierr)
      CHECK_ERROR(ierr)

      ! read first part, which is restricted or alpha density matrix
      call rchk_darray(iDataFile, "dense", nbasis, nbasis, 1, quick_qm_struct%dense, failed)

      if(quick_method%unrst) then
         failed = 0
         ! read second part, which is beta density matrix
         call rchk_darray(iDataFile, "denseb", nbasis, nbasis, 1, quick_qm_struct%dense, failed)
         if (failed .eq. 0) then
            call PrtWrn(iOutFile,"CONVERTING RESTRICTED DENSITY TO UNRESTRICTED")
            do I=1,nbasis
               do J =1,nbasis
                  quick_qm_struct%dense(J,I) = quick_qm_struct%dense(J,I)/2.d0
                  quick_qm_struct%denseb(J,I) = quick_qm_struct%dense(J,I)
               enddo
            enddo
         endif
      endif
      close(iDataFile)

   else


      ! MFCC Initial Guess
      !if(quick_method%MFCC)then
      !   call MFCC_initial_guess
      !endif

      !  SAD inital guess
      if (quick_method%SAD) then
         call getSadDense
      endif

      if(quick_method%unrst) then
        do I=1,nbasis
          do J =1,nbasis
            quick_qm_struct%dense(J,I) = quick_qm_struct%dense(J,I)/2.d0
            quick_qm_struct%denseb(J,I) = quick_qm_struct%dense(J,I)
          enddo
        enddo
      endif

   endif

   ! debug initial guess
   if (quick_method%debug) call debugInitialGuess
end subroutine initialGuess
