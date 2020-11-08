!
!	getMol.f90
!	new_quick
!        
!       Updated by Madu Manathunga on 04/20/2019
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

subroutine getMol()
   ! This subroutine is to get molecule information
   ! and assign basis function.
   use allmod
   implicit none

#ifdef MPIV
   include 'mpif.h'
#endif

   logical :: present
   integer :: i,j,k,itemp

   !-----------MPI/MASTER------------------------
   if (master) then

      ! Read info from AMBER
      if (amber_interface_logic) then
         call read_AMBER_crd
      else
         call PrtAct(iOutfile,"Begin Reading Molecular Information")

         ! read xyz coordinates from the .in file 
         if(.not. isTemplate) then
           call quick_open(infile,inFileName,'O','F','W',.true.)

           ! read molecule coordinates
           call read2(quick_molspec,inFile)
           close(inFile)
         endif

         quick_molspec%nbasis   => nbasis
         quick_qm_struct%nbasis => nbasis
         call set(quick_molspec)

         ! quick forward coordinates stored in namelist to instant variables
         xyz(1:3,1:natom)=quick_molspec%xyz(1:3,1:natom)
      end if   !amber_interface_logic

      ! Now read in another line
      ! to see the specific grid has been requested if it is a DFT job
      if ((quick_method%DFT .OR. quick_method%SEDFT).and.quick_method%isg.eq.1) call gridformSG1()        ! Xiao HE 01/09/07 SG1 SG0 grid

      ! check the correctness between molecular specification and method used
      call check_quick_method_and_molspec(iOutFile,quick_molspec,quick_method)
   endif
   !-----------END MPI/MASTER-----------------------
#ifdef MPIV
   !-----------MPI/ALL NODES------------------------
   if (bMPI)  call mpi_setup_mol1()
   !-----------END MPI/ALL NODES--------------------
#endif

#if defined CUDA || defined CUDA_MPIV 
   quick_method%bCUDA = .true.
#endif

   ! At this point we have the positions and identities of the atoms. We also
   ! have the number of electrons. Now we must assign basis functions. This
   ! is done in a subroutine.
   call readbasis(natom,0,0,0,0)

   quick_molspec%nbasis   => nbasis
   quick_qm_struct%nbasis => nbasis

   call allocate_basis(quick_method)
   call alloc(quick_qm_struct)
   call init(quick_qm_struct)



   !-----------MPI/MASTER------------------------
   if (master) then
      ! now print molecule specification to output file
      call print(quick_molspec,iOutFile)
      call print(quick_basis,iOutFile)

      ! the following some step are setup for basis and for ECP or DFT calculation
      ! and see comments for details
      ! Include the normalization constant in the coefficient.
      call normalize_basis()

      ! Store the normalized primitives coefficients for ECP calculations
      if (quick_method%ecp) call store_basis_to_ecp()      ! Alessandro GENONI 03/21/2007

   endif !Madu

      ! calculate the radius of the sphere of basis function signifigance.
!      if (quick_method%DFT .OR. quick_method%SEDFT) call get_sigrad

  ! calculate the radius of the sphere of basis function signifigance.
    if (quick_method%DFT .OR. quick_method%SEDFT) call get_sigrad

#ifdef MPIV
   if (bMPI) call mpi_setup_mol2()
#endif
!--------------Madu----------------

   if (master) then
      ! Read params for semi-emipeircal DFT
      if (quick_method%SEDFT) call read_sedft_parm

      !  initialize density matrix
      call initialGuess

      call PrtAct(iOutfile,"End Reading Molecular Information ")

   endif
   !-----------END MPI/MASTER------------------------

   return

end subroutine getmol


!--------------------
! check mol spec and method
!--------------------
subroutine check_quick_method_and_molspec(io,quick_molspec_arg,quick_method_arg)
   use quick_method_module
   use quick_molspec_module
   implicit none

   ! Argument
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
               call PrtWrn(io,"ATOM OUT OF RANGE FOR CORE")
      enddo
   endif


   ! check the correctness between elections, multiplicity and unrestricted calculation
   ! request. And try to correct possible error
   i = mod(dble(quick_molspec_arg%nelec),2.d0)
   if (i.ne.0 .and. .not. quick_method_arg%unrst) then
      call PrtWrn(io,"UNPAIRED ELECTRONS REQUIRE UNRESTRICTED CALCULATIONS.")
      quick_method_arg%UNRST=.true.
   endif

   if (quick_molspec_arg%imult.ne.1 .and. .not. quick_method_arg%unrst) then
      call PrtWrn(io,"HIGHER MULTIPLICITIES REQUIRE UNRESTRICTED CALCULATIONS.")
      call quick_exit(6,1)
   endif

   if (quick_method_arg%unrst) then
      quick_molspec_arg%nelecb = quick_molspec_arg%nelec
      quick_molspec_arg%nelec = 0

      do WHILE (quick_molspec_arg%nelec.lt.quick_molspec_arg%nelecb)
         quick_molspec_arg%nelecb = quick_molspec_arg%nelecb-1
         quick_molspec_arg%nelec = quick_molspec_arg%nelec +1
      enddo

      if (quick_molspec_arg%imult .eq. 2 .and. quick_molspec_arg%nelec-1 .ne. quick_molspec_arg%nelecb) then
         call PrtWrn(io,"INCORRECT NUMBER OF ELECTRONS FOR A DOUBLET.")
         call quick_exit(6,1)
      endif

      if (quick_molspec_arg%imult .eq. 3) then
         quick_molspec_arg%nelec = quick_molspec_arg%nelec+1
         quick_molspec_arg%nelecb = quick_molspec_arg%nelecb - 1
         if (quick_molspec_arg%nelec-2 .ne. quick_molspec_arg%nelecb) then
            call PrtWrn(io,"INCORRECT NUMBER OF ELECTRONS FOR A TRIPLET.")
            call quick_exit(6,1)
         endif
      endif

      if (quick_molspec_arg%imult .eq. 4) then
         quick_molspec_arg%nelec = quick_molspec_arg%nelec+1
         quick_molspec_arg%nelecb = quick_molspec_arg%nelecb - 1
         if (quick_molspec_arg%nelec-3 .ne. quick_molspec_arg%nelecb) then
            call PrtWrn(io,"INCORRECT NUMBER OF ELECTRONS FOR A QUADRUPLET.")
            call quick_exit(6,1)
         endif
      endif
   endif

end subroutine check_quick_method_and_molspec


!--------------------------------------
! Initial Densitry Matrix
!--------------------------------------
subroutine initialGuess
   use allmod
   implicit none
   logical :: present
   integer :: failed
   character(len=80) :: keyWD
   integer n,sadAtom
   integer Iatm,i,j
   double precision temp


   ! Initialize Density arrays. Create initial density matrix guess.
   call zeroMatrix(quick_qm_struct%dense,nbasis)
   if (quick_method%unrst) call zeroMatrix(quick_qm_struct%denseb,nbasis)

   present = .false.

   ! if read matrix is requested, begin to read dmx file
   if (quick_method%readdmx) inquire (file=dataFileName,exist=present)

   if (present) then
      call quick_open(iDataFile, dataFileName, 'O', 'U', 'W',.true.)

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
      if(quick_method%MFCC)then
         call MFCC_initial_guess
      endif

      !  SAD inital guess
      if (quick_method%SAD) then

         n=0
         do Iatm=1,natom
            do sadAtom=1,10

               if(symbol(quick_molspec%iattype(Iatm)).eq. &
                     quick_molspec%atom_type_sym(sadAtom))then
                  do i=1,atombasis(sadAtom)
                     do j=1,atombasis(sadAtom)

                        quick_qm_struct%dense(i+n,j+n)=atomdens(sadAtom,i,j)
                     enddo
                  enddo
                  n=n+atombasis(sadAtom)
               endif
            enddo
         enddo
      endif
   endif

            do I=1,nbasis
               do J =1,nbasis
                  quick_qm_struct%denseInt(J,I) = quick_qm_struct%dense(J,I)
               enddo
            enddo

   call deallocate_mol_sad

   ! debug initial guess
   if (quick_method%debug) call debugInitialGuess
end subroutine initialGuess
