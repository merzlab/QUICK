!---------------------------------------------------------------------!
! Created by Madu Manathunga on 06/24/2020                            !
!                                                                     ! 
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#include "util.fh"

! This module contains subroutines and data structures related to scf
! and diis procedures

module quick_scf_module

  implicit none
  private

  public :: allocate_quick_scf, deallocate_quick_scf, scf 
  public :: B, BSAVE, BCOPY, W, COEFF, RHS, allerror, alloperator

!  type quick_scf_type

    ! matrices required for diis procedure
    double precision, allocatable, dimension(:,:)   :: B

    double precision, allocatable, dimension(:,:)   :: BSAVE

    double precision, allocatable, dimension(:,:)   :: BCOPY

    double precision, allocatable, dimension(:)     :: W

    double precision, allocatable, dimension(:)     :: COEFF

    double precision, allocatable, dimension(:)     :: RHS

    double precision, allocatable, dimension(:,:,:) :: allerror

    double precision, allocatable, dimension(:,:,:) :: alloperator

!  end type quick_scf_type

!  type (quick_scf_type), save :: quick_scf

contains

! This subroutine allocates memory for quick_scf type and initializes them to zero. 
  subroutine allocate_quick_scf(ierr)
    use quick_method_module
    use quick_basis_module
    use quick_molspec_module
    use quick_scratch_module, only: quick_scratch

    implicit none 

    integer, intent(inout) :: ierr

    if(.not. allocated(B))           allocate(B(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(BSAVE))       allocate(BSAVE(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(BCOPY))       allocate(BCOPY(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(W))           allocate(W(quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(COEFF))       allocate(COEFF(quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(RHS))         allocate(RHS(quick_method%maxdiisscf+1), stat=ierr)
    if(.not. allocated(allerror))    allocate(allerror(NBSuse, NBSuse, quick_method%maxdiisscf), stat=ierr)
    if(.not. allocated(alloperator)) allocate(alloperator(nbasis, nbasis, quick_method%maxdiisscf), stat=ierr)

     ! hold3, hold4 are only needed in case of near-linear dependency
     ! path where NBSuse < nbasis.  In the standard case NBSuse == nbasis and hold/hold2
     ! (already nbasis x nbasis) are reused as intermediates, so these arrays are skipped.
     if(NBSuse .ne. nbasis) then
        if(.not. allocated(quick_scratch%hold3)) allocate(quick_scratch%hold3(nbasis, NBSuse))
        if(.not. allocated(quick_scratch%hold4)) allocate(quick_scratch%hold4(NBSuse, NBSuse))
     end if

    !initialize values to zero
    B           = 0.0d0
    BSAVE       = 0.0d0
    BCOPY       = 0.0d0
    W           = 0.0d0
    COEFF       = 0.0d0
    RHS         = 0.0d0
    allerror    = 0.0d0
    alloperator = 0.0d0
  end subroutine allocate_quick_scf 


  subroutine deallocate_quick_scf(ierr)
    use quick_scratch_module, only: quick_scratch
    implicit none

    integer, intent(inout) :: ierr

    if(allocated(B))           deallocate(B, stat=ierr)
    if(allocated(BSAVE))       deallocate(BSAVE, stat=ierr)
    if(allocated(BCOPY))       deallocate(BCOPY, stat=ierr)
    if(allocated(W))           deallocate(W, stat=ierr)
    if(allocated(COEFF))       deallocate(COEFF, stat=ierr)
    if(allocated(RHS))         deallocate(RHS, stat=ierr)
    if(allocated(allerror))    deallocate(allerror, stat=ierr)
    if(allocated(alloperator)) deallocate(alloperator, stat=ierr)

    if(allocated(quick_scratch%hold3)) deallocate(quick_scratch%hold3)
    if(allocated(quick_scratch%hold4)) deallocate(quick_scratch%hold4)
  end subroutine deallocate_quick_scf


  !-------------------------------------------------------
  ! this subroutine is to do scf job for restricted system
  !-------------------------------------------------------
  ! Ed Brothers. November 27, 2001
  ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  subroutine scf(ierr)
     use allmod
#if defined(RESTART_HDF5)
     use quick_io_module, only: read_hdf5_int_rank0, read_hdf5_real8_rank2
#else
     use quick_io_module, only: read_int_rank0, read_real8_rank3
#endif

     implicit none

     integer, intent(inout) :: ierr
  
     logical :: done
     integer :: jscf
     integer :: fail

     done = .false.

      if (quick_method%readden) then
        if (master) then
#if defined(RESTART_HDF5)
         call read_hdf5_int_rank0('molinfo', 2, nbasis)
         call read_hdf5_real8_rank2('dense', (/1,1/), (/nbasis,nbasis/), quick_qm_struct%dense)
#else
         open(unit=iDataFile, file=dataFileName, status='OLD', form='UNFORMATTED')
         call read_int_rank0(iDataFile, "nbasis", nbasis, fail)
         call read_real8_rank3(iDataFile, "dense", nbasis, nbasis, 1, quick_qm_struct%dense, fail)
         close(iDataFile)
#endif
       endif
     endif
  
     !-----------------------------------------------------------------
     ! The purpose of this subroutine is to perform scf cycles.  At this
     ! point, X has been formed. The remaining steps are:
     ! 1)  Form operator matrix.
     ! 2)  Calculate O' = Transpose[X] O X
     ! 3)  Diagonalize O' to obtain C' and eigenvalues.
     ! 4)  Calculate C = XC'
     ! 5)  Form new density matrix.
     ! 6)  Check for convergence.
     !-----------------------------------------------------------------
  
     ! Each location in the code that the step is occurring will be marked.
     ! The cycles stop when prms  is less than pmaxrms or when the maximum
     ! number of scfcycles has been reached.
     jscf=0
  
     ! Alessandro GENONI 03/21/2007
     ! ECP integrals computation exploiting Alexander V. Mitin Subroutine
     ! Note: the integrals are stored in the array ecp_int that corresponds
     !       to the lower triangular matrix of the ECP integrals
     if (quick_method%ecp) call ecpint
  
     ! if not direct SCF, generate 2e int file
     ! if (quick_method%nodirect) call aoint
  
     if (quick_method%diisscf .and. .not. quick_method%divcon) call electdiis(jscf,ierr)  ! normal scf
!     if (quick_method%diisscf .and. quick_method%divcon) call electdiisdc(jscf,PRMS)     ! div & con scf
  
     jscf=jscf+1
  
     if (quick_method%debug)  call debug_SCF(jscf)
  end subroutine scf
  

  ! electdiis
  !-------------------------------------------------------
  ! 11/02/2010 Yipu Miao: Add parallel option for HF calculation
  subroutine electdiis(jscf,ierr)
     use allmod
     use quick_gridpoints_module
     use quick_scf_operator_module, only: scf_operator
     use quick_oei_module, only: bCalc1e 
     use quick_lri_module, only: computeLRI
     use quick_molden_module, only: quick_molden
#ifdef CEW 
     use quick_cew_module, only : quick_cew
#endif
#if defined(MPIV)
     use mpi_f08
#endif
     implicit none
 
     ! variable inputed to return
     integer :: jscf                ! scf iteration
     integer, intent(inout) :: ierr
  
     logical :: LShift = .false.    ! flag if level shifting is being performed

     logical :: diisdone = .false.  ! flag to indicate if diis is done
     logical :: deltaO   = .false.  ! delta Operator
     integer :: idiis = 0           ! diis iteration
     integer :: IDIISfinal,iidiis,current_diis
     integer :: lsolerr = 0
     integer :: IDIIS_Error_Start, IDIIS_Error_End
     double precision :: BIJ,DENSEJI,errormax,OJK,temp
     double precision :: Sum2Mat,rms, shift, bandgap
     integer :: I,J,K,L,IERROR, homo
  
      double precision :: oldEnergy=0.0d0,E1e ! energy for last iteration, and 1e-energy
      double precision :: PRMS,PCHANGE, tmp

      double precision :: c_coords(3),c_zeta,c_chg

      ! Pointers to select the correct operator and scratch arrays depending on
      ! whether near-linear dependency has reduced the basis (NBSuse < nbasis).
      ! operator_ptr -> oeff (NBSuse x NBSuse) or o (nbasis x nbasis, used as NBSuse x NBSuse)
      ! scratch_sq   -> hold4 or hold2  (NBSuse x NBSuse square scratch)
      ! scratch_rect -> hold3 or hold   (nbasis x NBSuse rectangular scratch)
      double precision, pointer :: operator_ptr(:,:)
      double precision, pointer :: scratch_sq(:,:)
      double precision, pointer :: scratch_rect(:,:)

     !---------------------------------------------------------------------------
     ! The purpose of this subroutine is to utilize Pulay's accelerated
     ! scf convergence as detailed in J. Comp. Chem, Vol 3, #4, pg 566-60, 1982.
     ! At the beginning of this process, their is an approximate density
     ! matrix.
     ! The step in the procedure are:
     ! 1)  Form the operator matrix for step i, O(i).
     ! 2)  Form error matrix for step i.
     ! e(i) = ODS - SDO
     ! 3)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
     ! 4)  Store the e'(I) and O(i)
     ! 5)  Form matrix B, which is:
     !      _                                                 _
     !     |                                                   |
     !     |  B(1,1)      B(1,2)     . . .     B(1,J)      -1  |
     !     |  B(2,1)      B(2,2)     . . .     B(2,J)      -1  |
     !     |  .            .                     .          .  |
     ! B = |  .            .                     .          .  |
     !     |  .            .                     .          .  |
     !     |  B(I,1)      B(I,2)     . . .     B(I,J)      -1  |
     !     | -1            -1        . . .      -1          0  |
     !     |_                                                 _|
     ! Where B(i,j) = Trace(e(i) Transpose(e(j)) )
     ! 6)  Solve B*COEFF = RHS which is:
     ! _                                             _  _  _     _  _
     ! |                                               ||    |   |    |
     ! |  B(1,1)      B(1,2)     . . .     B(1,J)  -1  ||  C1|   |  0 |
     ! |  B(2,1)      B(2,2)     . . .     B(2,J)  -1  ||  C2|   |  0 |
     ! |  .            .                     .      .  ||  . |   |  0 |
     ! |  .            .                     .      .  ||  . | = |  0 |
     ! |  .            .                     .      .  ||  . |   |  0 |
     ! |  B(I,1)      B(I,2)     . . .     B(I,J)  -1  ||  Ci|   |  0 |
     ! | -1            -1        . . .      -1      0  || -L |   | -1 |
     ! |_                                             _||_  _|   |_  _|
     ! 7) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
     ! 8) Diagonalize the operator matrix to form a new density matrix.
     ! As in scf.F, each step wil be reviewed as we pass through the code.
     !---------------------------------------------------------------------------
  
     if(master) call allocate_quick_scf(ierr)

     ! Set up pointers so the DIIS loop uses a single code path regardless of
     ! whether near-linear dependency has reduced the basis (NBSuse < nbasis):
     !   operator_ptr -> oeff (NBSuse x NBSuse) or o  (treated as NBSuse x NBSuse)
     !   scratch_sq   -> hold4 or hold2  (NBSuse x NBSuse square scratch)
     !   scratch_rect -> hold3 or hold   (nbasis x NBSuse rectangular scratch)
     if(NBSuse .ne. nbasis) then
        operator_ptr  => quick_qm_struct%oeff
        scratch_sq    => quick_scratch%hold4
        scratch_rect  => quick_scratch%hold3
     else
        operator_ptr  => quick_qm_struct%o
        scratch_sq    => quick_scratch%hold2
        scratch_rect  => quick_scratch%hold
     end if
  
     if(master) then
        write(ioutfile,'(40x," SCF ENERGY")')
        if (quick_method%printEnergy) then
           write(ioutfile,'("| ",120("-"))')
        else
           write(ioutfile,'("| ",90("-"))')
        endif
        write(ioutfile,'("| ","NCYC",6x)',advance="no")
        if (quick_method%printEnergy) write(ioutfile,'(" ENERGY ",8x,"DELTA_E",5x)',advance="no")
        write(ioutfile,'(" SCF_TIME",2x,"DII_CYC",2x," DII_TIME ",2x,"O_TIME",2x, &
              "DIAG_TIME",4x,"MAX_ERR",4x,"RMS_CHG",4x,"MAX_CHG")')
        if (quick_method%printEnergy) then
           write(ioutfile,'("| ",120("-"))')
        else
           write(ioutfile,'("| ",90("-"))')
        endif
     endif
  
#ifdef MPIV
     !-------------- MPI / ALL NODE ---------------
     ! Setup MPI integral configuration
     if (bMPI) call MPI_setup_hfoperator
     !-------------- END MPI / ALL NODE -----------
#endif

#ifdef MPIV
     if (bMPI) then
  !      call MPI_BCAST(quick_qm_struct%o,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%dense,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%co,nbasis*NBSuse,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%E,NBSuse,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_method%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_method%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
     endif
#endif
  
#if defined(GPU) || defined(MPIV_GPU)
     if(quick_method%bGPU) then
  
        if (quick_method%DFT &
#ifdef CEW
        .or. quick_cew%use_cew &
#endif
        )then
  
        call gpu_upload_dft_grid(quick_dft_grid%gridxb, quick_dft_grid%gridyb,quick_dft_grid%gridzb, &
        quick_dft_grid%gridb_sswt, quick_dft_grid%gridb_weight, quick_dft_grid%gridb_atm, &
        quick_dft_grid%bin_locator, quick_dft_grid%basf, quick_dft_grid%primf, quick_dft_grid%basf_counter, &
        quick_dft_grid%primf_counter, quick_dft_grid%bin_counter,quick_dft_grid%gridb_count, quick_dft_grid%nbins, &
        quick_dft_grid%nbtotbf, quick_dft_grid%nbtotpf, quick_method%isg, sigrad2, quick_method%DMCutoff, &
        quick_method%XCCutoff)
  
#if defined(MPIV_GPU)
        call mgpu_get_xclb_time(timer_cumer%TDFTlb)
#endif
  
        endif
     endif
#endif

!-----------------------------------------------
!     c_chg=2.0000000000D+00
!     c_zeta=7.5000000000D-01
!     c_coords(1)=1.5000000000D+00
!     c_coords(2)=2.5000000000D+00
!     c_coords(3)=3.5000000000D+00  
!
!     call computeLRI(c_coords,c_zeta,c_chg)
!-----------------------------------------------
 
     bCalc1e = .true.
     diisdone = .false.
     deltaO = .false.
     idiis = 0
     ! Now Begin DIIS
     do while (.not.diisdone)
  
  
        RECORD_TIME(timer_begin%TSCF)
        !--------------------------------------------
        ! 1)  Form the operator matrix for step i, O(i).
        !--------------------------------------------

        ! Determine dii cycle and scf cycle
        idiis=idiis+1
        jscf=jscf+1
  
        if(idiis.le.quick_method%maxdiisscf)then
           IDIISfinal=idiis; iidiis=idiis
        else
           IDIISfinal=quick_method%maxdiisscf; iidiis=1
        endif

        ! Level shift is not performed by default
        LShift = .false.
  
        !-----------------------------------------------
        ! Before Delta Densitry Matrix, normal operator is implemented here
        !-----------------------------------------------
        ! Triger Operator timer
        RECORD_TIME(timer_begin%TOp)
  
        ! if want to calculate operator difference?
        if(jscf.ge.quick_method%ncyc) deltaO = .true.
  
        if (quick_method%debug)  call debug_SCF(jscf)
  
        call scf_operator(deltaO)

        quick_qm_struct%denseOld(:,:) = quick_qm_struct%dense(:,:)

        if (quick_method%debug)  call debug_SCF(jscf)
  
        ! Terminate Operator timer
        RECORD_TIME(timer_end%TOp)
        !------------- MASTER NODE -------------------------------
        if (master) then
           !-----------------------------------------------
           ! End of Delta Matrix
           !-----------------------------------------------
           RECORD_TIME(timer_begin%TDII)
  
           !if (quick_method%debug)  write(ioutfile,*) "hehe hf"
           !if (quick_method%debug)  call debug_SCF(jscf)
  
           !-----------------------------------------------
           ! 2)  Form error matrix for step i.
           ! e(i) = ODS - SDO
           !-----------------------------------------------
           ! The matrix multiplier comes from Steve Dixon. It calculates
           ! C = Transpose(A) B.  Thus to utilize this we have to make sure that the
           ! A matrix is symmetric. First, calculate DENSE*S and store in the scratch
           ! matrix hold. Then calculate O*(DENSE*S). Store this (the ODS term) in hold2.
  
           ! The first part is ODS

           call MAT_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%dense, &
                 nbasis, quick_qm_struct%s, nbasis, 0.0d0, quick_scratch%hold, nbasis)
   
           call MAT_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%o, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2, nbasis)

            ! Calculate D O, then calculate -S*(DO) and add to hold2.
            ! This means we now have the e(i) matrix.
            ! hold2 = ODS (stored above); we add -SDO below to get e = ODS - SDO
            call MAT_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%dense, &
                  nbasis, quick_qm_struct%o, nbasis, 0.0d0, quick_scratch%hold, nbasis)
            call MAT_DGEMM ('n', 'n', nbasis, nbasis, nbasis, -1.0d0, quick_qm_struct%s, &
                  nbasis, quick_scratch%hold, nbasis, 1.0d0, quick_scratch%hold2, nbasis)

           ! hold2 now contains e(i) = ODS - SDO
           errormax = 0.d0
           do I=1,nbasis
              do J=1,nbasis
                 errormax = max(quick_scratch%hold2(J,I), errormax)
              enddo
           enddo

           !-----------------------------------------------
           ! 3)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
           ! X is symmetric, but we do not know anything about the symmetry of e.
           ! The easiest way to do this is to calculate e(i) . X , store
           ! this in a scratch matrix, and then calculate Transpose[X] . (e(i) . X).
           !
           ! scratch_rect(nbasis,NBSuse) is used as the rectangular intermediate and
           ! scratch_sq(NBSuse,NBSuse) receives the result.
           !-----------------------------------------------
            call MAT_DGEMM ('n', 'n', nbasis, NBSuse, nbasis, 1.0d0, quick_scratch%hold2, &
                  nbasis, quick_qm_struct%x, nbasis, 0.0d0, scratch_rect, nbasis)

            call MAT_DGEMM ('t', 'n', NBSuse, NBSuse, nbasis, 1.0d0, quick_qm_struct%x, &
                  nbasis, scratch_rect, nbasis, 0.0d0, scratch_sq, NBSuse)

            allerror(:,:,iidiis) = scratch_sq(:,:)

           ! allerror matrix contains the error in orthogonal basis.
           ! allerror has dimension NBSuse,NBSuse,iidiis
           !-----------------------------------------------
           ! 4)  Store the e'(I) and O(i).
           ! e'(i) is already stored.  Simply store the operator matrix in
           ! all operator.
           !-----------------------------------------------
  
           if(idiis.le.quick_method%maxdiisscf)then
              alloperator(:,:,iidiis) = quick_qm_struct%o(:,:)
           else
              do K=1,quick_method%maxdiisscf-1
                 alloperator(:,:,K) = alloperator(:,:,K+1)
              enddo
              alloperator(:,:,quick_method%maxdiisscf) = quick_qm_struct%o(:,:)
           endif
  
           !-----------------------------------------------
           ! 5)  Form matrix B, which is:
           !       _                                                 _
           !       |                                                   |
           !       |  B(1,1)      B(1,2)     . . .     B(1,J)      -1  |
           !       |  B(2,1)      B(2,2)     . . .     B(2,J)      -1  |
           !       |  .            .                     .          .  |
           ! B =   |  .            .                     .          .  |
           !       |  .            .                     .          .  |
           !       |  .            .                     .          .  |
           !       |  B(I,1)      B(I,2)     . . .     B(I,J)      -1  |
           !       | -1            -1        . . .      -1          0  |
           !       |_                                                 _|
  
           ! Where B(i,j) = Trace(e(i) Transpose(e(j)))
           ! According to an example done in mathematica, B12 = B21.  Note that
           ! the rigorous proof of this phenomenon is left as an exercise for the
           ! reader.  Thus the first step is copying BCOPY to B.  In this way we
           ! only have to recalculate the new elements.
           !-----------------------------------------------
           do I=1,IDIISfinal
              do J=1,IDIISfinal
                 B(J,I) = BCOPY(J,I)
              enddo
           enddo
  
           if(IDIIS.gt.quick_method%maxdiisscf)then
              do I=1,IDIISfinal-1
                 do J=1,IDIISfinal-1
                    B(J,I) = BCOPY(J+1,I+1)
                 enddo
              enddo
           endif
  
            ! Copy the current error slice (j=iidiis) into a scratch array.
            scratch_sq(:,:) = allerror(:,:,iidiis)

            do I=1,IDIISfinal
               ! Calculate and sum together the diagonal elements of e(i) Transpose(e(j))).
               BIJ=Sum2Mat(scratch_sq,allerror(:,:,I),NBSuse)
              
              ! Now place this in the B matrix.
              if(idiis.le.quick_method%maxdiisscf)then
                 B(iidiis,I) = BIJ
                 B(I,iidiis) = BIJ
              else
                 if(I.gt.1)then
                    B(quick_method%maxdiisscf,I-1)=BIJ
                    B(I-1,quick_method%maxdiisscf)=BIJ
                 else
                    B(quick_method%maxdiisscf,quick_method%maxdiisscf)=BIJ
                 endif
              endif
           enddo
  
           if(idiis.gt.quick_method%maxdiisscf)then
              ! Roll allerror ring buffer: save slot 1, shift down, restore to last slot.
               scratch_sq(:,:) = allerror(:,:,1)
               do J=1,quick_method%maxdiisscf-1
                  allerror(:,:,J) = allerror(:,:,J+1)
               enddo
               allerror(:,:,quick_method%maxdiisscf) = scratch_sq(:,:)
           endif
  
           ! Now that all the BIJ elements are in place, fill in all the column
           ! and row ending -1, and fill up the rhs matrix.
  
           do I=1,IDIISfinal
              B(I,IDIISfinal+1) = -1.d0
              B(IDIISfinal+1,I) = -1.d0
           enddo
           do I=1,IDIISfinal
              RHS(I) = 0.d0
           enddo
           RHS(IDIISfinal+1) = -1.d0
           B(IDIISfinal+1,IDIISfinal+1) = 0.d0
  
           ! Now save the B matrix in Bcopy so it is available for subsequent
           ! iterations.
           do I=1,IDIISfinal
              do J=1,IDIISfinal
                 BCOPY(J,I)=B(J,I)
              enddo
           enddo
  
           !-----------------------------------------------
           ! 6)  Solve B*COEFF = RHS which is:
           ! _                                             _  _  _     _  _
           ! |                                               ||    |   |    |
           ! |  B(1,1)      B(1,2)     . . .     B(1,J)  -1  ||  C1|   |  0 |
           ! |  B(2,1)      B(2,2)     . . .     B(2,J)  -1  ||  C2|   |  0 |
           ! |  .            .                     .      .  ||  . |   |  0 |
           ! |  .            .                     .      .  ||  . | = |  0 |
           ! |  .            .                     .      .  ||  . |   |  0 |
           ! |  B(I,1)      B(I,2)     . . .     B(I,J)  -1  ||  Ci|   |  0 |
           ! | -1            -1        . . .      -1      0  || -L |   | -1 |
           ! |_                                             _||_  _|   |_  _|
           !
           !-----------------------------------------------
  
           BSAVE(:,:) = B(:,:)
           call LSOLVE(IDIISfinal+1,quick_method%maxdiisscf+1,B,RHS,W,quick_method%DMCutoff,COEFF,LSOLERR)
  
           IDIIS_Error_Start = 1
           IDIIS_Error_End   = IDIISfinal
           111     IF (LSOLERR.ne.0 .and. IDIISfinal > 0)then
              IDIISfinal=Idiisfinal-1
              do I=1,IDIISfinal+1
                 do J=1,IDIISfinal+1
                    B(I,J)=BSAVE(I+IDIIS_Error_Start,J+IDIIS_Error_Start)
                 enddo
              enddo
              IDIIS_Error_Start = IDIIS_Error_Start + 1
  
              do i=1,IDIISfinal
                 RHS(i)=0.0d0
              enddo
  
              RHS(IDIISfinal+1)=-1.0d0
  
  
              call LSOLVE(IDIISfinal+1,quick_method%maxdiisscf+1,B,RHS,W,quick_method%DMCutoff,COEFF,LSOLERR)
  
              goto 111
           endif
  
           !-----------------------------------------------
           ! 7) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
           ! If the solution to step eight failed, skip this step and revert
           ! to a standard scf cycle.
           !-----------------------------------------------
           ! Xiao HE 07/20/2007,if the B matrix is ill-conditioned, remove the first,second... error vector
           if (LSOLERR == 0) then
              do J=1,nbasis
                 do K=1,nbasis
                    OJK=0.d0
                    do I=IDIIS_Error_Start, IDIIS_Error_End
                       OJK = OJK + COEFF(I-IDIIS_Error_Start+1) * alloperator(K,J,I)
                    enddo
                    quick_qm_struct%o(J,K) = OJK
                 enddo
              enddo
              
           endif
            !-----------------------------------------------
            ! 8) Diagonalize the operator matrix to form a new density matrix.
            ! First you have to transpose this into an orthogonal basis, which
            ! is accomplished by calculating Transpose[X] . O . X.
            ! scratch_rect(nbasis,NBSuse) is used as the rectangular intermediate;
            ! operator_ptr(NBSuse,NBSuse) receives the result.
            !-----------------------------------------------
            call MAT_DGEMM ('n', 'n', nbasis, NBSuse, nbasis, 1.0d0, quick_qm_struct%o, &
                  nbasis, quick_qm_struct%x, nbasis, 0.0d0, scratch_rect, nbasis)

            call MAT_DGEMM ('t', 'n', NBSuse, NBSuse, nbasis, 1.0d0, quick_qm_struct%x, &
                  nbasis, scratch_rect, nbasis, 0.0d0, operator_ptr, NBSuse)

            !-----------------------------------------------
            !  Level shifting if the DIIS error is large.
            !  operator_ptr(NBSuse,NBSuse) is rotated into the eigenbasis via
            !  scratch_sq(NBSuse,NBSuse), virtual eigenvalues are shifted, then
            !  rotated back.
            !-----------------------------------------------
            homo = quick_molspec%nelec/2
            bandgap = quick_qm_struct%E(homo+1) - quick_qm_struct%E(homo)
            if(idiis .ge. quick_method%LShift_cycle .and. errormax .gt. quick_method%LShift_err .and. &
               quick_method%LShift_gap .gt. bandgap)then
               LShift = .true.
               call MAT_DGEMM ('n', 'n', NBSuse, NBSuse, NBSuse, 1.0d0, operator_ptr, &
                    NBSuse, quick_qm_struct%oldvec, NBSuse, 0.0d0, scratch_sq, NBSuse)

               call MAT_DGEMM ('t', 'n', NBSuse, NBSuse, NBSuse, 1.0d0, quick_qm_struct%oldvec, &
                    NBSuse, scratch_sq, NBSuse, 0.0d0, operator_ptr, NBSuse)

               shift = quick_method%LShift_gap - bandgap
               do I=homo+1,NBSuse
                  operator_ptr(I,I) = operator_ptr(I,I) + shift
               enddo
            endif

            ! Now diagonalize the operator matrix (operator_ptr points to oeff or o).
            RECORD_TIME(timer_begin%TDiag)

            call MAT_DIAG(operator_ptr, NBSuse, NBSuse, quick_qm_struct%E, quick_qm_struct%vec)

            RECORD_TIME(timer_end%TDiag)

           ! Calculate C = XC' and form a new density matrix.
           ! The C' is from the above diagonalization.  Also, save the previous
           ! Density matrix to check for convergence.
           !        call DMatMul(nbasis,X,VEC,CO)    ! C=XC'
           ! scratch_sq(NBSuse,NBSuse) is used as intermediate when level-shifting.
           if(LShift)then
              call MAT_DGEMM ('n', 'n', NBSuse, NBSuse, NBSuse, 1.0d0, quick_qm_struct%oldvec, &
                    NBSuse, quick_qm_struct%vec, NBSuse, 0.0d0, scratch_sq, NBSuse)
              call MAT_DGEMM ('n', 'n', nbasis, NBSuse, NBSuse, 1.0d0, quick_qm_struct%x, &
                    nbasis, scratch_sq, NBSuse, 0.0d0, quick_qm_struct%co, nbasis)
              quick_qm_struct%oldvec(:,:) = scratch_sq(:,:)
           else
               call MAT_DGEMM ('n', 'n', nbasis, NBSuse, NBSuse, 1.0d0, quick_qm_struct%x, &
                     nbasis, quick_qm_struct%vec, NBSuse, 0.0d0, quick_qm_struct%co,nbasis)
               quick_qm_struct%oldvec(:,:) = quick_qm_struct%vec(:,:)
           endif
           ! Form new density matrix using MO coefficients
           call MAT_DGEMM ('n', 't', nbasis, nbasis, quick_molspec%nelec/2, 2.0d0, quick_qm_struct%co, &
                 nbasis, quick_qm_struct%co, nbasis, 0.0d0, quick_qm_struct%dense,nbasis)         
           RECORD_TIME(timer_end%TDII)
  
           ! Now check for convergence. pchange is the max change
           ! and prms is the rms
           PCHANGE=0.d0
           do I=1,nbasis
              do J=1,nbasis
                 PCHANGE=max(PCHANGE,abs(quick_qm_struct%dense(J,I)-quick_qm_struct%denseold(J,I)))
              enddo
           enddo
           PRMS = rms(quick_qm_struct%dense,quick_qm_struct%denseold,nbasis)
  
           tmp = quick_method%integralCutoff
           call adjust_cutoff(PRMS,PCHANGE,quick_method,ierr)  !from quick_method_module
        endif
  
        !--------------- MPI/ALL NODES -----------------------------------------
        RECORD_TIME(timer_end%TSCF)
        timer_cumer%TOp=timer_end%TOp-timer_begin%TOp+timer_cumer%TOp
        timer_cumer%TSCF=timer_end%TSCF-timer_begin%TSCF+timer_cumer%TSCF
        timer_cumer%TDII=timer_end%TDII-timer_begin%TDII+timer_cumer%TDII
        timer_cumer%TDiag=timer_end%TDiag-timer_begin%TDiag+timer_cumer%TDiag
        !--------------- END MPI/ALL NODES -------------------------------------
  
        if (master) then
#ifdef USEDAT
           ! open data file then write calculated info to dat file
           SAFE_CALL(quick_open(iDataFile, dataFileName, 'R', 'U', 'R',.true.,ierr)
           rewind(iDataFile)
           call dat(quick_qm_struct, iDataFile)
           close(iDataFile)
#endif
           current_diis=mod(idiis-1,quick_method%maxdiisscf)
           current_diis=current_diis+1
           
           write (ioutfile,'("|",I3,1x)',advance="no") jscf
           if(quick_method%printEnergy)then
              write (ioutfile,'(F16.9,2x)',advance="no") quick_qm_struct%Eel+quick_qm_struct%Ecore
              if (jscf.ne.1) then
                 write(ioutFile,'(E12.6,2x)',advance="no") oldEnergy-quick_qm_struct%Eel-quick_qm_struct%Ecore
              else
                 write(ioutFile,'(4x,"------",4x)',advance="no")
              endif
              oldEnergy=quick_qm_struct%Eel+quick_qm_struct%Ecore
           endif
           write (ioutfile,'(F10.3,4x)',advance="no") timer_end%TSCF-timer_begin%TSCF
           write (ioutfile,'(I2,4x,F8.2,2x,F8.2,2x)',advance="no") current_diis,timer_end%TDII-timer_begin%TDII, &
                 timer_end%TOp-timer_begin%TOp
           write (ioutfile,'(F8.2,4x)',advance="no") timer_end%TDiag-timer_begin%TDiag
           write (ioutfile,'(E10.4,2x)',advance="no") errormax
           write (ioutfile,'(E10.4,2x,E10.4)')  PRMS,PCHANGE
  
           if(LShift) write (ioutfile,'("|   ***  Level shifting applied  ( HOMO-LUMO gap = ",F6.3," au ) ***")')bandgap
           if (lsolerr /= 0) write (ioutfile,'(" DIIS FAILED !!", &
                 & " PERFORM NORMAL SCF. (NOT FATAL.)")')
  
           if (PRMS < quick_method%pmaxrms .and. pchange < quick_method%pmaxrms*100.d0 .and. jscf.gt.MIN_SCF)then
              if (quick_method%printEnergy) then
                 write(ioutfile,'("| ",120("-"))')
              else
                 write(ioutfile,'("| ",90("-"))')
              endif
              write (ioutfile,'("| REACH CONVERGENCE AFTER ",i3," CYCLES")') jscf
              write (ioutfile,'("| MAX ERROR = ",E12.6,2x," RMS CHANGE = ",E12.6,2x," MAX CHANGE = ",E12.6)') &
                    errormax,prms,pchange
              write (ioutfile,'("| -----------------------------------------------")')
              if (quick_method%DFT .or. quick_method%SEDFT) then
                 write (ioutfile,'(" ALPHA ELECTRON DENSITY    = ",F16.10)') quick_qm_struct%aelec
                 write (ioutfile,'(" BETA ELECTRON DENSITY     = ",F16.10)') quick_qm_struct%belec
              endif
  
              if (quick_method%prtgap) write (ioutfile,'(" HOMO-LUMO GAP (EV) =",11x,F12.6)') &
                    (quick_qm_struct%E((quick_molspec%nelec/2)+1) - quick_qm_struct%E(quick_molspec%nelec/2))*AU_TO_EV
              diisdone=.true.
              quick_method%scf_conv=.true.
  
           endif
           if(jscf >= quick_method%iscf-1) then
              write (ioutfile,'(" RAN OUT OF CYCLES.  NO CONVERGENCE.")')
              write (ioutfile,'(" PERFORM FINAL NO INTERPOLATION ITERATION")')
              diisdone=.true.
              quick_method%scf_conv=.false.
           endif
  
           if((tmp .ne. quick_method%integralCutoff).and. .not.diisdone) then
              write(ioutfile, '("| -------------- 2E-INT CUTOFF CHANGE TO ", E10.4, " ------------")') quick_method%integralCutoff
           endif
 
           if(write_molden) quick_molden%e_snapshots(jscf, quick_molden%iexport_snapshot) &
                            = quick_qm_struct%Eel+quick_qm_struct%Ecore 
 
           flush(ioutfile)
  
        endif
  
#ifdef MPIV
        if (bMPI) then
           call MPI_BCAST(diisdone,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_method%scf_conv,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%dense,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%co,nbasis*NBSuse,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%E,NBSuse,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_method%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_method%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
        endif
#endif
        if (quick_method%debug)  call debug_SCF(jscf)
     enddo

     if(master .and. write_molden) then 
         quick_molden%nscf_snapshots(quick_molden%iexport_snapshot)=jscf 
     endif

#if (defined CUDA || defined CUDA_MPIV) && !defined(HIP)
     ! sign of the coefficient matrix resulting from cusolver is not consistent
     ! with rest of the code (e.g. gradients). We have to correct this.
     call scalarMatMul(quick_qm_struct%co,NBSuse,nbasis,-1.0d0)
#endif
  
#if defined(GPU) || defined(MPIV_GPU)
    if (quick_method%DFT &
#ifdef CEW
       .or. quick_cew%use_cew &
#endif
       )then
       if(quick_method%grad) then
         call gpu_delete_dft_dev_grid()
       else
         call gpu_delete_dft_grid()
       endif
    endif
#endif
  
     if(master) call deallocate_quick_scf(ierr)
  
     return
  end subroutine electdiis

end module quick_scf_module
