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

!---------------------------------------------------------------------!
! This module contains subroutines and data structures related to     !
! unrestricted scf procedure.                                         ! 
!---------------------------------------------------------------------!

#include "util.fh"

module quick_uscf_module

  implicit none
  private

  public :: uscf,allocate_quick_uscf,deallocate_quick_uscf,alloperatorB 

  double precision, allocatable, dimension(:,:,:) :: alloperatorB

contains

! This subroutine allocates memory for uscf data structures  and initializes them to zero. 
  subroutine allocate_quick_uscf(ierr)

    use quick_method_module, only: quick_method
    use quick_basis_module, only: nbasis
    use quick_scf_module, only: allocate_quick_scf

    implicit none 

    integer, intent(inout) :: ierr

    call allocate_quick_scf(ierr)

    if(.not. allocated(alloperatorB)) allocate(alloperatorB(nbasis, nbasis, quick_method%maxdiisscf), stat=ierr)

    !initialize values to zero
    alloperatorB = 0.0d0

  end subroutine allocate_quick_uscf 

! Deallocate memory allocated for uscf
  subroutine deallocate_quick_uscf(ierr)

    use quick_scf_module, only: deallocate_quick_scf
    implicit none

    integer, intent(inout) :: ierr

    call deallocate_quick_scf(ierr)

    if(allocated(alloperatorB)) deallocate(alloperatorB, stat=ierr)

  end subroutine deallocate_quick_uscf

  ! Ed Brothers. November 27, 2001
  ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  subroutine uscf(ierr)
     !-------------------------------------------------------
     ! this subroutine is to do scf job for restricted system
     !-------------------------------------------------------
     use allmod
     use quick_molden_module, only: quick_molden, exportMO, exportSCF
     implicit none
  
     logical :: done
     integer, intent(inout) :: ierr
     integer :: jscf
     done=.false.
  
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
     !if (quick_method%ecp) call ecpint
  
     ! if not direct SCF, generate 2e int file
     !if (quick_method%nodirect) call aoint
  
     if (quick_method%diisscf .and. .not. quick_method%divcon) call uelectdiis(jscf,ierr)       ! normal scf
  
     jscf=jscf+1
  
     !if (quick_method%debug)  call debug_SCF(jscf)

     if(write_molden) then
         call exportMO(quick_molden, ierr)
         if(.not.quick_method%opt) then
             quick_molden%iexport_snapshot=quick_molden%iexport_snapshot+1
             call exportSCF(quick_molden, ierr)
         endif
     endif  

     return
  
  end subroutine uscf
  
  ! electdiis
  !-------------------------------------------------------
  ! 11/02/2010 Yipu Miao: Add paralle option for HF calculation
  subroutine uelectdiis(jscf,ierr)
  
     use allmod
     use quick_gridpoints_module
     use quick_uscf_operator_module, only: uscf_operator
     use quick_scf_module  
     use quick_oei_module, only: bCalc1e
     use quick_molden_module, only: quick_molden

#if defined HIP || defined HIP_MPIV
     use quick_rocblas_module, only: rocDGEMM
#if defined WITH_MAGMA
     use quick_magma_module, only: magmaDIAG
#else
#if defined WITH_ROCSOLVER
     use quick_rocsolver_module, only: rocDIAG
#endif
#endif
#endif
#ifdef MPIV
     use mpi
#endif

     implicit none
  
     ! variable inputed to return
     integer :: jscf                ! scf interation
     integer, intent(inout) :: ierr
  
     logical :: diisdone = .false.  ! flag to indicate if diis is done
     logical :: deltaO   = .false.  ! delta Operator
     integer :: idiis = 0           ! diis iteration
     integer :: IDIISfinal,iidiis,current_diis
     integer :: lsolerr = 0
     integer :: IDIIS_Error_Start, IDIIS_Error_End
     double precision :: BIJ,DENSEJI,errormax,OJK,temp
     double precision :: Sum2Mat,rms
     integer :: I,J,K,L,IERROR
  
     double precision :: oldEnergy=0.0d0,E1e ! energy for last iteriation, and 1e-energy
     double precision :: PRMS,PRMS2,PCHANGE, tmp
  
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
  
     call allocate_quick_uscf(ierr)
  
     if(master) then
        write(ioutfile,'(40x," USCF ENERGY")')
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
        call MPI_BCAST(quick_qm_struct%dense,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%denseb,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%co,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%cob,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%Eb,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_method%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_method%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
     endif
#endif
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
     if(quick_method%bCUDA) then
  
        if (quick_method%DFT) then

        call gpu_upload_dft_grid(quick_dft_grid%gridxb, quick_dft_grid%gridyb,quick_dft_grid%gridzb, &
        quick_dft_grid%gridb_sswt, quick_dft_grid%gridb_weight, quick_dft_grid%gridb_atm, &
        quick_dft_grid%bin_locator, quick_dft_grid%basf, quick_dft_grid%primf, quick_dft_grid%basf_counter, &
        quick_dft_grid%primf_counter, quick_dft_grid%bin_counter,quick_dft_grid%gridb_count, quick_dft_grid%nbins, &
        quick_dft_grid%nbtotbf, quick_dft_grid%nbtotpf, quick_method%isg, sigrad2, quick_method%DMCutoff, &
        quick_method%XCCutoff)
  
#if defined CUDA_MPIV || defined HIP_MPIV
        call mgpu_get_xclb_time(timer_cumer%TDFTlb)
#endif
  
        endif
     endif
#endif
  
  
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
        !temp=Sum2Mat(quick_qm_struct%dense,quick_qm_struct%s,nbasis)
  
        ! Determine dii cycle and scf cycle
        idiis=idiis+1
        jscf=jscf+1
  
        if(idiis.le.quick_method%maxdiisscf)then
           IDIISfinal=idiis; iidiis=idiis
        else
           IDIISfinal=quick_method%maxdiisscf; iidiis=1
        endif
        !-----------------------------------------------
        ! Before Delta Densitry Matrix, normal operator is implemented here
        !-----------------------------------------------
        ! Triger Operator timer
        RECORD_TIME(timer_begin%TOp)
  
        ! if want to calculate operator difference?
        if(jscf.ge.quick_method%ncyc) deltaO = .true.
  
        !if (quick_method%debug)  call debug_SCF(jscf)
  
        call uscf_operator(deltaO)
  
        quick_qm_struct%denseOld(:,:) = quick_qm_struct%dense(:,:)
        quick_qm_struct%densebOld(:,:) = quick_qm_struct%denseb(:,:)

        !if (quick_method%debug)  call debug_SCF(jscf)
  
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
           ! A matrix is symetric. First, calculate DENSE*S and store in the scratch
           ! matrix hold.Then calculate O*(DENSE*S).  As the operator matrix is symmetric, the
           ! above code can be used. Store this (the ODS term) in the all error
           ! matrix.
  
           ! The first part is ODS
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%dense, &
                 nbasis, quick_qm_struct%s, nbasis, 0.0d0, quick_scratch%hold,nbasis)
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%o, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%dense, &
                 nbasis, quick_qm_struct%s, nbasis, 0.0d0, quick_scratch%hold,nbasis)
  
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%o, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#endif
  
           allerror(:,:,iidiis) = quick_scratch%hold2(:,:)
  
           ! Calculate D O. then calculate S (do) and subtract that from the allerror matrix.
           ! This means we now have the e(i) matrix.
           ! allerror=ODS-SDO
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%dense, &
                 nbasis, quick_qm_struct%o, nbasis, 0.0d0, quick_scratch%hold,nbasis)
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%s, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%dense, &
                 nbasis, quick_qm_struct%o, nbasis, 0.0d0, quick_scratch%hold,nbasis)
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%s, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#endif
  
           do I=1,nbasis
              do J=1,nbasis
                 allerror(J,I,iidiis) = allerror(J,I,iidiis) - quick_scratch%hold2(J,I) !e=ODS=SDO
              enddo
           enddo

           ! 3)  Form the beta operator matrix for step i, O(i).  (Store in alloperatorb array.)
           
           ! 4)  Form beta error matrix for step i.
           ! e(i) = e(i,alpha part)+Ob Db S - S Db Ob
  
           ! First, calculate quick_qm_struct%denseb*S and store in the scratch
           ! matrix hold. Then calculate O*(quick_qm_struct%denseb*S).  As the operator matrix is
           ! symmetric, the above code can be used. Add this (the ODS term) into the allerror
           ! matrix.

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%denseb, &
                 nbasis, quick_qm_struct%s, nbasis, 0.0d0, quick_scratch%hold,nbasis)
           
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%ob, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#else      
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%denseb, &
                    nbasis, quick_qm_struct%s, nbasis, 0.0d0, quick_scratch%hold,nbasis)
           
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%ob, &
                    nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#endif

           do I=1,nbasis
              do J=1,nbasis
                 allerror(J,I,iidiis) = allerror(J,I,iidiis) + quick_scratch%hold2(J,I) !e=ODS=SDO
              enddo
           enddo

           ! Calculate Db O.Then calculate S (DbO) and subtract that from the allerror matrix.
           ! This means we now have the complete e(i) matrix.

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%denseb, &
                 nbasis, quick_qm_struct%ob, nbasis, 0.0d0, quick_scratch%hold,nbasis)
        
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%s, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%denseb, &
                 nbasis, quick_qm_struct%ob, nbasis, 0.0d0, quick_scratch%hold,nbasis)
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%s, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#endif

           errormax = 0.d0
           do I=1,nbasis
              do J=1,nbasis
                 allerror(J,I,iidiis) = allerror(J,I,iidiis) - quick_scratch%hold2(J,I) !e=ODS=SDO
                 errormax = max(allerror(J,I,iidiis),errormax)
              enddo
           enddo
  
           !-----------------------------------------------
           ! 5)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
           ! X is symmetric, but we do not know anything about the symmetry of e.
           ! The easiest way to do this is to calculate e(i) . X , store
           ! this in HOLD, and then calculate Transpose[X] (.e(i) . X)
           !-----------------------------------------------
           quick_scratch%hold2(:,:) = allerror(:,:,iidiis)
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_scratch%hold2, &
                 nbasis, quick_qm_struct%x, nbasis, 0.0d0, quick_scratch%hold,nbasis)
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#else
  
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_scratch%hold2, &
                 nbasis, quick_qm_struct%x, nbasis, 0.0d0, quick_scratch%hold,nbasis)
  
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_scratch%hold2,nbasis)
#endif
           allerror(:,:,iidiis) = quick_scratch%hold2(:,:)
           !-----------------------------------------------
           ! 6)  Store the e'(I) and O(i).
           ! e'(i) is already stored.  Simply store the operator matrix in
           ! all operator.
           !-----------------------------------------------
  
           if(idiis.le.quick_method%maxdiisscf)then
              alloperator(:,:,iidiis) = quick_qm_struct%o(:,:)
              alloperatorB(:,:,iidiis) = quick_qm_struct%ob(:,:)
           else
              do K=1,quick_method%maxdiisscf-1
                 alloperator(:,:,K) = alloperator(:,:,K+1)
                 alloperatorB(:,:,K) = alloperatorB(:,:,K+1)
              enddo
              alloperator(:,:,quick_method%maxdiisscf) = quick_qm_struct%o(:,:)
              alloperatorB(:,:,quick_method%maxdiisscf) = quick_qm_struct%ob(:,:)
           endif
  
           !-----------------------------------------------
           ! 7)  Form matrix B, which is:
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
  
           ! Now copy the current matrix into HOLD2 transposed.  This will be the
           ! Transpose[ej] used in B(i,j) = Trace(e(i) Transpose(e(j)))
           quick_scratch%hold2(:,:) = allerror(:,:,iidiis)
  
           do I=1,IDIISfinal
              ! Copy the transpose of error matrix I into HOLD.
              quick_scratch%hold(:,:) = allerror(:,:,I) 
  
              ! Calculate and sum together the diagonal elements of e(i) Transpose(e(j))).
              BIJ=Sum2Mat(quick_scratch%hold2,quick_scratch%hold,nbasis)
              
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
              quick_scratch%hold(:,:) = allerror(:,:,1)
              do J=1,quick_method%maxdiisscf-1
                 allerror(:,:,J) = allerror(:,:,J+1)
              enddo
              allerror(:,:,quick_method%maxdiisscf) = quick_scratch%hold(:,:)
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
           ! 8)  Solve B*COEFF = RHS which is:
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
           ! 9) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
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
           ! 10) Diagonalize the operator matrix to form a new density matrix.
           ! First you have to transpose this into an orthogonal basis, which
           ! is accomplished by calculating Transpose[X] . O . X.
           !-----------------------------------------------
#if (defined(CUDA) || defined(CUDA_MPIV)) && !defined(HIP)
 
          RECORD_TIME(timer_begin%TDiag)
          call cuda_diag(quick_qm_struct%o, quick_qm_struct%x, quick_scratch%hold,&
                quick_qm_struct%E, quick_qm_struct%idegen, &
                quick_qm_struct%vec, quick_qm_struct%co, &
                V2, nbasis)
           RECORD_TIME(timer_end%TDiag)
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_qm_struct%o,nbasis)
#else
#if defined HIP || defined HIP_MPIV
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%o, &
                 nbasis, quick_qm_struct%x, nbasis, 0.0d0, quick_scratch%hold,nbasis)

           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_qm_struct%o,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%o, &
                 nbasis, quick_qm_struct%x, nbasis, 0.0d0, quick_scratch%hold,nbasis)
  
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_qm_struct%o,nbasis)
#endif
           ! Now diagonalize the operator matrix.
           RECORD_TIME(timer_begin%TDiag)
#if (defined HIP || defined HIP_MPIV) && defined WITH_MAGMA
           call magmaDIAG(nbasis,quick_qm_struct%o,quick_qm_struct%E,quick_qm_struct%vec,IERROR)
#else 
#if defined LAPACK || defined MKL
           call DIAGMKL(nbasis,quick_qm_struct%o,quick_qm_struct%E,quick_qm_struct%vec,IERROR)
#else
           call DIAG(nbasis,quick_qm_struct%o,nbasis,quick_method%DMCutoff,V2,quick_qm_struct%E,&
                 quick_qm_struct%idegen,quick_qm_struct%vec,IERROR)
#endif
#endif
           RECORD_TIME(timer_end%TDiag)
  
#endif

           timer_cumer%TDiag=timer_end%TDiag-timer_begin%TDiag
  
           ! Calculate C = XC' and form a new density matrix.
           ! The C' is from the above diagonalization.  Also, save the previous
           ! Density matrix to check for convergence.
           !        call DMatMul(nbasis,X,VEC,CO)    ! C=XC'
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_qm_struct%vec, nbasis, 0.0d0, quick_qm_struct%co,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_qm_struct%vec, nbasis, 0.0d0, quick_qm_struct%co,nbasis)
#endif
  
           quick_scratch%hold(:,:) = quick_qm_struct%dense(:,:) 
  
           ! Form new density matrix using MO coefficients
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
           call GPU_DGEMM ('n', 't', nbasis, nbasis, quick_molspec%nelec, 1.0d0, quick_qm_struct%co, &
                 nbasis, quick_qm_struct%co, nbasis, 0.0d0, quick_qm_struct%dense,nbasis)
#else
           call DGEMM ('n', 't', nbasis, nbasis, quick_molspec%nelec, 1.0d0, quick_qm_struct%co, &
                 nbasis, quick_qm_struct%co, nbasis, 0.0d0, quick_qm_struct%dense,nbasis)
#endif

           ! Now check for convergence. pchange is the max change
           ! and prms is the rms
           PCHANGE=0.d0
           do I=1,nbasis
              do J=1,nbasis
                 PCHANGE=max(PCHANGE,abs(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I)))
              enddo
           enddo
           PRMS = rms(quick_qm_struct%dense,quick_scratch%hold,nbasis)


           !-----------------------------------------------
           ! 11) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
           ! If the solution to step eight failed, skip this step and revert
           ! to a standard scf cycle.
           !-----------------------------------------------
           ! Xiao HE 07/20/2007,if the B matrix is ill-conditioned, remove the first,second... error vector
           if (LSOLERR == 0) then
              do J=1,nbasis
                 do K=1,nbasis
                    OJK=0.d0
                    do I=IDIIS_Error_Start, IDIIS_Error_End
                       OJK = OJK + COEFF(I-IDIIS_Error_Start+1) * alloperatorB(K,J,I)
                    enddo
                    quick_qm_struct%ob(J,K) = OJK
                 enddo
              enddo

           endif

           !-----------------------------------------------
           ! 12) Diagonalize the beta operator matrix to form a new beta density matrix.
           ! First you have to transpose this into an orthogonal basis, which
           ! is accomplished by calculating Transpose[X] . O . X.
           !-----------------------------------------------
#if (defined(CUDA) || defined(CUDA_MPIV)) && !defined(HIP)

          RECORD_TIME(timer_begin%TDiag)
          call cuda_diag(quick_qm_struct%ob, quick_qm_struct%x, quick_scratch%hold,&
                quick_qm_struct%EB, quick_qm_struct%idegen, &
                quick_qm_struct%vec, quick_qm_struct%cob, &
                V2, nbasis)
           RECORD_TIME(timer_end%TDiag)

           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_qm_struct%ob,nbasis)
#else
#if defined HIP || defined HIP_MPIV
           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%ob, &
                 nbasis, quick_qm_struct%x, nbasis, 0.0d0, quick_scratch%hold,nbasis)

           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_qm_struct%ob,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%ob, &
                 nbasis, quick_qm_struct%x, nbasis, 0.0d0, quick_scratch%hold,nbasis)

           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_scratch%hold, nbasis, 0.0d0, quick_qm_struct%ob,nbasis)
#endif
           ! Now diagonalize the operator matrix.
           RECORD_TIME(timer_begin%TDiag)
#if (defined HIP || defined HIP_MPIV) && defined WITH_MAGMA
           call magmaDIAG(nbasis,quick_qm_struct%ob,quick_qm_struct%EB,quick_qm_struct%vec,IERROR)
#else 
#if defined LAPACK || defined MKL
           call DIAGMKL(nbasis,quick_qm_struct%ob,quick_qm_struct%EB,quick_qm_struct%vec,IERROR)
#else
           call DIAG(nbasis,quick_qm_struct%ob,nbasis,quick_method%DMCutoff,V2,quick_qm_struct%EB,&
                 quick_qm_struct%idegen,quick_qm_struct%vec,IERROR)
#endif
#endif
           RECORD_TIME(timer_end%TDiag)

#endif
           timer_cumer%TDiag=timer_cumer%TDiag+timer_end%TDiag-timer_begin%TDiag

           ! Calculate C = XC' and form a new density matrix.
           ! The C' is from the above diagonalization.  Also, save the previous
           ! Density matrix to check for convergence.
           !        call DMatMul(nbasis,X,VEC,CO)    ! C=XC'

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

           call GPU_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_qm_struct%vec, nbasis, 0.0d0, quick_qm_struct%cob,nbasis)
#else
           call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_qm_struct%x, &
                 nbasis, quick_qm_struct%vec, nbasis, 0.0d0, quick_qm_struct%cob,nbasis)
#endif

           quick_scratch%hold(:,:) = quick_qm_struct%denseb(:,:)

           ! Form new density matrix using MO coefficients
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
           call GPU_DGEMM ('n', 't', nbasis, nbasis, quick_molspec%nelecb, 1.0d0, quick_qm_struct%cob, &
                 nbasis, quick_qm_struct%cob, nbasis, 0.0d0, quick_qm_struct%denseb,nbasis)
#else
           call DGEMM ('n', 't', nbasis, nbasis, quick_molspec%nelecb, 1.0d0, quick_qm_struct%cob, &
                 nbasis, quick_qm_struct%cob, nbasis, 0.0d0, quick_qm_struct%denseb,nbasis)
#endif

           ! Now check for convergence. pchange is the max change
           ! and prms is the rms
           do I=1,nbasis
              do J=1,nbasis
                 PCHANGE=max(PCHANGE,abs(quick_qm_struct%denseb(J,I)-quick_scratch%hold(J,I)))
              enddo
           enddo
           PRMS2 = rms(quick_qm_struct%denseb,quick_scratch%hold,nbasis)
           PRMS = MAX(PRMS,PRMS2)

           RECORD_TIME(timer_end%TDII)  

           tmp = quick_method%integralCutoff
           call adjust_cutoff(PRMS,PCHANGE,quick_method,ierr)  !from quick_method_module

        !do I=1,nbasis; do J=1,nbasis
        !  write(*,*) jscf,i,j,quick_qm_struct%dense(j,i),quick_qm_struct%denseb(j,i),&
        !  quick_qm_struct%co(j,i), quick_qm_struct%cob(j,i)
        !enddo; enddo

        endif
  
        !--------------- MPI/ALL NODES -----------------------------------------
        RECORD_TIME(timer_end%TSCF)
        timer_cumer%TOp=timer_end%TOp-timer_begin%TOp+timer_cumer%TOp
        timer_cumer%TSCF=timer_end%TSCF-timer_begin%TSCF+timer_cumer%TSCF
        timer_cumer%TDII=timer_end%TDII-timer_begin%TDII+timer_cumer%TDII
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
                 write (ioutfile,'(" ALPHA ELECTRON DENSITY    =",F16.10)') quick_qm_struct%aelec
                 write (ioutfile,'(" BETA ELECTRON DENSITY     =",F16.10)') quick_qm_struct%belec
              endif
  
              !if (quick_method%prtgap) write (ioutfile,'(" HOMO-LUMO GAP (EV) =",11x,F12.6)') &
              !      (quick_qm_struct%E((quick_molspec%nelec/2)+1) - quick_qm_struct%E(quick_molspec%nelec/2))*AU_TO_EV
              diisdone=.true.
              quick_method%uscf_conv=.true. 
  
           endif
           if(jscf >= quick_method%iscf-1) then
              write (ioutfile,'(" RAN OUT OF CYCLES.  NO CONVERGENCE.")')
              write (ioutfile,'(" PERFORM FINAL NO INTERPOLATION ITERATION")')
              diisdone=.true.
              quick_method%uscf_conv=.false.
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
           call MPI_BCAST(quick_method%uscf_conv,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%dense,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%denseb,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%co,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%cob,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_qm_struct%Eb,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_method%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BCAST(quick_method%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
           call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
        endif
#endif
        !if (quick_method%debug)  call debug_SCF(jscf)
     enddo
  
#if (defined CUDA || defined CUDA_MPIV) && !defined(HIP)
     if(master .and. write_molden) then
         quick_molden%nscf_snapshots(quick_molden%iexport_snapshot)=jscf
     endif  

     ! sign of the coefficient matrix resulting from cusolver is not consistent
     ! with rest of the code (e.g. gradients). We have to correct this.
     call scalarMatMul(quick_qm_struct%co,nbasis,nbasis,-1.0d0)
     call scalarMatMul(quick_qm_struct%cob,nbasis,nbasis,-1.0d0)
#endif
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
    if (quick_method%DFT) then
       if(quick_method%grad) then
         call gpu_delete_dft_dev_grid()
       else
         call gpu_delete_dft_grid()
       endif
    endif
#endif
  
     call deallocate_quick_uscf(ierr)
  
     return
  end subroutine uelectdiis

end module quick_uscf_module
