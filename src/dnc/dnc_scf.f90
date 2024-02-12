#include "util.fh"


!*******************************************************
! electdiisdc
!-------------------------------------------------------
! 10/25/2010 YIPU MIAO Successful run on mpi, begin to test
! 10/20/2010 YIPU MIAO Add MPI option to DIIS div&con
! this is dii for div & con
subroutine electdiisdc(jscf,ierr)
   use allmod
     use quick_gridpoints_module
     use quick_scf_operator_module, only: scf_operator
     use quick_oei_module, only: bCalc1e
     use quick_lri_module, only: computeLRI
     use quick_molden_module, only: quick_molden

#ifdef CEW 
     use quick_cew_module, only : quick_cew
#endif

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

   implicit double precision(a-h,o-z) 

#ifdef MPIV
     include "mpif.h"
#endif

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
     integer :: I,J,K,L,IERROR,nbasissave,itt,NtempN

     double precision :: oldEnergy=0.0d0,E1e ! energy for last iteriation, and 1e-energy
     double precision :: PRMS,PCHANGE, tmp,Ttmp

     double precision :: c_coords(3),c_zeta,c_chg

   dimension:: B(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1),BSAVE(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1)
   dimension:: BCOPY(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1),W(quick_method%maxdiisscf+1)
   dimension:: COEFF(quick_method%maxdiisscf+1),RHS(quick_method%maxdiisscf+1)
   dimension:: allerror(nbasis, nbasis, quick_method%maxdiisscf)
   dimension:: alloperator(nbasis, nbasis, quick_method%maxdiisscf)
   double precision,allocatable :: dcco(:,:), holddc(:,:),Xdcsubtemp(:,:)

   logical templog1,templog2
   logical :: diisoff=.false.
   integer elecs,itemp
   double precision efermi(10),oneElecO(nbasis,nbasis)


   !---------------------------------------------------------------------------
   ! The purpose of this subroutine is to utilize Pulay's accelerated
   ! scf convergence as detailed in J. Comp. Chem, Vol 3, #4, pg 566-60, 1982.
   ! At the beginning of this process, their is an approximate density
   ! matrix.
   ! The step in the procedure are:
   ! 1)  Form the operator matrix for step i, O(i).
   ! 2)  Form error matrix for step i.
   ! e(i) = ODS - Sdo
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
   !===========================================================

!     call allocate_quick_scf(ierr)

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
  !      call
  !      MPI_BCAST(quick_qm_struct%o,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%dense,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%co,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_qm_struct%E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_method%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(quick_method%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
     endif
#endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
     if(quick_method%bCUDA) then

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

#if defined CUDA_MPIV || defined HIP_MPIV
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
   diisdone=.false.
   deltaO = .false.
   idiis=0
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

   if (quick_method%debug)  call debug_SCF(jscf)

   call scf_operatordc(deltaO)

   quick_qm_struct%denseOld(:,:) = quick_qm_struct%dense(:,:)

   if (quick_method%debug)  call debug_SCF(jscf)

   ! Terminate Operator timer 
   RECORD_TIME(timer_end%TOp)
   !------------- MASTER NODE -------------------------------
   if (master) then

if (.not.diisoff) then

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
      ! C = Transpose(A) B.  Thus to utilize this we have to make sure that
      ! the
      ! A matrix is symetric. First, calculate DENSE*S and store in the
      ! scratch
      ! matrix hold.Then calculate O*(DENSE*S).  As the operator matrix is
      ! symmetric, the
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

      errormax = 0.d0
      do I=1,nbasis
         do J=1,nbasis
            allerror(J,I,iidiis) = allerror(J,I,iidiis) - quick_scratch%hold2(J,I) !e=ODS=SDO
            errormax = max(allerror(J,I,iidiis),errormax)
         enddo
      enddo

      !-----------------------------------------------
      ! 3)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
      ! X is symmetric, but we do not know anything about the symmetry of  e.
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

      ! Now copy the current matrix into HOLD2 transposed.  This will be the
      ! Transpose[ej] used in B(i,j) = Trace(e(i) Transpose(e(j)))
      quick_scratch%hold2(:,:) = allerror(:,:,iidiis)

      do I=1,IDIISfinal
         ! Copy the transpose of error matrix I into HOLD.
         quick_scratch%hold(:,:) = allerror(:,:,I)

         ! Calculate and sum together the diagonal elements of e(i)
         ! Transpose(e(j))).
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
      ! 7) Form a new operator matrix based on O(new) = [Sum over i]
      ! c(i)O(i)
      ! If the solution to step eight failed, skip this step and revert
      ! to a standard scf cycle.
      !-----------------------------------------------
      ! Xiao HE 07/20/2007,if the B matrix is ill-conditioned, remove the
      ! first,second... error vector
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

      ! TDII Timer needs to be checked when start MPI implementation
      RECORD_TIME(timer_end%TDII)

endif

      !-----------------------------------------------
      ! 8) Diagonalize the operator matrix to form a new density matrix.
      ! First you have to transpose this into an orthogonal basis, which
      ! is accomplished by calculating Transpose[X] . O . X.
      !-----------------------------------------------

  ! Original DnC implementation above used HF and deltaHF
  ! subroutines and had different structure.
  ! Need to carefully think about how MPI can be implemented
  ! in new structure of the DnC code.
 
         !--------------------------------------------
         ! Form extract subsystem operator from full system
         !--------------------------------------------
         call Odivided

         !--------------------------------------------
         ! recover basis number
         !--------------------------------------------
         nbasissave=nbasis

      !============================================
      ! Now begin to diag O of subsystems:
      ! which is to do DIV part
      !============================================

      Ttmp=0.0d0

      do itt = 1, np

         ! pass value for convience reason
         nbasis=nbasisdc(itt)    ! basis set for fragment

         allocate(Odcsubtemp(nbasisdc(itt),nbasisdc(itt)))
         allocate(Xdcsubtemp(nbasisdc(itt),nbasisdc(itt)))
         allocate(VECtemp(nbasisdc(itt),nbasisdc(itt)))
         allocate(Vtemp(3,nbasisdc(itt)))
         allocate(EVAL1temp(nbasisdc(itt)))
         allocate(IDEGEN1temp(nbasisdc(itt)))
         allocate(dcco(nbasisdc(itt),nbasisdc(itt)))
         allocate(holddc(nbasisdc(itt),nbasisdc(itt)))

         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               Odcsubtemp(I,J) = Odcsub(itt,I,J)
               Xdcsubtemp(I,J) = Xdcsub(itt,I,J)
            enddo
         enddo

         NtempN=nbasisdc(itt)

#if (defined(CUDA) || defined(CUDA_MPIV)) && !defined(HIP)

          RECORD_TIME(timer_begin%TDiag)
          call cuda_diag(Odcsubtemp, Xdcsubtemp, quick_scratch%hold,&
                EVAL1temp, IDEGEN1temp, &
                VECtemp, dcco, &
                Vtemp, NtempN)
           RECORD_TIME(timer_end%TDiag)

           call GPU_DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Xdcsubtemp, &
                 NtempN, holddc, NtempN, 0.0d0, Odcsubtemp,NtempN)
#else

#if defined HIP || defined HIP_MPIV

           call GPU_DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Odcsubtemp, &
                 NtempN, Xdcsubtemp, NtempN, 0.0d0, holddc,NtempN)

           call GPU_DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Xdcsubtemp, &
                 NtempN, holddc, NtempN, 0.0d0, Odcsubtemp,NtempN)

#else
           call DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Odcsubtemp, &
                 NtempN, Xdcsubtemp, NtempN, 0.0d0, holddc,NtempN)

           call DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Xdcsubtemp, &
                 NtempN, holddc, NtempN, 0.0d0, Odcsubtemp,NtempN)
#endif  
           ! Now diagonalize the operator matrix.
           RECORD_TIME(timer_begin%TDiag)
#if (defined HIP || defined HIP_MPIV) && defined WITH_MAGMA
           call magmaDIAG(NtempN,Odcsubtemp,EVAL1temp,VECtemp,IERROR)
#else
#if defined LAPACK || defined MKL
           call DIAGMKL(NtempN,Odcsubtemp,EVAL1temp,VECtemp,IERROR)
#else
           call DIAG(NtempN,Odcsubtemp,NtempN,quick_method%DMCutoff,Vtemp,i &
                     EVAL1temp,IDEGEN1temp,VECtemp,IERROR)
#endif

#endif
           RECORD_TIME(timer_end%TDiag)

#endif

         Ttmp=Ttmp+timer_end%TDiag-timer_begin%TDiag
         timer_cumer%TDiag=timer_cumer%TDiag+timer_end%TDiag-timer_begin%TDiag   ! Global dc diag time

         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               Odcsub(itt,I,J) = Odcsubtemp(I,J)
            enddo
            evaldcsub(itt,I)= EVAL1temp(I)
         enddo

         !---------------------------------------------
         ! Calculate C = XC' and form a new density matrix.
         ! The C' is from the above diagonalization.
         !---------------------------------------------

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

           call GPU_DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Xdcsubtemp, &
                 NtempN, VECtemp, NtempN, 0.0d0, dcco,NtempN)
#else
           call DGEMM ('n', 'n', NtempN, NtempN, NtempN, 1.0d0, Xdcsubtemp, &
                 NtempN, VECtemp, NtempN, 0.0d0, dcco,NtempN)
#endif

         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               codcsub(I,J,itt)=dcco(I,J)
               codcsubtran(I,J,itt)=dcco(I,J)
            enddo
         enddo

         if (allocated(Odcsubtemp)) deallocate(Odcsubtemp)
         if (allocated(Xdcsubtemp)) deallocate(Xdcsubtemp)
         if (allocated(VECtemp)) deallocate(VECtemp)
         if (allocated(Vtemp)) deallocate(Vtemp)
         if (allocated(EVAL1temp)) deallocate(EVAL1temp)
         if (allocated(IDEGEN1temp)) deallocate(IDEGEN1temp)
         if (allocated(dcco)) deallocate(dcco)
         if (allocated(holddc)) deallocate(holddc)
      enddo

      quick_scratch%hold(:,:) = quick_qm_struct%dense(:,:)

      nbasis=nbasissave
      !--------------------------------------------
      ! Next step is to calculate fermi energy and renormalize
      ! the density matrix
      !--------------------------------------------

      call fermiSCF(efermi,jscf)


      ! Now check for convergence. pchange is the max change
      ! and prms is the rms
      PCHANGE=0.d0
      do I=1,nbasis
         do J=1,nbasis
            PCHANGE=max(PCHANGE,abs(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I)))
         enddo
      enddo
      PRMS = rms(quick_qm_struct%dense,quick_scratch%hold,nbasis)

      tmp = quick_method%integralCutoff
      call adjust_cutoff(PRMS,PCHANGE,quick_method,ierr)  !from quick_method_module
   endif

      !------ MPI/ALL NODES --------------------------------------------
      RECORD_TIME(timer_end%TSCF)
      timer_cumer%TSCF=timer_cumer%TSCF+timer_end%TSCF-timer_begin%TSCF
      timer_cumer%TDII=timer_cumer%TDII+timer_end%TDII-timer_begin%TDII
      timer_cumer%TOp=timer_cumer%TOp+timer_end%TOp-timer_begin%TOp
      !------ END MPI/ALL NODES ----------------------------------------

   if (master) then

#ifdef USEDAT
      ! open data file then write calculated info to dat file
      SAFE_CALL(quick_open(iDataFile, dataFileName, 'R', 'U','R',.true.,ierr)
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
      write (ioutfile,'(F10.3,2x)',advance="no") timer_end%TSCF-timer_begin%TSCF
      if (.not.diisoff) then
         write (ioutfile,'(2x,I2,2x)',advance="no") current_diis
      else
         write (ioutfile,'(2x,"--",2x)',advance="no")
      endif
      write (ioutfile,'(2x,F8.2,2x,F8.2,2x)',advance="no") timer_end%TDII-timer_begin%TDII, &
            timer_end%TOp-timer_begin%TOp
      write (ioutfile,'(F8.2,4x)',advance="no") Ttmp
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
            write (ioutfile,'(" ALPHA ELECTRON DENSITY    = ",F16.10)') quick_qm_struct%aelec
            write (ioutfile,'(" BETA ELECTRON DENSITY     = ",F16.10)') quick_qm_struct%belec
         endif

         if (quick_method%prtgap) write (ioutfile,'(" HOMO-LUMO GAP (EV) =",11x,F12.6)') &
               (quick_qm_struct%E((quick_molspec%nelec/2)+1) - quick_qm_struct%E(quick_molspec%nelec/2))*AU_TO_EV
         diisdone=.true.
         quick_method%scf_conv=.true.

      endif
      if (PRMS < 1.0d-4) then
         diisoff=.true.
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
      call MPI_BCAST(quick_qm_struct%co,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_qm_struct%E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_method%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_method%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
   endif
#endif

   if (quick_method%debug)  call debug_SCF(jscf)
  enddo

#if (defined CUDA || defined CUDA_MPIV) && !defined(HIP)
    if(master .and. write_molden) then
        quick_molden%nscf_snapshots(quick_molden%iexport_snapshot)=jscf
    endif

     ! sign of the coefficient matrix resulting from cusolver is not consistent
     ! with rest of the code (e.g. gradients). We have to correct this.
    call scalarMatMul(quick_qm_struct%co,nbasis,nbasis,-1.0d0)
#endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
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

!    call deallocate_quick_scf(ierr)

    return
end subroutine electdiisdc



!*******************************************************
! fermiSCF(fermidone
!-------------------------------------------------------
! Use iteriation method to calculate fermi energy
!
!
subroutine fermiSCF(efermi,jscf)
   use allmod
   implicit double precision(a-h,o-z)
   logical fermidone
   integer jscf
   double precision :: efermi(10)

   ! Boltzmann constant in eV/Kelvin:
   boltz = 8.617335408d0*0.00001d0/27.2116d0
   tempk=1000.0d0
   betah = 1.0d0/(boltz*tempk)
   etoler = 1.0d-8
   maxit = 100
   elecs=quick_molspec%nelec


   ! Determine the observed range of eigenvalues for all subsystems.
   emin = 10000000.0d0
   emax =-10000000.0d0
   if(np > 1)then
      do itt=1,np
         imin = nelecdcsub(itt)/2
         imax = nelecdcsub(itt)/2+1
         if(mod(nelecdcsub(itt),2).eq.1)imax = nelecdcsub(itt)/2+2
         emin = min(emin,evaldcsub(itt,imin))
         emax = max(emax,evaldcsub(itt,imax))
      enddo
   endif

   ! Use a bisection technique to determine the fermi energy.
   !        emax =emax+2.0d0
   elecs = quick_molspec%nelec
   niter = 0
   fermidone = .false.

   ! Use iteriation method to calculate fermi energy
   do while ((niter .le. maxit).and.(.not.fermidone))

      efermi(1) = (emax + emin)/2.0d0

      ! Set fermi occupation numbers and use the sum of weighted squared
      ! eigenvector coefficients evecsq to get the number of electrons.
      niter = niter + 1
      etotal = 0.0d0

      do ii=1,nbasis
         do jj=1,nbasis
            quick_qm_struct%dense(ii,jj)=0.0d0
         enddo
      enddo

      !--------------------------------------------
      ! Now Begin to transfer DENSE of subsystem to full system
      ! Puv(a)=2puv*sigma(fb(Ef-ei)*Cui(sub a)*Cvi(sub a)
      ! where puv is partion matrix=1,1/2 or 0
      ! fb is Fermi function
      ! Ef is the eigen energy
      ! Ei is Fermi energy which is unknow but can be get from normlization constraint
      !--------------------------------------------

      do itt=1,np
         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               Pdcsubtran(I,J,itt)=0.0d0
            enddo
         enddo
      enddo
      do itt=1,np
         do K=1,nbasisdc(itt)
            deltae = evaldcsub(itt,K) - efermi(1)
            arg = betah*deltae
            ! Fermi Function part
            if(arg > 25.0d0)then
               fermitemp = 0.0d0
               elseif(arg < -25.0d0)then
               fermitemp = 1.0d0
            else
               fermitemp = 1.0d0/(1.0d0 + exp(arg))
            endif

            if(dabs(fermitemp).ge.0.00000000001d0)then
               do I=1,nbasisdc(itt)
                  if (quick_method%UNRST) then
                     temp1=COdcsub(I,K,itt)*fermitemp
                  else
                     temp1=COdcsub(I,K,itt)*fermitemp*2.0d0
                  endif
                  do J=I,nbasisdc(itt)
                     temp2=COdcsubtran(J,K,itt)
                     DENSEJI=temp1*temp2
                     Pdcsubtran(J,I,itt)=Pdcsubtran(J,I,itt)+DENSEJI
                  enddo
               enddo
            endif
         enddo
      enddo

      ! And Puv are symmatry
      do itt=1,np
         do I=1,nbasisdc(itt)
            do J=I,nbasisdc(itt)
               Pdcsub(itt,I,J)=Pdcsubtran(J,I,itt)
               Pdcsub(itt,J,I)=Pdcsub(itt,I,J)
            enddo
         enddo
      enddo

      !------------------------------------------------
      ! At this step, now we have Density matrix for subsystems for this cycle
      ! What we want to do is to transfer them to full-system density matrix
      !------------------------------------------------
      ! Xiao HE reconsider buffer area 07/17/2008
      do itt=1,np
         Kstart1=0
         do jtt=1,dcsubn(itt)
            Iblockatom=dcsub(itt,jtt)
            do itemp=quick_basis%first_basis_function(Iblockatom),quick_basis%last_basis_function(Iblockatom)
               Kstart2=0
               do jtt2=1,dcsubn(itt)
                  Jblockatom=dcsub(itt,jtt2)
                  if(dclogic(itt,Iblockatom,Jblockatom).eq.0)then
                     do jtemp=quick_basis%first_basis_function(Jblockatom),quick_basis%last_basis_function(Jblockatom)
                        quick_qm_struct%dense(itemp,jtemp)= quick_qm_struct%dense(itemp,jtemp)+ &
                              invdcoverlap(Iblockatom,Jblockatom)* &
                              Pdcsub(itt,Kstart1+itemp-quick_basis%first_basis_function(Iblockatom)+1, &
                              Kstart2+jtemp-quick_basis%first_basis_function(Jblockatom)+1)
                     enddo
                  endif
                  Kstart2=Kstart2+quick_basis%last_basis_function(Jblockatom)-quick_basis%first_basis_function(Jblockatom)+1
               enddo
            enddo
            Kstart1=Kstart1+quick_basis%last_basis_function(Iblockatom)-quick_basis%first_basis_function(Iblockatom)+1
         enddo
      enddo

      !------------------------------------------------
      ! The accuracy of this step is depend on the fermi energy with under normalization constraint
      ! Now calculate normalization and compare with no. of elections to rule the accurate of fermi energy
      !------------------------------------------------

      temp=0.0d0
      do i=1,nbasis
         do j=1,nbasis
            temp=temp+quick_qm_struct%dense(j,i)*quick_qm_struct%s(j,i)
         enddo
      enddo

      diff = abs(temp - elecs)

      !------------------------------------------------
      ! Throw away lower or upper half of interval, depending upon whether
      ! the computed number of electrons is too low or too high.  Note that
      ! raising the fermi energy increases the number of electrons because
      ! more mo's become fully occupied.
      !------------------------------------------------

      if (temp < elecs) then
         emin = efermi(1)
      else
         emax = efermi(1)
      endif


      if(diff < etoler) fermidone=.true.
   enddo

   ! If can't converge, it will lead to fatal error at late step, but quite common
   ! for first or second iteration
   if (.not.fermidone) then
      write(ioutfile,*) "Exceed the maximum interations"
      write(ioutfile,*) "IF IT APPEARS AT LATE STEP MAY LEAD TO WRONG RESULT!"
   endif

   ! At this point we get the fermi energy (Ei) and store for next cycle.
   write (ioutfile,'("E(Fermi)    = ",f12.7,"  AFTER ",i3, " N.C. Cycles")') efermi(1),niter

   call flush(ioutfile)

end subroutine fermiSCF
