#include "util.fh"
!
!	getEnergy.f90
!	new_quick
!
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   written by Ed Brothers. 08/15/02
!   This subroutine calculates and ouptus the energy.
!
subroutine getEnergy(isGuess, ierr)
   use allMod
   use quick_gridpoints_module
   use quick_scf_module
   use quick_uscf_module, only: uscf
   use quick_overlap_module, only: fullx
   use quick_dftd3_module, only: calculateDFTD3 
   use quick_exception_module
#ifdef CEW
   use quick_cew_module, only : quick_cew
#endif
#ifdef MPIV
   use mpi
#endif
#ifdef CUEST
   use quick_cuest_module, only: cuest_correct_P, CUEST_CORRECT_REORDER_AND_NORM_QUICK_TO_CUEST
#endif

   implicit none

   double precision :: distance
   double precision, external :: rootSquare
   integer i,j
   logical, intent(in) :: isGuess
   integer, intent(inout) :: ierr
   logical :: verbose
   integer nocc,alpha

   verbose = .true.
   if ( isGuess .and. (.not. quick_method%writeSAD) ) verbose = .false.

    !Form the exchange-correlation quadrature if DFT is requested
    if ( ( quick_method%DFT &
#ifdef CEW
        .or. quick_cew%use_cew &
#endif
        ) .and. .not. isGuess ) then

        if (master) call PrtAct(ioutfile,"Begin XC Quadrature Formation")

        call form_dft_grid(quick_dft_grid, quick_xcg_tmp)

        if (master) call print_grid_info(quick_dft_grid)

        if (master) call PrtAct(ioutfile,"End XC Quadrature Formation")
    endif

   if (master) then
      if (verbose) call PrtAct(ioutfile,"Begin Energy Calculation")
      ! Build a transformation matrix X and overlap matrix
      call fullX

      ! ------------------!
      ! force idempotency !
      ! ------------------!
      if (quick_method%sadmo) then
#ifdef CUEST
         if (quick_method%usecuest) then
            call cuest_correct_P(quick_qm_struct%dense, CUEST_CORRECT_REORDER_AND_NORM_QUICK_TO_CUEST)
            if (quick_method%unrst) then
               call cuest_correct_P(quick_qm_struct%denseb, CUEST_CORRECT_REORDER_AND_NORM_QUICK_TO_CUEST)
            endif
         endif
#endif

         if (.not. allocated(quick_scratch%hold)) allocate(quick_scratch%hold(nbasis, nbasis))
         if (.not. allocated(quick_scratch%hold2)) allocate(quick_scratch%hold2(nbasis, nbasis))
         if (.not. allocated(quick_scratch%hold3)) allocate(quick_scratch%hold3(nbasis, nbasis))
         if (.not. allocated(quick_scratch%hold4)) allocate(quick_scratch%hold4(nbasis, nbasis))
         if (.not. allocated(quick_scratch%tmphold)) allocate(quick_scratch%tmphold(nbasis, nbasis))
         if (.not. allocated(quick_scratch%Sminhalf)) allocate(quick_scratch%Sminhalf(nbasis))

         ! %hold  = S^{1/2}PS^{1/2} =: P'
         ! %hold4 = S^{1/2}  (do not modify)
         ! %hold3 = S^{-1/2} (do not modify)
         call lowdin_orth(nbasis, quick_qm_struct%s, quick_qm_struct%dense, &
                          quick_scratch%hold4, .true., quick_scratch%hold3, quick_scratch%hold)

         ! %hold2 =: C', where P' = C'NC'^T, N = diag(1,...,1,0,...0)
         call MAT_DIAG(quick_scratch%hold, nbasis, nbasis, quick_scratch%Sminhalf, quick_scratch%tmphold)
         do j = 1, nbasis
            quick_scratch%hold2(:, j) = quick_scratch%tmphold(:, nbasis - j + 1)
         enddo

         ! set (alpha) Nocc
         if (quick_method%unrst) then
            nocc = quick_molspec%nelec
            alpha = 1
         else
            nocc = quick_molspec%nelec/2
            alpha = 2
         endif

         quick_qm_struct%co = 0.0d0
         ! %co = (%hold3)*(%hold2) S^{-1/2}C' = C
         call MAT_DGEMM('n', 'n', nbasis, nocc, nbasis, 1.0d0, quick_scratch%hold3, nbasis, &
                        quick_scratch%hold2, nbasis, 0.0d0, quick_qm_struct%co, nbasis)

         ! %dense = (%co)*(%co)^T = CNC^T
         call MAT_DGEMM ('n', 't', nbasis, nbasis, nocc, dble(alpha), quick_qm_struct%co, &
                         nbasis, quick_qm_struct%co, nbasis, 0.0d0, quick_qm_struct%dense, nbasis)

         if (quick_method%unrst) then
            ! %hold2 = (%hold4)*(%denseb) = S^{1/2}P
            call MAT_DGEMM('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_scratch%hold4, nbasis, &
                           quick_qm_struct%denseb, nbasis, 0.0d0, quick_scratch%hold2, nbasis)
            ! %hold = (%hold2)*(%hold4) = S^{1/2}PS^{1/2}
            call MAT_DGEMM('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_scratch%hold2, nbasis, &
                           quick_scratch%hold4, nbasis, 0.0d0, quick_scratch%hold, nbasis)
            ! C' = %hold2
            call MAT_DIAG(quick_scratch%hold, nbasis, nbasis, quick_scratch%Sminhalf, quick_scratch%tmphold)
            do j = 1, nbasis
               quick_scratch%hold2(:, j) = quick_scratch%tmphold(:, nbasis - j + 1)
            enddo

            quick_qm_struct%cob = 0.0d0
            call MAT_DGEMM('n', 'n', nbasis, quick_molspec%nelecb, nbasis, 1.0d0, quick_scratch%hold3, nbasis, &
                           quick_scratch%hold2, nbasis, 0.0d0, quick_qm_struct%cob, nbasis)

            call MAT_DGEMM ('n', 't', nbasis, nbasis, quick_molspec%nelecb, dble(alpha), quick_qm_struct%cob, &
                            nbasis, quick_qm_struct%cob, nbasis, 0.0d0, quick_qm_struct%denseb, nbasis)
         endif

         deallocate(quick_scratch%hold4)
         deallocate(quick_scratch%hold3)
         deallocate(quick_scratch%tmphold)
         deallocate(quick_scratch%Sminhalf)
      endif

      ! if it's a div-con calculate, construct Div & Con matrices, Overlap,X, and PDC
      !if (quick_method%DivCon) then
      !   call DivideS
      !   call DivideX
      !   call PDCDivided
      !endif

      !Classical Nuclear-Nuclear interaction energy
      quick_qm_struct%Ecore=0.d0      ! atom-extcharge and atom-atom replusion
      quick_qm_struct%ECharge=0d0     ! extcharge-extcharge interaction


      if (natom > 1) then
         !                    qi*qj
         ! E=sigma(i,j=1,n)----------
         !                   |ri-rj|
         do I=1,natom+quick_molspec%nextatom
            do J=I+1,natom+quick_molspec%nextatom
               if(i<=natom .and. j<=natom)then                     ! the atom to atom replusion
                  distance = rootSquare(xyz(1:3,i), xyz(1:3,j), 3)
                  quick_qm_struct%Ecore = quick_molspec%chg(I)*quick_molspec%chg(J)/distance+quick_qm_struct%Ecore
                  elseif(i<=natom .and. j>natom)then                  ! the atom to external point charge replusion
                  distance = rootSquare(xyz(1:3,i), quick_molspec%extxyz(1:3,j-natom), 3)
                  quick_qm_struct%Ecore = quick_molspec%chg(I)*quick_molspec%extchg(J-natom)/distance+quick_qm_struct%Ecore
                  elseif(i>natom .and. j>natom)then                   ! external to external point charge repulsion
                  distance = rootSquare(quick_molspec%extxyz(1:3,i-natom), quick_molspec%extxyz(1:3,j-natom), 3)
                  quick_qm_struct%ECharge = quick_qm_struct%ECharge + &

                        quick_molspec%extchg(I-natom)*quick_molspec%extchg(J-natom)/distance
               endif
            enddo
         enddo
      endif
   endif
   ! Converge the density matrix.
#ifdef MPIV
   !-------------- MPI / ALL NODES ----------------------------------
   if (bMPI) then
      quick_qm_struct%NBSuse => NBSuse

      call MPI_BCAST(NBSuse,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

      if(.not. master) call allocate_quick_qm_struct_fullx(quick_qm_struct)

      call MPI_BCAST(quick_qm_struct%s,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_qm_struct%x,nbasis*NBSuse,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_qm_struct%Ecore,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
   endif
   !-------------- END MPI / ALL NODES ------------------------------
#endif
   ! scf process to obtain converged density matrix and coeffecient so that we can
   ! process to next step, the energy compuation. the restrited system will call scf and
   ! unrestred system will call uscf. the logical variable failed indicated failed convergence.
   ! convergence criteria can be set in the job or default value.
   if (quick_method%UNRST) then
      if(isGuess) then
        !call uscf_sad(verbose,ierr)
        !call uscf_sad_new(ierr)
      else
        call uscf(ierr)       ! unrestricted system
      endif
   else
      call scf(ierr)        ! restricted system
   endif

   !--------------- MPI/MASTER --------------------------
   if (master) then

      ! Fisrt, it is PB model, we need calculate the energy for PB Sol.
      !
      ! Blocked by Yipu Miao
      !
      if(quick_method%PBSOL)then
         if (quick_method%UNRST) then
            !       if (quick_method%HF) call UHFEnergy
            !       if (quick_method%DFT) call uDFTEnergy
            !        if (quick_method%SEDFT) call uSEDFTEnergy
         else
            !        if (quick_method%HF) call HFEnergy
            !        if (quick_method%DFT) call DFTenergy
            !        if (quick_method%SEDFT) call SEDFTenergy
         endif
      endif

      ! Now that we have a converged density matrix, it is time to
      ! calculate the energy.  It equals to the summation of different
      ! parts: electronic energy, core-core repulsion, and some other energy
      ! for specific job
      quick_qm_struct%Eelvac=quick_qm_struct%Eel
      if (quick_method%extcharges) then
         quick_qm_struct%Etot = quick_qm_struct%Etot + quick_qm_struct%ECharge
      endif
      quick_qm_struct%Etot = quick_qm_struct%Eel + quick_qm_struct%Ecore

      ! calculate emperical dispersion correction 
      if(quick_method%edisp) then
         SAFE_CALL(calculateDFTD3(ierr))
         quick_qm_struct%Etot=quick_qm_struct%Etot+quick_qm_struct%Edisp
      endif

      if (ioutfile.ne.0 .and. verbose) then
         write (ioutfile,'(" ELECTRONIC ENERGY    = ",F16.9)') quick_qm_struct%Eel
         write (ioutfile,'(" CORE_CORE REPULSION  = ",F16.9)') quick_qm_struct%Ecore
         if(quick_method%edisp) then
            write (ioutfile,'(" DISPERSION CORRECTION  = ",F16.9)') quick_qm_struct%Edisp
         endif
         if (quick_method%extcharges) then
            write (ioutfile,'(" EXT CHARGE REPULSION = ",F16.9)') quick_qm_struct%ECharge
         endif
         write (ioutfile,'(" TOTAL ENERGY         = ",F16.9)') quick_qm_struct%Etot
         call prtact(ioutfile,"End Energy calculation")
         call flush(ioutfile)
      endif
   endif
   !--------------- END MPI/MASTER ----------------------

#if defined(MPIV) || defined(MPIV_GPU)
  call MPI_BCAST(quick_qm_struct%Etot, 1, mpi_double_precision,0,MPI_COMM_WORLD,mpierror) 
#endif

   ! -----------------------------------  !
   ! print eigenvalues of S^{1/2}PS^{1/2} !
   ! -----------------------------------  !

   ! if (quick_method%usecuest) then
   !    if (.not. allocated(quick_scratch%hold2)) allocate(quick_scratch%hold2(nbasis, nbasis))
   !    if (.not. allocated(quick_scratch%tmphold)) allocate(quick_scratch%tmphold(nbasis, nbasis))
   !    if (.not. allocated(quick_scratch%Sminhalf)) allocate(quick_scratch%Sminhalf(nbasis))
   !
   !    ! %hold2 = S^{1/2}PS^{1/2}
   !    ! %tmphold = S^{1/2}
   !    call lowdin_orth(nbasis, quick_qm_struct%s, quick_qm_struct%dense/2.0d0, &
   !                     quick_scratch%tmphold, .false., quick_scratch%tmphold, quick_scratch%hold2)
   !    ! %Sminhalf = eigenvalues
   !    ! %tmphold = eigenvector matrix (not needed)
   !    call MAT_DIAG(quick_scratch%hold2, nbasis, nbasis, quick_scratch%Sminhalf, quick_scratch%hold2)
   !
   !    write (ioutfile,'(" EIGENVECTORS OF LOWDIN ORTHOGONALIZED DENSITY MATRIX: ")')
   !    write (ioutfile,'(53("-"))')
   !    do i=1, nbasis
   !       write(ioutfile, '(F14.10)') quick_scratch%Sminhalf(i)
   !    enddo
   !    write (ioutfile,'(53("-"))')
   !
   !    deallocate(quick_scratch%hold2)
   !    deallocate(quick_scratch%tmphold)
   !    deallocate(quick_scratch%Sminhalf)
   ! endif

end subroutine getenergy
