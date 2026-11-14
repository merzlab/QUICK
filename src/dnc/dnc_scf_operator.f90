#include "util.fh"

  subroutine scf_operatordc(deltaO)
  !-------------------------------------------------------
  !  The purpose of this subroutine is to form the operator matrix
  !  for a full Hartree-Fock/DFT calculation, i.e. the Fock matrix.  The
  !  Fock matrix is as follows:  O(I,J) =  F(I,J) = KE(I,J) + IJ attraction
  !  to each atom + repulsion_prim
  !  with each possible basis  - 1/2 exchange with each
  !  possible basis. Note that the Fock matrix is symmetric.
  !  This code now also does all the HF energy calculation. Ed.
  !-------------------------------------------------------
     use quick_calculated_module, only: quick_qm_struct
     use quick_method_module, only: quick_method
     use quick_scratch_module, only: quick_scratch
     use quick_basis_module, only: nbasis, jshell, cutmatrix
#if defined(MPIV)
     use quick_basis_module, only: mpi_jshelln, mpi_jshell
#endif
     use quick_timer_module, only: timer_begin, timer_end, timer_cumer
     use quick_cutoff_module, only: cshell_density_cutoff
     use quick_eri_cshell_module, only: getCshellEriDC, getCshellEriEnergy
     use quick_oei_module, only:get1eEnergy,get1e
     use quick_files_module, only: iOutfile
     use quick_mpi_module, only: master
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
     use quick_divcon_module, only: GPUUseCompactO, GPUCompactOBuf, GPUFragOffset
#endif

#if defined(MPIV)
     use quick_mpi_module, only: bMPI, quick_comm, quick_comm_rank, quick_mpi_error
     use mpi
#endif

     implicit none
  !   double precision oneElecO(nbasis,nbasis)
     logical :: deltaO
     integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, I, J
     common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
     double precision tst, te, tred
#ifdef MPIV
     integer ierror
     double precision :: Eelsum, Excsum, aelec, belec

     quick_scratch%osum=0.0d0
     Eelsum=0.0d0
     Excsum=0.0d0
     aelec=0.0d0
     belec=0.0d0
#endif

     quick_qm_struct%o = 0.0d0
     quick_qm_struct%Eel=0.0d0

  !-----------------------------------------------------------------
  !  Step 1. evaluate 1e integrals
  !-----------------------------------------------------------------

  !  if only calculate operation difference
     if (deltaO) then
  !     save density matrix
        quick_qm_struct%denseSave(:,:) = quick_qm_struct%dense(:,:)
        quick_qm_struct%dense=quick_qm_struct%dense-quick_qm_struct%denseOld

        if(quick_method%dft) then
          quick_qm_struct%o = quick_qm_struct%oSave-quick_qm_struct%oxc
        else
          quick_qm_struct%o(:,:) = quick_qm_struct%oSave(:,:)
        endif
     endif

  !  Delta density matrix cutoff
     call cshell_density_cutoff

#ifdef MPIV
     call MPI_BARRIER(quick_comm,quick_mpi_error)
#endif


#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
     if (quick_method%bGPU) then

        if (quick_method%DivCon .and. quick_method%densdiis .and. &
            .not.quick_method%dcmp2only .and. .not.quick_method%MP2 .and. &
            .not.quick_method%grad .and. .not.quick_method%opt .and. &
            .not.quick_method%freq .and. master) then
           write(iOutfile,'(" @ DEBUG [DnC+DENSDIIS, single-point]: uploading dense O+P(density) to &
              &GPU this iteration, nbasis=",i0,", ~",f10.3," GB each (~",f10.3," GB combined)")') &
              nbasis, dble(nbasis)*dble(nbasis)*8.0d0/1024.0d0**3, &
              2.0d0*dble(nbasis)*dble(nbasis)*8.0d0/1024.0d0**3
           call flush(iOutfile)
        endif

        call gpu_upload_calculated(quick_qm_struct%o,quick_qm_struct%co, &
        quick_qm_struct%vec,quick_qm_struct%dense)
        call gpu_upload_cutoff(cutmatrix,quick_method%integralCutoff,quick_method%primLimit,quick_method%DMCutoff, &
                                quick_method%coreIntegralCutoff)

     endif
#endif

     call get1e(deltaO)

     if(quick_method%printEnergy) call get1eEnergy(deltaO)

!write(*,*) "1e energy:", quick_qm_struct%Eel 

!     if (quick_method%nodirect) then
!#ifdef CUDA
!        call gpu_addint(quick_qm_struct%o, intindex, intFileName)
!#else
!#ifndef MPI
!        call addInt
!#endif
!#endif
!     else
  !-----------------------------------------------------------------
  ! Step 2. evaluate 2e integrals
  !-----------------------------------------------------------------
  !
  ! The previous two terms are the one electron part of the Fock matrix.
  ! The next two terms define the two electron part.
  !-----------------------------------------------------------------
  !  Start the timer for 2e-integrals
     RECORD_TIME(timer_begin%T2e)

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
        if (quick_method%bGPU) then
           call gpu_get_cshell_eri(deltaO, quick_qm_struct%o)

           ! O-compaction (9.14): when active, gpu_get_cshell_eri just
           ! downloaded the compact per-fragment-block O buffer on the C++
           ! side (gpu->gpu_calculated->oCompact) instead of folding a dense
           ! O into quick_qm_struct%o. Retrieve it and scatter+fold it in
           ! here. GPUUseCompactO reflects the actual per-build outcome
           ! (inidivcon.f90), not just the upload request, so this is a
           ! no-op dense-O build/open-shell/legacy-atomics.
           if (quick_method%DivCon .and. quick_method%densdiis .and. &
               .not.quick_method%dcmp2only .and. .not.quick_method%MP2 .and. &
               .not.quick_method%grad .and. .not.quick_method%opt .and. &
               .not.quick_method%freq .and. GPUUseCompactO) then
              call gpu_download_compact_o(GPUCompactOBuf)
              call GatherCompactO(GPUCompactOBuf, GPUFragOffset)
           endif
        else
#endif
  !  Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
  !  Reference: Strout DL and Scuseria JCP 102(1995),8448.

#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
  !  Every nodes will take about jshell/nodes shells integrals such as 1 water,
  !  which has 
  !  4 jshell, and 2 nodes will take 2 jshell respectively.
     if(bMPI) then
        do i=1,mpi_jshelln(quick_comm_rank)
           ii=mpi_jshell(quick_comm_rank,i)
           call getCshellEriDC(II)
        enddo
     else
        do II=1,jshell
           call getCshellEriDC(II)
        enddo
     endif
#else
        do II=1,jshell
           call getCshellEriDC(II)
        enddo
#endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
        endif
#endif
 !    endif

  !  Remember the operator is symmetric
     call copySym(quick_qm_struct%o,nbasis)

  !  recover density if calculate difference
     if (deltaO) quick_qm_struct%dense(:,:) = quick_qm_struct%denseSave(:,:)

  !  Give the energy, E=1/2*sigma[i,j](Pij*(Fji+Hcoreji))
     if(quick_method%printEnergy) call getCshellEriEnergy

!write(*,*) "2e Energy added", quick_qm_struct%Eel


#ifdef MPIV
     call MPI_BARRIER(quick_comm,quick_mpi_error)
#endif

  !  Terminate the timer for 2e-integrals
     RECORD_TIME(timer_end%T2e)

  !  add the time to cumer
     timer_cumer%T2e=timer_cumer%T2e+timer_end%T2e-timer_begin%T2e

  !-----------------------------------------------------------------
  !  Step 3. If DFT, evaluate the exchange/correlation contribution 
  !          to the operator
  !-----------------------------------------------------------------

#ifdef MPIV
     call MPI_BARRIER(quick_comm,quick_mpi_error)
#endif

     if (quick_method%DFT) then

  !  Start the timer for exchange correlation calculation
        RECORD_TIME(timer_begin%TEx)

  !  Calculate exchange correlation contribution & add to operator    
!        call get_xc(deltaO)

  !  Remember the operator is symmetric
        call copySym(quick_qm_struct%o,nbasis)

#ifdef MPIV
     call MPI_BARRIER(quick_comm,quick_mpi_error)
#endif

  !  Stop the exchange correlation timer
        RECORD_TIME(timer_end%TEx)

  !  Add time total time
        timer_cumer%TEx=timer_cumer%TEx+timer_end%TEx-timer_begin%TEx
     endif

     quick_qm_struct%oSave(:,:) = quick_qm_struct%o(:,:)

#ifdef MPIV
  !  MPI reduction operations

     call MPI_BARRIER(quick_comm,quick_mpi_error)

     RECORD_TIME(timer_begin%TEred)

     if (quick_method%DFT) then
     call MPI_REDUCE(quick_qm_struct%Exc, Excsum, 1, mpi_double_precision, MPI_SUM, 0, quick_comm, IERROR)
     call MPI_REDUCE(quick_qm_struct%aelec, aelec, 1, mpi_double_precision, MPI_SUM, 0, quick_comm, IERROR)
     call MPI_REDUCE(quick_qm_struct%belec, belec, 1, mpi_double_precision, MPI_SUM, 0, quick_comm, IERROR)

     if(master) then
       quick_qm_struct%Exc = Excsum
       quick_qm_struct%aelec  = aelec
       quick_qm_struct%belec  = belec
     endif
     endif

     call MPI_REDUCE(quick_qm_struct%o, quick_scratch%osum, nbasis*nbasis, mpi_double_precision, MPI_SUM, 0, quick_comm, IERROR)
     call MPI_REDUCE(quick_qm_struct%Eel, Eelsum, 1, mpi_double_precision, MPI_SUM, 0, quick_comm, IERROR)

     if(master) then
       quick_qm_struct%o(:,:) = quick_scratch%osum(:,:)
       quick_qm_struct%Eel    = Eelsum
     endif

     RECORD_TIME(timer_end%TEred)
     timer_cumer%TEred=timer_cumer%TEred+timer_end%TEred-timer_begin%TEred

#endif

  return

  end subroutine scf_operatordc
