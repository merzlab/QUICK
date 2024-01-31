#include "util.fh"
!

! hfoperatordeltadc
!-------------------------------------------------------
! Xiao HE, Delta density matrix increase is implemented here. 07/07/07 version
subroutine hfoperatordeltadc
   use allmod
   use quick_gaussian_class_module
   use quick_cutoff_module, only: cshell_dnscreen
   use quick_oei_module, only: ekinetic
   use quick_overlap_module, only: gpt
   use quick_cshell_eri_module, only: getCshellEri

   implicit double precision(a-h,o-z)

   double precision cutoffTest,testtmp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   double precision fmmonearrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   double precision fmmtwoarrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   double precision g_table(200)
   integer i,j,k,g_count

   ! The purpose of this subroutine is to form the operator matrix
   ! for a full Hartree-Fock calculation, i.e. the Fock matrix.  The
   ! Fock matrix is as follows:

   ! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
   ! with each possible basis  - 1/2 exchange with each
   ! possible basis.

   ! Note that the Fock matrix is symmetric.

   ! May 15,2002-This code now also does all the HF energy calculation. Ed.

   if(quick_method%printEnergy)then
      quick_qm_struct%Eel=0.d0
      do Ibas=1,nbasis

         Bx = xyz(1,quick_basis%ncenter(Ibas))
         By = xyz(2,quick_basis%ncenter(Ibas))
         Bz = xyz(3,quick_basis%ncenter(Ibas))
         ii = itype(1,Ibas)
         jj = itype(2,Ibas)
         kk = itype(3,Ibas)
         g_count = ii+ii+jj+jj+kk+kk+2

         do Icon=1,ncontract(Ibas)
            do Jcon=1,ncontract(Ibas)

               b = aexp(Icon,Ibas)
               a = aexp(Jcon,Ibas)
               call gpt(a,b,Bx,By,Bz,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      

               ! Kinetic energy.

               quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Ibas,Ibas)* &
                     dcoeff(Jcon,Ibas)*dcoeff(Icon,Ibas)* &
                      ekinetic(a,b,ii ,jj,kk,ii,jj,kk,Bx,By,Bz,Bx,By,Bz,Px,Py,Pz,g_table) 
!                     ekinetic(aexp(Jcon,Ibas),aexp(Icon,Ibas), &
!                     itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                     itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                     xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)), &
!                     xyz(3,quick_basis%ncenter(Ibas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                     xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

               ! Nuclear attraction.

               do iatom = 1,natom
                  quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Ibas,Ibas)* &
                        dcoeff(Jcon,Ibas)*dcoeff(Icon,Ibas)* &
                        attraction(aexp(Jcon,Ibas),aexp(Icon,Ibas), &
                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)), &
                        xyz(3,quick_basis%ncenter(Ibas)),xyz(1,quick_basis%ncenter(Ibas)), &
                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                        xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                        quick_molspec%chg(iatom))
               enddo
            enddo
         enddo
      enddo

      do Ibas=1,nbasis
         do Jbas=Ibas+1,nbasis

            Ax = xyz(1,quick_basis%ncenter(Jbas))
            Bx = xyz(1,quick_basis%ncenter(Ibas))
            Ay = xyz(2,quick_basis%ncenter(Jbas))
            By = xyz(2,quick_basis%ncenter(Ibas))
            Az = xyz(3,quick_basis%ncenter(Jbas))
            Bz = xyz(3,quick_basis%ncenter(Ibas))
            ii = itype(1,Ibas)
            jj = itype(2,Ibas)
            kk = itype(3,Ibas)
            i = itype(1,Jbas)
            j = itype(2,Jbas)
            k = itype(3,Jbas)
            g_count = i+ii+j+jj+k+kk+2

            do Icon=1,ncontract(ibas)
               do Jcon=1,ncontract(jbas)

                  b = aexp(Icon,Ibas)
                  a = aexp(Jcon,Jbas)
                  call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      
                  ! Kinetic energy.

                  quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Jbas,Ibas)* &
                        dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                        2.d0* &
                      ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                        ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                        itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                        xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                        xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

                  ! Nuclear attraction.

                  do iatom = 1,natom
                     quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Jbas,Ibas)* &
                           dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                           2.d0*attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                           itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                           itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                           xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                           xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                           xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                           xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                           quick_molspec%chg(iatom))
                  enddo
               enddo
            enddo
         enddo
      enddo

   endif
   !
   ! Alessandro GENONI 03/21/2007
   ! Sum the ECP integrals to the partial Fock matrix
   !
   if (quick_method%ecp) then
      call ecpoperator
   end if
   !
   ! The previous two terms are the one electron part of the Fock matrix.
   ! The next two terms define the two electron part.
   !

   do II=1,jshell
      do JJ=II,jshell
         DNtemp=0.0d0
         call cshell_dnscreen(II,JJ,DNtemp)
         Cutmatrix(II,JJ)=DNtemp
         Cutmatrix(JJ,II)=DNtemp
      enddo
   enddo

   ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
   ! Reference: Strout DL and Scuseria JCP 102(1995),8448.

   do II=1,jshell
      do JJ=II,jshell
         Testtmp=Ycutoff(II,JJ)
         !         tbd1=quick_basis%gcexpomin(II)+quick_basis%gcexpomin(JJ)
         do KK=II,jshell
            do LL=KK,jshell
               !               tbd2=quick_basis%gcexpomin(KK)+quick_basis%gcexpomin(LL)
               testCutoff = TESTtmp*Ycutoff(KK,LL)
               if(testCutoff.gt.quick_method%integralCutoff)then
                  DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                        cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                  if((dcconnect(II,JJ).eq.1.and.(4.0d0*cutmatrix(KK,LL)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(KK,LL).eq.1.and.(4.0d0*cutmatrix(II,JJ)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(II,KK).eq.1.and.(cutmatrix(JJ,LL)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(LL,II).eq.1.and.(cutmatrix(JJ,KK)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(JJ,KK).eq.1.and.(cutmatrix(II,LL)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(JJ,LL).eq.1.and.(cutmatrix(II,KK)*testCutoff).gt.quick_method%integralCutoff))then

                     call getCshellEri(II)
                  endif
               endif

            enddo
         enddo
      enddo
   enddo


   do Ibas=1,nbasis
      do Jbas=Ibas+1,nbasis
         quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
      enddo
   enddo

   if(quick_method%printEnergy)then
      do Ibas=1,nbasis
         do Jbas=1,nbasis
            quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Ibas,Jbas)*quick_qm_struct%o(Jbas,Ibas)
         enddo
      enddo

      quick_qm_struct%Eel=quick_qm_struct%Eel/2.0d0
   endif

   return
end subroutine hfoperatordeltadc

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
     use allmod
     use quick_cutoff_module, only: cshell_density_cutoff
     use quick_cshell_eri_module, only: getCshellEriDC, getCshellEriEnergy
     use quick_oei_module, only:get1eEnergy,get1e

     implicit none

#ifdef MPIV
     include "mpif.h"
#endif
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
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif


#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
     if (quick_method%bCUDA) then

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
        if (quick_method%bCUDA) then
           call gpu_get_cshell_eri(deltaO, quick_qm_struct%o)
        else
#endif
  !  Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
  !  Reference: Strout DL and Scuseria JCP 102(1995),8448.

#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
  !  Every nodes will take about jshell/nodes shells integrals such as 1 water,
  !  which has 
  !  4 jshell, and 2 nodes will take 2 jshell respectively.
     if(bMPI) then
        do i=1,mpi_jshelln(mpirank)
           ii=mpi_jshell(mpirank,i)
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
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
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
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif

     if (quick_method%DFT) then

  !  Start the timer for exchange correlation calculation
        RECORD_TIME(timer_begin%TEx)

  !  Calculate exchange correlation contribution & add to operator    
!        call get_xc(deltaO)

  !  Remember the operator is symmetric
        call copySym(quick_qm_struct%o,nbasis)

#ifdef MPIV
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif

  !  Stop the exchange correlation timer
        RECORD_TIME(timer_end%TEx)

  !  Add time total time
        timer_cumer%TEx=timer_cumer%TEx+timer_end%TEx-timer_begin%TEx
     endif

     quick_qm_struct%oSave(:,:) = quick_qm_struct%o(:,:)

#ifdef MPIV
  !  MPI reduction operations

     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

     RECORD_TIME(timer_begin%TEred)

     if (quick_method%DFT) then
     call MPI_REDUCE(quick_qm_struct%Exc, Excsum, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
     call MPI_REDUCE(quick_qm_struct%aelec, aelec, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
     call MPI_REDUCE(quick_qm_struct%belec, belec, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)

     if(master) then
       quick_qm_struct%Exc = Excsum
       quick_qm_struct%aelec  = aelec
       quick_qm_struct%belec  = belec
     endif
     endif

     call MPI_REDUCE(quick_qm_struct%o, quick_scratch%osum, nbasis*nbasis, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
     call MPI_REDUCE(quick_qm_struct%Eel, Eelsum, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)

     if(master) then
       quick_qm_struct%o(:,:) = quick_scratch%osum(:,:)
       quick_qm_struct%Eel    = Eelsum
     endif

     RECORD_TIME(timer_end%TEred)
     timer_cumer%TEred=timer_cumer%TEred+timer_end%TEred-timer_begin%TEred

#endif

  return

  end subroutine scf_operatordc
