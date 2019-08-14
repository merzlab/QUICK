#include "config.h"

subroutine newdftoperator(oneElecO, deltaO)
   !-------------------------------------------------------
   ! The purpose of this subroutine is to form the operator matrix
   ! for a full Hartree-Fock calculation, i.e. the Fock matrix.  The
   ! Fock matrix is as follows:  O(I,J) =  F(I,J) = KE(I,J) + IJ
   ! attraction
   ! to each atom + repulsion_prim
   ! with each possible basis  - 1/2 exchange with each
   ! possible basis. Note that the Fock matrix is symmetric.
   ! This code now also does all the HF energy calculation. Ed.
   !-------------------------------------------------------
   use allmod
   use quick_gaussian_class_module
   implicit none


   double precision oneElecO(nbasis,nbasis)
   logical :: deltaO
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, I, J
   integer iatm, Iradtemp, Irad, iang
   double precision rad, rad3, gridx, gridy, gridz, weight, phi, dphidx&
   ,dphidy, dphidz, density, densityb,gax,gay,gaz, gbx,gby,gbz, dfdr &
   ,dfdgaa,dfdgab, dfdr2,dfdgaa2,dfdgab2, xdot, ydot, zdot, quicktest&
   ,phi2, dphi2dx, dphi2dy, dphi2dz, temp, tempgx, tempgy, tempgz
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   !-----------------------------------------------------------------
   ! Step 1. evaluate 1e integrals
   !-----------------------------------------------------------------

!---------------------Madu-----------------
!   if(master) then
!    do i=1,nbasis
!     do j=1,nbasis
!       write(*,*) "Madu: CPU before 1e O",quick_qm_struct%o(i,j)
!     enddo
!    enddo
!   endif

!---------------------Madu-----------------

   ! fetch 1e-integral from 1st time
   call copyDMat(oneElecO,quick_qm_struct%o,nbasis)

   if(master) then
    do i=1,nbasis
     do j=1,nbasis
!       write(*,*) "Madu: CPU after 1e O",quick_qm_struct%o(i,j)
     enddo
    enddo
   endif

!stop

   ! Now calculate kinetic and attraction energy first.
   if (quick_method%printEnergy) call get1eEnergy()

   ! Alessandro GENONI 03/21/2007
   ! Sum the ECP integrals to the partial Fock matrix
   if (quick_method%ecp) call ecpoperator()

   ! if only calculate operation difference
   if (deltaO) then
      ! save density matrix
      call CopyDMat(quick_qm_struct%dense,quick_qm_struct%denseSave,nbasis)
      call CopyDMat(quick_qm_struct%oSave,quick_qm_struct%o,nbasis)

      do I=1,nbasis; do J=1,nbasis
         quick_qm_struct%dense(J,I)=quick_qm_struct%dense(J,I)-quick_qm_struct%denseOld(J,I)
      enddo; enddo

   endif


   ! Delta density matrix cutoff
   call densityCutoff()

   call cpu_time(timer_begin%T2e)  ! Terminate the timer for 2e-integrals

#ifdef CUDA
   if (quick_method%bCUDA) then
      call gpu_upload_method(0)
      call gpu_upload_calculated(quick_qm_struct%o,quick_qm_struct%co, &
            quick_qm_struct%vec,quick_qm_struct%dense)
      call gpu_upload_cutoff(cutmatrix,quick_method%integralCutoff,quick_method%primLimit)
   endif

#endif

   if (quick_method%nodirect) then
#ifdef CUDA
      call gpu_addint(quick_qm_struct%o, intindex, intFileName)
#else
      call addInt
#endif
   else
      !-----------------------------------------------------------------
      ! Step 2. evaluate 2e integrals
      !-----------------------------------------------------------------

      ! The previous two terms are the one electron part of the Fock
      ! matrix.
      ! The next two terms define the two electron part.


#ifdef CUDA
      if (quick_method%bCUDA) then
         call gpu_get2e(quick_qm_struct%o)
      else
#endif

      ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
      ! Reference: Strout DL and Scuseria JCP 102(1995),8448.
      do II=1,jshell
         call get2e(II)
      enddo

#ifdef CUDA
   endif
#endif
   endif


   if(quick_method%BLYP)then
        call newblyp
   endif



   ! Remember the operator is symmetry
   call copySym(quick_qm_struct%o,nbasis)

   ! Operator matrix
   !   write(ioutfile,'("OPERATOR MATRIX FOR CYCLE")')
   !   call PriSym(iOutFile,nbasis,quick_qm_struct%o,'f14.8')


   ! recover density if calculate difference
   if (deltaO) call CopyDMat(quick_qm_struct%denseSave,quick_qm_struct%dense,nbasis)

   ! Give the energy, E=1/2*sigma[i,j](Pij*(Fji+Hcoreji))
   if(quick_method%printEnergy) call get2eEnergy()


   call cpu_time(timer_end%T2e)  ! Terminate the timer for 2e-integrals
   timer_cumer%T2e=timer_cumer%T2e+timer_end%T2e-timer_begin%T2e ! add
!  the time to cumer

   return

end subroutine newdftoperator 

