#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 03/24/2021                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains all one electron integral (oei) & oei gradient ! 
! code.                                                               !
!---------------------------------------------------------------------!

module quick_oei_module

  implicit double precision(a-h,o-z)
  private

  public :: get1eEnergy, get1e, attrashellopt, attrashellfock1, ekinetic, kineticO, attrashell
  public :: bCalc1e

  logical :: bCalc1e = .false.

contains

  !------------------------------------------------
  ! get1eEnergy
  !------------------------------------------------
  subroutine get1eEnergy(deltaO)
     !------------------------------------------------
     ! This subroutine is to get 1e integral
     !------------------------------------------------
     use quick_calculated_module, only: quick_qm_struct
     use quick_method_module,only: quick_method
     use quick_timer_module, only:timer_begin, timer_end, timer_cumer
     use quick_basis_module

     implicit double precision(a-h,o-z)

     logical, intent(in) :: deltaO

     RECORD_TIME(timer_begin%tE)
  
     if(.not. deltaO) quick_qm_struct%E1e=0.0d0
     quick_qm_struct%E1e=quick_qm_struct%E1e+sum2mat(quick_qm_struct%dense,quick_qm_struct%oneElecO,nbasis)

     if (quick_method%unrst) then
       quick_qm_struct%E1e = quick_qm_struct%E1e+sum2mat(quick_qm_struct%denseb,quick_qm_struct%oneElecO,nbasis)
     endif

     quick_qm_struct%Eel=quick_qm_struct%E1e

     RECORD_TIME(timer_end%tE)
     timer_cumer%TE=timer_cumer%TE+timer_end%TE-timer_begin%TE
  
  end subroutine get1eEnergy

subroutine get1e(deltaO)
   use allmod

#ifdef CEW
   use quick_cew_module, only : quick_cew, quick_cew_prescf
#endif

#ifdef MPIV
   use mpi
#endif
   
   implicit double precision(a-h,o-z)
   double precision :: temp2d(nbasis,nbasis)
   logical, intent(in) :: deltaO

   !------------------------------------------------
   ! This subroutine is to obtain Hcore, and store it
   ! to oneElecO so we don't need to calculate it repeatly for
   ! every scf cycle
   !------------------------------------------------


#ifdef MPIV
   if ((.not.bMPI).or.(nbasis.le.MIN_1E_MPI_BASIS)) then
#endif

     if (master) then
       RECORD_TIME(timer_begin%T1e)
       if(bCalc1e) then

         !=================================================================
         ! Step 1. evaluate 1e integrals
         !-----------------------------------------------------------------
         ! The first part is kinetic part
         ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
         !-----------------------------------------------------------------
         RECORD_TIME(timer_begin%T1eT)
         do Ibas=1,nbasis
            call kineticO(Ibas)
         enddo
         RECORD_TIME(timer_end%T1eT)


         !-----------------------------------------------------------------
         ! The second part is attraction part
         !-----------------------------------------------------------------
         RECORD_TIME(timer_begin%T1eV)

#if defined CUDA || defined HIP
         if(.not. quick_method%hasF) then
           call gpu_get_oei(quick_qm_struct%o)
         else

           do IIsh=1,jshell
              do JJsh=IIsh,jshell
                 call attrashell(IIsh,JJsh)
              enddo
           enddo
         endif
#else
         do IIsh=1,jshell
            do JJsh=IIsh,jshell
               call attrashell(IIsh,JJsh)
            enddo
         enddo
#endif

         RECORD_TIME(timer_end%T1eV)

         timer_cumer%T1eT=timer_cumer%T1eT+timer_end%T1eT-timer_begin%T1eT
         timer_cumer%T1eV=timer_cumer%T1eV+timer_end%T1eV-timer_begin%T1eV

#ifdef CEW
         if ( quick_cew%use_cew ) then
            
            RECORD_TIME(timer_begin%Tcew)

            call quick_cew_prescf()

            RECORD_TIME(timer_end%Tcew)

            timer_cumer%Tcew=timer_cumer%Tcew+timer_end%Tcew-timer_begin%Tcew

         end if
#endif
         
         call copySym(quick_qm_struct%o,nbasis)

         quick_qm_struct%oneElecO(:,:) = quick_qm_struct%o(:,:)

         if (quick_method%debug) then
                write(iOutFile,*) "ONE ELECTRON MATRIX"
                call PriSym(iOutFile,nbasis,quick_qm_struct%oneElecO,'f14.8')
         endif
         bCalc1e=.false.

       else
         if (.not. deltaO) quick_qm_struct%o(:,:)=quick_qm_struct%oneElecO(:,:)
       endif
       RECORD_TIME(timer_end%t1e)

       timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
       timer_cumer%TOp = timer_cumer%TOp+timer_end%T1e-timer_begin%T1e
       timer_cumer%TSCF = timer_cumer%TSCF+timer_end%T1e-timer_begin%T1e

     endif
#ifdef MPIV
   else
    RECORD_TIME(timer_begin%t1e)
    if(bCalc1e) then

      !------- MPI/ ALL NODES -------------------

      !=================================================================
      ! Step 1. evaluate 1e integrals
      ! This job is only done on master node since it won't cost much resource
      ! and parallel will even waste more than it saves
      !-----------------------------------------------------------------
      ! The first part is kinetic part
      ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
      !-----------------------------------------------------------------
      RECORD_TIME(timer_begin%T1eT)

      do i=1,mpi_nbasisn(mpirank)
         Ibas=mpi_nbasis(mpirank,i)
         call kineticO(Ibas)
      enddo
      RECORD_TIME(timer_end%T1eT)

      !-----------------------------------------------------------------
      ! The second part is attraction part
      !-----------------------------------------------------------------
      RECORD_TIME(timer_begin%T1eV)

#if defined CUDA_MPIV || defined HIP_MPIV
      if(.not. quick_method%hasF) then
        call gpu_get_oei(quick_qm_struct%o)
      else
        do i=1,mpi_jshelln(mpirank)
           IIsh=mpi_jshell(mpirank,i)
           do JJsh=IIsh,jshell
              call attrashell(IIsh,JJsh)
           enddo
        enddo
      endif
#else
      do i=1,mpi_jshelln(mpirank)
         IIsh=mpi_jshell(mpirank,i)
         do JJsh=IIsh,jshell
            call attrashell(IIsh,JJsh)
         enddo
      enddo
#endif
      RECORD_TIME(timer_end%T1eV)

#ifdef CEW

         if ( quick_cew%use_cew ) then

            RECORD_TIME(timer_begin%Tcew)

            call quick_cew_prescf()

            RECORD_TIME(timer_end%Tcew)

            timer_cumer%Tcew=timer_cumer%Tcew+timer_end%Tcew-timer_begin%Tcew

         endif
#endif

      call copySym(quick_qm_struct%o,nbasis)

      quick_qm_struct%oneElecO(:,:) = quick_qm_struct%o(:,:)

      bCalc1e=.false.
      !------- END MPI/ALL NODES ------------
     else
       if (.not. deltaO) quick_qm_struct%o(:,:)=quick_qm_struct%oneElecO(:,:)
     endif


     RECORD_TIME(timer_end%t1e)
     timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
     timer_cumer%T1eT=timer_cumer%T1eT+timer_end%T1eT-timer_begin%T1eT
     timer_cumer%T1eV=timer_cumer%T1eV+timer_end%T1eV-timer_begin%T1eV

   endif
#endif
end subroutine get1e


subroutine kineticO(IBAS)

   !------------------------------------------------
   ! This subroutine is to get 1e integral Operator
   !------------------------------------------------
   use allmod
   use quick_overlap_module, only: gpt, opf
   implicit double precision(a-h,o-z)
   integer Ibas 
   integer g_count
   double precision g_table(200)
   double precision :: valopf

   ix = itype(1,Ibas)
   iy = itype(2,Ibas)
   iz = itype(3,Ibas)
   xyzxi = xyz(1,quick_basis%ncenter(Ibas))
   xyzyi = xyz(2,quick_basis%ncenter(Ibas))
   xyzzi = xyz(3,quick_basis%ncenter(Ibas))

   do Jbas=Ibas,nbasis

      jx = itype(1,Jbas)
      jy = itype(2,Jbas)
      jz = itype(3,Jbas)
      xyzxj = xyz(1,quick_basis%ncenter(Jbas))
      xyzyj = xyz(2,quick_basis%ncenter(Jbas))
      xyzzj = xyz(3,quick_basis%ncenter(Jbas))

      g_count = ix+iy+iz+jx+jy+jz+2

      OJI = 0.d0 
      do Icon=1,ncontract(ibas)
         ai = aexp(Icon,Ibas)

         do Jcon=1,ncontract(jbas)
            F = dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)
           aj = aexp(Jcon,Jbas)

           valopf = opf(ai, aj, dcoeff(Jcon,Jbas), dcoeff(Icon,Ibas), xyzxi, xyzyi, xyzzi, xyzxj, xyzyj, xyzzj)

           if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then 

             ! The first part is the kinetic energy.
             call gpt(aj,ai,xyzxj,xyzyj,xyzzj,xyzxi,xyzyi,xyzzi,Px,Py,Pz,g_count,g_table)

              OJI = OJI + F*ekinetic(aj,   ai, &
                  jx,   jy,   jz,&
                  ix,   iy,   iz, &
                  xyzxj,xyzyj,xyzzj,&
                  xyzxi,xyzyi,xyzzi,Px,Py,Pz,g_table)
           endif
         enddo
      enddo
      quick_qm_struct%o(Jbas,Ibas) = OJI
   enddo

end subroutine kineticO


subroutine attrashell(IIsh,JJsh)
   use allmod
   use quick_overlap_module, only: gpt, opf, overlap_core
   !    use xiaoconstants
   implicit double precision(a-h,o-z)
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   double precision RA(3),RB(3),RP(3),inv_g,g_table(200), valopf

   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))

   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))

   ! The purpose of this subroutine is to calculate the nuclear attraction
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
   ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
   ! arises from the recusion relationship.

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! The first step is generating all the necessary auxillary integrals.
   ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
   ! The values of m range from 0 to i+j+k+ii+jj+kk.

   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2

   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))

         valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)), &
         quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)

         if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then

           !Eqn 14 O&S
           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,0,g_table)
           g = a+b
           !Eqn 15 O&S
           inv_g = 1.0d0 / dble(g)

           !Calculate first two terms of O&S Eqn A20
           constanttemp=dexp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 + (Az - Bz)**2.d0))*inv_g))
           constant = overlap_core(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) * 2.d0 * sqrt(g/Pi)*constanttemp

           !nextatom=number of external MM point charges. set to 0 if none used
           do iatom=1,natom+quick_molspec%nextatom
             if(iatom<=natom)then
               Cx=xyz(1,iatom)
               Cy=xyz(2,iatom)
               Cz=xyz(3,iatom)
               Z=-1.0d0*quick_molspec%chg(iatom)
             else
               Cx=quick_molspec%extxyz(1,iatom-natom)
               Cy=quick_molspec%extxyz(2,iatom-natom)
               Cz=quick_molspec%extxyz(3,iatom-natom)
               Z=-quick_molspec%extchg(iatom-natom)
             endif
             constant2=constanttemp*Z

             !Calculate the last term of O&S Eqn A21
             PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
!            if(quick_method%fMM .and. a*b*PCsquare/g.gt.33.0d0)then
!               xdistance=1.0d0/dsqrt(PCsquare)
!               call fmmone(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
!                     Cx,Cy,Cz,Px,Py,Pz,iatom,constant2,a,b,xdistance)
!            else

               !Compute O&S Eqn A21
               U = g* PCsquare

               !Calculate the last term of O&S Eqn A20
               call FmT(Maxm,U,aux)

               !Calculate all the auxilary integrals and store in attraxiao
               !array
               do L = 0,maxm
                  aux(L) = aux(L)*constant*Z
                  attraxiao(1,1,L)=aux(L)
               enddo

               ! At this point all the auxillary integrals have been calculated.
               ! It is now time to decompase the attraction integral to it's
               ! auxillary integrals through the recursion scheme.  To do this we use
               ! a recursive function.

               !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
                     !    Cx,Cy,Cz,Px,Py,Pz,g)
               NIJ1=10*NII2+NJJ2

               call nuclearattra(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
                     Cx,Cy,Cz,Px,Py,Pz,iatom)

!            endif

           enddo
         endif
      enddo
   enddo

   ! Xiao HE remember to multiply Z   01/12/2008
   !    attraction = attraction*(-1.d0)* Z
   201 return
end subroutine attrashell



  subroutine attrashellopt(IIsh,JJsh)
     use allmod
     use quick_overlap_module, only: opf, overlap
     !    use xiaoconstants
#ifdef MPIV
     use mpi
#endif
     implicit double precision(a-h,o-z)
     dimension aux(0:20)
     double precision AA(3),BB(3),CC(3),PP(3)
     common /xiaoattra/attra,aux,AA,BB,CC,PP,g
  
     double precision RA(3),RB(3),RP(3), valopf, g_table(200)
  
     Ax=xyz(1,quick_basis%katom(IIsh))
     Ay=xyz(2,quick_basis%katom(IIsh))
     Az=xyz(3,quick_basis%katom(IIsh))
  
     Bx=xyz(1,quick_basis%katom(JJsh))
     By=xyz(2,quick_basis%katom(JJsh))
     Bz=xyz(3,quick_basis%katom(JJsh))
  
     ! The purpose of this subroutine is to calculate the nuclear attraction
     ! of an electron  distributed between gtfs with orbital exponents a
     ! and b on A and B with angular momentums defined by i,j,k (a's x, y
     ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
     ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
     ! arises from the recusion relationship.
  
     ! The this is taken from the recursive relation found in Obara and Saika,
     ! J. Chem. Phys. 84 (7) 1986, 3963.
  
     ! The first step is generating all the necessary auxillary integrals.
     ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
     ! The values of m range from 0 to i+j+k+ii+jj+kk.
  
     NII2=quick_basis%Qfinal(IIsh)
     NJJ2=quick_basis%Qfinal(JJsh)
     Maxm=NII2+NJJ2+1+1
  
     do ips=1,quick_basis%kprim(IIsh)
        a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
        do jps=1,quick_basis%kprim(JJsh)
           b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
  
           valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)),&
           quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)
  
           if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
  
             g = a+b
             Px = (a*Ax + b*Bx)/g
             Py = (a*Ay + b*By)/g
             Pz = (a*Az + b*Bz)/g
             g_table = g**(-1.5)
  
             constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
                   * 2.d0 * sqrt(g/Pi)
  
             do iatom=1,natom+quick_molspec%nextatom
                if(quick_basis%katom(IIsh).eq.iatom.and.quick_basis%katom(JJsh).eq.iatom)then
                    continue
                 else
                   if(iatom<=natom)then
                    Cx=xyz(1,iatom)
                    Cy=xyz(2,iatom)
                    Cz=xyz(3,iatom)
                    Z=-1.0d0*quick_molspec%chg(iatom)
                   else
                    Cx=quick_molspec%extxyz(1,iatom-natom)
                    Cy=quick_molspec%extxyz(2,iatom-natom)
                    Cz=quick_molspec%extxyz(3,iatom-natom)
                    Z=-1.0d0*quick_molspec%extchg(iatom-natom)
                   endif
  
                   PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
  
                   U = g* PCsquare
                   !    Maxm = i+j+k+ii+jj+kk
                   call FmT(Maxm,U,aux)
                   do L = 0,maxm
                      aux(L) = aux(L)*constant*Z
                      attraxiao(1,1,L)=aux(L)
                   enddo
  
                   do L = 0,maxm-1
                      attraxiaoopt(1,1,1,L)=2.0d0*g*(Px-Cx)*aux(L+1)
                      attraxiaoopt(2,1,1,L)=2.0d0*g*(Py-Cy)*aux(L+1)
                      attraxiaoopt(3,1,1,L)=2.0d0*g*(Pz-Cz)*aux(L+1)
                   enddo
  
                   ! At this point all the auxillary integrals have been calculated.
                   ! It is now time to decompase the attraction integral to it's
                   ! auxillary integrals through the recursion scheme.  To do this we use
                   ! a recursive function.
  
                   !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
  
                         !    Cx,Cy,Cz,Px,Py,Pz,g)
                   NIJ1=10*NII2+NJJ2
  
                   call nuclearattraopt(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
  
                         Cx,Cy,Cz,Px,Py,Pz,iatom)
  
                endif
  
             enddo
           endif
        enddo
     enddo
  
     ! Xiao HE remember to multiply Z   01/12/2008
     !    attraction = attraction*(-1.d0)* Z
     return
  end subroutine attrashellopt


  subroutine attrashellfock1(IIsh,JJsh)
     use allmod
     use quick_overlap_module, only: opf, overlap
     !    use xiaoconstants
     implicit double precision(a-h,o-z)
     dimension aux(0:20)
     double precision AA(3),BB(3),CC(3),PP(3)
     common /xiaoattra/attra,aux,AA,BB,CC,PP,g
  
     double precision RA(3),RB(3),RP(3), valopf, g_table(200)
#ifdef MPIV
     include "mpif.h"
#endif
  
     Ax=xyz(1,quick_basis%katom(IIsh))
     Ay=xyz(2,quick_basis%katom(IIsh))
     Az=xyz(3,quick_basis%katom(IIsh))
  
     Bx=xyz(1,quick_basis%katom(JJsh))
     By=xyz(2,quick_basis%katom(JJsh))
     Bz=xyz(3,quick_basis%katom(JJsh))
  
     ! The purpose of this subroutine is to calculate the nuclear attraction
     ! of an electron  distributed between gtfs with orbital exponents a
     ! and b on A and B with angular momentums defined by i,j,k (a's x, y
     ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
     ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
     ! arises from the recusion relationship.
  
     ! The this is taken from the recursive relation found in Obara and Saika,
     ! J. Chem. Phys. 84 (7) 1986, 3963.
  
     ! The first step is generating all the necessary auxillary integrals.
     ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
     ! The values of m range from 0 to i+j+k+ii+jj+kk.
  
     NII2=quick_basis%Qfinal(IIsh)
     NJJ2=quick_basis%Qfinal(JJsh)
     Maxm=NII2+NJJ2+1+1
  
     do ips=1,quick_basis%kprim(IIsh)
        a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
        do jps=1,quick_basis%kprim(JJsh)
           b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
  
           valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)),&
           quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)
  
           if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
  
             g = a+b
             Px = (a*Ax + b*Bx)/g
             Py = (a*Ay + b*By)/g
             Pz = (a*Az + b*Bz)/g
             g_table = g**(-1.5)
  
             constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
                   * 2.d0 * sqrt(g/Pi)
  
             do iatom=1,natom+quick_molspec%nextatom
                if(quick_basis%katom(IIsh).eq.iatom.and.quick_basis%katom(JJsh).eq.iatom)then
                    continue
                 else
                   if(iatom<=natom)then
                    Cx=xyz(1,iatom)
                    Cy=xyz(2,iatom)
                    Cz=xyz(3,iatom)
                    Z=-1.0d0*quick_molspec%chg(iatom)
                   else
                    Cx=quick_molspec%extxyz(1,iatom-natom)
                    Cy=quick_molspec%extxyz(2,iatom-natom)
                    Cz=quick_molspec%extxyz(3,iatom-natom)
                    Z=-1.0d0*quick_molspec%extchg(iatom-natom)
                   endif
  
                   PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
  
                   U = g* PCsquare
                   !    Maxm = i+j+k+ii+jj+kk
                   call FmT(Maxm,U,aux)
                   do L = 0,maxm
                      aux(L) = aux(L)*constant*Z
                      attraxiao(1,1,L)=aux(L)
                   enddo
  
                   do L = 0,maxm-1
                      attraxiaoopt(1,1,1,L)=2.0d0*g*(Px-Cx)*aux(L+1)
                      attraxiaoopt(2,1,1,L)=2.0d0*g*(Py-Cy)*aux(L+1)
                      attraxiaoopt(3,1,1,L)=2.0d0*g*(Pz-Cz)*aux(L+1)
                   enddo
  
                   ! At this point all the auxillary integrals have been calculated.
                   ! It is now time to decompase the attraction integral to it's
                   ! auxillary integrals through the recursion scheme.  To do this we use
                   ! a recursive function.
  
                   !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
  
                         !    Cx,Cy,Cz,Px,Py,Pz,g)
                   NIJ1=10*NII2+NJJ2
  
                   call nuclearattrafock1(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
  
                         Cx,Cy,Cz,Px,Py,Pz,iatom)
  
                endif
  
             enddo
           endif
        enddo
     enddo
  
     ! Xiao HE remember to multiply Z   01/12/2008
     !    attraction = attraction*(-1.d0)* Z
     return
  end subroutine attrashellfock1


double precision function ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   use quick_overlap_module, only: overlap_core
   implicit none
   double precision :: kinetic
   double precision :: a,b
   integer :: i,j,k,ii,jj,kk,g,g_count
   double precision :: Ax,Ay,Az,Bx,By,Bz
   double precision :: Px,Py,Pz

   double precision :: xi,xj,xk,g_table(200)

   ! The purpose of this subroutine is to calculate the kinetic energy
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,and kk on B.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset kinetic to 0.

   kinetic = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk)) &
         +(Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2
   if (kinetic .ne. 0.d0) then
      kinetic=0.d0

      ! Kinetic energy is the integral of an orbital times the second derivative
      ! over space of the other orbital.  For GTFs, this means that it is just a
      ! sum of various overlap integrals with the powers adjusted.

      xi = dble(i)
      xj = dble(j)
      xk = dble(k)

      kinetic = kinetic &
            +        (-1.d0+     xi)*xi  *overlap_core(a,b,i-2,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a*( 1.d0+2.d0*xi)     *overlap_core(a,b,i  ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i+2,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
      kinetic = kinetic &
            +         (-1.d0+     xj)*xj *overlap_core(a,b,i,j-2,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a* ( 1.d0+2.d0*xj)    *overlap_core(a,b,i,j  ,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i,j+2,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
      kinetic = kinetic &
            +         (-1.d0+     xk)*xk *overlap_core(a,b,i,j,k-2,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a* ( 1.d0+2.d0*xk)    *overlap_core(a,b,i,j,k  ,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i,j,k+2,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   endif
   ekinetic = kinetic*(-0.5d0)  *exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

   return
end function ekinetic

end module quick_oei_module
