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

#ifdef OSHELL
module quick_oshell_eri_module
#else
module quick_cshell_eri_module
#endif

   implicit double precision(a-h,o-z)
   private
#ifdef OSHELL
!   public :: get_oshell_eri
!   public :: get_oshell_eri_energy
#else
!   public :: cshell ! Should be private, but we still need this for deprecated
                    ! subroutines such as addInt in shell.f90
!   public :: get_cshell_eri
!   public :: get_cshell_eri_energy
   public :: getEriPrecomputables
#endif 

#ifndef OSHELL
interface getEriPrecomputables
  module procedure get_eri_precomputables
end interface getEriPrecomputables
#endif

contains

#ifndef OSHELL
subroutine get_eri_precomputables

   !-----------------------------------------------------------------
   ! This subroutine computes precomputable quantities given by
   ! HGP equations 8, 9 and 15. The resulting quantities are
   ! stored in Apri, Ppri and Kpri. Additionally we compute Xcoeff 
   ! which is a combined coefficient for two indices (a product of 
   ! Kpri and normalized coefficients).
   ! See HGP paper for equations and APPENDIX: COMPUTER IMPLEMENTATION
   ! section: 
   ! Head‐Gordon, M.; Pople, J. A. A Method for Two‐electron 
   ! Gaussian Integral and Integral Derivative Evaluation Using
   ! Recurrence Relations. J. Chem. Phys. 1988, 89, 5777–5786.
   !__________________________________________________________________

   use allmod
   implicit none

#ifdef MPIV
   include 'mpif.h'
#endif

   integer ics,ips,jcs,jps,itemp,itemp2,i
   integer NA,NB
   double precision AA,BB,XYZA(3),XYZB(3),DAB
   if (master) then
   ! ics cycle
   do ics=1,jshell                       ! ics is the shell no.
      do ips=1,quick_basis%kprim(ics)                ! ips is prim no. for certain shell

         ! jcs cycle
         do jcs=1,jshell
            do jps=1,quick_basis%kprim(jcs)

               ! We have ics,jcs, ips and jps, which is the prime for shell, so we can
               ! obtain its global prim. and its exponents and coeffecients
               NA=quick_basis%kstart(ics)+IPS-1         ! we can get globle prim no. for ips
               NB=quick_basis%kstart(jcs)+jps-1         ! and jps

               AA=quick_basis%gcexpo(ips,quick_basis%ksumtype(ics))    ! so we have the exponent part for ics shell ips prim
               BB=quick_basis%gcexpo(jps,quick_basis%ksumtype(jcs))    ! and jcs shell jps prim
               Apri(NA,NB)=AA+BB                     ! A'=expo(A)+expo(B)
               do i=1,3
                  XYZA(i)=xyz(i,quick_basis%katom(ics))     ! xyz(A)
                  XYZB(i)=xyz(i,quick_basis%katom(jcs))     ! xyz(B)
               enddo
               !DAB=quick_molspec%atomdistance(quick_basis%katom(ics),quick_basis%katom(jcs))
               DAB = dsqrt((xyz(1,quick_basis%katom(ics))-xyz(1,quick_basis%katom(jcs)))**2 + &
                           (xyz(2,quick_basis%katom(ics))-xyz(2,quick_basis%katom(jcs)))**2 + &
                           (xyz(3,quick_basis%katom(ics))-xyz(3,quick_basis%katom(jcs)))**2 )

               ! P' is the weighting center of NpriI and NpriJ
               !              expo(A)*xyz(A)+expo(B)*xyz(B)
               ! P'(A,B)  = ------------------------------
               !                 expo(A) + expo(B)
               do i=1,3
                  Ppri(i,NA,NB) = (XYZA(i)*AA + XYZB(i)*BB)/(AA+BB)
               enddo

               !                    expo(A)*expo(B)*(xyz(A)-xyz(B))^2              1
               ! K'(A,B) =  exp[ - ------------------------------------]* -------------------
               !                            expo(A)+expo(B)                  expo(A)+expo(B)
               Kpri(NA,NB) = dexp(-AA*BB/(AA+BB)*(DAB**2))/(AA+BB)

               do  itemp=quick_basis%Qstart(ics),quick_basis%Qfinal(ics)
                  do itemp2=quick_basis%Qstart(jcs),quick_basis%Qfinal(jcs)

                     ! Xcoeff(A,B,itmp1,itmp2)=K'(A,B)*a(itmp1)*a(itmp2)
                     quick_basis%Xcoeff(NA,NB,itemp,itemp2)=Kpri(NA,NB)* &

                           quick_basis%gccoeff(ips,quick_basis%ksumtype(ics)+Itemp )* &

                           quick_basis%gccoeff(jps,quick_basis%ksumtype(jcs)+Itemp2)
                  enddo
               enddo

            enddo
         enddo
      enddo
   enddo
   endif


#ifdef MPIV
      if (bMPI) then
         call MPI_BCAST(Ppri,3*jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(Kpri,jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(Apri,jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(quick_basis%Xcoeff,jbasis*jbasis*4*4,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      endif
#endif

end subroutine get_eri_precomputables
#endif


#ifdef OSHELL
subroutine get_oshell_eri(II_arg)
#else
subroutine get_cshell_eri(II_arg)
#endif

   !------------------------------------------------
   ! This subroutine is to get 2e integral
   !------------------------------------------------
   use allmod
   implicit double precision(a-h,o-z)
   double precision testtmp,cutoffTest
   integer II_arg
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   II = II_arg
   do JJ = II,jshell
      testtmp = Ycutoff(II,JJ)
      do KK = II,jshell
         do LL = KK,jshell

          cutoffTest = testtmp * Ycutoff(KK,LL)
          if (cutoffTest .gt. quick_method%integralCutoff) then
            DNmax =  max(4.0d0*cutmatrix(II,JJ), &
                  4.0d0*cutmatrix(KK,LL), &
                  cutmatrix(II,LL), &
                  cutmatrix(II,KK), &
                  cutmatrix(JJ,KK), &
                  cutmatrix(JJ,LL))
            ! (IJ|KL)^2<=(II|JJ)*(KK|LL) if smaller than cutoff criteria, then
            ! ignore the calculation to save computation time

            if ( cutoffTest * DNmax  .gt. quick_method%integralCutoff ) &
#ifdef OSHELL 
                call oshell
#else
                call cshell
#endif
           endif
         enddo
      enddo
   enddo
#ifdef OSHELL
end subroutine get_oshell_eri
#else
end subroutine get_cshell_eri
#endif

#ifdef OSHELL
end module quick_oshell_eri_module
#else
end module quick_cshell_eri_module
#endif

