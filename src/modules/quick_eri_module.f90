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
   public :: getOshellEri
!   public :: get_oshell_eri_energy
#else
!   public :: cshell ! Should be private, but we still need this for deprecated
                    ! subroutines such as addInt in shell.f90
   public :: getCshellEri
!   public :: get_cshell_eri_energy
   public :: getEriPrecomputables
#endif 

#ifdef OSHELL

interface getOshellEri
  module procedure get_oshell_eri
end interface getOshellEri

#else

interface getEriPrecomputables
  module procedure get_eri_precomputables
end interface getEriPrecomputables

interface getCshellEri
  module procedure get_cshell_eri
end interface getCshellEri

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

! Vertical Recursion by Xiao HE 07/07/07 version
#ifdef OSHELL
subroutine oshell
#else
subroutine cshell
#endif

   use allmod

   Implicit double precision(a-h,o-z)
   double precision P(3),Q(3),W(3),KAB,KCD,AAtemp(3)
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp
   COMMON /COM1/RA,RB,RC,RD

   do M=1,3
      RA(M)=xyz(M,quick_basis%katom(II))
      RB(M)=xyz(M,quick_basis%katom(JJ))
      RC(M)=xyz(M,quick_basis%katom(KK))
      RD(M)=xyz(M,quick_basis%katom(LL))
   enddo

  ! Get angular momenta quantum number for each of the 4 shells
  ! s=0~0, p=1~1, sp=0~1, d=2~2, f=3~3

   NII1=quick_basis%Qstart(II)
   NII2=quick_basis%Qfinal(II)
   NJJ1=quick_basis%Qstart(JJ)
   NJJ2=quick_basis%Qfinal(JJ)
   NKK1=quick_basis%Qstart(KK)
   NKK2=quick_basis%Qfinal(KK)
   NLL1=quick_basis%Qstart(LL)
   NLL2=quick_basis%Qfinal(LL)

   NNAB=(NII2+NJJ2)
   NNCD=(NKK2+NLL2)

   NABCDTYPE=NNAB*10+NNCD


   NNAB=sumindex(NNAB)
   NNCD=sumindex(NNCD)
   NNA=sumindex(NII1-1)+1
   NNC=sumindex(NKK1-1)+1

   !The summation of the highest angular momentum number 
   !of each shell. 
   NABCD=NII2+NJJ2+NKK2+NLL2
   ITT=0

   do JJJ=1,quick_basis%kprim(JJ)

      Nprij=quick_basis%kstart(JJ)+JJJ-1

      ! the second cycle is for i prim
      ! II and NpriI are the tracking indices
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         !For NpriI and NpriJ primitives, we calculate the following quantities
         AB=Apri(Nprii,Nprij)    ! AB = Apri = expo(NpriI)+expo(NpriJ). Eqn 8 of HGP.
         ABtemp=0.5d0/AB         ! ABtemp = 1/(2Apri) = 1/2(expo(NpriI)+expo(NpriJ))
         ! This is term is required for Eqn 6 of HGP. 
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)

         do M=1,3
            !Eqn 9 of HGP
            ! P' is the weighting center of NpriI and NpriJ
            !                           --->           --->
            ! ->  ------>       expo(I)*xyz(I)+expo(J)*xyz(J)
            ! P = P'(I,J)  = ------------------------------
            !                       expo(I) + expo(J)
            P(M)=Ppri(M,Nprii,Nprij)

            !Multiplication of Eqns 9  by Eqn 8 of HGP.. 
            !                        -->            -->
            ! ----->         expo(I)*xyz(I)+expo(J)*xyz(J)                                 -->            -->
            ! AAtemp = ----------------------------------- * (expo(I) + expo(J)) = expo(I)*xyz(I)+expo(J)*xyz(J)
            !                  expo(I) + expo(J)
            AAtemp(M)=P(M)*AB

            !Requires for HGP Eqn 6. 
            ! ----->   ->  ->
            ! Ptemp  = P - A
            Ptemp(M)=P(M)-RA(M)
         enddo

         ! the third cycle is for l prim
         ! LLL and npriL are the tracking indices
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1

            ! the forth cycle is for k prim
            ! the KKK and nprik are the tracking indices
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1

               ! prim cutoff: cutoffprim(I,J,K,L) = dnmax * cutprim(I,J) * cutprim(K,L)
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%primLimit)then

                  !Nita quantity of HGP Eqn 10. This is same as
                  !zita (AB) above. 
                  CD=Apri(Nprik,Npril)  ! CD = Apri = expo(NpriK) + expo(NpriL)

                  !First term of HGP Eqn 12 without sqrt. 
                  ABCD=AB+CD            ! ABCD = expo(NpriI)+expo(NpriJ)+expo(NpriK)+expo(NpriL)

                  !First term of HGP Eqn 13.
                  !         AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                  ! Rou = ----------- = ------------------------------------
                  !         AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                  ROU=AB*CD/ABCD

                  RPQ=0.0d0

                  !First term of HGP Eqn 12 with sqrt. 
                  !              _______________________________
                  ! ABCDxiao = \/expo(I)+expo(J)+expo(K)+expo(L)
                  ABCDxiao=dsqrt(ABCD)

                  !Not sure why we calculate the following. 
                  CDtemp=0.5d0/CD       ! CDtemp =  1/2(expo(NpriK)+expo(NpriL))

                  !These terms are required for HGP Eqn 6.
                  !                expo(I)+expo(J)                        expo(K)+expo(L)
                  ! ABcom = --------------------------------  CDcom = --------------------------------
                  !          expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                  ABcom=AB/ABCD
                  CDcom=CD/ABCD

                  ! ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                  ABCDtemp=0.5d0/ABCD

                  do M=1,3

                     !Calculate Q of HGP 10, which is same as P above. 
                     ! Q' is the weighting center of NpriK and NpriL
                     !                           --->           --->
                     ! ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                     ! Q = P'(K,L)  = ------------------------------
                     !                       expo(K) + expo(L)
                     Q(M)=Ppri(M,Nprik,Npril)

                     !HGP Eqn 10. 
                     ! W' is the weight center for NpriI,NpriJ,NpriK and NpriL
                     !                --->             --->             --->            --->
                     ! ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                     ! W = -------------------------------------------------------------------
                     !                    expo(I) + expo(J) + expo(K) + expo(L)
                     W(M)=(AAtemp(M)+Q(M)*CD)/ABCD

                     !Required for HGP Eqn 13.
                     !        ->  ->  2
                     ! RPQ =| P - Q |
                     XXXtemp=P(M)-Q(M)
                     RPQ=RPQ+XXXtemp*XXXtemp

                     !Not sure why we need the next two terms. 
                     ! ---->   ->  ->
                     ! Qtemp = Q - K
                     Qtemp(M)=Q(M)-RC(M)

                     ! ----->   ->  ->
                     ! WQtemp = W - Q
                     ! ----->   ->  ->
                     ! WPtemp = W - P
                     WQtemp(M)=W(M)-Q(M)

                     !Required for HGP Eqns 6 and 16.
                     WPtemp(M)=W(M)-P(M)
                  enddo

                  !HGP Eqn 13. 
                  !             ->  -> 2
                  ! T = ROU * | P - Q|
                  T=RPQ*ROU
                  !                         2m        2
                  ! Fm(T) = integral(1,0) {t   exp(-Tt )dt}
                  ! NABCD is the m value, and FM returns the FmT value
#ifdef MIRP
                  call mirp_fmt(NABCD,T,FM)
#else
                  call FmT(NABCD,T,FM)
#endif
                  !Go through all m values, obtain Fm values from FM array we
                  !just computed and calculate quantities required for HGP Eqn
                  !12. 
                  do iitemp=0,NABCD
                     ! Yxiaotemp(1,1,iitemp) is the starting point of recurrsion
                     Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                     !              _______________________________
                     ! ABCDxiao = \/expo(I)+expo(J)+expo(K)+expo(L)
                  enddo

                  ITT=ITT+1
                  ! now we will do vrr and and the double-electron integral
                  call vertical(NABCDTYPE)
                  do I2=NNC,NNCD
                     do I1=NNA,NNAB
                        Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                     enddo
                  enddo
               endif
            enddo
         enddo
      enddo
   enddo

   do I=NII1,NII2
      NNA=Sumindex(I-1)+1
      do J=NJJ1,NJJ2
         NNAB=SumINDEX(I+J)
         do K=NKK1,NKK2
            NNC=Sumindex(k-1)+1
            do L=NLL1,NLL2
               NNCD=SumIndex(K+L)
#ifdef OSHELL
                call iclass_oshell(I,J,K,L,NNA,NNC,NNAB,NNCD)
#else
                call iclass_cshell(I,J,K,L,NNA,NNC,NNAB,NNCD)
#endif
            enddo
         enddo
      enddo
   enddo
   201 return
#ifdef OSHELL
end subroutine oshell
#else
end subroutine cshell
#endif

#ifdef OSHELL
end module quick_oshell_eri_module
#else
end module quick_cshell_eri_module
#endif

