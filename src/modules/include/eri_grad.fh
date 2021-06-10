!---------------------------------------------------------------------!
! Created by Madu Manathunga on 06/29/2020                            !
!                                                                     ! 
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
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
! scf gradient calculation.                                           !
!---------------------------------------------------------------------!

#include "util.fh"

#ifdef OSHELL
module quick_oshell_eri_grad_module
#else
module quick_cshell_eri_grad_module
#endif

   implicit double precision(a-h,o-z)
#ifdef OSHELL
   public  :: oshell_eri_grad
   private :: iclass_grad_oshell
#else
   public  :: cshell_eri_grad
   private :: iclass_grad_cshell
#endif 
contains


! Vertical Recursion by Xiao HE 07/07/07 version
#ifdef OSHELL
subroutine oshell_eri_grad
#else
subroutine cshell_eri_grad
#endif

   use allmod

   Implicit double precision(a-h,o-z)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=14)
   double precision FM(0:14)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,NNABfirst,NNCDfirst
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

   COMMON /COM1/RA,RB,RC,RD

   do M=1,3
      RA(M)=xyz(M,quick_basis%katom(II))
      RB(M)=xyz(M,quick_basis%katom(JJ))
      RC(M)=xyz(M,quick_basis%katom(KK))
      RD(M)=xyz(M,quick_basis%katom(LL))
   enddo

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

   NNABfirst=sumindex(NII2+NJJ2+1)
   NNCDfirst=sumindex(NKK2+NLL2+1)

   NNA=Sumindex(NII1-2)+1

   NNC=Sumindex(NKK1-2)+1

   NABCD=NII2+NJJ2+NKK2+NLL2

   ! For first derivative of nuclui motion, the total angular momentum is raised by 1
   NABCD=NABCD+1+1


   !print*,'NABCD=',NABCD

   ITT=0
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         AB=Apri(Nprii,Nprij)
         ABtemp=0.5d0/AB
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do M=1,3
            P(M)=Ppri(M,Nprii,Nprij)
            Ptemp(M)=P(M)-RA(M)
         enddo
         !            KAB=Kpri(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%gradCutoff)then
                  CD=Apri(Nprik,Npril)
                  ABCD=AB+CD
                  ROU=AB*CD/ABCD
                  RPQ=0.0d0
                  ABCDxiao=dsqrt(ABCD)

                  CDtemp=0.5d0/CD
                  ABcom=AB/ABCD
                  CDcom=CD/ABCD
                  ABCDtemp=0.5d0/ABCD
                  do M=1,3
                     Q(M)=Ppri(M,Nprik,Npril)
                     W(M)=(P(M)*AB+Q(M)*CD)/ABCD
                     XXXtemp=P(M)-Q(M)
                     RPQ=RPQ+XXXtemp*XXXtemp
                     Qtemp(M)=Q(M)-RC(M)
                     WQtemp(M)=W(M)-Q(M)
                     WPtemp(M)=W(M)-P(M)
                  enddo
                  !                         KCD=Kpri(Nprik,Npril)

                  T=RPQ*ROU

                  call FmT(NABCD,T,FM)
                  do iitemp=0,NABCD
                     Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                  enddo

                  ITT=ITT+1

                  call vertical(NABCDTYPE+11)

                  !                           if(NABCDTYPE.eq.44)print*,'xiao',NABCD,FM

                  do I2=NNC,NNCDfirst
                     do I1=NNA,NNABfirst
                        Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                     enddo
                  enddo

               endif
            enddo
         enddo
      enddo
   enddo

   ! NNA=1
   ! NNC=1

   ! allocate scratch memory for X arrays
   call allocshellopt(quick_scratch,maxcontract)

   do I=NII1,NII2
      NNA=Sumindex(I-2)+1
      do J=NJJ1,NJJ2
         NNAB=SumINDEX(I+J)
         ! change for first derivative
         NNABfirst=SumINDEX(I+J+1)
         do K=NKK1,NKK2
            NNC=Sumindex(k-2)+1
            do L=NLL1,NLL2
               NNCD=SumIndex(K+L)
               NNCDfirst=SumIndex(K+L+1)
#ifdef OSHELL
               call iclass_grad_oshell(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
#else
               call iclass_grad_cshell(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
#endif 
            enddo
         enddo
      enddo
   enddo

   ! deallocate scratch memory for X arrays
   call deallocshellopt(quick_scratch)

#ifdef OSHELL
end subroutine oshell_eri_grad
#else
end subroutine cshell_eri_grad
#endif


! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
#ifdef OSHELL
subroutine iclass_grad_oshell(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
#else
subroutine iclass_grad_cshell(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
#endif

   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)

   double precision storeaa(120,120)
   double precision storebb(120,120)
   double precision storecc(120,120)
   double precision storedd(120,120)

   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=14)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision AA,BB,CC,DD

   COMMON /COM1/RA,RB,RC,RD
   COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
   COMMON /COM4/P,Q,W
   COMMON /COM5/FM

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /xiaostoreopt/storeaa,storebb,storecc,storedd
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   ITT=0
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1
      BB=quick_basis%gcexpo(JJJ,quick_basis%ksumtype(JJ))
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         AA=quick_basis%gcexpo(III,quick_basis%ksumtype(II))

         X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            DD=quick_basis%gcexpo(LLL,quick_basis%ksumtype(LL))
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               CC=quick_basis%gcexpo(KKK,quick_basis%ksumtype(KK))
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%gradCutoff)then
                  ITT=ITT+1
                  quick_scratch%X44(ITT)=X2*quick_basis%Xcoeff(Nprik,Npril,K,L)
                  quick_scratch%X44AA(ITT)=quick_scratch%X44(ITT)*AA*2.0d0
                  quick_scratch%X44BB(ITT)=quick_scratch%X44(ITT)*BB*2.0d0
                  quick_scratch%X44CC(ITT)=quick_scratch%X44(ITT)*CC*2.0d0
                  !                       X44DD(ITT)=X44(ITT)*DD*2.0d0
               endif
            enddo
         enddo
      enddo
   enddo

   do MM2=NNC,NNCD
      do MM1=NNA,NNAB
         Ytemp=0.0d0
         !                           YtempAA=0.0d0
         !                           YtempBB=0.0d0
         !                           YtempCC=0.0d0
         do itemp=1,ITT
            Ytemp=Ytemp+quick_scratch%X44(itemp)*Yxiao(itemp,MM1,MM2)
            !                           YtempAA=YtempAA+X44AA(itemp)*Yxiao(itemp,MM1,MM2)
            !                           YtempBB=YtempBB+X44BB(itemp)*Yxiao(itemp,MM1,MM2)
            !                           YtempCC=YtempCC+X44CC(itemp)*Yxiao(itemp,MM1,MM2)

         enddo
         store(MM1,MM2)=Ytemp
         !                         storeAA(MM1,MM2)=YtempAA
         !                         storeBB(MM1,MM2)=YtempBB
         !                         storeCC(MM1,MM2)=YtempCC

      enddo
   enddo

   do MM2=NNC,NNCDfirst
      do MM1=NNA,NNABfirst
         YtempAA=0.0d0
         YtempBB=0.0d0
         YtempCC=0.0d0
         do itemp=1,ITT
            YtempAA=YtempAA+quick_scratch%X44AA(itemp)*Yxiao(itemp,MM1,MM2)
            YtempBB=YtempBB+quick_scratch%X44BB(itemp)*Yxiao(itemp,MM1,MM2)
            YtempCC=YtempCC+quick_scratch%X44CC(itemp)*Yxiao(itemp,MM1,MM2)

            !                  if(II.eq.1.and.JJ.eq.1.and.KK.eq.13.and.LL.eq.13)then
            !                    print*,KKK-39,LLL-39,III+12,JJJ+12,itemp,X44AA(itemp),Yxiao(itemp,MM1,MM2)
            !                  endif

         enddo
         storeAA(MM1,MM2)=YtempAA
         storeBB(MM1,MM2)=YtempBB
         storeCC(MM1,MM2)=YtempCC

      enddo
   enddo

   NBI1=quick_basis%Qsbasis(II,I)
   NBI2=quick_basis%Qfbasis(II,I)
   NBJ1=quick_basis%Qsbasis(JJ,J)
   NBJ2=quick_basis%Qfbasis(JJ,J)
   NBK1=quick_basis%Qsbasis(KK,K)
   NBK2=quick_basis%Qfbasis(KK,K)
   NBL1=quick_basis%Qsbasis(LL,L)
   NBL2=quick_basis%Qfbasis(LL,L)

   Agrad1=0.d0
   Bgrad1=0.d0
   Cgrad1=0.d0
   Dgrad1=0.d0
   Agrad2=0.d0
   Bgrad2=0.d0
   Cgrad2=0.d0
   Dgrad2=0.d0
   Agrad3=0.d0
   Bgrad3=0.d0
   Cgrad3=0.d0
   Dgrad3=0.d0

   !       IJKLtype=1000*I+100*J+10*K+L
   IJtype=10*I+J
   KLtype=10*K+L
   IJKLtype=100*IJtype+KLtype

   III1=quick_basis%ksumtype(II)+NBI1
   III2=quick_basis%ksumtype(II)+NBI2
   JJJ1=quick_basis%ksumtype(JJ)+NBJ1
   JJJ2=quick_basis%ksumtype(JJ)+NBJ2
   KKK1=quick_basis%ksumtype(KK)+NBK1
   KKK2=quick_basis%ksumtype(KK)+NBK2
   LLL1=quick_basis%ksumtype(LL)+NBL1
   LLL2=quick_basis%ksumtype(LL)+NBL2

   iA = quick_basis%ncenter(III2)
   iB = quick_basis%ncenter(JJJ2)
   iC = quick_basis%ncenter(KKK2)
   iD = quick_basis%ncenter(LLL2)

   iAstart = (iA-1)*3
   iBstart = (iB-1)*3
   iCstart = (iC-1)*3
   iDstart = (iD-1)*3

   if(II.lt.JJ.and.II.lt.KK.and.KK.lt.LL)then

      do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
               do LLL=LLL1,LLL2

                  call hrrwholeopt

                  ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                  ! can be equal.
#ifdef OSHELL

                  DENSELK=quick_qm_struct%dense(LLL,KKK)+quick_qm_struct%denseb(LLL,KKK)
                  DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)

                  DENSEKIA=quick_qm_struct%dense(KKK,III)
                  DENSEKJA=quick_qm_struct%dense(KKK,JJJ)
                  DENSELJA=quick_qm_struct%dense(LLL,JJJ)
                  DENSELIA=quick_qm_struct%dense(LLL,III)

                  DENSEKIB=quick_qm_struct%denseb(KKK,III)
                  DENSEKJB=quick_qm_struct%denseb(KKK,JJJ)
                  DENSELJB=quick_qm_struct%denseb(LLL,JJJ)
                  DENSELIB=quick_qm_struct%denseb(LLL,III)

                  constant = (4.d0*DENSEJI*DENSELK- 2.0d0*quick_method%x_hybrid_coeff*DENSEKIA*DENSELJA &
                        -2.0d0*quick_method%x_hybrid_coeff*DENSELIA*DENSEKJA-2.0d0*quick_method%x_hybrid_coeff*DENSEKIB*DENSELJB &
                        -2.0d0*quick_method%x_hybrid_coeff*DENSELIB*DENSEKJB)
#else
                  DENSEKI=quick_qm_struct%dense(KKK,III)
                  DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                  DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                  DENSELI=quick_qm_struct%dense(LLL,III)
                  DENSELK=quick_qm_struct%dense(LLL,KKK)
                  DENSEJI=quick_qm_struct%dense(JJJ,III)

                  constant = (4.d0*DENSEJI*DENSELK-quick_method%x_hybrid_coeff*DENSEKI*DENSELJ &
                        -quick_method%x_hybrid_coeff*DENSELI*DENSEKJ)
#endif

                  Agrad1=Agrad1+Yaa(1)*constant
                  Agrad2=Agrad2+Yaa(2)*constant
                  Agrad3=Agrad3+Yaa(3)*constant
                  Bgrad1=Bgrad1+Ybb(1)*constant
                  Bgrad2=Bgrad2+Ybb(2)*constant
                  Bgrad3=Bgrad3+Ybb(3)*constant
                  Cgrad1=Cgrad1+Ycc(1)*constant
                  Cgrad2=Cgrad2+Ycc(2)*constant
                  Cgrad3=Cgrad3+Ycc(3)*constant

               enddo
            enddo
         enddo
      enddo

   else

      do III=III1,III2
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=max(III,KKK1),KKK2
               do LLL=max(KKK,LLL1),LLL2

                  if(III.LT.KKK)then

                     call hrrwholeopt

                     if(III.lt.JJJ.and.KKK.lt.LLL)then
                        ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                        ! can be equal.

#ifdef OSHELL

                        DENSELK=quick_qm_struct%dense(LLL,KKK)+quick_qm_struct%denseb(LLL,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)

                        DENSEKIA=quick_qm_struct%dense(KKK,III)
                        DENSEKJA=quick_qm_struct%dense(KKK,JJJ)
                        DENSELJA=quick_qm_struct%dense(LLL,JJJ)
                        DENSELIA=quick_qm_struct%dense(LLL,III)

                        DENSEKIB=quick_qm_struct%denseb(KKK,III)
                        DENSEKJB=quick_qm_struct%denseb(KKK,JJJ)
                        DENSELJB=quick_qm_struct%denseb(LLL,JJJ)
                        DENSELIB=quick_qm_struct%denseb(LLL,III)

                        constant = (4.d0*DENSEJI*DENSELK- 2.0d0*quick_method%x_hybrid_coeff*DENSEKIA*DENSELJA &
                              -2.0d0*quick_method%x_hybrid_coeff*DENSELIA*DENSEKJA &
                              -2.0d0*quick_method%x_hybrid_coeff*DENSEKIB*DENSELJB &
                              -2.0d0*quick_method%x_hybrid_coeff*DENSELIB*DENSEKJB)

#else
                        DENSEKI=quick_qm_struct%dense(KKK,III)
                        DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                        DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                        DENSELI=quick_qm_struct%dense(LLL,III)
                        DENSELK=quick_qm_struct%dense(LLL,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)

                        constant = (4.d0*DENSEJI*DENSELK-quick_method%x_hybrid_coeff*DENSEKI*DENSELJ &
                              -quick_method%x_hybrid_coeff*DENSELI*DENSEKJ)
#endif
                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant
                        !    ! do all the (ii|ii) integrals.
                        !        ! Set some variables to reduce access time for some of the more
                        !        ! used quantities. (AGAIN)
                        ElseIf(III.eq.JJJ.and.KKK.eq.LLL)then
                        ! Find  all the (ii|jj) integrals.

#ifdef OSHELL
                        DENSEJJ=quick_qm_struct%dense(KKK,KKK)+quick_qm_struct%denseb(KKK,KKK)
                        DENSEII=quick_qm_struct%dense(III,III)+quick_qm_struct%denseb(III,III)

                        DENSEJIA=quick_qm_struct%dense(KKK,III)
                        DENSEJIB=quick_qm_struct%denseb(KKK,III)

                        constant = (DENSEII*DENSEJJ-quick_method%x_hybrid_coeff*DENSEJIA*DENSEJIA &
                                    -quick_method%x_hybrid_coeff*DENSEJIB*DENSEJIB)
#else
                        DENSEJI=quick_qm_struct%dense(KKK,III)
                        DENSEJJ=quick_qm_struct%dense(KKK,KKK)
                        DENSEII=quick_qm_struct%dense(III,III)

                        constant = (DENSEII*DENSEJJ-.5d0*quick_method%x_hybrid_coeff*DENSEJI*DENSEJI)
#endif

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        elseif(JJJ.eq.KKK.and.JJJ.eq.LLL)then
                        ! Find  all the (ij|jj) integrals.

#ifdef OSHELL
                        DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)
                        DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)+quick_qm_struct%denseb(JJJ,JJJ)

                        DENSEJIA=quick_qm_struct%dense(JJJ,III)
                        DENSEJJA=quick_qm_struct%dense(JJJ,JJJ)

                        DENSEJIB=quick_qm_struct%denseb(JJJ,III)
                        DENSEJJB=quick_qm_struct%denseb(JJJ,JJJ)

                        constant = 2.0d0*DENSEJJ*DENSEJI-2.0d0*quick_method%x_hybrid_coeff*DENSEJJA*DENSEJIA &
                                   -2.0d0*quick_method%x_hybrid_coeff*DENSEJJB*DENSEJIB
#else
                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

                        constant = 2.0d0*DENSEJJ*DENSEJI-quick_method%x_hybrid_coeff*DENSEJJ*DENSEJI
#endif

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        !        ! Find  all the (ii|ij) integrals.
                        !
                        !        ! Find all the (ij|ij) integrals
                        !
                        ! Find all the (ij|ik) integrals where j>i,k>j
                        elseif(KKK.eq.LLL.and.III.lt.JJJ.and.JJJ.ne.KKK)then

                        ! Find all the (ij|kk) integrals where j>i, k>j.

#ifdef OSHELL
                        DENSEKK=quick_qm_struct%dense(KKK,KKK)+quick_qm_struct%denseb(KKK,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)

                        DENSEKIA=quick_qm_struct%dense(KKK,III)
                        DENSEKJA=quick_qm_struct%dense(KKK,JJJ)

                        DENSEKIB=quick_qm_struct%denseb(KKK,III)
                        DENSEKJB=quick_qm_struct%denseb(KKK,JJJ)

                        constant=(2.d0*DENSEJI*DENSEKK-2.0d0*quick_method%x_hybrid_coeff*DENSEKIA*DENSEKJA &
                                  -2.0d0*quick_method%x_hybrid_coeff*DENSEKIB*DENSEKJB)
#else
                        DENSEKI=quick_qm_struct%dense(KKK,III)
                        DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                        DENSEKK=quick_qm_struct%dense(KKK,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)

                        constant=(2.d0*DENSEJI*DENSEKK-quick_method%x_hybrid_coeff*DENSEKI*DENSEKJ)
#endif

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        !            ! Find all the (ik|jj) integrals where j>i, k>j.
                        elseif(III.eq.JJJ.and.KKK.lt.LLL)then
                        ! Find all the (ii|jk) integrals where j>i, k>j.
#ifdef OSHELL

                        DENSEKJ=quick_qm_struct%dense(LLL,KKK)+quick_qm_struct%denseb(LLL,KKK)
                        DENSEII=quick_qm_struct%dense(III,III)+quick_qm_struct%denseb(III,III)

                        DENSEJIA=quick_qm_struct%dense(KKK,III)
                        DENSEKIA=quick_qm_struct%dense(LLL,III)

                        DENSEJIB=quick_qm_struct%denseb(KKK,III)
                        DENSEKIB=quick_qm_struct%denseb(LLL,III)

                        constant = (2.d0*DENSEKJ*DENSEII-2.0d0*quick_method%x_hybrid_coeff*DENSEJIA*DENSEKIA &
                                   -2.0d0*quick_method%x_hybrid_coeff*DENSEJIB*DENSEKIB)
#else
                        DENSEII=quick_qm_struct%dense(III,III)
                        DENSEJI=quick_qm_struct%dense(KKK,III)
                        DENSEKI=quick_qm_struct%dense(LLL,III)
                        DENSEKJ=quick_qm_struct%dense(LLL,KKK)

                        constant = (2.d0*DENSEKJ*DENSEII-quick_method%x_hybrid_coeff*DENSEJI*DENSEKI)
#endif

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                     endif

                  else
                     if(JJJ.LE.LLL)then

                        !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                        call hrrwholeopt

                        if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then
                           ! do all the (ii|ii) integrals.
#ifdef OSHELL
                           DENSEII=quick_qm_struct%dense(III,III)+quick_qm_struct%denseb(III,III)
#else
                           DENSEII=quick_qm_struct%dense(III,III)
#endif

                           constant=0.0d0

                           elseif(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then
                           ! Find  all the (ii|ij) integrals.
#ifdef OSHELL
                           DENSEJI=quick_qm_struct%dense(LLL,III)+quick_qm_struct%denseb(LLL,III)
                           DENSEII=quick_qm_struct%dense(III,III)+quick_qm_struct%denseb(III,III)

                           DENSEJIA=quick_qm_struct%dense(LLL,III)
                           DENSEIIA=quick_qm_struct%dense(III,III)

                           DENSEJIB=quick_qm_struct%denseb(LLL,III)
                           DENSEIIB=quick_qm_struct%denseb(III,III)

                           constant = 2.0d0*DENSEJI*DENSEII-2.0d0*quick_method%x_hybrid_coeff*DENSEJIA*DENSEIIA &
                                      -2.0d0*quick_method%x_hybrid_coeff*DENSEJIB*DENSEIIB

#else
                           DENSEJI=quick_qm_struct%dense(LLL,III)
                           DENSEII=quick_qm_struct%dense(III,III)

                           constant = 2.0d0*DENSEJI*DENSEII-quick_method%x_hybrid_coeff*DENSEJI*DENSEII
#endif


                           Agrad1=Agrad1+Yaa(1)*constant
                           Agrad2=Agrad2+Yaa(2)*constant
                           Agrad3=Agrad3+Yaa(3)*constant
                           Bgrad1=Bgrad1+Ybb(1)*constant
                           Bgrad2=Bgrad2+Ybb(2)*constant
                           Bgrad3=Bgrad3+Ybb(3)*constant
                           Cgrad1=Cgrad1+Ycc(1)*constant
                           Cgrad2=Cgrad2+Ycc(2)*constant
                           Cgrad3=Cgrad3+Ycc(3)*constant

                           elseif(III.eq.KKK.and.JJJ.eq.LLL.and.III.lt.JJJ)then
                           ! Find all the (ij|ij) integrals

#ifdef OSHELL
                           DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)

                           DENSEJIA=quick_qm_struct%dense(JJJ,III)
                           DENSEJJA=quick_qm_struct%dense(JJJ,JJJ)
                           DENSEIIA=quick_qm_struct%dense(III,III)

                           DENSEJIB=quick_qm_struct%denseb(JJJ,III)
                           DENSEJJB=quick_qm_struct%denseb(JJJ,JJJ)
                           DENSEIIB=quick_qm_struct%denseb(III,III)

                           constant =(2.0d0*DENSEJI*DENSEJI-quick_method%x_hybrid_coeff*DENSEJIA*DENSEJIA &
                           -quick_method%x_hybrid_coeff*DENSEJJA*DENSEIIA &
                           -quick_method%x_hybrid_coeff*DENSEJIB*DENSEJIB &
                           -quick_method%x_hybrid_coeff*DENSEJJB*DENSEIIB)

#else
                           DENSEJI=quick_qm_struct%dense(JJJ,III)
                           DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)
                           DENSEII=quick_qm_struct%dense(III,III)

                           constant =(2.0d0*DENSEJI*DENSEJI-0.50d0*quick_method%x_hybrid_coeff*DENSEJI*DENSEJI &
                           -0.50d0*quick_method%x_hybrid_coeff*DENSEJJ*DENSEII)
#endif

                           Agrad1=Agrad1+Yaa(1)*constant
                           Agrad2=Agrad2+Yaa(2)*constant
                           Agrad3=Agrad3+Yaa(3)*constant
                           Bgrad1=Bgrad1+Ybb(1)*constant
                           Bgrad2=Bgrad2+Ybb(2)*constant
                           Bgrad3=Bgrad3+Ybb(3)*constant
                           Cgrad1=Cgrad1+Ycc(1)*constant
                           Cgrad2=Cgrad2+Ycc(2)*constant
                           Cgrad3=Cgrad3+Ycc(3)*constant

                           elseif(III.eq.KKK.and.III.lt.JJJ.and.JJJ.lt.LLL)then
                           ! Find all the (ij|ik) integrals where j>i,k>j
#ifdef OSHELL

                           DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)
                           DENSEKI=quick_qm_struct%dense(LLL,III)+quick_qm_struct%denseb(LLL,III)

                           DENSEKIA=quick_qm_struct%dense(LLL,III)
                           DENSEKJA=quick_qm_struct%dense(LLL,JJJ)
                           DENSEIIA=quick_qm_struct%dense(III,III)
                           DENSEJIA=quick_qm_struct%dense(JJJ,III)

                           DENSEKIB=quick_qm_struct%denseb(LLL,III)
                           DENSEKJB=quick_qm_struct%denseb(LLL,JJJ)
                           DENSEIIB=quick_qm_struct%denseb(III,III)
                           DENSEJIB=quick_qm_struct%denseb(JJJ,III)

                           constant = (4.0d0*DENSEJI*DENSEKI-2.0d0*quick_method%x_hybrid_coeff*DENSEJIA*DENSEKIA &
                           -2.0d0*quick_method%x_hybrid_coeff*DENSEKJA*DENSEIIA &
                           -2.0d0*quick_method%x_hybrid_coeff*DENSEJIB*DENSEKIB &
                           -2.0d0*quick_method%x_hybrid_coeff*DENSEKJB*DENSEIIB )

#else
                           DENSEKI=quick_qm_struct%dense(LLL,III)
                           DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                           DENSEII=quick_qm_struct%dense(III,III)
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           constant = (4.0d0*DENSEJI*DENSEKI-quick_method%x_hybrid_coeff*DENSEJI*DENSEKI &
                           -quick_method%x_hybrid_coeff*DENSEKJ*DENSEII)
#endif

                           Agrad1=Agrad1+Yaa(1)*constant
                           Agrad2=Agrad2+Yaa(2)*constant
                           Agrad3=Agrad3+Yaa(3)*constant
                           Bgrad1=Bgrad1+Ybb(1)*constant
                           Bgrad2=Bgrad2+Ybb(2)*constant
                           Bgrad3=Bgrad3+Ybb(3)*constant
                           Cgrad1=Cgrad1+Ycc(1)*constant
                           Cgrad2=Cgrad2+Ycc(2)*constant
                           Cgrad3=Cgrad3+Ycc(3)*constant

                        endif

                     endif
                  endif

               enddo
            enddo
         enddo
      enddo
   endif

   quick_qm_struct%gradient(iASTART+1) = quick_qm_struct%gradient(iASTART+1)+ AGrad1
   quick_qm_struct%gradient(iBSTART+1) = quick_qm_struct%gradient(iBSTART+1)+ BGrad1
   quick_qm_struct%gradient(iCSTART+1) = quick_qm_struct%gradient(iCSTART+1)+ CGrad1
   quick_qm_struct%gradient(iDSTART+1) = quick_qm_struct%gradient(iDSTART+1)- AGrad1-BGrad1-CGrad1

   quick_qm_struct%gradient(iASTART+2) = quick_qm_struct%gradient(iASTART+2)+ AGrad2
   quick_qm_struct%gradient(iBSTART+2) = quick_qm_struct%gradient(iBSTART+2)+ BGrad2
   quick_qm_struct%gradient(iCSTART+2) = quick_qm_struct%gradient(iCSTART+2)+ CGrad2
   quick_qm_struct%gradient(iDSTART+2) = quick_qm_struct%gradient(iDSTART+2)- AGrad2-BGrad2-CGrad2

   quick_qm_struct%gradient(iASTART+3) = quick_qm_struct%gradient(iASTART+3)+ AGrad3
   quick_qm_struct%gradient(iBSTART+3) = quick_qm_struct%gradient(iBSTART+3)+ BGrad3
   quick_qm_struct%gradient(iCSTART+3) = quick_qm_struct%gradient(iCSTART+3)+ CGrad3
   quick_qm_struct%gradient(iDSTART+3) = quick_qm_struct%gradient(iDSTART+3)- AGrad3-BGrad3-CGrad3

#ifdef OSHELL
end subroutine iclass_grad_oshell
#else
end subroutine iclass_grad_cshell
#endif

#ifdef OSHELL
end module quick_oshell_eri_grad_module
#else
end module quick_cshell_eri_grad_module
#endif
