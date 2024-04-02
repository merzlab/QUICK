!---------------------------------------------------------------------!
! Created by Madu Manathunga on 01/26/2021                            !
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

#include "util.fh"

! Xiao HE 07/07/07
! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
! Reference: Strout DL and Scuseria JCP 102(1995),8448.

module quick_cutoff_module
  implicit double precision(a-h,o-z)
  private
  public :: cshell_density_cutoff,oshell_density_cutoff
  public :: cshell_dnscreen,oshell_dnscreen
  public :: schwarzoff

  ! temporary variables required only for subroutines in this module
  double precision, dimension(:), allocatable :: X44
  double precision, dimension(:,:), allocatable :: X4444

contains

subroutine allocate_quick_cutoff
  use quick_basis_module
  implicit none
  
  if(.not. allocated(X44)) allocate(X44(maxcontract**4))
  if(.not. allocated(X4444)) allocate(X4444(maxcontract,maxcontract))

  X44 = 0.0d0
  X4444 = 0.0d0  

end subroutine allocate_quick_cutoff

subroutine deallocate_quick_cutoff
  implicit none

  if(allocated(X44)) deallocate(X44)
  if(allocated(X4444)) deallocate(X4444)

end subroutine deallocate_quick_cutoff

subroutine schwarzoff
  use allmod
#ifdef MPIV
  use mpi
#endif

  Implicit none

  integer ii,jj
  double precision Ymaxtemp

  if (master) then

  call allocate_quick_cutoff

  do II=1,nshell
     do JJ=II,nshell
        call shellcutoff(II,JJ,Ymaxtemp)
        Ycutoff(II,JJ)=dsqrt(Ymaxtemp)  ! Ycutoff(II,JJ) stands for (IJ|IJ)
        Ycutoff(JJ,II)=dsqrt(Ymaxtemp)
     enddo
  enddo

  call deallocate_quick_cutoff

  endif

#ifdef MPIV
      if (bMPI) then
         call MPI_BCAST(YCutoff,nshell*nshell,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(cutprim,jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      endif
#endif

end subroutine schwarzoff

subroutine shellcutoff(II,JJ,Ymax)
  use allmod

  Implicit double precision(a-h,o-z)
  double precision P(3),Q(3),W(3),KAB,KCD
  Parameter(NN=13)
  double precision FM(0:13)
  double precision RA(3),RB(3),RC(3),RD(3)

  double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp

  COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

  COMMON /COM1/RA,RB,RC,RD

  KK=II
  LL=JJ

  Ymax=0.0d0

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

  NNA=Sumindex(NII1-1)+1

  NNC=Sumindex(NKK1-1)+1

  NABCD=NII2+NJJ2+NKK2+NLL2
  ITT=0

  do JJJ=1,quick_basis%kprim(JJ)
     Nprij=quick_basis%kstart(JJ)+JJJ-1
     do III=1,quick_basis%kprim(II)
        Nprii=quick_basis%kstart(II)+III-1
        AB=Apri(Nprii,Nprij)
        ABtemp=0.5d0/AB
        do M=1,3
           P(M)=Ppri(M,Nprii,Nprij)
           Ptemp(M)=P(M)-RA(M)
        enddo
        do LLL=1,quick_basis%kprim(LL)
           Npril=quick_basis%kstart(LL)+LLL-1
           do KKK=1,quick_basis%kprim(KK)
              Nprik=quick_basis%kstart(KK)+KKK-1
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
              T=RPQ*ROU

              call FmT(NABCD,T,FM)
              do iitemp=0,NABCD
                 Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
              enddo

              ITT=ITT+1

              call vertical(NABCDTYPE)

              do I2=NNC,NNCD
                 do I1=NNA,NNAB
                    Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                 enddo
              enddo

              if(KKK.eq.III.and.JJJ.eq.LLL)then

                 do I2=NNC,NNCD
                    do I1=NNA,NNAB
                       Yxiaoprim(III,JJJ,I1,I2)=Yxiaotemp(I1,I2,0)
                    enddo
                 enddo

              endif
           enddo
        enddo
     enddo
  enddo
  do IIxiao=1,quick_basis%kprim(II)
     Nprii=quick_basis%kstart(II)+IIxiao-1
     do JJxiao=1,quick_basis%kprim(JJ)
        Nprij=quick_basis%kstart(JJ)+JJxiao-1

        Ymaxprim=0.0d0
        do I=NII1,NII2
           if(I.eq.0)then
              NNA=1
           else
              NNA=Sumindex(I-1)+1
           endif
           do J=NJJ1,NJJ2
              NNAB=SumINDEX(I+J)
              K=I
              L=J
              NNC=NNA
              NNCD=SumIndex(K+L)
              call classprim(I,J,K,L,II,JJ,KK,LL,NNA,NNC,NNAB,NNCD,Ymaxprim,IIxiao,JJxiao)
           enddo
        enddo

        cutprim(Nprii,Nprij)=dsqrt(Ymaxprim)

     enddo
  enddo

  do I=NII1,NII2
     if(I.eq.0)then
        NNA=1
     else
        NNA=Sumindex(I-1)+1
     endif


     do J=NJJ1,NJJ2
        NNAB=SumINDEX(I+J)
        NNC=NNA
        K=I
        L=J
        NNCD=SumIndex(K+L)
        call classcutoff(I,J,K,L,II,JJ,KK,LL,NNA,NNC,NNAB,NNCD,Ymax)
     enddo


  enddo

end subroutine shellcutoff

subroutine classcutoff(I,J,K,L,II,JJ,KK,LL,NNA,NNC,NNAB,NNCD,Ymax)
  use allmod

  Implicit double precision(A-H,O-Z)
  double precision store(120,120)
  INTEGER NA(3),NB(3),NC(3),ND(3)
  double precision P(3),Q(3),W(3),KAB,KCD
  Parameter(NN=13)
  double precision FM(0:13)
  double precision RA(3),RB(3),RC(3),RD(3)
!  double precision X44(100000)

  double precision coefangxiaoL(20),coefangxiaoR(20)
  integer angxiaoL(20),angxiaoR(20),numangularL,numangularR

  COMMON /COM1/RA,RB,RC,RD
  COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
  COMMON /COM4/P,Q,W
  COMMON /COM5/FM

  common /xiaostore/store

  ITT=0
  do JJJ=1,quick_basis%kprim(JJ)
     Nprij=quick_basis%kstart(JJ)+JJJ-1
     do III=1,quick_basis%kprim(II)
        Nprii=quick_basis%kstart(II)+III-1

        X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
        do LLL=1,quick_basis%kprim(LL)
           Npril=quick_basis%kstart(LL)+LLL-1
           do KKK=1,quick_basis%kprim(KK)
              Nprik=quick_basis%kstart(KK)+KKK-1
              ITT=ITT+1
              X44(ITT)=X2*quick_basis%Xcoeff(Nprik,Npril,K,L)
            enddo
        enddo
     enddo
  enddo
  do MM2=NNC,NNCD
     do MM1=NNA,NNAB
        Ytemp=0.0d0
        do itemp=1,ITT
           Ytemp=Ytemp+X44(itemp)*Yxiao(itemp,MM1,MM2)
        enddo
        store(MM1,MM2)=Ytemp
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

  do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
     do JJJ=quick_basis%ksumtype(JJ)+NBJ1,quick_basis%ksumtype(JJ)+NBJ2
        KKK=III
        LLL=JJJ

        if((I.eq.0.and.J.eq.0.and.K.eq.0.and.L.eq.0).or. &
             (I.eq.1.and.J.eq.0.and.K.eq.1.and.L.eq.0))then

           do M=1,3
              NA(M)=quick_basis%KLMN(M,III)
              NC(M)=quick_basis%KLMN(M,KKK)
           enddo

           M1=trans(NA(1),NA(2),NA(3))
           M3=trans(NC(1),NC(2),NC(3))
           Y=store(M1,M3)

        elseif(I.eq.0.and.J.eq.1.and.K.eq.0.and.L.eq.1)then

           do M=1,3
              NB(M)=quick_basis%KLMN(M,JJJ)
              ND(M)=quick_basis%KLMN(M,LLL)
           enddo
           M1=trans(NB(1),NB(2),NB(3))
           M3=trans(ND(1),ND(2),ND(3))

           do itemp=1,3
              if(ND(itemp).ne.0)then
                 ctemp=(RC(itemp)-RD(itemp))
                 Y1=store(M1,M3)+ctemp*store(M1,1)
                 Y2=store(1,M3)+ctemp*store(1,1)
                 goto 117
              endif
           enddo
117        continue

           do jtemp=1,3
              if(NB(jtemp).ne.0)then
                 Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
                 goto 118
              endif
           enddo
118        continue

        elseif(I.eq.1.and.J.eq.1.and.K.eq.1.and.L.eq.1)then

           do M=1,3
              NA(M)=quick_basis%KLMN(M,III)
              NB(M)=quick_basis%KLMN(M,JJJ)
              NC(M)=quick_basis%KLMN(M,KKK)
              ND(M)=quick_basis%KLMN(M,LLL)
           enddo

           MA=trans(NA(1),NA(2),NA(3))
           MAB=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
           MCX=trans(NC(1),NC(2),NC(3))
           MCD=trans(NC(1)+ND(1),NC(2)+ND(2),NC(3)+ND(3))

           do itemp=1,3
              if(ND(itemp).ne.0)then
                 ctemp=(RC(itemp)-RD(itemp))
                 Y1=store(MAB,MCD)+ctemp*store(MAB,MCX)
                 Y2=store(MA,MCD)+ctemp*store(MA,MCX)
                 goto 141
              endif
           enddo
141        continue

           do jtemp=1,3
              if(NB(jtemp).ne.0)then
                 Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
                 goto 142
              endif
           enddo
142        continue

        elseif((I.eq.2.and.J.eq.0.and.K.eq.2.and.L.eq.0).or. &
             (I.eq.0.and.J.eq.2.and.K.eq.0.and.L.eq.2).or. &
             (I.eq.2.and.J.eq.1.and.K.eq.2.and.L.eq.1).or. &
             (I.eq.1.and.J.eq.2.and.K.eq.1.and.L.eq.2).or. &
             (I.eq.2.and.J.eq.2.and.K.eq.2.and.L.eq.2).or. &
             (I.eq.3.and.J.eq.0.and.K.eq.3.and.L.eq.0).or. &
             (I.eq.0.and.J.eq.3.and.K.eq.0.and.L.eq.3).or. &
             (I.eq.3.and.J.eq.1.and.K.eq.3.and.L.eq.1).or. &
             (I.eq.1.and.J.eq.3.and.K.eq.1.and.L.eq.3).or. &
             (I.eq.3.and.J.eq.2.and.K.eq.3.and.L.eq.2).or. &
             (I.eq.2.and.J.eq.3.and.K.eq.2.and.L.eq.3).or. &
             (I.eq.3.and.J.eq.3.and.K.eq.3.and.L.eq.3))then

           IJtype=10*I+J
           KLtype=10*K+L

           call lefthrr(RA,RB,quick_basis%KLMN(1:3,III),quick_basis%KLMN(1:3,JJJ),IJtype,coefangxiaoL,angxiaoL,numangularL)
           call lefthrr(RC,RD,quick_basis%KLMN(1:3,KKK),quick_basis%KLMN(1:3,LLL),KLtype,coefangxiaoR,angxiaoR,numangularR)

           Y=0.0d0
           do ixiao=1,numangularL
              do jxiao=1,numangularR
                 Y=Y+coefangxiaoL(ixiao)*coefangxiaoR(jxiao)*store(angxiaoL(ixiao),angxiaoR(jxiao))
              enddo
           enddo

           Y=Y*quick_basis%cons(III)*quick_basis%cons(JJJ)*quick_basis%cons(KKK)*quick_basis%cons(LLL)

        endif
        Ytemp=dabs(Y)
        if(dabs(Ytemp).gt.Ymax) Ymax=Ytemp

     enddo
  enddo

End subroutine classcutoff

subroutine classprim(I,J,K,L,II,JJ,KK,LL,NNA,NNC,NNAB,NNCD,Ymax1,IIIxiao,JJJxiao)
  use allmod

  Implicit double precision(A-H,O-Z)
  double precision store(120,120)
  INTEGER NA(3),NB(3),NC(3),ND(3)
  double precision P(3),Q(3),W(3),KAB,KCD
  Parameter(NN=13)
  double precision FM(0:13)
  double precision RA(3),RB(3),RC(3),RD(3)
!  double precision X44(12960)
!  double precision X4444(MAXPRIM,MAXPRIM)

  double precision coefangxiaoL(20),coefangxiaoR(20)
  integer angxiaoL(20),angxiaoR(20),numangularL,numangularR

  COMMON /COM1/RA,RB,RC,RD
  COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
  COMMON /COM4/P,Q,W
  COMMON /COM5/FM

  common /xiaostore/store


  ITT=0
  do JJJ=1,quick_basis%kprim(JJ)
     Nprij=quick_basis%kstart(JJ)+JJJ-1
     do III=1,quick_basis%kprim(II)
        Nprii=quick_basis%kstart(II)+III-1
        X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
        X4444(III,JJJ)=X2
     enddo
  enddo

  do MM2=NNC,NNCD
     do MM1=NNA,NNAB
        Ytemp=Yxiaoprim(IIIxiao,JJJxiao,MM1,MM2)*X4444(IIIxiao,JJJxiao)**2.0d0
        store(MM1,MM2)=Ytemp
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

  do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
     do JJJ=quick_basis%ksumtype(JJ)+NBJ1,quick_basis%ksumtype(JJ)+NBJ2
        KKK=III
        LLL=JJJ

        if((I.eq.0.and.J.eq.0.and.K.eq.0.and.L.eq.0).or. &
             (I.eq.1.and.J.eq.0.and.K.eq.1.and.L.eq.0))then

           do M=1,3
              NA(M)=quick_basis%KLMN(M,III)
              NC(M)=quick_basis%KLMN(M,KKK)
           enddo

           M1=trans(NA(1),NA(2),NA(3))
           M3=trans(NC(1),NC(2),NC(3))
           Y=store(M1,M3)

        elseif(I.eq.0.and.J.eq.1.and.K.eq.0.and.L.eq.1)then

           do M=1,3
              NB(M)=quick_basis%KLMN(M,JJJ)
              ND(M)=quick_basis%KLMN(M,LLL)
           enddo
           M1=trans(NB(1),NB(2),NB(3))
           M3=trans(ND(1),ND(2),ND(3))

           do itemp=1,3
              if(ND(itemp).ne.0)then
                 ctemp=(RC(itemp)-RD(itemp))
                 Y1=store(M1,M3)+ctemp*store(M1,1)
                 Y2=store(1,M3)+ctemp*store(1,1)
                 goto 117
              endif
           enddo
117        continue

           do jtemp=1,3
              if(NB(jtemp).ne.0)then
                 Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
                 goto 118
              endif
           enddo
118        continue

        elseif(I.eq.1.and.J.eq.1.and.K.eq.1.and.L.eq.1)then

           do M=1,3
              NA(M)=quick_basis%KLMN(M,III)
              NB(M)=quick_basis%KLMN(M,JJJ)
              NC(M)=quick_basis%KLMN(M,KKK)
              ND(M)=quick_basis%KLMN(M,LLL)
           enddo
           MA=trans(NA(1),NA(2),NA(3))
           MAB=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
           MCX=trans(NC(1),NC(2),NC(3))
           MCD=trans(NC(1)+ND(1),NC(2)+ND(2),NC(3)+ND(3))

          do itemp=1,3
              if(ND(itemp).ne.0)then
                 ctemp=(RC(itemp)-RD(itemp))
                 Y1=store(MAB,MCD)+ctemp*store(MAB,MCX)
                 Y2=store(MA,MCD)+ctemp*store(MA,MCX)
                 goto 141
              endif
           enddo
141        continue

           do jtemp=1,3
              if(NB(jtemp).ne.0)then
                 Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
                 goto 142
              endif
           enddo
142        continue

        elseif((I.eq.2.and.J.eq.0.and.K.eq.2.and.L.eq.0).or. &
             (I.eq.0.and.J.eq.2.and.K.eq.0.and.L.eq.2).or. &
             (I.eq.2.and.J.eq.1.and.K.eq.2.and.L.eq.1).or. &
             (I.eq.1.and.J.eq.2.and.K.eq.1.and.L.eq.2).or. &
             (I.eq.2.and.J.eq.2.and.K.eq.2.and.L.eq.2).or. &
             (I.eq.3.and.J.eq.0.and.K.eq.3.and.L.eq.0).or. &
             (I.eq.0.and.J.eq.3.and.K.eq.0.and.L.eq.3).or. &
             (I.eq.3.and.J.eq.1.and.K.eq.3.and.L.eq.1).or. &
             (I.eq.1.and.J.eq.3.and.K.eq.1.and.L.eq.3).or. &
             (I.eq.3.and.J.eq.2.and.K.eq.3.and.L.eq.2).or. &
             (I.eq.2.and.J.eq.3.and.K.eq.2.and.L.eq.3).or. &
             (I.eq.3.and.J.eq.3.and.K.eq.3.and.L.eq.3))then

           IJtype=10*I+J
           KLtype=10*K+L

           call lefthrr(RA,RB,quick_basis%KLMN(1:3,III),quick_basis%KLMN(1:3,JJJ),IJtype,coefangxiaoL,angxiaoL,numangularL)
           call lefthrr(RC,RD,quick_basis%KLMN(1:3,KKK),quick_basis%KLMN(1:3,LLL),KLtype,coefangxiaoR,angxiaoR,numangularR)

           Y=0.0d0
           do ixiao=1,numangularL
              do jxiao=1,numangularR
                 Y=Y+coefangxiaoL(ixiao)*coefangxiaoR(jxiao)* &
                      store(angxiaoL(ixiao),angxiaoR(jxiao))
              enddo
           enddo

           Y=Y*quick_basis%cons(III)*quick_basis%cons(JJJ)*quick_basis%cons(KKK)*quick_basis%cons(LLL)

        endif

        Ytemp=dabs(Y)
        if(dabs(Ytemp).gt.Ymax1)Ymax1=Ytemp

     enddo
  enddo

end subroutine classprim

#define OSHELL
#include "./include/cutoff.fh"
#undef OSHELL
#include "./include/cutoff.fh"

end module quick_cutoff_module
