#include "util.fh"


! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shellmp2divcon(i33,ittsub)
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
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do M=1,3
            P(M)=Ppri(M,Nprii,Nprij)
            AAtemp(M)=P(M)*AB
            Ptemp(M)=P(M)-RA(M)
         enddo
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%primLimit)then
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
                     W(M)=(AAtemp(M)+Q(M)*CD)/ABCD
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
               call classmp2divcon(I,J,K,L,NNA,NNC,NNAB,NNCD,i33,ittsub)
            enddo
         enddo
      enddo
   enddo

end subroutine shellmp2divcon

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine classmp2divcon(I,J,K,L,NNA,NNC,NNAB,NNCD,i33,ittsub)
   ! subroutine class
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)
   double precision X44(129600)

   COMMON /COM1/RA,RB,RC,RD
   COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
   COMMON /COM4/P,Q,W
   COMMON /COM5/FM

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   ITT=0

   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%primLimit)then
                  ITT=ITT+1
                  X44(ITT)=X2*quick_basis%Xcoeff(Nprik,Npril,K,L)
               endif
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

   !       IJKLtype=1000*I+100*J+10*K+L
   IJtype=10*I+J
   KLtype=10*K+L
   IJKLtype=100*IJtype+KLtype

   !*****       if(max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0))IJKLtype=999
   if((max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0)).or.(max(I,J,K,L).ge.3))IJKLtype=999
   !       IJKLtype=999
   !      if(J.eq.0.and.L.eq.0)then

   III1=quick_basis%ksumtype(II)+NBI1
   III2=quick_basis%ksumtype(II)+NBI2
   JJJ1=quick_basis%ksumtype(JJ)+NBJ1
   JJJ2=quick_basis%ksumtype(JJ)+NBJ2
   KKK1=quick_basis%ksumtype(KK)+NBK1
   KKK2=quick_basis%ksumtype(KK)+NBK2
   LLL1=quick_basis%ksumtype(LL)+NBL1
   LLL2=quick_basis%ksumtype(LL)+NBL2


   NII1=quick_basis%Qstart(II)
   NJJ1=quick_basis%Qstart(JJ)

   NBI1=quick_basis%Qsbasis(II,NII1)
   NBJ1=quick_basis%Qsbasis(JJ,NJJ1)

   II111=quick_basis%ksumtype(II)+NBI1
   JJ111=quick_basis%ksumtype(JJ)+NBJ1

   if(II.lt.JJ.and.KK.lt.LL)then

      do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
               do LLL=LLL1,LLL2

                  call hrrwhole

                  KKKsub=wtospoint(ittsub,KKK)
                  LLLsub=wtospoint(ittsub,LLL)

                  atemp=quick_qm_struct%co(KKKsub,i33)*Y
                  btemp=quick_qm_struct%co(LLLsub,i33)*Y

                  IIInew=III-II111+1
                  JJJnew=JJJ-JJ111+1

                  orbmp2i331(1,LLLsub,IIInew,JJJnew,1)=orbmp2i331(1,LLLsub,IIInew,JJJnew,1)+atemp
                  orbmp2i331(1,LLLsub,JJJnew,IIInew,2)=orbmp2i331(1,LLLsub,JJJnew,IIInew,2)+atemp
                  orbmp2i331(1,KKKsub,IIInew,JJJnew,1)=orbmp2i331(1,KKKsub,IIInew,JJJnew,1)+btemp
                  orbmp2i331(1,KKKsub,JJJnew,IIInew,2)=orbmp2i331(1,KKKsub,JJJnew,IIInew,2)+btemp

               enddo
            enddo
         enddo
      enddo

   else

      do III=III1,III2
         !         if(max(III,JJJ1).le.JJJ2)then
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=KKK1,KKK2
               !            if(max(KKK,LLL1).le.LLL2)then
               do LLL=max(KKK,LLL1),LLL2

                  call hrrwhole

                  KKKsub=wtospoint(ittsub,KKK)
                  LLLsub=wtospoint(ittsub,LLL)

                  atemp=quick_qm_struct%co(KKKsub,i33)*Y
                  btemp=quick_qm_struct%co(LLLsub,i33)*Y

                  IIInew=III-II111+1
                  JJJnew=JJJ-JJ111+1

                  !                 mp2shell(KKK)=.true.
                  !                 mp2shell(LLL)=.true.

                  orbmp2i331(1,LLLsub,IIInew,JJJnew,1)=orbmp2i331(1,LLLsub,IIInew,JJJnew,1)+atemp
                  if(JJJ.ne.III)then
                     orbmp2i331(1,LLLsub,JJJnew,IIInew,2)=orbmp2i331(1,LLLsub,JJJnew,IIInew,2)+atemp
                  endif
                  if(KKK.ne.LLL)then
                     orbmp2i331(1,KKKsub,IIInew,JJJnew,1)=orbmp2i331(1,KKKsub,IIInew,JJJnew,1)+btemp
                     if(III.ne.JJJ)then
                        orbmp2i331(1,KKKsub,JJJnew,IIInew,2)=orbmp2i331(1,KKKsub,JJJnew,IIInew,2)+btemp
                     endif
                  endif

               enddo
               !            endif
            enddo
         enddo
         !        endif
      enddo

   endif

End subroutine classmp2divcon


