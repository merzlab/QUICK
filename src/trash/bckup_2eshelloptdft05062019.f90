!!!!!!!!BE careful of IJKLtype and Fmt.f!!!!!!!That's the difference of 6d and 10f
! vertical and hrr for gradient 10/01/2007

!Be careful of store(1,1),IASTART,size of FM, STORE, coefangxiaoL(4),coefangxiaoR(4)!!!!
! Vertical Recursion by Xiao HE 07/07/07 version
 subroutine shelloptdft(IItemp,JJtemp,KKtemp,LLtemp)
 use allmod

 implicit real*8(a-h,o-z)
 real*8 P(3),Q(3),W(3),KAB,KCD
 Parameter(NN=14)
 real*8 FM(0:14)
 real*8 RA(3),RB(3),RC(3),RD(3)

 real*8 Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
 integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,NNABfirst,NNCDfirst
 common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

 common /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

 common /COM1/RA,RB,RC,RD

 II=IItemp
 JJ=JJtemp
 KK=KKtemp
 LL=LLtemp

!    logical same
!    same = .false.

! print*,II,JJ,KK,LL

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

!     print*,'NNCDfirst=',KK,LL,NKK1,NKK2,NLL1,NLL2,NNCD,NNCDfirst

!      NNA=Sumindex(NII1-1)+1

!            NNC=Sumindex(NKK1-1)+1

!       NNA=1
!       NNC=1

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
                     if(cutoffprim.gt.gradcutoff)then
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
            call classoptdft(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
!                   call class
          enddo
        enddo
   enddo
 enddo
 
 end

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
 subroutine classoptdft(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
! subroutine class
 use allmod

 implicit real*8(A-H,O-Z)
 real*8 store(120,120)
 
 real*8 storeaa(120,120)
 real*8 storebb(120,120)
 real*8 storecc(120,120) 
 real*8 storedd(120,120)

 integer NA(3),NB(3),NC(3),ND(3)
 real*8 P(3),Q(3),W(3),KAB,KCD
 parameter(NN=14)
 real*8 FM(0:14)
 real*8 RA(3),RB(3),RC(3),RD(3)
 real*8 X44(1296)

 real*8 X44aa(1296)
 real*8 X44bb(1296)
 real*8 X44cc(1296)
 real*8 X44dd(1296)
 real*8 AA,BB,CC,DD

 common /COM1/RA,RB,RC,RD
 common /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
 common /COM4/P,Q,W
 common /COM5/FM

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

      X2=X0*quick_basis%xcoeff(Nprii,Nprij,I,J)
      cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            DD=quick_basis%gcexpo(LLL,quick_basis%ksumtype(LL))
              do KKK=1,quick_basis%kprim(KK)
                Nprik=quick_basis%kstart(KK)+KKK-1
                CC=quick_basis%gcexpo(KKK,quick_basis%ksumtype(KK))
                     cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
                     if(cutoffprim.gt.quick_method%gradcutoff)then
                       ITT=ITT+1
                       X44(ITT)=X2*quick_basis%xcoeff(Nprik,Npril,K,L)
                       X44AA(ITT)=X44(ITT)*AA*2.0d0
                       X44BB(ITT)=X44(ITT)*BB*2.0d0
                       X44CC(ITT)=X44(ITT)*CC*2.0d0
!                       X44DD(ITT)=X44(ITT)*DD*2.0d0
                     endif
               enddo
            enddo
        enddo
     enddo

!       do III=1,Kprim(II)
!         AA=gcexpo(III,ksumtype(II))


                       do MM2=NNC,NNCD
                         do MM1=NNA,NNAB
                           Ytemp=0.0d0
!                           YtempAA=0.0d0
!                           YtempBB=0.0d0
!                           YtempCC=0.0d0
                           do itemp=1,ITT
                           Ytemp=Ytemp+X44(itemp)*Yxiao(itemp,MM1,MM2)
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
                           YtempAA=YtempAA+X44AA(itemp)*Yxiao(itemp,MM1,MM2)
                           YtempBB=YtempBB+X44BB(itemp)*Yxiao(itemp,MM1,MM2)
                           YtempCC=YtempCC+X44CC(itemp)*Yxiao(itemp,MM1,MM2)

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
  
!    iA = ncenter(II)
!    iB = ncenter(JJ)
!    iC = ncenter(KK)
!    iD = ncenter(LL)

!    same = iA.eq.iB .and. iB.eq.iC.and. iC.eq.iD

!    print*,II,JJ,KK,LL

!    IF (same.eq..true.) return

!    iAstart = (iA-1)*3
!    iBstart = (iB-1)*3
!    iCstart = (iC-1)*3
!    iDstart = (iD-1)*3

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

!       IF(max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0))IJKLtype=999
!       IJKLtype=999
!      If(J.eq.0.and.L.eq.0)then

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

!       Do III=Ksumtype(II)+NBI1,Ksumtype(II)+NBI2
!         Do JJJ=Ksumtype(JJ)+NBJ1,Ksumtype(JJ)+NBJ2
!            Do KKK=Ksumtype(KK)+NBK1,Ksumtype(KK)+NBK2
!              Do LLL=Ksumtype(LL)+NBL1,Ksumtype(LL)+NBL2

       do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
              do LLL=LLL1,LLL2

!                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                call hrrwholeopt

                    DENSELK=quick_qm_struct%dense(LLL,KKK)
                    DENSEJI=quick_qm_struct%dense(JJJ,III)
                ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                ! can be equal.

!                    O(JJJ,III) = O(JJJ,III)+2.d0*DENSELK*Y
!                    O(LLL,KKK) = O(LLL,KKK)+2.d0*DENSEJI*Y
!                    O(KKK,III) = O(KKK,III)-.5d0*DENSELJ*Y
!                    O(LLL,III) = O(LLL,III)-.5d0*DENSEKJ*Y
!                        O(JJJ,KKK) = O(JJJ,KKK)-.5d0*DENSELI*Y
!                        O(JJJ,LLL) = O(JJJ,LLL)-.5d0*DENSEKI*Y
!                        O(KKK,JJJ) = O(KKK,JJJ)-.5d0*DENSELI*Y
!                        O(LLL,JJJ) = O(LLL,JJJ)-.5d0*DENSEKI*Y

                    constant = 4.d0*DENSEJI*DENSELK

!                    print*,'here',constant                    
 
                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant
 
!                      print*,III,JJJ,KKK,LLL,Yaa(1)*constant,Ybb(1)*constant,Ycc(1)*constant, &
!                             Yaa(2)*constant,Ybb(2)*constant,Ycc(2)*constant, &
!                             Yaa(3)*constant,Ybb(3)*constant,Ycc(3)*constant

!                      print*,III,JJJ,KKK,LLL,Yaa(1),Ybb(1),Ycc(1), &
!                             Yaa(2),Ybb(2),Ycc(2), &
!                             Yaa(3),Ybb(3),Ycc(3)


!            if(III.eq.4.and.JJJ.eq.8.and.KKK.eq.8.and.LLL.eq.10)then
!                do ixiao=1,20
!                  do jxiao=1,10
!                    print*,ixiao,jxiao,storeAA(ixiao,jxiao),KLMN(1:3,III),KLMN(1:3,JJJ),&
!                           KLMN(1:3,KKK),KLMN(1:3,LLL),store(ixiao,jxiao),Yaa(1), &
!                           Yaa(2),Yaa(3)
!                  enddo
!                enddo
!            endif

!    iA = ncenter(III)        
!    iB = ncenter(JJJ)        
!    iC = ncenter(KKK)
!    iD = ncenter(LLL)
!
!    iAstart = (iA-1)*3
!    iBstart = (iB-1)*3
!    iCstart = (iC-1)*3    
!    iDstart = (iD-1)*3      

                      enddo
                    enddo
                  enddo
                enddo

       else

!       Do III=Ksumtype(II)+NBI1,Ksumtype(II)+NBI2
!         Do JJJ=max(III,Ksumtype(JJ)+NBJ1),Ksumtype(JJ)+NBJ2
!            Do KKK=max(III,Ksumtype(KK)+NBK1),Ksumtype(KK)+NBK2
!              Do LLL=max(KKK,Ksumtype(LL)+NBL1),Ksumtype(LL)+NBL2

       do III=III1,III2                          
         do JJJ=max(III,JJJ1),JJJ2                          
            do KKK=max(III,KKK1),KKK2                          
              do LLL=max(KKK,LLL1),LLL2                          

!    iA = ncenter(III)
!    iB = ncenter(JJJ)
!    iC = ncenter(KKK)
!    iD = ncenter(LLL)
!
!    iAstart = (iA-1)*3
!    iBstart = (iB-1)*3
!    iCstart = (iC-1)*3
!    iDstart = (iD-1)*3

                 if(III.LT.KKK)then

!                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                call hrrwholeopt

         if(III.lt.JJJ.and.KKK.lt.LLL)then
                    DENSELK=quick_qm_struct%dense(LLL,KKK)
                    DENSEJI=quick_qm_struct%dense(JJJ,III)
                ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                ! can be equal.

!                    O(JJJ,III) = O(JJJ,III)+2.d0*DENSELK*Y
!                    O(LLL,KKK) = O(LLL,KKK)+2.d0*DENSEJI*Y
!                    O(KKK,III) = O(KKK,III)-.5d0*DENSELJ*Y
!                    O(LLL,III) = O(LLL,III)-.5d0*DENSEKJ*Y
!                        O(JJJ,KKK) = O(JJJ,KKK)-.5d0*DENSELI*Y
!                        O(JJJ,LLL) = O(JJJ,LLL)-.5d0*DENSEKI*Y
!                        O(KKK,JJJ) = O(KKK,JJJ)-.5d0*DENSELI*Y
!                        O(LLL,JJJ) = O(LLL,JJJ)-.5d0*DENSEKI*Y

                    constant = 4.d0*DENSEJI*DENSELK

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

!                      print*,III,JJJ,KKK,LLL,Yaa(1)*constant,Ybb(1)*constant,Ycc(1)*constant, &
!                             Yaa(2)*constant,Ybb(2)*constant,Ycc(2)*constant, &
!                             Yaa(3)*constant,Ybb(3)*constant,Ycc(3)*constant

!        ! DO all the (ii|ii) integrals.
!        ! Set some variables to reduce access time for some of the more
!        ! used quantities. (AGAIN)
         elseif(III.eq.JJJ.and.KKK.eq.LLL)then
            DENSEJJ=quick_qm_struct%dense(KKK,KKK)
            DENSEII=quick_qm_struct%dense(III,III)

        ! Find  all the (ii|jj) integrals.
!            O(III,III) = O(III,III)+DENSEJJ*Y
!            O(KKK,KKK) = O(KKK,KKK)+DENSEII*Y
!            O(KKK,III) = O(KKK,III)-.5d0*DENSEJI*Y

            constant = DENSEII*DENSEJJ

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
            DENSEJI=quick_qm_struct%dense(JJJ,III)
            DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

        ! Find  all the (ij|jj) integrals.
!            O(JJJ,III) = O(JJJ,III)+.5d0*DENSEJJ*Y
!            O(JJJ,JJJ) = O(JJJ,JJJ)+DENSEJI*Y

            constant =  2.0d0*DENSEJJ*DENSEJI

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
                DENSEKK=quick_qm_struct%dense(KKK,KKK)
                DENSEJI=quick_qm_struct%dense(JJJ,III)

            ! Find all the (ij|kk) integrals where j>i, k>j.
!                O(JJJ,III) = O(JJJ,III)+DENSEKK*Y
!                O(KKK,KKK) = O(KKK,KKK)+2.d0*DENSEJI*Y
!                O(KKK,III) = O(KKK,III)-.5d0*DENSEKJ*Y
!                O(KKK,JJJ) = O(KKK,JJJ)-.5d0*DENSEKI*Y
!                O(JJJ,KKK) = O(JJJ,KKK)-.5d0*DENSEKI*Y

                constant=2.d0*DENSEJI*DENSEKK

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
                DENSEII=quick_qm_struct%dense(III,III)
                DENSEKJ=quick_qm_struct%dense(LLL,KKK)

            ! Find all the (ii|jk) integrals where j>i, k>j.
!                O(LLL,KKK) = O(LLL,KKK)+DENSEII*Y
!                O(III,III) = O(III,III)+2.d0*DENSEKJ*Y
!                O(KKK,III) = O(KKK,III)-.5d0*DENSEKI*Y
!                O(LLL,III) = O(LLL,III)-.5d0*DENSEJI*Y

                constant = 2.d0*DENSEKJ*DENSEII

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
                    If(JJJ.LE.LLL)then

!                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                call hrrwholeopt

      if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then
        DENSEII=quick_qm_struct%dense(III,III)

    ! DO all the (ii|ii) integrals.
!        O(III,III) = O(III,III)+.5d0*DENSEII*Y

        constant=0.0d0          
 
        elseif(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then
            DENSEJI=quick_qm_struct%dense(LLL,III)
            DENSEII=quick_qm_struct%dense(III,III)

        ! Find  all the (ii|ij) integrals.
!            O(LLL,III) = O(LLL,III)+.5d0*DENSEII*Y
!            O(III,III) = O(III,III)+DENSEJI*Y

            constant= 2.0d0*DENSEJI*DENSEII

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
            DENSEJI=quick_qm_struct%dense(JJJ,III)

        ! Find all the (ij|ij) integrals
!            O(JJJ,III) = O(JJJ,III)+1.50*DENSEJI*Y
!            O(JJJ,JJJ) = O(JJJ,JJJ)-.5d0*DENSEII*Y
!            O(III,III) = O(III,III)-.5d0*DENSEJJ*Y

            constant =2.0d0*DENSEJI*DENSEJI

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
                DENSEKI=quick_qm_struct%dense(LLL,III)
                DENSEJI=quick_qm_struct%dense(JJJ,III)

            ! Find all the (ij|ik) integrals where j>i,k>j
!                O(JJJ,III) = O(JJJ,III)+1.5d0*DENSEKI*Y
!                O(LLL,III) = O(LLL,III)+1.5d0*DENSEJI*Y
!                O(III,III) = O(III,III)-1.d0*DENSEKJ*Y
!                O(LLL,JJJ) = O(LLL,JJJ)-.5d0*DENSEII*Y

                constant = 4.0d0*DENSEJI*DENSEKI

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
   
     quick_qm_struct%gradient(iASTART+1) = quick_qm_struct%gradient(iASTART+1)+ &
        AGrad1
        quick_qm_struct%gradient(iBSTART+1) = quick_qm_struct%gradient(iBSTART+1)+ &
        BGrad1
        quick_qm_struct%gradient(iCSTART+1) = quick_qm_struct%gradient(iCSTART+1)+ &
        CGrad1
        quick_qm_struct%gradient(iDSTART+1) = quick_qm_struct%gradient(iDSTART+1) &
         -AGrad1-BGrad1-CGrad1

         quick_qm_struct%gradient(iASTART+2) = quick_qm_struct%gradient(iASTART+2)+ &
        AGrad2
        quick_qm_struct%gradient(iBSTART+2) = quick_qm_struct%gradient(iBSTART+2)+ &
        BGrad2
        quick_qm_struct%gradient(iCSTART+2) = quick_qm_struct%gradient(iCSTART+2)+ &
        CGrad2
        quick_qm_struct%gradient(iDSTART+2) = quick_qm_struct%gradient(iDSTART+2) &
         -AGrad2-BGrad2-CGrad2

         quick_qm_struct%gradient(iASTART+3) = quick_qm_struct%gradient(iASTART+3)+ &
        AGrad3
        quick_qm_struct%gradient(iBSTART+3) = quick_qm_struct%gradient(iBSTART+3)+ &
        BGrad3
        quick_qm_struct%gradient(iCSTART+3) = quick_qm_struct%gradient(iCSTART+3)+ &
        CGrad3
        quick_qm_struct%gradient(iDSTART+3) = quick_qm_struct%gradient(iDSTART+3) &
         -AGrad3-BGrad3-CGrad3
!   print*,Agrad1,BGrad1,CGrad1,AGrad2,BGrad2,CGrad2,AGrad3,BGrad3,CGrad3

!    print*,'here' 
!    print*,constant,Yaa1,Yaa2,Yaa3,Ybb1,Ybb2,Ybb3,Ycc1,Ycc2,Ycc3
    
!      stop 
     end
      
