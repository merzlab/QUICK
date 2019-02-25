! Be careful of (NB(itemp).eq.2.and.NB(jtemp).eq.2)
! Xiao HE 07/07/07 version
! BE careful of the array coefangxiaoL(20),allocate,main,module array,2eshell(opt),hrrsub,vertical
! subroutine hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
!Horrizontal Recursion subroutines by hand, these parts can be optimized by MAPLE
subroutine hrrwhole
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision RA(3),RB(3),RC(3),RD(3)
   Integer M1,M2,M3,M4

   double precision coefangxiaoL(20),coefangxiaoR(20)
   integer angxiaoL(20),angxiaoR(20),numangularL,numangularR

   COMMON /COM1/RA,RB,RC,RD

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   select case (IJKLtype)

   case (0,10,1000,1010)

      M1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
      M3=trans(quick_basis%KLMN(1,KKK),quick_basis%KLMN(2,KKK),quick_basis%KLMN(3,KKK))
      Y=store(M1,M3)
   case (2000,20,2010,1020,2020)
      M1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
      M3=trans(quick_basis%KLMN(1,KKK),quick_basis%KLMN(2,KKK),quick_basis%KLMN(3,KKK))
      Y=store(M1,M3)

      Y=Y*quick_basis%cons(III)*quick_basis%cons(JJJ)*quick_basis%cons(KKK)*quick_basis%cons(LLL)

   case(100)
      do M=1,3
         NB(M)=quick_basis%KLMN(M,JJJ)
      enddo
      M1=trans(NB(1),NB(2),NB(3))
      do itemp=1,3
         if(NB(itemp).ne.0)then
            Y=store(M1,1)+(RA(itemp)-RB(itemp))*store(1,1)
            goto 111
         endif
      enddo

   case(110)
      do M=1,3
         NB(M)=quick_basis%KLMN(M,JJJ)
         NC(M)=quick_basis%KLMN(M,KKK)
      enddo
      M1=trans(NB(1),NB(2),NB(3))
      M3=trans(NC(1),NC(2),NC(3))
      do itemp=1,3
         if(NB(itemp).ne.0)then
            Y=store(M1,M3)+(RA(itemp)-RB(itemp))*store(1,M3)
            goto 111
         endif
      enddo

   case(101)
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
      117  continue

      do jtemp=1,3
         if(NB(jtemp).ne.0)then
            Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
            goto 111
         endif
      enddo

   case(111)
      do M=1,3
         NB(M)=quick_basis%KLMN(M,JJJ)
         NC(M)=quick_basis%KLMN(M,KKK)
         ND(M)=quick_basis%KLMN(M,LLL)
      enddo
      MB=trans(NB(1),NB(2),NB(3))
      MCX=trans(NC(1),NC(2),NC(3))
      MCD=trans(NC(1)+ND(1),NC(2)+ND(2),NC(3)+ND(3))

      do itemp=1,3
         if(ND(itemp).ne.0)then
            ctemp=(RC(itemp)-RD(itemp))
            Y1=store(MB,MCD)+ctemp*store(MB,MCX)
            Y2=store(1,MCD)+ctemp*store(1,MCX)
            goto 1230
         endif
      enddo
      1230 continue

      do jtemp=1,3
         if(NB(jtemp).ne.0)then
            Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
            goto 111
         endif
      enddo

   case(1100)
      do M=1,3
         NA(M)=quick_basis%KLMN(M,III)
         NB(M)=quick_basis%KLMN(M,JJJ)
      enddo
      MA=trans(NA(1),NA(2),NA(3))
      MAB=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      do itemp=1,3
         if(NB(itemp).ne.0)then
            Y=store(MAB,1)+(RA(itemp)-RB(itemp))*store(MA,1)
            goto 111
         endif
      enddo

   case(1110)
      do M=1,3
         NA(M)=quick_basis%KLMN(M,III)
         NB(M)=quick_basis%KLMN(M,JJJ)
         NC(M)=quick_basis%KLMN(M,KKK)
      enddo
      MA=trans(NA(1),NA(2),NA(3))
      MAB=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      MCX=trans(NC(1),NC(2),NC(3))
      do itemp=1,3
         if(NB(itemp).ne.0)then
            Y=store(MAB,MCX)+(RA(itemp)-RB(itemp))*store(MA,MCX)
            goto 111
         endif
      enddo

   case(1101)
      do M=1,3
         NA(M)=quick_basis%KLMN(M,III)
         NB(M)=quick_basis%KLMN(M,JJJ)
         ND(M)=quick_basis%KLMN(M,LLL)
      enddo
      MA=trans(NA(1),NA(2),NA(3))
      MAB=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      MCX=trans(ND(1),ND(2),ND(3))

      do itemp=1,3
         if(ND(itemp).ne.0)then
            ctemp=(RC(itemp)-RD(itemp))
            Y1=store(MAB,MCX)+ctemp*store(MAB,1)
            Y2=store(MA,MCX)+ctemp*store(MA,1)
            goto 135
         endif
      enddo
      135  continue

      do jtemp=1,3
         if(NB(jtemp).ne.0)then
            Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
            goto 111
         endif
      enddo

   case(1111)
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
      141  continue

      do jtemp=1,3
         if(NB(jtemp).ne.0)then
            Y=Y1+(RA(jtemp)-RB(jtemp))*Y2
            goto 111
         endif
      enddo

   case(1)
      do M=1,3
         ND(M)=quick_basis%KLMN(M,LLL)
      enddo

      MD=trans(ND(1),ND(2),ND(3))
      do itemp=1,3
         if(ND(itemp).ne.0)then
            Y=store(1,MD)+(RC(itemp)-RD(itemp))*store(1,1)
            goto 111
         endif
      enddo

   case(11)
      do M=1,3
         NC(M)=quick_basis%KLMN(M,KKK)
         ND(M)=quick_basis%KLMN(M,LLL)
      enddo

      MCD=trans(NC(1)+ND(1),NC(2)+ND(2),NC(3)+ND(3))
      MCX=trans(NC(1),NC(2),NC(3))
      do itemp=1,3
         if(ND(itemp).ne.0)then
            Y=store(1,MCD)+(RC(itemp)-RD(itemp))*store(1,MCX)
            goto 111
         endif
      enddo

   case(1001)
      do M=1,3
         NA(M)=quick_basis%KLMN(M,III)
         ND(M)=quick_basis%KLMN(M,LLL)
      enddo

      MA=trans(NA(1),NA(2),NA(3))
      MD=trans(ND(1),ND(2),ND(3))
      do itemp=1,3
         if(ND(itemp).ne.0)then
            Y=store(MA,MD)+(RC(itemp)-RD(itemp))*store(MA,1)
            goto 111
         endif
      enddo

   case(1011)
      do M=1,3
         NA(M)=quick_basis%KLMN(M,III)
         NC(M)=quick_basis%KLMN(M,KKK)
         ND(M)=quick_basis%KLMN(M,LLL)
      enddo
      MA=trans(NA(1),NA(2),NA(3))
      MCX=trans(NC(1),NC(2),NC(3))
      MCD=trans(NC(1)+ND(1),NC(2)+ND(2),NC(3)+ND(3))
      do itemp=1,3
         if(ND(itemp).ne.0)then
            Y=store(MA,MCD)+(RC(itemp)-RD(itemp))*store(MA,MCX)
            goto 111
         endif
      enddo

   case(999)

      call lefthrr(RA,RB,quick_basis%KLMN(1:3,III),quick_basis%KLMN(1:3,JJJ),IJtype,coefangxiaoL,angxiaoL,numangularL)
      call lefthrr(RC,RD,quick_basis%KLMN(1:3,KKK),quick_basis%KLMN(1:3,LLL),KLtype,coefangxiaoR,angxiaoR,numangularR)

      Y=0.0d0
      do i=1,numangularL
         do jxiao=1,numangularR
            Y=Y+coefangxiaoL(i)*coefangxiaoR(jxiao)* &
                  store(angxiaoL(i),angxiaoR(jxiao))

!if (III.eq.7.and.JJJ.eq.82.and.KKK.eq.18.and.LLL.eq.91) then
!        write(*,*) "hrr=",Y,coefangxiaoL(i),coefangxiaoR(jxiao), &
!                  store(angxiaoL(i),angxiaoR(jxiao)), &
!                angxiaoL(i),angxiaoR(jxiao)
!endif

         enddo
      enddo
      Y=Y*quick_basis%cons(III)*quick_basis%cons(JJJ)*quick_basis%cons(KKK)*quick_basis%cons(LLL)

   end select
   111 continue
!  write(*,*) IJKLtype,mpirank, iii,jjj,kkk,lll,Y
end subroutine hrrwhole


subroutine lefthrr(RA,RB,KLMNA,KLMNB,IKnumber,coefangxiao,angxiao,numangular)

   use allmod

   Implicit double precision(A-H,O-Z)
   INTEGER KLMNA(3),KLMNB(3),NA(3),NB(3)
   double precision RA(3),RB(3)
   Integer M1,M2,M3,M4

   double precision coefangxiao(20)
   integer angxiao(20),numangular,IKnumber

   select case (IKnumber)

   case (0)
      numangular=1
      coefangxiao(1)=1.0d0
      angxiao(1)=1

   case (1)
      do M=1,3
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NB(1),NB(2),NB(3))

      do itemp=1,3
         if(NB(itemp).ne.0)then
            Y=(RA(itemp)-RB(itemp))
            goto 111
         endif
      enddo

      111  numangular=2
      coefangxiao(1)=1.0d0
      angxiao(1)=M1
      coefangxiao(2)=Y
      angxiao(2)=1


   case (10)
      do M=1,3
         NA(M)=KLMNA(M)
      enddo

      M1=trans(NA(1),NA(2),NA(3))

      numangular=1
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

   case (11)
      do M=1,3
         NA(M)=KLMNA(M)
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      M2=trans(NA(1),NA(2),NA(3))

      do itemp=1,3
         if(NB(itemp).ne.0)then
            Y=(RA(itemp)-RB(itemp))
            goto 222
         endif
      enddo

      222  numangular=2
      coefangxiao(1)=1.0d0
      angxiao(1)=M1
      coefangxiao(2)=Y
      angxiao(2)=M2

   case (20,30,40)
      do M=1,3
         NA(M)=KLMNA(M)
      enddo

      M1=trans(NA(1),NA(2),NA(3))

      numangular=1
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

   case (2)
      do M=1,3
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NB(1),NB(2),NB(3))
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

      do itemp=1,3
         if(NB(itemp).eq.2)then
            numangular=3
            Y=(RA(itemp)-RB(itemp))
            coefangxiao(2)=2.0d0*Y
            angxiao(2)=itemp+1
            coefangxiao(3)=Y*Y
            angxiao(3)=1
            goto 333
         endif
         do jtemp=itemp+1,3
            if(NB(itemp).eq.1.and.NB(jtemp).eq.1)then
               numangular=4
               coefangxiao(2)=(RA(itemp)-RB(itemp))
               angxiao(2)=jtemp+1
               coefangxiao(3)=(RA(jtemp)-RB(jtemp))
               angxiao(3)=itemp+1
               coefangxiao(4)=coefangxiao(2)*coefangxiao(3)
               angxiao(4)=1
               goto 333
            endif
         enddo
      enddo

      333  continue

   case (21,31,41)
      do M=1,3
         NA(M)=KLMNA(M)
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      M2=trans(NA(1),NA(2),NA(3))
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

      do itemp=1,3
         if(NB(itemp).eq.1)then
            numangular=2
            coefangxiao(2)=(RA(itemp)-RB(itemp))
            angxiao(2)=M2
            goto 444
         endif
      enddo

      444  continue


   case (12)
      do M=1,3
         NA(M)=KLMNA(M)
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      M2=trans(NA(1),NA(2),NA(3))
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

      do itemp=1,3
         if(NB(itemp).eq.2)then
            numangular=3
            Y=(RA(itemp)-RB(itemp))
            coefangxiao(2)=2.0d0*Y
            NA(itemp)=NA(itemp)+1
            angxiao(2)=trans(NA(1),NA(2),NA(3))
            coefangxiao(3)=Y*Y
            angxiao(3)=M2
            goto 555
         endif
         do jtemp=itemp+1,3
            if(NB(itemp).eq.1.and.NB(jtemp).eq.1)then
               numangular=4
               coefangxiao(2)=(RA(itemp)-RB(itemp))
               NA(jtemp)=NA(jtemp)+1
               angxiao(2)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(3)=(RA(jtemp)-RB(jtemp))
               NA(itemp)=NA(itemp)+1
               angxiao(3)=trans(NA(1),NA(2),NA(3))
               coefangxiao(4)=coefangxiao(2)*coefangxiao(3)
               angxiao(4)=M2
               goto 555
            endif
         enddo
      enddo

      555  continue


   case (22,32,42)
      do M=1,3
         NA(M)=KLMNA(M)
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      M2=trans(NA(1),NA(2),NA(3))
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

      do itemp=1,3
         if(NB(itemp).eq.2)then
            numangular=3
            Y=(RA(itemp)-RB(itemp))
            coefangxiao(2)=2.0d0*Y
            NA(itemp)=NA(itemp)+1
            angxiao(2)=trans(NA(1),NA(2),NA(3))
            coefangxiao(3)=Y*Y
            angxiao(3)=M2
            goto 666
         endif
         do jtemp=itemp+1,3
            if(NB(itemp).eq.1.and.NB(jtemp).eq.1)then
               numangular=4
               coefangxiao(2)=(RA(itemp)-RB(itemp))
               NA(jtemp)=NA(jtemp)+1
               angxiao(2)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(3)=(RA(jtemp)-RB(jtemp))
               NA(itemp)=NA(itemp)+1
               angxiao(3)=trans(NA(1),NA(2),NA(3))
               coefangxiao(4)=coefangxiao(2)*coefangxiao(3)
               angxiao(4)=M2
               goto 666
            endif
         enddo
      enddo

      666  continue

   case(3,13,23,33,43)
      do M=1,3
         NA(M)=KLMNA(M)
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      M2=trans(NA(1),NA(2),NA(3))
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

      do itemp=1,3
         if(NB(itemp).eq.3)then
            numangular=4
            Y=(RA(itemp)-RB(itemp))

            coefangxiao(2)=3.0d0*Y
            NA(itemp)=NA(itemp)+2
            angxiao(2)=trans(NA(1),NA(2),NA(3))
            NA(itemp)=NA(itemp)-2

            coefangxiao(3)=3.0d0*Y*Y
            NA(itemp)=NA(itemp)+1
            angxiao(3)=trans(NA(1),NA(2),NA(3))
            NA(itemp)=NA(itemp)-1

            coefangxiao(4)=Y*Y*Y
            angxiao(4)=M2
            goto 777
         endif

         do jtemp=1,3
            if(NB(itemp).eq.1.and.NB(jtemp).eq.2)then
               numangular=6

               Yxiaotemp1=(RA(itemp)-RB(itemp))
               coefangxiao(2)=Yxiaotemp1
               NA(jtemp)=NA(jtemp)+2
               angxiao(2)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-2

               Yxiaotemp2=(RA(jtemp)-RB(jtemp))
               coefangxiao(3)=2.0d0*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               NA(jtemp)=NA(jtemp)+1
               angxiao(3)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(4)=2.0d0*Yxiaotemp1*Yxiaotemp2
               NA(jtemp)=NA(jtemp)+1
               angxiao(4)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(5)=Yxiaotemp2*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               angxiao(5)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1

               coefangxiao(6)=Yxiaotemp1*Yxiaotemp2*Yxiaotemp2
               angxiao(6)=M2

               goto 777
            endif

         enddo
      enddo

      if(NB(1).eq.1.and.NB(2).eq.1)then
         numangular=8

         Yxiaotemp1=(RA(1)-RB(1))
         Yxiaotemp2=(RA(2)-RB(2))
         Yxiaotemp3=(RA(3)-RB(3))

         coefangxiao(2)=Yxiaotemp1
         NA(2)=NA(2)+1
         NA(3)=NA(3)+1
         angxiao(2)=trans(NA(1),NA(2),NA(3))
         NA(2)=NA(2)-1
         NA(3)=NA(3)-1

         coefangxiao(3)=Yxiaotemp2
         NA(1)=NA(1)+1
         NA(3)=NA(3)+1
         angxiao(3)=trans(NA(1),NA(2),NA(3))
         NA(1)=NA(1)-1
         NA(3)=NA(3)-1

         coefangxiao(4)=Yxiaotemp3
         NA(1)=NA(1)+1
         NA(2)=NA(2)+1
         angxiao(4)=trans(NA(1),NA(2),NA(3))
         NA(1)=NA(1)-1
         NA(2)=NA(2)-1

         coefangxiao(5)=Yxiaotemp1*Yxiaotemp2
         NA(3)=NA(3)+1
         angxiao(5)=trans(NA(1),NA(2),NA(3))
         NA(3)=NA(3)-1

         coefangxiao(6)=Yxiaotemp1*Yxiaotemp3
         NA(2)=NA(2)+1
         angxiao(6)=trans(NA(1),NA(2),NA(3))
         NA(2)=NA(2)-1

         coefangxiao(7)=Yxiaotemp2*Yxiaotemp3
         NA(1)=NA(1)+1
         angxiao(7)=trans(NA(1),NA(2),NA(3))
         NA(1)=NA(1)-1

         coefangxiao(8)=Yxiaotemp1*Yxiaotemp2*Yxiaotemp3
         angxiao(8)=M2

         goto 777
      endif

      777  continue

   case(4,14,24,34,44)
      do M=1,3
         NA(M)=KLMNA(M)
         NB(M)=KLMNB(M)
      enddo

      M1=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
      M2=trans(NA(1),NA(2),NA(3))
      coefangxiao(1)=1.0d0
      angxiao(1)=M1

      do itemp=1,3
         if(NB(itemp).eq.4)then
            numangular=5
            Y=(RA(itemp)-RB(itemp))

            coefangxiao(2)=4.0d0*Y
            NA(itemp)=NA(itemp)+3
            angxiao(2)=trans(NA(1),NA(2),NA(3))
            NA(itemp)=NA(itemp)-3

            coefangxiao(3)=6.0d0*Y*Y
            NA(itemp)=NA(itemp)+2
            angxiao(3)=trans(NA(1),NA(2),NA(3))
            NA(itemp)=NA(itemp)-2

            coefangxiao(4)=4.0d0*Y*Y*Y
            NA(itemp)=NA(itemp)+1
            angxiao(4)=trans(NA(1),NA(2),NA(3))
            NA(itemp)=NA(itemp)-1

            coefangxiao(5)=Y*Y*Y*Y
            angxiao(5)=M2

            goto 888
         endif

         do jtemp=1,3
            if(NB(itemp).eq.1.and.NB(jtemp).eq.3)then
               numangular=8

               Yxiaotemp1=(RA(itemp)-RB(itemp))
               coefangxiao(2)=Yxiaotemp1
               NA(jtemp)=NA(jtemp)+3
               angxiao(2)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-3

               Yxiaotemp2=(RA(jtemp)-RB(jtemp))
               coefangxiao(3)=3.0d0*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               NA(jtemp)=NA(jtemp)+2
               angxiao(3)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1
               NA(jtemp)=NA(jtemp)-2

               coefangxiao(4)=3.0d0*Yxiaotemp1*Yxiaotemp2
               NA(jtemp)=NA(jtemp)+2
               angxiao(4)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-2

               coefangxiao(5)=3.0d0*Yxiaotemp2*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               NA(jtemp)=NA(jtemp)+1
               angxiao(5)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(6)=3.0d0*Yxiaotemp1*Yxiaotemp2*Yxiaotemp2
               NA(jtemp)=NA(jtemp)+1
               angxiao(6)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(7)=Yxiaotemp2*Yxiaotemp2*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               angxiao(7)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1

               coefangxiao(8)=Yxiaotemp1*Yxiaotemp2*Yxiaotemp2*Yxiaotemp2
               angxiao(8)=M2

               goto 888
            endif

            if(NB(itemp).eq.2.and.NB(jtemp).eq.2.and.itemp.ne.jtemp)then
               numangular=9

               Yxiaotemp1=(RA(itemp)-RB(itemp))
               Yxiaotemp2=(RA(jtemp)-RB(jtemp))

               coefangxiao(2)=2.0d0*Yxiaotemp1
               NA(itemp)=NA(itemp)+1
               NA(jtemp)=NA(jtemp)+2
               angxiao(2)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1
               NA(jtemp)=NA(jtemp)-2

               coefangxiao(3)=2.0d0*Yxiaotemp2
               NA(itemp)=NA(itemp)+2
               NA(jtemp)=NA(jtemp)+1
               angxiao(3)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-2
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(4)=4.0d0*Yxiaotemp1*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               NA(jtemp)=NA(jtemp)+1
               angxiao(4)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(5)=Yxiaotemp1*Yxiaotemp1
               NA(jtemp)=NA(jtemp)+2
               angxiao(5)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-2

               coefangxiao(6)=Yxiaotemp2*Yxiaotemp2
               NA(itemp)=NA(itemp)+2
               angxiao(6)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-2

               coefangxiao(7)=2.0d0*Yxiaotemp1*Yxiaotemp2*Yxiaotemp2
               NA(itemp)=NA(itemp)+1
               angxiao(7)=trans(NA(1),NA(2),NA(3))
               NA(itemp)=NA(itemp)-1

               coefangxiao(8)=2.0d0*Yxiaotemp1*Yxiaotemp1*Yxiaotemp2
               NA(jtemp)=NA(jtemp)+1
               angxiao(8)=trans(NA(1),NA(2),NA(3))
               NA(jtemp)=NA(jtemp)-1

               coefangxiao(9)=Yxiaotemp1*Yxiaotemp1*Yxiaotemp2*Yxiaotemp2
               angxiao(9)=M2

               goto 888
            endif

            do ktemp=1,3
               if(NB(itemp).eq.1.and.NB(jtemp).eq.1.and.NB(ktemp).eq.2.and.itemp.ne.jtemp)then
                  numangular=12

                  Yxiaotemp1=(RA(itemp)-RB(itemp))
                  Yxiaotemp2=(RA(jtemp)-RB(jtemp))
                  Yxiaotemp3=(RA(ktemp)-RB(ktemp))

                  coefangxiao(2)=Yxiaotemp1
                  NA(jtemp)=NA(jtemp)+1
                  NA(ktemp)=NA(ktemp)+2
                  angxiao(2)=trans(NA(1),NA(2),NA(3))
                  NA(jtemp)=NA(jtemp)-1
                  NA(ktemp)=NA(ktemp)-2

                  coefangxiao(3)=Yxiaotemp2
                  NA(itemp)=NA(itemp)+1
                  NA(ktemp)=NA(ktemp)+2
                  angxiao(3)=trans(NA(1),NA(2),NA(3))
                  NA(itemp)=NA(itemp)-1
                  NA(ktemp)=NA(ktemp)-2

                  coefangxiao(4)=2.0d0*Yxiaotemp3
                  NA(itemp)=NA(itemp)+1
                  NA(jtemp)=NA(jtemp)+1
                  NA(ktemp)=NA(ktemp)+1
                  angxiao(4)=trans(NA(1),NA(2),NA(3))
                  NA(itemp)=NA(itemp)-1
                  NA(jtemp)=NA(jtemp)-1
                  NA(ktemp)=NA(ktemp)-1

                  coefangxiao(5)=Yxiaotemp1*Yxiaotemp2
                  NA(ktemp)=NA(ktemp)+2
                  angxiao(5)=trans(NA(1),NA(2),NA(3))
                  NA(ktemp)=NA(ktemp)-2

                  coefangxiao(6)=2.0d0*Yxiaotemp1*Yxiaotemp3
                  NA(jtemp)=NA(jtemp)+1
                  NA(ktemp)=NA(ktemp)+1
                  angxiao(6)=trans(NA(1),NA(2),NA(3))
                  NA(jtemp)=NA(jtemp)-1
                  NA(ktemp)=NA(ktemp)-1

                  coefangxiao(7)=2.0d0*Yxiaotemp2*Yxiaotemp3
                  NA(itemp)=NA(itemp)+1
                  NA(ktemp)=NA(ktemp)+1
                  angxiao(7)=trans(NA(1),NA(2),NA(3))
                  NA(itemp)=NA(itemp)-1
                  NA(ktemp)=NA(ktemp)-1

                  coefangxiao(8)=Yxiaotemp3*Yxiaotemp3
                  NA(itemp)=NA(itemp)+1
                  NA(jtemp)=NA(jtemp)+1
                  angxiao(8)=trans(NA(1),NA(2),NA(3))
                  NA(itemp)=NA(itemp)-1
                  NA(jtemp)=NA(jtemp)-1

                  coefangxiao(9)=2.0d0*Yxiaotemp1*Yxiaotemp2*Yxiaotemp3
                  NA(ktemp)=NA(ktemp)+1
                  angxiao(9)=trans(NA(1),NA(2),NA(3))
                  NA(ktemp)=NA(ktemp)-1

                  coefangxiao(10)=Yxiaotemp1*Yxiaotemp3*Yxiaotemp3
                  NA(jtemp)=NA(jtemp)+1
                  angxiao(10)=trans(NA(1),NA(2),NA(3))
                  NA(jtemp)=NA(jtemp)-1

                  coefangxiao(11)=Yxiaotemp2*Yxiaotemp3*Yxiaotemp3
                  NA(itemp)=NA(itemp)+1
                  angxiao(11)=trans(NA(1),NA(2),NA(3))
                  NA(itemp)=NA(itemp)-1

                  coefangxiao(12)=Yxiaotemp1*Yxiaotemp2*Yxiaotemp3*Yxiaotemp3
                  angxiao(12)=M2

                  goto 888
               endif

            enddo
         enddo
      enddo

      888  continue

   end select

End subroutine lefthrr

! PAY ATTENTION TO THE INDICE OF IJTYPE AND KLTYPE
! Xiao HE 07/07/07 version
! subroutine hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
!Horrizontal Recursion subroutines by hand, these parts can be optimized by MAPLE
subroutine hrrwholeopt
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision RA(3),RB(3),RC(3),RD(3)
   Integer M1,M2,M3,M4

   double precision storeaa(120,120)
   double precision storebb(120,120)
   double precision storecc(120,120)
   double precision storedd(120,120)

   double precision coefangxiaoL(20),coefangxiaoR(20)
   integer angxiaoL(20),angxiaoR(20),numangularL,numangularR

   double precision coefangxiaoLnew(20),coefangxiaoRnew(20)
   integer angxiaoLnew(20),angxiaoRnew(20),numangularLnew,numangularRnew

   COMMON /COM1/RA,RB,RC,RD

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /xiaostoreopt/storeaa,storebb,storecc,storedd
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   tempconstant=quick_basis%cons(III)*quick_basis%cons(JJJ)*quick_basis%cons(KKK)*quick_basis%cons(LLL)

   call lefthrr(RA,RB,quick_basis%KLMN(1:3,III),quick_basis%KLMN(1:3,JJJ),IJtype,coefangxiaoL,angxiaoL,numangularL)
   call lefthrr(RC,RD,quick_basis%KLMN(1:3,KKK),quick_basis%KLMN(1:3,LLL),KLtype,coefangxiaoR,angxiaoR,numangularR)


   do itemp=1,3

      do itempxiao=1,3
         NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
      enddo

      NA(itemp)=quick_basis%KLMN(itemp,III)+1

      call lefthrr(RA,RB,NA(1:3),quick_basis%KLMN(1:3,JJJ),IJtype+10,coefangxiaoLnew,angxiaoLnew,numangularLnew)
      Yaa(itemp)=0.0d0
      do i=1,numangularLnew
         do jxiao=1,numangularR
            Yaa(itemp)=Yaa(itemp)+coefangxiaoLnew(i)*coefangxiaoR(jxiao)* &
                  storeAA(angxiaoLnew(i),angxiaoR(jxiao))


         enddo
      enddo


      if(quick_basis%KLMN(itemp,III).ge.1)then

         do itempxiao=1,3
            NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
         enddo

         NA(itemp)=quick_basis%KLMN(itemp,III)-1

         call lefthrr(RA,RB,NA(1:3),quick_basis%KLMN(1:3,JJJ),IJtype-10,coefangxiaoLnew,angxiaoLnew,numangularLnew)
         do i=1,numangularLnew
            do jxiao=1,numangularR
               Yaa(itemp)=Yaa(itemp)-quick_basis%KLMN(itemp,III)*coefangxiaoLnew(i)* &

                     coefangxiaoR(jxiao)*store(angxiaoLnew(i),angxiaoR(jxiao))
            enddo
         enddo
      endif

      Yaa(itemp)=Yaa(itemp)*tempconstant
   enddo

   do itemp=1,3
      do itempxiao=1,3
         NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
         NB(itempxiao)=quick_basis%KLMN(itempxiao,JJJ)
      enddo
      NB(itemp)=quick_basis%KLMN(itemp,JJJ)+1

      call lefthrr(RA,RB,NA(1:3),NB(1:3),IJtype+1,coefangxiaoLnew,angxiaoLnew,numangularLnew)

      Ybb(itemp)=0.0d0
      do i=1,numangularLnew
         do jxiao=1,numangularR
            Ybb(itemp)=Ybb(itemp)+coefangxiaoLnew(i)*coefangxiaoR(jxiao)* &
                  storeBB(angxiaoLnew(i),angxiaoR(jxiao))
         enddo
      enddo

      if(quick_basis%KLMN(itemp,JJJ).ge.1)then

         do itempxiao=1,3
            NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
            NB(itempxiao)=quick_basis%KLMN(itempxiao,JJJ)
         enddo

         NB(itemp)=quick_basis%KLMN(itemp,JJJ)-1

         call lefthrr(RA,RB,NA(1:3),NB(1:3),IJtype-1,coefangxiaoLnew,angxiaoLnew,numangularLnew)

         do i=1,numangularLnew
            do jxiao=1,numangularR
               Ybb(itemp)=Ybb(itemp)-quick_basis%KLMN(itemp,JJJ)*coefangxiaoLnew(i)* &
                     coefangxiaoR(jxiao)*store(angxiaoLnew(i),angxiaoR(jxiao))
            enddo
         enddo
      endif

      Ybb(itemp)=Ybb(itemp)*tempconstant
   enddo

   do itemp=1,3

      do itempxiao=1,3
         NC(itempxiao)=quick_basis%KLMN(itempxiao,KKK)
      enddo

      NC(itemp)=quick_basis%KLMN(itemp,KKK)+1

      call lefthrr(RC,RD,NC(1:3),quick_basis%KLMN(1:3,LLL),KLtype+10,coefangxiaoRnew,angxiaoRnew,numangularRnew)

      Ycc(itemp)=0.0d0
      do i=1,numangularL
         do jxiao=1,numangularRnew
            Ycc(itemp)=Ycc(itemp)+coefangxiaoL(i)*coefangxiaoRnew(jxiao)* &
                  storeCC(angxiaoL(i),angxiaoRnew(jxiao))
         enddo
      enddo
      if(quick_basis%KLMN(itemp,KKK).ge.1)then

         do itempxiao=1,3
            NC(itempxiao)=quick_basis%KLMN(itempxiao,KKK)
         enddo

         NC(itemp)=quick_basis%KLMN(itemp,KKK)-1
         call lefthrr(RC,RD,NC(1:3),quick_basis%KLMN(1:3,LLL),KLtype-10,coefangxiaoRnew,angxiaoRnew,numangularRnew)

         do i=1,numangularL
            do jxiao=1,numangularRnew
               Ycc(itemp)=Ycc(itemp)-quick_basis%KLMN(itemp,KKK)*coefangxiaoL(i)* &
                     coefangxiaoRnew(jxiao)*store(angxiaoL(i),angxiaoRnew(jxiao))
            enddo
         enddo
      endif

      Ycc(itemp)=Ycc(itemp)*tempconstant
   enddo

100 continue

end subroutine hrrwholeopt



subroutine vertical(NABCDTYPE)
   implicit none
   integer i, NABCDTYPE

   select case (NABCDTYPE)
   case(0)
   case(10)
      call PSSS(0)
   case(1)
      call SSPS(0)
   case(11)
      call PSSS(0)

      call SSPS(0)
      call SSPS(1)
      call PSPS(0)
   case(20)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)
   case(2)
      call SSPS(0)
      call SSPS(1)
      call SSDS(0)
   case(21)
      call SSPS(0)

      call PSSS(0)
      call SSPS(1)
      call PSPS(0)

      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)
      call DSPS(0)
   case(12)
      call SSPS(0)

      call PSSS(0)
      call SSPS(1)
      call PSPS(0)

      call SSDS(0)

      call SSPS(2)
      call SSDS(1)
      call PSDS(0)
   case(22)
      call SSPS(0)

      call PSSS(0)
      call SSPS(1)
      call PSPS(0)

      call SSDS(0)
      call SSPS(2)
      call SSDS(1)
      call PSDS(0)

      call PSSS(1)
      call DSSS(0)
      call PSSS(2)
      call DSSS(1)
      call DSPS(0)

      call SSPS(3)
      call SSDS(2)
      call PSDS(1)

      call PSPS(1)
      call DSDS(0)

   case(30)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)

      call FSSS(0)

   case(3)
      call SSPS(0)
      call SSPS(1)
      call SSDS(0)

      call SSPS(2)
      call SSDS(1)

      call SSFS(0)

   case(40)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)

      call FSSS(0)

      call PSSS(3)
      call DSSS(2)

      call FSSS(1)

      call GSSS(0)

   case(4)
      call SSPS(0)
      call SSPS(1)
      call SSDS(0)

      call SSPS(2)
      call SSDS(1)

      call SSFS(0)

      call SSPS(3)
      call SSDS(2)

      call SSFS(1)

      call SSGS(0)

   case(31)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)

      call FSSS(0)

      call PSSS(3)
      call DSSS(2)

      call FSSS(1)

      call SSPS(0)
      call SSPS(1)
      call PSPS(0)

      call DSPS(0)

      call FSPS(0)

   case(13)
      call SSPS(0)
      call SSPS(1)
      call SSDS(0)

      call SSPS(2)
      call SSDS(1)

      call SSFS(0)

      call SSPS(3)
      call SSDS(2)

      call SSFS(1)

      call PSSS(0)
      call PSPS(0)
      call PSDS(0)
      call PSFS(0)

   case(41)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)

      call FSSS(0)

      call PSSS(3)
      call DSSS(2)

      call FSSS(1)

      call GSSS(0)

      call PSSS(4)
      call DSSS(3)

      call FSSS(2)

      call GSSS(1)

      call SSPS(0)
      call SSPS(1)
      call PSPS(0)

      call DSPS(0)

      call FSPS(0)

      call GSPS(0)

   case(14)
      call SSPS(0)
      call SSPS(1)
      call SSDS(0)

      call SSPS(2)
      call SSDS(1)

      call SSFS(0)

      call SSPS(3)
      call SSDS(2)

      call SSFS(1)

      call PSSS(0)
      call PSSS(1)
      call PSPS(0)

      call SSGS(0)

      call SSPS(4)
      call SSDS(3)

      call SSFS(2)

      call SSGS(1)

      call PSDS(0)

      call PSFS(0)

      call PSGS(0)

   case(32)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)

      call FSSS(0)

      call PSSS(3)
      call DSSS(2)

      call FSSS(1)

      call SSPS(0)
      call SSPS(1)
      call PSPS(0)

      call DSPS(0)

      call FSPS(0)

      call PSSS(4)
      call DSSS(3)

      call FSSS(2)

      call FSPS(1)

      call DSPS(1)

      call SSDS(0)

      call SSPS(2)
      call SSDS(1)
      call PSDS(0)

      call SSPS(3)
      call SSDS(2)
      call PSDS(1)

      call PSPS(1)
      call DSDS(0)

      call FSDS(0)

   case(23)
      call SSPS(0)
      call SSPS(1)
      call SSDS(0)

      call SSPS(2)
      call SSDS(1)

      call SSFS(0)

      call SSPS(3)
      call SSDS(2)

      call SSFS(1)

      call PSSS(0)
      call PSSS(1)
      call PSPS(0)

      call PSDS(0)

      call PSFS(0)

      call SSPS(4)
      call SSDS(3)

      call SSFS(2)

      call PSFS(1)

      call PSDS(1)

      call DSSS(0)

      call PSSS(2)
      call DSSS(1)
      call DSPS(0)
      call PSPS(1)
      call DSDS(0)

      call DSFS(0)

   case(42)
      call PSSS(0)
      call PSSS(1)
      call DSSS(0)

      call PSSS(2)
      call DSSS(1)

      call FSSS(0)

      call PSSS(3)
      call DSSS(2)

      call FSSS(1)

      call GSSS(0)

      call PSSS(4)
      call DSSS(3)

      call FSSS(2)

      call GSSS(1)

      call DSPS(0)

      call FSPS(0)

      call GSPS(0)

      call PSSS(5)
      call DSSS(4)
      call FSSS(3)

      call GSSS(2)
      call DSPS(1)

      call FSPS(1)

      call GSPS(1)


      call SSPS(0)
      call SSPS(1)
      call PSPS(0)

      call SSDS(0)
      call SSPS(2)
      call SSDS(1)
      call PSDS(0)

      call SSPS(3)
      call SSDS(2)
      call PSDS(1)

      call PSPS(1)
      call DSDS(0)

      call FSDS(0)

      call GSDS(0)


   case(24)

      call SSPS(0)
      call SSPS(1)
      call SSDS(0)

      call SSPS(2)
      call SSDS(1)

      call SSFS(0)

      call SSPS(3)
      call SSDS(2)

      call SSFS(1)

      call SSGS(0)

      call SSPS(4)
      call SSDS(3)

      call SSFS(2)

      call SSGS(1)

      call PSDS(0)

      call PSFS(0)

      call PSGS(0)

      call SSPS(5)
      call SSDS(4)

      call SSFS(3)

      call SSGS(2)
      call PSDS(1)
      call PSFS(1)

      call PSGS(1)

      call PSSS(0)

      call PSSS(1)
      call PSPS(0)

      call DSSS(0)
      call PSSS(2)
      call DSSS(1)
      call DSPS(0)

      call PSSS(3)
      call DSSS(2)
      call DSPS(1)

      call PSPS(1)
      call DSDS(0)

      call DSFS(0)
      call DSGS(0)


   case(33)
      do i=0,5
         call PSSS(i)
      enddo
      do i=0,4
         call SSPS(i)
      enddo
      do i=0,3
         call PSPS(i)
      enddo
      do i=0,4
         call DSSS(i)
      enddo
      do i=0,3
         call SSDS(i)
      enddo
      do i=0,3
         call DSPS(i)
      enddo
      do i=0,2
         call PSDS(i)
      enddo
      do i=0,1
         call DSDS(i)
      enddo
      do i=0,3
         call FSSS(i)
      enddo
      do i=0,2
         call SSFS(i)
      enddo
      do i=0,2
         call FSPS(i)
      enddo
      do i=0,1
         call PSFS(i)
      enddo
      call FSDS(0)
      call DSFS(0)
      call FSDS(1)
      call FSFS(0)

   case(43)
      do i=0,6
         call PSSS(i)
      enddo
      do i=0,5
         call SSPS(i)
      enddo
      do i=0,4
         call PSPS(i)
      enddo
      do i=0,5
         call DSSS(i)
      enddo
      do i=0,4
         call SSDS(i)
      enddo
      do i=0,4
         call DSPS(i)
      enddo
      do i=0,3
         call PSDS(i)
      enddo
      do i=0,2
         call DSDS(i)
      enddo
      do i=0,4
         call FSSS(i)
      enddo
      do i=0,3
         call SSFS(i)
      enddo
      do i=0,3
         call FSPS(i)
      enddo
      do i=0,2
         call PSFS(i)
      enddo
      call FSDS(0)
      call DSFS(0)
      call FSDS(1)
      call DSFS(1)
      call FSDS(2)
      call FSFS(0)
      do i=0,3
         call GSSS(i)
      enddo
      do i=0,2
         call GSPS(i)
      enddo
      do i=0,1
         call GSDS(i)
      enddo
      call GSFS(0)

   case(34)
      do i=0,6
         call SSPS(i)
      enddo
      do i=0,5
         call PSSS(i)
      enddo
      do i=0,4
         call PSPS(i)
      enddo
      do i=0,5
         call SSDS(i)
      enddo
      do i=0,4
         call DSSS(i)
      enddo
      do i=0,4
         call PSDS(i)
      enddo
      do i=0,3
         call DSPS(i)
      enddo
      do i=0,2
         call DSDS(i)
      enddo
      do i=0,4
         call SSFS(i)
      enddo
      do i=0,3
         call FSSS(i)
      enddo
      do i=0,3
         call PSFS(i)
      enddo
      do i=0,2
         call FSPS(i)
      enddo
      call FSDS(0)
      call DSFS(0)
      call FSDS(1)
      call DSFS(1)
      call DSFS(2)
      call FSFS(0)
      do i=0,3
         call SSGS(i)
      enddo
      do i=0,2
         call PSGS(i)
      enddo
      do i=0,1
         call DSGS(i)
      enddo
      call FSGS(0)

   case(44)
      do i=0,7
         call PSSS(i)
      enddo
      do i=0,6
         call SSPS(i)
      enddo
      do i=0,5
         call PSPS(i)
      enddo
      do i=0,6
         call DSSS(i)
      enddo
      do i=0,5
         call SSDS(i)
      enddo
      do i=0,5
         call DSPS(i)
      enddo
      do i=0,4
         call PSDS(i)
      enddo
      do i=0,3
         call DSDS(i)
      enddo
      do i=0,5
         call FSSS(i)
      enddo
      do i=0,4
         call SSFS(i)
      enddo
      do i=0,4
         call FSPS(i)
      enddo
      do i=0,3
         call PSFS(i)
      enddo
      do i=0,3
         call FSDS(i)
      enddo
      do i=0,2
         call DSFS(i)
      enddo
      do i=0,1
         call FSFS(i)
      enddo
      do i=0,4
         call GSSS(i)
      enddo
      do i=0,3
         call SSGS(i)
      enddo
      do i=0,3
         call GSPS(i)
      enddo
      do i=0,2
         call PSGS(i)
      enddo
      do i=0,2
         call GSDS(i)
      enddo
      do i=0,1
         call DSGS(i)
      enddo
      do i=0,1
         call GSFS(i)
      enddo
      call FSGS(0)
      call GSGS(0)

   case(50)
      do i=0,4
         call PSSS(i)
      enddo
      do i=0,3
         call DSSS(i)
      enddo
      do i=0,2
         call FSSS(i)
      enddo
      do i=0,1
         call GSSS(i)
      enddo
      call BSLS(5,0,0)

   case(5)
      do i=0,4
         call SSPS(i)
      enddo
      do i=0,3
         call SSDS(i)
      enddo
      do i=0,2
         call SSFS(i)
      enddo
      do i=0,1
         call SSGS(i)
      enddo
      call LSBS(0,5,0)

   case(51)
      do i=0,5
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,4
         call PSPS(i)
      enddo
      do i=0,4
         call DSSS(i)
      enddo
      do i=0,3
         call DSPS(i)
         call FSSS(i)
      enddo
      do i=0,2
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,1
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      call BSLS(5,1,0)

   case(15)
      do i=0,5
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,4
         call PSPS(i)
         call SSDS(i)
      enddo
      do i=0,3
         call PSDS(i)
         call SSFS(i)
      enddo
      do i=0,2
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,1
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      call LSBS(1,5,0)

   case(52)
      do i=0,6
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,5
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,4
         call PSDS(i)
         call DSPS(i)
         call FSSS(i)
      enddo
      do i=0,3
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,2
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,1
         call GSDS(i)
         call BSLS(5,1,i)
      enddo
      call BSLS(5,2,0)

   case(25)
      do i=0,6
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,5
         call PSPS(i)
         call SSDS(i)
         call DSSS(i)
      enddo
      do i=0,4
         call DSPS(i)
         call PSDS(i)
         call SSFS(i)
      enddo
      do i=0,3
         call DSDS(i)
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,2
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,1
         call DSGS(i)
         call LSBS(1,5,i)
      enddo
      call LSBS(2,5,0)

   case(53)
      do i=0,7
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,6
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,5
         call PSDS(i)
         call DSPS(i)
         call FSSS(i)
         call SSFS(i)
      enddo
      do i=0,4
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,3
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,2
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
      enddo
      do i=0,1
         call GSFS(i)
         call BSLS(5,2,i)
      enddo
      call BSLS(5,3,0)

   case(35)
      do i=0,7
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,6
         call PSPS(i)
         call SSDS(i)
         call DSSS(i)
      enddo
      do i=0,5
         call DSPS(i)
         call PSDS(i)
         call SSFS(i)
         call FSSS(i)
      enddo
      do i=0,4
         call FSPS(i)
         call DSDS(i)
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,3
         call FSDS(i)
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,2
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
      enddo
      do i=0,1
         call FSGS(i)
         call LSBS(2,5,i)
      enddo
      call LSBS(3,5,0)

   case(54)
      do i=0,8
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,7
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,6
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,5
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,4
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,3
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
      enddo
      do i=0,2
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
      enddo
      do i=0,1
         call GSGS(i)
         call BSLS(5,3,i)
      enddo
      call BSLS(5,4,0)

   case(45)
      do i=0,8
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,7
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,6
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,5
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,4
         call GSPS(i)
         call FSDS(i)
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,3
         call GSDS(i)
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
      enddo
      do i=0,2
         call GSFS(i)
         call FSGS(i)
         call LSBS(2,5,i)
      enddo
      do i=0,1
         call GSGS(i)
         call LSBS(3,5,i)
      enddo
      call LSBS(4,5,0)

   case(55)
      do i=0,9
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,8
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,7
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,6
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,5
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,4
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
      enddo
      do i=0,3
         call LSBS(2,5,i)
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
      enddo
      do i=0,2
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
      enddo
      do i=0,1
         call LSBS(4,5,i)
         call BSLS(5,4,i)
      enddo
      call BSLS(5,5,0)

   case(60)
      do i=0,5
         call PSSS(i)
      enddo
      do i=0,4
         call DSSS(i)
      enddo
      do i=0,3
         call FSSS(i)
      enddo
      do i=0,2
         call GSSS(i)
      enddo
      do i=0,1
         call BSLS(5,0,i)
      enddo
      call BSLS(6,0,0)

   case(6)
      do i=0,5
         call SSPS(i)
      enddo
      do i=0,4
         call SSDS(i)
      enddo
      do i=0,3
         call SSFS(i)
      enddo
      do i=0,2
         call SSGS(i)
      enddo
      do i=0,1
         call LSBS(0,5,i)
      enddo
      call LSBS(0,6,0)

   case(61)
      do i=0,6
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,5
         call PSPS(i)
         call DSSS(i)
      enddo
      do i=0,4
         call DSPS(i)
         call FSSS(i)
      enddo
      do i=0,3
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,2
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,1
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      call BSLS(6,1,0)

   case(16)
      do i=0,6
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,5
         call PSPS(i)
         call SSDS(i)
      enddo
      do i=0,4
         call PSDS(i)
         call SSFS(i)
      enddo
      do i=0,3
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,2
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,1
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      call LSBS(1,6,0)

   case(62)
      do i=0,7
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,6
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,5
         call PSDS(i)
         call DSPS(i)
         call FSSS(i)
      enddo
      do i=0,4
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,3
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,2
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,1
         call BSLS(5,2,i)
         call BSLS(6,1,i)
      enddo
      call BSLS(6,2,0)

   case(26)
      do i=0,7
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,6
         call PSPS(i)
         call SSDS(i)
         call DSSS(i)
      enddo
      do i=0,5
         call DSPS(i)
         call PSDS(i)
         call SSFS(i)
      enddo
      do i=0,4
         call DSDS(i)
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,3
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,2
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,1
         call LSBS(2,5,i)
         call LSBS(1,6,i)
      enddo
      call LSBS(2,6,0)

   case(63)
      do i=0,8
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,7
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,6
         call PSDS(i)
         call DSPS(i)
         call FSSS(i)
         call SSFS(i)
      enddo
      do i=0,5
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,4
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,3
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,2
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
      enddo
      do i=0,1
         call BSLS(5,3,i)
         call BSLS(6,2,i)
      enddo
      call BSLS(6,3,0)

   case(36)
      do i=0,8
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,7
         call PSPS(i)
         call SSDS(i)
         call DSSS(i)
      enddo
      do i=0,6
         call DSPS(i)
         call PSDS(i)
         call SSFS(i)
         call FSSS(i)
      enddo
      do i=0,5
         call FSPS(i)
         call DSDS(i)
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,4
         call FSDS(i)
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,3
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,2
         call FSGS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
      enddo
      do i=0,1
         call LSBS(3,5,i)
         call LSBS(2,6,i)
      enddo
      call LSBS(3,6,0)

   case(64)
      do i=0,9
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,8
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,7
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,6
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,5
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,4
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,3
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
      enddo
      do i=0,2
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
      enddo
      do i=0,1
         call BSLS(5,4,i)
         call BSLS(6,3,i)
      enddo
      call BSLS(6,4,0)

   case(46)
      do i=0,9
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,8
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,7
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,6
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,5
         call GSPS(i)
         call FSDS(i)
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,4
         call GSDS(i)
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,3
         call GSFS(i)
         call FSGS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
      enddo
      do i=0,2
         call GSGS(i)
         call LSBS(3,5,i)
         call LSBS(2,6,i)
      enddo
      do i=0,1
         call LSBS(4,5,i)
         call LSBS(3,6,i)
      enddo
      call LSBS(4,6,0)

   case(65)
      do i=0,10
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,9
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,8
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,7
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,6
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,5
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,4
         call LSBS(2,5,i)
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
      enddo
      do i=0,3
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
      enddo
      do i=0,2
         call LSBS(4,5,i)
         call BSLS(5,4,i)
         call BSLS(6,3,i)
      enddo
      do i=0,1
         call BSLS(5,5,i)
         call BSLS(6,4,i)
      enddo
      call BSLS(6,5,0)

   case(56)
      do i=0,10
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,9
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,8
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,7
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,6
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,5
         call BSLS(5,1,i)
         call GSDS(i)
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,4
         call BSLS(5,2,i)
         call FSGS(i)
         call GSFS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
      enddo
      do i=0,3
         call BSLS(5,3,i)
         call GSGS(i)
         call LSBS(3,5,i)
         call LSBS(2,6,i)
      enddo
      do i=0,2
         call BSLS(5,4,i)
         call LSBS(4,5,i)
         call LSBS(3,6,i)
      enddo
      do i=0,1
         call BSLS(5,5,i)
         call LSBS(4,6,i)
      enddo
      call LSBS(5,6,0)

   case(66)
      do i=0,11
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,10
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,9
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,8
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,7
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,6
         call LSBS(0,6,i)
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,5
         call LSBS(1,6,i)
         call LSBS(2,5,i)
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
      enddo
      do i=0,4
         call LSBS(2,6,i)
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
      enddo
      do i=0,3
         call LSBS(3,6,i)
         call LSBS(4,5,i)
         call BSLS(5,4,i)
         call BSLS(6,3,i)
      enddo
      do i=0,2
         call LSBS(4,6,i)
         call BSLS(5,5,i)
         call BSLS(6,4,i)
      enddo
      do i=0,1
         call LSBS(5,6,i)
         call BSLS(6,5,i)
      enddo
      call BSLS(6,6,0)

   case(70)
      do i=0,6
         call PSSS(i)
      enddo
      do i=0,5
         call DSSS(i)
      enddo
      do i=0,4
         call FSSS(i)
      enddo
      do i=0,3
         call GSSS(i)
      enddo
      do i=0,2
         call BSLS(5,0,i)
      enddo
      do i=0,1
         call BSLS(6,0,i)
      enddo
      call BSLS(7,0,0)

   case(7)
      do i=0,6
         call SSPS(i)
      enddo
      do i=0,5
         call SSDS(i)
      enddo
      do i=0,4
         call SSFS(i)
      enddo
      do i=0,3
         call SSGS(i)
      enddo
      do i=0,2
         call LSBS(0,5,i)
      enddo
      do i=0,1
         call LSBS(0,6,i)
      enddo
      call LSBS(0,7,0)

   case(71)
      do i=0,7
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,6
         call PSPS(i)
         call DSSS(i)
      enddo
      do i=0,5
         call DSPS(i)
         call FSSS(i)
      enddo
      do i=0,4
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,3
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,2
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,1
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      call BSLS(7,1,0)

   case(17)
      do i=0,7
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,6
         call PSPS(i)
         call SSDS(i)
      enddo
      do i=0,5
         call PSDS(i)
         call SSFS(i)
      enddo
      do i=0,4
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,3
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,2
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,1
         call LSBS(1,6,i)
         call LSBS(0,7,i)
      enddo
      call LSBS(1,7,0)

   case(72)
      do i=0,8
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,7
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,6
         call PSDS(i)
         call DSPS(i)
         call FSSS(i)
      enddo
      do i=0,5
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,4
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,3
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,2
         call BSLS(5,2,i)
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      do i=0,1
         call BSLS(6,2,i)
         call BSLS(7,1,i)
      enddo
      call BSLS(7,2,0)

   case(27)
      do i=0,8
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,7
         call PSPS(i)
         call SSDS(i)
         call DSSS(i)
      enddo
      do i=0,6
         call DSPS(i)
         call PSDS(i)
         call SSFS(i)
      enddo
      do i=0,5
         call DSDS(i)
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,4
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,3
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,2
         call LSBS(2,5,i)
         call LSBS(1,6,i)
         call LSBS(0,7,i)
      enddo
      do i=0,1
         call LSBS(2,6,i)
         call LSBS(1,7,i)
      enddo
      call LSBS(2,7,0)

   case(73)
      do i=0,9
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,8
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,7
         call PSDS(i)
         call DSPS(i)
         call FSSS(i)
         call SSFS(i)
      enddo
      do i=0,6
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,5
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,4
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,3
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      do i=0,2
         call BSLS(5,3,i)
         call BSLS(6,2,i)
         call BSLS(7,1,i)
      enddo
      do i=0,1
         call BSLS(6,3,i)
         call BSLS(7,2,i)
      enddo
      call BSLS(7,3,0)

   case(37)
      do i=0,9
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,8
         call PSPS(i)
         call SSDS(i)
         call DSSS(i)
      enddo
      do i=0,7
         call DSPS(i)
         call PSDS(i)
         call SSFS(i)
         call FSSS(i)
      enddo
      do i=0,6
         call FSPS(i)
         call DSDS(i)
         call PSFS(i)
         call SSGS(i)
      enddo
      do i=0,5
         call FSDS(i)
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,4
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,3
         call FSGS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
         call LSBS(0,7,i)
      enddo
      do i=0,2
         call LSBS(3,5,i)
         call LSBS(2,6,i)
         call LSBS(1,7,i)
      enddo
      do i=0,1
         call LSBS(3,6,i)
         call LSBS(2,7,i)
      enddo
      call LSBS(3,7,0)

   case(74)
      do i=0,10
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,9
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,8
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,7
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,6
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,5
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,4
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      do i=0,3
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
         call BSLS(7,1,i)
      enddo
      do i=0,2
         call BSLS(5,4,i)
         call BSLS(6,3,i)
         call BSLS(7,2,i)
      enddo
      do i=0,1
         call BSLS(6,4,i)
         call BSLS(7,3,i)
      enddo
      call BSLS(7,4,0)

   case(47)
      do i=0,10
         call SSPS(i)
         call PSSS(i)
      enddo
      do i=0,9
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,8
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,7
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,6
         call GSPS(i)
         call FSDS(i)
         call DSFS(i)
         call PSGS(i)
         call LSBS(0,5,i)
      enddo
      do i=0,5
         call GSDS(i)
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,4
         call GSFS(i)
         call FSGS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
         call LSBS(0,7,i)
      enddo
      do i=0,3
         call GSGS(i)
         call LSBS(3,5,i)
         call LSBS(2,6,i)
         call LSBS(1,7,i)
      enddo
      do i=0,2
         call LSBS(4,5,i)
         call LSBS(3,6,i)
         call LSBS(2,7,i)
      enddo
      do i=0,1
         call LSBS(4,6,i)
         call LSBS(3,7,i)
      enddo
      call LSBS(4,7,0)

   case(75)
      do i=0,11
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,10
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,9
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,8
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,7
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,6
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,5
         call LSBS(2,5,i)
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      do i=0,4
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
         call BSLS(7,1,i)
      enddo
      do i=0,3
         call LSBS(4,5,i)
         call BSLS(5,4,i)
         call BSLS(6,3,i)
         call BSLS(7,2,i)
      enddo
      do i=0,2
         call BSLS(5,5,i)
         call BSLS(6,4,i)
         call BSLS(7,3,i)
      enddo
      do i=0,1
         call BSLS(6,5,i)
         call BSLS(7,4,i)
      enddo
      call BSLS(7,5,0)

   case(57)
      do i=0,11
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,10
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,9
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,8
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,7
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,6
         call BSLS(5,1,i)
         call GSDS(i)
         call FSFS(i)
         call DSGS(i)
         call LSBS(1,5,i)
         call LSBS(0,6,i)
      enddo
      do i=0,5
         call BSLS(5,2,i)
         call FSGS(i)
         call GSFS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
         call LSBS(0,7,i)
      enddo
      do i=0,4
         call BSLS(5,3,i)
         call GSGS(i)
         call LSBS(3,5,i)
         call LSBS(2,6,i)
         call LSBS(1,7,i)
      enddo
      do i=0,3
         call BSLS(5,4,i)
         call LSBS(4,5,i)
         call LSBS(3,6,i)
         call LSBS(2,7,i)
      enddo
      do i=0,2
         call BSLS(5,5,i)
         call LSBS(4,6,i)
         call LSBS(3,7,i)
      enddo
      do i=0,1
         call LSBS(5,6,i)
         call LSBS(4,7,i)
      enddo
      call LSBS(5,7,0)

   case(76)
      do i=0,12
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,11
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,10
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,9
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,8
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,7
         call LSBS(0,6,i)
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,6
         call LSBS(1,6,i)
         call LSBS(2,5,i)
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      do i=0,5
         call LSBS(2,6,i)
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
         call BSLS(7,1,i)
      enddo
      do i=0,4
         call LSBS(3,6,i)
         call LSBS(4,5,i)
         call BSLS(5,4,i)
         call BSLS(6,3,i)
         call BSLS(7,2,i)
      enddo
      do i=0,3
         call LSBS(4,6,i)
         call BSLS(5,5,i)
         call BSLS(6,4,i)
         call BSLS(7,3,i)
      enddo
      do i=0,2
         call LSBS(5,6,i)
         call BSLS(6,5,i)
         call BSLS(7,4,i)
      enddo
      do i=0,1
         call BSLS(6,6,i)
         call BSLS(7,5,i)
      enddo
      call BSLS(7,6,0)

   case(67)
      do i=0,12
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,11
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,10
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,9
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,8
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,7
         call LSBS(0,6,i)
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,6
         call BSLS(6,1,i)
         call BSLS(5,2,i)
         call FSGS(i)
         call GSFS(i)
         call LSBS(2,5,i)
         call LSBS(1,6,i)
         call LSBS(0,7,i)
      enddo
      do i=0,5
         call LSBS(2,6,i)
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
         call LSBS(1,7,i)
      enddo
      do i=0,4
         call LSBS(3,6,i)
         call LSBS(4,5,i)
         call BSLS(5,4,i)
         call BSLS(6,3,i)
         call LSBS(2,7,i)
      enddo
      do i=0,3
         call LSBS(4,6,i)
         call BSLS(5,5,i)
         call BSLS(6,4,i)
         call LSBS(3,7,i)
      enddo
      do i=0,2
         call LSBS(5,6,i)
         call BSLS(6,5,i)
         call LSBS(4,7,i)
      enddo
      do i=0,1
         call BSLS(6,6,i)
         call LSBS(5,7,i)
      enddo
      call LSBS(6,7,0)

   case(77)
      do i=0,13
         call PSSS(i)
         call SSPS(i)
      enddo
      do i=0,12
         call PSPS(i)
         call DSSS(i)
         call SSDS(i)
      enddo
      do i=0,11
         call FSSS(i)
         call PSDS(i)
         call DSPS(i)
         call SSFS(i)
      enddo
      do i=0,10
         call SSGS(i)
         call PSFS(i)
         call DSDS(i)
         call FSPS(i)
         call GSSS(i)
      enddo
      do i=0,9
         call LSBS(0,5,i)
         call PSGS(i)
         call DSFS(i)
         call FSDS(i)
         call GSPS(i)
         call BSLS(5,0,i)
      enddo
      do i=0,8
         call LSBS(0,6,i)
         call LSBS(1,5,i)
         call DSGS(i)
         call FSFS(i)
         call GSDS(i)
         call BSLS(5,1,i)
         call BSLS(6,0,i)
      enddo
      do i=0,7
         call LSBS(0,7,i)
         call LSBS(1,6,i)
         call LSBS(2,5,i)
         call FSGS(i)
         call GSFS(i)
         call BSLS(5,2,i)
         call BSLS(6,1,i)
         call BSLS(7,0,i)
      enddo
      do i=0,6
         call LSBS(1,7,i)
         call LSBS(2,6,i)
         call LSBS(3,5,i)
         call GSGS(i)
         call BSLS(5,3,i)
         call BSLS(6,2,i)
         call BSLS(7,1,i)
      enddo
      do i=0,5
         call LSBS(2,7,i)
         call LSBS(3,6,i)
         call LSBS(4,5,i)
         call BSLS(5,4,i)
         call BSLS(6,3,i)
         call BSLS(7,2,i)
      enddo
      do i=0,4
         call LSBS(3,7,i)
         call LSBS(4,6,i)
         call BSLS(5,5,i)
         call BSLS(6,4,i)
         call BSLS(7,3,i)
      enddo
      do i=0,3
         call LSBS(4,7,i)
         call LSBS(5,6,i)
         call BSLS(6,5,i)
         call BSLS(7,4,i)
      enddo
      do i=0,2
         call LSBS(5,7,i)
         call BSLS(6,6,i)
         call BSLS(7,5,i)
      enddo
      call LSBS(6,7,0)
      call BSLS(7,6,0)
      call LSBS(6,7,1)
      call BSLS(7,6,1)
      call BSLS(7,7,0)

   end select
end subroutine vertical
