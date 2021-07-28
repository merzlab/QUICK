#include "util.fh"
! prepare all combinations SS, PS, SP, PP, Xiao HE 01/14/2008
! Be careful of coeff
! nuclearspdf.f90
subroutine nuclearattra(Ips,Jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,iatom)
   use allmod

   implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g


   AA(1)=Ax
   AA(2)=Ay
   AA(3)=Az
   BB(1)=Bx
   BB(2)=By
   BB(3)=Bz
   CC(1)=Cx
   CC(2)=Cy
   CC(3)=Cz
   PP(1)=Px
   PP(2)=Py
   PP(3)=Pz

   select case (NIJ1)

   case(0)
   case(10)
      call PSattra(0)
   case(1)
      call SPattra(0)
   case(11)
      call SPattra(0)
      call PSattra(0)
      call PSattra(1)
      call PPattra(0)
   case(20)
      call PSattra(0)
      call PSattra(1)
      call DSattra(0)
   case(2)
      call SPattra(0)
      call SPattra(1)
      call SDattra(0)
   case(21)
      call PSattra(0)
      call PSattra(1)
      call PSattra(2)
      call DSattra(0)
      call DSattra(1)
      call DPattra(0)
   case(12)
      call SPattra(0)
      call SPattra(1)
      call SPattra(2)
      call SDattra(0)
      call SDattra(1)
      call PDattra(0)
   case(22)
      do itempt=0,3
         call PSattra(itempt)
      enddo
      do itempt=0,1
         call PPattra(itempt)
      enddo
      do itempt=0,2
         call DSattra(itempt)
      enddo
      do itempt=0,1
         call DPattra(itempt)
      enddo
      call DDattra(0)
   case(30)
      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call DSattra(itemp)
      enddo
      call FSattra(0)
   case(3)
      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call SDattra(itemp)
      enddo
      call SFattra(0)
   case(31)
      do itemp=0,3
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo
      call FPattra(0)
   case(13)
      do itemp=0,3
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo
      call PFattra(0)
   case(32)
      do itemp=0,4
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo
      call FDattra(0)
   case(23)
      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo
      call DFattra(0)
   case(33)
      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo
      call FFattra(0)
   end select

   do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
      X1temp=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
      do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
         NBI1=quick_basis%Qsbasis(IIsh,Iang)
         NBI2=quick_basis%Qfbasis(IIsh,Iang)
         NBJ1=quick_basis%Qsbasis(JJsh,Jang)
         NBJ2=quick_basis%Qfbasis(JJsh,Jang)

         III1=quick_basis%ksumtype(IIsh)+NBI1
         III2=quick_basis%ksumtype(IIsh)+NBI2
         JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
         JJJ2=quick_basis%ksumtype(JJsh)+NBJ2

         Xconstant=X1temp*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)
         do III=III1,III2
            itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
            do JJJ=max(III,JJJ1),JJJ2
               itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))

!              write(*,'(I5,2X,F20.10,2X,F20.10,2X,F20.10,2X,F20.10,2X,F20.10)') JJJ,&
!              quick_qm_struct%o(JJJ,III),Xconstant,quick_basis%cons(III),quick_basis%cons(JJJ)&
!              ,attraxiao(itemp1,itemp2,0) 

               quick_qm_struct%o(JJJ,III)=quick_qm_struct%o(JJJ,III)+ &
                     Xconstant*quick_basis%cons(III)*quick_basis%cons(JJJ)*attraxiao(itemp1,itemp2,0)
!write(*,*) "Madu O:", quick_qm_struct%o(JJJ,III)
            enddo
         enddo

      enddo
   enddo
!write(*,*) "Returning.."
   201 return
End subroutine nuclearattra



!***Xiao HE******** 07/07/07 version
! new lesson: be careful of HSSS,ISSS,JSSS
!*Lesson1,angular momentum;2,angular momentum factor;3.All possibilties in order.
!Vertical Recursion subroutines by hand, these parts can be optimized by MAPLE
subroutine PSattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do i=1,3
      attraxiao(i+1,1,mtemp)=(PP(i)-AA(i))*(aux(mtemp))-(PP(i)-CC(i))*(aux(mtemp+1))
   enddo

end subroutine PSattra


subroutine SPattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do i=1,3
      attraxiao(1,i+1,mtemp)=(PP(i)-BB(i))*(aux(mtemp))-(PP(i)-CC(i))*(aux(mtemp+1))
   enddo

end subroutine SPattra


subroutine PPattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=1,3
      do jtemp=1,3
         attraxiao(itemp+1,jtemp+1,mtemp)=(PP(jtemp)-BB(jtemp))*(attraxiao(itemp+1,1,mtemp))- &
               (PP(jtemp)-CC(jtemp))*(attraxiao(itemp+1,1,mtemp+1))
         if(itemp.eq.jtemp)then
            attraxiao(itemp+1,jtemp+1,mtemp)=attraxiao(itemp+1,jtemp+1,mtemp)+ &
                  0.5d0/g*(aux(mtemp)-aux(mtemp+1))
         endif
      enddo
   enddo

   !111  continue

end subroutine PPattra


subroutine DSattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=5,10
      a(1)=Mcal(1,itemp)
      a(2)=Mcal(2,itemp)
      a(3)=Mcal(3,itemp)
      do j=1,3
         if(Mcal(j,itemp).ne.0)then
            a(j)=a(j)-1
            itempnew=trans(a(1),a(2),a(3))
            attraxiao(itemp,1,mtemp)=(PP(j)-AA(j))*(attraxiao(itempnew,1,mtemp))- &
                  (PP(j)-CC(j))*(attraxiao(itempnew,1,mtemp+1))
            if(Mcal(j,itemp).eq.2)then
               attraxiao(itemp,1,mtemp)=attraxiao(itemp,1,mtemp)+0.5d0/g*(aux(mtemp)-aux(mtemp+1))
            endif
            goto 111
         endif
      enddo
      111  continue
   enddo

end subroutine DSattra


subroutine SDattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=5,10
      a(1)=Mcal(1,itemp)
      a(2)=Mcal(2,itemp)
      a(3)=Mcal(3,itemp)
      do j=1,3
         if(Mcal(j,itemp).ne.0)then
            a(j)=a(j)-1
            itempnew=trans(a(1),a(2),a(3))
            attraxiao(1,itemp,mtemp)=(PP(j)-BB(j))*(attraxiao(1,itempnew,mtemp))- &
                  (PP(j)-CC(j))*(attraxiao(1,itempnew,mtemp+1))
            if(Mcal(j,itemp).eq.2)then
               attraxiao(1,itemp,mtemp)=attraxiao(1,itemp,mtemp)+0.5d0/g*(aux(mtemp)-aux(mtemp+1))
            endif
            goto 111
         endif
      enddo
      111  continue
   enddo

end subroutine SDattra


subroutine DPattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=5,10
      do i=2,4
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,1,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,1,mtemp+1))
               if(a(jj).ne.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,jtemp))*(attraxiao(itempnew,1,mtemp)-attraxiao(itempnew,1,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

end subroutine DPattra


subroutine PDattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=5,10
      do i=2,4
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               attraxiao(i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiao(1,jtemp,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(1,jtemp,mtemp+1))
               if(a(jj).ne.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(1,itempnew,mtemp)-attraxiao(1,itempnew,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine PDattra


subroutine DDattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=5,10
      do i=5,10
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Axiao(jj)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,ixiao,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,ixiao,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,i)-1)*(attraxiao(jtemp,1,mtemp)-attraxiao(jtemp,1,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(itempnew,ixiao,mtemp)-attraxiao(itempnew,ixiao,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine DDattra

subroutine FSattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=11,20
      a(1)=Mcal(1,itemp)
      a(2)=Mcal(2,itemp)
      a(3)=Mcal(3,itemp)
      do j=1,3
         if(Mcal(j,itemp).ne.0)then
            a(j)=Mcal(j,itemp)-1
            itempnew=trans(a(1),a(2),a(3))
            attraxiao(itemp,1,mtemp)=(PP(j)-AA(j))*(attraxiao(itempnew,1,mtemp))- &
                  (PP(j)-CC(j))*(attraxiao(itempnew,1,mtemp+1))
            if(Mcal(j,itemp).gt.1)then
               a(j)=Mcal(j,itemp)-2
               itempnew=trans(a(1),a(2),a(3))
               attraxiao(itemp,1,mtemp)=attraxiao(itemp,1,mtemp)+ &
                     0.5d0/g*(Mcal(j,itemp)-1)*(attraxiao(itempnew,1,mtemp)-attraxiao(itempnew,1,mtemp+1))
            endif
            goto 111
         endif
      enddo
      111  continue
   enddo

end subroutine FSattra


subroutine SFattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=11,20
      a(1)=Mcal(1,itemp)
      a(2)=Mcal(2,itemp)
      a(3)=Mcal(3,itemp)
      do j=1,3
         if(Mcal(j,itemp).ne.0)then
            a(j)=Mcal(j,itemp)-1
            itempnew=trans(a(1),a(2),a(3))
            attraxiao(1,itemp,mtemp)=(PP(j)-BB(j))*(attraxiao(1,itempnew,mtemp))- &
                  (PP(j)-CC(j))*(attraxiao(1,itempnew,mtemp+1))
            if(Mcal(j,itemp).gt.1)then
               a(j)=Mcal(j,itemp)-2
               itempnew=trans(a(1),a(2),a(3))
               attraxiao(1,itemp,mtemp)=attraxiao(1,itemp,mtemp)+ &
                     0.5d0/g*(Mcal(j,itemp)-1)*(attraxiao(1,itempnew,mtemp)-attraxiao(1,itempnew,mtemp+1))
            endif
            goto 111
         endif
      enddo
      111  continue
   enddo

end subroutine SFattra


subroutine FPattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=2,4
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,1,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,1,mtemp+1))
               if(a(jj).ne.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,jtemp))*(attraxiao(itempnew,1,mtemp)-attraxiao(itempnew,1,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

end subroutine FPattra


subroutine PFattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=2,4
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               attraxiao(i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiao(1,jtemp,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(1,jtemp,mtemp+1))
               if(a(jj).ne.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*(Mcal(jj,jtemp))*(attraxiao(1,itempnew,mtemp)-attraxiao(1,itempnew,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine PFattra


subroutine FDattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=5,10
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,ixiao,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,ixiao,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(attraxiao(jtemp,1,mtemp)-attraxiao(jtemp,1,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(itempnew,ixiao,mtemp)-attraxiao(itempnew,ixiao,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine FDattra

subroutine DFattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=5,10
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiao(ixiao,jtemp,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(ixiao,jtemp,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*(attraxiao(1,jtemp,mtemp)-attraxiao(1,jtemp,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(ixiao,itempnew,mtemp)-attraxiao(ixiao,itempnew,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine DFattra


subroutine FFattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=11,20
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,ixiao,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,ixiao,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  Axiao(jj)=Mcal(jj,i)-2
                  inewtemp=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,i)-1)*(attraxiao(jtemp,inewtemp,mtemp)- &
                        attraxiao(jtemp,inewtemp,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(itempnew,ixiao,mtemp)-attraxiao(itempnew,ixiao,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine FFattra


subroutine GSattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=21,35
      a(1)=Mcal(1,itemp)
      a(2)=Mcal(2,itemp)
      a(3)=Mcal(3,itemp)
      do j=1,3
         if(Mcal(j,itemp).ne.0)then
            a(j)=Mcal(j,itemp)-1
            itempnew=trans(a(1),a(2),a(3))
            attraxiao(itemp,1,mtemp)=(PP(j)-AA(j))*(attraxiao(itempnew,1,mtemp))- &
                  (PP(j)-CC(j))*(attraxiao(itempnew,1,mtemp+1))
            if(Mcal(j,itemp).gt.1)then
               a(j)=Mcal(j,itemp)-2
               itempnew=trans(a(1),a(2),a(3))
               attraxiao(itemp,1,mtemp)=attraxiao(itemp,1,mtemp)+ &
                     0.5d0/g*(Mcal(j,itemp)-1)*(attraxiao(itempnew,1,mtemp)-attraxiao(itempnew,1,mtemp+1))
            endif
            goto 111
         endif
      enddo
      111  continue
   enddo

end subroutine GSattra


subroutine SGattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=21,35
      a(1)=Mcal(1,itemp)
      a(2)=Mcal(2,itemp)
      a(3)=Mcal(3,itemp)
      do j=1,3
         if(Mcal(j,itemp).ne.0)then
            a(j)=Mcal(j,itemp)-1
            itempnew=trans(a(1),a(2),a(3))
            attraxiao(1,itemp,mtemp)=(PP(j)-BB(j))*(attraxiao(1,itempnew,mtemp))- &
                  (PP(j)-CC(j))*(attraxiao(1,itempnew,mtemp+1))
            if(Mcal(j,itemp).gt.1)then
               a(j)=Mcal(j,itemp)-2
               itempnew=trans(a(1),a(2),a(3))
               attraxiao(1,itemp,mtemp)=attraxiao(1,itemp,mtemp)+ &
                     0.5d0/g*(Mcal(j,itemp)-1)*(attraxiao(1,itempnew,mtemp)-attraxiao(1,itempnew,mtemp+1))
            endif
            goto 111
         endif
      enddo
      111  continue
   enddo

end subroutine SGattra


subroutine GPattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=2,4
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,1,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,1,mtemp+1))
               if(a(jj).ne.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,jtemp))*(attraxiao(itempnew,1,mtemp)-attraxiao(itempnew,1,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

end subroutine GPattra


subroutine PGattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=2,4
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               attraxiao(i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiao(1,jtemp,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(1,jtemp,mtemp+1))
               if(a(jj).ne.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*(Mcal(jj,jtemp))*(attraxiao(1,itempnew,mtemp)-attraxiao(1,itempnew,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine PGattra


subroutine GDattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=5,10
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,ixiao,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,ixiao,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(attraxiao(jtemp,1,mtemp)-attraxiao(jtemp,1,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(itempnew,ixiao,mtemp)-attraxiao(itempnew,ixiao,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine GDattra

subroutine DGattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=5,10
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiao(ixiao,jtemp,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(ixiao,jtemp,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*(attraxiao(1,jtemp,mtemp)-attraxiao(1,jtemp,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(ixiao,itempnew,mtemp)-attraxiao(ixiao,itempnew,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine DGattra


subroutine GFattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=11,20
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,ixiao,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,ixiao,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  Axiao(jj)=Mcal(jj,i)-2
                  inewtemp=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,i)-1)*(attraxiao(jtemp,inewtemp,mtemp)- &
                        attraxiao(jtemp,inewtemp,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(itempnew,ixiao,mtemp)-attraxiao(itempnew,ixiao,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine GFattra


subroutine FGattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=11,20
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiao(ixiao,jtemp,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(ixiao,jtemp,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  !          a(i)=a(i)-1
                  !          itempnew=trans(a(1),a(2),a(3))
                  Axiao(jj)=Mcal(jj,i)-2
                  inewtemp=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*(Mcal(jj,i)-1)*(attraxiao(inewtemp,jtemp,mtemp)- &
                        attraxiao(inewtemp,jtemp,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(i,jtemp,mtemp)=attraxiao(i,jtemp,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(ixiao,itempnew,mtemp)- &
                        attraxiao(ixiao,itempnew,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine FGattra


subroutine GGattra(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=21,35
      do i=21,35
         a(1)=Mcal(1,jtemp)
         a(2)=Mcal(2,jtemp)
         a(3)=Mcal(3,jtemp)
         Axiao(1)=Mcal(1,i)
         Axiao(2)=Mcal(2,i)
         Axiao(3)=Mcal(3,i)
         do jj=1,3
            if(Mcal(jj,i).ne.0)then
               Axiao(jj)=Mcal(jj,i)-1
               ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
               attraxiao(jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiao(jtemp,ixiao,mtemp))- &
                     (PP(jj)-CC(jj))*(attraxiao(jtemp,ixiao,mtemp+1))
               if(Mcal(jj,i).gt.1)then
                  !          a(i)=a(i)-1
                  !          itempnew=trans(a(1),a(2),a(3))
                  Axiao(jj)=Mcal(jj,i)-2
                  inewtemp=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*(Mcal(jj,i)-1)*(attraxiao(jtemp,inewtemp,mtemp)- &
                        attraxiao(jtemp,inewtemp,mtemp+1))
               endif

               if(a(jj).gt.0)then
                  a(jj)=a(jj)-1
                  itempnew=trans(a(1),a(2),a(3))
                  attraxiao(jtemp,i,mtemp)=attraxiao(jtemp,i,mtemp)+ &
                        0.5d0/g*Mcal(jj,jtemp)*(attraxiao(itempnew,ixiao,mtemp)-attraxiao(itempnew,ixiao,mtemp+1))
               endif

               goto 111
            endif
         enddo

         111     continue

      enddo
   enddo

End subroutine GGattra


! prepare all combinations SS, PS, SP, PP, Xiao HE 01/14/2008
! Be careful of coeff
! nuclearspdf.f90
subroutine nuclearattraenergy(Ips,Jps,IIsh,JJsh,NIJ1, &
      Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,iatom)
   use allmod

   implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   AA(1)=Ax
   AA(2)=Ay
   AA(3)=Az
   BB(1)=Bx
   BB(2)=By
   BB(3)=Bz
   CC(1)=Cx
   CC(2)=Cy
   CC(3)=Cz
   PP(1)=Px
   PP(2)=Py
   PP(3)=Pz

   select case (NIJ1)

   case(0)
   case(10)
      call PSattra(0)
   case(1)
      call SPattra(0)
   case(11)
      call SPattra(0)
      call PSattra(0)
      call PSattra(1)
      call PPattra(0)
   case(20)
      call PSattra(0)
      call PSattra(1)
      call DSattra(0)
   case(2)
      call SPattra(0)
      call SPattra(1)
      call SDattra(0)
   case(21)
      call PSattra(0)
      call PSattra(1)
      call PSattra(2)
      call DSattra(0)
      call DSattra(1)
      call DPattra(0)
   case(12)
      call SPattra(0)
      call SPattra(1)
      call SPattra(2)
      call SDattra(0)
      call SDattra(1)
      call PDattra(0)
   case(22)
      do itempt=0,3
         call PSattra(itempt)
      enddo
      do itempt=0,1
         call PPattra(itempt)
      enddo
      do itempt=0,2
         call DSattra(itempt)
      enddo
      do itempt=0,1
         call DPattra(itempt)
      enddo

      call DDattra(0)

   case(30)

      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call DSattra(itemp)
      enddo

      call FSattra(0)

   case(3)

      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call SDattra(itemp)
      enddo

      call SFattra(0)

   case(31)

      do itemp=0,3
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo

      call FPattra(0)

   case(13)

      do itemp=0,3
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo

      call PFattra(0)

   case(32)

      do itemp=0,4
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo

      call FDattra(0)

   case(23)

      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo

      call DFattra(0)

   case(33)

      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo

      call FFattra(0)


   end select


   do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
      X1temp=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
      do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
         NBI1=quick_basis%Qsbasis(IIsh,Iang)
         NBI2=quick_basis%Qfbasis(IIsh,Iang)
         NBJ1=quick_basis%Qsbasis(JJsh,Jang)
         NBJ2=quick_basis%Qfbasis(JJsh,Jang)

         III1=quick_basis%ksumtype(IIsh)+NBI1
         III2=quick_basis%ksumtype(IIsh)+NBI2
         JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
         JJJ2=quick_basis%ksumtype(JJsh)+NBJ2

         Xconstant=X1temp*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)

         do III=III1,III2
            itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
            do JJJ=max(III,JJJ1),JJJ2

               if (quick_method%UNRST) then
                  DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)
               else
                  DENSEJI=quick_qm_struct%dense(JJJ,III)
               endif

               if(JJJ.ne.III)DENSEJI=DENSEJI*2.0d0
               itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))
               quick_qm_struct%Eel=quick_qm_struct%Eel+DENSEJI*Xconstant*quick_basis%cons(III)*quick_basis%cons(JJJ)* &
                     attraxiao(itemp1,itemp2,0)
            enddo
         enddo

      enddo
   enddo


End subroutine nuclearattraenergy

! be careful of sequence PP DD FF
! prepare all combinations SS, PS, SP, PP, Xiao HE 01/14/2008
! Be careful of coeff
! nuclearspdf.f90
subroutine nuclearattraopt(Ips,Jps,IIsh,JJsh,NIJ1, &
      Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,iatom)
   use allmod

   implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   AA(1)=Ax
   AA(2)=Ay
   AA(3)=Az
   BB(1)=Bx
   BB(2)=By
   BB(3)=Bz
   CC(1)=Cx
   CC(2)=Cy
   CC(3)=Cz
   PP(1)=Px
   PP(2)=Py
   PP(3)=Pz


   select case (NIJ1+11)

   case(0)
   case(10)
      call PSattra(0)
   case(1)
      call SPattra(0)
   case(11)
      call SPattra(0)
      call PSattra(0)
      call PSattra(1)
      call PPattra(0)
   case(20)
      call PSattra(0)
      call PSattra(1)
      call DSattra(0)
   case(2)
      call SPattra(0)
      call SPattra(1)
      call SDattra(0)
   case(21)
      call PSattra(0)
      call PSattra(1)
      call PSattra(2)
      call DSattra(0)
      call DSattra(1)
      call DPattra(0)
      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call PPattra(itemp)
      enddo
   case(12)
      call SPattra(0)
      call SPattra(1)
      call SPattra(2)
      call SDattra(0)
      call SDattra(1)
      call PDattra(0)
      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call PPattra(itemp)
      enddo
   case(22)
      do itempt=0,3
         call PSattra(itempt)
      enddo
      do itempt=0,1
         call PPattra(itempt)
      enddo
      do itempt=0,2
         call DSattra(itempt)
      enddo
      do itempt=0,1
         call DPattra(itempt)
      enddo

      call DDattra(0)

      ! new
      do itemp=0,3
         call SPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo

   case(30)

      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call DSattra(itemp)
      enddo

      call FSattra(0)

   case(3)

      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call SDattra(itemp)
      enddo

      call SFattra(0)

   case(31)

      do itemp=0,3
         call PSattra(itemp)
         call SPattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo

      call FPattra(0)

   case(13)

      do itemp=0,3
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo

      call PFattra(0)



   case(32)

      do itemp=0,4
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo

      call FDattra(0)

      do itemp=0,4
         call SPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call DDattra(itemp)
      enddo

   case(23)

      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo

      call DFattra(0)
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call DDattra(itemp)
      enddo


   case(33)

      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo

      call FFattra(0)

      do itemp=0,5
         call SPattra(itemp)
      enddo
      do itemp=0,4
         call SDattra(itemp)
      enddo
      do itemp=0,3
         call PDattra(itemp)
      enddo
      do itemp=0,3
         call SFattra(itemp)
      enddo
      do itemp=0,2
         call PFattra(itemp)
      enddo
      do itemp=0,1
         call DFattra(itemp)
      enddo

   case(40)

      do itemp=0,3
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo

      call GSattra(0)

   case(4)

      do itemp=0,3
         call SPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo

      call SGattra(0)

   case(41)

      do itemp=0,4
         call PSattra(itemp)
         call SPattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call GSattra(itemp)
      enddo

      call GPattra(0)

   case(14)

      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo
      do itemp=0,1
         call SGattra(itemp)
      enddo

      call PGattra(0)


   case(42)

      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo
      do itemp=0,2
         call GSattra(itemp)
      enddo
      do itemp=0,1
         call GPattra(itemp)
      enddo

      call GDattra(0)

      ! new
      do itemp=0,5
         call SPattra(itemp)
      enddo
      do itemp=0,4
         call SDattra(itemp)
      enddo
      do itemp=0,3
         call PDattra(itemp)
      enddo


   case(24)

      do itemp=0,5
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call SDattra(itemp)
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call PDattra(itemp)
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call SFattra(itemp)
      enddo
      do itemp=0,2
         call PFattra(itemp)
      enddo
      do itemp=0,1
         call DFattra(itemp)
      enddo
      do itemp=0,2
         call SGattra(itemp)
      enddo
      do itemp=0,1
         call PGattra(itemp)
      enddo

      call DGattra(0)

   case(43)

      do itemp=0,6
         call PSattra(itemp)
      enddo
      do itemp=0,5
         call PPattra(itemp)
      enddo
      do itemp=0,5
         call DSattra(itemp)
      enddo
      do itemp=0,4
         call DPattra(itemp)
      enddo
      do itemp=0,3
         call DDattra(itemp)
      enddo
      do itemp=0,4
         call FSattra(itemp)
      enddo
      do itemp=0,3
         call FPattra(itemp)
      enddo
      do itemp=0,2
         call FDattra(itemp)
      enddo
      do itemp=0,1
         call FFattra(itemp)
      enddo
      do itemp=0,3
         call GSattra(itemp)
      enddo
      do itemp=0,2
         call GPattra(itemp)
      enddo
      do itemp=0,1
         call GDattra(itemp)
      enddo
      call GFattra(0)

      ! new
      do itemp=0,6
         call SPattra(itemp)
      enddo
      do itemp=0,5
         call SDattra(itemp)
      enddo
      do itemp=0,4
         call PDattra(itemp)
      enddo
      do itemp=0,4
         call SFattra(itemp)
      enddo
      do itemp=0,3
         call PFattra(itemp)
      enddo
      do itemp=0,2
         call DFattra(itemp)
      enddo


   case(34)

      do itemp=0,6
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,5
         call PPattra(itemp)
      enddo
      do itemp=0,5
         call SDattra(itemp)
         call DSattra(itemp)
      enddo
      do itemp=0,4
         call PDattra(itemp)
         call DPattra(itemp)
      enddo
      do itemp=0,3
         call DDattra(itemp)
      enddo
      do itemp=0,4
         call SFattra(itemp)
         call FSattra(itemp)
      enddo
      do itemp=0,3
         call PFattra(itemp)
         call FPattra(itemp)
      enddo
      do itemp=0,2
         call DFattra(itemp)
         call FDattra(itemp)
      enddo
      do itemp=0,1
         call FFattra(itemp)
      enddo
      do itemp=0,3
         call SGattra(itemp)
      enddo
      do itemp=0,2
         call PGattra(itemp)
      enddo
      do itemp=0,1
         call DGattra(itemp)
      enddo
      call FGattra(0)

   case(44)

      do itemp=0,7
         call PSattra(itemp)
      enddo
      do itemp=0,6
         call PPattra(itemp)
      enddo
      do itemp=0,6
         call DSattra(itemp)
      enddo
      do itemp=0,5
         call DPattra(itemp)
      enddo
      do itemp=0,4
         call DDattra(itemp)
      enddo
      do itemp=0,5
         call FSattra(itemp)
      enddo
      do itemp=0,4
         call FPattra(itemp)
      enddo
      do itemp=0,3
         call FDattra(itemp)
      enddo
      do itemp=0,2
         call FFattra(itemp)
      enddo
      do itemp=0,4
         call GSattra(itemp)
      enddo
      do itemp=0,3
         call GPattra(itemp)
      enddo
      do itemp=0,2
         call GDattra(itemp)
      enddo
      do itemp=0,1
         call GFattra(itemp)
      enddo
      call GGattra(0)

      ! new
      do itemp=0,7
         call SPattra(itemp)
      enddo
      do itemp=0,6
         call SDattra(itemp)
      enddo
      do itemp=0,5
         call PDattra(itemp)
      enddo
      do itemp=0,5
         call SFattra(itemp)
      enddo
      do itemp=0,4
         call PFattra(itemp)
      enddo
      do itemp=0,3
         call DFattra(itemp)
      enddo
      do itemp=0,4
         call SGattra(itemp)
      enddo
      do itemp=0,3
         call PGattra(itemp)
      enddo
      do itemp=0,2
         call DGattra(itemp)
      enddo
      do itemp=0,1
         call FGattra(itemp)
      enddo

   end select


   ! new opt
!   select case (NIJ1)
!
!   case(0)
!   case(10)
!      call PSattraopt(0)
!   case(1)
!      call SPattraopt(0)
!   case(11)
!
!      call SPattraopt(0)
!      call PSattraopt(0)
!      call PSattraopt(1)
!      call PPattraopt(0)
!
!   case(20)
!
!      call PSattraopt(0)
!      call PSattraopt(1)
!      call DSattraopt(0)
!
!   case(2)
!
!      call SPattraopt(0)
!      call SPattraopt(1)
!      call SDattraopt(0)
!
!   case(21)
!
!      call PSattraopt(0)
!      call PSattraopt(1)
!      call PSattraopt(2)
!      call DSattraopt(0)
!      call DSattraopt(1)
!      call DPattraopt(0)
!
!   case(12)
!
!      call SPattraopt(0)
!      call SPattraopt(1)
!      call SPattraopt(2)
!      call SDattraopt(0)
!      call SDattraopt(1)
!      call PDattraopt(0)
!
!
!   case(22)
!
!      do itempt=0,3
!         call PSattraopt(itempt)
!      enddo
!      do itempt=0,1
!         call PPattraopt(itempt)
!      enddo
!      do itempt=0,2
!         call DSattraopt(itempt)
!      enddo
!      do itempt=0,1
!         call DPattraopt(itempt)
!      enddo
!
!      call DDattraopt(0)
!
!   case(30)
!
!      do itemp=0,2
!         call PSattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call DSattraopt(itemp)
!      enddo
!
!      call FSattraopt(0)
!
!   case(3)
!
!      do itemp=0,2
!         call SPattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call SDattraopt(itemp)
!      enddo
!
!      call SFattraopt(0)
!
!   case(31)
!
!      do itemp=0,3
!         call PSattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call PPattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call DSattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call DPattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call FSattraopt(itemp)
!      enddo
!
!      call FPattraopt(0)
!
!   case(13)
!
!      do itemp=0,3
!         call SPattraopt(itemp)
!         call PSattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call PPattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call SDattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call PDattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call SFattraopt(itemp)
!      enddo
!
!      call PFattraopt(0)
!
!   case(32)
!
!      do itemp=0,4
!         call PSattraopt(itemp)
!      enddo
!      do itemp=0,3
!         call PPattraopt(itemp)
!      enddo
!      do itemp=0,3
!         call DSattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call DPattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call FSattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call FPattraopt(itemp)
!      enddo
!
!      call FDattraopt(0)
!
!   case(23)
!
!      do itemp=0,4
!         call SPattraopt(itemp)
!         call PSattraopt(itemp)
!      enddo
!      do itemp=0,3
!         call PPattraopt(itemp)
!      enddo
!      do itemp=0,3
!         call SDattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call PDattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call SFattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call PFattraopt(itemp)
!      enddo
!
!      call DFattraopt(0)
!
!   case(33)
!
!      do itemp=0,5
!         call PSattraopt(itemp)
!      enddo
!      do itemp=0,4
!         call PPattraopt(itemp)
!      enddo
!      do itemp=0,4
!         call DSattraopt(itemp)
!      enddo
!      do itemp=0,3
!         call DPattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call DDattraopt(itemp)
!      enddo
!      do itemp=0,3
!         call FSattraopt(itemp)
!      enddo
!      do itemp=0,2
!         call FPattraopt(itemp)
!      enddo
!      do itemp=0,1
!         call FDattraopt(itemp)
!      enddo
!
!      call FFattraopt(0)
!
!
!   end select

   Agrad1=0.0d0
   Agrad2=0.0d0
   Agrad3=0.0d0
   Bgrad1=0.0d0
   Bgrad2=0.0d0
   Bgrad3=0.0d0
   Cgrad1=0.0d0
   Cgrad2=0.0d0
   Cgrad3=0.0d0

   do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
      X1temp=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
      do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
         NBI1=quick_basis%Qsbasis(IIsh,Iang)
         NBI2=quick_basis%Qfbasis(IIsh,Iang)
         NBJ1=quick_basis%Qsbasis(JJsh,Jang)
         NBJ2=quick_basis%Qfbasis(JJsh,Jang)

         III1=quick_basis%ksumtype(IIsh)+NBI1
         III2=quick_basis%ksumtype(IIsh)+NBI2
         JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
         JJJ2=quick_basis%ksumtype(JJsh)+NBJ2

         iA=quick_basis%katom(IIsh)
         iB=quick_basis%katom(JJsh)
         iC=iatom

         iAstart = (iA-1)*3
         iBstart = (iB-1)*3
         iCstart = (iC-1)*3

         Xconstant=X1temp*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)

         do III=III1,III2
            Xconstant1=Xconstant*quick_basis%cons(III)
            itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
            do JJJ=max(III,JJJ1),JJJ2

               if (quick_method%UNRST) then
                  DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)
               else
                  DENSEJI=quick_qm_struct%dense(JJJ,III)
               endif

               if(III.ne.JJJ)DENSEJI=2.0d0*DENSEJI
               Xconstant2=Xconstant1*quick_basis%cons(JJJ)*DENSEJI
               itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))

!               Cgrad1=Cgrad1+Xconstant2*attraxiaoopt(1,itemp1,itemp2,0)
!               Cgrad2=Cgrad2+Xconstant2*attraxiaoopt(2,itemp1,itemp2,0)
!               Cgrad3=Cgrad3+Xconstant2*attraxiaoopt(3,itemp1,itemp2,0)

               itemp1new=trans(quick_basis%KLMN(1,III)+1,quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
               Agrad1=Agrad1+2.0d0*Xconstant2* &
                     quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))*attraxiao(itemp1new,itemp2,0)
               if(quick_basis%KLMN(1,III).ge.1)then
                  itemp1new=trans(quick_basis%KLMN(1,III)-1,quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
                  Agrad1=Agrad1-Xconstant2* &
                        quick_basis%KLMN(1,III)*attraxiao(itemp1new,itemp2,0)
               endif

               itemp1new=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III)+1,quick_basis%KLMN(3,III))
               Agrad2=Agrad2+2.0d0*Xconstant2* &
                     quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))*attraxiao(itemp1new,itemp2,0)
               if(quick_basis%KLMN(2,III).ge.1)then
                  itemp1new=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III)-1,quick_basis%KLMN(3,III))
                  Agrad2=Agrad2-Xconstant2* &
                        quick_basis%KLMN(2,III)*attraxiao(itemp1new,itemp2,0)
               endif

               itemp1new=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III)+1)
               Agrad3=Agrad3+2.0d0*Xconstant2* &
                     quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))*attraxiao(itemp1new,itemp2,0)
               if(quick_basis%KLMN(3,III).ge.1)then
                  itemp1new=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III)-1)
                  Agrad3=Agrad3-Xconstant2* &
                        quick_basis%KLMN(3,III)*attraxiao(itemp1new,itemp2,0)
               endif

               itemp2new=trans(quick_basis%KLMN(1,JJJ)+1,quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))
               Bgrad1=Bgrad1+2.0d0*Xconstant2* &
                     quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))*attraxiao(itemp1,itemp2new,0)
               if(quick_basis%KLMN(1,JJJ).ge.1)then
                  itemp2new=trans(quick_basis%KLMN(1,JJJ)-1,quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))
                  Bgrad1=Bgrad1-Xconstant2* &
                        quick_basis%KLMN(1,JJJ)*attraxiao(itemp1,itemp2new,0)
               endif

               itemp2new=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ)+1,quick_basis%KLMN(3,JJJ))
               Bgrad2=Bgrad2+2.0d0*Xconstant2* &
                     quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))*attraxiao(itemp1,itemp2new,0)
               if(quick_basis%KLMN(2,JJJ).ge.1)then
                  itemp2new=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ)-1,quick_basis%KLMN(3,JJJ))
                  Bgrad2=Bgrad2-Xconstant2* &
                        quick_basis%KLMN(2,JJJ)*attraxiao(itemp1,itemp2new,0)
               endif

               itemp2new=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ)+1)
               Bgrad3=Bgrad3+2.0d0*Xconstant2* &
                     quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))*attraxiao(itemp1,itemp2new,0)
               if(quick_basis%KLMN(3,JJJ).ge.1)then
                  itemp2new=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ)-1)
                  Bgrad3=Bgrad3-Xconstant2* &
                        quick_basis%KLMN(3,JJJ)*attraxiao(itemp1,itemp2new,0)
               endif
            enddo
         enddo
      enddo
   enddo

   ! use the gradient invariance to determine the derivatives wrt nuclei/external
   ! point charge.
   CGrad1 = -(AGrad1+BGrad1)
   CGrad2 = -(AGrad2+BGrad2)
   CGrad3 = -(AGrad3+BGrad3)

   quick_qm_struct%gradient(iASTART+1) = quick_qm_struct%gradient(iASTART+1)+ AGrad1
   quick_qm_struct%gradient(iASTART+2) = quick_qm_struct%gradient(iASTART+2)+ AGrad2
   quick_qm_struct%gradient(iASTART+3) = quick_qm_struct%gradient(iASTART+3)+ AGrad3

   quick_qm_struct%gradient(iBSTART+1) = quick_qm_struct%gradient(iBSTART+1)+ BGrad1
   quick_qm_struct%gradient(iBSTART+2) = quick_qm_struct%gradient(iBSTART+2)+ BGrad2
   quick_qm_struct%gradient(iBSTART+3) = quick_qm_struct%gradient(iBSTART+3)+ BGrad3

if(iatom<=natom)then
   quick_qm_struct%gradient(iCSTART+1) = quick_qm_struct%gradient(iCSTART+1)+ CGrad1
   quick_qm_struct%gradient(iCSTART+2) = quick_qm_struct%gradient(iCSTART+2)+ CGrad2
   quick_qm_struct%gradient(iCSTART+3) = quick_qm_struct%gradient(iCSTART+3)+ CGrad3
else
!  One electron-point charge attraction grdients, update point charge gradient vector 
   iCSTART = (iatom-natom-1)*3
   quick_qm_struct%ptchg_gradient(iCSTART+1) = quick_qm_struct%ptchg_gradient(iCSTART+1)+ CGrad1
   quick_qm_struct%ptchg_gradient(iCSTART+2) = quick_qm_struct%ptchg_gradient(iCSTART+2)+ CGrad2
   quick_qm_struct%ptchg_gradient(iCSTART+3) = quick_qm_struct%ptchg_gradient(iCSTART+3)+ CGrad3
endif

End subroutine nuclearattraopt


! Big mistake itempnew
!  be aware of the initial cycle
! be careful of                   0.5d0/g*(Mcal(jj,jtemp))*(attraxiaoopt(idxiao,itempnew,1,mtemp)- &
      !***Xiao HE******** 07/07/07 version
! new lesson: be careful of HSSS,ISSS,JSSS
!*Lesson1,angular momentum;2,angular momentum factor;3.All possibilties in order.
!Vertical Recursion subroutines by hand, these parts can be optimized by MAPLE
subroutine PSattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do i=1,3
      do idxiao=1,3
         attraxiaoopt(idxiao,i+1,1,mtemp)=(PP(i)-AA(i))*(attraxiaoopt(idxiao,1,1,mtemp))- &
               (PP(i)-CC(i))*(attraxiaoopt(idxiao,1,1,mtemp+1))
         if(idxiao.eq.i)then
            attraxiaoopt(idxiao,i+1,1,mtemp)=attraxiaoopt(idxiao,i+1,1,mtemp)+attraxiao(1,1,mtemp+1)
         endif
      enddo
   enddo

end subroutine PSattraopt


subroutine SPattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do i=1,3
      do idxiao=1,3
         attraxiaoopt(idxiao,1,i+1,mtemp)=(PP(i)-BB(i))*(attraxiaoopt(idxiao,1,1,mtemp))- &
               (PP(i)-CC(i))*(attraxiaoopt(idxiao,1,1,mtemp+1))
         if(idxiao.eq.i)then
            attraxiaoopt(idxiao,1,i+1,mtemp)=attraxiaoopt(idxiao,1,i+1,mtemp)+attraxiao(1,1,mtemp+1)
         endif
      enddo
   enddo

end subroutine SPattraopt


subroutine PPattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=1,3
      do jtemp=1,3
         do idxiao=1,3
            attraxiaoopt(idxiao,itemp+1,jtemp+1,mtemp)=(PP(jtemp)-BB(jtemp))* &
                  (attraxiaoopt(idxiao,itemp+1,1,mtemp))- &
                  (PP(jtemp)-CC(jtemp))*(attraxiaoopt(idxiao,itemp+1,1,mtemp+1))
            if(itemp.eq.jtemp)then
               attraxiaoopt(idxiao,itemp+1,jtemp+1,mtemp)=attraxiaoopt(idxiao,itemp+1,jtemp+1,mtemp)+ &
                     0.5d0/g*(attraxiaoopt(idxiao,1,1,mtemp)-attraxiaoopt(idxiao,1,1,mtemp+1))
            endif
            if(idxiao.eq.jtemp)then
               attraxiaoopt(idxiao,itemp+1,jtemp+1,mtemp)=attraxiaoopt(idxiao,itemp+1,jtemp+1,mtemp)+ &
                     attraxiao(itemp+1,1,mtemp+1)
            endif
         enddo
      enddo
   enddo

end subroutine PPattraopt


subroutine DSattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=5,10
      do idxiao=1,3
         a(1)=Mcal(1,itemp)
         a(2)=Mcal(2,itemp)
         a(3)=Mcal(3,itemp)
         do j=1,3
            if(Mcal(j,itemp).ne.0)then
               a(j)=a(j)-1
               itempnew=trans(a(1),a(2),a(3))
               attraxiaoopt(idxiao,itemp,1,mtemp)=(PP(j)-AA(j))*(attraxiaoopt(idxiao,itempnew,1,mtemp))- &
                     (PP(j)-CC(j))*(attraxiaoopt(idxiao,itempnew,1,mtemp+1))
               if(Mcal(j,itemp).eq.2)then
                  attraxiaoopt(idxiao,itemp,1,mtemp)=attraxiaoopt(idxiao,itemp,1,mtemp)+ &
                        0.5d0/g*(attraxiaoopt(idxiao,1,1,mtemp)-attraxiaoopt(idxiao,1,1,mtemp+1))
               endif
               if(idxiao.eq.j)then
                  attraxiaoopt(idxiao,itemp,1,mtemp)=attraxiaoopt(idxiao,itemp,1,mtemp)+ &
                        attraxiao(itempnew,1,mtemp+1)
               endif
               goto 111
            endif
         enddo
         111     continue
      enddo
   enddo

end subroutine DSattraopt


subroutine SDattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=5,10
      do idxiao=1,3
         a(1)=Mcal(1,itemp)
         a(2)=Mcal(2,itemp)
         a(3)=Mcal(3,itemp)
         do j=1,3
            if(Mcal(j,itemp).ne.0)then
               a(j)=a(j)-1
               itempnew=trans(a(1),a(2),a(3))
               attraxiaoopt(idxiao,1,itemp,mtemp)=(PP(j)-BB(j))*(attraxiaoopt(idxiao,1,itempnew,mtemp))- &
                     (PP(j)-CC(j))*(attraxiaoopt(idxiao,1,itempnew,mtemp+1))
               if(Mcal(j,itemp).eq.2)then
                  attraxiaoopt(idxiao,1,itemp,mtemp)=attraxiaoopt(idxiao,1,itemp,mtemp)+ &
                        0.5d0/g*(attraxiaoopt(idxiao,1,1,mtemp)-attraxiaoopt(idxiao,1,1,mtemp+1))
               endif
               if(idxiao.eq.j)then
                  attraxiaoopt(idxiao,1,itemp,mtemp)=attraxiaoopt(idxiao,1,itemp,mtemp)+ &
                        attraxiao(1,itempnew,mtemp+1)
               endif
               goto 111
            endif
         enddo
         111     continue
      enddo
   enddo

end subroutine SDattraopt


subroutine DPattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=5,10
      do i=2,4
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  attraxiaoopt(idxiao,jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiaoopt(idxiao,jtemp,1,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,jtemp,1,mtemp+1))
                  if(a(jj).ne.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*(Mcal(jj,jtemp))*(attraxiaoopt(idxiao,itempnew,1,mtemp)- &
                           attraxiaoopt(idxiao,itempnew,1,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           attraxiao(jtemp,1,mtemp+1)
                  endif
                  goto 111
               endif
            enddo

            111        continue
         enddo
      enddo
   enddo

end subroutine DPattraopt



subroutine PDattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=5,10
      do i=2,4
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  attraxiaoopt(idxiao,i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiaoopt(idxiao,1,jtemp,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,1,jtemp,mtemp+1))
                  if(a(jj).ne.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           0.5d0/g*(Mcal(jj,jtemp))*(attraxiaoopt(idxiao,1,itempnew,mtemp)- &
                           attraxiaoopt(idxiao,1,itempnew,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           attraxiao(1,jtemp,mtemp+1)
                  endif
                  goto 111
               endif
            enddo

            111        continue
         enddo
      enddo
   enddo

End subroutine PDattraopt


subroutine DDattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=5,10
      do i=5,10
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            Axiao(1)=Mcal(1,i)
            Axiao(2)=Mcal(2,i)
            Axiao(3)=Mcal(3,i)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  Axiao(jj)=Axiao(jj)-1
                  ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiaoopt(idxiao,jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiaoopt(idxiao,jtemp,ixiao,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,jtemp,ixiao,mtemp+1))
                  if(Mcal(jj,i).gt.1)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*(Mcal(jj,i)-1)*(attraxiaoopt(idxiao,jtemp,1,mtemp)- &
                           attraxiaoopt(idxiao,jtemp,1,mtemp+1))
                  endif

                  if(a(jj).gt.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*Mcal(jj,jtemp)*(attraxiaoopt(idxiao,itempnew,ixiao,mtemp)- &
                           attraxiaoopt(idxiao,itempnew,ixiao,mtemp+1))
                  endif

                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           attraxiao(jtemp,ixiao,mtemp+1)
                  endif

                  goto 111
               endif
            enddo

            111        continue

         enddo
      enddo
   enddo

End subroutine DDattraopt

subroutine FSattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=11,20
      do idxiao=1,3
         a(1)=Mcal(1,itemp)
         a(2)=Mcal(2,itemp)
         a(3)=Mcal(3,itemp)
         do j=1,3
            if(Mcal(j,itemp).ne.0)then
               a(j)=Mcal(j,itemp)-1
               itempnew=trans(a(1),a(2),a(3))
               attraxiaoopt(idxiao,itemp,1,mtemp)=(PP(j)-AA(j))*(attraxiaoopt(idxiao,itempnew,1,mtemp))- &
                     (PP(j)-CC(j))*(attraxiaoopt(idxiao,itempnew,1,mtemp+1))
               if(Mcal(j,itemp).gt.1)then
                  a(j)=Mcal(j,itemp)-2
                  inewtemp=trans(a(1),a(2),a(3))
                  attraxiaoopt(idxiao,itemp,1,mtemp)=attraxiaoopt(idxiao,itemp,1,mtemp)+ &
                        0.5d0/g*(Mcal(j,itemp)-1)*(attraxiaoopt(idxiao,inewtemp,1,mtemp)- &
                        attraxiaoopt(idxiao,inewtemp,1,mtemp+1))
               endif
               if(idxiao.eq.j)then
                  attraxiaoopt(idxiao,itemp,1,mtemp)=attraxiaoopt(idxiao,itemp,1,mtemp)+ &
                        attraxiao(itempnew,1,mtemp+1)
               endif
               goto 111
            endif
         enddo
         111     continue
      enddo
   enddo

end subroutine FSattraopt


subroutine SFattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do itemp=11,20
      do idxiao=1,3
         a(1)=Mcal(1,itemp)
         a(2)=Mcal(2,itemp)
         a(3)=Mcal(3,itemp)
         do j=1,3
            if(Mcal(j,itemp).ne.0)then
               a(j)=Mcal(j,itemp)-1
               itempnew=trans(a(1),a(2),a(3))
               attraxiaoopt(idxiao,1,itemp,mtemp)=(PP(j)-BB(j))*(attraxiaoopt(idxiao,1,itempnew,mtemp))- &
                     (PP(j)-CC(j))*(attraxiaoopt(idxiao,1,itempnew,mtemp+1))
               if(Mcal(j,itemp).gt.1)then
                  a(j)=Mcal(j,itemp)-2
                  inewtemp=trans(a(1),a(2),a(3))
                  attraxiaoopt(idxiao,1,itemp,mtemp)=attraxiaoopt(idxiao,1,itemp,mtemp)+ &
                        0.5d0/g*(Mcal(j,itemp)-1)*(attraxiaoopt(idxiao,1,inewtemp,mtemp)- &
                        attraxiaoopt(idxiao,1,inewtemp,mtemp+1))
               endif
               if(idxiao.eq.j)then
                  attraxiaoopt(idxiao,1,itemp,mtemp)=attraxiaoopt(idxiao,1,itemp,mtemp)+ &
                        attraxiao(1,itempnew,mtemp+1)
               endif
               goto 111
            endif
         enddo
         111     continue
      enddo
   enddo

end subroutine SFattraopt


subroutine FPattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=2,4
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  attraxiaoopt(idxiao,jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiaoopt(idxiao,jtemp,1,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,jtemp,1,mtemp+1))
                  if(a(jj).ne.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*(Mcal(jj,jtemp))*(attraxiaoopt(idxiao,itempnew,1,mtemp)- &
                           attraxiaoopt(idxiao,itempnew,1,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           attraxiao(jtemp,1,mtemp+1)
                  endif

                  goto 111
               endif
            enddo

            111        continue

         enddo
      enddo
   enddo

end subroutine FPattraopt


subroutine PFattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=2,4
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  attraxiaoopt(idxiao,i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiaoopt(idxiao,1,jtemp,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,1,jtemp,mtemp+1))
                  if(a(jj).ne.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           0.5d0/g*(Mcal(jj,jtemp))*(attraxiaoopt(idxiao,1,itempnew,mtemp)- &
                           attraxiaoopt(idxiao,1,itempnew,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           attraxiao(1,jtemp,mtemp+1)
                  endif

                  goto 111
               endif
            enddo

            111        continue

         enddo
      enddo
   enddo

End subroutine PFattraopt


subroutine FDattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=5,10
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            Axiao(1)=Mcal(1,i)
            Axiao(2)=Mcal(2,i)
            Axiao(3)=Mcal(3,i)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  Axiao(jj)=Mcal(jj,i)-1
                  ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiaoopt(idxiao,jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiaoopt(idxiao,jtemp,ixiao,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,jtemp,ixiao,mtemp+1))
                  if(Mcal(jj,i).gt.1)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*(attraxiaoopt(idxiao,jtemp,1,mtemp)-attraxiaoopt(idxiao,jtemp,1,mtemp+1))
                  endif

                  if(a(jj).gt.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*Mcal(jj,jtemp)*(attraxiaoopt(idxiao,itempnew,ixiao,mtemp)- &
                           attraxiaoopt(idxiao,itempnew,ixiao,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           attraxiao(jtemp,ixiao,mtemp+1)
                  endif

                  goto 111
               endif
            enddo

            111        continue

         enddo
      enddo
   enddo

End subroutine FDattraopt

subroutine DFattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=5,10
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            Axiao(1)=Mcal(1,i)
            Axiao(2)=Mcal(2,i)
            Axiao(3)=Mcal(3,i)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  Axiao(jj)=Mcal(jj,i)-1
                  ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiaoopt(idxiao,i,jtemp,mtemp)=(PP(jj)-AA(jj))*(attraxiaoopt(idxiao,ixiao,jtemp,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,ixiao,jtemp,mtemp+1))
                  if(Mcal(jj,i).gt.1)then
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           0.5d0/g*(attraxiaoopt(idxiao,1,jtemp,mtemp)-attraxiaoopt(idxiao,1,jtemp,mtemp+1))
                  endif

                  if(a(jj).gt.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           0.5d0/g*Mcal(jj,jtemp)*(attraxiaoopt(idxiao,ixiao,itempnew,mtemp)- &
                           attraxiaoopt(idxiao,ixiao,itempnew,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,i,jtemp,mtemp)=attraxiaoopt(idxiao,i,jtemp,mtemp)+ &
                           attraxiao(ixiao,jtemp,mtemp+1)
                  endif

                  goto 111
               endif
            enddo

            111        continue

         enddo
      enddo
   enddo

End subroutine DFattraopt


subroutine FFattraopt(mtemp)
   use allmod
   Implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3),Axiao(3)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   do jtemp=11,20
      do i=11,20
         do idxiao=1,3
            a(1)=Mcal(1,jtemp)
            a(2)=Mcal(2,jtemp)
            a(3)=Mcal(3,jtemp)
            Axiao(1)=Mcal(1,i)
            Axiao(2)=Mcal(2,i)
            Axiao(3)=Mcal(3,i)
            do jj=1,3
               if(Mcal(jj,i).ne.0)then
                  Axiao(jj)=Mcal(jj,i)-1
                  ixiao=trans(Axiao(1),Axiao(2),Axiao(3))
                  attraxiaoopt(idxiao,jtemp,i,mtemp)=(PP(jj)-BB(jj))*(attraxiaoopt(idxiao,jtemp,ixiao,mtemp))- &
                        (PP(jj)-CC(jj))*(attraxiaoopt(idxiao,jtemp,ixiao,mtemp+1))
                  if(Mcal(jj,i).gt.1)then
                     Axiao(jj)=Mcal(jj,i)-2
                     inewtemp=trans(Axiao(1),Axiao(2),Axiao(3))
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*(Mcal(jj,i)-1)*(attraxiaoopt(idxiao,jtemp,inewtemp,mtemp)- &
                           attraxiaoopt(idxiao,jtemp,inewtemp,mtemp+1))
                  endif

                  if(a(jj).gt.0)then
                     a(jj)=a(jj)-1
                     itempnew=trans(a(1),a(2),a(3))
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           0.5d0/g*Mcal(jj,jtemp)*(attraxiaoopt(idxiao,itempnew,ixiao,mtemp)- &
                           attraxiaoopt(idxiao,itempnew,ixiao,mtemp+1))
                  endif
                  if(idxiao.eq.jj)then
                     attraxiaoopt(idxiao,jtemp,i,mtemp)=attraxiaoopt(idxiao,jtemp,i,mtemp)+ &
                           attraxiao(jtemp,ixiao,mtemp+1)
                  endif

                  goto 111
               endif
            enddo

            111        continue

         enddo
      enddo
   enddo

End subroutine FFattraopt


