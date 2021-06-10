#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/14/2021                            !
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
! This module contains subroutines related to fast multipole method.  ! 
!---------------------------------------------------------------------!

module quick_fmm_module

  implicit double precision(a-h,o-z)
  private  
  public :: fmmone

contains

  subroutine fmmone(Ips,Jps,IIsh,JJsh,NIJ1, &
      Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,iatom,constant2,a,b,xdistance)
  use allmod

  implicit double precision(a-h,o-z)

  integer a(3),b(3)
  double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g  
  double precision AA(3),BB(3),CC(3),PP(3)
  double precision fmmonearray(0:2,0:2,1:2)
   common /xiaofmm/fmmonearray,AA,BB,CC,PP,g
 
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

       xdis3=xdistance**3.0d0
       xdis5=xdis3*xdis3/xdistance               

               do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh) 
                X1temp=constant2*quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
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

                   Nvalue=Iang+Jang

                   do III=III1,III2
                    itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
                     do JJJ=max(III,JJJ1),JJJ2
                      itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))

                       select case (Nvalue)

                       case(0)
                         call overlapzero(a,b,fmmonearray)
                         valfmmone=fmmonearray(0,0,1)*xdistance
                       case(1)
                         call overlapone(a,b,quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III), &
                              quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ),Ax,Ay,Az,Bx, &
                              By,Bz,fmmonearray)
                          !call overlapzero(a,b)
                         valfmmone=fmmonearray(0,0,1)*xdistance
                         valfmmone=valfmmone+fmmonearray(1,0,1)*xdis3* &
                                   (Cz-Pz)
                         valfmmone=valfmmone+fmmonearray(1,1,1)*xdis3* &
                                   (Cx-Px)
                         valfmmone=valfmmone+fmmonearray(1,1,2)*xdis3* &
                                   (Cy-Py)
                       case(2)
                         call overlaptwo(a,b,quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III), &
                              quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ),Ax,Ay,Az,Bx, &
                              By,Bz,fmmonearray)
                         ! call overlapzero(a,b)
                         valfmmone=fmmonearray(0,0,1)*xdistance
                         valfmmone=valfmmone+fmmonearray(1,0,1)*xdis3* &
                                   (Cz-Pz)
                         valfmmone=valfmmone+fmmonearray(1,1,1)*xdis3* &
                                   (Cx-Px)
                         valfmmone=valfmmone+fmmonearray(1,1,2)*xdis3* &
                                   (Cy-Py)
                         valfmmone=valfmmone+fmmonearray(2,0,1)*xdis5* &
                                   0.5d0*(2.0d0*(Cz-Pz)**2.0d0-(Cx-Px)**2.0d0- &
                                   (Cy-Py)**2.0d0)
                         valfmmone=valfmmone+fmmonearray(2,1,1)*xdis5* &
                                   dsqrt(3.0d0)*(Cx-Px)*(Cz-Pz)
                         valfmmone=valfmmone+fmmonearray(2,1,2)*xdis5* &
                                   dsqrt(3.0d0)*(Cy-Py)*(Cz-Pz)
                         valfmmone=valfmmone+fmmonearray(2,2,1)*xdis5* &
                                   dsqrt(0.75d0)*((Cx-Px)*(Cx-Px)-(Cy-Py)*(Cy-Py))
                         valfmmone=valfmmone+fmmonearray(2,2,2)*xdis5* &
                                   dsqrt(3.0d0)*(Cx-Px)*(Cy-Py)
                        !case(3)
                        !  call overlapthree
                        !case(4)
                        !  call overlapfour
                      
                       end select                                 

                       quick_qm_struct%o(JJJ,III)=quick_qm_struct%o(JJJ,III)+ &
                               Xconstant*quick_basis%cons(III)*quick_basis%cons(JJJ)*valfmmone
                     enddo
                   enddo

                enddo
               enddo


  end subroutine fmmone


subroutine overlapone(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,fmmtemparray)
   use quick_constants_module
   implicit double precision(a-h,o-z)

   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   double precision fmmtemparray(0:2,0:2,1:2)

   ! The purpose of this subroutine is to calculate the overlap between
   ! two normalized gaussians. i,j and k are the x,y,
   ! and z exponents for the gaussian with exponent a, and ii,jj, and kk
   ! have the same order for b.


   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset overlap to 0.

   ! If it is not zero, construct P and g values.  The gaussian product
   ! theory states the product of two s gaussians on centers A and B
   ! with exponents a and b forms a new s gaussian on P with exponent
   ! g.  (g comes from gamma, as is "alpha,beta, gamma" and P comes
   ! from "Product." Also needed are the PA differences.

   g = a+b
   Px = (a*Ax + b*Bx)/g
   Py = (a*Ay + b*By)/g
   Pz = (a*Az + b*Bz)/g

   PAx= Px-Ax
   PAy= Py-Ay
   PAz= Pz-Az
   PBx= Px-Bx
   PBy= Py-By
   PBz= Pz-Bz

   Y1=0.0d0
   Y2=0.0d0
   Y3=0.0d0
   Y4=0.0d0

   ! There is also a few factorials that are needed in the integral many
   ! times.  Calculate these as well.

   xnumfact=fact(i)*fact(ii)*fact(j)*fact(jj)*fact(k)*fact(kk)

   ! Now start looping over i,ii,j,jj,k,kk to form all the required elements.

   do iloop=0,i
      do iiloop=0,ii
         do jloop=0,j
            do jjloop=0,jj
               do kloop=0,k
                  do kkloop=0,kk
                     ix=iloop+iiloop
                     jy=jloop+jjloop
                     kz=kloop+kkloop

                     ! Continue calculating the elements.  The next elements arise from the
                     ! different angular momentums portion of the GPT.

                     element00=PAx**(i-iloop) &
                           *PBx**(ii-iiloop) &
                           *PAy**(j-jloop) &
                           *PBy**(jj-jjloop) &
                           *PAz**(k-kloop) &
                           *PBz**(kk-kkloop) &
                           *xnumfact &
                           /(fact(iloop)*fact(iiloop)* &
                           fact(jloop)*fact(jjloop)* &
                           fact(kloop)*fact(kkloop)* &
                           fact(i-iloop)*fact(ii-iiloop)* &
                           fact(j-jloop)*fact(jj-jjloop)* &
                           fact(k-kloop)*fact(kk-kkloop))

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 50
                     endif


                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y1=Y1+element
                     50 continue
                     !                            fmmtemparray(0,0,1) = fmmtemparray(0,0,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy))*(1+(-1)**(kz+1))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 60
                     endif
                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz-1)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,(kz+1)/2
                        element = element * (dble(kz+1)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y2=Y2+element
                     60 continue
                     !                            fmmtemparray(1,0,1) = fmmtemparray(1,0,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix+1))*(1+(-1)**(jy))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 70
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix -1 - jy - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,(ix+1)/2
                        element = element * (dble(ix+1)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y3=Y3+element
                     70 continue
                     !                            fmmtemparray(1,1,1) = fmmtemparray(1,1,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy+1))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 80
                     endif
                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy -1 - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,(jy+1)/2
                        element = element * (dble(jy+1)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y4=Y4+element
                     80 continue
                     !                            fmmtemparray(1,1,2) = fmmtemparray(1,1,2) + element


                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo

   ! The final step is multiplying in the K factor (from the gpt)

   !    overlap = overlap* Exp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 &
         !    + (Az - Bz)**2.d0))/(a + b)))

   fmmtemparray(0,0,1)=Y1
   fmmtemparray(1,0,1)=Y2
   fmmtemparray(1,1,1)=Y3
   fmmtemparray(1,1,2)=Y4

   100 continue
   return
end subroutine overlapone


subroutine overlaptwo(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx, &
      By,Bz,fmmonearray)
   use quick_constants_module
   implicit double precision(a-h,o-z)

   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   double precision fmmonearray(0:2,0:2,1:2)
   !           common /xiaofmm/fmmonearray,AA,BB,CC,PP,g

   ! The purpose of this subroutine is to calculate the overlap between
   ! two normalized gaussians. i,j and k are the x,y,
   ! and z exponents for the gaussian with exponent a, and ii,jj, and kk
   ! have the same order for b.

   ! Constants:

   !    pi=3.1415926535897932385
   !    pito3half=5.568327996831707845284817982118835702014
   !    pito3half = pi**(1.5)

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset overlap to 0.

   ! If it is not zero, construct P and g values.  The gaussian product
   ! theory states the product of two s gaussians on centers A and B
   ! with exponents a and b forms a new s gaussian on P with exponent
   ! g.  (g comes from gamma, as is "alpha,beta, gamma" and P comes
   ! from "Product." Also needed are the PA differences.

   g = a+b
   Px = (a*Ax + b*Bx)/g
   Py = (a*Ay + b*By)/g
   Pz = (a*Az + b*Bz)/g

   PAx= Px-Ax
   PAy= Py-Ay
   PAz= Pz-Az
   PBx= Px-Bx
   PBy= Py-By
   PBz= Pz-Bz

   Y1=0.0d0
   Y2=0.0d0
   Y3=0.0d0
   Y4=0.0d0
   Y5=0.0d0
   Y6=0.0d0
   Y7=0.0d0
   Y8=0.0d0
   Y9=0.0d0

   ! There is also a few factorials that are needed in the integral many
   ! times.  Calculate these as well.

   xnumfact=fact(i)*fact(ii)*fact(j)*fact(jj)*fact(k)*fact(kk)

   ! Now start looping over i,ii,j,jj,k,kk to form all the required elements.

   do iloop=0,i
      do iiloop=0,ii
         do jloop=0,j
            do jjloop=0,jj
               do kloop=0,k
                  do kkloop=0,kk
                     ix=iloop+iiloop
                     jy=jloop+jjloop
                     kz=kloop+kkloop

                     ! Continue calculating the elements.  The next elements arise from the
                     ! different angular momentums portion of the GPT.

                    element00=PAx**(i-iloop) &
                           *PBx**(ii-iiloop) &
                           *PAy**(j-jloop) &
                           *PBy**(jj-jjloop) &
                           *PAz**(k-kloop) &
                           *PBz**(kk-kkloop) &
                           *xnumfact &
                           /(fact(iloop)*fact(iiloop)* &
                           fact(jloop)*fact(jjloop)* &
                           fact(kloop)*fact(kkloop)* &
                           fact(i-iloop)*fact(ii-iiloop)* &
                           fact(j-jloop)*fact(jj-jjloop)* &
                           fact(k-kloop)*fact(kk-kkloop))

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 50
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y1=Y1+element
                     50 continue
                     !                            fmmonearray(0,0,1) = fmmonearray(0,0,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy))*(1+(-1)**(kz+1))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 60
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz-1)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2


                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,(kz+1)/2
                        element = element * (dble(kz+1)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y2=Y2+element
                     60 continue
                     !                            fmmonearray(1,0,1) = fmmonearray(1,0,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix+1))*(1+(-1)**(jy))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 70
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix -1 - jy - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,(ix+1)/2
                        element = element * (dble(ix+1)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y3=Y3+element
                     70 continue
                     !                            fmmonearray(1,1,1) = fmmonearray(1,1,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy+1))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 80
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy -1 - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,(jy+1)/2
                        element = element * (dble(jy+1)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y4=Y4+element
                     80 continue
                     !                            fmmonearray(1,1,2) = fmmonearray(1,1,2) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy))*(1+(-1)**(kz+2))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 90
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz -2)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,(kz+2)/2
                        element = element * (dble(kz+2)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y5=Y5+element
                     90 continue
                     !                            fmmonearray(2,0,1) = fmmonearray(2,0,1) + element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix+2))*(1+(-1)**(jy))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 100
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix -2 - jy - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                    do iG=1,(ix+2)/2
                        element = element * (dble(ix+2)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y5=Y5 - 0.5d0*element
                     Y8=Y8 + dsqrt(0.75d0)*element
                     100 continue
                     !                            fmmonearray(2,0,1) = fmmonearray(2,0,1) - 0.5d0*element
                     !                            fmmonearray(2,2,1) = fmmonearray(2,2,1) + dsqrt(0.75d0)*element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy+2))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 110
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy -2 - kz)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                    do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,(jy+2)/2
                        element = element * (dble(jy+2)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y5=Y5 - 0.5d0*element
                     Y8=Y8 - dsqrt(0.75d0)*element

                     110 continue
                     !                            fmmonearray(2,0,1) = fmmonearray(2,0,1) - 0.5d0*element
                     !                            fmmonearray(2,2,1) = fmmonearray(2,2,1) - dsqrt(0.75d0)*element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix+1))*(1+(-1)**(jy))*(1+(-1)**(kz+1))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 120
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz -2)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,(ix+1)/2
                        element = element * (dble(ix+1)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,jy/2
                        element = element * (dble(jy)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,(kz+1)/2
                        element = element * (dble(kz+1)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y6=Y6 + dsqrt(3.0d0)*element
                     120 continue
                     !                            fmmonearray(2,1,1) = fmmonearray(2,1,1) + dsqrt(3.0d0)*element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix))*(1+(-1)**(jy+1))*(1+(-1)**(kz+1))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 130
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz -2)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2


                     do iG=1,ix/2
                        element = element * (dble(ix)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,(jy+1)/2
                        element = element * (dble(jy+1)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,(kz+1)/2
                        element = element * (dble(kz+1)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y7=Y7+dsqrt(3.0d0)*element
                     130 continue
                     !                            fmmonearray(2,1,2) = fmmonearray(2,1,2) + dsqrt(3.0d0)*element

                     ! Check to see if this element is zero.

                     elementtemp=(1+(-1)**(ix+1))*(1+(-1)**(jy+1))*(1+(-1)**(kz))/8
                     if (elementtemp == 0)then
                        element=0.0d0
                        goto 140
                     endif

                     ! The next part arises from the integratation of a gaussian of arbitrary
                     ! angular momentum.

                     element=element00*g**(dble(-3 - ix - jy - kz -2)/2.d0)

                     ! Before the Gamma function code, a quick note. All gamma functions are
                     ! of the form:
                     ! 1
                     ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                     ! 2

                     ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                     ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                     ! just requires a loop and multiplying by Pi^3/2

                     do iG=1,(ix+1)/2
                        element = element * (dble(ix+1)/2.d0-dble(iG) + .5d0)
                     enddo
                     do jG=1,(jy+1)/2
                        element = element * (dble(jy+1)/2.d0-dble(jG) + .5d0)
                     enddo
                     do kG=1,kz/2
                        element = element * (dble(kz)/2.d0-dble(kG) + .5d0)
                     enddo
                     element=element*pito3half

                     ! Now sum the whole thing into the overlap.

                     Y9=Y9 + dsqrt(3.0d0)*element

                     140 continue
                     !                            fmmonearray(2,2,2) = fmmonearray(2,2,2) + dsqrt(3.0d0)*element


                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo

   ! The final step is multiplying in the K factor (from the gpt)

   !    overlap = overlap* Exp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 &
         !    + (Az - Bz)**2.d0))/(a + b)))

   !    100 continue

   fmmonearray(0,0,1)=Y1
   fmmonearray(1,0,1)=Y2
   fmmonearray(1,1,1)=Y3
   fmmonearray(1,1,2)=Y4
   fmmonearray(2,0,1)=Y5
   fmmonearray(2,1,1)=Y6
   fmmonearray(2,1,2)=Y7
   fmmonearray(2,2,1)=Y8
   fmmonearray(2,2,2)=Y9

   return

end subroutine overlaptwo 

subroutine overlapzero(a,b,fmmtemparray)
   use quick_constants_module
   implicit double precision(a-h,o-z)

   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   double precision fmmtemparray(0:2,0:2,1:2)
   !           common /xiaofmm/fmmonearray,AA,BB,CC,PP,g

   !    pito3half=5.568327996831707845284817982118835702014
   !    pito3half = pi**1.5
   g = a+b

   ssoverlap = pito3half*g**(-1.5d0)

   !    ssoverlap = ssoverlap* Exp(-((a*b*((Ax - Bx)**2.d0 + &
         !    (Ay - By)**2.d0 &
         !    + (Az - Bz)**2.d0))/(a + b)))

   fmmtemparray(0,0,1)=ssoverlap

end subroutine overlapzero



end module quick_fmm_module

