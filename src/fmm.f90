#include "util.fh"
! prepare all combinations SS, PS, SP, PP, Xiao HE 01/14/2008
! Be careful of coeff
! nuclearspdf.f90
           subroutine fmmone(Ips,Jps,IIsh,JJsh,NIJ1, &
               Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,iatom,constant2,a,b,xdistance)
           use allmod

           implicit double precision(a-h,o-z)

           integer a(3),b(3)
           double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g  
           double precision AA(3),BB(3),CC(3),PP(3)
           double precision fmmonearray(0:2,0:2,1:2)
!           common /xiaofmm/fmmonearray,AA,BB,CC,PP,g
          
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
!                                  call overlapzero(a,b)
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
!                                  call overlapzero(a,b)
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
!                                case(3)
!                                  call overlapthree
!                                case(4)
!                                  call overlapfour
                               
                                end select                                 

                                quick_qm_struct%o(JJJ,III)=quick_qm_struct%o(JJJ,III)+ &
                                        Xconstant*quick_basis%cons(III)*quick_basis%cons(JJJ)*valfmmone
                              enddo
                            enddo

                         enddo
                        enddo


               End
