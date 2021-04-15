#include "util.fh"
!
!	quick_one_electron_integral.f90
!	new_quick
!
!	Created by Yipu Miao on 4/12/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   this subroutine is a collection of one-electron integeral
!   subroutine inventory:
!           FullX       :       calculate transformation matrix X and overlap matrix S
!           ekinetic    :       calculate kinetic energy
!           overlap     :       calculate overlap matrix element
!           ssoverlap   :
!           overlapone, overlaptwo, overlapzero
!                       :       overlap for FMM
!           repulsion   :       calculate repulsion element (only used in unrestricted)



! Ed Brothers. November 2, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
double precision function repulsion(gi,gj,gk,gl,pos1,pos2,pos3,pos4) result(repint)
   use quick_gaussian_class_module
   implicit none

   type(gaussian) :: gi,gj,gk,gl
   double precision, dimension(3) :: pos1,pos2,pos3,pos4
   double precision :: repulsion_prim
   integer :: i,j,k,l

   repint = 0.d0

   do I=1,gi%ncontract
      do J=1,gj%ncontract
         do K=1,gk%ncontract
            do L=1,gl%ncontract
               repint = repint+ &
                        gi%dcoeff(i)*gj%dcoeff(j)*gk%dcoeff(k)*gl%dcoeff(l)* &
                        (repulsion_prim(gi%aexp(i), gj%aexp(j),&
                                        gk%aexp(k), gl%aexp(l), &
                                        gi%itype(1),gi%itype(2),gi%itype(3), &
                                        gj%itype(1),gj%itype(2),gj%itype(3), &
                                        gk%itype(1),gk%itype(2),gk%itype(3), &
                                        gl%itype(1),gl%itype(2),gl%itype(3), &
                                        pos1(1),pos1(2),pos1(3),&
                                        pos2(1),pos2(2),pos2(3),&
                                        pos3(1),pos3(2),pos3(3),&
                                        pos4(1),pos4(2),pos4(3)))
            enddo
         enddo
      enddo
   enddo
end function repulsion


double precision function repulsion_prim(a, b, c, d, &
                                         i,  j,  k,  &
                                         ii, jj, kk, &
                                         i2, j2, k2, &
                                         ii2,jj2,kk2,&
                                         Ax, Ay, Az, &
                                         Bx, By, Bz, &
                                         Cx, Cy, Cz, &
                                         Dx, Dy, Dz) result(repulsion)
   use quick_constants_module
   use quick_overlap_module, only: ssoverlap
   implicit double precision(a-h,o-z)
   dimension aux(0:20)

   ! Variables needed later:

   !    pi = 3.1415926535897932385

   g = a+b
   Px = (a*Ax + b*Bx)/g
   Py = (a*Ay + b*By)/g
   Pz = (a*Az + b*Bz)/g

   h = c+d
   Qx = (c*Cx + d*Dx)/h
   Qy = (c*Cy + d*Dy)/h
   Qz = (c*Cz + d*Dz)/h

   Wx = (g*Px + h*Qx)/(g+h)
   Wy = (g*Py + h*Qy)/(g+h)
   Wz = (g*Pz + h*Qz)/(g+h)

   rho = (g*h)/(g+h)

   PQsquare = (Px-Qx)**2.d0 + (Py -Qy)**2.d0 + (Pz -Qz)**2.d0

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! The first step is generating all the necessary auxillary integrals.
   ! These are:
   ! (0a0b|1/r12|0c0d)^(m) = 2 Sqrt(rho/Pi) (0a||0b)(0c||0d) Fm(rho(Rpq)^2)
   ! The values of m range from 0 to i+j+k+ii+jj+kk.

   T = rho* PQsquare
   Maxm = i+j+k+ii+jj+kk+i2+j2+k2+ii2+jj2+kk2
   call FmT(Maxm,T,aux)
   ! constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz)
   ! .          *overlap(c,d,0,0,0,0,0,0,Cx,Cy,Cz,Dx,Dy,Dz)
   constant = ssoverlap(a,b,Ax,Ay,Az,Bx,By,Bz) &
         *ssoverlap(c,d,Cx,Cy,Cz,Dx,Dy,Dz) &
         * 2.d0 * (rho/Pi)**0.5d0
   do L = 0,maxm
      aux(L) = aux(L)*constant
   enddo

   ! At this point all the auxillary integrals have been calculated.
   ! It is now time to decompose the repulsion integral to it's
   ! auxillary integrals through the recursion scheme.  To do this we use
   ! a recursive function.

   repulsion = reprecurse(i,j,k,ii,jj,kk,i2,j2,k2,ii2,jj2,kk2, &
         0,aux, &
         Ax,Ay,Az,Bx,By,Bz, &
         Cx,Cy,Cz,Dx,Dy,Dz, &
         Px,Py,Pz,Qx,Qy,Qz, &
         Wx,Wy,Wz, &
         g,h,rho)

   return
end function repulsion_prim



! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

double precision recursive function reprecurse(i,j,k,ii,jj,kk, &
      i2,j2,k2,ii2,jj2,kk2, &
      m,aux, &
      Ax,Ay,Az,Bx,By,Bz, &
      Cx,Cy,Cz,Dx,Dy,Dz, &
      Px,Py,Pz,Qx,Qy,Qz, &
      Wx,Wy,Wz, &
      g,h,rho) &
      result(reprec)

   implicit double precision(a-h,o-z)
   dimension iexponents(12),center(21),aux(0:20)

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! If this is one of the auxillary integrals (s||s)^(m), assign value and
   ! return.

   if (i+j+k+ii+jj+kk+i2+j2+k2+ii2+jj2+kk2 == 0) then
      reprec=aux(m)

      ! Otherwise, use the recusion relation from Obara and Siaka.  The first
      ! step is to find the lowest nonzero angular momentum exponent.  This is
      ! because the more exponents equal zero the fewer terms need to be
      ! calculated, and each recursive loop reduces the angular momentum
      ! exponents. This therefore reorders the atoms and sets the exponent
      ! to be reduced.

   else
      iexponents(1) = i
      iexponents(2) = j
      iexponents(3) = k
      iexponents(4) = ii
      iexponents(5) = jj
      iexponents(6) = kk
      iexponents(7) = i2
      iexponents(8) = j2
      iexponents(9) = k2
      iexponents(10) = ii2
      iexponents(11) = jj2
      iexponents(12) = kk2
      center(19)= Wx
      center(20)= Wy
      center(21)= Wz
      ilownum=300
      ilowex=300
      do L=1,12
         if (iexponents(L) < ilowex .AND. iexponents(L) /= 0) then
            ilowex=iexponents(L)
            ilownum=L
         endif
      enddo
      if (ilownum < 4) then
         center(1)=Ax
         center(2)=Ay
         center(3)=Az
         center(4)=Bx
         center(5)=By
         center(6)=Bz
         center(7)=Cx
         center(8)=Cy
         center(9)=Cz
         center(10)=Dx
         center(11)=Dy
         center(12)=Dz
         center(13)=Px
         center(14)=Py
         center(15)=Pz
         center(16)=Qx
         center(17)=Qy
         center(18)=Qz
         elseif (ilownum > 3 .AND. ilownum < 7) then
         center(4)=Ax
         center(5)=Ay
         center(6)=Az
         center(1)=Bx
         center(2)=By
         center(3)=Bz
         center(7)=Cx
         center(8)=Cy
         center(9)=Cz
         center(10)=Dx
         center(11)=Dy
         center(12)=Dz
         center(13)=Px
         center(14)=Py
         center(15)=Pz
         center(16)=Qx
         center(17)=Qy
         center(18)=Qz
         iexponents(4) = i
         iexponents(5) = j
         iexponents(6) = k
         iexponents(1) = ii
         iexponents(2) = jj
         iexponents(3) = kk
         ilownum = ilownum - 3
         elseif (ilownum > 6 .AND. ilownum < 10) then
         center(7)=Ax
         center(8)=Ay
         center(9)=Az
         center(10)=Bx
         center(11)=By
         center(12)=Bz
         center(1)=Cx
         center(2)=Cy
         center(3)=Cz
         center(4)=Dx
         center(5)=Dy
         center(6)=Dz
         center(16)=Px
         center(17)=Py
         center(18)=Pz
         center(13)=Qx
         center(14)=Qy
         center(15)=Qz
         iexponents(1) = i2
         iexponents(2) = j2
         iexponents(3) = k2
         iexponents(4) = ii2
         iexponents(5) = jj2
         iexponents(6) = kk2
         iexponents(7) = i
         iexponents(8) = j
         iexponents(9) = k
         iexponents(10) = ii
         iexponents(11) = jj
         iexponents(12) = kk
         ilownum = ilownum - 6
         temp=g
         g = h
         h = temp
      else
         center(7)=Ax
         center(8)=Ay
         center(9)=Az
         center(10)=Bx
         center(11)=By
         center(12)=Bz
         center(4)=Cx
         center(5)=Cy
         center(6)=Cz
         center(1)=Dx
         center(2)=Dy
         center(3)=Dz
         center(16)=Px
         center(17)=Py
         center(18)=Pz
         center(13)=Qx
         center(14)=Qy
         center(15)=Qz
         iexponents(1) = ii2
         iexponents(2) = jj2
         iexponents(3) = kk2
         iexponents(4) = i2
         iexponents(5) = j2
         iexponents(6) = k2
         iexponents(7) = i
         iexponents(8) = j
         iexponents(9) = k
         iexponents(10) = ii
         iexponents(11) = jj
         iexponents(12) = kk
         ilownum = ilownum - 9
         temp=g
         g = h
         h = temp
      endif

      ! BUG FIX:  If you pass values to a funtion, and these values are
      ! modified in the course of the function they return changed. EG
      ! FUNCTION TEST(I)
      ! TEST=I+1

      ! If I=1,TEST=2 and on return I=1.

      ! On the other hand:
      ! FUNCTION TEST(I)
      ! I=I+1
      ! TEST=I

      ! If I=1,TEST=2 and on return I=2.

      ! This phenomenon caused the problem that the switching of g and h
      ! carried out by one call to the recursive function would be preserved
      ! to the next functional call.
      ! E.G. Assume that g and h are not equal for this example.
      ! (px px|px px) will produce two (s s |px px) functions (m=0 and m=1).
      ! The (s s |px px) (m=0) call will swap the g and h as it should and
      ! the g and h stay swapped the whole way down through the recursion.
      ! The next function to be solved will be the (s s |px px) (m=1).  When
      ! it starts down through the recursion the g and h are still swapped
      ! from the previous recusion.  The code therefore swaps them again
      ! (or unswaps them if you prefer) which puts them in the wrong place.
      ! Thus we need to set up a dummy set of variables (gpass and hpass) to
      ! avoid this.

      ! If you've read this far and it is unitelligible, simply think of it as
      ! passing a copy to keep the original pristine.

      gpass=g
      hpass=h

      ! The first step is lowering the orbital exponent by one.
      ! This is what actually makes this method work:  The idea
      ! that all electron repulsion integrals can be expressed
      ! as a sum of electron repulsion integrals of gtfs of lower
      ! angular momentum.

      iexponents(ilownum) = iexponents(ilownum)-1

      ! At this point, calculate the the terms of the recusion
      ! relation.

      reprec=0.d0
      PA = center(12+ilownum)-center(ilownum)
      if (PA /= 0) reprec = reprec + PA * &
            reprecurse(iexponents(1),iexponents(2), &
            iexponents(3),iexponents(4), &
            iexponents(5),iexponents(6), &
            iexponents(7),iexponents(8), &
            iexponents(9),iexponents(10), &
            iexponents(11),iexponents(12), &
            m,aux, &
            center(1),center(2),center(3), &
            center(4),center(5),center(6), &
            center(7),center(8),center(9), &
            center(10),center(11),center(12), &
            center(13),center(14),center(15), &
            center(16),center(17),center(18), &
            center(19),center(20),center(21), &
            gpass,hpass,rho)
      gpass = g
      hpass = h

      WP = center(18+ilownum)-center(12+ilownum)
      if (WP /= 0) reprec = reprec + WP * &
            reprecurse(iexponents(1),iexponents(2), &
            iexponents(3),iexponents(4), &
            iexponents(5),iexponents(6), &
            iexponents(7),iexponents(8), &
            iexponents(9),iexponents(10), &
            iexponents(11),iexponents(12), &
            m+1,aux, &
            center(1),center(2),center(3), &
            center(4),center(5),center(6), &
            center(7),center(8),center(9), &
            center(10),center(11),center(12), &
            center(13),center(14),center(15), &
            center(16),center(17),center(18), &
            center(19),center(20),center(21), &
            gpass,hpass,rho)
      gpass = g
      hpass = h



      if (iexponents(ilownum) /= 0) then
         coeff = dble(iexponents(ilownum))/(2.d0*g)
         iexponents(ilownum) = iexponents(ilownum)-1
         reprec = reprec + coeff*( &
               reprecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               iexponents(7),iexponents(8), &
               iexponents(9),iexponents(10), &
               iexponents(11),iexponents(12), &
               m,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12), &
               center(13),center(14),center(15), &
               center(16),center(17),center(18), &
               center(19),center(20),center(21), &
               gpass,hpass,rho))
         gpass = g
         hpass = h


         reprec = reprec + coeff*( &
               -(rho/g)*reprecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               iexponents(7),iexponents(8), &
               iexponents(9),iexponents(10), &
               iexponents(11),iexponents(12), &
               m+1,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12), &
               center(13),center(14),center(15), &
               center(16),center(17),center(18), &
               center(19),center(20),center(21), &
               gpass,hpass,rho))
         iexponents(ilownum) = iexponents(ilownum)+1
         gpass = g
         hpass = h
      endif

      if (iexponents(ilownum+3) /= 0) then
         coeff = dble(iexponents(ilownum+3))/(2.d0*g)
         iexponents(ilownum+3) = iexponents(ilownum+3)-1
         reprec = reprec + coeff*( &
               reprecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               iexponents(7),iexponents(8), &
               iexponents(9),iexponents(10), &
               iexponents(11),iexponents(12), &
               m,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12), &
               center(13),center(14),center(15), &
               center(16),center(17),center(18), &
               center(19),center(20),center(21), &
               gpass,hpass,rho))
         gpass=g
         hpass=h
         reprec = reprec + coeff*( &
               -(rho/g)*reprecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               iexponents(7),iexponents(8), &
               iexponents(9),iexponents(10), &
               iexponents(11),iexponents(12), &
               m+1,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12), &
               center(13),center(14),center(15), &
               center(16),center(17),center(18), &
               center(19),center(20),center(21), &
               gpass,hpass,rho))
         iexponents(ilownum+3) = iexponents(ilownum+3)+1
         gpass=g
         hpass=h
      endif

      if (iexponents(ilownum+6) /= 0) then
         coeff = dble(iexponents(ilownum+6))/(2.d0*(g+h))
         iexponents(ilownum+6) = iexponents(ilownum+6)-1
         reprec = reprec + coeff*( &
               reprecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               iexponents(7),iexponents(8), &
               iexponents(9),iexponents(10), &
               iexponents(11),iexponents(12), &
               m+1,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12), &
               center(13),center(14),center(15), &
               center(16),center(17),center(18), &
               center(19),center(20),center(21), &
               gpass,hpass,rho) &
               )
         iexponents(ilownum+6) = iexponents(ilownum+6)+1
         gpass=g
         hpass=h
      endif

      if (iexponents(ilownum+9) /= 0) then
         coeff = dble(iexponents(ilownum+9))/(2.d0*(g+h))
         iexponents(ilownum+9) = iexponents(ilownum+9)-1
         reprec = reprec + coeff*( &
               reprecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               iexponents(7),iexponents(8), &
               iexponents(9),iexponents(10), &
               iexponents(11),iexponents(12), &
               m+1,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12), &
               center(13),center(14),center(15), &
               center(16),center(17),center(18), &
               center(19),center(20),center(21), &
               gpass,hpass,rho) &
               )
         iexponents(ilownum+9) = iexponents(ilownum+9)+1
         gpass=g
         hpass=h
      endif


   endif

   return
end

! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
double precision function attraction(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az, &
      Bx,By,Bz,Cx,Cy,Cz,Z)
   use quick_constants_module
   use quick_overlap_module, only: gpt, overlap
   implicit double precision(a-h,o-z)
   dimension aux(0:20)
   double precision g_table(200)

   ! Variables needed later:
   !    pi=3.1415926535897932385

   g = a+b
   call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,0,g_table)

   PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2

   ! The purpose of this subroutine is to calculate the nuclear attraction
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
   ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
   ! arises from the recusion relationship.

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! The first step is generating all the necessary auxillary integrals.
   ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
   ! The values of m range from 0 to i+j+k+ii+jj+kk.

   U = g* PCsquare
   Maxm = i+j+k+ii+jj+kk
   call FmT(Maxm,U,aux)

!   g_table = g**(-1.5)
   constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
         * 2.d0 * sqrt(g/Pi)
   do L = 0,maxm
      aux(L) = aux(L)*constant
   enddo

   ! At this point all the auxillary integrals have been calculated.
   ! It is now time to decompase the attraction integral to it's
   ! auxillary integrals through the recursion scheme.  To do this we use
   ! a recursive function.

   attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
         Cx,Cy,Cz,Px,Py,Pz,g)
   attraction = attraction*(-1.d0)* Z
   return
end function attraction



! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

double precision recursive function attrecurse(i,j,k,ii,jj,kk,m,aux,Ax,Ay, &
      Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g) result(attrec)
   implicit double precision(a-h,o-z)
   dimension iexponents(6),center(12),aux(0:20)

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! If this is one of the auxillary integrals (s||s)^(m), assign value and
   ! return.

   if (i+j+k+ii+jj+kk == 0) then
      attrec=aux(m)

      ! Otherwise, use the recusion relation from Obara and Saika.  The first
      ! step is to find the lowest nonzero angular momentum exponent.  This is
      ! because the more exponents equal zero the fewer terms need to be
      ! calculated, and each recursive loop reduces the angular momentum
      ! exponents. This therefore reorders the atoms and sets the exponent
      ! to be reduced.

   else
      iexponents(1) = i
      iexponents(2) = j
      iexponents(3) = k
      iexponents(4) = ii
      iexponents(5) = jj
      iexponents(6) = kk
      center(7) = Cx
      center(8) = Cy
      center(9) = Cz
      center(10)= Px
      center(11)= Py
      center(12)= Pz
      ilownum=300
      ilowex=300
      do L=1,6
         if (iexponents(L) < ilowex .AND. iexponents(L) /= 0) then
            ilowex=iexponents(L)
            ilownum=L
         endif
      enddo
      if (ilownum <= 3) then
         center(1)=Ax
         center(2)=Ay
         center(3)=Az
         center(4)=Bx
         center(5)=By
         center(6)=Bz
      else
         center(4)=Ax
         center(5)=Ay
         center(6)=Az
         center(1)=Bx
         center(2)=By
         center(3)=Bz
         iexponents(4) = i
         iexponents(5) = j
         iexponents(6) = k
         iexponents(1) = ii
         iexponents(2) = jj
         iexponents(3) = kk
         ilownum = ilownum - 3
      endif

      ! The first step is lowering the orbital exponent by one.

      iexponents(ilownum) = iexponents(ilownum)-1

      ! At this point, calculate the first two terms of the recusion
      ! relation.

      attrec=0.d0
      PA = center(9+ilownum)-center(ilownum)
      if (PA /= 0) attrec = attrec + PA * &
            attrecurse(iexponents(1),iexponents(2), &
            iexponents(3),iexponents(4), &
            iexponents(5),iexponents(6), &
            m,aux, &
            center(1),center(2),center(3), &
            center(4),center(5),center(6), &
            center(7),center(8),center(9), &
            center(10),center(11),center(12),g)

      PC = center(9+ilownum)-center(6+ilownum)
      if (PC /= 0) attrec = attrec - PC * &
            attrecurse(iexponents(1),iexponents(2), &
            iexponents(3),iexponents(4), &
            iexponents(5),iexponents(6), &
            m+1,aux, &
            center(1),center(2),center(3), &
            center(4),center(5),center(6), &
            center(7),center(8),center(9), &
            center(10),center(11),center(12),g)

      ! The next two terms only arise is the angual momentum of the dimension
      ! of A that has already been lowered is not zero.  In other words, if a
      ! (px|1/rc|px) was passed to this subroutine, we are now considering
      ! (s|1/rc|px), and the following term does not arise, as the x expoent
      ! on A is zero.

      if (iexponents(ilownum) /= 0) then
         coeff = dble(iexponents(ilownum))/(2.d0*g)
         iexponents(ilownum) = iexponents(ilownum)-1
         attrec = attrec + coeff*( &
               attrecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               m,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12),g) &
               -attrecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               m+1,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12),g) &
               )
         iexponents(ilownum) = iexponents(ilownum)+1
      endif

      ! The next two terms only arise is the angual momentum of the dimension
      ! of A that has already been lowered is not zero in B.  If a
      ! (px|1/rc|px) was passed to this subroutine, we are now considering
      ! (s|1/rc|px), and the following term does arise, as the x exponent on
      ! B is 1.

      if (iexponents(ilownum+3) /= 0) then
         coeff = dble(iexponents(ilownum+3))/(2.d0*g)
         iexponents(ilownum+3) = iexponents(ilownum+3)-1
         attrec = attrec + coeff*( &
               attrecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               m,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12),g) &
               -attrecurse(iexponents(1),iexponents(2), &
               iexponents(3),iexponents(4), &
               iexponents(5),iexponents(6), &
               m+1,aux, &
               center(1),center(2),center(3), &
               center(4),center(5),center(6), &
               center(7),center(8),center(9), &
               center(10),center(11),center(12),g) &
               )
      endif
   endif

   return
end

