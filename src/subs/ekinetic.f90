!
!	ekinetic.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!-----------------------------------------------------------
! ekinetic
!-----------------------------------------------------------
! Ed Brothers. October 12, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
!-----------------------------------------------------------
double precision function ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx, &
     By,Bz)
  implicit double precision(a-h,o-z)
  double precision :: kinetic

  ! The purpose of this subroutine is to calculate the kinetic energy
  ! of an electron  distributed between gtfs with orbital exponents a
  ! and b on A and B with angular momentums defined by i,j,k (a's x, y
  ! and z exponents, respectively) and ii,jj,and kk on B.

  ! The first step is to see if this function is zero due to symmetry.
  ! If it is not, reset kinetic to 0.

  kinetic = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk)) &
       +(Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2
  if (kinetic == 0.d0) goto 100
  kinetic=0.d0

  ! Kinetic energy is the integral of an orbital times the second derivative
  ! over space of the other orbital.  For GTFs, this means that it is just a
  ! sum of various overlap integrals with the powers adjusted.

  xi = dble(i)
  xj = dble(j)
  xk = dble(k)
  kinetic = kinetic + (-1.d0+xi)*xi*overlap(a,b,i-2,j,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz) &
       - 2.d0*a*(1.d0+2.d0*xi)*overlap(a,b,i,j,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz) &
       + 4.d0*(a**2.d0)*overlap(a,b,i+2,j,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz)
  kinetic = kinetic + (-1.d0+xj)*xj*overlap(a,b,i,j-2,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz) &
       - 2.d0*a*(1.d0+2.d0*xj)*overlap(a,b,i,j,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz) &
       + 4.d0*(a**2.d0)*overlap(a,b,i,j+2,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz)
  kinetic = kinetic + (-1.d0+xk)*xk*overlap(a,b,i,j,k-2,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz) &
       - 2.d0*a*(1.d0+2.d0*xk)*overlap(a,b,i,j,k,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz) &
       + 4.d0*(a**2.d0)*overlap(a,b,i,j,k+2,ii,jj,kk, &
       Ax,Ay,Az,Bx,By,Bz)

100 continue
  ekinetic = kinetic/(-2.d0)
  return
end function ekinetic

