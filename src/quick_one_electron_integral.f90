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

subroutine fullx
   !   The purpose of this subroutine is to calculate the transformation
   !   matrix X.  The first step is forming the overlap matrix (Smatrix).
   !
   use allmod
   implicit none

   double precision :: overlap
   double precision :: Sminhalf(nbasis)
   double precision :: V(3,nbasis)
   double precision :: IDEGEN1(nbasis)
   double precision :: SJI,sum
   integer Ibas,Jbas,Icon,Jcon,i,j,k,IERROR
   double precision g_table(200),Px,Py,Pz
   integer g_count,ii,jj,kk
   double precision a,b,Ax,Ay,Az,Bx,By,Bz

   call cpu_time(timer_begin%T1eS)

   do Ibas=1,nbasis
      ii = itype(1,Ibas)
      jj = itype(2,Ibas)
      kk = itype(3,Ibas)
      do Jbas=Ibas,nbasis
         i = itype(1,Jbas)
         j = itype(2,Jbas)
         k = itype(3,Jbas)
         g_count = i+ii+j+jj+k+kk

         Ax = xyz(1,quick_basis%ncenter(Jbas))
         Bx = xyz(1,quick_basis%ncenter(Ibas))
         Ay = xyz(2,quick_basis%ncenter(Jbas))
         By = xyz(2,quick_basis%ncenter(Ibas))
         Az = xyz(3,quick_basis%ncenter(Jbas))
         Bz = xyz(3,quick_basis%ncenter(Ibas))

         SJI =0.d0
         do Icon=1,ncontract(ibas)
            b = aexp(Icon,Ibas)
            do Jcon=1,ncontract(jbas)
               a = aexp(Jcon,Jbas)
               call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

               SJI =SJI + &
                     dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
!                     *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                     *overlap(a,b, i,j,k,ii,jj,kk, &
!                     itype(1,Jbas),       itype(2,Jbas),       itype(3,Jbas), &
!                     itype(1,Ibas),       itype(2,Ibas),       itype(3,Ibas), &
!                     xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)),xyz(3,quick_basis%ncenter(Jbas)), &
!                     xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)),
                     Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!            *exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))*inv_g))
            enddo
         enddo
         quick_qm_struct%s(Jbas,Ibas) = SJI
      enddo
   enddo

   call cpu_time(timer_end%T1eS)
   timer_cumer%T1eS=timer_cumer%T1eS+timer_end%T1eS-timer_begin%T1eS

   call copySym(quick_qm_struct%s,nbasis)


   ! copy s matrix to scratch
   call copyDMat(quick_qm_struct%s,quick_scratch%hold,nbasis)

   ! Now diagonalize HOLD to generate the eigenvectors and eigenvalues.
   call cpu_time(timer_begin%T1eSD)

   call DIAG(NBASIS,quick_scratch%hold,NBASIS,quick_method%DMCutoff,V,Sminhalf,IDEGEN1,quick_scratch%hold2,IERROR)

   call cpu_time(timer_end%T1eSD)
   timer_cumer%T1eSD=timer_cumer%T1eSD+timer_end%T1eSD-timer_begin%T1eSD
   ! Consider the following:

   ! X = U * s^(-.5) * transpose(U)

   ! s^-.5 is a diagonal matrix filled with the eigenvalues of S taken to
   ! to the 1/square root.  If we define an intermediate matrix A for the
   ! purposes of this discussion:

   ! A   = U * s^(-.5)
   ! or Aij = Sum(k=1,m) Uik * s^(-.5)kj

   ! s^(-.5)kj = 0 unless k=j so

   ! Aij = Uij * s^(-.5)jj

   ! X   = A * transpose(U)
   ! Xij = Sum(k=1,m) Aik * transpose(U)kj
   ! Xij = Sum(k=1,m) Uik * s^(-.5)kk * transpose(U)kj
   ! Xij = Sum(k=1,m) Uik * s^(-.5)kk * Ujk

   ! Similarly:
   ! Xji = Sum(k=1,m) Ajk * transpose(U)ki
   ! Xji = Sum(k=1,m) Ujk * s^(-.5)kk * transpose(U)ki
   ! Xji = Sum(k=1,m) Ujk * s^(-.5)kk * Uik

   ! This aggravating little demonstration contains two points:
   ! 1)  X can be calculated without crossing columns in the array
   ! which adds to speed.
   ! 2)  X has to be symmetric. Thus we only have to fill the bottom
   ! half. (Lower Diagonal)

   do I=1,nbasis
      if (Sminhalf(I).gt.1E-4) then
      Sminhalf(I) = Sminhalf(I)**(-.5d0)
      else
      Sminhalf(I) = 0.0d0
      endif
   enddo


   ! Transpose U onto X then copy on to U.  Now U contains U transpose.

   call transpose(quick_scratch%hold2,quick_qm_struct%x,nbasis)
   call copyDMat(quick_qm_struct%x,quick_scratch%hold2,nbasis)
   ! Now calculate X.
   ! Xij = Sum(k=1,m) Transpose(U)kj * s^(-.5)kk * Transpose(U)ki

   do I = 1,nbasis
      do J=I,nbasis
         sum = 0.d0
         do K=1,nbasis
            sum = sum+quick_scratch%hold2(K,I)*quick_scratch%hold2(K,J)*Sminhalf(K)
!write(*,*)  k,j,quick_scratch%hold2(K,J),Sminhalf(K),sum
         enddo
         quick_qm_struct%x(I,J) = sum
         quick_qm_struct%x(J,I) = quick_qm_struct%x(I,J)
      enddo
   enddo

   if (quick_method%debug) call debugFullX

   ! At this point we have the transformation matrix (X) which is necessary
   ! to orthogonalize the operator matrix, and the overlap matrix (S) which
   ! is used in the DIIS-SCF procedure.

   return
end subroutine fullx

double precision function ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   implicit none
   double precision :: kinetic
   double precision :: a,b
   integer :: i,j,k,ii,jj,kk,g,g_count
   double precision :: Ax,Ay,Az,Bx,By,Bz
   double precision :: Px,Py,Pz

   double precision :: xi,xj,xk,overlap_core,g_table(200)

   ! The purpose of this subroutine is to calculate the kinetic energy
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,and kk on B.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset kinetic to 0.

   kinetic = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk)) &
         +(Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2
   if (kinetic .ne. 0.d0) then
      kinetic=0.d0

      ! Kinetic energy is the integral of an orbital times the second derivative
      ! over space of the other orbital.  For GTFs, this means that it is just a
      ! sum of various overlap integrals with the powers adjusted.

      xi = dble(i)
      xj = dble(j)
      xk = dble(k)

      kinetic = kinetic &
            +        (-1.d0+     xi)*xi  *overlap_core(a,b,i-2,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a*( 1.d0+2.d0*xi)     *overlap_core(a,b,i  ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i+2,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
      kinetic = kinetic &
            +         (-1.d0+     xj)*xj *overlap_core(a,b,i,j-2,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a* ( 1.d0+2.d0*xj)    *overlap_core(a,b,i,j  ,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i,j+2,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
      kinetic = kinetic &
            +         (-1.d0+     xk)*xk *overlap_core(a,b,i,j,k-2,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a* ( 1.d0+2.d0*xk)    *overlap_core(a,b,i,j,k  ,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i,j,k+2,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   endif
   ekinetic = kinetic*(-0.5d0)  *exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

   return
end function ekinetic

subroutine gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)
  implicit none

  double precision a,b,Ax,Ay,Bx,By,Az,Bz,Px,Py,Pz,g_table(200),g,inv_g
  integer g_count,ig

  g = a+b
  do ig=0,g_count
     g_table(1+ig) = g**(dble(-3-ig)*0.5)
  enddo
  inv_g = 1.0d0 / dble(g)
  Px = (a*Ax + b*Bx)*inv_g
  Py = (a*Ay + b*By)*inv_g
  Pz = (a*Az + b*Bz)*inv_g

  return
end subroutine gpt


! Ed Brothers. October 3, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
double precision function overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   !use quick_constants_module
   use allmod

   implicit none
   ! INPUT PARAMETERS
   double precision a,b                 ! exponent of basis set 1 and 2
   integer i,j,k,ii,jj,kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
   double precision Ax,Ay,Az,Bx,By,Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2

   ! INNER VARIBLES
   double precision g_table(200)
   double precision Px,py,pz,overlap_core

   overlap = overlap_core(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   overlap = overlap*exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

   return
end function overlap


double precision function overlap_core (a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   use quick_constants_module
   implicit none
   ! INPUT PARAMETERS
   double precision a,b                 ! exponent of basis set 1 and 2
   integer i,j,k,ii,jj,kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
   double precision Ax,Ay,Az,Bx,By,Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2

   ! INNER VARIBLES
   double precision element,g,g_table(217),elfactor
   integer ig,jg,kg,ng
   integer iiloop,iloop,jloop,jjloop,kloop,kkloop,ix,jy,kz
   double precision pAx,pAy,pAz,pBx,pBy,pBz
   double precision Px,py,pz,overlap

   double precision xnumfact


   ! The purpose of this subroutine is to calculate the overlap between
   ! two normalized gaussians. i,j and k are the x,y,
   ! and z exponents for the gaussian with exponent a, and ii,jj, and kk
   ! have the same order for b.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset overlap to 0.
   overlap = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk))+(Ax-Bx)**2+(Ay-By)**2+(Az-Bz)**2
   if (overlap.ne.zero) then

      overlap=zero
      ! If it is not zero, construct P and g values.  The gaussian product
      ! theory states the product of two s gaussians on centers A and B
      ! with exponents a and b forms a new s gaussian on P with exponent
      ! g.  (g comes from gamma, as is "alpha,beta, gamma" and P comes
      ! from "Product." Also needed are the PA differences.


      PAx= Px-Ax
      PAy= Py-Ay
      PAz= Pz-Az
      PBx= Px-Bx
      PBy= Py-By
      PBz= Pz-Bz

      ! There is also a few factorials that are needed in the integral many
      ! times.  Calculate these as well.

      xnumfact=fact(i)*fact(ii)*fact(j)*fact(jj)*fact(k)*fact(kk)

      ! Now start looping over i,ii,j,jj,k,kk to form all the required elements

      do iloop=0,i
         do iiloop=mod(iloop,2),ii,2
            do jloop=0,j
               do jjloop=mod(jloop,2),jj,2
                  do kloop=0,k
                     do kkloop=mod(kloop,2),kk,2


                        ix=iloop+iiloop
                        jy=jloop+jjloop
                        kz=kloop+kkloop

                        element = pito3half * g_table(1+ix+jy+kz)

                        ! Check to see if this element is zero.


                        ! Continue calculating the elements.  The next elements arise from the
                        ! different angular momentums portion of the GPT.

                        element=element *PAx**(i-iloop)*PBx**(ii-iiloop) &
                               *PAy**(j-jloop)*PBy**(jj-jjloop) &
                               *PAz**(k-kloop)*PBz**(kk-kkloop) &
                               *xnumfact &
                               /(fact(iloop)*fact(iiloop)* &
                                 fact(jloop)*fact(jjloop)* &
                                 fact(kloop)*fact(kkloop)* &
                                 fact(i-iloop)*fact(ii-iiloop)* &
                                 fact(j-jloop)*fact(jj-jjloop)* &
                                 fact(k-kloop)*fact(kk-kkloop))

                        ! The next part arises from the integratation of a gaussian of arbitrary
                        ! angular momentum.


                        ! Before the Gamma function code, a quick note. All gamma functions are
                        ! of the form:
                        ! 1
                        ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                        ! 2

                        ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                        ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                        ! just requires a loop and multiplying by Pi^3/2

                        do iG=1,ix/2
                           element = element * (dble(ix)*0.5-dble(iG) + .5d0)
                        enddo
                        do jG=1,jy/2
                           element = element * (dble(jy)*0.5-dble(jG) + .5d0)
                        enddo
                        do kG=1,kz/2
                           element = element * (dble(kz)*0.5-dble(kG) + .5d0)
                        enddo

                        ! Now sum the whole thing into the overlap.

                        overlap = overlap + element
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo

      ! The final step is multiplying in the K factor (from the gpt)

!      overlap = overlap*exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

   endif

   overlap_core = overlap

   return
end function overlap_core

double precision function ssoverlap(a,b,Ax,Ay,Az,Bx,By,Bz)
   use quick_constants_module
   implicit none
   double precision a,b,Ax,Ay,Az,Bx,By,Bz

   ssoverlap = pito3half*1.d0*(a+b)**(-3.d0/2.d0)*Exp(-((a*b*((Ax-Bx)**2.d0+(Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

end function ssoverlap


! Ed Brothers. October 3, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
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
end

! Ed Brothers. October 3, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

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
end

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

end

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


!------------------------------------------------
! get1eO
!------------------------------------------------
subroutine get1eO(IBAS)

   !------------------------------------------------
   ! This subroutine is to get 1e integral Operator
   !------------------------------------------------
   use allmod
   implicit double precision(a-h,o-z)
   integer Ibas
   integer g_count
   double precision g_table(200)

   ix = itype(1,Ibas)
   iy = itype(2,Ibas)
   iz = itype(3,Ibas)
   xyzxi = xyz(1,quick_basis%ncenter(Ibas))
   xyzyi = xyz(2,quick_basis%ncenter(Ibas))
   xyzzi = xyz(3,quick_basis%ncenter(Ibas))

   do Jbas=Ibas,nbasis
      jx = itype(1,Jbas)
      jy = itype(2,Jbas)
      jz = itype(3,Jbas)
      xyzxj = xyz(1,quick_basis%ncenter(Jbas))
      xyzyj = xyz(2,quick_basis%ncenter(Jbas))
      xyzzj = xyz(3,quick_basis%ncenter(Jbas))

      g_count = ix+iy+iz+jx+jy+jz+2

      OJI = 0.d0
      do Icon=1,ncontract(ibas)
         ai = aexp(Icon,Ibas)

         do Jcon=1,ncontract(jbas)
            F = dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)
           aj = aexp(Jcon,Jbas)
            ! The first part is the kinetic energy.
           call gpt(aj,ai,xyzxj,xyzyj,xyzzj,xyzxi,xyzyi,xyzzi,Px,Py,Pz,g_count,g_table)

            OJI = OJI + F*ekinetic(aj,   ai, &
                  jx,   jy,   jz,&
                  ix,   iy,   iz, &
                  xyzxj,xyzyj,xyzzj,&
                  xyzxi,xyzyi,xyzzi,Px,Py,Pz,g_table)
         enddo
      enddo
      quick_qm_struct%o(Jbas,Ibas) = OJI
   enddo

end subroutine get1eO


!------------------------------------------------
! get1eEnergy
!------------------------------------------------
subroutine get1eEnergy()
   !------------------------------------------------
   ! This subroutine is to get 1e integral
   !------------------------------------------------
   use allmod
   implicit double precision(a-h,o-z)
   call cpu_time(timer_begin%tE)

   call copySym(quick_qm_struct%o,nbasis)
   quick_qm_struct%Eel=0.d0
   quick_qm_struct%Eel=quick_qm_struct%Eel+sum2mat(quick_qm_struct%dense,quick_qm_struct%o,nbasis)
   call cpu_time(timer_end%tE)
   timer_cumer%TE=timer_cumer%TE+timer_end%TE-timer_begin%TE

end subroutine get1eEnergy


!------------------------------------------------
! get1e
!------------------------------------------------
subroutine get1e(oneElecO)
   use allmod
   implicit double precision(a-h,o-z)
   double precision oneElecO(nbasis,nbasis),temp2d(nbasis,nbasis)

#ifdef MPIV
   include "mpif.h"
#endif

   !------------------------------------------------
   ! This subroutine is to obtain Hcore, and store it
   ! to oneElecO so we don't need to calculate it repeatly for
   ! every scf cycle
   !------------------------------------------------


#ifdef MPIV
   if ((.not.bMPI).or.(nbasis.le.MIN_1E_MPI_BASIS)) then
#endif

     if (master) then

         !=================================================================
         ! Step 1. evaluate 1e integrals
         !-----------------------------------------------------------------
         ! The first part is kinetic part
         ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
         !-----------------------------------------------------------------
         call cpu_time(timer_begin%T1e)
!         call cpu_time(timer_begin%T1eT)
         do Ibas=1,nbasis
            call get1eO(Ibas)
         enddo
!         call cpu_time(timer_end%T1eT)

         !-----------------------------------------------------------------
         ! The second part is attraction part
         !-----------------------------------------------------------------
         call cpu_time(timer_begin%T1eV)
         do IIsh=1,jshell
            do JJsh=IIsh,jshell
               call attrashell(IIsh,JJsh)
            enddo
         enddo
         call cpu_time(timer_end%T1eV)

         call cpu_time(timer_end%t1e)
         timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
         timer_cumer%T1eT=timer_cumer%T1eT+timer_end%T1eT-timer_begin%T1eT
         timer_cumer%T1eV=timer_cumer%T1eV+timer_end%T1eV-timer_begin%T1eV
         timer_cumer%TOp = timer_cumer%T1e
         timer_cumer%TSCF = timer_cumer%T1e

         call copySym(quick_qm_struct%o,nbasis)
         call CopyDMat(quick_qm_struct%o,oneElecO,nbasis)
         !if (quick_method%debug) then
                write(100,*) "ONE ELECTRON MATRIX"
                call PriSym(100,nbasis,oneElecO,'f14.8')
         !endif
      endif
#ifdef MPIV
   else

      !------- MPI/ ALL NODES -------------------

      !=================================================================
      ! Step 1. evaluate 1e integrals
      ! This job is only done on master node since it won't cost much resource
      ! and parallel will even waste more than it saves
      !-----------------------------------------------------------------
      ! The first part is kinetic part
      ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
      !-----------------------------------------------------------------
      call cpu_time(timer_begin%t1e)
      do i=1,nbasis
         do j=1,nbasis
            quick_qm_struct%o(i,j)=0
         enddo
      enddo
      do i=1,mpi_nbasisn(mpirank)
         Ibas=mpi_nbasis(mpirank,i)
         call get1eO(Ibas)
      enddo

      !-----------------------------------------------------------------
      ! The second part is attraction part
      !-----------------------------------------------------------------
      do i=1,mpi_jshelln(mpirank)
         IIsh=mpi_jshell(mpirank,i)
         do JJsh=IIsh,jshell
            call attrashell(IIsh,JJsh)
         enddo
      enddo

      call cpu_time(timer_end%t1e)
      timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e

      ! slave node will send infos
      if(.not.master) then

         ! Copy Opertor to a temp array and then send it to master
         call copyDMat(quick_qm_struct%o,temp2d,nbasis)
         ! send operator to master node
         call MPI_SEND(temp2d,nbasis*nbasis,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
      else
         ! master node will receive infos from every nodes
         do i=1,mpisize-1
            ! receive opertors from slave nodes
            call MPI_RECV(temp2d,nbasis*nbasis,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
            ! and sum them into operator
            do ii=1,nbasis
               do jj=1,nbasis
                  quick_qm_struct%o(ii,jj)=quick_qm_struct%o(ii,jj)+temp2d(ii,jj)
               enddo
            enddo
         enddo
         call copySym(quick_qm_struct%o,nbasis)
         call copyDMat(quick_qm_struct%o,oneElecO,nbasis)
      endif
      !------- END MPI/ALL NODES ------------
   endif
#endif
end subroutine get1e

! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

!    subroutine attrashell(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az, &
      !    Bx,By,Bz,Cx,Cy,Cz,Z)
subroutine attrashell(IIsh,JJsh)
   use allmod
   !    use xiaoconstants
   implicit double precision(a-h,o-z)
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   double precision RA(3),RB(3),RP(3),inv_g,g_table

   ! Variables needed later:
   !    pi=3.1415926535897932385


   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))

   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))

   !   Cx=sumx
   !   Cy=sumy
   !   Cz=sumz

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

   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2


   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))

         !Eqn 14 O&S
         call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,0,g_table)
         g = a+b
         !Eqn 15 O&S
         inv_g = 1.0d0 / dble(g)

         !Calculate first two terms of O&S Eqn A20
         constanttemp=dexp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 + (Az - Bz)**2.d0))*inv_g))
         constant = overlap_core(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) * 2.d0 * sqrt(g/Pi)*constanttemp

         !nextatom=number of external MM point charges. set to 0 if none used
         do iatom=1,natom+quick_molspec%nextatom
            if(iatom<=natom)then
               Cx=xyz(1,iatom)
               Cy=xyz(2,iatom)
               Cz=xyz(3,iatom)
               Z=-1.0d0*quick_molspec%chg(iatom)
            else
               Cx=quick_molspec%extxyz(1,iatom-natom)
               Cy=quick_molspec%extxyz(2,iatom-natom)
               Cz=quick_molspec%extxyz(3,iatom-natom)
               Z=-quick_molspec%extchg(iatom-natom)
            endif
            constant2=constanttemp*Z

            !Calculate the last term of O&S Eqn A21
            PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
!            if(quick_method%fMM .and. a*b*PCsquare/g.gt.33.0d0)then
!               xdistance=1.0d0/dsqrt(PCsquare)
!               call fmmone(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
!                     Cx,Cy,Cz,Px,Py,Pz,iatom,constant2,a,b,xdistance)
!            else

               !Compute O&S Eqn A21
               U = g* PCsquare

               !Calculate the last term of O&S Eqn A20
               call FmT(Maxm,U,aux)

               !Calculate all the auxilary integrals and store in attraxiao
               !array
               do L = 0,maxm
                  aux(L) = aux(L)*constant*Z
                  attraxiao(1,1,L)=aux(L)
               enddo

               ! At this point all the auxillary integrals have been calculated.
               ! It is now time to decompase the attraction integral to it's
               ! auxillary integrals through the recursion scheme.  To do this we use
               ! a recursive function.

               !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
                     !    Cx,Cy,Cz,Px,Py,Pz,g)
               NIJ1=10*NII2+NJJ2

               !    write(*,'(I2,I2,I2,I2,I2,I2)')ips,jps,IIsh,JJsh,NIJ1,iatom

!write(*,'(A8,2X,I3,2X,I3,2X,I3,2X,I3,2X,I3,F20.10,2X,F20.10,2X,F20.10)') "Ax,Ay,Az",&
!ips,jps,IIsh,JJsh,NIJ1,Bx,By,Bz

               call nuclearattra(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
                     Cx,Cy,Cz,Px,Py,Pz,iatom)

!            endif

         enddo

      enddo
   enddo

!stop

   ! Xiao HE remember to multiply Z   01/12/2008
   !    attraction = attraction*(-1.d0)* Z
   201 return
end subroutine attrashell



! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
double precision function attraction(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az, &
      Bx,By,Bz,Cx,Cy,Cz,Z)
   use quick_constants_module
   implicit double precision(a-h,o-z)
   dimension aux(0:20)
   double precision g_table

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
