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
! This module contains subroutines that compute overlap integrals and ! 
! for the transformation matrix.                                      !
!---------------------------------------------------------------------!

module quick_overlap_module

  implicit double precision(a-h,o-z)
  private  
  public :: fullx, overlap, overlap_core, ssoverlap, gpt, opf

contains

function overlap_core (a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) result(ovlp_core)
   use quick_constants_module
   implicit none
   ! INPUT PARAMETERS
   double precision a,b                 ! exponent of basis set 1 and 2
   integer i,j,k,ii,jj,kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
   double precision Ax,Ay,Az,Bx,By,Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2

   ! INNER VARIBLES
   double precision element,g,g_table(200),elfactor
   integer ig,jg,kg,ng
   integer iiloop,iloop,jloop,jjloop,kloop,kkloop,ix,jy,kz
   double precision pAx,pAy,pAz,pBx,pBy,pBz
   double precision fpAx,fpAy,fpAz,fpBx,fpBy,fpBz
   double precision Px,py,pz,ovlp,ovlp_core

   double precision xnumfact


   ! The purpose of this subroutine is to calculate the overlap between
   ! two normalized gaussians. i,j and k are the x,y,
   ! and z exponents for the gaussian with exponent a, and ii,jj, and kk
   ! have the same order for b.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset ovlp to 0.
   ovlp = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk))+(Ax-Bx)**2+(Ay-By)**2+(Az-Bz)**2
   if (ovlp.ne.zero) then

      ovlp=zero
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
         fPAx=PAx**(i-iloop)/(fact(iloop)*fact(i-iloop))

         do iiloop=mod(iloop,2),ii,2
            ix=iloop+iiloop
            fPBx=PBx**(ii-iiloop)/(fact(iiloop)*fact(ii-iiloop))

            do jloop=0,j
               fPAy=PAy**(j-jloop)/(fact(jloop)*fact(j-jloop))

               do jjloop=mod(jloop,2),jj,2
                  jy=jloop+jjloop
                  fPBy=PBy**(jj-jjloop)/(fact(jjloop)*fact(jj-jjloop))

                  do kloop=0,k
                     fPAz=PAz**(k-kloop)/(fact(kloop)*fact(k-kloop))

                     do kkloop=mod(kloop,2),kk,2
                        kz=kloop+kkloop
                        fPBz=PBz**(kk-kkloop)/(fact(kkloop)*fact(kk-kkloop))

                        element = pito3half * g_table(1+ix+jy+kz)

                        ! Check to see if this element is zero.
                        ! Continue calculating the elements.  The next elements arise from the
                        ! different angular momentums portion of the GPT.

                        element=element*fPAx*fPBx*fPAy*fPBy*fPAz*fPBz*xnumfact

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

                        ovlp = ovlp + element
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo


      ! The final step is multiplying in the K factor (from the gpt)

!      ovlp = ovlp*exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

   endif

   ovlp_core = ovlp

end function overlap_core


function overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) result(ovlp)
   !use quick_constants_module
   use allmod

   implicit none
   ! INPUT PARAMETERS
   double precision a,b                 ! exponent of basis set 1 and 2
   integer i,j,k,ii,jj,kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
   double precision Ax,Ay,Az,Bx,By,Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2

   ! INNER VARIBLES
   double precision g_table(200)
   double precision Px,py,pz,ovlp

   ovlp = overlap_core(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   ovlp = ovlp*exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

end function overlap

subroutine fullx
   !   The purpose of this subroutine is to calculate the transformation
   !   matrix X.  The first step is forming the overlap matrix (Smatrix).
   !
   use allmod
   implicit none

   double precision :: SJI,sum, SJI_temp
   integer Ibas,Jbas,Icon,Jcon,i,j,k,IERROR
   double precision g_table(200),Px,Py,Pz
   integer g_count,ii,jj,kk
   double precision a,b,Ax,Ay,Az,Bx,By,Bz

   call cpu_time(timer_begin%T1eS)

   call allocfullx(quick_scratch,nbasis)

   do Ibas=1,nbasis
      ii = itype(1,Ibas)
      jj = itype(2,Ibas)
      kk = itype(3,Ibas)

      Bx = xyz(1,quick_basis%ncenter(Ibas))
      By = xyz(2,quick_basis%ncenter(Ibas))
      Bz = xyz(3,quick_basis%ncenter(Ibas))

      do Jbas=Ibas,nbasis
         i = itype(1,Jbas)
         j = itype(2,Jbas)
         k = itype(3,Jbas)
         g_count = i+ii+j+jj+k+kk

         Ax = xyz(1,quick_basis%ncenter(Jbas))
         Ay = xyz(2,quick_basis%ncenter(Jbas))
         Az = xyz(3,quick_basis%ncenter(Jbas))

         SJI =0.d0
         do Icon=1,ncontract(ibas)
            b = aexp(Icon,Ibas)
            do Jcon=1,ncontract(jbas)
               a = aexp(Jcon,Jbas)
               call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

               SJI =SJI + &
                     dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                     *overlap(a,b, i,j,k,ii,jj,kk, &
                     Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
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

#if defined CUDA || defined CUDA_MPIV

   call cuda_diag(quick_scratch%hold, quick_scratch%tmpx,quick_scratch%tmphold,&
   quick_scratch%Sminhalf, quick_scratch%IDEGEN1, quick_scratch%hold2,quick_scratch%tmpco, quick_scratch%V, nbasis)
#else

#if defined LAPACK || defined MKL
   call DIAGMKL(nbasis,quick_scratch%hold,quick_scratch%Sminhalf,quick_scratch%hold2,IERROR)
#else
   call DIAG(NBASIS,quick_scratch%hold,NBASIS,quick_method%DMCutoff,quick_scratch%V,quick_scratch%Sminhalf,&
   quick_scratch%IDEGEN1,quick_scratch%hold2,IERROR)
#endif
#endif

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
      if (quick_scratch%Sminhalf(I).gt.1E-4) then
      quick_scratch%tmphold(i,i)= quick_scratch%Sminhalf(I)**(-.5d0)
      endif
   enddo

#if defined CUDA || defined CUDA_MPIV

   call cublas_DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0,quick_scratch%hold2, &
   nbasis, quick_scratch%tmphold, nbasis, 0.0d0, quick_scratch%tmpco,nbasis)

   call cublas_DGEMM ('n', 't', nbasis, nbasis, nbasis, 1.0d0,quick_scratch%tmpco, &
   nbasis, quick_scratch%hold2, nbasis, 0.0d0, quick_qm_struct%x,nbasis)
#else
   call DGEMM ('n', 'n', nbasis, nbasis, nbasis, 1.0d0, quick_scratch%hold2, &
   nbasis, quick_scratch%tmphold, nbasis, 0.0d0, quick_scratch%tmpco,nbasis)

   call DGEMM ('n', 't', nbasis, nbasis, nbasis, 1.0d0, quick_scratch%tmpco, &
   nbasis, quick_scratch%hold2, nbasis, 0.0d0, quick_qm_struct%x,nbasis)
#endif

   ! Transpose U onto X then copy on to U.  Now U contains U transpose.

!   call transpose(quick_scratch%hold2,quick_qm_struct%x,nbasis)
!   call copyDMat(quick_qm_struct%x,quick_scratch%hold2,nbasis)
   ! Now calculate X.
   ! Xij = Sum(k=1,m) Transpose(U)kj * s^(-.5)kk * Transpose(U)ki

   if (quick_method%debug) call debugFullX

   ! At this point we have the transformation matrix (X) which is necessary
   ! to orthogonalize the operator matrix, and the overlap matrix (S) which
   ! is used in the DIIS-SCF procedure.

   call deallocfullx(quick_scratch)

   return
end subroutine fullx

!double precision function overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!   !use quick_constants_module
!   use allmod
!
!   implicit none
!   ! INPUT PARAMETERS
!   double precision a,b                 ! exponent of basis set 1 and 2
!   integer i,j,k,ii,jj,kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
!   double precision Ax,Ay,Az,Bx,By,Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2
!
!   ! INNER VARIBLES
!   double precision g_table(200)
!   double precision Px,py,pz,overlap_core
!
!   overlap = overlap_core(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!   overlap = overlap*exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))
!
!   return
!end function overlap
!
!double precision function overlap_core(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!   use quick_constants_module
!   implicit none
!   ! INPUT PARAMETERS
!   double precision a,b                 ! exponent of basis set 1 and 2
!   integer i,j,k,ii,jj,kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
!   double precision Ax,Ay,Az,Bx,By,Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2
!
!   ! INNER VARIBLES
!   double precision element,g,g_table(200),elfactor
!   integer ig,jg,kg,ng
!   integer iiloop,iloop,jloop,jjloop,kloop,kkloop,ix,jy,kz
!   double precision pAx,pAy,pAz,pBx,pBy,pBz
!   double precision fpAx,fpAy,fpAz,fpBx,fpBy,fpBz
!   double precision Px,py,pz,overlap
!
!   double precision xnumfact
!
!
!   ! The purpose of this subroutine is to calculate the overlap between
!   ! two normalized gaussians. i,j and k are the x,y,
!   ! and z exponents for the gaussian with exponent a, and ii,jj, and kk
!   ! have the same order for b.
!
!   ! The first step is to see if this function is zero due to symmetry.
!   ! If it is not, reset overlap to 0.
!   overlap = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk))+(Ax-Bx)**2+(Ay-By)**2+(Az-Bz)**2
!   if (overlap.ne.zero) then
!
!      overlap=zero
!      ! If it is not zero, construct P and g values.  The gaussian product
!      ! theory states the product of two s gaussians on centers A and B
!      ! with exponents a and b forms a new s gaussian on P with exponent
!      ! g.  (g comes from gamma, as is "alpha,beta, gamma" and P comes
!      ! from "Product." Also needed are the PA differences.
!      PAx= Px-Ax
!      PAy= Py-Ay
!      PAz= Pz-Az
!      PBx= Px-Bx
!      PBy= Py-By
!      PBz= Pz-Bz
!
!      ! There is also a few factorials that are needed in the integral many
!      ! times.  Calculate these as well.
!
!      xnumfact=fact(i)*fact(ii)*fact(j)*fact(jj)*fact(k)*fact(kk)
!
!      ! Now start looping over i,ii,j,jj,k,kk to form all the required elements
!
!      do iloop=0,i
!         fPAx=PAx**(i-iloop)/(fact(iloop)*fact(i-iloop))
!
!         do iiloop=mod(iloop,2),ii,2
!            ix=iloop+iiloop
!            fPBx=PBx**(ii-iiloop)/(fact(iiloop)*fact(ii-iiloop))
!
!            do jloop=0,j
!               fPAy=PAy**(j-jloop)/(fact(jloop)*fact(j-jloop))
!
!               do jjloop=mod(jloop,2),jj,2
!                  jy=jloop+jjloop
!                  fPBy=PBy**(jj-jjloop)/(fact(jjloop)*fact(jj-jjloop))
!
!                  do kloop=0,k
!                     fPAz=PAz**(k-kloop)/(fact(kloop)*fact(k-kloop))
!
!                     do kkloop=mod(kloop,2),kk,2
!                        kz=kloop+kkloop
!                        fPBz=PBz**(kk-kkloop)/(fact(kkloop)*fact(kk-kkloop))
!
!                        element = pito3half * g_table(1+ix+jy+kz)
!
!                        ! Check to see if this element is zero.
!                        ! Continue calculating the elements.  The next elements arise from the
!                        ! different angular momentums portion of the GPT.
!
!                        element=element*fPAx*fPBx*fPAy*fPBy*fPAz*fPBz*xnumfact
!
!                        ! The next part arises from the integratation of a gaussian of arbitrary
!                        ! angular momentum.
!
!
!                        ! Before the Gamma function code, a quick note. All gamma functions are
!                        ! of the form:
!                        ! 1
!                        ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
!                        ! 2
!
!                        ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
!                        ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
!                        ! just requires a loop and multiplying by Pi^3/2
!
!                        do iG=1,ix/2
!                           element = element * (dble(ix)*0.5-dble(iG) + .5d0)
!                        enddo
!                        do jG=1,jy/2
!                           element = element * (dble(jy)*0.5-dble(jG) + .5d0)
!                        enddo
!                        do kG=1,kz/2
!                           element = element * (dble(kz)*0.5-dble(kG) + .5d0)
!                        enddo
!
!                        ! Now sum the whole thing into the overlap.
!
!                        overlap = overlap + element
!                     enddo
!                  enddo
!               enddo
!            enddo
!         enddo
!      enddo
!
!
!      ! The final step is multiplying in the K factor (from the gpt)
!
!!      overlap = overlap*exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))
!
!   endif
!
!   overlap_core = overlap
!
!   return
!end function overlap_core

double precision function ssoverlap(a,b,Ax,Ay,Az,Bx,By,Bz)
   use quick_constants_module
   implicit none
   double precision a,b,Ax,Ay,Az,Bx,By,Bz

   ssoverlap = pito3half*1.d0*(a+b)**(-3.d0/2.d0)*Exp(-((a*b*((Ax-Bx)**2.d0+(Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

end function ssoverlap

double precision function opf(ai, aj, ci, cj, xyzxi, xyzyi, xyzzi, &
xyzxj,xyzyj,xyzzj)

  !------------------------------------------------
  ! This function computes the overlap prefactor 
  ! required for one electron integral prescreening
  !------------------------------------------------
  use quick_constants_module
  implicit none

  double precision, intent(in) :: ai, aj, ci, cj, xyzxi, xyzyi, xyzzi, &
  xyzxj,xyzyj,xyzzj
  double precision :: dist2, oog

  dist2 = (xyzxi-xyzxj)**2 + (xyzyi-xyzyj)**2 + (xyzzi-xyzzj)**2
  oog = 1.0d0/(ai+aj)

  opf = exp(-ai*aj*dist2*oog)*sqrt(PI*oog)*PI*oog*ci*cj

  return
end function opf

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

end module quick_overlap_module

