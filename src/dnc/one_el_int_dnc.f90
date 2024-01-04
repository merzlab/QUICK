!
!        quick_one_electron_integral.f90
!        new_quick
!
!        Created by Yipu Miao on 4/12/11.
!        Copyright 2011 University of Florida. All rights reserved.
!
!   this module is one-electron integrals 
!   from QUICK-2.1.0.0 that are used by DnC method

!------------------------------------------------
! get1eO
!------------------------------------------------
subroutine get1eO(IBAS)

   !------------------------------------------------
   ! This subroutine is to get 1e integral Operator
   !------------------------------------------------
   use allmod

   implicit double precision(a - h, o - z)
   integer Ibas

   ix = itype(1, Ibas)
   iy = itype(2, Ibas)
   iz = itype(3, Ibas)
   xyzxi = xyz(1, quick_basis%ncenter(Ibas))
   xyzyi = xyz(2, quick_basis%ncenter(Ibas))
   xyzzi = xyz(3, quick_basis%ncenter(Ibas))

   do Jbas = Ibas, nbasis
      jx = itype(1, Jbas)
      jy = itype(2, Jbas)
      jz = itype(3, Jbas)
      xyzxj = xyz(1, quick_basis%ncenter(Jbas))
      xyzyj = xyz(2, quick_basis%ncenter(Jbas))
      xyzzj = xyz(3, quick_basis%ncenter(Jbas))
      OJI = 0.d0
      do Icon = 1, ncontract(ibas)
         ai = aexp(Icon, Ibas)
         do Jcon = 1, ncontract(jbas)
            F = dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas)
            aj = aexp(Jcon, Jbas)
            ! The first part is the kinetic energy.
            OJI = OJI + F*ekindc(aj, ai, &
                                   jx, jy, jz, &
                                   ix, iy, iz, &
                                   xyzxj, xyzyj, xyzzj, &
                                   xyzxi, xyzyi, xyzzi)
         enddo
      enddo
      quick_qm_struct%o(Jbas, Ibas) = OJI
   enddo

end subroutine get1eO

!------------------------------------------------
! get1e
!------------------------------------------------
subroutine get1e(oneElecO)
   use allmod
   use quick_oei_module, only: attrashell
   implicit double precision(a - h, o - z)
   double precision oneElecO(nbasis, nbasis), temp2d(nbasis, nbasis)

#ifdef MPI
   include "mpif.h"
#endif

   !------------------------------------------------
   ! This subroutine is to obtain Hcore, and store it
   ! to oneElecO so we don't need to calculate it repeatly for
   ! every scf cycle
   !------------------------------------------------
#ifdef MPI
   if ((.not. bMPI) .or. (nbasis .le. MIN_1E_MPI_BASIS)) then
#endif
      if (master) then
         !=================================================================
         ! Step 1. evaluate 1e integrals
         !-----------------------------------------------------------------
         ! The first part is kinetic part
         ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
         !-----------------------------------------------------------------
         call cpu_time(timer_begin%T1e)
         do Ibas = 1, nbasis
            call get1eO(Ibas)
         enddo
         !-----------------------------------------------------------------
         ! The second part is attraction part
         !-----------------------------------------------------------------
         do IIsh = 1, jshell
            do JJsh = IIsh, jshell
               call attrashell(IIsh, JJsh)
            enddo
         enddo

         call cpu_time(timer_end%t1e)
         timer_cumer%T1e = timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
         timer_cumer%TOp = timer_cumer%T1e
         timer_cumer%TSCF = timer_cumer%T1e

         call copySym(quick_qm_struct%o, nbasis)
         call CopyDMat(quick_qm_struct%o, oneElecO, nbasis)
         if (quick_method%debug) then
            write (iOutFile, *) "ONE ELECTRON MATRIX"
            call PriSym(iOutFile, nbasis, oneElecO, 'f14.8')
         endif
      endif
#ifdef MPI
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
      do i = 1, nbasis
         do j = 1, nbasis
            quick_qm_struct%o(i, j) = 0
         enddo
      enddo
      do i = 1, mpi_nbasisn(mpirank)
         Ibas = mpi_nbasis(mpirank, i)
         call get1eO(Ibas)
      enddo

      !-----------------------------------------------------------------
      ! The second part is attraction part
      !-----------------------------------------------------------------
      do i = 1, mpi_jshelln(mpirank)
         IIsh = mpi_jshell(mpirank, i)
         do JJsh = IIsh, jshell
            call attrashell(IIsh, JJsh)
         enddo
      enddo

      call cpu_time(timer_end%t1e)
      timer_cumer%T1e = timer_cumer%T1e+timer_end%T1e-timer_begin%T1e

      ! slave node will send infos
      if (.not. master) then

         ! Copy Opertor to a temp array and then send it to master
         call copyDMat(quick_qm_struct%o, temp2d, nbasis)
         ! send operator to master node
         call MPI_SEND(temp2d, nbasis*nbasis, mpi_double_precision, 0, mpirank, MPI_COMM_WORLD, IERROR)
      else
         ! master node will receive infos from every nodes
         do i = 1, mpisize - 1
            ! receive opertors from slave nodes
            call MPI_RECV(temp2d, nbasis*nbasis, mpi_double_precision, i, i, MPI_COMM_WORLD, MPI_STATUS, IERROR)
            ! and sum them into operator
            do ii = 1, nbasis
               do jj = 1, nbasis
                  quick_qm_struct%o(ii, jj) = quick_qm_struct%o(ii, jj) + temp2d(ii, jj)
               enddo
            enddo
         enddo
         call copySym(quick_qm_struct%o, nbasis)
         call copyDMat(quick_qm_struct%o, oneElecO, nbasis)
      endif
      !------- END MPI/ALL NODES ------------
   endif
#endif
end subroutine get1e

double precision function ekindc(a, b, i, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz)
   implicit none
   double precision :: kinetic
   double precision :: a, b
   integer :: i, j, k, ii, jj, kk
   double precision :: Ax, Ay, Az, Bx, By, Bz

   double precision :: xi, xj, xk, overlapdc

   ! The purpose of this subroutine is to calculate the kinetic energy
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,and kk on B.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset kinetic to 0.

   kinetic = (1 + (-1)**(i + ii))*(1 + (-1)**(j + jj))*(1 + (-1)**(k + kk)) &
             + (Ax - Bx)**2 + (Ay - By)**2 + (Az - Bz)**2
   if (kinetic .ne. 0.d0) then
      kinetic = 0.d0

      ! Kinetic energy is the integral of an orbital times the second derivative
      ! over space of the other orbital.  For GTFs, this means that it is just a
      ! sum of various overlapdc integrals with the powers adjusted.

      xi = dble(i)
      xj = dble(j)
      xk = dble(k)
      kinetic = kinetic &
                + (-1.d0 + xi)*xi*overlapdc(a, b, i - 2, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz) &
                - 2.d0*a*(1.d0 + 2.d0*xi)*overlapdc(a, b, i, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz) &
                + 4.d0*(a**2.d0)*overlapdc(a, b, i + 2, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz)
      kinetic = kinetic &
                + (-1.d0 + xj)*xj*overlapdc(a, b, i, j - 2, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz) &
                - 2.d0*a*(1.d0 + 2.d0*xj)*overlapdc(a, b, i, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz) &
                + 4.d0*(a**2.d0)*overlapdc(a, b, i, j + 2, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz)
      kinetic = kinetic &
                + (-1.d0 + xk)*xk*overlapdc(a, b, i, j, k - 2, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz) &
                - 2.d0*a*(1.d0 + 2.d0*xk)*overlapdc(a, b, i, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz) &
                + 4.d0*(a**2.d0)*overlapdc(a, b, i, j, k + 2, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz)
   endif
   ekindc = kinetic/(-2.d0)
   return
end function ekindc

! Ed Brothers. October 3, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
double precision function overlapdc(a, b, i, j, k, ii, jj, kk, Ax, Ay, Az, Bx, By, Bz)
   use quick_constants_module
   implicit none
   ! INPUT PARAMETERS
   double precision a, b                 ! exponent of basis set 1 and 2
   integer i, j, k, ii, jj, kk               ! i,j,k are itype for basis set 1 and ii,jj,kk for 2
   double precision Ax, Ay, Az, Bx, By, Bz   ! Ax,Ay,Az are position for basis set 1 and Bx,By,Bz for 2

   ! INNER VARIBLES
   double precision element, g
   integer ig, jg, kg
   integer iiloop, iloop, jloop, jjloop, kloop, kkloop, ix, jy, kz
   double precision pAx, pAy, pAz, pBx, pBy, pBz
   double precision Px, py, pz
   double precision xnumfact

   ! The purpose of this subroutine is to calculate the overlapdc between
   ! two normalized gaussians. i,j and k are the x,y,
   ! and z exponents for the gaussian with exponent a, and ii,jj, and kk
   ! have the same order for b.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset overlapdc to 0.
   overlapdc = (1 + (-1)**(i + ii))*(1 + (-1)**(j + jj))*(1 + (-1)**(k + kk)) + (Ax - Bx)**2 + (Ay - By)**2 + (Az - Bz)**2
   if (overlapdc .ne. zero) then

      overlapdc = zero
      ! If it is not zero, construct P and g values.  The gaussian product
      ! theory states the product of two s gaussians on centers A and B
      ! with exponents a and b forms a new s gaussian on P with exponent
      ! g.  (g comes from gamma, as is "alpha,beta, gamma" and P comes
      ! from "Product." Also needed are the PA differences.

      g = a + b
      Px = (a*Ax + b*Bx)/g
      Py = (a*Ay + b*By)/g
      Pz = (a*Az + b*Bz)/g

      PAx = Px - Ax
      PAy = Py - Ay
      PAz = Pz - Az
      PBx = Px - Bx
      PBy = Py - By
      PBz = Pz - Bz

      ! There is also a few factorials that are needed in the integral many
      ! times.  Calculate these as well.

      xnumfact = fact(i)*fact(ii)*fact(j)*fact(jj)*fact(k)*fact(kk)

      ! Now start looping over i,ii,j,jj,k,kk to form all the required elements.

      do iloop = 0, i
         do iiloop = 0, ii
            do jloop = 0, j
               do jjloop = 0, jj
                  do kloop = 0, k
                     do kkloop = 0, kk
                        ix = iloop + iiloop
                        jy = jloop + jjloop
                        kz = kloop + kkloop

                        ! Check to see if this element is zero.

                        element = (1 + (-1)**(ix))*(1 + (-1)**(jy))*(1 + (-1)**(kz))/8
                        if (element .ne. zero) then

                           ! Continue calculating the elements.  The next elements arise from the
                           ! different angular momentums portion of the GPT.

                           element = PAx**(i - iloop)*PBx**(ii - iiloop) &
                                     *PAy**(j - jloop)*PBy**(jj - jjloop) &
                                     *PAz**(k - kloop)*PBz**(kk - kkloop) &
                                     *xnumfact &
                                     /(fact(iloop)*fact(iiloop)* &
                                       fact(jloop)*fact(jjloop)* &
                                       fact(kloop)*fact(kkloop)* &
                                       fact(i - iloop)*fact(ii - iiloop)* &
                                       fact(j - jloop)*fact(jj - jjloop)* &
                                       fact(k - kloop)*fact(kk - kkloop))

                           ! The next part arises from the integratation of a gaussian of arbitrary
                           ! angular momentum.

                           element = element*g**(dble(-3 - ix - jy - kz)/2.d0)

                           ! Before the Gamma function code, a quick note. All gamma functions are
                           ! of the form:
                           ! 1
                           ! Gamma[- + integer].  Now since Gamma[z] = (z-1)Gamma(z-1)
                           ! 2

                           ! We can say Gamma(0.5 + i) = (i-1+.5)(i-2+.5)...(i-i+.5)Gamma(0.5)
                           ! and Gamma(.5) is Sqrt(Pi).  Thus to calculate the three gamma
                           ! just requires a loop and multiplying by Pi^3/2

                           do iG = 1, ix/2
                              element = element*(dble(ix)/2.d0 - dble(iG) + .5d0)
                           enddo
                           do jG = 1, jy/2
                              element = element*(dble(jy)/2.d0 - dble(jG) + .5d0)
                           enddo
                           do kG = 1, kz/2
                              element = element*(dble(kz)/2.d0 - dble(kG) + .5d0)
                           enddo
                           element = element*pito3half

                           ! Now sum the whole thing into the overlapdc.

                        endif
                        overlapdc = overlapdc + element
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo

      ! The final step is multiplying in the K factor (from the gpt)

      overlapdc = overlapdc*exp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 + (Az - Bz)**2.d0))/(a + b)))

   endif
   return
end function overlapdc