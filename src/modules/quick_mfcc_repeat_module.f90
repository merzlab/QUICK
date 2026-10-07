#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors
! All rights reserved.
!
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.
!
!---------------------------------------------------------------------!
! quick_mfcc_repeat_module                                            !
!                                                                     !
! Support for the RPT (repeat) option of the MFCC guess.              !
!                                                                     !
! An MFCC standalone fragment is an isolated molecule: the guess for a !
! water, an ion or a ligand is built from that species on its own,     !
! with no reference to its surroundings. A solvated protein therefore  !
! runs the same calculation thousands of times, once per water, and    !
! the only thing that distinguishes the copies is where they sit and   !
! how they are turned.                                                !
!                                                                     !
! Position is free: the density matrix is indexed by basis functions   !
! that are already centred on the atoms, so a pure translation leaves  !
! it unchanged. Orientation is not. QUICK uses cartesian gaussians     !
! (itype holds the x, y and z exponents of each basis function), so    !
! rotating a molecule mixes the components of every shell with angular !
! momentum above s. The density of a turned copy is therefore          !
!                                                                     !
!     D' = U^T D U                                                    !
!                                                                     !
! where U is block diagonal over shells and carries the cartesian      !
! monomials of each shell into each other.                            !
!                                                                     !
! This module provides the three pieces that needs: a snapshot of the  !
! shell layout, the superposition that recovers the rotation relating  !
! two copies, and U itself.                                           !
!_____________________________________________________________________!

module quick_mfcc_repeat_module

   implicit none
   private

   public :: mfcc_layout_type
   public :: mfcc_rpt_snapshot_layout, mfcc_rpt_free_layout
   public :: mfcc_rpt_align, mfcc_rpt_ao_rotation, mfcc_rpt_rotate_density
   public :: MFCC_RPT_RMSD_TOL, MFCC_RPT_PERM_MAXAT, MFCC_RPT_MAXREP

   ! Two copies count as the same rigid species when their atoms superpose to
   ! within this root mean square deviation, in Angstrom.
   !
   ! A pdb stores coordinates to three decimals, so even a perfectly rigid
   ! solvent model arrives with a few thousandths of an Angstrom of rounding
   ! noise: the waters of the test systems here span 0.8711 to 0.8737 Angstrom
   ! in their O-H distances and a quarter of a degree in their H-O-H angles,
   ! entirely from that rounding. 1.0d-2 clears it with room to spare while
   ! still rejecting a genuinely different conformer. Anything that fails the
   ! test is solved on its own, so the tolerance trades speed, never accuracy.
   double precision, parameter :: MFCC_RPT_RMSD_TOL = 1.0d-2

   ! Atom ordering normally matches between copies, because they come from the
   ! same pdb written by the same tool. When it does not, the mapping is
   ! recovered by trying permutations, which is only affordable for the small
   ! species that actually repeat: water, ions, and the like. Larger fragments
   ! fall back to the order given.
   integer, parameter :: MFCC_RPT_PERM_MAXAT = 4

   ! Beyond this many distinct species the scan stops creating representatives
   ! and the leftovers are solved one by one. It bounds both the work that is
   ! repeated on every rank and the number of alignments tried per fragment, so
   ! an input where every fragment happens to be unique degrades to the ordinary
   ! path instead of becoming quadratic.
   integer, parameter :: MFCC_RPT_MAXREP = 64

   !-------------------------------------------------------------------!
   ! mfcc_layout_type                                                  !
   !                                                                   !
   ! The part of quick_basis that the AO rotation needs, copied out so  !
   ! it survives the deallocate_calculated that every following         !
   ! fragment performs. Every copy of a species has exactly this        !
   ! layout, since it is fixed by the elements and the basis set.       !
   !-------------------------------------------------------------------!
   type mfcc_layout_type
      integer :: nbasis = 0
      integer :: nshell = 0
      integer, allocatable :: ktype(:)    ! cartesian functions in each shell
      integer, allocatable :: kfirst(:)   ! first basis function of each shell
      integer, allocatable :: klmn(:,:)   ! (3,nbasis) cartesian exponents
   end type mfcc_layout_type

contains

   !-------------------------------------------------------------------!
   ! mfcc_rpt_snapshot_layout                                          !
   !                                                                   !
   ! Record the current sub-molecule's shell layout. Must be called     !
   ! while that sub-molecule is still the active one, i.e. straight     !
   ! after its SCF and before the next deallocate_calculated.          !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_snapshot_layout(lay, ierr)
      use quick_basis_module, only: quick_basis, itype, nbasis, nshell
      implicit none
      type(mfcc_layout_type), intent(inout) :: lay
      integer, intent(inout) :: ierr
      integer :: i, ioff, ia

      call mfcc_rpt_free_layout(lay)

      lay%nbasis = nbasis
      lay%nshell = nshell
      allocate(lay%ktype(nshell), lay%kfirst(nshell), lay%klmn(3,nbasis), stat=ia)
      if (ia /= 0) then
         ierr = 34
         return
      endif

      ! basis.f90 lays the basis functions out shell by shell, advancing by
      ! ktype each time, so the first function of a shell is just the running
      ! total of the shells before it. Deriving it here rather than reading
      ! Ksumtype keeps this independent of that array's offset convention.
      ioff = 0
      do i = 1, nshell
         lay%ktype(i) = quick_basis%ktype(i)
         lay%kfirst(i) = ioff + 1
         ioff = ioff + quick_basis%ktype(i)
      enddo

      do i = 1, nbasis
         lay%klmn(1:3,i) = itype(1:3,i)
      enddo

   end subroutine mfcc_rpt_snapshot_layout


   subroutine mfcc_rpt_free_layout(lay)
      implicit none
      type(mfcc_layout_type), intent(inout) :: lay
      if (allocated(lay%ktype))  deallocate(lay%ktype)
      if (allocated(lay%kfirst)) deallocate(lay%kfirst)
      if (allocated(lay%klmn))   deallocate(lay%klmn)
      lay%nbasis = 0
      lay%nshell = 0
   end subroutine mfcc_rpt_free_layout


   !-------------------------------------------------------------------!
   ! mfcc_rpt_align                                                    !
   !                                                                   !
   ! Find the rotation that carries the reference copy of a species     !
   ! onto the target copy, together with the residual rms deviation.    !
   !                                                                   !
   ! rot is returned such that, after both sets are centred on their    !
   ! centroids, ctgt(:,k) = rot .matmul. cref(:,k). Translation is not  !
   ! returned because the density does not depend on it.                !
   !                                                                   !
   ! ok comes back false when the two are not the same rigid body,      !
   ! which the caller answers by solving the target on its own.         !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_align(nat, zref, cref, ztgt, ctgt, rot, rmsd, ok)
      implicit none
      integer, intent(in) :: nat
      integer, intent(in) :: zref(nat), ztgt(nat)
      double precision, intent(in) :: cref(3,nat), ctgt(3,nat)
      double precision, intent(out) :: rot(3,3), rmsd
      logical, intent(out) :: ok

      integer :: perm(nat), best(nat), i
      double precision :: trial(3,3), trmsd
      logical :: found

      rot = 0.0d0
      do i = 1, 3
         rot(i,i) = 1.0d0
      enddo
      rmsd = huge(1.0d0)
      ok = .false.
      if (nat .le. 0) return

      ! The common case by far: the two copies list their atoms in the same
      ! order, because they were written by the same program from the same
      ! topology. Try that first and stop as soon as it works.
      do i = 1, nat
         perm(i) = i
      enddo
      if (all(ztgt .eq. zref)) then
         call mfcc_rpt_superpose(nat, cref, ctgt, perm, trial, trmsd)
         if (trmsd .le. MFCC_RPT_RMSD_TOL) then
            rot = trial
            rmsd = trmsd
            ok = .true.
            return
         endif
         rot = trial
         rmsd = trmsd
      endif

      ! Otherwise the atoms may be listed in a different order, or two
      ! equivalent atoms may be interchanged, as the two hydrogens of a water
      ! can be. Search the orderings for small fragments only: the cost is
      ! factorial and the species that repeat in a solvated structure are all
      ! tiny. Larger ones keep whatever the identity pairing gave.
      if (nat .gt. MFCC_RPT_PERM_MAXAT) then
         ok = (rmsd .le. MFCC_RPT_RMSD_TOL)
         return
      endif

      found = .false.
      call mfcc_rpt_perm_init(nat, perm)
      do
         ! Only orderings that line up the elements can possibly superpose.
         if (mfcc_rpt_elements_match(nat, zref, ztgt, perm)) then
            call mfcc_rpt_superpose(nat, cref, ctgt, perm, trial, trmsd)
            if (.not.found .or. trmsd .lt. rmsd) then
               found = .true.
               rmsd = trmsd
               rot = trial
               best = perm
            endif
         endif
         if (.not.mfcc_rpt_perm_next(nat, perm)) exit
      enddo

      ok = (found .and. rmsd .le. MFCC_RPT_RMSD_TOL)

   end subroutine mfcc_rpt_align


   !-------------------------------------------------------------------!
   ! mfcc_rpt_superpose                                                !
   !                                                                   !
   ! Optimal rotation carrying cref onto ctgt under the pairing perm,   !
   ! by Horn's quaternion method: the eigenvector of the largest        !
   ! eigenvalue of a 4x4 symmetric matrix built from the covariance.    !
   !                                                                   !
   ! A quaternion always describes a proper rotation, which is why this !
   ! is used in preference to an svd of the covariance: there is no     !
   ! reflected solution to detect and correct.                         !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_superpose(nat, cref, ctgt, perm, rot, rmsd)
      implicit none
      integer, intent(in) :: nat, perm(nat)
      double precision, intent(in) :: cref(3,nat), ctgt(3,nat)
      double precision, intent(out) :: rot(3,3), rmsd

      double precision :: p(3,nat), q(3,nat), cp(3), cq(3)
      double precision :: h(3,3), nmat(4,4), evec(4,4), eval(4)
      double precision :: w, x, y, z, nrm, dev(3)
      integer :: k, i, j, imax

      cp = 0.0d0
      cq = 0.0d0
      do k = 1, nat
         cp = cp + cref(1:3,k)
         cq = cq + ctgt(1:3,perm(k))
      enddo
      cp = cp/dble(nat)
      cq = cq/dble(nat)
      do k = 1, nat
         p(1:3,k) = cref(1:3,k) - cp
         q(1:3,k) = ctgt(1:3,perm(k)) - cq
      enddo

      h = 0.0d0
      do k = 1, nat
         do i = 1, 3
            do j = 1, 3
               h(i,j) = h(i,j) + p(i,k)*q(j,k)
            enddo
         enddo
      enddo

      nmat(1,1) =  h(1,1) + h(2,2) + h(3,3)
      nmat(2,2) =  h(1,1) - h(2,2) - h(3,3)
      nmat(3,3) = -h(1,1) + h(2,2) - h(3,3)
      nmat(4,4) = -h(1,1) - h(2,2) + h(3,3)
      nmat(1,2) =  h(2,3) - h(3,2)
      nmat(1,3) =  h(3,1) - h(1,3)
      nmat(1,4) =  h(1,2) - h(2,1)
      nmat(2,3) =  h(1,2) + h(2,1)
      nmat(2,4) =  h(3,1) + h(1,3)
      nmat(3,4) =  h(2,3) + h(3,2)
      do i = 2, 4
         do j = 1, i-1
            nmat(i,j) = nmat(j,i)
         enddo
      enddo

      call mfcc_rpt_jacobi4(nmat, eval, evec)
      imax = 1
      do i = 2, 4
         if (eval(i) .gt. eval(imax)) imax = i
      enddo

      w = evec(1,imax)
      x = evec(2,imax)
      y = evec(3,imax)
      z = evec(4,imax)
      nrm = dsqrt(w*w + x*x + y*y + z*z)
      if (nrm .lt. 1.0d-12) then
         rot = 0.0d0
         do i = 1, 3
            rot(i,i) = 1.0d0
         enddo
      else
         w = w/nrm; x = x/nrm; y = y/nrm; z = z/nrm
         rot(1,1) = w*w + x*x - y*y - z*z
         rot(1,2) = 2.0d0*(x*y - w*z)
         rot(1,3) = 2.0d0*(x*z + w*y)
         rot(2,1) = 2.0d0*(x*y + w*z)
         rot(2,2) = w*w - x*x + y*y - z*z
         rot(2,3) = 2.0d0*(y*z - w*x)
         rot(3,1) = 2.0d0*(x*z - w*y)
         rot(3,2) = 2.0d0*(y*z + w*x)
         rot(3,3) = w*w - x*x - y*y + z*z
      endif

      ! Report the deviation that actually results from the rotation returned,
      ! rather than reconstructing it from the eigenvalue. It costs nothing and
      ! it checks the quaternion too: a wrong conversion shows up as a large
      ! rmsd and the caller falls back to a real SCF instead of using a bad
      ! density.
      rmsd = 0.0d0
      do k = 1, nat
         dev = q(1:3,k) - matmul(rot, p(1:3,k))
         rmsd = rmsd + dot_product(dev, dev)
      enddo
      rmsd = dsqrt(rmsd/dble(nat))

   end subroutine mfcc_rpt_superpose


   !-------------------------------------------------------------------!
   ! mfcc_rpt_jacobi4                                                  !
   !                                                                   !
   ! Cyclic Jacobi diagonalisation of a symmetric 4x4 matrix. Written   !
   ! out rather than calling the library diagonaliser because this runs !
   ! before the rest of QUICK's linear algebra is set up for the        !
   ! sub-molecule, and because a 4x4 is not worth a dispatch.           !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_jacobi4(ain, eval, evec)
      implicit none
      double precision, intent(in) :: ain(4,4)
      double precision, intent(out) :: eval(4), evec(4,4)

      double precision :: a(4,4), theta, t, c, s, tau, g, h, aip, aiq
      integer :: i, j, p, q, isweep

      a = ain
      evec = 0.0d0
      do i = 1, 4
         evec(i,i) = 1.0d0
      enddo

      do isweep = 1, 50
         g = 0.0d0
         do p = 1, 3
            do q = p+1, 4
               g = g + dabs(a(p,q))
            enddo
         enddo
         if (g .lt. 1.0d-14) exit

         do p = 1, 3
            do q = p+1, 4
               if (dabs(a(p,q)) .lt. 1.0d-16) cycle
               theta = (a(q,q) - a(p,p))/(2.0d0*a(p,q))
               t = dsign(1.0d0,theta)/(dabs(theta) + dsqrt(theta*theta + 1.0d0))
               c = 1.0d0/dsqrt(t*t + 1.0d0)
               s = t*c
               tau = s/(1.0d0 + c)

               h = t*a(p,q)
               a(p,p) = a(p,p) - h
               a(q,q) = a(q,q) + h
               a(p,q) = 0.0d0
               a(q,p) = 0.0d0

               do i = 1, 4
                  if (i .eq. p .or. i .eq. q) cycle
                  aip = a(i,p)
                  aiq = a(i,q)
                  a(i,p) = aip - s*(aiq + tau*aip)
                  a(p,i) = a(i,p)
                  a(i,q) = aiq + s*(aip - tau*aiq)
                  a(q,i) = a(i,q)
               enddo

               do i = 1, 4
                  aip = evec(i,p)
                  aiq = evec(i,q)
                  evec(i,p) = aip - s*(aiq + tau*aip)
                  evec(i,q) = aiq + s*(aip - tau*aiq)
               enddo
            enddo
         enddo
      enddo

      do i = 1, 4
         eval(i) = a(i,i)
      enddo

   end subroutine mfcc_rpt_jacobi4


   !-------------------------------------------------------------------!
   ! mfcc_rpt_ao_rotation                                              !
   !                                                                   !
   ! Build the matrix U with                                           !
   !                                                                   !
   !     chi_i^ref(R^-1 (r - t)) = sum_j U_ij chi_j^tgt(r)             !
   !                                                                   !
   ! for the basis layout lay and the rotation rot carrying reference   !
   ! onto target. U is block diagonal: a basis function only mixes with !
   ! the functions of its own shell that share its angular momentum,    !
   ! since a rotation cannot change either the centre or the degree of  !
   ! the polynomial prefactor.                                         !
   !                                                                   !
   ! Within such a block, writing v for r - A and w for R^T v,          !
   !                                                                   !
   !     w_x^l w_y^m w_z^n = sum T_(lmn),(l'm'n') v_x^l' v_y^m' v_z^n'  !
   !                                                                   !
   ! by the multinomial theorem, and U = N_i T_ij / N_j corrects for    !
   ! the fact that QUICK normalises each cartesian component            !
   ! separately: the ratio of two normalisation constants of the same   !
   ! shell is the square root of the ratio of their double factorial    !
   ! products, independent of the exponent and the contraction.         !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_ao_rotation(lay, rot, u, ierr)
      implicit none
      type(mfcc_layout_type), intent(in) :: lay
      double precision, intent(in) :: rot(3,3)
      double precision, intent(out) :: u(lay%nbasis,lay%nbasis)
      integer, intent(inout) :: ierr

      integer :: ish, ib, jb, i0, i1, lsum, ii, jj, nblk
      integer :: idx(10), ll
      double precision :: tmat(10,10)

      u = 0.0d0

      do ish = 1, lay%nshell
         i0 = lay%kfirst(ish)
         i1 = i0 + lay%ktype(ish) - 1

         ! An sp shell holds both an s and a p function, so split the shell by
         ! total angular momentum before rotating: the two do not mix.
         do ll = 0, 3
            nblk = 0
            do ib = i0, i1
               lsum = lay%klmn(1,ib) + lay%klmn(2,ib) + lay%klmn(3,ib)
               if (lsum .ne. ll) cycle
               nblk = nblk + 1
               if (nblk .gt. 10) then
                  ! Beyond f there is no basis in QUICK, so this cannot happen
                  ! unless the layout is inconsistent.
                  ierr = 44
                  return
               endif
               idx(nblk) = ib
            enddo
            if (nblk .eq. 0) cycle

            if (ll .eq. 0) then
               ! An s function is invariant, so no expansion is needed.
               u(idx(1),idx(1)) = 1.0d0
               cycle
            endif

            call mfcc_rpt_cart_rot(ll, nblk, lay%klmn, idx, rot, tmat)

            do ii = 1, nblk
               do jj = 1, nblk
                  ib = idx(ii)
                  jb = idx(jj)
                  u(ib,jb) = tmat(ii,jj) * &
                        dsqrt(mfcc_rpt_dfprod(lay%klmn(1:3,jb))/mfcc_rpt_dfprod(lay%klmn(1:3,ib)))
               enddo
            enddo
         enddo
      enddo

   end subroutine mfcc_rpt_ao_rotation


   !-------------------------------------------------------------------!
   ! mfcc_rpt_cart_rot                                                 !
   !                                                                   !
   ! The unnormalised cartesian monomial transformation of one shell    !
   ! block: tmat(ii,jj) is the coefficient of the jj-th monomial in the !
   ! expansion of the ii-th monomial evaluated at R^T v.                !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_cart_rot(ll, nblk, klmn, idx, rot, tmat)
      implicit none
      integer, intent(in) :: ll, nblk, klmn(3,*), idx(*)
      double precision, intent(in) :: rot(3,3)
      double precision, intent(out) :: tmat(10,10)

      integer :: ii, jj, li, mi, ni
      integer :: i1, i2, i3, j1, j2, j3, k1, k2, k3
      integer :: e1, e2, e3
      double precision :: ca, cb, cc, coef

      tmat(1:nblk,1:nblk) = 0.0d0

      do ii = 1, nblk
         li = klmn(1,idx(ii))
         mi = klmn(2,idx(ii))
         ni = klmn(3,idx(ii))

         ! w_x = sum_beta R(beta,x) v_beta, so the x, y and z columns of rot
         ! supply the coefficients of the three linear forms being raised to
         ! the powers li, mi and ni.
         do i1 = 0, li
         do i2 = 0, li - i1
            i3 = li - i1 - i2
            ca = mfcc_rpt_multinom(li,i1,i2,i3) * &
                 rot(1,1)**i1 * rot(2,1)**i2 * rot(3,1)**i3

            do j1 = 0, mi
            do j2 = 0, mi - j1
               j3 = mi - j1 - j2
               cb = mfcc_rpt_multinom(mi,j1,j2,j3) * &
                    rot(1,2)**j1 * rot(2,2)**j2 * rot(3,2)**j3

               do k1 = 0, ni
               do k2 = 0, ni - k1
                  k3 = ni - k1 - k2
                  cc = mfcc_rpt_multinom(ni,k1,k2,k3) * &
                       rot(1,3)**k1 * rot(2,3)**k2 * rot(3,3)**k3

                  coef = ca*cb*cc
                  if (dabs(coef) .lt. 1.0d-300) cycle

                  e1 = i1 + j1 + k1
                  e2 = i2 + j2 + k2
                  e3 = i3 + j3 + k3

                  do jj = 1, nblk
                     if (klmn(1,idx(jj)) .eq. e1 .and. &
                         klmn(2,idx(jj)) .eq. e2 .and. &
                         klmn(3,idx(jj)) .eq. e3) then
                        tmat(ii,jj) = tmat(ii,jj) + coef
                        exit
                     endif
                  enddo
               enddo
               enddo
            enddo
            enddo
         enddo
         enddo
      enddo

   end subroutine mfcc_rpt_cart_rot


   !-------------------------------------------------------------------!
   ! mfcc_rpt_rotate_density                                           !
   !                                                                   !
   ! dout = u^T dref u, the reference density expressed in the basis of !
   ! the rotated copy.                                                 !
   !-------------------------------------------------------------------!
   subroutine mfcc_rpt_rotate_density(n, dref, u, dout)
      implicit none
      integer, intent(in) :: n
      double precision, intent(in) :: dref(n,n), u(n,n)
      double precision, intent(out) :: dout(n,n)

      dout = matmul(transpose(u), matmul(dref, u))

   end subroutine mfcc_rpt_rotate_density


   !-------------------------------------------------------------------!
   ! Small helpers.                                                    !
   !-------------------------------------------------------------------!

   ! Product of the double factorials (2l-1)!!(2m-1)!!(2n-1)!! that QUICK's
   ! cartesian normalisation carries, with (-1)!! taken as 1. The powers of two
   ! in that normalisation depend only on l+m+n, which is fixed across a shell,
   ! so they cancel in every ratio taken here and are left out.
   double precision function mfcc_rpt_dfprod(lmn)
      implicit none
      integer, intent(in) :: lmn(3)
      integer :: i, k
      mfcc_rpt_dfprod = 1.0d0
      do i = 1, 3
         do k = 1, lmn(i)
            mfcc_rpt_dfprod = mfcc_rpt_dfprod * dble(2*k - 1)
         enddo
      enddo
   end function mfcc_rpt_dfprod


   ! n! / (i! j! k!) for i+j+k = n, which never exceeds a handful for the
   ! angular momenta QUICK supports.
   double precision function mfcc_rpt_multinom(n,i,j,k)
      implicit none
      integer, intent(in) :: n,i,j,k
      mfcc_rpt_multinom = mfcc_rpt_fact(n)/(mfcc_rpt_fact(i)*mfcc_rpt_fact(j)*mfcc_rpt_fact(k))
   end function mfcc_rpt_multinom


   double precision function mfcc_rpt_fact(n)
      implicit none
      integer, intent(in) :: n
      integer :: i
      mfcc_rpt_fact = 1.0d0
      do i = 2, n
         mfcc_rpt_fact = mfcc_rpt_fact * dble(i)
      enddo
   end function mfcc_rpt_fact


   logical function mfcc_rpt_elements_match(nat, zref, ztgt, perm)
      implicit none
      integer, intent(in) :: nat, zref(nat), ztgt(nat), perm(nat)
      integer :: k
      mfcc_rpt_elements_match = .true.
      do k = 1, nat
         if (ztgt(perm(k)) .ne. zref(k)) then
            mfcc_rpt_elements_match = .false.
            return
         endif
      enddo
   end function mfcc_rpt_elements_match


   subroutine mfcc_rpt_perm_init(nat, perm)
      implicit none
      integer, intent(in) :: nat
      integer, intent(out) :: perm(nat)
      integer :: k
      do k = 1, nat
         perm(k) = k
      enddo
   end subroutine mfcc_rpt_perm_init


   ! Next permutation in lexicographic order; false once the last one has been
   ! handed out.
   logical function mfcc_rpt_perm_next(nat, perm)
      implicit none
      integer, intent(in) :: nat
      integer, intent(inout) :: perm(nat)
      integer :: i, j, t

      mfcc_rpt_perm_next = .false.
      if (nat .lt. 2) return

      i = nat - 1
      do while (i .ge. 1)
         if (perm(i) .lt. perm(i+1)) exit
         i = i - 1
      enddo
      if (i .lt. 1) return

      j = nat
      do while (perm(j) .le. perm(i))
         j = j - 1
      enddo
      t = perm(i); perm(i) = perm(j); perm(j) = t

      i = i + 1
      j = nat
      do while (i .lt. j)
         t = perm(i); perm(i) = perm(j); perm(j) = t
         i = i + 1
         j = j - 1
      enddo
      mfcc_rpt_perm_next = .true.
   end function mfcc_rpt_perm_next

end module quick_mfcc_repeat_module
