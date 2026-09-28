#include "util.fh"
!
!	MFCC.f90
!	new_quick
!
!	Created by Yipu Miao on 3/8/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! here is every thing about MFCC

! Allocate the MFCC density blocks.
!
! Sized from the actual fragmentation instead of the fixed 40/600/400/200
! magic numbers this used to carry, which silently capped the method at 40
! fragments and 600 basis functions while reserving ~115 MB per array.
!
! nfrag  : number of fragments (caps are nfrag-1, so this covers them too)
! ncon   : number of connection blocks
! maxbas : largest per-fragment basis function count
!
! The connection blocks hold the I and J sub-blocks side by side, which is
! why MFCC_initial_guess indexes them with an offset of the I block size,
! so they get 2*maxbas.
subroutine allocate_MFCC(nfrag,ncon,maxbas)
   use allmod
   implicit none
   integer, intent(in) :: nfrag,ncon,maxbas

   allocate(MFCCDens(nfrag,maxbas,maxbas))
   allocate(MFCCDensCap(nfrag,maxbas,maxbas))
   allocate(MFCCDensCon(ncon,2*maxbas,2*maxbas))
   allocate(MFCCDensCon2(ncon,2*maxbas,2*maxbas))
   allocate(MFCCDensConI(ncon,maxbas,maxbas))
   allocate(MFCCDensConJ(ncon,maxbas,maxbas))

   ! MFCC_initial_guess accumulates into these, so they must start at zero.
   MFCCDens = 0.0d0
   MFCCDensCap = 0.0d0
   MFCCDensCon = 0.0d0
   MFCCDensCon2 = 0.0d0
   MFCCDensConI = 0.0d0
   MFCCDensConJ = 0.0d0

end subroutine

subroutine MFCC_initial_guess
   use allmod
   call PrtAct(ioutfile,"Begin MFCC initial guess")

   ! The fragment blocks minus the cap blocks must cover every global basis
   ! function exactly once. matombases/matombasef are only valid once getMol
   ! has run readbasis on the whole molecule.
   if (quick_method%debug) then
      write(ioutfile,'(" MFCC index map (global / local basis ranges)")')
      do ixiao=1,npmfcc
         write(ioutfile,'("   fragment ",i3," global ",i5," -",i5,"   local ",i5," -",i5)') &
               ixiao,matombases(ixiao),matombasef(ixiao),mfccbases(ixiao),mfccbasef(ixiao)
      enddo
      do ixiao=1,npmfcc-1
         write(ioutfile,'("   cap      ",i3," global ",i5," -",i5,"   local ",i5," -",i5)') &
               ixiao,matombasescap(ixiao),matombasefcap(ixiao),mfccbasescap(ixiao),mfccbasefcap(ixiao)
      enddo
   endif
   ! mfcc_fragment_scf now computes the connection densities (pass 3), so the
   ! connection loops below are live. The guard remains as a tripwire: if the
   ! local basis ranges are somehow unset, skip rather than index dense(0,...).
   nconuse = kxiaoconnect
   do ixiao = 1, kxiaoconnect
      if (mfccbasesconi(ixiao) .le. 0 .or. mfccbasesconj(ixiao) .le. 0) nconuse = 0
   enddo

   if (kxiaoconnect .gt. 0 .and. nconuse .eq. 0) then
      call PrtWrn(iOutFile,'MFCC connection terms were identified but are NOT included in the guess.')
      write(ioutfile,'("|          MFCCCUT identified ",i5," connection blocks, but their")') kxiaoconnect
      write(ioutfile,'("|          densities are not computed yet, so this guess uses the two")')
      write(ioutfile,'("|          term formula (fragments minus caps) only.")')
      write(ioutfile,'("|")')
      write(ioutfile,'("|          The SCF result stays valid, but the guess is less accurate for")')
      write(ioutfile,'("|          systems with non-sequential residue contacts, which is exactly")')
      write(ioutfile,'("|          where the connection terms would help.")')
      write(ioutfile,'(a)')
      call flush(ioutfile)
   endif

   do i=1,nbasis
      do j=1,nbasis
         quick_qm_struct%dense(i,j)=0.0d0
      enddo
   enddo

   do ixiao=1,npmfcc
      do i=mfccbases(ixiao),mfccbasef(ixiao)
         do j=mfccbases(ixiao),mfccbasef(ixiao)
            quick_qm_struct%dense(matombases(ixiao)+i-mfccbases(ixiao),matombases(ixiao)+j-mfccbases(ixiao)) &
                  =quick_qm_struct%dense(matombases(ixiao)+i-mfccbases(ixiao),matombases(ixiao)+j-mfccbases(ixiao))+ &
                  mfccdens(ixiao,i-mfccbases(ixiao)+1,j-mfccbases(ixiao)+1)
            if(quick_method%debug .and. mfccdens(ixiao,i-mfccbases(ixiao)+1,j-mfccbases(ixiao)+1).gt.0.3d0)then
               print*,'fragment',ixiao,matombases(ixiao)+i-mfccbases(ixiao), &
                     matombases(ixiao)+j-mfccbases(ixiao),mfccdens(ixiao,i-mfccbases(ixiao)+1, &
                     j-mfccbases(ixiao)+1)
            endif
         enddo
      enddo
   enddo

   do ixiao=1,npmfcc-1
      do i=mfccbasescap(ixiao),mfccbasefcap(ixiao)
         do j=mfccbasescap(ixiao),mfccbasefcap(ixiao)
            quick_qm_struct%dense(matombasescap(ixiao)+i-mfccbasescap(ixiao),matombasescap(ixiao)+j-mfccbasescap(ixiao))= &
                  quick_qm_struct%dense(matombasescap(ixiao)+i-mfccbasescap(ixiao),matombasescap(ixiao)+j-mfccbasescap(ixiao)) &
                  -mfccdenscap(ixiao,i-mfccbasescap(ixiao)+1,j-mfccbasescap(ixiao)+1)
            if(quick_method%debug .and. mfccdenscap(ixiao,i-mfccbasescap(ixiao)+1,j-mfccbasescap(ixiao)+1).gt.0.3d0)then
               print*,'cap',ixiao,matombasescap(ixiao)+i-mfccbasescap(ixiao), &
                     matombasescap(ixiao)+j-mfccbasescap(ixiao),mfccdenscap(ixiao,i-mfccbasescap(ixiao)+1, &
                     j-mfccbasescap(ixiao)+1)
            endif
         enddo
      enddo
   enddo


   do ixiao=1,nconuse
      do i=mfccbasesconi(ixiao),mfccbasefconi(ixiao)
         do j=mfccbasesconi(ixiao),mfccbasefconi(ixiao)
            quick_qm_struct%dense(matombasesconi(ixiao)+i-mfccbasesconi(ixiao),matombasesconi(ixiao)+j-mfccbasesconi(ixiao))= &
                  quick_qm_struct%dense(matombasesconi(ixiao)+i-mfccbasesconi(ixiao),matombasesconi(ixiao)+j-mfccbasesconi(ixiao)) &
                  -mfccdensconi(ixiao,i-mfccbasesconi(ixiao)+1,j-mfccbasesconi(ixiao)+1)
            if(quick_method%debug .and. mfccdensconi(ixiao,i-mfccbasesconi(ixiao)+1,j-mfccbasesconi(ixiao)+1).gt.0.3d0)then
               print*,'connect-I',ixiao,matombasesconi(ixiao)+i-mfccbasesconi(ixiao), &
                     matombasesconi(ixiao)+j-mfccbasesconi(ixiao),mfccdensconi(ixiao,i-mfccbasesconi(ixiao)+1, &
                     j-mfccbasesconi(ixiao)+1)
            endif
         enddo
      enddo
   enddo

   do ixiao=1,nconuse
      do i=mfccbasesconj(ixiao),mfccbasefconj(ixiao)
         do j=mfccbasesconj(ixiao),mfccbasefconj(ixiao)
            quick_qm_struct%dense(matombasesconj(ixiao)+i-mfccbasesconj(ixiao),matombasesconj(ixiao)+j-mfccbasesconj(ixiao))= &
                  quick_qm_struct%dense(matombasesconj(ixiao)+i-mfccbasesconj(ixiao),matombasesconj(ixiao)+j-mfccbasesconj(ixiao)) &
                  -mfccdensconj(ixiao,i-mfccbasesconj(ixiao)+1,j-mfccbasesconj(ixiao)+1)
            if(quick_method%debug .and. mfccdensconj(ixiao,i-mfccbasesconj(ixiao)+1,j-mfccbasesconj(ixiao)+1).gt.0.3d0)then
               print*,'connect-J',ixiao,matombasesconj(ixiao)+i-mfccbasesconj(ixiao), &
                     matombasesconj(ixiao)+j-mfccbasesconj(ixiao),mfccdensconj(ixiao,i-mfccbasesconj(ixiao)+1, &
                     j-mfccbasesconj(ixiao)+1)
            endif
         enddo
      enddo
   enddo

   do ixiao=1,nconuse
      do i=mfccbasesconi(ixiao),mfccbasefconi(ixiao)
         do j=mfccbasesconi(ixiao),mfccbasefconi(ixiao)
            quick_qm_struct%dense(matombasesconi(ixiao)+i-mfccbasesconi(ixiao),matombasesconi(ixiao)+j-mfccbasesconi(ixiao))= &
                  quick_qm_struct%dense(matombasesconi(ixiao)+i-mfccbasesconi(ixiao),matombasesconi(ixiao)+j-mfccbasesconi(ixiao)) &
                  +mfccdenscon(ixiao,i-mfccbasesconi(ixiao)+1,j-mfccbasesconi(ixiao)+1)
            if(quick_method%debug .and. mfccdenscon(ixiao,i-mfccbasesconi(ixiao)+1,j-mfccbasesconi(ixiao)+1).gt.0.3d0)then
               print*,'connect-IJ',ixiao,matombasesconi(ixiao)+i-mfccbasesconi(ixiao), &
                     matombasesconi(ixiao)+j-mfccbasesconi(ixiao),mfccdenscon(ixiao,i-mfccbasesconi(ixiao)+1, &
                     j-mfccbasesconi(ixiao)+1)
            endif
         enddo
      enddo
   enddo


   do ixiao=1,nconuse
      do i=mfccbasesconj(ixiao),mfccbasefconj(ixiao)
         do j=mfccbasesconj(ixiao),mfccbasefconj(ixiao)

            iixiaotemp=mfccbasefconi(ixiao)-mfccbasesconi(ixiao)+1

            quick_qm_struct%dense(matombasesconj(ixiao)+i-mfccbasesconj(ixiao),matombasesconj(ixiao)+j-mfccbasesconj(ixiao))= &
                  quick_qm_struct%dense(matombasesconj(ixiao)+i-mfccbasesconj(ixiao),matombasesconj(ixiao)+j-mfccbasesconj(ixiao)) &
                  +mfccdenscon(ixiao,iixiaotemp+i-mfccbasesconj(ixiao)+1, &
                  iixiaotemp+j-mfccbasesconj(ixiao)+1)
            if(quick_method%debug .and. mfccdenscon(ixiao,iixiaotemp+i-mfccbasesconj(ixiao)+1, &
                  iixiaotemp+j-mfccbasesconj(ixiao)+1).gt.0.3d0)then
               print*,'connect-IJ',ixiao,matombasesconj(ixiao)+i-mfccbasesconj(ixiao), &
                     !                     iixiaotemp+i-mfccbasesconj(ixiao)+1,iixiaotemp+j-mfccbasesconj(ixiao)+1, &
                     matombasesconj(ixiao)+j-mfccbasesconj(ixiao),mfccdenscon(ixiao,iixiaotemp+ &
                     i-mfccbasesconj(ixiao)+1, &
                     iixiaotemp+j-mfccbasesconj(ixiao)+1)
            endif
         enddo
      enddo
   enddo

call PrtAct(ioutfile,"Finish MFCC initial guess")
end subroutine

!-------------------------------------------------------
! mfcc_purify_density
!-------------------------------------------------------
! McWeeny purification of the assembled MFCC guess density.
!
! MFCC builds the global density as a sum of fragment blocks minus cap blocks.
! Each block is idempotent on its own, but the sum is not: the assembled density
! has occupation numbers outside the physical range. On the glycine hexamer the
! measured idempotency error (trace(DSDS) - 2 trace(DS), zero for a valid closed
! shell density) is about 17.7, against 1.1 for the SAD guess. DIIS copes badly
! with that, which is why MFCC started closer to the answer than SAD yet needed
! more cycles.
!
! Working with P = D/2 so the target eigenvalues are 0 and 1, each sweep applies
!
!     P <- 3 P S P - 2 P S P S P
!
! which is a contraction toward 0 and 1 for eigenvalues already in (-0.5, 1.5).
! A few sweeps are enough; more can amplify components far outside that range,
! so the sweep is rejected if it makes the idempotency error worse.
!
! Requires the overlap matrix, so this must run after fullX has built it.
!-------------------------------------------------------

subroutine mfcc_purify_density()
   use allmod
   use quick_mpi_module, only: master
   implicit none

   integer, parameter :: MAXSWEEP = 8
   double precision, allocatable :: p(:,:), ps(:,:), psp(:,:), pspsp(:,:), pbest(:,:)
   double precision, allocatable :: cand(:,:,:)
   integer :: ic, ibest
   double precision :: errbest
   character(len=9) :: mapname(3)
   double precision :: err, errprev, t1, t2, nocc

   integer :: i, j, isweep, n

   if (.not.master) return
   if (.not.quick_method%MFCC) return

   n = nbasis
   allocate(p(n,n),ps(n,n),psp(n,n),pspsp(n,n),pbest(n,n),cand(n,n,3))
   mapname(1)='McWeeny  '
   mapname(2)='TC2 down '
   mapname(3)='TC2 up   '

   p = 0.5d0*quick_qm_struct%dense
   pbest = p
   call mfcc_idem(p,n,t1,t2,errprev)

   if (master) write(ioutfile,'(" MFCC purification: error before ",f14.6,"   trace(DS) ",f12.4, &
         &"   target ",f12.4)') 2.0d0*errprev,2.0d0*t1,dble(quick_molspec%nelec)

   nocc = 0.5d0*dble(quick_molspec%nelec)

   nocc = 0.5d0*dble(quick_molspec%nelec)

   ! Which sweep helps depends on where the occupations actually are, so rather
   ! than fix an order, try all three candidate maps each sweep and keep the one
   ! that reduces the idempotency error most:
   !
   !   McWeeny   3 PSP - 2 PSPSP    contracts only inside (-0.5, 1.5)
   !   TC2 down  PSP                pulls negative occupations up toward 0
   !   TC2 up    2P - PSP           pulls occupations above 1 down toward 1
   !
   ! The two TC2 maps are complementary: squaring repairs negative occupations
   ! and the other repairs occupations above 1, so a fixed McWeeny-then-TC2
   ! order would be arbitrary. Greedy selection needs no spectral information.
   !
   ! Selection is on idempotency alone. Purification moves the trace, but the
   ! divide and conquer Fermi step renormalises the electron count every cycle,
   ! so idempotency is the part the SCF cannot repair for itself.
   do isweep = 1, MAXSWEEP
      call DGEMM('n','n',n,n,n,1.0d0,p,n,quick_qm_struct%s,n,0.0d0,ps,n)
      call DGEMM('n','n',n,n,n,1.0d0,ps,n,p,n,0.0d0,psp,n)
      call DGEMM('n','n',n,n,n,1.0d0,ps,n,psp,n,0.0d0,pspsp,n)

      cand(:,:,1) = 3.0d0*psp - 2.0d0*pspsp
      cand(:,:,2) = psp
      cand(:,:,3) = 2.0d0*p - psp

      ibest = 0
      errbest = errprev
      do ic = 1, 3
         call mfcc_idem(cand(:,:,ic),n,t1,t2,err)
         if (err .lt. errbest) then
            errbest = err
            ibest = ic
         endif
      enddo

      if (ibest .eq. 0) then
         if (master) write(ioutfile,'("   sweep ",i2," no candidate improves; stopping")') isweep
         exit
      endif

      p = cand(:,:,ibest)
      pbest = p
      errprev = errbest
      call mfcc_idem(p,n,t1,t2,err)
      if (master) write(ioutfile,'("   sweep ",i2," ",a," -> error ",f14.6, &
            &"   trace(DS) ",f12.4)') isweep,trim(mapname(ibest)),2.0d0*err,2.0d0*t1
      if (err .lt. 1.0d-8) exit
   enddo

   quick_qm_struct%dense = 2.0d0*pbest
   call mfcc_idem(0.5d0*quick_qm_struct%dense,n,t1,t2,err)
   if (master) then
      write(ioutfile,'(" MFCC purification: idempotency error after  ",f14.6, &
            &"   trace(DS) ",f12.4)') 2.0d0*err,2.0d0*t1
      call flush(ioutfile)
   endif

   deallocate(p,ps,psp,pspsp,pbest,cand)

end subroutine mfcc_purify_density


!-------------------------------------------------------
! mfcc_idem
!-------------------------------------------------------
! For P with target eigenvalues 0 and 1: t1 = trace(PS), t2 = trace(PSPS),
! err = |t2 - t1|, which is zero when P is idempotent.
!-------------------------------------------------------

subroutine mfcc_idem(p,n,t1,t2,err)
   use allmod
   implicit none

   integer, intent(in) :: n
   double precision, intent(in) :: p(n,n)
   double precision, intent(out) :: t1, t2, err

   double precision, allocatable :: ps(:,:)
   integer :: i, j

   allocate(ps(n,n))
   call DGEMM('n','n',n,n,n,1.0d0,p,n,quick_qm_struct%s,n,0.0d0,ps,n)
   t1 = 0.0d0
   do i = 1, n
      t1 = t1 + ps(i,i)
   enddo
   t2 = 0.0d0
   do i = 1, n
      do j = 1, n
         t2 = t2 + ps(i,j)*ps(j,i)
      enddo
   enddo
   err = dabs(t2-t1)
   deallocate(ps)

end subroutine mfcc_idem
