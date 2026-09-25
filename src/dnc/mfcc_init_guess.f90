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


   do ixiao=1,kxiaoconnect
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

   do ixiao=1,kxiaoconnect
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

   do ixiao=1,kxiaoconnect
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


   do ixiao=1,kxiaoconnect
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
