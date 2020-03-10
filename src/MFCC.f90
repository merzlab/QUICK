!
!        MFCC.f90
!        new_quick
!
!        Created by Yipu Miao on 3/8/11.
!        Copyright 2011 University of Florida. All rights reserved.
!

! here is every thing about MFCC

subroutine allocate_MFCC()
   use allmod
   allocate (MFCCDens(40, 600, 600))
   allocate (MFCCDensCap(40, 400, 400))
   allocate (MFCCDensCon(40, 200, 200))
   allocate (MFCCDensCon2(40, 200, 200))
   allocate (MFCCDensConI(40, 200, 200))
   allocate (MFCCDensConJ(40, 200, 200))
end subroutine

subroutine MFCC_initial_guess
   use allmod
   do i = 1, nbasis
      do j = 1, nbasis
         quick_qm_struct%dense(i, j) = 0.0d0
      enddo
   enddo

   do ixiao = 1, npmfcc
      do i = mfccbases(ixiao), mfccbasef(ixiao)
         do j = mfccbases(ixiao), mfccbasef(ixiao)
            quick_qm_struct%dense(matombases(ixiao) + i - mfccbases(ixiao), matombases(ixiao) + j - mfccbases(ixiao)) &
               = quick_qm_struct%dense(matombases(ixiao) + i - mfccbases(ixiao), matombases(ixiao) + j - mfccbases(ixiao)) + &
                 mfccdens(ixiao, i - mfccbases(ixiao) + 1, j - mfccbases(ixiao) + 1)
            if (mfccdens(ixiao, i - mfccbases(ixiao) + 1, j - mfccbases(ixiao) + 1) .gt. 0.3d0) then
               print *, 'fragment', ixiao, matombases(ixiao) + i - mfccbases(ixiao), &
                  matombases(ixiao) + j - mfccbases(ixiao), mfccdens(ixiao, i - mfccbases(ixiao) + 1, &
                                                                     j - mfccbases(ixiao) + 1)
            endif
         enddo
      enddo
   enddo

   do ixiao = 1, npmfcc - 1
      do i = mfccbasescap(ixiao), mfccbasefcap(ixiao)
         do j = mfccbasescap(ixiao), mfccbasefcap(ixiao)
           quick_qm_struct%dense(matombasescap(ixiao) + i - mfccbasescap(ixiao), matombasescap(ixiao) + j - mfccbasescap(ixiao)) = &
             quick_qm_struct%dense(matombasescap(ixiao) + i - mfccbasescap(ixiao), matombasescap(ixiao) + j - mfccbasescap(ixiao)) &
               - mfccdenscap(ixiao, i - mfccbasescap(ixiao) + 1, j - mfccbasescap(ixiao) + 1)
            if (mfccdenscap(ixiao, i - mfccbasescap(ixiao) + 1, j - mfccbasescap(ixiao) + 1) .gt. 0.3d0) then
               print *, 'cap', ixiao, matombasescap(ixiao) + i - mfccbasescap(ixiao), &
                  matombasescap(ixiao) + j - mfccbasescap(ixiao), mfccdenscap(ixiao, i - mfccbasescap(ixiao) + 1, &
                                                                              j - mfccbasescap(ixiao) + 1)
            endif
         enddo
      enddo
   enddo

   do ixiao = 1, kxiaoconnect
      do i = mfccbasesconi(ixiao), mfccbasefconi(ixiao)
         do j = mfccbasesconi(ixiao), mfccbasefconi(ixiao)
       quick_qm_struct%dense(matombasesconi(ixiao) + i - mfccbasesconi(ixiao), matombasesconi(ixiao) + j - mfccbasesconi(ixiao)) = &
         quick_qm_struct%dense(matombasesconi(ixiao) + i - mfccbasesconi(ixiao), matombasesconi(ixiao) + j - mfccbasesconi(ixiao)) &
               - mfccdensconi(ixiao, i - mfccbasesconi(ixiao) + 1, j - mfccbasesconi(ixiao) + 1)
            if (mfccdensconi(ixiao, i - mfccbasesconi(ixiao) + 1, j - mfccbasesconi(ixiao) + 1) .gt. 0.3d0) then
               print *, 'connect-I', ixiao, matombasesconi(ixiao) + i - mfccbasesconi(ixiao), &
                  matombasesconi(ixiao) + j - mfccbasesconi(ixiao), mfccdensconi(ixiao, i - mfccbasesconi(ixiao) + 1, &
                                                                                 j - mfccbasesconi(ixiao) + 1)
            endif
         enddo
      enddo
   enddo

   do ixiao = 1, kxiaoconnect
      do i = mfccbasesconj(ixiao), mfccbasefconj(ixiao)
         do j = mfccbasesconj(ixiao), mfccbasefconj(ixiao)
       quick_qm_struct%dense(matombasesconj(ixiao) + i - mfccbasesconj(ixiao), matombasesconj(ixiao) + j - mfccbasesconj(ixiao)) = &
         quick_qm_struct%dense(matombasesconj(ixiao) + i - mfccbasesconj(ixiao), matombasesconj(ixiao) + j - mfccbasesconj(ixiao)) &
               - mfccdensconj(ixiao, i - mfccbasesconj(ixiao) + 1, j - mfccbasesconj(ixiao) + 1)
            if (mfccdensconj(ixiao, i - mfccbasesconj(ixiao) + 1, j - mfccbasesconj(ixiao) + 1) .gt. 0.3d0) then
               print *, 'connect-J', ixiao, matombasesconj(ixiao) + i - mfccbasesconj(ixiao), &
                  matombasesconj(ixiao) + j - mfccbasesconj(ixiao), mfccdensconj(ixiao, i - mfccbasesconj(ixiao) + 1, &
                                                                                 j - mfccbasesconj(ixiao) + 1)
            endif
         enddo
      enddo
   enddo

   do ixiao = 1, kxiaoconnect
      do i = mfccbasesconi(ixiao), mfccbasefconi(ixiao)
         do j = mfccbasesconi(ixiao), mfccbasefconi(ixiao)
       quick_qm_struct%dense(matombasesconi(ixiao) + i - mfccbasesconi(ixiao), matombasesconi(ixiao) + j - mfccbasesconi(ixiao)) = &
         quick_qm_struct%dense(matombasesconi(ixiao) + i - mfccbasesconi(ixiao), matombasesconi(ixiao) + j - mfccbasesconi(ixiao)) &
               + mfccdenscon(ixiao, i - mfccbasesconi(ixiao) + 1, j - mfccbasesconi(ixiao) + 1)
            if (mfccdenscon(ixiao, i - mfccbasesconi(ixiao) + 1, j - mfccbasesconi(ixiao) + 1) .gt. 0.3d0) then
               print *, 'connect-IJ', ixiao, matombasesconi(ixiao) + i - mfccbasesconi(ixiao), &
                  matombasesconi(ixiao) + j - mfccbasesconi(ixiao), mfccdenscon(ixiao, i - mfccbasesconi(ixiao) + 1, &
                                                                                j - mfccbasesconi(ixiao) + 1)
            endif
         enddo
      enddo
   enddo

   do ixiao = 1, kxiaoconnect
      do i = mfccbasesconj(ixiao), mfccbasefconj(ixiao)
         do j = mfccbasesconj(ixiao), mfccbasefconj(ixiao)

            iixiaotemp = mfccbasefconi(ixiao) - mfccbasesconi(ixiao) + 1

       quick_qm_struct%dense(matombasesconj(ixiao) + i - mfccbasesconj(ixiao), matombasesconj(ixiao) + j - mfccbasesconj(ixiao)) = &
         quick_qm_struct%dense(matombasesconj(ixiao) + i - mfccbasesconj(ixiao), matombasesconj(ixiao) + j - mfccbasesconj(ixiao)) &
               + mfccdenscon(ixiao, iixiaotemp + i - mfccbasesconj(ixiao) + 1, &
                             iixiaotemp + j - mfccbasesconj(ixiao) + 1)
            if (mfccdenscon(ixiao, iixiaotemp + i - mfccbasesconj(ixiao) + 1, &
                            iixiaotemp + j - mfccbasesconj(ixiao) + 1) .gt. 0.3d0) then
               print *, 'connect-IJ', ixiao, matombasesconj(ixiao) + i - mfccbasesconj(ixiao), &
                  !                     iixiaotemp+i-mfccbasesconj(ixiao)+1,iixiaotemp+j-mfccbasesconj(ixiao)+1, &
                  matombasesconj(ixiao) + j - mfccbasesconj(ixiao), mfccdenscon(ixiao, iixiaotemp + &
                                                                                i - mfccbasesconj(ixiao) + 1, &
                                                                                iixiaotemp + j - mfccbasesconj(ixiao) + 1)
            endif
         enddo
      enddo
   enddo

end subroutine
