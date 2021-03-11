#include "util.fh"
!
!	matMul.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! IMatMul
!-----------------------------------------------------------
!multiplication for two matrix m3=m1xm2 [integer]
!-----------------------------------------------------------
      subroutine IMatMul(n,m1,m2,m3)
      implicit none
      integer n,i,j,k,m1(n,n),m2(n,n),m3(n,n),P

      do i=1,n; do j=1,n
         P=0
         do k=1,n
            P=P+m1(i,k)*m2(k,j)
         enddo
         m3(i,j)=P
      enddo; enddo

      end

!-----------------------------------------------------------
! DMatMul
!-----------------------------------------------------------
!multiplication for two matrix m3=m1xm2 double precision
!-----------------------------------------------------------

      subroutine DMatMul(n,m1,m2,m3)
      implicit none
      integer n,i,j,k
      double precision m1(n,n),m2(n,n),m3(n,n),P

      do j=1,n
        do i=1,n
            P=0d0
            do k=1,n
                P=P+m1(i,k)*m2(k,j)
            enddo
            m3(i,j)=P
        enddo
      enddo
      return
      end subroutine DMatMul
