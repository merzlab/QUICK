#include "util.fh"
!
!	order.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! IOrder
!-----------------------------------------------------------
! 2004.12.23 find order for the first n1 a dimension arr(n)
!     incr=1: increasing order; incr=-1: decreasing order
!-----------------------------------------------------------

subroutine IOrder(n,arr)
  implicit none
  integer n,incr,i,j,l,k1,k2
  integer arr(n),PP,k

  do i=1,n-1
     do j=i+1,n
        if (arr(i).ge.arr(j)) then
           k=arr(i);arr(i)=arr(j);arr(j)=k
        endif
     enddo
  enddo

end subroutine IOrder
