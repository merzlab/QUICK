!
!	sum2Mat.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!-----------------------------------------------------------
! sum2Mat
!-----------------------------------------------------------
! sum2Mat=sigma(i,j) Mat1(i,j)*Mat2(j,i)
!-----------------------------------------------------------
function sum2Mat(Mat1,Mat2,n)
implicit none
integer n,i,j
double precision Mat1(n,n),Mat2(n,n),sum2Mat

sum2Mat=0.0d0
do j=1,n
    do i=1,n
        sum2Mat=sum2Mat+Mat1(i,j)*Mat2(j,i)
    enddo
enddo
end function sum2Mat
