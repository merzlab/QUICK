!
!	transpose.f90
!	new_quick
!
!	Created by Yipu Miao on 4/11/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
subroutine transpose(x,xt,n)
    implicit none
    integer n,i,j
    double precision x(n,n),xt(n,n)
    
    do i=1,n
        do j=1,n
            xt(j,i)=x(i,j)
        enddo
    enddo
    
end subroutine
