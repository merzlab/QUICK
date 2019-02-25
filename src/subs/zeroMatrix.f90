!
!	zeroMatrix.f90
!	new_quick
!
!	Created by Yipu Miao on 3/11/11.
!	Copyright 2011 University of Florida. All rights reserved.
!


subroutine zeroMatrix(O,n)
    integer i,j,n
    double precision O(n,n)
    
    do i=1,n
        do j=1,n
            O(i,j)=0.0d0
        enddo
    enddo
    
    return
end subroutine

subroutine zeroVec(V,n)
    integer n,i
    double precision V(n)
    
    do i=1,n
        V(i)=0.0d0
    enddo
    
    return
end subroutine

subroutine zeroMatrix2(O,n1,n2)
    integer i,j,n1,n2
    double precision O(n1,n2)
    
    do i=1,n1
        do j=1,n2
            O(i,j)=0.0d0
        enddo
    enddo
    
    return
end subroutine

subroutine zeroiMatrix(O,n)
    integer i,j,n
    integer O(n,n)
    
    do i=1,n
        do j=1,n
            O(i,j)=0
        enddo
    enddo
    
    return
end subroutine

subroutine zeroiVec(V,n)
    integer n,i
    integer V(n)
    
    do i=1,n
        V(i)=0
    enddo
    
    return
end subroutine
