#include "util.fh"
!
!       prterr.f90
!       new_quick
!
!       Created by Madu Manathunga on 05/21/2020.
!       Copyright 2020 Michigan State University. All rights reserved.
!

!-----------------------------------------------------------
! scalarMatMul
!-----------------------------------------------------------
! Multiplies a matrix by a scalar
!-----------------------------------------------------------

subroutine scalarMatMul(O,n1,n2,scalar)
    integer i,j,n1,n2
    double precision scalar
    double precision O(n1,n2)

    do i=1,n1
        do j=1,n2
            O(j,i)= O(j,i)*scalar
        enddo
    enddo

    return
end subroutine
