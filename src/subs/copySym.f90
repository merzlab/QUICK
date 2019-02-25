!
!	copySym.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! Complete the symmetry of matrix
!-----------------------------------------------------------
! if we have symmetry matrix but only get right-top part infos
! we can simply copy it to left-bottom
!-----------------------------------------------------------
subroutine copySym(O,n)
implicit none
integer n,i,j
double precision O(n,n)

do i=1,n
    do j=i+1,n
        O(i,j)=O(j,i)
    enddo
enddo

end subroutine copySym
