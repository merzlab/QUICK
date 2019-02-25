!
!	copyDMat.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! CopyDMat
!-----------------------------------------------------------
! 2010.10.26 Copy matrix(double precision)
!-----------------------------------------------------------
subroutine CopyDMat(fromMat,toMat,n)
  implicit none
  integer n,i,j
  double precision fromMat(n,n),toMat(n,n)

  do i=1,n
     do j=1,n
        toMat(j,i)=fromMat(j,i)
     enddo
  enddo

end subroutine CopyDMat

!-----------------------------------------------------------
! CopyDMat
!-----------------------------------------------------------
! 2010.10.26 Copy matrix(double precision)
!-----------------------------------------------------------
subroutine CopyIMat(fromMat,toMat,n)
  implicit none
  integer n,i,j
  integer fromMat(n,n),toMat(n,n)

  do i=1,n
     do j=1,n
        toMat(j,i)=fromMat(j,i)
     enddo
  enddo

end subroutine CopyIMat