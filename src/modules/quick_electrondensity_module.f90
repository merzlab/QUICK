!
!	quick_electrondensity_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! electron densitry modules. Written by John F. 11/25/2008
module quick_electrondensity_module
    implicit none
   
    double precision, dimension(:,:,:), allocatable :: elecDense,elecDenseLap
    double precision :: xStart,yStart,zStart
    integer :: numPointsX,numPointsY,numPointsZ
    character(len=80) :: dxFileName,xyzFileName
     
end module quick_electrondensity_module