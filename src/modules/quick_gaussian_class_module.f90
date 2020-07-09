!
!	quick_gaussian_class_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! Gaussian Module, Kenneth Ayers 1/23/04
module quick_gaussian_class_module

! This file defines the data types needed for a higher abstraction level
! of gaussian orbitals.

    implicit none

    type gaussian
        integer :: ncontract
        integer, dimension(3) :: itype
        double precision, pointer, dimension(:) :: aexp   => null()
        double precision, pointer, dimension(:) :: dcoeff => null()
    end type gaussian
    
end module quick_gaussian_class_module

