!!****h* utilities/dlf_sort
!!
!! NAME
!! sort
!!
!! FUNCTION
!! Simple bubble sort
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
!!  Joanne Carr (j.m.carr@dl.ac.uk)
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as 
!!  published by the Free Software Foundation, either version 3 of the 
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public 
!!  License along with DL-FIND.  If not, see 
!!  <http://www.gnu.org/licenses/>.
!!
!! SOURCE
!!****
module dlf_sort_module
  use dlf_parameter_module, only: rk
  implicit none
  private

  public :: dlf_sort

  interface dlf_sort
    module procedure dlf_sort_1
    module procedure dlf_sort_2
  end interface

  integer  :: nswap, ii
  real(rk) :: tmp
contains

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_1
!! 
!! FUNCTION 
!! Simple bubble sort, on the 2-dimensional array (1st argument), 
!! according to the corresponding values in the 1-D array (2nd argument). 
!! The lower the value the smaller the index in the resulting
!! sorted array.
!! The next 2 arguments are the upper bounds of the 1st and 2nd 
!! dimensions of the array in the first argument.  The upper bounds of the
!! first dimensions of all the passed arrays must coincide.
!! 
!! Designed for use with dlf_parallel_opt.
!! 
!! INPUTS
!!
!! a(index_1,index_2)
!! values(index_1)
!! index_1
!! index_2
!!
!! OUTPUTS
!!
!! the sorted array a(index_1,index_2)
!!
!! SYNOPSIS
subroutine dlf_sort_1(a,values,index_1,index_2)
!! SOURCE

  integer, intent(in)  :: index_1 ! upper bound of the 1st dimension of a
  integer, intent(in)  :: index_2 ! upper bound of the 2nd dimension of a
  real(rk), dimension(index_2)                         :: swap
  real(rk), dimension(index_1)         , intent(inout) :: values
  real(rk), dimension(index_1, index_2), intent(inout) :: a

  do
     nswap = 0
     do ii = 1, index_1 - 1
        if (values(ii) > values(ii+1)) then
            swap(:) = a(ii,:)
            a(ii,:) = a(ii+1,:)
            a(ii+1,:) = swap(:)
            tmp = values(ii)
            values(ii) = values(ii+1)
            values(ii+1) = tmp
            nswap = nswap + 1
        end if
     end do
     if (nswap == 0) exit
  end do

end subroutine dlf_sort_1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_2
!!
!! FUNCTION
!! Simple bubble sort, on the 2-dimensional array (1st argument) and 
!! 3-dimensional array (2nd argument),
!! according to the corresponding values in the 1-D array (3rd argument).
!! The lower the value the smaller the index in the resulting
!! sorted arrays.
!! The next 2 arguments are the upper bounds of the 1st and 2nd
!! dimensions of the array in the first argument.  The next 2 arguments 
!! are the upper bounds of the 2nd and 3rd dimensions of the second 
!! array.  The upper bounds of the 
!! first dimensions of all the passed arrays must coincide.
!!
!! Designed for use with dlf_parallel_opt.
!!
!! INPUTS
!!
!! a(index_1,index_2)
!! b(index_1,index_3,index_4)
!! values(index_1)
!! index_1
!! index_2
!! index_3
!! index_4
!!
!! OUTPUTS
!!
!! sorted arrays a(index_1,index_2)
!! b(index_1,index_3,index_4)
!!
!! SYNOPSIS
subroutine dlf_sort_2(a,b,values,index_1,index_2,index_3,index_4)
!! SOURCE

  integer, intent(in)  :: index_1 ! upper bound of the 1st dimension of a
  integer, intent(in)  :: index_2 ! upper bound of the 2nd dimension of a
  integer, intent(in)  :: index_3 ! upper bound of the 2nd dimension of b
  integer, intent(in)  :: index_4 ! upper bound of the 3rd dimension of b
  real(rk), dimension(index_2)                                  :: swap_a
  real(rk), dimension(index_3, index_4)                         :: swap_b
  real(rk), dimension(index_1)                  , intent(inout) :: values
  real(rk), dimension(index_1, index_2)         , intent(inout) :: a
  real(rk), dimension(index_1, index_3, index_4), intent(inout) :: b
  
  do
     nswap = 0
     do ii = 1, index_1 - 1
        if (values(ii) > values(ii+1)) then
            swap_a(:) = a(ii,:)
            a(ii,:) = a(ii+1,:)
            a(ii+1,:) = swap_a(:)

            swap_b(:,:) = b(ii,:,:)
            b(ii,:,:) = b(ii+1,:,:)
            b(ii+1,:,:) = swap_b(:,:)

            tmp = values(ii)
            values(ii) = values(ii+1)
            values(ii+1) = tmp
            nswap = nswap + 1
        end if
     end do 
     if (nswap == 0) exit 
  end do

end subroutine dlf_sort_2
!!****

end module dlf_sort_module
