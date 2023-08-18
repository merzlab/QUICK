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
  public :: dlf_sort_shell_ind, dlf_sort_shell

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

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_si
!!
!! FUNCTION
!! Simple in-place straight insertion sorting algorithm for 1D real array
!!
!! INPUTS
!!
!! real array arr, arbitrary size
!!
!! OUTPUTS
!!
!! arr is overwritten with sorted array (ascending order)
!!
!! SYNOPSIS
subroutine dlf_sort_si(arr)
  implicit none
  real(rk), intent(inout), dimension(:) :: arr
!! SOURCE
  integer :: N,i,j,k
  real(rk) :: y
  N=size(arr)
  
  outer: do i=2,N
    y=arr(i)
    k=i
    inner: do j=i-1,1,-1
      if (y.ge.arr(j)) exit inner
      k=j
    enddo inner 
    if (k.eq.i) cycle outer
    arr(k+1:i)=arr(k:i-1)
    arr(k)=y
  enddo outer
  
  return
end subroutine dlf_sort_si
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_si_ri
!!
!! FUNCTION
!! Auxiliar routine for dlf_sort_shell_ind. Does a simple straight 
!! insertion sort on the real array 'arr' while the integer array
!! 'pasv_int' is sorted along with 'arr' for bookkeeping purposes.
!!
!! INPUTS
!!
!! real 1D array 'arr', arbitrary size
!! integer 1D array 'pasv_int', must have the same size as arr 
!!   (no checks performed within the routine)
!!
!! OUTPUTS
!!
!! arr is overwritten with sorted array (ascending order)
!! pasv_int is overwritten as well 
!!
!! SYNOPSIS
subroutine dlf_sort_si_ri(arr,pasv_int)
  implicit none
  real(rk), intent(inout), dimension(:) :: arr
  integer, intent(inout), dimension(:) :: pasv_int
!! SOURCE
  integer :: N,i,j,k,iy
  real(rk) :: y
  N=size(arr)
  
  outer: do i=2,N
    y=arr(i)
    iy=pasv_int(i)
    k=i
    inner: do j=i-1,1,-1
      if (y.ge.arr(j)) exit inner
      k=j
    enddo inner 
    if (k.eq.i) cycle outer
    arr(k+1:i)=arr(k:i-1)
    arr(k)=y
    pasv_int(k+1:i)=pasv_int(k:i-1)
    pasv_int(k)=iy
  enddo outer
  
  return
end subroutine dlf_sort_si_ri
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_si_ind
!!
!! FUNCTION
!! Sort index generating routine for 1D real array via straight insertion
!! algorithm.
!!
!! INPUTS
!!
!! real 1D array 'arr', arbitrary size
!!
!! OUTPUTS
!!
!! Sort index (integer 1D array) 'ind', such that (arr(ind(i)), i=1,...,N)
!! is sorted in ascending order. Must have same size as 'arr'.
!!
!! SYNOPSIS
subroutine dlf_sort_si_ind(arr,ind)
  implicit none
  real(rk), intent(in), dimension(:) :: arr
  integer, intent(out), dimension(:) :: ind
!! SOURCE
  integer :: N,i,j,k,iy
  real(rk) :: y
  N=size(arr)
  
  if (size(ind).ne.N) then
    write(*,'(A)') 'Error in dlf_sort_si_ind: size mismatch!'
    call dlf_error()
  endif
  
  do i=1,N
   ind(i)=i
  enddo
  
  outer: do i=2,N
    y=arr(ind(i))
    iy=ind(i)
    k=i
    inner: do j=i-1,1,-1
      if (y.ge.arr(ind(j))) exit inner
      k=j
    enddo inner 
    if (k.eq.i) cycle outer
    ind(k+1:i)=ind(k:i-1)
    ind(k)=iy
  enddo outer
  
  return
end subroutine dlf_sort_si_ind
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_shell
!!
!! FUNCTION
!! In-place Shell-sort algorithm for 1D real array
!!
!! INPUTS
!!
!! real array arr, arbitrary size
!!
!! OUTPUTS
!!
!! arr is overwritten with sorted array (ascending order)
!!
!! SYNOPSIS
subroutine dlf_sort_shell(arr)
  use dlf_allocate, only: allocate, deallocate
  implicit none
  real(rk), intent(inout), dimension(:), target :: arr
!! SOURCE
  logical, dimension(size(arr)) :: mask
  integer :: N,i,j,k,del,maxinc,nel
  integer, allocatable, dimension(:) :: incr
  real(rk), pointer :: tmparr(:)
  nullify(tmparr)
  N=size(arr)
  
  ! build increment table
  maxinc=floor(log(2*real(N)+1._rk)/log(3._rk))
  call allocate(incr,maxinc)
  
  incr(1)=1
  do k=2,maxinc
    incr(k)=3*incr(k-1)+1
  enddo
  
  do k=maxinc,1,-1
    del=incr(k)
    do i=1,del
      nullify(tmparr)
      tmparr => arr(i:N:del)
      call dlf_sort_si(tmparr)
    enddo
  enddo
  
  call deallocate(incr)
  nullify(tmparr)
  
  return
end subroutine dlf_sort_shell
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_sort/dlf_sort_shell_ind
!!
!! FUNCTION
!! Sort index generating routine for 1D real array via shell-sort
!! algorithm.
!!
!! INPUTS
!!
!! real 1D array 'arr', arbitrary size
!!
!! OUTPUTS
!!
!! Sort index (integer 1D array) 'ind', such that (arr(ind(i)), i=1,...,N)
!! is sorted in ascending order. Must have same size as 'arr'.
!!
!! SYNOPSIS
subroutine dlf_sort_shell_ind(arr,ind)
  use dlf_allocate, only: allocate, deallocate
  implicit none
  real(rk), intent(in), dimension(:) :: arr
  integer, intent(out), dimension(:),target :: ind
!! SOURCE
  real(rk), dimension(size(arr)), target :: arrc
  integer,dimension(:),pointer :: tmpind
  logical, dimension(size(arr)) :: mask
  integer :: N,i,j,k,del,maxinc,nel
  integer, allocatable, dimension(:) :: incr
  real(rk), pointer :: tmparr(:)
  
  nullify(tmparr,tmpind)
  N=size(arr)
  
  arrc=arr
  
  if (size(ind).ne.N) then
    write(*,'(A)') 'Error in dlf_sort_shell_ind: size mismatch!'
    call dlf_error()
  endif
  
  do i=1,N
    ind(i)=i
  enddo
  
  ! build increment table
  maxinc=floor(log(2*real(N)+1._rk)/log(3._rk))
  call allocate(incr,maxinc)
  
  incr(1)=1
  do k=2,maxinc
    incr(k)=3*incr(k-1)+1
    !write(*,*) incr(k)
  enddo
  
  do k=maxinc,1,-1
    del=incr(k)
    do i=1,del
      nullify(tmparr)
      nullify(tmpind)
      tmparr => arrc(i:N:del)
      tmpind => ind(i:N:del)
      call dlf_sort_si_ri(tmparr,tmpind)
    enddo
  enddo
  
  call deallocate(incr)
  nullify(tmparr,tmpind)
  
  return
end subroutine dlf_sort_shell_ind
!!****

end module dlf_sort_module
