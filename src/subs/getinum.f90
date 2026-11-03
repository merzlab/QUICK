#include "util.fh"
!
!	getinum.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! getinum(string,istart,ivalue,ierror)
!-----------------------------------------------------------
! read integer number
!-----------------------------------------------------------
subroutine getinum(string,ib,ie,ivalue,ierror)

  implicit none
  character, intent(in) :: string*(*)
  integer, intent(inout) :: ib
  integer, intent(in) :: ie
  integer, intent(out) :: ivalue, ierror
  character :: char*1, efield(4)*1
  integer :: isign

  ierror = 0
  ivalue = 0
  if (string(ib:ib) == '-') then
     ib = ib + 1
     isign = -1
  elseif (string(ib:ib) == '+') then
     ib = ib + 1
     isign = 1
  else
     isign = 1
  endif

  call iwhole(string,ib,ie,ivalue,ierror)
  ivalue = ivalue * isign

  return
end subroutine getinum
