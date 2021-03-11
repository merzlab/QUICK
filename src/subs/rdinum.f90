#include "util.fh"
!
!	rdinum.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!-----------------------------------------------------------
! rdinum(string,istart,ivalue,ierror)
!-----------------------------------------------------------
! Read integer number
!-----------------------------------------------------------

subroutine rdinum(string,istart,ivalue,ierror)

  implicit double precision (a-h,o-z)
  character string*(*),char*1,efield(4)*1

  ierror = 0
  ibeg = istart
  istop = len(string)
  iend = istop
  do 10 i=istart,istop
     if(string(i:i) == ' ')then
        iend = i-1
        go to 20
     endif
10 enddo
20 if(iend < ibeg)then
     ierror = 1
     go to 1000
  endif
  ieq = index(string(ibeg:iend),'=')
  if(ieq /= 0) ibeg = ibeg + ieq
  call getinum(string,ibeg,iend,ivalue,ierror)
1000 return
end subroutine rdinum

