#include "util.fh"
!
!	rdnum.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!--------------------------------------------------------
! rdnum
!--------------------------------------------------------
! contains all the routines to convert strings to numbers
!--------------------------------------------------------
subroutine rdnum(string,istart,value,ierror)

  ! extracts a double precision floating point number from a character
  ! string.  the field of search starts at string(istart:istart) or
  ! after the first equals sign following the istart position.  the
  ! number is returned in value.  if an error is encountered, ierror
  ! is set to one.  this routine expects that there are no blank spaces
  ! embedded anywhere within the numerical field.

  implicit double precision (a-h,o-z)
  character string*(*),char*1,efield(4)*1
  data efield /'E','e','D','d'/
  save efield
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
  call getnum(string,ibeg,iend,value,ierror)
1000 return
end subroutine rdnum

