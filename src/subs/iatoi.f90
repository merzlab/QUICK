!
!	iatoi.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!------------------------------------------------------------
! iatioi
!------------------------------------------------------------

subroutine iatoi(str, istart, lstr, integ, ierror)

  implicit double precision(a-h,o-z)

  character str*(*), ch
  logical :: int, min

  integ = 0
  izero = ichar('0')
  nstr = len(str)
  do i=istart,nstr
     ch = str(i:i)
     call whatis2(ch, int, min)
     if ( .not. int) goto 20
  enddo
20 lstr = i-1
  if (lstr == 0) return
  call getinum(str,istart,lstr,integ,ierror)

end subroutine iatoi

