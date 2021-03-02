#include "util.fh"
!
!	iwhole.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! iwhole
!-----------------------------------------------------------
! returns the whole number in the field string(ibeg:iend).  only
! the numbers 0-9 are allowed to be present.
!-----------------------------------------------------------
subroutine iwhole(string,ibeg,iend,ivalue,ierror)

  character string*(*)
  ivalue = 0
  ichar0 = ichar('0')
  do 10 i=ibeg,iend
     idigit = ichar(string(i:i)) - ichar0
     if(idigit < 0 .OR. idigit > 9)then
        ierror = 1
        go to 1000
     endif
     ivalue = 10*ivalue + idigit
10 enddo
1000 return
end subroutine iwhole
