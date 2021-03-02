#include "util.fh"
!
!	upcase.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!--------------------------------------------------------
! Upcase(string,iend)
!--------------------------------------------------------
! CHANGES LOWER CASE CHARACTERS IN STRING(1:IEND) TO UPPER CASE.
! -- S. DIXON.
!--------------------------------------------------------
SUBROUTINE UPCASE(STRING,IEND)
  CHARACTER STRING*(*),AUPP,ALOW,DUMMY
  INTEGER :: IALOW,IAUPP,IDUMMY
  DATA AUPP,ALOW /'A','a'/
  SAVE AUPP,ALOW
  IALOW = ICHAR(ALOW)
  IAUPP = ICHAR(AUPP)
  ISHifT = IALOW - IAUPP
  do I=1,IEND
     DUMMY = STRING(I:I)
     IDUMMY = ICHAR(DUMMY)
     if(IDUMMY >= IALOW)then
        DUMMY = CHAR(IDUMMY - ISHifT)
        STRING(I:I) = DUMMY
     endif
  enddo
  RETURN
end SUBROUTINE UPCASE