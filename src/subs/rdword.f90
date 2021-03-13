#include "util.fh"
!
!	rdword.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

SUBROUTINE RDWORD(STRING,ISTART,ISTOP)

  ! LOCATES THE NEXT WORD IN STRING STARTING AT STRING(ISTART:ISTART).
  ! ON RETURN ISTART AND ISTOP WILL BE UPDATED AND THE WORD WILL BE
  ! IN STRING(ISTART:ISTOP).  if THERE ARE NO MORE WORDS IN THE STRING,
  ! then BOTH ISTART AND ISTOP WILL BE RETURNED WITH VALUES OF ZERO.

  CHARACTER STRING*(*)
  LOGICAL :: INWORD

  ! GET DECLARED LENGTH OF STRING AND FIND THE NEXT CONTIGUOUS BLOCK
  ! OF NONBLANK CHARACTERS.

  IBEGIN = MAX(ISTART,1)
  IEND = LEN(STRING)
  if(IEND < IBEGIN)then
     ISTART = 0
     ISTOP = 0
     RETURN
  endif
  INWORD = .FALSE.
  IBEGIN = ISTART
  do I=IBEGIN,IEND
     if(STRING(I:I) == ' ' .or. (string(i:i) == achar(9)))then
        if(INWORD)then
           ISTOP = I-1
           RETURN
        endif
     else
        if( .not. INWORD)then
           INWORD = .TRUE.
           ISTART = I
        endif
     endif
  enddo

  ! if WE GET HERE, then EITHER THE WORD FOUND EXTENDS ALL THE WAY TO
  ! THE END OF THE STRING, OR NO WORD WAS FOUND IN THE REMAINING
  ! PORTION OF THE STRING.

  if(INWORD)then
     ISTOP = IEND
  else
     ISTART = 0
     ISTOP = 0
  endif
  RETURN
end SUBROUTINE RDWORD
