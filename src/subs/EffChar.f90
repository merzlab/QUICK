#include "util.fh"
!
!	EffChar.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!   This subroutine is from Wei Li, Nanjing University

!-----------------------------------------------------------
! EffChar
!-----------------------------------------------------------
! 2005.01.07 move blank of two sides in a line
!-----------------------------------------------------------
subroutine EffChar(line,ini,ifi,k1,k2)
  implicit none
  integer ini,ifi,k1,k2,i,j
  character line*(*)

  ! If every character in ini..ifi is blank, neither loop below assigns
  ! anything and both results used to come back as whatever the caller's
  ! locals happened to hold. Every one of the callers then indexes a substring
  ! with them unconditionally, so a blank line meant reading uninitialised
  ! memory as substring bounds. quick_open did worse than read: it builds an
  ! 'mv' command out of line(k1:k2) and hands it to execute_command_line.
  !
  ! 1 and 0 make the empty case a well defined zero length range, line(1:0).
  ! The lower bound has to stay at 1 rather than 0, because inidivcon.f90 asks
  ! for linetmp(k1:k1+15) rather than linetmp(k1:k2), and k1=0 there would be a
  ! genuine out of bounds read. Callers that need to tell empty from non-empty
  ! can test k2 .ge. k1.
  k1 = 1
  k2 = 0

  do i=ini,ifi
     if (line(i:i).ne.' ') then
        k1=i; exit
     endif
  enddo

  do i=ifi,ini,-1
     if (line(i:i).ne.' ') then
        k2=i; exit
     endif
  enddo

end subroutine EffChar