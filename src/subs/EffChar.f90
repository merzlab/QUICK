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