!
!	prterr.f90
!	new_quick
!
!	Created by Madu Manathunga on 03/25/2020.
!	Copyright 2020 Michigan State University. All rights reserved.
!

!-----------------------------------------------------------
! PrtErr
!-----------------------------------------------------------
! print error
!-----------------------------------------------------------

subroutine PrtErr(io,line)
  implicit none
  integer io,L,leng,i
  character line*(*)

  leng=len(line)
  L=0
  write(io,'(a)')
  write(io,'(" ERROR: ",a)') line
  write(io,'(a)')
  call flush(io)
  return
end subroutine PrtErr
