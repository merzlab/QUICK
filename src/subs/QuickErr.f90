#include "util.fh"
!
!	QuickErr.f90
!	new_quick
!
!	Created by Vikrant Tripathy on 10/02/2024.
!	Copyright 2020 Michigan State University. All rights reserved.
!

!-----------------------------------------------------------
! QuickErr
!-----------------------------------------------------------
! print error and exit
!-----------------------------------------------------------

subroutine QuickErr(message)
  implicit none
  character message*(*)

  call PrtErr(OUTFILEHANDLE, trim(message))
  call quick_exit(OUTFILEHANDLE,1)

end subroutine QuickErr
