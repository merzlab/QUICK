!
!	prtTim.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!   This subroutine is from Wei Li, Nanjing University

!-----------------------------------------------------------
! PrtTime
!-----------------------------------------------------------
!  Print out total Time used by the job
!*     in parent program def. "double precision Tim0,CPUTim"; "Tim0=CPUTim(0)" for initial time
!-----------------------------------------------------------
subroutine PrtTim(IOut,sec)
  Implicit double precision(A-H,O-Z)

1000 Format('| Job cpu time:',I3,' days ',I2,' hours ',I2,' minutes ',F4.1,' seconds.')

  Time = sec
  NDays = (Time / (3600.0d0*24.0d0))
  Time = Time - (NDays*(3600.0d0*24.0d0))
  NHours = (Time / 3600.0d0)
  Time = Time - (NHours*3600.0d0)
  NMin = (Time / 60.0d0)
  Time = Time - (NMin*60.0d0)
  Write(IOut,1000) NDays, NHours, NMin, Time
  Return
End subroutine PrtTim
