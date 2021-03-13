#include "util.fh"
!
!	PrtDat.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!   This subroutine is from Wei Li, Nanjing University

!-----------------------------------------------------------
! PrtDate
!-----------------------------------------------------------
! 2004.12.24 Print current time for system
!-----------------------------------------------------------

subroutine PrtDate(io,note,ierr)
  implicit none
  integer io,i
  character datim*26,note*(*)
  integer, intent(inout) :: ierr

  i=len(note)
  call GDate(datim)
  write (io,'(a)') '| '//note(1:i)//' '//datim(1:24)
  call flush(io)
end subroutine PrtDate


!-----------------------------------------------------------
! GDate
!-----------------------------------------------------------
!*Deck GDate
Subroutine GDate(Date1)
  Implicit Integer(A-Z)
  !C
  !C     This wrapper routine either calls FDate (on bsd systems) or
  !C     builds the 24-character date in some other way.
  !C
  Character*(*) Date1
  !C
  !C#ifdef IBM_RS6K
  !C#define GDATE_doNE
  !C      Character*26 LDate
  !C      LDate = ' '
  !C      Junk = GCTime(LDate)
  !C      Date1 = LDate
  !C#endif
  !C#ifndef GDATE_doNE
  Call FDate(Date1)
  !C#endif
  if(Len(Date1).gt.24) Date1(25:) = ' '
  Return
end Subroutine GDate
