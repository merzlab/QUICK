#include "util.fh"
!
!	wrtRestart.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! wrtrestart
!-----------------------------------------------------------
! Ed Brothers. August 18, 2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine wrtrestart
  use allmod
  use quick_gridpoints_module
  implicit none
  integer istart,ifinal
  integer i

  !    logical :: present
  character(len=80) keywd
  !    character(len=20) tempstring

  ! The purpose of this routine is to write out an input file based
  ! on the result of a previous calculation.  Thus the restart file
  ! should be the same as the input file, except gifferent geometry.

  open(infile,file=infilename,status='old')
  open(irstfile,file=rstfilename,status='unknown')
  write (ioutfile,'(" WROTE A RESTART FILE. ")')

  istart = 1
  ifinal = 80
  read (infile,'(A80)') keywd
  call upcase(keywd,80)

  do WHILE(istart.ne.0.and.ifinal.ne.0)
     write(irstfile,'(A80)') keywd
     read (infile,'(A80)') keywd
     call upcase(keywd,80)
     call rdword(keywd,ISTART,IFINAL)
  enddo

  ! After copying the keywords, put in a blank space.

  write(irstfile,'("  ")')

  ! Now we can write the molecular geometry.

  do I=1,natom
     write (irstfile,'(A2,6x,F14.9,3x,F14.9,3x,F14.9)') symbol(quick_molspec%iattype(I)), &
            xyz(1,I)*BOHRS_TO_A,xyz(2,I)*BOHRS_TO_A,xyz(3,I)*BOHRS_TO_A
  enddo

  ! Now another blank line.

  write(irstfile,'("  ")')

  ! If this is DFT calculation, write the grid specification out.
  ! This has to be copied from the input file, as otherwise the radial
  ! number would get smaller as you restarted multiple times.  For
  ! instance, if you specify 75 radial points, the code will chop off
  ! the insignifigant ones, giving you 72 points.  If that is written to
  ! the restart file, when you use the restart file you will actually
  ! have a coarser grid.

  if (quick_method%DFT .OR. quick_method%SEDFT) then
     ! skip all geometry specification and the terminating blank line.
     do I=1,natom+1
        read (infile,'(A80)') keywd
     enddo

     ! Now read in the grid specifiaction and write it out.  Not that the
     ! number of lines is the number of regions+2.
     do I=1,iregion+2
        read (infile,'(A80)') keywd
        write(irstfile,'(A80)') keywd
     enddo

  endif

  close(infile)
  close(irstfile)

  return
end subroutine wrtrestart
