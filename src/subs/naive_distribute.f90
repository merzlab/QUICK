#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/23/2021                            !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! Simply distributes items across a give number of candidates         !
!_____________________________________________________________________!

subroutine naive_distribute(nitem, commsize, arrll, arrul)

  implicit none
  integer, intent(in) :: nitem
  integer, intent(in) :: commsize
  integer, intent(out) :: arrll(commsize)
  integer, intent(out) :: arrul(commsize)
  integer :: tmp_counter(1:commsize)
  integer :: i, icount

  tmp_counter = 0
  arrll = 0
  arrul = 0

  icount=nitem
  do while(icount .gt. 0)
     do i=1, commsize
        tmp_counter(i)=tmp_counter(i)+1
        icount=icount-1
        if (icount .lt. 1) exit
     enddo
  enddo

  icount=0
  do i=1, commsize
     icount=icount+tmp_counter(i)
     arrul(i)=icount
     if(i .eq. 1) then
        arrll(i)=1
     else
        arrll(i)=arrul(i-1)+1
     endif
  enddo

end subroutine naive_distribute

