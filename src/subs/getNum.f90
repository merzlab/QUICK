!
!	getNum.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!--------------------------------------------------------
! getnum
!--------------------------------------------------------
! routine to read number from string
!--------------------------------------------------------
subroutine getnum(string,ibeg,iend,value,ierror)
  implicit none

  character string*(*),char*1,efield(4)*1
  double precision value, asign, xleft, xright, esign, expart
  integer ibeg, iend, ierror, idecml, i1, i2, i, ie
  data efield /'E','e','D','d'/
  save efield

  value = 0.0d0

  ! check for algebraic sign.

  char = string(ibeg:ibeg)
  if(char == '-')then
     asign = -1.0d0
     ibeg = ibeg + 1
  elseif(char == '+')then
     asign = 1.0d0
     ibeg = ibeg + 1
  else
     asign = 1.0d0
  endif
  if(ibeg > iend)then
     ierror = 1
     go to 1000
  endif

  ! first determine the whole number equivalent of whatever is
  ! to the left of any decimal point.

  idecml = index(string(ibeg:iend),'.')
  if(idecml == 1)then
     if(ibeg == iend)then

        ! number is just a decimal point.  assume a value of zero.

        value = 0.0d0
        go to 1000
     endif
     xleft = 0.0d0
     ibeg = ibeg+1
  else
     i1 = ibeg
     if(idecml == 0)then
        i2 = iend
     else
        i2 = ibeg+idecml-2
     endif
     call whole(string,i1,i2,xleft,ierror)
     if(ierror /= 0) go to 1000
     value = xleft*asign
     if(idecml == 0 .OR. i2 == (iend-1)) go to 1000
     ibeg = i2+2
  endif

  ! determine the whole number equivalent of whatever is to the
  ! right of the decimal point.  account for e or d field format.

  do 30 i=1,4
     ie = index(string(ibeg:iend),efield(i))
     if(ie /= 0) go to 40
30 enddo
40 if(ie == 1)then
     value = xleft*asign
     ibeg = ibeg + 1
  else
     i1 = ibeg
     if(ie == 0)then
        i2 = iend
     else
        i2 = ibeg+ie-2
     endif
     call whole(string,i1,i2,xright,ierror)
     if(ierror /= 0) go to 1000
     xright = xright*10.0d0**(i1-i2-1)
     value = value + xright*asign
     if(ie == 0 .OR. i2 == (iend-1)) go to 1000
     ibeg = i2+2
  endif

  ! get the exponential portion.

  char = string(ibeg:ibeg)
  if(char == '-')then
     esign = -1.0d0
     ibeg = ibeg + 1
  elseif(char == '+')then
     esign = 1.0d0
     ibeg = ibeg + 1
  else
     esign = 1.0d0
  endif
  if(ibeg > iend) go to 1000
  i1 = ibeg
  i2 = iend
  call whole(string,i1,i2,expart,ierror)
  if(ierror /= 0) go to 1000
  value = value*10.0d0**(esign*expart)
1000 return
end subroutine getnum
