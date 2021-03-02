#include "util.fh"
!
!	rdnml.f90
!	new_quick
!
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! this subroutine is to read real number value from keywords, 
! like "**** ABC=1.0 ****", then call 
! rdnml(keywd,"ABC") will return val=1.0
function rdnml(keywd,nml)
    implicit none
    character nml*(*),keywd*(*)
    double precision rdnml
    integer i,j,k,ierror
    

    if (index(keywd,nml//'=') .ne. 0) then
        i=index(keywd,nml//'=')
        k=index(keywd(i:len(keywd)),'=')+i-1
        j=index(keywd(i:len(keywd)),' ')+i-1
        call rdnum(keywd(k+1:j-1),1,rdnml,ierror)
        return
    else
        rdnml = 0d0
        return
    endif
end function rdnml

! this subroutine is to read integer number value from keywords, 
! like "**** ABC=100 ****", then call 
! rdnml(keywd,"ABC") will return val=100
function rdinml(keywd,nml)
    implicit none
    character nml*(*),keywd*(*)
    integer rdinml
    integer i,j,k,ierror

    if (index(keywd,nml//'=') .ne. 0) then
        i=index(keywd,nml//'=')
        k=index(keywd(i:len(keywd)),'=')+i-1
        j=index(keywd(i:len(keywd)),' ')+i-1
        call rdinum(keywd(k+1:j-1),1,rdinml,ierror)
        return
    else
        rdinml=0
        return
    endif
end function rdinml