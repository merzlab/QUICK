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
    use quick_input_parser_module, only:index_keyword, found_keyword

    implicit none
    character nml*(*),keywd*(*)
    double precision rdnml
    integer i,j,k,ierror
    

    if (index_keyword(keywd,nml//'=') .ne. 0) then
        i=index_keyword(keywd,nml//'=')
        k=index_keyword(keywd(i:len(keywd)),'=')+i-1
        j=index_keyword(keywd(i:len(keywd)),' ')+i-1
        call rdnum(keywd(k+1:j-1),1,rdnml,ierror)
        return
    else
        rdnml = 0d0
        return
    endif
end function rdnml
