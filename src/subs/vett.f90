#include "util.fh"
!
!	vett.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! vett
!------------------------------------------------------------
! Alessandro GENONI 03/12/007
! Subroutine to build up the array kvett, whose elemts are kvett(i)=i*(i-1)/2
!-----------------------------------------------------------
Subroutine vett
  use quick_ecp_module
  implicit double precision (a-h,o-z)
  do i=1,nbf12
     kvett(i)=i*(i-1)/2
  end do
  return
end Subroutine vett