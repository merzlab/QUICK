!
!	quick_amber_interface_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   

#include "util.fh"

module amber_interface_module

!------------------------------------------------------------------------
!  ATTRIBUTES  : AMBER_interface_logic
!                quick_first_call
!  SUBROUTINES : none
!  FUNCTIONS   : none
!  DESCRIPTION : This module is to provide an interface between AMBER and 
!                quick to impliment ab initio QM/MM calculation
!  AUTHOR      : Yipu Miao
!------------------------------------------------------------------------
  implicit none
  logical:: AMBER_interface_logic = .false.   ! flag to enable AMBER-quick interface 
  logical:: quick_first_call = .true.         ! flag to indicate if quick is first called

  contains
    
end module amber_interface_module
