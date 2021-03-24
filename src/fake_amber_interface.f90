#include "util.fh"
!
!	fake_amber_interface.f90
!	fake_amber_interface
!
!	Created by Yipu Miao on 1/19/11.
!	Copyright 2011 University of Florida. All rights reserved.
!


subroutine qm2_quick_energy(escf,scf_mchg)
! quick mod
   use allmod
   implicit none
   

   double precision   :: escf
   double precision   :: scf_mchg(1)

   return
end subroutine qm2_quick_energy

!-------------------------------
! Read mol info from Amber
!-------------------------------
subroutine read_AMBER_crd

end subroutine read_AMBER_crd

!--------------------------------
! read charge info from AMBER to quick
!--------------------------------
subroutine read_AMBER_charge

end subroutine read_AMBER_charge


!--------------------------------------------
! connect AMBER namelist with quick
!--------------------------------------------
subroutine read_AMBER_job
    
end subroutine read_AMBER_job

!
subroutine AMBER_interface_get_qm_forces(dxyzqm)
      implicit none
      double precision dxyzqm(1,1)
    
end subroutine AMBER_interface_get_qm_forces
