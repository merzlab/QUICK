#include "util.fh"
!---------------------------------------------------------------------!
! Created by Etienne Palos on  01/20/2024                             !
! Contributor: Vikrant Tripathy                                       !
!                                                                     !
! Purpose:  " Compute electrostatic properties on grid points "       !
!                                                                     !
! Capabilities:                                                       !
!              - ESP        Serial and MPI                            !
!              - EField     Serial                                    !
!                                                                     ! 
! Copyright (C) 2024-2025                                             !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!---------------------------------------------------------------------!
module quick_oeproperties_module
 private
 public :: compute_oeprop

 contains

!--------------------------------------------------------------------!
!  The following subroutine "compute_oeprop" is the only routine     !
!  called from outside. This routine performs calculations as per    !
!  user provided keywords.                                           !
!--------------------------------------------------------------------!

 Subroutine compute_oeprop()
   use quick_method_module, only: quick_method
   use quick_files_module, only : ioutfile

   implicit none

   if (quick_method%esp_grid .or. quick_method%efield_grid) then
      call compute_oeprop_grid()
   else
      write (ioutfile,'("  Skipping one-electron property calculation.")')
   end if

 end Subroutine

 Subroutine compute_oeprop_grid()
   use quick_molspec_module, only: quick_molspec
   use quick_method_module, only: quick_method
   use quick_mpi_module, only: master
   use quick_files_module, only: iESPFile, espFileName

   implicit none
   integer :: ierr
   double precision, allocatable :: esp_on_points(:)

   allocate(esp_on_points(quick_molspec%nextpoint))

   ierr = 0

   ! Electrostatic Potential
   if (quick_method%esp_grid) then
     call compute_esp(quick_molspec%nextpoint,quick_molspec%extpointxyz,esp_on_points)
     ! Print ESP at external points
     if (master) then
       call quick_open(iESPFile,espFileName,'U','F','R',.false.,ierr)
       call print_esp(esp_on_points, quick_molspec%nextpoint, quick_molspec%extpointxyz)
       close(iESPFile)
     endif
   end if

   ! Electric field
   if (quick_method%efield_grid) then
     call compute_efield()
   end if

   deallocate(esp_on_points)

 end Subroutine

!--------------------------------------------------------------------!
!   The subroutines esp_shell_pair, efield_shell_pair and            !
!   esp_1pdm, efield_1pdm are present in ./include/attrashell.fh     !
!   and ./include/nuclearattra.fh header files respectively.         !
!                                                                    !
!   The header files are called with OEPROP being defined.           !
!--------------------------------------------------------------------!

#define OEPROP
#include "./include/attrashell.fh"
#include "./include/nuclearattra.fh"
#undef OEPROP

 !----------------------------------------------------------------------------!
 ! This is the subroutine that "computes" the Electrostatic Potential (ESP)   !
 ! at a given point , V(r) = V_nuc(r) + V_elec(r), and prints it to file.prop !
 !                                                                            !
 ! This subroutine is called from the main program.                           !
 ! It calls the following subroutines:                                        !
 !     1. esp_nuc: Computes the nuclear contribution to the ESP               !
 !     2. esp_shell_pair: Computes the electronic contribution to the ESP     !
 !----------------------------------------------------------------------------!
 subroutine compute_esp(npoints,xyz_points,esp)
   use quick_timer_module, only : timer_begin, timer_end, timer_cumer
   use quick_molspec_module, only : quick_molspec
   use quick_basis_module, only: jshell
   use quick_mpi_module, only: master
 
   use quick_calculated_module, only: quick_qm_struct

#ifdef MPIV
    use mpi
    use quick_basis_module, only: mpi_jshelln, mpi_jshell
    use quick_mpi_module, only: mpirank, mpierror 
#endif

#if defined CUDA || defined CUDA_MPIV
    use quick_method_module, only: quick_method
#endif


   implicit none
   integer :: ierr
   integer :: IIsh, JJsh
   integer :: igridpoint, npoints

   double precision :: xyz_points(3,npoints), esp(npoints)

   double precision, allocatable :: esp_electronic(:)
   double precision, allocatable :: esp_nuclear(:)
#ifdef MPIV
   double precision, allocatable :: esp_electronic_aggregate(:)
#endif
   integer :: Ish

!   Write(6,*)quick_qm_struct%osave(1,1)

   ierr = 0
   
   ! Allocates ESP_NUC and ESP_ELEC arrays
   allocate(esp_nuclear(npoints))
   allocate(esp_electronic(npoints))
#ifdef MPIV
   allocate(esp_electronic_aggregate(npoints))
#endif

   ! ESP_ELEC array need initialization as we will be iterating
   ! over shells and updating ESP_ELEC.
   esp_electronic(:) = 0.0d0

   RECORD_TIME(timer_begin%TESPGrid)

   ! Computes ESP_NUC 
!   do igridpoint=1,npoints
   call esp_nuc(npoints, xyz_points, esp_nuclear)
!   end do

   ! Computes ESP_ELEC
#if defined CUDA || defined CUDA_MPIV
   call gpu_upload_oeprop(npoints, xyz_points, esp_electronic, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop(esp_electronic)
#if defined MPIV
   call MPI_REDUCE(esp_electronic, esp_electronic_aggregate, npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#endif
   ! Sum over contributions from different shell pairs
#elif defined MPIV
   ! MPI parallellization is performed over shell-pairs
   ! Different processes consider different shell-pairs
   do Ish=1,mpi_jshelln(mpirank)
      IIsh=mpi_jshell(mpirank,Ish)
      do JJsh=IIsh,jshell
         call esp_shell_pair(IIsh, JJsh, npoints, xyz_points, esp_electronic)
      enddo
   enddo
   ! MPI_REDUCE is called to sum over esp_electronic obtained from all the processes
   call MPI_REDUCE(esp_electronic, esp_electronic_aggregate, npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call esp_shell_pair(IIsh, JJsh, npoints, xyz_points, esp_electronic)
      end do
   end do
#endif

   ! Sum the nuclear and electronic part of ESP
   do igridpoint=1,npoints
#ifdef MPIV
     esp(igridpoint) = esp_nuclear(igridpoint)+esp_electronic_aggregate(igridpoint)
#else
     esp(igridpoint) = esp_nuclear(igridpoint)+esp_electronic(igridpoint)
#endif
   end do

   RECORD_TIME(timer_end%TESPGrid)
   timer_cumer%TESPGrid=timer_cumer%TESPGrid+timer_end%TESPGrid-timer_begin%TESPGrid

   RECORD_TIME(timer_begin%TESPCharge)

   if (master) then
     call compute_ESP_charge(esp)
   end if

   RECORD_TIME(timer_end%TESPCharge)
   timer_cumer%TESPCharge=timer_cumer%TESPCharge+timer_end%TESPCharge-timer_begin%TESPCharge

   deallocate(esp_electronic)
   deallocate(esp_nuclear)
#ifdef MPIV
   deallocate(esp_electronic_aggregate)
#endif
 end subroutine compute_esp

!----------------------------------------------------------!
!  Obtain ESP charge by solving:                           !
!             Aq=B                                         !
!  B is a column vector of dimension (natom+1)             !
!  A is a symmetric matrix of dimension (natom+1,natom+1)  !
!  q is a column vector of charges. Dimension: (natom+1)   !
!                                                          !
!  Only upper triangle of A is stored.                     !
!----------------------------------------------------------!

 subroutine compute_ESP_charge(esp)
   use quick_molspec_module, only: quick_molspec, natom, xyz
   use quick_files_module, only: ioutfile
   use quick_mpi_module, only: master
   use quick_exception_module, only: RaiseException

   implicit none

   integer, external :: ILAENV
   double precision, external :: rootSquare

   integer, allocatable :: IPIV(:)
   integer :: iatom, jatom, igridpoint, ierr, NB, LWORK, LDA
   double precision, intent(in) :: esp(quick_molspec%nextpoint)
   double precision, allocatable :: WORK(:)
   double precision :: A(natom+1,natom+1), B(natom+1), q(natom+1)
   double precision :: distance, distanceb, invdistance, Net_charge

   double precision :: One = 1.0d0, Zero = 0.0d0

!  A and B are initialized. A(natom+1,natom+1) is set to a small number
!  instead of zero to facilitate diagonalization of A.

   do iatom = 1, natom
     B(iatom) = 0
     do jatom = 1, natom
       A(jatom,iatom) = 0
     end do
   end do

   B(natom+1) = quick_molspec%molchg

   do jatom = 1, natom
     A(jatom,natom+1) = 1
   end do
   A(natom+1,natom+1) = 0

! The matrix A and vector B is formed.

   do iatom = 1, natom  
     do igridpoint = 1, quick_molspec%nextpoint
       distance = rootSquare(xyz(1:3,iatom), quick_molspec%extpointxyz(1:3,igridpoint), 3)
       invdistance = 1/distance
       B(iatom) = B(iatom) + esp(igridpoint) * invdistance
       do jatom = 1, iatom
         distanceb = rootSquare(xyz(1:3,jatom), quick_molspec%extpointxyz(1:3,igridpoint), 3)
         A(jatom,iatom) = A(jatom,iatom) + invdistance/distanceb
       end do
     end do
   end do

   call symmetrize('U',A,natom+1)

!  A is inverted.

   NB = ILAENV(1,'DGETRI',' ',natom+1,-1,-1,-1)

   LWORK = (natom+1)*NB

   allocate(WORK(LWORK))
   allocate(IPIV(natom+1))

   LDA = natom+1
   CALL dgetrf(natom+1,natom+1,A,LDA,IPIV,ierr)
   call DGETRI(natom+1,A,LDA,IPIV,WORK,LWORK,ierr)

   if (ierr /= 0) then
     ierr = 40
     call RaiseException(ierr)
   end if

!  q = A-1*B

   call DGEMV('N',natom+1,natom+1,One,A,LDA,B,1,Zero,q,1)

!  B is copied to charge array.

   Net_charge = 0.d0

   write (ioutfile,'("  ESP charges:")')
   write (ioutfile,'("  ----------------")')
   do iatom = 1, natom
     Net_charge = Net_charge + q(iatom)
     write (ioutfile,'(3x,I3,3x,F10.6)')iatom,q(iatom)
   end do
   write (ioutfile,'("  ----------------")')
   write (ioutfile,'("  Net charge = ",F10.6)')Net_charge
   write (ioutfile,'("  ")')

 end subroutine compute_ESP_charge

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the ESP data to "file.esp"                               !
 !---------------------------------------------------------------------------------------------!
 subroutine print_esp(esp, npoints, xyz_points)
   use quick_molspec_module, only: quick_molspec
   use quick_method_module, only: quick_method
   use quick_files_module, only: ioutfile,  iESPFile, espFileName
   use quick_constants_module, only: BOHRS_TO_A

   implicit none
   integer, intent(in) :: npoints

   double precision :: xyz_points(3,npoints)
   double precision :: esp(npoints)

   integer :: igridpoint
   double precision :: Cx, Cy, Cz

   ! If ESP_GRID is true, print to table X, Y, Z, V(r)
   write (ioutfile,'(" *** Printing Electrostatic Potential (ESP) [a.u.] at external points to ",A,x,"***")') &
           trim(espFileName)
   write (iESPFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) [atomic units] ")')
   write (iESPFile,'(100("-"))')
   if (quick_method%extgrid_angstrom)  then
     write (iESPFile,'(6x,"X[A]",10x ,"Y[A]",9x,"Z[A]",13x, "ESP_TOTAL [a.u.] ")')
   else
     ! Default is X, Y, and V_total in a.u.
     write (iESPFile,'(9x,"X",13x,"Y",12x,"Z",16x,"ESP")')
   endif

   ! Collect ESP and print
   do igridpoint = 1, npoints
     if (quick_method%extgrid_angstrom)  then
       Cx = (xyz_points(1, igridpoint)*BOHRS_TO_A)
       Cy = (xyz_points(2, igridpoint)*BOHRS_TO_A)
       Cz = (xyz_points(3, igridpoint)*BOHRS_TO_A)
     else
       Cx = xyz_points(1, igridpoint)
       Cy = xyz_points(2, igridpoint)
       Cz = xyz_points(3, igridpoint)
     endif
     write(iESPFile, '(2x,3(F14.10, 1x), 3F14.10)') Cx, Cy, Cz, esp(igridpoint)
   end do
 end subroutine print_esp

 !-----------------------------------------------------------------------!
 ! This subroutine calculates V_nuc(r) = sum Z_k/|r-Rk|                  !
 !-----------------------------------------------------------------------!
 subroutine esp_nuc(npoints, xyz_points, esp_nuclear)
   use quick_molspec_module, only: natom, quick_molspec, xyz

   implicit none
   integer, intent(in) :: npoints
   double precision, intent(in) :: xyz_points(3,npoints)
   double precision, intent(out) :: esp_nuclear(npoints)

   double precision :: distance
   double precision, external :: rootSquare
   integer :: inucleus, igridpoint

   do igridpoint = 1, npoints
     esp_nuclear(igridpoint) = 0.d0
     do inucleus=1,natom+quick_molspec%nextatom
       if(inucleus<=natom)then
         distance = rootSquare(xyz(1:3,inucleus), xyz_points(1:3,igridpoint), 3)
         esp_nuclear(igridpoint) = esp_nuclear(igridpoint) + quick_molspec%chg(inucleus) / distance
       else
         distance = rootSquare(quick_molspec%extxyz(1:3,inucleus-natom), xyz_points(1:3,igridpoint), 3)
         esp_nuclear(igridpoint) = esp_nuclear(igridpoint) + quick_molspec%extchg(inucleus-natom) / distance
       endif
     enddo
   enddo
 end subroutine esp_nuc


 !----------------------------------------------------------------------------------!
 ! This is the subroutine that "computes" the Electric Field (EFIELD)               !
 ! at a given point , E(x,y,z) = E_nuc(x,y,z) + E_elec(x,y,z), printing the         !
 ! result to file.efield                                                            !
 !                                                                                  !
 !----------------------------------------------------------------------------------!
 subroutine compute_efield()
  use quick_timer_module, only : timer_begin, timer_end, timer_cumer
  use quick_molspec_module, only : quick_molspec
  use quick_files_module, only : ioutfile, iPropFile, propFileName, iEFIELDFile, efieldFileName
  use quick_basis_module, only: jshell
  use quick_mpi_module, only: master

#ifdef MPIV
   use mpi
   use quick_basis_module, only: mpi_jshelln, mpi_jshell
   use quick_mpi_module, only: mpirank, mpierror
#endif

   implicit none
   integer :: ierr
   integer :: IIsh, JJsh
   integer :: igridpoint
  
   double precision, allocatable :: efield_electronic(:,:)
   double precision, allocatable :: efield_nuclear(:,:)
#ifdef MPIV
   double precision, allocatable :: efield_electronic_aggregate(:,:)
#endif
   integer :: Ish

   ierr = 0
  
   ! Allocates efield_nuclear and efield_electronic arrays
   allocate(efield_electronic(3,quick_molspec%nextpoint))
   allocate(efield_nuclear(3,quick_molspec%nextpoint))

#ifdef MPIV
   allocate(efield_electronic_aggregate(3,quick_molspec%nextpoint))
#endif

   ! Initilizes efield_electronic as it will be updated to account
   ! for contributions from different shell-pairs
   efield_electronic(:,:) = 0.0d0

   RECORD_TIME(timer_begin%TEFIELDGrid)

   ! Computes efield_nuclear 
   do igridpoint=1,quick_molspec%nextpoint
     call efield_nuc(igridpoint, efield_nuclear(1,igridpoint))
   end do

   ! Computes EField_ELEC by summing over contrbutions from individual shell-pairs

#ifdef MPIV
   do Ish=1,mpi_jshelln(mpirank)
      IIsh=mpi_jshell(mpirank,Ish)
      do JJsh=IIsh,jshell
         call efield_shell_pair(IIsh, JJsh, efield_electronic)
      enddo
   enddo
   call MPI_REDUCE(efield_electronic, efield_electronic_aggregate, 3 * quick_molspec%nextpoint, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call efield_shell_pair(IIsh, JJsh, efield_electronic)
      end do
   end do
#endif

   RECORD_TIME(timer_end%TEFIELDGrid)
   timer_cumer%TEFIELDGrid=timer_cumer%TEFIELDGrid+timer_end%TEFIELDGrid-timer_begin%TEFIELDGrid

   ! Sum the nuclear and electronic part of EField and print
   if (master) then
    ! for now, back to 'R' mode
     call quick_open(iEFIELDFile,efieldFileName,'U','F','R',.false.,ierr)
#ifdef MPIV
     call print_efield(efield_nuclear, efield_electronic_aggregate, quick_molspec%nextpoint)
#else
     call print_efield(efield_nuclear, efield_electronic, quick_molspec%nextpoint)
#endif
     close(iEFIELDFile)
   endif

   deallocate(efield_electronic)
   deallocate(efield_nuclear)
#ifdef MPIV
   deallocate(efield_electronic_aggregate)
#endif
 end subroutine compute_efield

!------------------------------------------------------------------------!
! This subroutine calculates EField_nuc(r) = sum Z_k*(r-Rk)/(|r-Rk|^3)   !
!------------------------------------------------------------------------!
 subroutine efield_nuc(igridpoint, efield_nuclear_term)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none

  integer, intent(in) :: igridpoint
  double precision, external :: rootSquare
  double precision, intent(out) :: efield_nuclear_term(3)

  double precision :: distance
  double precision :: inv_dist_cube, rx_nuc_gridpoint, ry_nuc_gridpoint, rz_nuc_gridpoint
  integer :: inucleus

  efield_nuclear_term = 0.0d0

  do inucleus = 1, natom+quick_molspec%nextatom
    if(inucleus<=natom)then
      distance = rootSquare(xyz(1:3, inucleus), quick_molspec%extpointxyz(1:3, igridpoint), 3)
      inv_dist_cube = 1.0d0/(distance**3)

      rx_nuc_gridpoint = (quick_molspec%extpointxyz(1, igridpoint) - xyz(1, inucleus))
      ry_nuc_gridpoint = (quick_molspec%extpointxyz(2, igridpoint) - xyz(2, inucleus))
      rz_nuc_gridpoint = (quick_molspec%extpointxyz(3, igridpoint) - xyz(3, inucleus))

     ! Compute nuclear components to EFIELD_NUCLEAR
      efield_nuclear_term(1) = efield_nuclear_term(1) + quick_molspec%chg(inucleus) * (rx_nuc_gridpoint * inv_dist_cube)
      efield_nuclear_term(2) = efield_nuclear_term(2) + quick_molspec%chg(inucleus) * (ry_nuc_gridpoint * inv_dist_cube)
      efield_nuclear_term(3) = efield_nuclear_term(3) + quick_molspec%chg(inucleus) * (rz_nuc_gridpoint * inv_dist_cube)
    else
      distance = rootSquare(quick_molspec%extxyz(1:3, inucleus-natom), quick_molspec%extpointxyz(1:3, igridpoint), 3)
      inv_dist_cube = 1.0d0/(distance**3)

      rx_nuc_gridpoint = (quick_molspec%extpointxyz(1, igridpoint) - quick_molspec%extxyz(1, inucleus-natom))
      ry_nuc_gridpoint = (quick_molspec%extpointxyz(2, igridpoint) - quick_molspec%extxyz(2, inucleus-natom))
      rz_nuc_gridpoint = (quick_molspec%extpointxyz(3, igridpoint) - quick_molspec%extxyz(3, inucleus-natom))

     ! Compute nuclear components to EFIELD_NUCLEAR
      efield_nuclear_term(1) = efield_nuclear_term(1) + quick_molspec%extchg(inucleus-natom) * (rx_nuc_gridpoint * inv_dist_cube)
      efield_nuclear_term(2) = efield_nuclear_term(2) + quick_molspec%extchg(inucleus-natom) * (ry_nuc_gridpoint * inv_dist_cube)
      efield_nuclear_term(3) = efield_nuclear_term(3) + quick_molspec%extchg(inucleus-natom) * (rz_nuc_gridpoint * inv_dist_cube)
    endif
  end do

end subroutine efield_nuc

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the EFIELD data to file.efield                           !
 !---------------------------------------------------------------------------------------------!
subroutine print_efield(efield_nuclear, efield_electronic, nextpoint)
  use quick_molspec_module, only: quick_molspec
  use quick_method_module, only: quick_method
  use quick_files_module, only: ioutfile, iPropFile, propFileName, iEFIELDFile, efieldFileName
  use quick_constants_module, only: BOHRS_TO_A

  implicit none
  integer, intent(in) :: nextpoint

  double precision :: efield_nuclear(3,nextpoint), efield_electronic(3,nextpoint)

  integer :: igridpoint
  double precision :: Cx, Cy, Cz

  ! If ESP_GRID is true, print to table X, Y, Z, V(r)
  write (ioutfile,'(" *** Printing Electric Field (EFIELD) [a.u.] on grid ",A,x,"***")') trim(efieldFileName)
  write (iEFIELDFile,'(/," ELECTRIC FIELD CALCULATION (EFIELD) [atomic units] ")')
  write (iEFIELDFile,'(100("-"))')

  if (quick_method%efield_grid) then
    write (iEFIELDFile,'(9x,"X",13x,"Y",12x,"Z",16x, "EFIELD_X",12x, "EFIELD_Y",8x,"EFIELD_Z")')
  else if (quick_method%efield_esp) then
      write (iEFIELDFile,'(9x,"X",13x,"Y",12x,"Z",16x, "ESP",8x, "EFIELD_X",12x, "EFIELD_Y",8x,"EFIELD_Z")')
  endif

  ! Collect ESP and print
  do igridpoint = 1, nextpoint 
    if (quick_method%extgrid_angstrom)  then
      Cx = (quick_molspec%extpointxyz(1, igridpoint)*BOHRS_TO_A)
      Cy = (quick_molspec%extpointxyz(2, igridpoint)*BOHRS_TO_A)
      Cz = (quick_molspec%extpointxyz(3, igridpoint)*BOHRS_TO_A)
    else
      Cx = quick_molspec%extpointxyz(1, igridpoint)
      Cy = quick_molspec%extpointxyz(2, igridpoint)
      Cz = quick_molspec%extpointxyz(3, igridpoint)
    endif

    ! Sum nuclear and electric components of EField and print.
    if (quick_method%efield_grid) then
      write(iEFIELDFile, '(3x,ES14.6,3x,ES14.6,3x,ES14.6)') &
      (efield_nuclear(1,igridpoint)+efield_electronic(1,igridpoint)), &
      (efield_nuclear(2,igridpoint)+efield_electronic(2,igridpoint)), &
      (efield_nuclear(3,igridpoint)+efield_electronic(3,igridpoint))
    endif

  end do
end subroutine print_efield

end module quick_oeproperties_module
