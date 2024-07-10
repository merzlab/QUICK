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
 public :: compute_esp, compute_efield

 contains

 !----------------------------------------------------------------------------!
 ! This is the subroutine that "computes" the Electrostatic Potential (ESP)   !
 ! at a given point , V(r) = V_nuc(r) + V_elec(r), and prints it to file.prop !
 !                                                                            !
 ! This subroutine is called from the main program.                           !
 ! It calls the following subroutines:                                        !
 !     1. esp_nuc: Computes the nuclear contribution to the ESP               !
 !     2. esp_shell_pair: Computes the electronic contribution to the ESP     !
 !     3. print_esp: Prints the ESP to the output file.prop                   !
 !----------------------------------------------------------------------------!
 subroutine compute_esp(ierr)
   use quick_timer_module, only : timer_begin, timer_end, timer_cumer
   use quick_molspec_module, only : quick_molspec
   use quick_files_module, only : ioutfile, iPropFile, propFileName, iESPFile, espFileName
   use quick_basis_module, only: jshell
   use quick_mpi_module, only: master
 
#ifdef MPIV
    use mpi
    use quick_basis_module, only: mpi_jshelln, mpi_jshell
    use quick_mpi_module, only: mpirank, mpierror 
#endif

   implicit none
   integer, intent(out) :: ierr
   integer :: IIsh, JJsh
   integer :: igridpoint
   
   double precision, allocatable :: esp_electronic(:)
   double precision, allocatable :: esp_nuclear(:)
#ifdef MPIV
   double precision, allocatable :: esp_electronic_aggregate(:)
#endif
   integer :: Ish

   ierr = 0
   
   ! Allocates ESP_NUC and ESP_ELEC arrays
   allocate(esp_nuclear(quick_molspec%nextpoint))
   allocate(esp_electronic(quick_molspec%nextpoint))
#ifdef MPIV
   allocate(esp_electronic_aggregate(quick_molspec%nextpoint))
#endif

   ! ESP_ELEC array need initialization as we will be iterating
   ! over shells and updating ESP_ELEC.
   esp_electronic(:) = 0.0d0

   RECORD_TIME(timer_begin%TESPGrid)

   ! Computes ESP_NUC 
   do igridpoint=1,quick_molspec%nextpoint
     call esp_nuc(ierr, igridpoint, esp_nuclear(igridpoint))
   end do

   ! Computes ESP_ELEC
   ! Sum over contributions from different shell pairs
#ifdef MPIV
   ! MPI parallellization is performed over shell-pairs
   ! Different processes consider different shell-pairs
   do Ish=1,mpi_jshelln(mpirank)
      IIsh=mpi_jshell(mpirank,Ish)
      do JJsh=IIsh,jshell
         call esp_shell_pair(IIsh, JJsh, esp_electronic)
      enddo
   enddo
   ! MPI_REDUCE is called to sum over esp_electronic obtained from all the processes
   call MPI_REDUCE(esp_electronic, esp_electronic_aggregate, quick_molspec%nextpoint, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call esp_shell_pair(IIsh, JJsh, esp_electronic)
      end do
   end do
#endif

   RECORD_TIME(timer_end%TESPGrid)
   timer_cumer%TESPGrid=timer_cumer%TESPGrid+timer_end%TESPGrid-timer_begin%TESPGrid

   ! Sum the nuclear and electronic part of ESP and print
   if (master) then
     call quick_open(iESPFile,espFileName,'U','F','R',.false.,ierr)
#ifdef MPIV
     call print_esp(esp_nuclear,esp_electronic_aggregate, quick_molspec%nextpoint, ierr)
#else
     call print_esp(esp_nuclear,esp_electronic, quick_molspec%nextpoint, ierr)
#endif
     close(iESPFile)
   endif

   deallocate(esp_electronic)
   deallocate(esp_nuclear)
#ifdef MPIV
   deallocate(esp_electronic_aggregate)
#endif
 end subroutine compute_esp

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the ESP data to "file.esp"                               !
 !---------------------------------------------------------------------------------------------!
 subroutine print_esp(esp_nuclear, esp_electronic, nextpoint, ierr)
   use quick_molspec_module, only: quick_molspec
   use quick_method_module, only: quick_method
   use quick_files_module, only: ioutfile, iPropFile, propFileName,  iESPFile, espFileName
   use quick_constants_module, only: BOHRS_TO_A

   implicit none
   integer, intent(out) :: ierr
   integer, intent(in) :: nextpoint

   double precision :: esp_nuclear(nextpoint)
   double precision :: esp_electronic(nextpoint)

   integer :: igridpoint
   double precision :: Cx, Cy, Cz

   ! If ESP_GRID is true, print to table X, Y, Z, V(r)
   write (ioutfile,'(" *** Printing Electrostatic Potential (ESP) [a.u.] at external points to ",A,x,"***")') &
           trim(espFileName)
   write (iESPFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) [atomic units] ")')
   write (iESPFile,'(100("-"))')
   ! Do you want V_nuc and V_elec?
   if (quick_method%esp_print_terms) then
     write (iESPFile,'(9x,"X",13x,"Y",12x,"Z",16x, "ESP_NUC",12x, "ESP_ELEC",8x,"ESP_TOTAL")')
     ! Do you want  X, Y, and Z in Angstrom?
   else if (quick_method%extgrid_angstrom)  then
     write (iESPFile,'(6x,"X[A]",10x ,"Y[A]",9x,"Z[A]",13x, "ESP_TOTAL [a.u.] ")')
   else
     ! Default is X, Y, and V_total in a.u.
     write (iESPFile,'(9x,"X",13x,"Y",12x,"Z",16x,"ESP")')
   endif

   ! Collect ESP and print
   do igridpoint = 1, nextpoint 
     if (quick_method%extgrid_angstrom)  then
       Cx = (quick_molspec%extxyz(1, igridpoint)*BOHRS_TO_A)
       Cy = (quick_molspec%extxyz(2, igridpoint)*BOHRS_TO_A)
       Cz = (quick_molspec%extxyz(3, igridpoint)*BOHRS_TO_A)
     else
       Cx = quick_molspec%extxyz(1, igridpoint)
       Cy = quick_molspec%extxyz(2, igridpoint)
       Cz = quick_molspec%extxyz(3, igridpoint)
     endif

     ! Additional option 1 : PRINT ESP_NUC, ESP_ELEC, and ESP_TOTAL
     if (quick_method%esp_print_terms) then
       write(iESPFile, '(2x,3(F14.10, 1x), 3x,F14.10,3x,F14.10,3x,3F14.10)') Cx, Cy, Cz,  &
       esp_nuclear(igridpoint), esp_electronic(igridpoint), (esp_nuclear(igridpoint)+esp_electronic(igridpoint))
     else
       write(iESPFile, '(2x,3(F14.10, 1x), 3F14.10)') Cx, Cy, Cz,  &
         (esp_nuclear(igridpoint)+esp_electronic(igridpoint))
     endif

   end do
 end subroutine print_esp

 !-----------------------------------------------------------------------!
 ! This subroutine calculates V_nuc(r) = sum Z_k/|r-Rk|                  !
 !-----------------------------------------------------------------------!
 subroutine esp_nuc(ierr, igridpoint, esp_nuclear_term)
   use quick_molspec_module, only: natom, quick_molspec, xyz

   implicit none
   integer, intent(inout) :: ierr

   double precision :: distance
   double precision, external :: rootSquare
   integer inucleus

   double precision, intent(out) :: esp_nuclear_term
   integer ,intent(in) :: igridpoint
   
   esp_nuclear_term = 0.d0

     do inucleus=1,natom
        distance = rootSquare(xyz(1:3,inucleus), quick_molspec%extxyz(1:3,igridpoint), 3)
        esp_nuclear_term = esp_nuclear_term + quick_molspec%chg(inucleus) / distance
     enddo
 end subroutine esp_nuc


 !----------------------------------------------------------------------------------!
 ! This is the subroutine that "computes" the Electric Field (EFIELD)               !
 ! at a given point , E(x,y,z) = E_nuc(x,y,z) + E_elec(x,y,z), printing the         !
 ! result to file.efield                                                            !
 !                                                                                  !
 !----------------------------------------------------------------------------------!
 subroutine compute_efield(ierr)
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
   integer, intent(out) :: ierr
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
     call efield_nuc(ierr, igridpoint, efield_nuclear(1,igridpoint))
   end do

   ! Computes EField_ELEC by summing over contrbutions from individual shell-pairs

#ifdef MPIV
!   do Ish=1,mpi_jshelln(mpirank)
!      IIsh=mpi_jshell(mpirank,i)
!      do JJsh=IIsh,jshell
!         call esp_shell_pair(IIsh, JJsh, esp_electronic)
!      enddo
!   enddo
!   call MPI_REDUCE(esp_electronic, esp_electronic_aggregate, quick_molspec%nextpoint, &
!     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
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

     call print_efield(efield_nuclear, efield_electronic, quick_molspec%nextpoint, ierr)
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
 subroutine efield_nuc(ierr, igridpoint, efield_nuclear_term)
  use quick_molspec_module
  implicit none

  integer, intent(inout) :: ierr
  integer, intent(in) :: igridpoint
  double precision, external :: rootSquare
  double precision, intent(out) :: efield_nuclear_term(3)

  double precision :: distance
  double precision :: inv_dist_cube, rx_nuc_gridpoint, ry_nuc_gridpoint, rz_nuc_gridpoint
  integer :: inucleus

  efield_nuclear_term = 0.0d0

  do inucleus = 1, natom
      distance = rootSquare(xyz(1:3, inucleus), quick_molspec%extxyz(1:3, igridpoint), 3)
      inv_dist_cube = 1.0d0/(distance**3)

      rx_nuc_gridpoint = (quick_molspec%extxyz(1, igridpoint) - xyz(1, inucleus))
      ry_nuc_gridpoint = (quick_molspec%extxyz(2, igridpoint) - xyz(2, inucleus))
      rz_nuc_gridpoint = (quick_molspec%extxyz(3, igridpoint) - xyz(3, inucleus))

     ! Compute nuclear components to EFIELD_NUCLEAR
      efield_nuclear_term(1) = efield_nuclear_term(1) + quick_molspec%chg(inucleus) * (rx_nuc_gridpoint * inv_dist_cube)
      efield_nuclear_term(2) = efield_nuclear_term(2) + quick_molspec%chg(inucleus) * (ry_nuc_gridpoint * inv_dist_cube)
      efield_nuclear_term(3) = efield_nuclear_term(3) + quick_molspec%chg(inucleus) * (rz_nuc_gridpoint * inv_dist_cube)
  end do

end subroutine efield_nuc

 !----------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the EFIELD data to file.efield                            !
 !----------------------------------------------------------------------------------------------!
subroutine print_efield(efield_nuclear, efield_electronic, nextpoint, ierr)
  use quick_molspec_module, only: quick_molspec
  use quick_method_module, only: quick_method
  use quick_files_module, only: ioutfile, iPropFile, propFileName, iEFIELDFile, efieldFileName
  use quick_constants_module, only: BOHRS_TO_A

  implicit none
  integer, intent(out) :: ierr
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
      Cx = (quick_molspec%extxyz(1, igridpoint)*BOHRS_TO_A)
      Cy = (quick_molspec%extxyz(2, igridpoint)*BOHRS_TO_A)
      Cz = (quick_molspec%extxyz(3, igridpoint)*BOHRS_TO_A)
    else
      Cx = quick_molspec%extxyz(1, igridpoint)
      Cy = quick_molspec%extxyz(2, igridpoint)
      Cz = quick_molspec%extxyz(3, igridpoint)
    endif

    ! Sum nuclear and electric components of EField and print.
    if (quick_method%efield_grid) then
      write(iEFIELDFile, '(3x,ES14.6,3x,ES14.6,3x,ES14.6)') &
      efield_nuclear(1,igridpoint)+efield_electronic(1,igridpoint), &
      efield_nuclear(2,igridpoint)+efield_electronic(2,igridpoint), &
      efield_nuclear(3,igridpoint)+efield_electronic(3,igridpoint)
    endif

  end do
end subroutine print_efield

!--------------------------------------------------------------------!
!   The subroutines esp_shell_pair and esp_1pdm are present in       !
!   ./include/attrashell.fh and ./include/nuclearattra.fh include    !
!   files respectively.                                              !
!                                                                    !
!   The header files are called with OEPROP being defined.           !
!--------------------------------------------------------------------!

#define OEPROP
#include "./include/attrashell.fh"
#include "./include/nuclearattra.fh"
#undef OEPROP

subroutine efield_1pdm(Ips,Jps,IIsh,JJsh,NIJ1, &
      Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,efield)
   use quick_params_module, only: trans
   use quick_method_module, only: quick_method
   use quick_basis_module, only: quick_basis, attraxiaoopt
   use quick_calculated_module, only : quick_qm_struct

   implicit none

   double precision attra,aux(0:20)

   integer a(3),b(3)
   integer Ips, Jps, IIsh, JJsh, NIJ1
   integer iA,iB,III,III1,III2,JJJ,JJJ1,JJJ2,NBI1,NBI2,NBJ1,NBJ2
   integer Iang,Jang,iAstart,iBstart,itemp,itemp1,itemp2,itempt
   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g,DENSEJI
   double precision AA(3),BB(3),CC(3),PP(3)
   double precision Agrad1,Agrad2,Agrad3,Bgrad1,Bgrad2,Bgrad3,Cgrad1,Cgrad2,Cgrad3
   double precision X1temp,Xconstant,Xconstant1,Xconstant2
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   double precision, intent(inout) :: efield(3)

   AA(1)=Ax
   AA(2)=Ay
   AA(3)=Az
   BB(1)=Bx
   BB(2)=By
   BB(3)=Bz
   CC(1)=Cx
   CC(2)=Cy
   CC(3)=Cz
   PP(1)=Px
   PP(2)=Py
   PP(3)=Pz


   select case (NIJ1+11)

   case(0)
   case(10)
      call PSattra(0)
   case(1)
      call SPattra(0)
   case(11)
      call SPattra(0)
      call PSattra(0)
      call PSattra(1)
      call PPattra(0)
   case(20)
      call PSattra(0)
      call PSattra(1)
      call DSattra(0)
   case(2)
      call SPattra(0)
      call SPattra(1)
      call SDattra(0)
   case(21)
      call PSattra(0)
      call PSattra(1)
      call PSattra(2)
      call DSattra(0)
      call DSattra(1)
      call DPattra(0)
      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call PPattra(itemp)
      enddo
   case(12)
      call SPattra(0)
      call SPattra(1)
      call SPattra(2)
      call SDattra(0)
      call SDattra(1)
      call PDattra(0)
      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call PPattra(itemp)
      enddo
   case(22)
      do itempt=0,3
         call PSattra(itempt)
      enddo
      do itempt=0,1
         call PPattra(itempt)
      enddo
      do itempt=0,2
         call DSattra(itempt)
      enddo
      do itempt=0,1
         call DPattra(itempt)
      enddo

      call DDattra(0)

      ! new
      do itemp=0,3
         call SPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo

   case(30)

      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call DSattra(itemp)
      enddo

      call FSattra(0)

   case(3)

      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call SDattra(itemp)
      enddo

      call SFattra(0)

   case(31)

      do itemp=0,3
         call PSattra(itemp)
         call SPattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo

      call FPattra(0)

   case(13)

      do itemp=0,3
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo

      call PFattra(0)



   case(32)

      do itemp=0,4
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo

      call FDattra(0)

      do itemp=0,4
         call SPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call DDattra(itemp)
      enddo

   case(23)

      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo

      call DFattra(0)
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call DDattra(itemp)
      enddo


   case(33)

      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo

      call FFattra(0)

      do itemp=0,5
         call SPattra(itemp)
      enddo
      do itemp=0,4
         call SDattra(itemp)
      enddo
      do itemp=0,3
         call PDattra(itemp)
      enddo
      do itemp=0,3
         call SFattra(itemp)
      enddo
      do itemp=0,2
         call PFattra(itemp)
      enddo
      do itemp=0,1
         call DFattra(itemp)
      enddo

   case(40)

      do itemp=0,3
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo

      call GSattra(0)

   case(4)

      do itemp=0,3
         call SPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo

      call SGattra(0)

   case(41)

      do itemp=0,4
         call PSattra(itemp)
         call SPattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call GSattra(itemp)
      enddo

      call GPattra(0)

   case(14)

      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo
      do itemp=0,1
         call SGattra(itemp)
      enddo

      call PGattra(0)


   case(42)

      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo
      do itemp=0,2
         call GSattra(itemp)
      enddo
      do itemp=0,1
         call GPattra(itemp)
      enddo

      call GDattra(0)

      ! new
      do itemp=0,5
         call SPattra(itemp)
      enddo
      do itemp=0,4
         call SDattra(itemp)
      enddo
      do itemp=0,3
         call PDattra(itemp)
      enddo


   case(24)

      do itemp=0,5
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call SDattra(itemp)
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call PDattra(itemp)
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call SFattra(itemp)
      enddo
      do itemp=0,2
         call PFattra(itemp)
      enddo
      do itemp=0,1
         call DFattra(itemp)
      enddo
      do itemp=0,2
         call SGattra(itemp)
      enddo
      do itemp=0,1
         call PGattra(itemp)
      enddo

      call DGattra(0)

   case(43)

      do itemp=0,6
         call PSattra(itemp)
      enddo
      do itemp=0,5
         call PPattra(itemp)
      enddo
      do itemp=0,5
         call DSattra(itemp)
      enddo
      do itemp=0,4
         call DPattra(itemp)
      enddo
      do itemp=0,3
         call DDattra(itemp)
      enddo
      do itemp=0,4
         call FSattra(itemp)
      enddo
      do itemp=0,3
         call FPattra(itemp)
      enddo
      do itemp=0,2
         call FDattra(itemp)
      enddo
      do itemp=0,1
         call FFattra(itemp)
      enddo
      do itemp=0,3
         call GSattra(itemp)
      enddo
      do itemp=0,2
         call GPattra(itemp)
      enddo
      do itemp=0,1
         call GDattra(itemp)
      enddo
      call GFattra(0)

      ! new
      do itemp=0,6
         call SPattra(itemp)
      enddo
      do itemp=0,5
         call SDattra(itemp)
      enddo
      do itemp=0,4
         call PDattra(itemp)
      enddo
      do itemp=0,4
         call SFattra(itemp)
      enddo
      do itemp=0,3
         call PFattra(itemp)
      enddo
      do itemp=0,2
         call DFattra(itemp)
      enddo


   case(34)

      do itemp=0,6
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,5
         call PPattra(itemp)
      enddo
      do itemp=0,5
         call SDattra(itemp)
         call DSattra(itemp)
      enddo
      do itemp=0,4
         call PDattra(itemp)
         call DPattra(itemp)
      enddo
      do itemp=0,3
         call DDattra(itemp)
      enddo
      do itemp=0,4
         call SFattra(itemp)
         call FSattra(itemp)
      enddo
      do itemp=0,3
         call PFattra(itemp)
         call FPattra(itemp)
      enddo
      do itemp=0,2
         call DFattra(itemp)
         call FDattra(itemp)
      enddo
      do itemp=0,1
         call FFattra(itemp)
      enddo
      do itemp=0,3
         call SGattra(itemp)
      enddo
      do itemp=0,2
         call PGattra(itemp)
      enddo
      do itemp=0,1
         call DGattra(itemp)
      enddo
      call FGattra(0)

   case(44)

      do itemp=0,7
         call PSattra(itemp)
      enddo
      do itemp=0,6
         call PPattra(itemp)
      enddo
      do itemp=0,6
         call DSattra(itemp)
      enddo
      do itemp=0,5
         call DPattra(itemp)
      enddo
      do itemp=0,4
         call DDattra(itemp)
      enddo
      do itemp=0,5
         call FSattra(itemp)
      enddo
      do itemp=0,4
         call FPattra(itemp)
      enddo
      do itemp=0,3
         call FDattra(itemp)
      enddo
      do itemp=0,2
         call FFattra(itemp)
      enddo
      do itemp=0,4
         call GSattra(itemp)
      enddo
      do itemp=0,3
         call GPattra(itemp)
      enddo
      do itemp=0,2
         call GDattra(itemp)
      enddo
      do itemp=0,1
         call GFattra(itemp)
      enddo
      call GGattra(0)

      ! new
      do itemp=0,7
         call SPattra(itemp)
      enddo
      do itemp=0,6
         call SDattra(itemp)
      enddo
      do itemp=0,5
         call PDattra(itemp)
      enddo
      do itemp=0,5
         call SFattra(itemp)
      enddo
      do itemp=0,4
         call PFattra(itemp)
      enddo
      do itemp=0,3
         call DFattra(itemp)
      enddo
      do itemp=0,4
         call SGattra(itemp)
      enddo
      do itemp=0,3
         call PGattra(itemp)
      enddo
      do itemp=0,2
         call DGattra(itemp)
      enddo
      do itemp=0,1
         call FGattra(itemp)
      enddo

   end select


   ! new opt
   select case (NIJ1)

   case(0)
   case(10)
      call PSattraopt(0)
   case(1)
      call SPattraopt(0)
   case(11)

      call SPattraopt(0)
      call PSattraopt(0)
      call PSattraopt(1)
      call PPattraopt(0)

   case(20)

      call PSattraopt(0)
      call PSattraopt(1)
      call DSattraopt(0)

   case(2)

      call SPattraopt(0)
      call SPattraopt(1)
      call SDattraopt(0)

   case(21)

      call PSattraopt(0)
      call PSattraopt(1)
      call PSattraopt(2)
      call DSattraopt(0)
      call DSattraopt(1)
      call DPattraopt(0)

   case(12)

      call SPattraopt(0)
      call SPattraopt(1)
      call SPattraopt(2)
      call SDattraopt(0)
      call SDattraopt(1)
      call PDattraopt(0)


   case(22)

      do itempt=0,3
         call PSattraopt(itempt)
      enddo
      do itempt=0,1
         call PPattraopt(itempt)
      enddo
      do itempt=0,2
         call DSattraopt(itempt)
      enddo
      do itempt=0,1
         call DPattraopt(itempt)
      enddo

      call DDattraopt(0)

   case(30)

      do itemp=0,2
         call PSattraopt(itemp)
      enddo
      do itemp=0,1
         call DSattraopt(itemp)
      enddo

      call FSattraopt(0)

   case(3)

      do itemp=0,2
         call SPattraopt(itemp)
      enddo
      do itemp=0,1
         call SDattraopt(itemp)
      enddo

      call SFattraopt(0)

   case(31)

      do itemp=0,3
         call PSattraopt(itemp)
      enddo
      do itemp=0,2
         call PPattraopt(itemp)
      enddo
      do itemp=0,2
         call DSattraopt(itemp)
      enddo
      do itemp=0,1
         call DPattraopt(itemp)
      enddo
      do itemp=0,1
         call FSattraopt(itemp)
      enddo

      call FPattraopt(0)

   case(13)

      do itemp=0,3
         call SPattraopt(itemp)
         call PSattraopt(itemp)
      enddo
      do itemp=0,2
         call PPattraopt(itemp)
      enddo
      do itemp=0,2
         call SDattraopt(itemp)
      enddo
      do itemp=0,1
         call PDattraopt(itemp)
      enddo
      do itemp=0,1
         call SFattraopt(itemp)
      enddo

      call PFattraopt(0)

   case(32)

      do itemp=0,4
         call PSattraopt(itemp)
      enddo
      do itemp=0,3
         call PPattraopt(itemp)
      enddo
      do itemp=0,3
         call DSattraopt(itemp)
      enddo
      do itemp=0,2
         call DPattraopt(itemp)
      enddo
      do itemp=0,2
         call FSattraopt(itemp)
      enddo
      do itemp=0,1
         call FPattraopt(itemp)
      enddo

      call FDattraopt(0)

   case(23)

      do itemp=0,4
         call SPattraopt(itemp)
         call PSattraopt(itemp)
      enddo
      do itemp=0,3
         call PPattraopt(itemp)
      enddo
      do itemp=0,3
         call SDattraopt(itemp)
      enddo
      do itemp=0,2
         call PDattraopt(itemp)
      enddo
      do itemp=0,2
         call SFattraopt(itemp)
      enddo
      do itemp=0,1
         call PFattraopt(itemp)
      enddo

      call DFattraopt(0)

   case(33)

      do itemp=0,5
         call PSattraopt(itemp)
      enddo
      do itemp=0,4
         call PPattraopt(itemp)
      enddo
      do itemp=0,4
         call DSattraopt(itemp)
      enddo
      do itemp=0,3
         call DPattraopt(itemp)
      enddo
      do itemp=0,2
         call DDattraopt(itemp)
      enddo
      do itemp=0,3
         call FSattraopt(itemp)
      enddo
      do itemp=0,2
         call FPattraopt(itemp)
      enddo
      do itemp=0,1
         call FDattraopt(itemp)
      enddo

      call FFattraopt(0)


   end select

   Agrad1=0.0d0
   Agrad2=0.0d0
   Agrad3=0.0d0
   Bgrad1=0.0d0
   Bgrad2=0.0d0
   Bgrad3=0.0d0
   Cgrad1=0.0d0
   Cgrad2=0.0d0
   Cgrad3=0.0d0

   do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
      X1temp=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
      do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
         NBI1=quick_basis%Qsbasis(IIsh,Iang)
         NBI2=quick_basis%Qfbasis(IIsh,Iang)
         NBJ1=quick_basis%Qsbasis(JJsh,Jang)
         NBJ2=quick_basis%Qfbasis(JJsh,Jang)

         III1=quick_basis%ksumtype(IIsh)+NBI1
         III2=quick_basis%ksumtype(IIsh)+NBI2
         JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
         JJJ2=quick_basis%ksumtype(JJsh)+NBJ2

         iA=quick_basis%katom(IIsh)
         iB=quick_basis%katom(JJsh)
!         iC=iatom

         iAstart = (iA-1)*3
         iBstart = (iB-1)*3
!         iCstart = (iC-1)*3

         Xconstant=X1temp*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)

         do III=III1,III2
            Xconstant1=Xconstant*quick_basis%cons(III)
            itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
            do JJJ=max(III,JJJ1),JJJ2

               if (quick_method%UNRST) then
                  DENSEJI=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)
               else
                  DENSEJI=quick_qm_struct%dense(JJJ,III)
               endif

               if(III.ne.JJJ)DENSEJI=2.0d0*DENSEJI
               Xconstant2=Xconstant1*quick_basis%cons(JJJ)*DENSEJI
               itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))

               Cgrad1=Cgrad1+Xconstant2*attraxiaoopt(1,itemp1,itemp2,0)
               Cgrad2=Cgrad2+Xconstant2*attraxiaoopt(2,itemp1,itemp2,0)
               Cgrad3=Cgrad3+Xconstant2*attraxiaoopt(3,itemp1,itemp2,0)

            enddo
         enddo
      enddo
   enddo

   efield(1) = efield(1)- CGrad1
   efield(2) = efield(2)- CGrad2
   efield(3) = efield(3)- CGrad3

End subroutine efield_1pdm

  subroutine efield_shell_pair(IIsh,JJsh,efield_electronic)
     use quick_overlap_module, only: opf, overlap
     use quick_molspec_module, only: quick_molspec, xyz
     use quick_basis_module, only: quick_basis, attraxiaoopt, attraxiao
     use quick_method_module, only: quick_method
     use quick_constants_module, only : Pi
     !    use xiaoconstants
#ifdef MPIV
     use mpi
#endif
     
     implicit none

     integer :: igridpoint
     integer :: IIsh, JJsh, ips, jps, L, Maxm, NII2, NIJ1, NJJ2
     double precision :: a, b, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, g, U
     double precision :: attra, constant, constanttemp, PCsquare, Px, Py, Pz

     double precision, dimension(0:20) :: aux
     double precision AA(3),BB(3),CC(3),PP(3)
     common /xiaoattra/attra,aux,AA,BB,CC,PP,g
  
     double precision RA(3),RB(3),RP(3), valopf, g_table(200)
  
     double precision, intent(inout) :: efield_electronic(3,quick_molspec%nextpoint)

     Ax=xyz(1,quick_basis%katom(IIsh))
     Ay=xyz(2,quick_basis%katom(IIsh))
     Az=xyz(3,quick_basis%katom(IIsh))
  
     Bx=xyz(1,quick_basis%katom(JJsh))
     By=xyz(2,quick_basis%katom(JJsh))
     Bz=xyz(3,quick_basis%katom(JJsh))
  
     ! The purpose of this subroutine is to calculate the nuclear attraction
     ! of an electron  distributed between gtfs with orbital exponents a
     ! and b on A and B with angular momentums defined by i,j,k (a's x, y
     ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
     ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
     ! arises from the recusion relationship.
  
     ! The this is taken from the recursive relation found in Obara and Saika,
     ! J. Chem. Phys. 84 (7) 1986, 3963.
  
     ! The first step is generating all the necessary auxillary integrals.
     ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
     ! The values of m range from 0 to i+j+k+ii+jj+kk.
  
     NII2=quick_basis%Qfinal(IIsh)
     NJJ2=quick_basis%Qfinal(JJsh)
     Maxm=NII2+NJJ2+1+1
  
     do ips=1,quick_basis%kprim(IIsh)
        a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
        do jps=1,quick_basis%kprim(JJsh)
           b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
  
           valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)),&
           quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)
  
           if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
  
             g = a+b
             Px = (a*Ax + b*Bx)/g
             Py = (a*Ay + b*By)/g
             Pz = (a*Az + b*Bz)/g
             g_table = g**(-1.5)
  
             constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
                   * 2.d0 * sqrt(g/Pi)
  
             do igridpoint=1,quick_molspec%nextpoint
                    Cx=quick_molspec%extxyz(1,igridpoint)
                    Cy=quick_molspec%extxyz(2,igridpoint)
                    Cz=quick_molspec%extxyz(3,igridpoint)
  
                   PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
  
                   U = g* PCsquare
                   !    Maxm = i+j+k+ii+jj+kk
                   call FmT(Maxm,U,aux)
                   do L = 0,maxm
                      aux(L) = -1.0d0*aux(L)*constant
                      attraxiao(1,1,L)=aux(L)
                   enddo
  
                   do L = 0,maxm-1
                      attraxiaoopt(1,1,1,L)=2.0d0*g*(Px-Cx)*aux(L+1)
                      attraxiaoopt(2,1,1,L)=2.0d0*g*(Py-Cy)*aux(L+1)
                      attraxiaoopt(3,1,1,L)=2.0d0*g*(Pz-Cz)*aux(L+1)
                   enddo
  
                   ! At this point all the auxillary integrals have been calculated.
                   ! It is now time to decompase the attraction integral to it's
                   ! auxillary integrals through the recursion scheme.  To do this we use
                   ! a recursive function.
  
                   !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
  
                         !    Cx,Cy,Cz,Px,Py,Pz,g)
                   NIJ1=10*NII2+NJJ2
  
                   call efield_1pdm(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
                         Cx,Cy,Cz,Px,Py,Pz,efield_electronic(1,igridpoint))
  
!                endif
  
             enddo
           endif
        enddo
     enddo
  
     ! Xiao HE remember to multiply Z   01/12/2008
     !    attraction = attraction*(-1.d0)* Z
     return
  end subroutine efield_shell_pair

end module quick_oeproperties_module
