#include "util.fh"
!---------------------------------------------------------------------!
! Created by Etienne Palos on   xx/xx/2024                            !
!                                                                     ! 
! Copyright (C) 2024-2025 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains all one electron properties code.              ! 
! This includes quantities required for polarizable QM/MM.            !
!---------------------------------------------------------------------!

module quick_oeproperties_module
  implicit double precision(a-h,o-z)
  private

  public some_function
  public :: electrostatic_potential , print_esp

contains

integer function some_function()
write(6,'(a)') 'testing something'
some_function = 0

end function

  !------------------------------------------------------
  ! One-electron properties realted to SCF claculations ! 
  !------------------------------------------------------

  !------------------------------------------------------
  ! Electrostatic potential (ESP) subroutine
  !           int d3r [phi_mu(r) * phi_nu(r)] / |r-R| 
  !------------------------------------------------------
subroutine electrostatic_potential(IIsh,JJsh, esp_array)
   use allmod
   use quick_overlap_module, only: gpt, opf, overlap_core
   implicit double precision(a-h,o-z)
   
   integer :: igridpoint
   
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g 
   double precision RA(3),RB(3),RP(3),inv_g,g_table(200), valopf
 
   double precision, dimension(:), intent(out) :: esp_array
 
   ! For quantities not explicitly commented, see attrashell

   ! Related to positions of "QM" atoms
   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))
 
   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))
 
   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2
 
   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
 
         valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)), &
         quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)
 
         if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
 
           !Eqn 14 O&S
           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,0,g_table)
           g = a+b
           !Eqn 15 O&S
           inv_g = 1.0d0 / dble(g)
 
           !Calculate first two terms of O&S Eqn A20
           constanttemp=dexp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 + (Az - Bz)**2.d0))*inv_g))
           constant = overlap_core(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) * 2.d0 * sqrt(g/Pi)*constanttemp
 
           !Related to positions of environment (ext points, or MM atoms)
           !nextpoint=number of external grid points. set to 0 if none used
           do igridpoint=1,quick_molspec%nextpoint
               Cx=quick_molspec%extxyz(1,igridpoint)
               Cy=quick_molspec%extxyz(2,igridpoint)
               Cz=quick_molspec%extxyz(3,igridpoint)

               ! dummy charge (value of Z in attrashell)
               Q=1.0d0
             constant2=constanttemp*Q
 
             !Calculate the last term of O&S Eqn A21
             PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
 
             !Compute O&S Eqn A21
               U = g* PCsquare
 
               !Calculate the last term of O&S Eqn A20
               call FmT(Maxm,U,aux)
 
               !Calculate all the auxilary integrals and store in attraxiao
               !array
               do L = 0,maxm
                  aux(L) = aux(L)*constant*Z
                  attraxiao(1,1,L)=aux(L)
               enddo
 
               NIJ1=10*NII2+NJJ2
 
               ! Calls nucelearattra as in attrashell but not passes the computed
               ! potential value to the ith element in esp_array
               call nuclearattra(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
                     Cx,Cy,Cz,Px,Py,Pz, esp_array(igridpoint))
 
           enddo
         endif
      enddo
   enddo
   201 return
 end subroutine electrostatic_potential

 subroutine print_esp(esp_array)
   double precision, dimension(:), intent(in) :: esp_array

   ! Output file handling
   integer :: unit_number
   character(len=15) :: output_file

   output_file = 'esp_data.dat'

   ! Open the file for writing
   open(newunit=unit_number, file=output_file, status='replace', action='write', iostat=iostatus)
   if (iostatus /= 0) then
      write(*,*) 'Error opening file ', output_file
      return
   end if

   ! Write esp_array values to the file
   write(unit_number, *) 'ESP_ARRAY:'
   write(unit_number, *) (esp_array(igridpoint), igridpoint=1,size(esp_array))

   ! Close the file
   close(unit_number)
 end subroutine print_esp

end module quick_oeproperties_module
