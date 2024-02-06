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

module quick_oeproperties_module
implicit double precision(a-h,o-z)
private 

public :: esp_1pdm, electrostatic_potential, compute_esp, print_esp, logger

contains

subroutine esp_1pdm(Ips,Jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,esp)
!-----------------------------------------------------------------------------------------!
! Computes the 1 particle contribution to the ESP: P_{mu nu} V_{mu nu}                    !
! Specifically, see how the Coulomb integral is defiend for cartesian gaussian functions  !
! See Eq. A14 of O&S [J. Chem. Phys. 84, 3963 (1986)]                                     !
! First, calculates 〈 psi_mu | psi_nu 〉 for all mu and nu                               !
! 〈 psi_mu | 1/|r-R| | psi_nu 〉 ; let R =  1/|r-R|                                      !
!  P_{mu nu} * 〈 psi_mu | R | psi_nu 〉                                                  !
!-----------------------------------------------------------------------------------------!
   use allmod

   implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision, intent(inout) :: esp
   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g


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

   select case (NIJ1)

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
   case(12)
      call SPattra(0)
      call SPattra(1)
      call SPattra(2)
      call SDattra(0)
      call SDattra(1)
      call PDattra(0)
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
   end select

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

         Xconstant=X1temp*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)
         do III=III1,III2
            itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
            do JJJ=max(III,JJJ1),JJJ2
               itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))

              ! esp <--- P_{mu nu} * (mu | 1/C | nu) contribution
              esp= esp + quick_qm_struct%dense(JJJ,III)* &
               Xconstant*quick_basis%cons(III)*quick_basis%cons(JJJ)*attraxiao(itemp1,itemp2,0)
            enddo
         enddo

      enddo
   enddo
   201 return
End subroutine esp_1pdm

!---------------------------------------------------------------!
!      Electrostatic Potential (ESP) on grid subroutine         !
!---------------------------------------------------------------!
subroutine electrostatic_potential(IIsh, JJsh, esp_array)
   use allmod
   use quick_overlap_module, only: gpt, opf, overlap_core
   use quick_files_module, only: ioutfile
   use quick_mpi_module, only: master
   implicit double precision(a-h,o-z)
   
   integer :: igridpoint
!   integer, intent(out) :: ierr

!   logical :: debug = .true.
   
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g 
   double precision RA(3),RB(3),RP(3),inv_g,g_table(200), valopf
 
   double precision, dimension(:), intent(out) :: esp_array

!   if (debug) then
!      call logger('electrostatic_potential', 'enter')
!   end if
 
   esp_array(:) = 0.0d0

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
              

               ! Calls esp_1pdm : P_{mu nu} * 〈 psi_mu | R | psi_nu 〉 to compute V(r_i) = sum_{i} V_i at each point

               call esp_1pdm(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
                     Cx,Cy,Cz,Px,Py,Pz, esp_array(igridpoint))
            enddo

         endif
      enddo
   enddo
end subroutine electrostatic_potential

!! new stuff

subroutine compute_esp(esp_array, ierr)
   use quick_files_module, only: ioutfile
   use quick_mpi_module, only: master
!  implicit none
   implicit double precision(a-h,o-z)
   
   double precision, intent(out) :: esp_array(:)

   integer, intent(out) :: ierr
    logical :: debug = .true.

    if (debug) then
      call logger('compute_esp', 'enter')
   end if

   if (debug) then
        write(ioutfile, '(a)') '>>>DEBUG entered subroutine compute_esp'
   end if

   call electrostatic_potential(IIsh, JJsh, esp_array)

   call print_esp(esp_array, ierr)
end subroutine compute_esp

!subroutine print_esp(ierr)
    
!    implicit none
!   integer, intent(out) :: ierr

!    logical :: debug = .true.

!    if (debug) then
!       call logger('print_esp', 'enter')
!    end if

    ! NEED TO IMPLEMENT

!        if (debug) then
!       call logger('print_esp', 'exit')
!    end if

!  end subroutine print_esp

subroutine print_esp(esp_array, ierr)
    use quick_files_module, only: ioutfile
    implicit none
    integer, intent(out) :: ierr
    logical :: debug = .true.
    double precision, dimension(:), intent(in) :: esp_array(:)
    
    integer :: igridpoint

    if (debug) then
        write(ioutfile, '(a)') '>>> DEBUG Entering print_esp subroutine'
    endif

    ! Loop over the elements of esp_array and print them to the output file
    do igridpoint = 1, size(esp_array)
        write(ioutfile, *) esp_array(igridpoint)
    end do

    if (debug) then
        write(ioutfile, '(a)') '>>> DEBUG Exiting print_esp subroutine'
    endif
end subroutine print_esp

subroutine logger(name, status)

    use quick_files_module
    implicit none
    character (len=*), intent(in) :: name
    character (len=*), intent(in) :: status
    write(ioutfile, '(3(a,x))') '>>> DEBUG', name, status
  
end subroutine logger

end module quick_oeproperties_module
