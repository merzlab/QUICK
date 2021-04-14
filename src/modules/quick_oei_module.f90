#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 03/24/2021                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains all one electron integral (oei) & oei gradient ! 
! code.                                                               !
!---------------------------------------------------------------------!

module quick_oei_module

  implicit double precision(a-h,o-z)
  private

  public :: get1eEnergy, attrashellopt

contains

  !------------------------------------------------
  ! get1eEnergy
  !------------------------------------------------
  subroutine get1eEnergy()
     !------------------------------------------------
     ! This subroutine is to get 1e integral
     !------------------------------------------------
     use quick_calculated_module, only: quick_qm_struct
     use quick_method_module,only: quick_method
     use quick_timer_module, only:timer_begin, timer_end, timer_cumer
     use quick_basis_module

     implicit double precision(a-h,o-z)

     call cpu_time(timer_begin%tE)
  
     quick_qm_struct%Eel=0.d0
     call copySym(quick_qm_struct%o,nbasis)
     quick_qm_struct%Eel=quick_qm_struct%Eel+sum2mat(quick_qm_struct%dense,quick_qm_struct%o,nbasis)

     if (quick_method%unrst) then
       call copySym(quick_qm_struct%ob,nbasis)
       quick_qm_struct%Eel = quick_qm_struct%Eel+sum2mat(quick_qm_struct%denseb,quick_qm_struct%ob,nbasis)
     endif

     call cpu_time(timer_end%tE)
     timer_cumer%TE=timer_cumer%TE+timer_end%TE-timer_begin%TE
  
  end subroutine get1eEnergy

  subroutine attrashellopt(IIsh,JJsh)
     use allmod
     !    use xiaoconstants
     implicit double precision(a-h,o-z)
     dimension aux(0:20)
     double precision AA(3),BB(3),CC(3),PP(3)
     common /xiaoattra/attra,aux,AA,BB,CC,PP,g
  
     double precision RA(3),RB(3),RP(3), valopf
#ifdef MPIV
     include "mpif.h"
#endif
  
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
  
             do iatom=1,natom+quick_molspec%nextatom
                if(quick_basis%katom(IIsh).eq.iatom.and.quick_basis%katom(JJsh).eq.iatom)then
                    continue
                 else
                   if(iatom<=natom)then
                    Cx=xyz(1,iatom)
                    Cy=xyz(2,iatom)
                    Cz=xyz(3,iatom)
                    Z=-1.0d0*quick_molspec%chg(iatom)
                   else
                    Cx=quick_molspec%extxyz(1,iatom-natom)
                    Cy=quick_molspec%extxyz(2,iatom-natom)
                    Cz=quick_molspec%extxyz(3,iatom-natom)
                    Z=-1.0d0*quick_molspec%extchg(iatom-natom)
                   endif
  
                   PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
  
                   U = g* PCsquare
                   !    Maxm = i+j+k+ii+jj+kk
                   call FmT(Maxm,U,aux)
                   do L = 0,maxm
                      aux(L) = aux(L)*constant*Z
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
  
                   call nuclearattraopt(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
  
                         Cx,Cy,Cz,Px,Py,Pz,iatom)
  
                endif
  
             enddo
           endif
        enddo
     enddo
  
     ! Xiao HE remember to multiply Z   01/12/2008
     !    attraction = attraction*(-1.d0)* Z
     return
  end subroutine attrashellopt

  
end module quick_oei_module
