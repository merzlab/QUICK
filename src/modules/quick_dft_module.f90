!---------------------------------------------------------------------!
! Created by Madu Manathunga on 03/26/2021                            !
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

! This module should contain stuff relevent for DFT
module quick_dft_module

  implicit double precision
  private

  public :: b3lypf, b3lyp_e, becke, becke_e, lyp, lyp_e 

contains

  subroutine b3lypf (rhoa1,sigmaaa1, vrhoa,vsigmaaa)
     !
     !     P.J. Stephens, F.J. Devlin, C.F. Chabalowski, M.J. Frisch
     !     Ab initio calculation of vibrational absorption and circular
     !     dichroism spectra using density functional force fields
     !     J. Phys. Chem. 98 (1994) 11623-11627
     !
     !
     !     CITATION:
     !
     !     Functionals were obtained from the Density Functional Repository
     !     as developed and distributed by the Quantum Chemistry Group,
     !     CCLRC Daresbury Laboratory, Daresbury, Cheshire, WA4 4AD
     !     United Kingdom. Contact Huub van Dam (h.j.j.vandam@dl.ac.uk) or
     !     Paul Sherwood for further information.
     !
     !     COPYRIGHT:
     !
     !     Users may incorporate the source code into software packages and
     !     redistribute the source code provided the source code is not
     !     changed in anyway and is properly cited in any documentation or
     !     publication related to its use.
     !
     !     ACKNOWLEDGEMENT:
     !
     !     The source code was generated using Maple 8 through a modified
     !     version of the dfauto script published in:
     !
     !        R. Strange, F.R. Manby, P.J. Knowles
     !        Automatic code generation in density functional theory
     !        Comp. Phys. Comm. 136 (2001) 310-318.
     !
     implicit double precision (a-h,o-z)
     integer,parameter::npt=1
     integer ideriv
     double precision rhoa1
     double precision sigmaaa1
     double precision zk,vrhoa,vsigmaaa
     double precision v2rhoa2(npt),v2rhoasigmaaa(npt),v2sigmaaa2(npt)
     parameter(tol=1.0d-20)
  
  
     !      do i=1,npt
     i=1
     rho = dmax1(0.D0,rhoa1)
     !      if(rho.gt.tol) then
     sigma = dmax1(0.D0,sigmaaa1)
     t2 = rho**(1.D0/3.D0)
     t3 = t2*rho
     t5 = 1/t3
     t6 = t5*sigma
     t7 = dsqrt(sigma)
     t8 = t7*t5
     t10 = dlog(0.1259921049894873D1*t8+dsqrt(1+0.1587401051968199D1 &
           *t8**2))
     t13 = 1.D0+0.317500104573508D-1*t8*t10
     t14 = 1/t13
     t17 = 1/t2
     t19 = 1.D0+0.349D0*t17
     t20 = 1/t19
     t23 = 0.2533D0*t17
     t24 = dexp(-t23)
     t25 = t24*t20
     t26 = rho**2
     t28 = t2**2
     t30 = 1/t28/t26/rho
     t31 = t28*t26
     t34 = t17*t20
     t36 = 0.2611111111111111D1-0.9850555555555556D-1*t17 &
           -0.1357222222222222D0*t34
     t44 = t23+0.349D0*t34-11.D0
     t47 = 0.1148493600075277D2*t31+t36*sigma-0.5D0*(0.25D1 &
           -0.1407222222222222D-1*t17-0.1938888888888889D-1*t34)*sigma &
           -0.2777777777777778D-1*t44*sigma
     t52 = 0.25D0*t26*t47-0.4583333333333333D0*t26*sigma
     t56 = 1/rho
     t57 = t56**(1.D0/3.D0)
     t59 = t56**(1.D0/6.D0)
     t61 = 0.6203504908994D0*t57+0.1029581201158544D2*t59+0.427198D2
     t62 = 1/t61
     t65 = dlog(0.6203504908994D0*t57*t62)
     t68 = 0.1575246635799487D1*t59+0.13072D2
     t71 = datan(0.448998886412873D-1/t68)
     t74 = 0.7876233178997433D0*t59+0.409286D0
     t75 = t74**2
     t77 = dlog(t75*t62)
     !      zk(i) = -0.5908470131056179D0*t3-0.3810001254882096D-2*t6*t14 &
           !      -0.398358D-1*t20*rho-0.52583256D-2*t25*t30*t52+0.19D0*rho* &
           !      (0.310907D-1*t65+0.205219729378375D2*t71+0.4431373767749538D-2 &
           !      *t77)
     t84 = 1/t2/t26
     t88 = t13**2
     t89 = 1/t88
     t94 = 1/t31
     t98 = dsqrt(1.D0+0.1587401051968199D1*sigma*t94)
     t99 = 1/t98
     t109 = t28*rho
     t112 = t44*t56*sigma
     t117 = rho*sigma
     t123 = t19**2
     t124 = 1/t123
     t127 = t26**2
     t129 = 1/t127/rho
     t144 = t5*t20
     t146 = 1/t109
     t147 = t146*t124
     t175 = t57**2
     t176 = 1/t175
     t178 = 1/t26
     t181 = t61**2
     t182 = 1/t181
     t186 = t59**2
     t187 = t186**2
     t189 = 1/t187/t59
     t190 = t189*t178
     t192 = -0.2067834969664667D0*t176*t178-0.1715968668597574D1*t190
     t200 = t68**2
     t201 = 1/t200
     s1 = -0.7877960174741572D0*t2+0.5080001673176129D-2*t84*sigma &
           *t14+0.1905000627441048D-2*t6*t89*(-0.8466669455293548D-1*t7*t84 &
           *t10-0.106673350692263D0*sigma*t30*t99)-0.398358D-1*t20 &
           -0.52583256D-2*t25*t30*(0.5D0*rho*t47+0.25D0*t26* &
           (0.3062649600200738D2*t109-0.2777777777777778D-1*t112)-0.25D0 &
           *t117)-0.46342314D-2*t124*t17-0.44397795816D-3*t129*t24*t20*t52
     s2 = s1-0.6117185448D-3*t24*t124*t129*t52+0.192805272D-1*t25/t28 &
           /t127*t52-0.52583256D-2*t25*t30*(0.25D0*t26*( &
           (0.3283518518518519D-1*t5+0.4524074074074074D-1*t144 &
           -0.1578901851851852D-1*t147)*sigma-0.5D0*(0.4690740740740741D-2 &
           *t5+0.6462962962962963D-2*t144-0.2255574074074074D-2*t147)*sigma &
           -0.2777777777777778D-1*(-0.8443333333333333D-1*t5 &
           -0.1163333333333333D0*t144+0.4060033333333333D-1*t147)*sigma &
           +0.2777777777777778D-1*t112)-0.6666666666666667D0*t117)
     vrhoa = s2+0.5907233D-2*t65+0.3899174858189126D1*t71
     vrhoa = vrhoa &
           +0.8419610158724123D-3*t77
     vrhoa = vrhoa +0.19D0*rho*(0.5011795824473985D-1*( &
           -0.2067834969664667D0*t176*t62*t178-0.6203504908994D0*t57*t182 &
           *t192)/t57*t61+0.2419143800947354D0*t201*t189*t178/(1.D0 &
           +0.2016D-2*t201)+0.4431373767749538D-2*(-0.2625411059665811D0 &
           *t74*t62*t190-1.D0*t75*t182*t192)/t75*t61)
     vsigmaaa = -0.1524000501952839D-1*t5*t14 &
           +0.3810001254882096D-2*t6*t89*(0.6350002091470161D-1/t7*t5*t10 &
           +0.8000501301919725D-1*t94*t99)+0.5842584D-3*t25*t146 &
           -0.210333024D-1*t25*t30*(0.25D0*t26*t36-0.6666666666666667D0*t26)
     !      else ! rho
     !      zk(i) = 0.0d0
     !      vrhoa(i) = 0.0d0
     !      vsigmaaa(i) = 0.0d0
     !      endif ! rho
     !      enddo
  
     return
  end subroutine b3lypf
  
  subroutine b3lyp_e(rhoa1,sigmaaa1,zk)
  
     implicit double precision (a-h,o-z)
     integer,parameter::npt=1
     integer ideriv
     double precision rhoa1
     double precision sigmaaa1
     double precision zk,vrhoa(npt),vsigmaaa(npt)
     double precision v2rhoa2(npt),v2rhoasigmaaa(npt),v2sigmaaa2(npt)
     parameter(tol=1.0d-20)
  
     i=1
     rho = dmax1(0.D0,rhoa1)
     sigma = dmax1(0.D0,sigmaaa1)
     t2 = rho**(1.D0/3.D0)
     t3 = t2*rho
     t5 = 1/t3
     t7 = dsqrt(sigma)
     t8 = t7*t5
     t10 = dlog(0.1259921049894873D1*t8+dsqrt(1+0.1587401051968199D1 &
           *t8**2))
     t17 = 1/t2
     t20 = 1/(1.D0+0.349D0*t17)
     t23 = 0.2533D0*t17
     t24 = dexp(-t23)
     t26 = rho**2
     t28 = t2**2
     t34 = t17*t20
     t56 = 1/rho
     t57 = t56**(1.D0/3.D0)
     t59 = t56**(1.D0/6.D0)
     t62 = 1/(0.6203504908994D0*t57+0.1029581201158544D2*t59 &
           +0.427198D2)
     t65 = dlog(0.6203504908994D0*t57*t62)
     t71 = datan(0.448998886412873D-1/(0.1575246635799487D1*t59 &
           +0.13072D2))
     t75 = (0.7876233178997433D0*t59+0.409286D0)**2
     t77 = dlog(t75*t62)
     zk = -0.5908470131056179D0*t3-0.3810001254882096D-2*t5*sigma/ &
           (1.D0+0.317500104573508D-1*t8*t10)-0.398358D-1*t20*rho &
           -0.52583256D-2*t24*t20/t28/t26/rho*(0.25D0*t26* &
           (0.1148493600075277D2*t28*t26+(0.2611111111111111D1 &
           -0.9850555555555556D-1*t17-0.1357222222222222D0*t34)*sigma-0.5D0 &
           *(0.25D1-0.1407222222222222D-1*t17-0.1938888888888889D-1*t34) &
           *sigma-0.2777777777777778D-1*(t23+0.349D0*t34-11.D0)*sigma) &
           -0.4583333333333333D0*t26*sigma)+0.19D0*rho*(0.310907D-1*t65 &
           +0.205219729378375D2*t71+0.4431373767749538D-2*t77)
  
  end subroutine b3lyp_e
  
  ! Ed Brothers. January 28, 2002
  ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  subroutine becke(density,gx,gy,gz,gotherx,gothery,gotherz,dfdr,dfdgg,dfdggo)
     implicit double precision(a-h,o-z)
  
     ! Given either density and the two gradients, (with gother being for
     ! the spin that density is not, i.e. beta if density is alpha) return
     ! the derivative of beckes 1988 functional with regard to the density
     ! and the derivatives with regard to the gradient invariants.
  
     ! Example:  If becke() is passed the alpha density and the alpha and beta
     ! density gradients, return the derivative of f with regard to the alpha
     ! denisty, the alpha-alpha gradient invariant, and the alpha beta gradient
     ! invariant.
  
     fourPi= 12.5663706143591729538505735331d0
     b = 0.0042d0
     dfdggo=0.d0
  
     rhothirds=density**(1.d0/3.d0)
     rho4thirds=rhothirds**(4.d0)
  
     x = Dsqrt(gx*gx+gy*gy+gz*gz)/rho4thirds
  
     arcsinhx = Dlog(x+DSQRT(x*x+1))
     denom = 1.d0 + 6.d0*b*x*arcsinhx
  
     ! gofx = -1.5d0*(3.d0/fourPi)**(1.d0/3.d0) -b*x*x/denom
     gofx = -0.93052573634910002500d0 -b*x*x/denom
  
     gprimeofx =(6.d0*b*b*x*x*(x/Dsqrt(x*x+1)-arcsinhx) &
           -2.d0*b*x) &
           /(denom*denom)
  
     dfdr=(4.d0/3.d0)*rhothirds*(gofx -x*gprimeofx)
  
     dfdgg= .5d0 * gprimeofx /(x * rho4thirds)
  
  end subroutine becke
  
  ! Ed Brothers. January 28, 2002
  ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  
  subroutine becke_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,e)
     implicit double precision(a-h,o-z)
  
     ! Given the densities and the two gradients, return the energy.
  
     b = 0.0042d0
  
     rho4thirds=density**(4.d0/3.d0)
     x = Dsqrt(gax*gax+gay*gay+gaz*gaz)/rho4thirds
     gofx = -0.93052573634910002500d0 &
           -b*x*x/(1+6.d0*b*x*Dlog(x+DSQRT(x*x+1.d0)))
     e = rho4thirds*gofx
  
     if (densityb == 0.d0) goto 100
  
     rhob4thirds=densityb**(4.d0/3.d0)
     xb = Dsqrt(gbx*gbx+gby*gby+gbz*gbz)/rhob4thirds
     gofxb = -0.93052573634910002500d0 &
           -b*xb*xb/(1+6.d0*b*xb*Dlog(xb+DSQRT(xb*xb+1.d0)))
     e = e + rhob4thirds * gofxb
  
     100 continue
     return
  end subroutine becke_e
  
  ! Ed Brothers. January 28, 2002
  ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  subroutine lyp(pa,pb,gax,gay,gaz,gbx,gby,gbz, &
        dfdr,dfdgg,dfdggo)
     implicit double precision(a-h,o-z)
  
     ! Given the densites and the two gradients, (with gother being for
     ! the spin that density is not, i.e. beta if density is alpha) return
     ! the derivative of the lyp correlation functional with regard to the density
     ! and the derivatives with regard to the gradient invariants.
  
  
     ! Some params:
  
     pi=3.1415926535897932385d0
  
     a = .04918d0
     b = .132d0
     c = .2533d0
     d = .349d0
     CF = .3d0*(3.d0*Pi*Pi)**(2.d0/3.d0)
  
     ! And some required quanitities.
  
     gaa = (gax*gax+gay*gay+gaz*gaz)
     gbb = (gbx*gbx+gby*gby+gbz*gbz)
     gab = (gax*gbx+gay*gby+gaz*gbz)
     ptot = pa+pb
     ptone3rd = ptot**(1.d0/3.d0)
     third = 1.d0/3.d0
     third2 = 2.d0/3.d0
  
     w = Dexp(-c/ptone3rd)*(ptone3rd**(-11.d0))/(1.d0 + d/ptone3rd)
     abw = a*b*w
     abwpapb = abw*pa*pb
  
     dabw = (a*b*Dexp(-c/ptone3rd)*(c*d*ptone3rd**2.d0+ &
           (c-10.d0*d)*ptot -11.d0 *ptone3rd**4.d0)) &
           /(3.d0*(ptone3rd**16.d0)*(d + ptone3rd)**2.d0)
     dabwpapb = (a*b*Dexp(-c/ptone3rd)*pb*(c*d*pa*ptone3rd**2.d0 &
           + (ptone3rd**4.d0)*(-8.d0*pa+3.d0*pb) + ptot &
           * (c*pa - 7.d0*d*pa + 3.d0*d*pb)))/ &
           (3.d0*(ptone3rd**16.d0)*(d + ptone3rd)**2.d0)
  
     delta = c/ptone3rd + (d/ptone3rd)/ (1 + d/ptone3rd)
  
  
     dfdr = 0.d0-dabwpapb*CF*(2.d0**(11.d0/3.d0))*(pa**(8.d0/3.d0) &
           + pb**(8.d0/3.d0)) -dabwpapb*(47.d0/18.d0 - 7.d0*delta/18.d0) &
           *(gaa + gbb + 2.d0*gab)+dabwpapb*(2.5d0 -delta/18.d0)*(gaa + gbb) &
           +dabwpapb*((delta - 11.d0)/9.d0)*(pa*gaa/ptot+pb*gbb/ptot) &
           +dabw*(third2)*ptot*ptot*(gaa + gbb + 2.d0*gab)-dabw*((third2) &
           *ptot*ptot - pa*pa)*gbb-dabw*((third2)*ptot*ptot - pb*pb)*gaa &
           +(-4.d0*a*pb*(3.d0*pb*(ptot)**(third) + d*(pa + 3.d0*pb)))/ &
           (3.d0*(ptot)**(5.d0/3.d0)*(d + (ptot)**(third))**2.d0) &
           +(-64.d0*2.d0**(third2)*abwpapb*CF*pa**(5.d0/3.d0))/3.d0 &
           - (7.d0*abwpapb*(gaa+2.d0*gab + gbb)*(c + (d*(ptot)**(third2))/ &
           (d + (ptot   )**(third    ))**2.d0))/(54.d0*(ptot)**(4.d0/3.d0)) &
           +(abwpapb*(gaa + gbb)*(c +(d*(ptot   )**(third2   ))/ &
           (d + (ptot)**(third))**2.d0))/(54.d0*(ptot )**(4.d0/3.d0)) &
           +(abwpapb*(-30.d0*d**2.d0*(gaa-gbb)*pb*(ptot )**(third    )- &
           33.d0*(gaa - gbb)*pb*(ptot) -c*(gaa*(pa-3.d0*pb) + 4.d0*gbb*pb)* &
           (d + (ptot   )**(third    ))**2.d0 - &
           d*(pa+pb)**(third2   )*(-62.d0*gbb*pb+gaa*(pa+63.d0*pb)))) &
           /(27.d0*(ptot   )**(7.d0/3.d0)*(d + (ptot)**(third))**2.d0) &
           + (4.d0*abw*(gaa + 2.d0*gab + gbb)*(ptot))/3.d0 &
           +(2.d0*abw*gbb*(pa-2.d0*pb))/3.d0+ (-4.d0*abw*gaa*(ptot))/3.d0
  
     dfdgg =  0.d0 &
           -abwpapb*(47.d0/18.d0 - 7.d0*delta/18.d0) &
           +abwpapb*(2.5d0 - delta/18.d0) &
           +abwpapb*((delta - 11.d0)/9.d0)*(pa/ptot) &
           +abw*(2.d0/3.d0)*ptot*ptot &
           -abw*((2.d0/3.d0)*ptot*ptot - pb*pb)
  
     dfdggo= 0.d0 &
           -abwpapb*(47.d0/18.d0 - 7.d0*delta/18.d0)*2.d0 &
           +abw*(2.d0/3.d0)*ptot*ptot*2.d0
  
  end subroutine lyp
  
  ! Ed Brothers. January 31, 2003.
  ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  
  subroutine lyp_e(pa,pb,gax,gay,gaz,gbx,gby,gbz,e)
     implicit double precision(a-h,o-z)
  
     ! Given the densities and the two gradients, return the energy, return
     ! the LYP correlation energy.
  
     ! Note the code is kind of garbled, as Mathematic was used to write it.
  
     ! Some params:
  
     pi=3.1415926535897932385d0
  
     a = .04918d0
     b = .132d0
     c = .2533d0
     d = .349d0
     CF = .3d0*(3.d0*Pi*Pi)**(2.d0/3.d0)
  
     ! And some required quanitities.
  
     gaa = (gax*gax+gay*gay+gaz*gaz)
     gbb = (gbx*gbx+gby*gby+gbz*gbz)
     gab = (gax*gbx+gay*gby+gaz*gbz)
     ptot = pa+pb
     ptone3rd = ptot**(1.d0/3.d0)
  
     w = Dexp(-c/ptone3rd)*(ptone3rd**(-11.d0))/(1.d0 + d/ptone3rd)
     abw = a*b*w
     abwpapb = abw*pa*pb
  
     delta = c/ptone3rd + (d/ptone3rd)/ (1 + d/ptone3rd)
  
     e = -4.d0*a*pa*pb/(ptot*(1.d0 + d/ptone3rd)) &
           -abwpapb*CF*(2.d0**(11.d0/3.d0)) &
           *(pa**(8.d0/3.d0) + pb**(8.d0/3.d0)) &
           -abwpapb*(47.d0/18.d0 - 7.d0*delta/18.d0) &
           *(gaa + gbb + 2.d0*gab) &
           +abwpapb*(2.5d0 - delta/18.d0)*(gaa + gbb) &
           +abwpapb*((delta - 11.d0)/9.d0)*(pa*gaa/ptot+pb*gbb/ptot) &
           +abw*(2.d0/3.d0)*ptot*ptot*(gaa + gbb + 2.d0*gab) &
           -abw*((2.d0/3.d0)*ptot*ptot - pa*pa)*gbb &
           -abw*((2.d0/3.d0)*ptot*ptot - pb*pb)*gaa
  
  end subroutine lyp_e

end module quick_dft_module
