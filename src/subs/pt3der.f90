#include "util.fh"
    ! Ed Brothers. July 11, 2002
    ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine pt3dr(gridx,gridy,gridz,phi,phix,phixx,phixxx,Iphi,icount)
    use allmod
    use quick_gridpoints_module
    implicit double precision(a-h,o-z)
    double precision :: phi
    double precision, dimension(3) :: phix
    double precision, dimension(6) :: phixx
    double precision, dimension(10):: phixxx
 
    ! Given a point in space, this function calculates the value of basis
    ! function I and the value of its cartesian derivatives in all three
    ! derivatives.

    x1=(gridx-xyz(1,quick_basis%ncenter(Iphi)))
    y1=(gridy-xyz(2,quick_basis%ncenter(Iphi)))
    z1=(gridz-xyz(3,quick_basis%ncenter(Iphi)))
    rsquared=x1*x1+y1*y1+z1*z1

!
!  The order of the first derivatives is x, y, z
!  The order of the second derivatives is xx, xy, yy, xz, yz, zz
!  The order of the third derivatives is xxx, xxy, xyy, yyy, xxz,
!                                        xyz, yyz, xzz, yzz, zzz
!
    phi    = 0.d0
    phix   = 0.d0
    phixx  = 0.d0
    phixxx = 0.d0

    if (rsquared > sigrad2(Iphi)) then
        continue
    else
        if (itype(1,Iphi) == 0) then
            x1imin3=0.d0
            x1imin2=0.d0
            x1imin1=0.d0
            x1i=1.d0
            x1iplus1=x1i*x1
            x1iplus2=x1iplus1*x1
            x1iplus3=x1iplus2*x1
        elseif (itype(1,Iphi) == 1) then
            x1imin3=0.d0
            x1imin2=0.d0
            x1imin1=1.d0
            x1i=x1imin1*x1
            x1iplus1=x1i*x1
            x1iplus2=x1iplus1*x1
            x1iplus3=x1iplus2*x1
        elseif (itype(1,Iphi) == 2) then
            x1imin3=0.d0
            x1imin2=1.d0
            x1imin1=x1imin2*x1
            x1i=x1imin1*x1
            x1iplus1=x1i*x1
            x1iplus2=x1iplus1*x1
            x1iplus3=x1iplus2*x1
        else
            x1imin3=x1**(itype(1,Iphi)-3)
            x1imin2=x1imin3*x1
            x1imin1=x1imin2*x1
            x1i=x1imin1*x1
            x1iplus1=x1i*x1
            x1iplus2=x1iplus1*x1
            x1iplus3=x1iplus2*x1
        endif

        if (itype(2,Iphi) == 0) then
            y1imin3=0.d0
            y1imin2=0.d0
            y1imin1=0.d0
            y1i=1.d0
            y1iplus1=y1i*y1
            y1iplus2=y1iplus1*y1
            y1iplus3=y1iplus2*y1
        elseif (itype(2,Iphi) == 1) then
            y1imin3=0.d0
            y1imin2=0.d0
            y1imin1=1.d0
            y1i=y1imin1*y1
            y1iplus1=y1i*y1
            y1iplus2=y1iplus1*y1
            y1iplus3=y1iplus2*y1
        elseif (itype(2,Iphi) == 2) then
            y1imin3=0.d0
            y1imin2=1.d0
            y1imin1=y1imin2*y1
            y1i=y1imin1*y1
            y1iplus1=y1i*y1
            y1iplus2=y1iplus1*y1
            y1iplus3=y1iplus2*y1
        else
            y1imin3=y1**(itype(2,Iphi)-3)
            y1imin2=y1imin3*y1
            y1imin1=y1imin2*y1
            y1i=y1imin1*y1
            y1iplus1=y1i*y1
            y1iplus2=y1iplus1*y1
            y1iplus3=y1iplus2*y1
        endif

        if (itype(3,Iphi) == 0) then
            z1imin3=0.d0
            z1imin2=0.d0
            z1imin1=0.d0
            z1i=1.d0
            z1iplus1=z1i*z1
            z1iplus2=z1iplus1*z1
            z1iplus3=z1iplus2*z1
        elseif (itype(3,Iphi) == 1) then
            z1imin3=0.d0
            z1imin2=0.d0
            z1imin1=1.d0
            z1i=z1imin1*z1
            z1iplus1=z1i*z1
            z1iplus2=z1iplus1*z1
            z1iplus3=z1iplus2*z1
        elseif (itype(3,Iphi) == 2) then
            z1imin3=0.d0
            z1imin2=1.d0
            z1imin1=z1imin2*z1
            z1i=z1imin1*z1
            z1iplus1=z1i*z1
            z1iplus2=z1iplus1*z1
            z1iplus3=z1iplus2*z1
        else
            z1imin3=z1**(itype(3,Iphi)-3)
            z1imin2=z1imin3*z1
            z1imin1=z1imin2*z1
            z1i=z1imin1*z1
            z1iplus1=z1i*z1
            z1iplus2=z1iplus1*z1
            z1iplus3=z1iplus2*z1
        endif

        xtype=dble(itype(1,Iphi))
        ytype=dble(itype(2,Iphi))
        ztype=dble(itype(3,Iphi))

        kcount=quick_dft_grid%primf_counter(icount)+1
        do while(kcount<quick_dft_grid%primf_counter(icount+1)+1)
            Icon = quick_dft_grid%primf(kcount)+1
            temp = dcoeff(Icon,IPhi)*DExp((-aexp(Icon,IPhi))*rsquared)
            twoA = 2.d0*aexp(Icon,IPhi)
            fourAsqr=4.d0*aexp(Icon,IPhi)*aexp(Icon,IPhi)
            eightAcu=8.d0*aexp(Icon,IPhi)*aexp(Icon,IPhi)*aexp(Icon,IPhi)
!
            phi = phi+temp
!
            phix(1) = phix(1)+temp*(-twoA*x1iplus1+xtype*x1imin)
            phix(2) = phix(2)+temp*(-twoA*y1iplus1+ytype*y1imin1)
            phix(3) = phix(3)+temp*(-twoA*z1iplus1+ztype*z1imin1)
!
            phixx(1) = phixx(1)+temp*(xtype*(xtype-1.d0)*x1imin2 &
                       -twoA*(2.d0*xtype+1.d0)*x1i &
                       +fourAsqr*x1iplus2)             !xx
            phixx(2) = phixx(2)+temp*(xtype*ytype*x1imin1*y1imin1 &
                       -twoA*xtype*x1imin1*y1iplus1 &
                       -twoA*ytype*x1iplus1*y1imin1 &
                       +fourAsqr*x1iplus1*y1iplus1)    !xy
            phixx(3) = phixx(3)+temp*(ytype*(ytype-1.d0)*y1imin2 &
                       -twoA*(2.d0*ytype+1.d0)*y1i &
                       +fourAsqr*y1iplus2)             !yy
            phixx(4) = phixx(4)+temp*(xtype*ztype*x1imin1*z1imin1 &
                       -twoA*xtype*x1imin1*z1iplus1 &
                       -twoA*ztype*x1iplus1*z1imin1 &
                       +fourAsqr*x1iplus1*z1iplus1)    !xz
            phixx(5) = phixx(5)+temp*(ytype*ztype*y1imin1*z1imin1 &
                      -twoA*ytype*y1imin1*z1iplus1 &
                      -twoA*ztype*y1iplus1*z1imin1 &
                      +fourAsqr*y1iplus1*z1iplus1)     !yz
            phixx(6) = phixx(6)+temp*(ztype*(ztype-1.d0)*z1imin2 &
                       -twoA*(2.d0*ztype+1.d0)*z1i &
                       +fourAsqr*z1iplus2)             !zz
!
            phixxx(1)  = phixxx(1)+temp*(xtype*(xtype-1.d0)*(xtype-2.d0)*x1imin3 &
                         -twoA*3.d0*xtype*xtype*x1imin1 &
                         +fourAsqr*(3.d0*xtype+3.d0)*x1iplus1 &
                         -eightAcu*x1iplus3) !xxx
            phixxx(2)  = phixxx(2)+temp*(xtype*(xtype-1.d0)*ytype*x1imin2*y1imin1 &
                         -twoA*(2.d0*xtype+1.d0)*ytype*y1imin1*x1i &
                         -twoA*xtype*(xtype-1.d0)*x1imin2*y1ipuls1 &
                         +fourAsqr*(2.d0*xtype+1.d0)*x1i*y1iplus1 &
                         +fourAsqr*ytype*x1iplus2*y1imin1 &
                         -eightAcu*x1iplus2*y1iplus1)  !xxy
            phixxx(3)  = phixxx(3)+temp*(ytype*(ytype-1.d0)*xtype*y1imin2*x1imin1 & 
                         -twoA*(2.d0*ytype+1.d0)*xtype*x1imin1*y1i &
                         -twoA*ytype*(ytype-1.d0)*y1imin2*x1ipuls1 & 
                         +fourAsqr*(2.d0*ytype+1.d0)*y1i*x1iplus1 &
                         +fourAsqr*xtype*y1iplus2*x1imin1 &
                         -eightAcu*y1iplus2*x1iplus1)  !xyy
            phixxx(4)  = phixxx(4)+temp*(ytype*(ytype-1.d0)*(ytype-2.d0)*y1imin3 &  
                         -twoA*3.d0*ytype*ytype*y1imin1 &  
                         +fourAsqr*(3.d0*ytype+3.d0)*y1iplus1 &
                         -eightAcu*y1iplus3)  !yyy
            phixxx(5)  = phixxx(5)+temp*(xtype*(xtype-1.d0)*ztype*x1imin2*z1imin1 & 
                         -twoA*(2.d0*xtype+1.d0)*ztype*z1imin1*x1i &
                         -twoA*xtypei*(xtype-1.d0)*x1imin2*z1ipuls1 & 
                         +fourAsqr*(2.d0*xtype+1.d0)*x1i*z1iplus1 &
                         +fourAsqr*ztype*x1iplus2*z1imin1 &
                         -eightAcu*x1iplus2*z1iplus1)  !xxz
            phixxx(6)  = phixxx(6)+temp*(xtype*ytpe*ztype*x1imin1*y1imin1*z1imin1 &
                         -twoA*xtype*ytype*x1imin1*y1imin1*z1iplus1 &
                         -twoA*ytype*ztype*x1iplus1*y1imin1*z1imin1 &
                         -twoA*ztype*xtype*x1imin1*y1iplus1*z1imin1 &  
                         +fourAsqr*xtype*x1imin1*y1iplus1*z1iplus1 &
                         +fourAsqr*ytype*y1imin1*x1iplus1*z1iplus1 &
                         +fourAsqr*ztype*z1imin1*x1iplus1*y1iplus1 &
                         -eightAcu*x1iplus1*y1iplus1*z1iplus1)   !xyz
            phixxx(7)  = phixxx(7)+temp*(ytype*(ytype-1.d0)*ztype*y1imin2*z1imin1 & 
                         -twoA*(2.d0*ytype+1.d0)*ztype*z1imin1*y1i &
                         -twoA*ytype*(ytype-1.d0)*y1imin2*z1ipuls1 &
                         +fourAsqr*(2.d0*ytype+1.d0)*y1i*z1iplus1 &
                         +fourAsqr*ztype*y1iplus2*z1imin1 &
                         -eightAcu*y1iplus2*z1iplus1)  !yyz
            phixxx(8)  = phixxx(8)+temp*(ztype*(ztype-1.d0)*xtype*z1imin2*x1imin1 & 
                         -twoA*(2.d0*ztype+1.d0)*xtype*x1imin1*z1i &
                         -twoA*ztypei*(ztype-1.d0)*z1imin2*x1ipuls1 &
                         +fourAsqr*(2.d0*ztype+1.d0)*z1i*x1iplus1 &
                         +fourAsqr*xtype*z1iplus2*x1imin1 &
                         -eightAcu*z1iplus2*x1iplus1)  !xzz
            phixxx(9)  = phixxx(9)+temp*(ztype*(ztype-1.d0)*ytype*z1imin2*y1imin1 &
                         -twoA*(2.d0*ztype+1.d0)*ytype*y1imin1*z1i &
                         -twoA*ztype*(ztype-1.d0)*z1imin2*y1ipuls1 &
                         +fourAsqr*(2.d0*ztype+1.d0)*z1i*y1iplus1 &
                         +fourAsqr*ytype*z1iplus2*y1imin1 &
                         -eightAcu*z1iplus2*y1iplus1)  !yzz
            phixxx(10) = phixxx(10)+temp*(ztype*(ztype-1.d0)*(ztype-2.d0)*z1imin3 &  
                         -twoA*ztype*ztype*ztype*z1imin1 &  
                         +fourAsqr*(3.d0*ztype+3.d0)*z1iplus1 & 
                         -eightAcu*z1iplus3) !zzz

            kcount=kcount+1
        enddo

        phi = phi*x1i*y1i*z1i

        phix(1) = phix(1)*y1i*z1i
        phix(2) = phix(2)*x1i*z1i
        phix(3) = phix(3)*x1i*y1i

        phixx(1) = phixx(1)*y1i*z1i
        phixx(2) = phixx(2)*z1i
        phixx(3) = phixx(3)*x1i*z1i
        phixx(4) = phixx(4)*y1i
        phixx(5) = phixx(5)*x1i
        phixx(6) = phixx(6)*x1i*y1i

        phixxx(1)  = phixxx(1)*y1i*z1i
        phixxx(2)  = phixxx(2)*z1i
        phixxx(3)  = phixxx(3)*z1i
        phixxx(4)  = phixxx(4)*x1i*z1i
        phixxx(5)  = phixxx(5)*y1i
        phixxx(6)  = phixxx(6)*1.d0
        phixxx(7)  = phixxx(7)*x1i
        phixxx(8)  = phixxx(8)*y1i
        phixxx(9)  = phixxx(9)*x1i
        phixxx(10) = phixxx(10)*x1i*y1i

    endif

    end subroutine pt3dr

