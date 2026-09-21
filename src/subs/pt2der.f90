#include "util.fh"
    ! Ed Brothers. July 11, 2002
    ! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine pt2der(gridx,gridy,gridz,dxdx,dxdy,dxdz,dydy,dydz,dzdz,Iphi,icount)
    use allmod
    use quick_gridpoints_module
    implicit double precision(a-h,o-z)

    ! Given a point in space, this function calculates the value of basis
    ! function I and the value of its cartesian derivatives in all three
    ! derivatives.

    x1=(gridx-xyz(1,quick_basis%ncenter(Iphi)))
    y1=(gridy-xyz(2,quick_basis%ncenter(Iphi)))
    z1=(gridz-xyz(3,quick_basis%ncenter(Iphi)))
    rsquared=x1*x1+y1*y1+z1*z1

    dxdx = 0.d0
    dxdy = 0.d0
    dxdz = 0.d0
    dydy = 0.d0
    dydz = 0.d0
    dzdz = 0.d0

    if (rsquared > sigrad2(Iphi)) then
        continue
    else
        if (itype(1,Iphi) == 0) then
            x1imin2=0.d0
            x1imin1=0.d0
            x1i=1.d0
            x1iplus1=x1
            x1iplus2=x1*x1
        elseif (itype(1,Iphi) == 1) then
            x1imin2=0.d0
            x1imin1=1.d0
            x1i=x1
            x1iplus1=x1*x1
            x1iplus2=x1*x1*x1
        else
            x1imin2=x1**(itype(1,Iphi)-2)
            x1imin1=x1imin2*x1
            x1i=x1imin1*x1
            x1iplus1=x1i*x1
            x1iplus2=x1iplus1*x1
        endif

        if (itype(2,Iphi) == 0) then
            y1imin2=0.d0
            y1imin1=0.d0
            y1i=1.d0
            y1iplus1=y1
            y1iplus2=y1*y1
        elseif (itype(2,Iphi) == 1) then
            y1imin2=0.d0
            y1imin1=1.d0
            y1i=y1
            y1iplus1=y1*y1
            y1iplus2=y1*y1*y1
        else
            y1imin2=y1**(itype(2,Iphi)-2)
            y1imin1=y1imin2*y1
            y1i=y1imin1*y1
            y1iplus1=y1i*y1
            y1iplus2=y1iplus1*y1
        endif

        if (itype(3,Iphi) == 0) then
            z1imin2=0.d0
            z1imin1=0.d0
            z1i=1.d0
            z1iplus1=z1
            z1iplus2=z1*z1
        elseif (itype(3,Iphi) == 1) then
            z1imin2=0.d0
            z1imin1=1.d0
            z1i=z1
            z1iplus1=z1*z1
            z1iplus2=z1*z1*z1
        else
            z1imin2=z1**(itype(3,Iphi)-2)
            z1imin1=z1imin2*z1
            z1i=z1imin1*z1
            z1iplus1=z1i*z1
            z1iplus2=z1iplus1*z1
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
            dxdx = dxdx+temp*(xtype*(xtype-1.d0)*x1imin2 &
            -twoA*(2.d0*xtype+1.d0)*x1i &
            +fourAsqr*x1iplus2)
            dydy = dydy+temp*(ytype*(ytype-1.d0)*y1imin2 &
            -twoA*(2.d0*ytype+1.d0)*y1i &
            +fourAsqr*y1iplus2)
            dzdz = dzdz+temp*(ztype*(ztype-1.d0)*z1imin2 &
            -twoA*(2.d0*ztype+1.d0)*z1i &
            +fourAsqr*z1iplus2)
            dxdy = dxdy+temp*(xtype*ytype*x1imin1*y1imin1 &
            -twoA*xtype*x1imin1*y1iplus1 &
            -twoA*ytype*x1iplus1*y1imin1 &
            +fourAsqr*x1iplus1*y1iplus1)
            dxdz = dxdz+temp*(xtype*ztype*x1imin1*z1imin1 &
            -twoA*xtype*x1imin1*z1iplus1 &
            -twoA*ztype*x1iplus1*z1imin1 &
            +fourAsqr*x1iplus1*z1iplus1)
            dydz = dydz+temp*(ytype*ztype*y1imin1*z1imin1 &
            -twoA*ytype*y1imin1*z1iplus1 &
            -twoA*ztype*y1iplus1*z1imin1 &
            +fourAsqr*y1iplus1*z1iplus1)

            kcount=kcount+1
        enddo
        dxdx = dxdx*y1i*z1i
        dydy = dydy*x1i*z1i
        dzdz = dzdz*x1i*y1i
        dxdy = dxdy*z1i
        dxdz = dxdz*y1i
        dydz = dydz*x1i
    endif

    end subroutine pt2der

