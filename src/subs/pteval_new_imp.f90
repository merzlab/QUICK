#include "util.fh"
! Ed Brothers. January 17, 2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine pteval_new_imp(gridx,gridy,gridz,phi,dphidx,dphidy,dphidz,Iphi,icount)
    use allmod
    implicit double precision(a-h,o-z)

    ! Given a point in space, this function calculates the value of basis
    ! function I and the value of its cartesian derivatives in all three
    ! derivatives.

    x1=(gridx-xyz(1,quick_basis%ncenter(Iphi)))
    y1=(gridy-xyz(2,quick_basis%ncenter(Iphi)))
    z1=(gridz-xyz(3,quick_basis%ncenter(Iphi)))
    rsquared=x1*x1+y1*y1+z1*z1
    

    phi=0.d0
    dphidx=0.d0
    dphidy=0.d0
    dphidz=0.d0

    if (rsquared > sigrad2(Iphi)) then
        continue
    else
        if (itype(1,Iphi) == 0) then
            x1imin1=0.d0
            x1i=1.d0
            x1iplus1=x1
        else
            x1imin1=x1**(itype(1,Iphi)-1)
            x1i=x1imin1*x1
            x1iplus1=x1i*x1
        endif

        if (itype(2,Iphi) == 0) then
            y1imin1=0.d0
            y1i=1.d0
            y1iplus1=y1
        else
            y1imin1=y1**(itype(2,Iphi)-1)
            y1i=y1imin1*y1
            y1iplus1=y1i*y1
        endif

        if (itype(3,Iphi) == 0) then
            z1imin1=0.d0
            z1i=1.d0
            z1iplus1=z1
        else
            z1imin1=z1**(itype(3,Iphi)-1)
            z1i=z1imin1*z1
            z1iplus1=z1i*z1
        endif
        
        xtype=dble(itype(1,Iphi))
        ytype=dble(itype(2,Iphi))
        ztype=dble(itype(3,Iphi))

        temp=0.d0
        kcount=quick_dft_grid%primf_counter(icount)+1
        do while(kcount<quick_dft_grid%primf_counter(icount+1)+1)
            Icon = quick_dft_grid%primf(kcount)+1 

            temp = dcoeff(Icon,IPhi)*DExp((-aexp(Icon,IPhi))*rsquared)

            Phi=Phi+temp
            dphidx=dphidx+temp*(-2.d0*(aexp(Icon,IPhi))*x1iplus1+xtype*x1imin1)
            dphidy=dphidy+temp*(-2.d0*(aexp(Icon,IPhi))*y1iplus1+ytype*y1imin1)
            dphidz=dphidz+temp*(-2.d0*(aexp(Icon,IPhi))*z1iplus1+ztype*z1imin1)

            kcount=kcount+1
        enddo
        Phi=phi*x1i*y1i*z1i
 

        dphidx=dphidx*y1i*z1i
        dphidy=dphidy*x1i*z1i
        dphidz=dphidz*x1i*y1i
    endif

    return
    end subroutine pteval_new_imp



