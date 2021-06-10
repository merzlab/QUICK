#include "util.fh"
!*******************************************************
! dipole
!-------------------------------------------------------
! Ed Brothers. March 18,2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
!-------------------------------------------------------
    subroutine dipole
    use allmod
    implicit double precision(a-h,o-z)
    real*8 xyzdipole(3,natom)
!-------------------------------------------------------
! The purpose of this subroutine is to generate the Mulliken and Lowdin
! charges, and then calculate the dipole moment.
!-------------------------------------------------------

    call prtact(ioutfile,"Begin Charge and Dipole Calculation")
    call cpu_time(timer_begin%TDip)
    !-------------------------------------------------------
    ! Part 1. Mulliken and Lowdin charge
    !-------------------------------------------------------
    ! First, calculate P*S.  Note P is symmetric, so we can use the fast
    ! muliplier.  Also note we are using the total denisity matrix. Store
    ! this is HOLD.
    IF ( .NOT. quick_method%unrst) THEN
        DO I=1,nbasis
            DO J=1,nbasis
                HOLDIJ = 0.0D0
                DO K=1,nbasis
                    HOLDIJ = HOLDIJ + quick_qm_struct%dense(K,I)*quick_qm_struct%s(K,J)
                ENDDO
                quick_scratch%hold(I,J) = HOLDIJ
            ENDDO
        ENDDO
    ELSE
        DO I=1,nbasis
            DO J=1,nbasis
                quick_scratch%hold2(J,I) = quick_qm_struct%dense(J,I)+quick_qm_struct%denseB(J,I)
            ENDDO
        ENDDO
        DO I=1,nbasis
            DO J=1,nbasis
                HOLDIJ = 0.0D0
                DO K=1,nbasis
                    HOLDIJ = HOLDIJ + quick_scratch%hold2(K,I)*quick_qm_struct%s(K,J)
                ENDDO
                quick_scratch%hold(I,J) = HOLDIJ
            ENDDO
        ENDDO
    ENDIF

    ! Mulliken Charge of atom A = core charge A - (Sum over u on A) PS(uu)
    DO I=1,natom
        quick_qm_struct%Mulliken(I) = quick_molspec%chg(I)
        DO J=quick_basis%first_basis_function(I),quick_basis%last_basis_function(I)
            quick_qm_struct%Mulliken(I) = quick_qm_struct%Mulliken(I) - quick_scratch%hold(J,J)
        ENDDO
    ENDDO

    ! At this point we have the Mulliken charges. Now calculate the Lowdin
    ! charges:
    ! Lowdin Charge of atom A = core charge A -
    ! - (Sum over u on A)[S^(1/2)PS^(1/2)](uu)
    ! Now remember S^(-1/2) = X.  Thus we have to calculate
    ! XSPSX = S^(-1/2)SPSS^(-1/2)= S^(1/2)PS^(1/2)

    ! Currently, HOLD contains PS.  Use the fast multiplier to get SPS and place
    ! it in HOLD2.
    DO I=1,nbasis
        DO J=1,nbasis
            HOLDIJ = 0.0D0
            DO K=1,nbasis
                HOLDIJ = HOLDIJ + quick_qm_struct%s(K,I)*quick_scratch%hold(K,J)
            ENDDO
            quick_scratch%hold2(I,J) = HOLDIJ
        ENDDO
    ENDDO

    ! Now we have two slow multiplication steps to get to XSPSX.
    DO I=1,nbasis
        DO J=1,nbasis
            HOLDIJ = 0.0D0
            DO K=1,nbasis
                HOLDIJ = HOLDIJ + quick_qm_struct%x(I,K)*quick_scratch%hold2(K,J)
            ENDDO
            quick_scratch%hold(I,J) = HOLDIJ
        ENDDO
    ENDDO

    DO I=1,nbasis
        DO J=1,nbasis
            HOLDIJ = 0.0D0
            DO K=1,nbasis
                HOLDIJ = HOLDIJ + quick_scratch%hold(I,K)*quick_qm_struct%x(K,J)
            ENDDO
            quick_scratch%hold2(I,J) = HOLDIJ
        ENDDO
    ENDDO

    ! So now HOLD2 contains XSPSX.  Use this to calculate the Lowdin charges.
    DO I=1,natom
        quick_qm_struct%Lowdin(I) = quick_molspec%chg(I)
        DO J=quick_basis%first_basis_function(I),quick_basis%last_basis_function(I)
            quick_qm_struct%Lowdin(I) = quick_qm_struct%Lowdin(I) - quick_scratch%hold2(J,J)
        ENDDO
    ENDDO

    write (ioutfile,'(4x,"ATOMIC CHARGES")')
    write (ioutfile,'(3x,"ATOM          MULLIKEN            LOWDIN")')
    DO I=1,natom
        write (ioutfile,'(3x,A2,7x,F12.4,7x,F12.4)') symbol(quick_molspec%iattype(I)), &
        quick_qm_struct%Mulliken(I),quick_qm_struct%Lowdin(I)
    ENDDO
    write (ioutfile,'(3x,A5,3x,F12.4,7x,F12.4)') "TOTAL", &
        sum(quick_qm_struct%Mulliken(1:natom)),sum(quick_qm_struct%Lowdin(1:natom))

    !-------------------------------------------------------
    ! Part 2. Dipole calculation
    !-------------------------------------------------------
    ! First move the molecule's center of mass to the origin.
    totweight=0.d0
    cmx = 0.d0
    cmy = 0.d0
    cmz = 0.d0
    DO I=1,natom
        weight = emass(quick_molspec%iattype(I))
        totweight = weight+totweight
        cmx = cmx+xyz(1,I)*weight
        cmy = cmy+xyz(2,I)*weight
        cmz = cmz+xyz(3,I)*weight
    ENDDO
    cmx = cmx/totweight
    cmy = cmy/totweight
    cmz = cmz/totweight
    DO I=1,natom
        xyzdipole(1,I)=xyz(1,I)-cmx
        xyzdipole(2,I)=xyz(2,I)-cmy
        xyzdipole(3,I)=xyz(3,I)-cmz
    ENDDO


    xdip=0.d0
    ydip=0.d0
    zdip=0.d0

    DO I=1,natom
        xdip = xdip+quick_molspec%chg(I)*xyzdipole(1,I)
        ydip = ydip+quick_molspec%chg(I)*xyzdipole(2,I)
        zdip = zdip+quick_molspec%chg(I)*xyzdipole(3,I)
    ENDDO

    DO Ibas=1,nbasis
        DO Jbas=Ibas,nbasis
            Sx =0.d0
            Sy =0.d0
            Sz =0.d0
            DENSEJI=quick_qm_struct%dense(Jbas,Ibas)
            IF (quick_method%unrst) DENSEJI=DENSEJI+quick_qm_struct%denseB(Jbas,Ibas)
            IF (Jbas /= Ibas) DENSEJI=DENSEJI*2.d0
            DO Icon=1,ncontract(ibas)
                DO Jcon=1,ncontract(jbas)
                    Sx =Sx + &
                    dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                    *xmoment(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                    itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                    1,0,0, &
                    xyzdipole(1,quick_basis%ncenter(Jbas)),xyzdipole(2,quick_basis%ncenter(Jbas)), &
                    xyzdipole(3,quick_basis%ncenter(Jbas)),xyzdipole(1,quick_basis%ncenter(Ibas)), &
                    xyzdipole(2,quick_basis%ncenter(Ibas)),xyzdipole(3,quick_basis%ncenter(Ibas)), &
                    0.d0,0.d0,0.d0)
                    Sy =Sy + &
                    dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                    *xmoment(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                    itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                    0,1,0, &
                    xyzdipole(1,quick_basis%ncenter(Jbas)),xyzdipole(2,quick_basis%ncenter(Jbas)), &
                    xyzdipole(3,quick_basis%ncenter(Jbas)),xyzdipole(1,quick_basis%ncenter(Ibas)), &
                    xyzdipole(2,quick_basis%ncenter(Ibas)),xyzdipole(3,quick_basis%ncenter(Ibas)), &
                    0.d0,0.d0,0.d0)
                    Sz =Sz + &
                    dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                    *xmoment(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                    itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                    0,0,1, &
                    xyzdipole(1,quick_basis%ncenter(Jbas)),xyzdipole(2,quick_basis%ncenter(Jbas)), &
                    xyzdipole(3,quick_basis%ncenter(Jbas)),xyzdipole(1,quick_basis%ncenter(Ibas)), &
                    xyzdipole(2,quick_basis%ncenter(Ibas)),xyzdipole(3,quick_basis%ncenter(Ibas)), &
                    0.d0,0.d0,0.d0)

                ENDDO
            ENDDO
            xdip = xdip - Sx*DENSEJI
            ydip = ydip - Sy*DENSEJI
            zdip = zdip - Sz*DENSEJI
        ENDDO
    ENDDO

    totdip = ((xdip*xdip+ydip*ydip+zdip*zdip)**.5d0)*2.541765d0
    write (ioutfile,'(/,4x,"DIPOLE (DEBYE)")')
    write (ioutfile,'(6x,"X",9x,"Y",9x,"Z",8x,"TOTAL")')
    write (ioutfile,'(4f10.4)') xdip*2.541765d0,ydip*2.541765d0,zdip*2.541765d0,totdip

    call cpu_time(timer_end%TDip)
    timer_cumer%TDip=timer_cumer%TDip+(timer_end%TDip-timer_begin%TDip)

    call prtact(ioutfile,"End Charge and Dipole Calculation")
    
    return
    end subroutine dipole
    
!*******************************************************
! xmoment   
!-------------------------------------------------------
! Ed Brothers. September 25, 2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    double precision function xmoment(a,b,i,j,k,ii,jj,kk, &
    iux,iuy,iuz,Ax,Ay,Az, &
    Bx,By,Bz,Cx,Cy,Cz)
    use quick_constants_module
    implicit double precision(a-h,o-z)

    ! Variables needed later:
    !    pi=3.1415926535897932385
    g = a+b
    Px = (a*Ax + b*Bx)/g
    Py = (a*Ay + b*By)/g
    Pz = (a*Az + b*Bz)/g

    ! The purpose of this subroutine is to calculate moment around point
    ! C.  This is used for calculating dipole moments.

    ! The notation is the same used throughout: gtfs with orbital exponents a
    ! and b on A and B with angular momentums defined by i,j,k (a's x, y
    ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
    ! (Cx,Cy,Cz) with charge Z. New to this are the iux, iuy, and iuz terms
    ! which determine the moment type being calculated.

    ! The this is taken from the recursive relation found in Obara and Saika,
    ! J. Chem. Phys. 84 (7) 1986, 3963.

    ! apass and bpass are used to avoid swapping the two variables when
    ! control passes back to the main program.

    apass=a
    bpass=b

    xmoment = xmomentrecurse(apass,bpass,i,j,k,ii,jj,kk, &
    iux,iuy,iuz, &
    Ax,Ay,Az,Bx,By,Bz, &
    Cx,Cy,Cz,Px,Py,Pz,g)

    return
    end function xmoment

!*******************************************************
! xmomentrecurse
!-------------------------------------------------------
! Ed Brothers. September 25, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    double precision recursive function xmomentrecurse(a,b, &
    i,j,k,ii,jj,kk,iux,iuy,iuz, &
    Ax,Ay,Az,Bx,By,Bz, &
    Cx,Cy,Cz,Px,Py,Pz,g) &
    result(xmomentrec)
    
    use quick_overlap_module, only: overlap
    implicit double precision(a-h,o-z)
    dimension iexponents(9),center(12)
    double precision g_table(200)

! The this is taken from the recursive relation found in Obara and Saika,
! J. Chem. Phys. 84 (7) 1986, 3963.

! If i=j=k=ii=jj=kk=iux=iuy=iuz=0, this is simply an overlap integral.

    IF (i+j+k+ii+jj+kk+iux+iuy+iuz == 0) THEN
        g_table = g**(-1.5)
        xmomentrec=overlap(a,b,0,0,0,0,0,0, &
        Ax,Ay,Az,Bx,By,Bz, Px,Py,Pz,g_table)

    ! Otherwise, use the recusion relation from Obara and Saika.  The first
    ! step is to find the lowest nonzero angular momentum exponent or,
    ! if all of those are zero, the lowest nonzeo iux, iuy or iuz. This is
    ! because the more exponents equal zero the fewer terms need to be
    ! calculated, and each recursive loop reduces the angular momentum
    ! exponents. This therefore reorders the atoms and sets the exponent
    ! to be reduced.

    ELSE
        xmomentrec=0.d0
        iexponents(1) = i
        iexponents(2) = j
        iexponents(3) = k
        iexponents(4) = ii
        iexponents(5) = jj
        iexponents(6) = kk
        iexponents(7) = iux
        iexponents(8) = iuy
        iexponents(9) = iuz
        center(7) = Cx
        center(8) = Cy
        center(9) = Cz
        center(10)= Px
        center(11)= Py
        center(12)= Pz
        ilownum=300
        ilowex=300
        DO L=1,6
            IF (iexponents(L) < ilowex .AND. iexponents(L) /= 0) THEN
                ilowex=iexponents(L)
                ilownum=L
            ENDIF
        ENDDO
        IF (ilownum <= 3) THEN
            center(1)=Ax
            center(2)=Ay
            center(3)=Az
            center(4)=Bx
            center(5)=By
            center(6)=Bz
        ELSE
            center(4)=Ax
            center(5)=Ay
            center(6)=Az
            center(1)=Bx
            center(2)=By
            center(3)=Bz
            iexponents(4) = i
            iexponents(5) = j
            iexponents(6) = k
            iexponents(1) = ii
            iexponents(2) = jj
            iexponents(3) = kk
            IF(ilownum <= 6) ilownum = ilownum - 3
            temp=b
            b=a
            a=temp
        ENDIF

    ! The reson for the following two variables is explained fully in the
    ! repulsion code.

        apass=a
        bpass=b

    ! There are two possibilities:
    ! 1)  ilownum is equal to 300.  This meant that i=j=k=ii=jj=kk=0.
    ! If iux,iuy,or iuz was equal to zero, we would not be this far
    ! in the code. Thus the integral is of the form (s|u|s).
    ! If this is the case:

        IF (ilownum == 300) THEN
            DO L=7,9
                IF (iexponents(L) < ilowex .AND. iexponents(L) /= 0) THEN
                    ilowex=iexponents(L)
                    ilownum=L
                ENDIF
            ENDDO
            PC = center(ilownum+3)-center(ilownum)
            iexponents(ilownum)=iexponents(ilownum)-1
            IF (PC /= 0.d0) THEN
                xmomentrec=xmomentrec+PC* &
                xmomentrecurse(apass,bpass,0,0,0,0,0,0, &
                iexponents(7),iexponents(8),iexponents(9), &
                center(1),center(2),center(3), &
                center(4),center(5),center(6), &
                center(7),center(8),center(9), &
                center(10),center(11),center(12),g)
                apass=a
                bpass=b
            ENDIF
            IF(iexponents(ilownum) /= 0) THEN
                coeff = dble(iexponents(ilownum))/(2.d0*g)
                iexponents(ilownum)=iexponents(ilownum)-1
                xmomentrec=xmomentrec+coeff* &
                xmomentrecurse(apass,bpass,0,0,0,0,0,0, &
                iexponents(7),iexponents(8),iexponents(9), &
                center(1),center(2),center(3), &
                center(4),center(5),center(6), &
                center(7),center(8),center(9), &
                center(10),center(11),center(12),g)
                iexponents(ilownum)=iexponents(ilownum)+1
                a=apass
                b=bpass
            ENDIF


        ! Otherwise:
        ! 2)  The ilow number is referring to an angular momentum orbital
        ! exponent.

        ELSE

        ! The first step is lowering the orbital exponent by one.

            iexponents(ilownum) = iexponents(ilownum)-1

        ! At this point, calculate the first term of the recursion
        ! relation.

            PA = center(9+ilownum)-center(ilownum)
            IF (PA /= 0) THEN
                xmomentrec=xmomentrec+PA* &
                xmomentrecurse(apass,bpass, &
                iexponents(1),iexponents(2),iexponents(3), &
                iexponents(4),iexponents(5),iexponents(6), &
                iexponents(7),iexponents(8),iexponents(9), &
                center(1),center(2),center(3), &
                center(4),center(5),center(6), &
                center(7),center(8),center(9), &
                center(10),center(11),center(12),g)
                apass=a
                bpass=b
            ENDIF

        ! The next term only arises if the angual momentum of the dimension
        ! of A that has already been lowered is not zero.  In other words, if a
        ! (px|ux|px) was passed to this subroutine, we are now considering
        ! (s|ux|px), and the following term does not arise, as the x expoent
        ! on A is zero.

            IF (iexponents(ilownum) /= 0) THEN
                coeff = dble(iexponents(ilownum))/(2.d0*g)
                iexponents(ilownum) = iexponents(ilownum)-1
                xmomentrec=xmomentrec+coeff* &
                xmomentrecurse(apass,bpass, &
                iexponents(1),iexponents(2),iexponents(3), &
                iexponents(4),iexponents(5),iexponents(6), &
                iexponents(7),iexponents(8),iexponents(9), &
                center(1),center(2),center(3), &
                center(4),center(5),center(6), &
                center(7),center(8),center(9), &
                center(10),center(11),center(12),g)
                iexponents(ilownum) = iexponents(ilownum)+1
                apass=a
                bpass=b
            ENDIF

        ! The next two terms only arise is the angual momentum of the dimension
        ! of A that has already been lowered is not zero in B.  If a
        ! (px|u1x|px) was passed to this subroutine, we are now considering
        ! (s|u1x|px), and the following term does arise, as the x exponent on
        ! B is 1.

            IF (iexponents(ilownum+3) /= 0) THEN
                coeff = dble(iexponents(ilownum+3))/(2.d0*g)
                iexponents(ilownum+3) = iexponents(ilownum+3)-1
                xmomentrec=xmomentrec+coeff* &
                xmomentrecurse(apass,bpass, &
                iexponents(1),iexponents(2),iexponents(3), &
                iexponents(4),iexponents(5),iexponents(6), &
                iexponents(7),iexponents(8),iexponents(9), &
                center(1),center(2),center(3), &
                center(4),center(5),center(6), &
                center(7),center(8),center(9), &
                center(10),center(11),center(12),g)
                iexponents(ilownum+3) = iexponents(ilownum+3)+1
                apass=a
                bpass=b
            ENDIF

        ! The next two terms only arises if the exponent of the operator
        ! corresponding to the dimension of A that has already been lowered
        ! is not zero.  If a (px|u1x|px) was passed to this subroutine,
        ! we are now considering (s|u1x|px), and the following term
        ! does arise. If we had started with (px|u1y|px), it would now be
        ! (s|u1y|px), and it would not arise.

            IF (iexponents(ilownum+6) /= 0) THEN
                coeff = dble(iexponents(ilownum+6))/(2.d0*g)
                iexponents(ilownum+6) = iexponents(ilownum+6)-1
                xmomentrec=xmomentrec+coeff* &
                xmomentrecurse(apass,bpass, &
                iexponents(1),iexponents(2),iexponents(3), &
                iexponents(4),iexponents(5),iexponents(6), &
                iexponents(7),iexponents(8),iexponents(9), &
                center(1),center(2),center(3), &
                center(4),center(5),center(6), &
                center(7),center(8),center(9), &
                center(10),center(11),center(12),g)
                iexponents(ilownum+6) = iexponents(ilownum+6)+1
                apass=a
                bpass=b
            ENDIF
        ENDIF
    ENDIF
    return
    end 

