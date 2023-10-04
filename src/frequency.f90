#include "util.fh"
!---------------------------------------------------------------------!
! Created by Akhil Shajan on 03/10/2023                               !
!                                                                     !
! Previous contributors: Ed Brothers                                  !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

    subroutine frequency
    use allmod
    implicit double precision(a-h,o-z)

    dimension Vib(natom*3),SVec(natom*3,4), ATMASS(natom)
    dimension RMode(3*natom,3*natom), P(3*natom,3*natom)
 
    call prtAct(ioutfile,"Begin Frequency calculation")

! Note:  This can be changed.

    TempK = 298.15d0

! Conversion factor to get from atomic units:
! from atomic charge to ESU = 4.803242D-10
! from particle to mole = 6.0221367D23
! from Bohrs to cm = .529177249D-8
! And since the eigenvalues are 4 Pi^2 (freq)^2 you also need the speed
! of light in cm = 2.99792458D10

    convfact = (4.803242D-10)**2.d0
    convfact = convfact* 6.0221367D23
    convfact = convfact/ (4 *pi*pi)
    convfact = convfact/ (2.99792458D10)**2.d0
    convfact = convfact/ (.529177249D-8)**3.d0

! This procedure calculates the frequencies of the molecule given the
! hessian.

!    do I=1, natom
!       ATMASS(I)=emass(quick_molspec%iattype(I))
!    enddo

! First, mass weight the hessian.

    write (ioutfile,'(/" MASS WEIGHT THE HESSIAN. ")')
    do Iatm=1,natom
       AMASI = 1.0d0/sqrt(quick_molspec%iatmass(Iatm))
       II = 3*(Iatm-1)
       do I=1,3
          do Jatm=1,natom
             AMAS = AMASI/sqrt(quick_molspec%iatmass(Jatm))
             JJ = 3*(Jatm-1)
             do J=1,3
                quick_qm_struct%hessian(II+I,JJ+J)=quick_qm_struct%hessian(II+I,JJ+J)*AMAS
             enddo
           enddo
        enddo
    enddo

    call PriHessian(ioutfile,3*natom,quick_qm_struct%hessian,'f12.6')

! Second, project out translations/rotations from Mass-Weighted Hessian

  call ProjTRM(natom, xyz, quick_molspec%iatmass, P, RMode, quick_qm_struct%hessian)


! Diagonalize the Hessian.

    call MATDIAG(quick_qm_struct%hessian,3*natom,P,SVec,Vib,IErr)

!! print out vibrations



! Remove Zero frequencies

!    call ChkHES

! Find number of negative frequencies

!    call FndNEG

! Convert frequencies to cm**-1
! Mass Weight and normalize the normal modes
!  Calculated IR intensities (using unnormalized modes)

    write (ioutfile,'(/" THE HARMONIC FREQUENCIES (1/cm): ")')
    write (ioutfile,*)
    do I=1,natom*3
        Vib(I) = convfact*Vib(I)
        if (Vib(I) < 0.d0) then
            Vib(I) = -1.d0* DABS(Vib(I))**.5d0
        else
            Vib(I) = Vib(I)**.5d0
        endif
        write (ioutfile,'(6x,F15.5)') Vib(I)
    enddo
    write (ioutfile,*)
    
! Now we have the frequencies.  Before continuing, we need to see if the
! molecule is linear.  Note that this assumes the molecule is polyatomic.

    if (natom == 2) then
        ignore=5
    else
        ignore=5
        do I=3,natom
            CALL BNDANG(1,2,I,ANGLE)
            ANGLE = ANGLE*180.d0/pi
            if (ANGLE > 5.d0 .and. ANGLE < 175.d0) ignore=6
        enddo
    endif
    if (ignore == 5) write (ioutfile,'(/" MOLECULE IS LINEAR!! ")')

! Now we can calculate the zero point energy.
! ZPE = 1/2 (Sum over i) Harmonic(i)
! This is in wavenumbers, so:
! 1/cm * h * c
! 1/cm * J/sec * cm/sec = Joules/particle, and convert to Hartrees.
! This is from Radom and Scott, J. Chem. Phys. 1996, 100, 16502-13.

    write(ioutfile,'(2x,"------------------------")')
    write(ioutfile,'(5x,"Thermodynamic Data")')
    write(ioutfile,'(2x,"------------------------")')
    write (ioutfile,'("TEMPERATURE          = ",F15.7,"K")') tempK
    Ezp = 0.d0
    do I=ignore+1,natom*3
        Ezp = Ezp + Vib(I)
    enddo
    Ezp = Ezp*.5d0
    Ezp = Ezp*6.6260755D-34*2.99792458D10/4.3597482D-18
    write (ioutfile,'("ZERO POINT ENERGY    = ",F15.7)') Ezp

! We can also calculate the vibrational contribution to internal energy.

    Evib = 0.d0
    do I=ignore+1,natom*3
        vibtemp = Vib(I)*(6.6260755D-34)/1.380658D-23*2.99792458D10
        ratio = vibtemp/TempK
        Evib = Evib +vibtemp*(.5d0 + 1.d0/(DEXP(ratio)-1.d0))
    enddo
    Evib = Evib*1.380658D-23/4.3597482D-18
    write (ioutfile,'("VIBRATIONAL ENERGY   = ",F15.7)') Evib

! Now we need the translation and rotation.

    Etrans=1.5d0*TempK*1.380658D-23/4.3597482D-18
    write (ioutfile,'("TRANSLATIONAL ENERGY = ",F15.7)') Etrans

    if (ignore == 5) then
        Erot = TempK*1.380658D-23/4.3597482D-18
    else
        Erot = 1.5d0*TempK*1.380658D-23/4.3597482D-18
    endif
    write (ioutfile,'("ROTATIONAL ENERGY    = ",F15.7)') Etrans


    write (ioutfile,'("INTERNAL ENERGY      = ",F15.7)') &
    quick_qm_struct%Etot+Ezp+Etrans+Erot+Evib
    write (ioutfile,'("INTERNAL ENTHALPY    = ",F15.7)') &
    quick_qm_struct%Etot+Ezp+Etrans+Erot+Evib+TempK*1.380658D-23/4.3597482D-18

    call prtAct(ioutfile,"Finish Frequency calculation")
    end subroutine frequency


    SUBROUTINE ProjTRM(NATOMS, XC, ATMASS,  P, TRVec,  HESS)
    USE allmod
    IMPLICIT REAL*8(A-H,O-Z)

! Projects out from the Mass-Weighted Hessian matrix in Cartesian
! coordinates vectors corresponding to translations and
! infinitesimal rotations

    DIMENSION XC(3,NATOMS),ATMASS(NATOMS),P(3*NATOMS,*),& 
              TRVec(3*NATOMS,*),HESS(3*NATOMS,3*NATOMS)
    DIMENSION T(9),PMom(3)

! Transform to centre-of--mass coordinate system

    CALL COM(NATOMS,ATMASS,XC,CX,CY,CZ)

! Find the principal momenta and rotation generators

    CALL zeroVec(T,9)
    DO I=1, NATOMS
       X = XC(1,I)
       Y = XC(2,I)
       Z = XC(3,I)
       ami = ATMASS(I)
       T(1) = T(1) + ami*(Y*Y + Z*Z)
       T(5) = T(5) + ami*(X*X + Z*Z)
       T(9) = T(9) + ami*(X*X + Y*Y)
       T(2) = T(2) - ami*X*Y
       T(3) = T(3) - ami*X*Z
       T(6) = T(6) - ami*Y*Z
    ENDDO
    T(4) = T(2)
    T(7) = T(3)
    T(8) = T(6)

! Diagonalize T

    CALL MATDIAG(T,3,TRVec,P,PMom,IErr)

!! error out
    

! Set up Orthogonal coordinate vectors for translation and
! rotation about principal axes of inertia

    NAT3 = 3*NATOMS
    CALL zeroMatrix(TRVec,NAT3)
    CALL FormTRM(NATOMS,ATMASS,XC,T,TRVec)

!! print matrix

    

! Now form the Projection Matrix
    CALL zeroMatrix(P,NAT3)

    DO K = 1, 6
       DO J = 1, NAT3
          DO I = 1, NAT3
             P(I,J) = P(I,J) - TRVec(I,K)*TRVec(J,K)
          ENDDO
       ENDDO
    ENDDO

    DO I = 1, NAT3
       P(I,I) = 1.0d0 + P(I,I)
    ENDDO

!! print matrix

    

! Project out the translations/rotations from Hessian
!     HESS = P * HESS * P(t)

      CALL DGemm('N',    'N',    NAT3,   NAT3,   NAT3,&
                  1.0d0,    HESS,  NAT3,   P,      NAT3,&
                  0.0d0,   TRVec, NAT3)
      CALL DGemm('N',    'N',    NAT3,   NAT3,   NAT3,&
                  1.0d0,    P,     NAT3,   TRVec,  NAT3,&
                  0.0d0,   HESS,  NAT3)

!! Translations and Rotations Projected Out of Hessian

! Restore original coordinates

    DO I = 1, NATOMS
       XC(1,I) = XC(1,I) + CX
       XC(2,I) = XC(2,I) + CY
       XC(3,I) = XC(3,I) + CZ
    ENDDO

    RETURN

    END SUBROUTINE

    SUBROUTINE COM(NATOMS,ATMASS,XC,X,Y,Z)
    IMPLICIT REAL*8(A-H,O-Z)

!  Transform into centre-of-mass coordinate system

!  ARGUMENTS
!
!  NATOMS  -  number of atoms
!  ATMASS  -  atomic masses
!  XC      -  on input  original coordinates
!             on exit   centre-of-mass coordinates
!  X       -  on exit contains X centre-of-mass in original coordinate frame
!  Y       -  on exit contains Y centre-of-mass in original coordinate frame
!  Z       -  on exit contains Z centre-of-mass in original coordinate frame
!
    REAL*8 ATMASS(NATOMS),XC(3,NATOMS)



    TOTMAS = 0.0d0
    X = Zero
    Y = Zero
    Z = Zero

    DO IAtm=1,NATOMS
    ami = AtMASS(IAtm)
    TOTMAS = TOTMAS + ami
    X = X + ami*XC(1,IAtm)
    Y = Y + ami*XC(2,IAtm)
    Z = Z + ami*XC(3,IAtm)
    ENDDO

    X = X/TOTMAS
    Y = Y/TOTMAS
    Z = Z/TOTMAS

    DO IAtm=1,NATOMS
    XC(1,IAtm) = XC(1,IAtm) - X
    XC(2,IAtm) = XC(2,IAtm) - Y
    XC(3,IAtm) = XC(3,IAtm) - Z
    ENDDO

    RETURN
    END


    SUBROUTINE FormTRM(NATOMS,ATMASS,XC,T,V)
    USE allmod
    IMPLICIT REAL*8(A-H,O-Z)

    DIMENSION XC(3,NATOMS),ATMASS(NATOMS),T(9),V(3,NATOMS,6)

    PARAMETER (TollZERO=1.0d-8)


!  This routine generates vectors corresponding to translations
!  and infinitesimal rotations given the coordinates (in centre
!  of mass frame) and the eigenvectors of the inertia tensor

    NAT3 = 3*NATOMS

    DO I=1,NATOMS
       X = XC(1,I)
       Y = XC(2,I)
       Z = XC(3,I)
       ami = SQRT(ATMASS(I))
       CX = ami*(X*T(1) + Y*T(2) + Z*T(3))
       CY = ami*(X*T(4) + Y*T(5) + Z*T(6))
       CZ = ami*(X*T(7) + Y*T(8) + Z*T(9))
       V(1,I,1) = ami
       V(2,I,2) = ami
       V(3,I,3) = ami
       V(1,I,4) = CY*T(7) - CZ*T(4)
       V(2,I,4) = CY*T(8) - CZ*T(5)
       V(3,I,4) = CY*T(9) - CZ*T(6)
       V(1,I,5) = CZ*T(1) - CX*T(7)
       V(2,I,5) = CZ*T(2) - CX*T(8)
       V(3,I,5) = CZ*T(3) - CX*T(9)
       V(1,I,6) = CX*T(4) - CY*T(1)
       V(2,I,6) = CX*T(5) - CY*T(2)
       V(3,I,6) = CX*T(6) - CY*T(3)
    ENDDO

    DO I=1,6
       skal = SProd(NAT3,V(1,1,I),V(1,1,I))
       IF(skal.GT.TollZERO) THEN
           skal = 1.0d0/SQRT(skal)
           CALL VScal(NAT3,skal,V(1,1,I))
       ENDIF
    ENDDO

    RETURN

    END SUBROUTINE

    SUBROUTINE VScal(N,skal,V)
    IMPLICIT REAL*8(A-H,O-Z)

!  Scales all elements of a vector by skal
!    V = V * skal

    REAL*8 V(N)

    DO I=1,N
       V(I) = V(I)*skal
    ENDDO

    RETURN
    END SUBROUTINE

    DOUBLE PRECISION FUNCTION SProd(N,V1,V2)
    IMPLICIT REAL*8(A-H,O-Z)

!   Forms scalar (dot) product of two vectors

    REAL*8 V1(N),V2(N)

    PARAMETER (ZERO=0.0d0)

    SProd = ZERO
    DO I=1,N
      SProd = SProd + V1(I)*V2(I)
    ENDDO

    RETURN
    END FUNCTION
