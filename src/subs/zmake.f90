!********************************************
! zmake
!--------------------------------------------
! by Ed Brothers. June 1998.
! 34567890123456789012345678901234567890123345678901234567890123456789012<stop

    subroutine zmake
    use quick_molspec_module
    use quick_files_module
    use quick_constants_module, only : symbol
    implicit double precision(a-h,o-z)
    double precision, dimension(3,natom) :: zmat
    integer, dimension(3,natom) :: izmat,ioptions
    double precision, dimension(3) :: ztemp
    double precision :: degree = 57.2957795131d0

! Initialization block
    r=0.d0
    angle=0.d0
    dih=0.d0
    K=1
    L=2
    M=3
    call PrtAct(ioutfile,"Begin Zmake conversion")
    
! The purpose of this subroutine is to return internal coordinates to
! the output file.  Conversion from cartesian to internal is not automatic,
! although various subprograms do it.  This is why there are so many calls in
! this subroutine.  Note that the routine checks to see if the values already
! present, and, if so simply returns.


! Initialize the z-matrix values if they have not been filled already.

    do J=1,natom
        do I=1,3
            zmat(i,j)=0.d0
            izmat(i,j)=0
            ioptions(i,j)=0
        enddo
    enddo

! Calculate distance from atom 1 to atom J.

    do 10 J=2,natom
        do 20 I=1,3
            r=r+(xyz(i,j)-xyz(i,1))**2
        20 enddo
        zmat(1,j)=sqrt(r)*0.5291772083d0
        izmat(1,j)=K
        ioptions(1,j)=1
        r=0.d0
    10 enddo

! Calculate angle j-1-2  This is the angle used in the z-matrix.

    do 30 J=3,natom
        call bndang(L,K,J,angle)
        zmat(2,j)=angle
        izmat(2,j)=L
        ioptions(2,j)=1
        angle=0.d0
    30 enddo

! Calculate dihderal 1-2-3-j.

    do 40 J=4,natom
        call dihedr(xyz,J,K,L,M,DIH)
        zmat(3,j)=dih
        izmat(3,j)=M
        ioptions(3,j)=1
        dih=0.d0
    40 enddo

    write(ioutfile,*) ' '
    write(ioutfile,*) 'Z-MATRIX:'

    WRITE(IOUTfile,50)
    50 FORMAT(/3X,'ATOM',2X,'ELEMENTAL',2X,'BOND LENGTH',4X, &
    'BOND ANGLE',4X,'DIHEDRAL ANGLE'/2X,'NUMBER',2X,'SYMBOL', &
    4X,'(ANGSTROMS)',4X,'(DEGREES)',7X,'(DEGREES)'/5X,'I', &
    17X,'I-NA',9X,'I-NA-NB',8X,'I-NA-NB-NC',6X,'NA',3X,'NB', &
    3X,'NC'/)
    do I=1,NATOM
        IAT = quick_molspec%iattype(I)
        ZTEMP(1) = ZMAT(1,I)
        ZTEMP(2) = ZMAT(2,I)*DEGREE
        ZTEMP(3) = ZMAT(3,I)*DEGREE
    
    ! KEEP DIHEDRAL IN RANGE -180 TO +180
    
        ABSZ = ABS(ZTEMP(3))
        if(ABSZ > 180.0D0)then
            ZTEMP(3) = ZTEMP(3) - SIGN(360.0D0,ZTEMP(3))
        endif
    
        WRITE(IOUTfile,60) I,SYMBOL(IAT),(ZTEMP(J),IOPTions(J,I),J=1,3) &
        ,(IZMAT(J,I),J=1,3)
        60 FORMAT(1X,I5,6X,A2,5X,F9.5,2X,I1,2X,F10.5,2X,I1,3X,F10.5,2X, &
        I1,2X,3I5)
    enddo
    call PrtAct(ioutfile,"End Zmake conversion")

    end subroutine zmake


