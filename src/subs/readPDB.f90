!
!	readPDB.f90
!	new_quick
!
!	Created by Yipu Miao on 3/1/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!********************************************************
! readpdb
!--------------------------------------------------------
! Subroutines to read pdb and put it to input file
!
! Yipu Miao: A bug is can't find elements with two letters like "CL". 
!
    subroutine readpdb(inputfile)
    use allmod
    implicit none
!
! input- (integer) inputfile the QUICK-stype input file the this subroutine will port.
!    
    integer inputfile
    
    integer i,j,number
    character(len=200) :: keyWD                 ! Key words input file contains
    character(len=200) :: tmpWD                 ! temp character words
    character*6,allocatable:: sn(:)             ! series no.
    double precision,allocatable::coord(:,:)    ! cooridnates
    integer,allocatable::class(:),ttnumber(:)   ! class and residue number
    character*4,allocatable::atomname(:)        ! atom name
    character*3,allocatable::residue(:)         ! residue name


    open(inputfile,file=inFileName)
    
    read(inputfile,'(A200)') keyWD
    
    allocate(sn(number))
    allocate(coord(3,number))
    allocate(class(number),ttnumber(number))
    allocate(atomname(number))
    allocate(residue(number))
        
    open(iPDBFile,file=PDBFileName)
    read(iPDBFile,'(A200)') tmpWD
    i=1
    do while (index(tmpWD,"TER").eq.0)
      read(tmpWD,100) sn(i),ttnumber(i),atomname(i),residue(i),class(i), &
              (coord(j,i),j=1,3)
      i=i+1
      read(iPDBFile,'(A200)') tmpWD
    enddo

    number=i-1
    close(iPDBFile)
    
    call PrtAct(iOutFile," Read PDB file")
    rewind(inputfile)
    write(inputfile,'(A200)') keyWD
    write(inputfile,*)
    
    do i=1,number
      tmpWD=atomname(i)
      atomname(i)=tmpWD(2:2)
    enddo
    
    do i=1,number
        write(inputfile,200) atomname(i),(coord(j,i),j=1,3)
    enddo
    write(inputfile,*)
    write(iOutFile,*) "Total Atom Read from PDB=",number
    write(iOutFile,*) "Residues Read from PDB=",class(number)
    call PrtAct(iOutFile," done PDB Reading")
    call flush(iOutFile)

    deallocate(sn)
    deallocate(coord)
    deallocate(class,ttnumber)
    deallocate(atomname)
    deallocate(residue)
    
    close(inputfile)
100 format(a6,1x,I4,1x,a4,1x,a3,3x,I3,4x,3f8.3)
200 format(a4,4x,3f8.3)
    
    return
    
    end subroutine