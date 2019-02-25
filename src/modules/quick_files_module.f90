!
!	quick_files_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!  File module.
module quick_files_module
!------------------------------------------------------------------------
!  ATTRIBUTES  : inFileName,outFileName,dmxFileName,rstFileName,CPHFFileName
!                basisDir,BasisFileName,ECPDir,ECPFileName,BasisCustName,PDBFileName
!  SUBROUTINES : set_quick_files
!                print_quick_io_files
!  FUNCTIONS   : none
!  DESCRIPTION : This module is to read argument from command and save file information.
!  AUTHOR      : Yipu Miao
!------------------------------------------------------------------------

    implicit none

    character(len=80) :: inFileName
    character(len=80) :: outFileName
    character(len=80) :: dmxFileName
    character(len=80) :: rstFileName
    character(len=80) :: CPHFFileName
    character(len=80) :: dataFileName
    character(len=80) :: intFileName
    
    
    ! Basis set and directory
    character(len=80) :: basisDir
    character(len=80) :: basisFileName
    
    ! ecp basis set and directory
    character(len=80) :: ECPDir
    character(len=80) :: ECPFileName
    
    ! custom basis set and directory
    character(len=80) :: basisCustName
    character(len=80) :: PDBFileName
    
    integer :: inFile = 15            ! input file
    integer :: iOutFile = 16          ! output file
    integer :: iDmxFile = 17          ! density matrix file
    integer :: iRstFile = 18          ! Restricted file
    integer :: iCPHFFile = 19         ! CPHF file
    integer :: iBasisFile = 20        ! basis set file
    integer :: iECPFile = 21          ! ECP file
    integer :: iBasisCustFile = 22    ! custom basis set file
    integer :: iPDBFile = 23          ! PDB input file
    integer :: iDataFile = 24         ! Data file, similar to chk file in gaussian
    integer :: iIntFile = 25          ! integral file
    
    contains
    
    !------------
    ! Setup input output files and basis dir
    !------------
    subroutine set_quick_files(ierr)
        implicit none
        ! Pass-in parameter:
        integer ierr    ! Error Flag
        
        ! Local Varibles
        integer i
        
        ierr=1
        
        ! Read enviromental variables: QUICK_BASIS and ECPs
        ! those can be defined in ~/.bashrc
        call getenv("QUICK_BASIS",basisdir)
        call getenv("ECPs",ecpdir)
      
        ! Read argument, which is input file, usually ".in" file and prepare files:
        ! .out: output file
        ! .dmx: density matrix file
        ! .rst: coordinates file
        ! .pdb: PDB file (can be input if use PDB keyword)
        ! .cphf: CPHF file
        call getarg(1,inFileName)
        i = index(inFileName,'.')
        if(i .eq. 0) i = index(inFileName,' ')
      
        outFileName=inFileName(1:i-1)//'.out'
        dmxFileName=inFileName(1:i-1)//'.dmx'
        rstFileName=inFileName(1:i-1)//'.rst'
        CPHFFileName=inFileName(1:i-1)//'.cphf'
        pdbFileName=inFileName(1:i-1)//'.pdb'
        dataFileName=inFileName(1:i-1)//'.dat'
        intFileName=inFileName(1:i-1)//'.int'

        ierr=0
        return
        
    end subroutine
    
    subroutine read_basis_file(keywd)
        implicit none
        
        !Pass-in Parameter
        character keywd*(*)
        
        ! local variables
        integer i,j,k1,k2,k3,k4
        logical present
        
        i = 0
        j = 100 
        ! read basis directory and ECP basis directory
        call rdword(basisdir,i,j)
        call EffChar(basisdir,i,j,k1,k2)
        
        call rdword(ecpdir,i,j) !AG 03/05/2007
        call EffChar(ecpdir,i,j,k3,k4)
              
        ! Gaussian Style Basis. Written by Alessandro GENONI 03/07/2007
        if (index(keywd,'BASIS=') /= 0) then
            i = index(keywd,'BASIS=')
            call rdword(keywd,i,j)
            write(basisfilename,*) basisdir(k1+1:k2),"/",keywd(i+6:j) 
        else
            basisfilename = basisdir(k1:k2) // '/STO-3G.BAS'    ! default
        endif
        
        if (index(keywd,'ECP=') /= 0) then
            i = index(keywd,'ECP=')
            call rdword(keywd,i,j)
            ecpfilename = ecpdir(k3:k4) // '/' // keywd(i+4:j)
            basisfilename = basisdir(k1:k2) // '/' //keywd(i+4:j)
            if (keywd(i+4:j) == "CUSTOM") BasisCustName = basisdir(k1:k2)// '/CUSTOM'
        endif
        
        ! a compatible mode for basis set file if files like STO-3G.BAS didn't exist, 
        ! the program will search STO-3G
        inquire(file=basisfilename,exist=present)
        if (.not.present) then
            i=index(basisfilename,'.')
!            basisfilename(k1:i+3)=basisfilename(k1:i-1)//'    '
            inquire(file=basisfilename,exist=present)
            if (.not.present) then
        !        call quick_exit(6,1)
            end if
        endif
      
    end subroutine
    
    subroutine print_basis_file(io)
        implicit none
        
        ! pass-in Parameter
        integer io
        
        ! instant variables
        integer i,j,k1,k2
        
        call EffChar(basisfilename,1,80,k1,k2)
        do i=k1,k2
            if (basisfilename(i:i).eq.'/') j=i
        enddo
        
        write(io,'("| BASIS SET = ",a)') basisfilename(j+1:k2)
        write(io,'("| BASIS FILE = ",a)') basisfilename(k1:k2)
    end subroutine
    
    subroutine print_ecp_file(io)
        implicit none
        
        ! pass-in Parameter
        integer io
        
        ! instant variables
        integer i,j,k1,k2
        
        call EffChar(ecpfilename,1,80,k1,k2)
        write(io,'("| ECP FILE = ",a)') ecpfilename(k1:k2)
    end subroutine
    
    
    subroutine print_quick_io_file(io,ierr)
        implicit none
        
        ! Pass-in parameters:
        integer ierr    ! Error Flag
        integer io      ! file to write
        
        integer k1,k2
        
        ierr=1
        
        call EffChar(inFileName,1,30,k1,k2)
        write (io,'(" INPUT FILE :    ",a)') inFileName(k1:k2)
        call EffChar(outFileName,1,30,k1,k2)
        write (io,'(" OUTPUT FILE:    ",a)') outFileName(k1:k2)
        call EffChar(dataFileName,1,30,k1,k2)
        write (io,'(" DATE FILE  :    ",a)') dataFileName(k1:k2)
        call EffChar(basisdir,1,80,k1,k2)
        write (io,'(" BASIS SET PATH: ",a)') basisdir(k1:k2)
        
        ierr=0
        return
    end subroutine print_quick_io_file
    
    

end module quick_files_module
