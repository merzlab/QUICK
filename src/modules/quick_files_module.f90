!---------------------------------------------------------------------!
! Updated by Madu Manathunga on 05/28/2020                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     ! 
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

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

    character(len=80) :: inFileName   = ''
    character(len=80) :: outFileName  = ''
    character(len=80) :: dmxFileName  = ''
    character(len=80) :: rstFileName  = ''
    character(len=80) :: CPHFFileName = ''
    character(len=80) :: dataFileName = ''
    character(len=80) :: intFileName  = ''
    
    
    ! Basis set and directory
    character(len=80) :: basisDir
    character(len=120) :: basisFileName
    character(len=80) :: basisSetName    
    
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

    logical :: fexist = .false.          ! Check if file exists

    logical :: isTemplate = .false.   ! is input file a template (i.e. only the keywords)
    integer :: wrtStep = 1            ! current step for writing to output file. 
    
    contains
    
    !------------
    ! Setup input output files and basis dir
    !------------
    subroutine set_quick_files(ierr)

        implicit none
        ! Pass-in parameter:
        integer :: ierr    ! Error Flag
        
        ! Local Varibles
        integer :: i
        
        ierr=1

        ! Read enviromental variables: QUICK_BASIS and ECPs
        ! those can be defined in ~/.bashrc
        call getenv("QUICK_BASIS",basisdir)
        basisdir=trim(basisdir)
        
        call getenv("ECPs",ecpdir)
      
        ! Read argument, which is input file, usually ".in" file and prepare files:
        ! .out: output file
        ! .dmx: density matrix file
        ! .rst: coordinates file
        ! .pdb: PDB file (can be input if use PDB keyword)
        ! .cphf: CPHF file

        ! if quick is in libary mode, use .qin and .qout extensions 
        ! for input and output files.  

        if(.not. isTemplate) call getarg(1,inFileName)

        i = index(inFileName,'.')

        if(i .eq. 0) i = index(inFileName,' ')
      
        if(isTemplate) then
          outFileName=inFileName(1:i-1)//'.qout'
        else
          outFileName=inFileName(1:i-1)//'.out'
        endif

        dmxFileName=inFileName(1:i-1)//'.dmx'
        rstFileName=inFileName(1:i-1)//'.rst'
        CPHFFileName=inFileName(1:i-1)//'.cphf'
        pdbFileName=inFileName(1:i-1)//'.pdb'
        dataFileName=inFileName(1:i-1)//'.dat'
        intFileName=inFileName(1:i-1)//'.int'
        

!        write(*,*) inFileName, outFileName

        ierr=0
        return
        
    end subroutine
    
    subroutine read_basis_file(keywd)
        implicit none
        
        !Pass-in Parameter
        character keywd*(*)
        character(len=80) :: line
        character(len=120) :: basis_sets  !stores full path to basis_sets file
        character(len=16) :: search_keywd !keywd packed with '=',used for searching basis file name
        character(len=16) :: tmp_keywd
        character(len=16) :: tmp_basisfilename

        ! local variables
        integer i,j,k1,k2,k3,k4,iofile,io,flen,f0,f1,lenkwd
        logical present
        
        i = 1
        j = 100 
        iofile = 0
        io = 0
        tmp_basisfilename = "NULL"


!        call EffChar(basisdir,i,j,k1,k2)
        
!        call rdword(ecpdir,k3,k4) !AG 03/05/2007
!        call EffChar(ecpdir,i,j,k3,k4)
              
        ! Gaussian Style Basis. Written by Alessandro GENONI 03/07/2007
        if (index(keywd,'BASIS=') /= 0) then

            !Get the length of keywd
            lenkwd=len_trim(keywd)

            i = index(keywd,'BASIS=',.false.)

            j = scan(keywd(i:lenkwd),' ',.false.)
           
            !write(basis_sets,*)  trim(basisdir),"/basis_link"
            basis_sets=trim(basisdir) // "/basis_link"

            basisSetName = keywd(i+6:i+j-2)
            search_keywd= "#" // trim(basisSetName)

            ! Check if the basis_link file exists
            !flen=len(basis_sets)
            !call EffChar(basis_sets,1,flen,f0,f1)

            inquire(file=trim(basis_sets),exist=fexist)
            if (.not.fexist) then
                call PrtErr(iOutFile,'basis_link file is not accessible.')                
                call PrtMsg(iOutFile,'Check if QUICK_BASIS environment variable is set.')
                call quick_exit(iOutFile,1)
            end if   

            call quick_open(ibasisfile,basis_sets,'O','F','W',.true.)
            
            do while (iofile  == 0 )
                read(ibasisfile,'(A80)',iostat=iofile) line
                
                    call upcase(line,80)
                    if(index(line,trim(search_keywd)) .ne. 0) then
                        tmp_basisfilename=trim(line(39:74))
                        iofile=1
                    endif
            enddo

            close(ibasisfile)

            basisfilename=trim(basisdir) // "/" // tmp_basisfilename

            ! Check if basis file exists. Otherwise, quit program.
            inquire(file=trim(basisfilename),exist=fexist)
            write(*,*) trim(basisfilename)
            if (.not.fexist) then
                call PrtErr(iOutFile,'Requested basis set does not exist or basis_link file not properly configured.')
                call PrtMsg(iOutFile,'Fix the basis_link file or add your basis set as a new entry. Check the user manual.')
                call quick_exit(iOutFile,1)
            end if

        else
            basisfilename = trim(basisdir) // '/STO-3G.BAS'    ! default
        endif
        
        if (index(keywd,'ECP=') /= 0) then
            i = index(keywd,'ECP=')
            call rdword(keywd,i,j)
            ecpfilename = ecpdir(k3:k4) // '/' // keywd(i+4:j)
            basisfilename = trim(basisdir) // '/' //keywd(i+4:j)
            if (keywd(i+4:j) == "CUSTOM") BasisCustName = trim(basisdir) // '/CUSTOM'
        endif
        
        ! a compatible mode for basis set file if files like STO-3G.BAS didn't exist, 
        ! the program will search STO-3G
        inquire(file=basisfilename,exist=present)
        if (.not.present) then
            i=index(basisfilename,'.')
!            basisfilename(k1:i+3)=basisfilename(k1:i-1)//'    '
        !    inquire(file=basisfilename,exist=present)
        !    if (.not.present) then
        !        call quick_exit(6,1)
        !    end if
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
        
        !write(io,'("| BASIS SET = ",a)') basisfilename(j+1:k2)
        write(io,'("| BASIS SET = ",a)') basisSetName
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
