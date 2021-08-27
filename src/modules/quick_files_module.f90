!---------------------------------------------------------------------!
! Updated by Madu Manathunga on 07/07/2020                            !
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

#include "util.fh"

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

    character(len=80) :: inFileName     = ''
    character(len=80) :: outFileName    = ''
    character(len=80) :: dmxFileName    = ''
    character(len=80) :: rstFileName    = ''
    character(len=80) :: CPHFFileName   = ''
    character(len=80) :: dataFileName   = ''
    character(len=80) :: intFileName    = ''


    ! Basis set and directory
    character(len=240) :: basisDir       = ''
    character(len=240) :: sadGuessDir    = ''
    character(len=320) :: basisFileName = ''
    character(len=80) :: basisSetName   = ''

    ! ecp basis set and directory
    character(len=80) :: ECPDir         = ''
    character(len=80) :: ECPFileName    = ''

    ! custom basis set and directory
    character(len=80) :: basisCustName  = ''
    character(len=80) :: PDBFileName    = ''

    integer :: inFile         = INFILEHANDLE     ! input file
    integer :: iOutFile       = OUTFILEHANDLE    ! output file
    integer :: iDmxFile       = DMXFILEHANDLE    ! density matrix file
    integer :: iRstFile       = RSTFILEHANDLE    ! Restricted file
    integer :: iCPHFFile      = CPHFFILEHANDLE   ! CPHF file
    integer :: iBasisFile     = BASISFILEHANDLE  ! basis set file
    integer :: iECPFile       = ECPFILEHANDLE    ! ECP file
    integer :: iBasisCustFile = BASISCFILEHANDLE ! custom basis set file
    integer :: iPDBFile       = PDBFILEHANDLE    ! PDB input file
    integer :: iDataFile      = DATAFILEHANDLE   ! Data file, similar to chk file in gaussian
    integer :: iIntFile       = INTFILEHANDLE    ! integral file

    logical :: fexist = .false.         ! Check if file exists

    logical :: isTemplate = .false.   ! is input file a template (i.e. only the keywords)
    integer :: wrtStep = 1            ! current step for writing to output file.

    contains

    !------------
    ! Setup input output files and basis dir
    !------------
    subroutine set_quick_files(api,ierr)

        implicit none
        ! Pass-in parameter:
        integer, intent(inout) :: ierr    ! Error Flag
        logical, intent(in) :: api

        ! Local Varibles
        integer :: i

        ! Read enviromental variables: QUICK_BASIS and ECPs
        ! those can be defined in ~/.bashrc
        call get_environment_variable("QUICK_BASIS",basisdir)

        call get_environment_variable("ECPs",ecpdir)

        ! Read argument, which is input file, usually ".in" file and prepare files:
        ! .out: output file
        ! .dmx: density matrix file
        ! .rst: coordinates file
        ! .pdb: PDB file (can be input if use PDB keyword)
        ! .cphf: CPHF file

        ! if quick is in libary mode, use .qin and .qout extensions
        ! for input and output files.

        if(.not.api) then
          call getarg(1,inFileName)
          i = index(inFileName,'.')
          if(i .eq. 0) then
            write(0,'("| Error: Invalid input file name.")')
            call quick_exit(0,1)
          endif
        else
          i = index(inFileName,'.')  
        endif

        outFileName=inFileName(1:i-1)//'.out'

        dmxFileName=inFileName(1:i-1)//'.dmx'
        rstFileName=inFileName(1:i-1)//'.rst'
        CPHFFileName=inFileName(1:i-1)//'.cphf'
        pdbFileName=inFileName(1:i-1)//'.pdb'
        dataFileName=inFileName(1:i-1)//'.dat'
        intFileName=inFileName(1:i-1)//'.int'

        return

    end subroutine

    subroutine read_basis_file(keywd,ierr)

        use quick_exception_module
        use quick_mpi_module

        implicit none

        !Pass-in Parameter
        character keywd*(*)
        character(len=80) :: line
        character(len=320) :: basis_sets  !stores full path to basis_sets file
        character(len=50) :: search_keywd !keywd packed with '=',used for searching basis file name
        character(len=50) :: tmp_keywd
        character(len=50) :: tmp_basisfilename
        integer, intent(inout) :: ierr

        ! local variables
        integer i,j,k1,k2,k3,k4,iofile,io,flen,f0,f1,lenkwd
        logical present

        i = 1
        j = 100
        iofile = 0
        io = 0
        tmp_basisfilename = "NULL"

        ! Gaussian Style Basis. Written by Alessandro GENONI 03/07/2007
        if (index(keywd,'BASIS=') /= 0) then

            !Get the length of keywd
            lenkwd=len_trim(keywd)

            i = index(keywd,'BASIS=',.false.)

            j = scan(keywd(i:lenkwd),' ',.false.)

            basis_sets=trim(basisdir) // "/basis_link"

            basisSetName = keywd(i+6:i+j-2)
            search_keywd= "#" // trim(basisSetName)
            ! Check if the basis_link file exists

            inquire(file=trim(basis_sets),exist=fexist)
            if (.not.fexist) then
                call PrtErr(iOutFile,'basis_link file is not accessible.')
                call PrtMsg(iOutFile,'Check if QUICK_BASIS environment variable is set.')
                call quick_exit(iOutFile,1)
            end if

            SAFE_CALL(quick_open(ibasisfile,basis_sets,'O','F','W',.true.,ierr))

            do while (iofile  == 0 )
                read(ibasisfile,'(A80)',iostat=iofile) line

                
                    call upcase(line,80)
                    
                    if(index(line,trim(search_keywd)) .ne. 0) then
                        tmp_basisfilename=trim(line(39:74))
                        iofile=1
                    endif
            enddo

            close(ibasisfile)

            tmp_basisfilename = tmp_basisfilename(1:len_trim(tmp_basisfilename)-4)

            basisfilename=trim(basisdir) // "/" // trim(tmp_basisfilename) //'.BAS'

            ! also set the sad guess directory
            sadGuessDir=trim(basisdir) // "/" // trim(tmp_basisfilename) // '.SAD'

            ! Check if basis file exists. Otherwise, quit program.
            inquire(file=trim(basisfilename),exist=fexist)

            if (.not.fexist) then
                call PrtErr(iOutFile,'Requested basis set does not exist or basis_link file not properly configured.')
                call PrtMsg(iOutFile,'Fix the basis_link file or add your basis set as a new entry. Check the user manual.')
                call quick_exit(iOutFile,1)
            end if

        else
            basisfilename = trim(basisdir) // '/STO-3G.BAS'    ! default
            sadGuessDir   = trim(basisdir) // '/STO-3G.SAD'
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

        write(io,'(" BASIS SET = ",a,",",2X,"TYPE = CARTESIAN")') trim(basisSetName)
        write(io,'("| BASIS FILE = ",a)') trim(basisfilename)

    end subroutine

    subroutine print_ecp_file(io)
        implicit none

        ! pass-in Parameter
        integer io

        ! instant variables
        integer i,j,k1,k2

        write(io,'("| ECP FILE = ",a)') trim(ecpfilename)
    end subroutine


    subroutine print_quick_io_file(io,ierr)
        implicit none

        ! Pass-in parameters:
        integer, intent(inout) :: ierr    ! Error Flag
        integer io      ! file to write

        write (io,'("| INPUT FILE :    ",a)') trim(inFileName)
        write (io,'("| OUTPUT FILE:    ",a)') trim(outFileName)
        write (io,'("| DATA FILE  :    ",a)') trim(dataFileName)
        write (io,'("| BASIS SET PATH: ",a)') trim(basisdir)

        return
    end subroutine print_quick_io_file



end module quick_files_module
