!
!	quick_open.f90
!	new_quick
!
!	Created by Yipu Miao on 2/24/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! a univeral open subroutine and backup file automatically if it exists
subroutine quick_open(funit,filename,filestat,fileform,fileacc,log_ow)
   
    implicit none
   
    integer funit   !    logical unit number
    character(len=*) filename   ! file name 
    character(len=1) filestat   ! file status: 'N' new, 'O' old, 'R' replace, 'U' unknow.
    character(len=1) fileform   ! file form: 'F' formatted, 'U' unformatted
    character(len=1) fileacc    ! file access: 'R' write, 'W' write/read, 'A',append
    character(len=60) errInfo    ! error information

    character(len=7) fstat       !status keyword
    character(len=11) fform      !form keyword
    character(len=11) pos       !position keyword
    integer ios                 !i/o status variable
    integer i,flen,k1,k2
   
    character(len=1) ch
    character(len=100) run
    logical log_exist
    logical log_ow              ! if overwrite ?

#ifndef GNU 
    integer,external :: system
#endif

    ch='~'
    i=0
    
    ! file stat
    select case(filestat)
    case('N')
        fstat='NEW'
    case('O')
        fstat='OLD'
    case('R')
        fstat='REPLACE'
    case('U')
        fstat='UNKNOWN'
    case default
        fstat='UNKNOWN'
    end select
    
    ! file formation
    select case(fileform)
    case('U')
        fform='UNFORMATTED'
    case('F')
        fform='FORMATTED'
    case default
        fform='FORMATTED'
    end select
   

    ! file access
    if (fileacc == 'A') then
        pos = "APPEND"
    else
        pos = "ASIS"
    end if
    
    flen=len(filename)
    call EffChar(filename,1,flen,k1,k2)

    inquire(file=filename(k1:k2),exist=log_exist)   
   
    run='mv '//filename(k1:k2)//' '//filename(k1:k2)//ch
    if (log_exist.and.(.not.log_ow)) then
        i=system(run)
    endif 

    if (i /= 0) then
         write(6,'(2x,a,a)') 'Error: Fail to overwrite file ',filename
         call quick_exit(6,1)
    end if
    
    open(unit=funit,file=filename(k1:k2),status=fstat,form=fform,iostat=ios,position=pos)
   
   
    if (ios /= 0) then
         write(6,'(2x,a,i4,a,a)') 'Error: Fail to opne Unit=',funit, ',file name=', filename
         call quick_exit(6,1)
    end if
   
    if (pos /= "APPEND") rewind(funit)
   
    return
end subroutine quick_open 
