#include "util.fh"
module quick_input_parser_module

    implicit none
    
    private
    public :: read

    interface read
        module procedure read_integer_keyword
        module procedure read_float_keyword
        module procedure read_string_keyword
    end interface read

    contains

        subroutine read_float_keyword(keywdline, keyword, val)
            implicit none
            character(len=*), intent(in) :: keywdline
            character(len=*), intent(in) :: keyword
            double precision :: val
            integer :: i,j,ierror
            
            if(index(keywdline,keyword) /= 0) then
                i = index(keywdline, trim(keyword)//trim('='))
                if(i==0) then
                    call PrtErr(OUTFILEHANDLE, "USE keyword=val format in input")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
                j = scan(keywdline(i:len_trim(keywdline)), ' ', .false.)
                read(keywdline(i+len_trim(keyword)+1:i+j-2),*, iostat=ierror) val              
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "USE keyword=val format in input")
                    call quick_exit(OUTFILEHANDLE,1)
                endif                
            endif
        end subroutine read_float_keyword

        subroutine read_integer_keyword(keywdline, keyword, val)
            implicit none
            character(len=*), intent(in) :: keywdline
            character(len=*), intent(in) :: keyword
            integer :: val
            integer :: i,j,ierror

            if(index(keywdline,keyword) /= 0) then
                i = index(keywdline, trim(keyword)//trim('='))
                if(i==0) then
                    call PrtErr(OUTFILEHANDLE, "USE keyword=val format in input")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
                j = scan(keywdline(i:len_trim(keywdline)), ' ', .false.)
                read(keywdline(i+len_trim(keyword)+1:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "USE keyword=val format in input")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif
        end subroutine read_integer_keyword    
    
        subroutine read_string_keyword(keywdline, keyword, val)
            implicit none
            character(len=*), intent(in) :: keywdline
            character(len=*), intent(in) :: keyword
            character(len=50) :: val
            integer :: i,j,ierror

            if(index(keywdline,keyword) /= 0) then
                i = index(keywdline, trim(keyword)//trim('='))
                if(i==0) then
                    call PrtErr(OUTFILEHANDLE, "USE keyword=val format in input")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
                j = scan(keywdline(i:len_trim(keywdline)), ' ', .false.)
                read(keywdline(i+len_trim(keyword)+1:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "USE keyword=val format in input")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif
        end subroutine read_string_keyword

end module quick_input_parser_module
