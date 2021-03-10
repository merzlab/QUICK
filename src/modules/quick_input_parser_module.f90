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

        subroutine checkformat(i,line,keyword)
            implicit none
            integer, intent(inout) :: i
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword

            i = index(line, trim(keyword)//'=')
            if(i==0) then
                call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                call quick_exit(OUTFILEHANDLE,1)
            endif
        end subroutine checkformat


        subroutine read_float_keyword(line, keyword, val)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            double precision :: val
            integer :: i,j,ierror
            
            call checkformat(i,line,keyword)
            j = scan(line(i:len_trim(line)), ' ', .false.)
            read(line(i+len_trim(keyword)+1:i+j-2),*, iostat=ierror) val
            if(ierror/=0) then
                call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                call quick_exit(OUTFILEHANDLE,1)
            endif         
                 
        end subroutine read_float_keyword

        subroutine read_integer_keyword(line, keyword, val)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            integer :: val
            integer :: i,j,ierror

            call checkformat(i,line,keyword)
            j = scan(line(i:len_trim(line)), ' ', .false.)
            read(line(i+len_trim(keyword)+1:i+j-2),*, iostat=ierror) val
            if(ierror/=0) then
                call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                call quick_exit(OUTFILEHANDLE,1)
            endif
        end subroutine read_integer_keyword    
    
        subroutine read_string_keyword(line, keyword, val)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            character(len=50) :: val
            integer :: i,j,ierror

            call checkformat(i,line,keyword)
            j = scan(line(i:len_trim(line)), ' ', .false.)
            read(line(i+len_trim(keyword)+1:i+j-2),*, iostat=ierror) val
            if(ierror/=0) then
                call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                call quick_exit(OUTFILEHANDLE,1)
            endif        
        end subroutine read_string_keyword

end module quick_input_parser_module
