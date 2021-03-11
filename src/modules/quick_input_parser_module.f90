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

        subroutine trimSpace(i,j,line,keyword,found)
            implicit none
            integer, intent(out) :: i,j
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            logical, intent(out) :: found

            print *, "keyword is ", keyword
    
            !first, go to the right to the end of the keyword
            i = index(line, trim(keyword))+len_trim(keyword)
            print *, "line(i:i+4) is", line(i:i+4)
        
            !ignore all spaces left to the equal sign
            do while(line(i:i)==' ')
                i=i+1
            end do

            !if equal sign not found, return
            if(line(i:i) /= '=') then
                print *, "didn't find assigned value"
                found = .false.
                return
            endif

            !ignore all spaces right to the equal sign
            i=i+1
            do while(line(i:i)==' ')
                i=i+1
            end do

            !read value
            !print *, "for j, line(i:len_trim(line)) is",line(i:len_trim(line))
            j = scan(line(i:len_trim(line)), ' ', .false.)  
            !if hit the end of the line so no space any more on the right
            if(j==0) then
                j = len_trim(line)
            endif
            
            found = .true.
            
        end subroutine trimSpace


        subroutine read_float_keyword(line, keyword, val)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            double precision :: val
            integer :: i,j,ierror
            logical :: found            

            call trimSpace(i,j,line,keyword,found)
            if(found) then
                read(line(i:i+j-2),*, iostat=ierror) val
                print *, "val is ", val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif         
        end subroutine read_float_keyword

        subroutine read_integer_keyword(line, keyword, val)
            implicit none
            character(len=*),intent(in) :: line
            character(len=*), intent(in) :: keyword
            integer :: val
            integer :: i,j,ierror
            logical :: found 
          
            call trimSpace(i,j,line,keyword,found)
            if(found) then
                read(line(i:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif
        end subroutine read_integer_keyword    
    
        subroutine read_string_keyword(line, keyword, val)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            character(len=50) :: val
            integer :: i,j,ierror
            logical :: found            

            call trimSpace(i,j,line,keyword,found)
            if(found) then
                read(line(i:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif        
        end subroutine read_string_keyword

end module quick_input_parser_module
