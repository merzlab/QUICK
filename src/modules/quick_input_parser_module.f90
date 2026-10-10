#include "util.fh"
module quick_input_parser_module

    implicit none
    
    private
    public :: read, index_keyword, found_keyword

    interface read
        module procedure read_integer_keyword
        module procedure read_float_keyword
        module procedure read_string_keyword
        module procedure read_string_endpoints_keyword
    end interface read

    contains

        integer function index_keyword(line, keyword) result(id)
            implicit none

            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword

            integer:: ind, pos
            integer:: len_keywd, len_line
            logical :: valid_left, valid_right
            character(len=:), allocatable :: key

            id = 0

            key = trim(adjustl(keyword))

            len_keywd = len(key)
            len_line  = len(line)

            if (len_keywd == 0) return

            ind = 1

            ! Takes into account the multiple occurences
            do
                pos = index(line(ind:),key)

                if (pos == 0) return

                ! Convert position relative to line(ind:) into position in line.
                pos = pos + ind - 1
 
                ! Check the character before the keyword.
                if (pos == 1) then
                    valid_left = .true.
                else
                    valid_left = line(pos-1:pos-1) == ' ' .or. line(pos-1:pos-1) == achar(9)
                end if

                ! Check the character after the keyword.
                if (len_keywd==len(line(pos:))) then
                    valid_right = .true.
                else
                    valid_right = line(pos+len_keywd:pos+len_keywd) == ' ' .or. &
                                  line(pos+len_keywd:pos+len_keywd) == achar(9) .or. &
                                  line(pos+len_keywd:pos+len_keywd) == '='
                end if

                if (valid_left .and. valid_right) then
                    id = pos
                    return
                endif

                ! Continue searching after the beginning of this rejected match.
                ind = pos + 1

                if (ind > len_line) return
            enddo
        end function index_keyword

        logical function found_keyword(line, keyword) result(found)
            implicit none

            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword

            integer:: ind, pos
            integer:: len_keywd, len_line
            logical :: valid_left, valid_right
            character(len=:), allocatable :: key

            found = .false.

            key = trim(adjustl(keyword))

            len_keywd = len(key)
            len_line  = len(line)

            if (len_keywd == 0) return

            ind = 1

            ! Takes into account the multiple occurences
            do
                pos = index(line(ind:),key)

                if (pos == 0) return

                ! Convert position relative to line(ind:) into position in line.
                pos = pos + ind - 1
 
                ! Check the character before the keyword.
                if (pos == 1) then
                    valid_left = .true.
                else
                    valid_left = line(pos-1:pos-1) == ' ' .or. line(pos-1:pos-1) == achar(9)
                end if

                ! Check the character after the keyword.
                if (len_keywd==len(line(pos:))) then
                    valid_right = .true.
                else
                    valid_right = line(pos+len_keywd:pos+len_keywd) == ' ' .or. &
                                  line(pos+len_keywd:pos+len_keywd) == achar(9) .or. &
                                  line(pos+len_keywd:pos+len_keywd) == '='
                end if

                if (valid_left .and. valid_right) then
                    found = .true.
                    return
                endif

                ! Continue searching after the beginning of this rejected match.
                ind = pos + 1

                if (ind > len_line) return
            enddo
        end function found_keyword

        subroutine trimSpace(i,j,line,keyword,found)
            implicit none
            integer, intent(out) :: i,j
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            logical, intent(out) :: found
    
            !first, go to the right to the end of the keyword
            i = index_keyword(line, keyword)+len_trim(adjustl(keyword))
        
            !ignore all spaces left to the equal sign
            do while(line(i:i)==' ' .or. line(i:i)==achar(9))
                i=i+1
            end do

            !if equal sign not found, return
            if(line(i:i) /= '=') then
                found = .false.
                return
            endif

            !ignore all spaces right to the equal sign
            i=i+1
            do while(line(i:i)==' ' .or. line(i:i)==achar(9))
                i=i+1
            end do

            !read value
            j = scan(line(i:len_trim(line)), ' ', .false.)  
            !if hit the end of the line so no space any more on the right
            if(j==0) then
                j = len_trim(line)
            endif

            found = .true.
            
        end subroutine trimSpace


        subroutine read_float_keyword(line, keyword, val, required)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            logical, intent(in), optional :: required
            double precision, intent(inout) :: val
            integer :: i,j,ierror
            logical :: found            
            logical :: reqdef !default value of required
            
            reqdef = .true.
            if(present(required)) then
                reqdef = required
            endif

            call trimSpace(i,j,line,keyword,found)

            if(reqdef .and. .not. found) then
                call PrtErr(OUTFILEHANDLE, "Keyword "//trim(keyword)//" needs an input value.")
                call quick_exit(OUTFILEHANDLE,1)
            endif

            if(found) then
                read(line(i:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif         
        end subroutine read_float_keyword

        subroutine read_integer_keyword(line, keyword, val, required)
            implicit none
            character(len=*),intent(in) :: line
            character(len=*), intent(in) :: keyword
            logical, intent(in), optional :: required
            integer, intent(inout) :: val
            integer :: i,j,ierror
            logical :: found 
            logical :: reqdef !default value of required          

            reqdef = .true.
            if(present(required)) then
                reqdef=required
            endif
    
            call trimSpace(i,j,line,keyword,found)

            if(reqdef .and. .not. found) then
                call PrtErr(OUTFILEHANDLE, "Keyword "//trim(keyword)//" needs an input value.")
                call quick_exit(OUTFILEHANDLE,1)
            endif 

            if(found) then
                read(line(i:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif
        end subroutine read_integer_keyword    
    
        subroutine read_string_keyword(line, keyword, val, required)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            logical, intent(in), optional :: required
            character(len=50), intent(inout) :: val
            integer :: i,j,ierror
            logical :: found   
            logical :: reqdef !default value of required         

            reqdef = .true.
            if(present(required)) then
                reqdef=required
            endif

            call trimSpace(i,j,line,keyword,found)
            
            if(reqdef .and. .not. found) then
                call PrtErr(OUTFILEHANDLE, "Keyword "//trim(keyword)//" needs an input value.")
                call quick_exit(OUTFILEHANDLE,1)
            endif

            if(found) then
                read(line(i:i+j-2),*, iostat=ierror) val
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif        
        end subroutine read_string_keyword

        subroutine read_string_endpoints_keyword(line, keyword, i, j, required, found)
            implicit none
            character(len=*), intent(in) :: line
            character(len=*), intent(in) :: keyword
            logical, intent(in), optional :: required
            integer, intent(out) :: i,j
            integer :: ierror
            logical, intent(out) :: found   
            logical :: reqdef !default value of required         

            reqdef = .true.
            if(present(required)) then
                reqdef=required
            endif

            call trimSpace(i,j,line,keyword,found)
            
            if(reqdef .and. .not. found) then
                call PrtErr(OUTFILEHANDLE, "Keyword "//trim(keyword)//" needs an input value.")
                call quick_exit(OUTFILEHANDLE,1)
            endif

            if(found) then
                if(ierror/=0) then
                    call PrtErr(OUTFILEHANDLE, "Error with keyword "//trim(keyword)//" encountered.")
                    call quick_exit(OUTFILEHANDLE,1)
                endif
            endif        
        end subroutine read_string_endpoints_keyword

end module quick_input_parser_module
