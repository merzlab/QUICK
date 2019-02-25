!
!	findBlock.f90
!	new_quick
!
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!  This subroutine is to find the n th non-empty block

subroutine  findBlock(Inp,n)
      implicit none
      character(len=100) line
      integer inp,n,i,j,k
      integer check_char

      rewind(Inp)
      i=0
      j=0

      do k=1,100000
         read(Inp,'(a)',err=110,end=110) line
         i=check_char(line,1,100)
         if (i.eq.0) j=j+1
         if (j.eq.n) return
      enddo

 110  stop 'Error in inp file!'
end

! check if a line of char is blank
function check_char(line,ini,ifi)
      character line*(*)
      integer check_char

      check_char=0
      do i=ini,ifi
         if (line(i:i).ne.' ') check_char=check_char+1
      enddo

      return
end

function is_blank(line,ini,ifi)
    implicit none
    character line*(*)
    integer i,ini,ifi,check_char
    logical is_blank
    
    is_blank=.false.
    i=check_char(line,ini,ifi)
    if (i.eq.0) is_blank=.true.
    
    return
    
end function is_blank