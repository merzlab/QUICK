#include "util.fh"
!-----------------------------------------------------------
! whatis1
!-----------------------------------------------------------

subroutine whatis1(this, float)

  implicit none

  character, intent(in) :: this
  logical, intent(out) :: float
  integer :: ithis, i0, i9

  float = .false.

  ithis = ichar(this)
  i0 = ichar('0')
  i9 = ichar('9')
  if ((ithis >= i0) .and. (ithis <= i9)) then
     float = .true.
  elseif (this == '.') then
     float = .true.
  elseif (this == '-') then
     float = .true.
  elseif (this == '+') then
     float = .true.
  elseif (this == 'E') then
     float = .true.
  elseif (this == 'e') then
     float = .true.
  elseif (this == 'D') then
     float = .true.
  elseif (this == 'd') then
     float = .true.
  endif

end subroutine whatis1



!C------------------------------------------------------------CC
! whatis1i
!C------------------------------------------------------------CC

subroutine whatis1i(this, int)

  implicit none

  character, intent(in) :: this
  logical, intent(out) :: int
  integer :: ithis, i0, i9

  int = .false.

  if (this == '-') then
     int = .true.
  elseif (this == '+') then
     int = .true.
  else
     ithis = ichar(this)
     i0 = ichar('0')
     i9 = ichar('9')
     if ((ithis >= i0) .and. (ithis <= i9)) int = .TRUE. 
  endif

end subroutine whatis1i

!C------------------------------------------------------------CC
! whatis2
!C------------------------------------------------------------CC

subroutine whatis2(this, int, min)

  implicit none

  character, intent(in) :: this
  logical, intent(out) :: int, min
  integer :: ithis, i0, i9

  int = .false.
  min = .false.

  if (this == '-') then
     min = .true.
  else
     ithis = ichar(this)
     i0 = ichar('0')
     i9 = ichar('9')
     if ((ithis >= i0) .and. (ithis <= i9)) int = .TRUE. 
  endif

end subroutine whatis2


!C------------------------------------------------------------CC
! whatis7
!C------------------------------------------------------------CC

subroutine whatis7(this,char,num,parl,parr,comma,eq,white)
  implicit none

  character, intent(in) :: this
  logical, intent(out) :: char, num, parl, parr, comma, eq, white
  integer :: ithis, ia, iz, iaa, izz, i0, i9

  !C--------------------------------------------CC

  char = .false.
  num = .false.
  parl = .false.
  parr = .false.
  comma = .false.
  eq = .false.
  white = .false.
  if (this == ' ') then
     white = .true.
  elseif (this == ',') then
     comma = .true.
  elseif (this == '=') then
     eq = .true.
  elseif (this == '(') then
     parl = .true.
  elseif (this == ')') then
     parr = .true.
  elseif (this == '/') then
     eq = .true.
  elseif (this == '+') then
     num = .true.
  elseif (this == '.') then
     num = .true.
  elseif (this == '-') then
     num = .true.
  else
     ithis = ichar(this)
     ia = ichar('a')
     iz = ichar('z')
     iaa = ichar('A')
     izz = ichar('Z')
     i0 = ichar('0')
     i9 = ichar('9')
     if (((ithis >= ia) .and. (ithis <= iz)) .OR. &
          ((ithis >= iaa) .and. (ithis <= izz))) then
        char = .true.
     elseif ((ithis >= i0) .and. (ithis <= i9)) then
        num = .true.
     else
        white = .true.
     endif
  endif

end subroutine whatis7