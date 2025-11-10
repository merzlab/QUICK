#include "util.fh"   
module quick_io_module
  implicit none
contains

  subroutine read_quick_out(fname, natoms, tot_ene, grad)
    implicit none
    ! ---- arguments ----
    character(len=*), intent(in)  :: fname
    integer,          intent(in)  :: natoms
    double precision, intent(out) :: tot_ene
    double precision, allocatable, intent(out) :: grad(:,:)  ! (natoms,3)

    ! ---- locals ----
    integer :: u, ios, p
    integer :: in_grad, nvals, aidx, cidx
    character(len=256) :: line
    character(len=32)  :: lbl
    double precision :: v_xyz, v_grad

    tot_ene = 0.0d0
    in_grad = 0
    nvals   = 0

    if (natoms < 0) stop "read_quick_out: natoms must be >= 0"
    if (allocated(grad)) deallocate(grad)
    allocate(grad(3, max(natoms,0)))
    if (natoms > 0) grad = 0.0d0

    open(newunit=u, file=fname, status='old', action='read', iostat=ios)
    if (ios /= 0) stop "read_quick_out: cannot open file"

    do
      read(u,'(A)', iostat=ios) line
      if (ios /= 0) exit

      ! ---- TOTAL ENERGY ----
      if (index(line,'TOTAL ENERGY') > 0) then
        p = index(line,'=')
        if (p > 0) then
          read(line(p+1:),*,iostat=ios) tot_ene
          if (ios /= 0) then
            ios = 0
            read(line,*,iostat=ios) lbl, lbl, lbl, tot_ene
          end if
        end if
        cycle
      end if

      ! ---- Start of analytical gradient ----
      if (index(line,'ANALYTICAL GRADIENT:') > 0) then
        in_grad = 1
        cycle
      end if

      if (in_grad == 1) then
        ! Skip headers / separators
        if (index(line,'COORDINATE') > 0) cycle
        if (index(line,'---') > 0) then
          if (nvals >= 3*natoms) exit
          cycle
        end if
        if (len_trim(line) == 0) then
          if (nvals >= 3*natoms) exit
          cycle
        end if

        ! Data line: label, xyz, gradient
        read(line,*,iostat=ios) lbl, v_xyz, v_grad
        if (ios == 0) then
          if (nvals < 3*natoms) then
            nvals = nvals + 1
            aidx = (nvals - 1)/3 + 1        ! 1..natoms
            cidx = mod(nvals - 1, 3) + 1     ! 1..3
            grad(cidx, aidx) = v_grad
            if (nvals == 3*natoms) exit      ! stop immediately when full
          else
            exit
          end if
        end if
      end if
    end do

    close(u)

    ! Defensive: warn if file had fewer entries than expected
    if (nvals < 3*natoms) then
      write(*,*) 'read_quick_out: WARNING: expected ', 3*natoms, ' gradient values, got ', nvals
    end if
  end subroutine read_quick_out

end module quick_io_module

