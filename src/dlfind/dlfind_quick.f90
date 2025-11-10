module dlfind_quick
    implicit none 
    private
    public::quick_opt
contains
subroutine quick_opt
    use dlf_global 
    implicit none
    integer :: nvar
    integer :: nvar2
    integer :: nspec
    integer :: master
    integer :: natm
    integer :: nz
!    integer :: i, j
!    character, dimension(natm) ::  atyp
!    real(rk) :: coords(nvar)
   
    print*,"QUICK Geometry Optimization with  DL-Find"

    natm = 3
    nz=natm
    nvar = 3*natm
    nvar2 = natm
    nspec = 2*natm+nz

    master = 1
    call dl_find(nvar,nvar2, nspec, master) 

end subroutine quick_opt

end module dlfind_quick
