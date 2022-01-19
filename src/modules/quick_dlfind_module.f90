module quick_dlfind_module

    use dlf_parameter_module
    implicit none
    private
    public :: dlfind_interface

    contains

        subroutine dlfind_interface(ierr)

            use quick_molspec_module, only: natom, quick_molspec

            implicit none
            integer                :: nvarin ! number of variables to read in                      
            integer                :: nvarin2! number of variables to read in                      
            integer                :: nspec  ! number of values in the integer                     
            integer, intent(inout) :: ierr


            nspec = 3*quick_molspec%natom
            nvarin = 3*quick_molspec%natom
            nvarin2 = quick_molspec%natom

            call dl_find(nvarin, nvarin2, nspec, ierr, 1)


        end subroutine dlfind_interface

end module quick_dlfind_module
