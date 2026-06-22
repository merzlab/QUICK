module quick_cuest_module
    use, intrinsic::iso_c_binding, only: c_int64_t, c_double
    implicit none
    
    interface
        subroutine cuest_init_basis(natom, nshell, ncenter, first_basis_function, &
                                    last_basis_function, katom, ktype, kprim, gcexpo, gccoeff, xyz) &
                                    bind(c, name="cuest_init_basis")
            use, intrinsic::iso_c_binding, only: c_int64_t, c_double
            implicit none
            integer(c_int64_t), intent(in), value :: natom
            integer(c_int64_t), intent(in), value :: nshell
            integer(c_int64_t), intent(in) :: ncenter(*)
            integer(c_int64_t), intent(in) :: first_basis_function(*)
            integer(c_int64_t), intent(in) :: last_basis_function(*)
            integer(c_int64_t), intent(in) :: katom(*)
            integer(c_int64_t), intent(in) :: ktype(*)
            integer(c_int64_t), intent(in) :: kprim(*)
            real(kind=c_double), intent(in) :: gcexpo(*)
            real(kind=c_double), intent(in) :: gccoeff(*)
            real(kind=c_double), intent(in) :: xyz(*)
        end subroutine
    end interface

    ! call cuest_init_basis(int(natom_, c_int64_t), int(nshell_, c_int64_t), int(ncenter_, c_int64_t), &
    !                       int(first_basis_function_, c_int64_t), int(last_basis_function_, c_int64_t), &
    !                       int(katom_, c_int64_t), int(ktype_, c_int64_t), int(kprim_, c_int64_t), &
    !                       real(gcexpo_, c_double), real(gccoeff_, c_double), real(xyz_, c_double))
end module quick_cuest_module
