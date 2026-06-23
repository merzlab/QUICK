module quick_cuest_module
    use, intrinsic::iso_c_binding, only: c_int64_t, c_double
    implicit none
    
    interface
        subroutine cuest_init_basis(ncenter, first_basis_function, last_basis_function, katom, &
                                    ktype, kprim, gcexpo, gccoeff)                             &
                                    bind(c, name="cuest_init_basis")
            use, intrinsic::iso_c_binding, only: c_int64_t, c_double
            implicit none
            integer(c_int64_t), intent(in) :: ncenter(*)
            integer(c_int64_t), intent(in) :: first_basis_function(*)
            integer(c_int64_t), intent(in) :: last_basis_function(*)
            integer(c_int64_t), intent(in) :: katom(*)
            integer(c_int64_t), intent(in) :: ktype(*)
            integer(c_int64_t), intent(in) :: kprim(*)
            real(kind=c_double), intent(in) :: gcexpo(*)
            real(kind=c_double), intent(in) :: gccoeff(*)
        end subroutine cuest_init_basis
    end interface

    interface
        subroutine cuest_init(natom, nshell, MAXPRIM, xyz) bind(c, name="cuest_init")
            use, intrinsic::iso_c_binding, only: c_int64_t, c_double
            implicit none
            integer(c_int64_t), intent(in), value :: natom
            integer(c_int64_t), intent(in), value :: nshell
            integer(c_int64_t), intent(in), value :: MAXPRIM
            real(kind=c_double), intent(in) :: xyz(*)
        end subroutine cuest_init
    end interface 

    interface
        subroutine cuest_init_oei_plan() bind(c, name="cuest_init_oei_plan")
        end subroutine cuest_init_oei_plan
    end interface

    interface
        subroutine cuest_get_oei_S(o) bind(c, name="cuest_get_oei_S")
            use, intrinsic::iso_c_binding, only: c_double
            real(kind=c_double), intent(inout) :: o(*)
        end subroutine cuest_get_oei_S
    end interface

    interface
        subroutine cuest_get_oei_V(o) bind(c, name="cuest_get_oei_V")
            use, intrinsic::iso_c_binding, only: c_double
            real(kind=c_double), intent(inout) :: o(*)
        end subroutine cuest_get_oei_V
    end interface

    interface
        subroutine cuest_deinit() bind(c, name="cuest_deinit")
        end subroutine
    end interface
end module quick_cuest_module
