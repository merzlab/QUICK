module quick_cuest_module
   !
   ! This module contains the fortran bindings for C functions that call cuEST
   !
   use, intrinsic::iso_c_binding, only: c_int64_t, c_ptr
   implicit none

   ! unless commented otherwise, C will not modify the memory a pointer points to

   interface
      subroutine cuest_init(natom, nshell, nauxshell, MAXPRIM, MAXPRIM_AUX, xyz, chg, nextatom, extxyz, extchg) &
                 bind(c, name="cuest_init")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_double, c_ptr
         implicit none
         integer(c_int64_t), intent(in), value :: natom
         integer(c_int64_t), intent(in), value :: nshell
         integer(c_int64_t), intent(in), value :: nauxshell
         integer(c_int64_t), intent(in), value :: MAXPRIM
         integer(c_int64_t), intent(in), value :: MAXPRIM_AUX
         type(c_ptr), intent(in), value :: xyz ! double
         real(c_double), intent(in) :: chg(*)
         integer(c_int64_t), intent(in), value :: nextatom
         real(c_double), intent(in) :: extxyz(*)
         real(c_double), intent(in) :: extchg(*)
      end subroutine cuest_init
   end interface

   interface
      subroutine cuest_deinit() bind(c, name="cuest_deinit")
      end subroutine
   end interface

   interface
      subroutine cuest_init_basis(ncenter, first_basis_function, last_basis_function, katom, &
                                  ktype, kprim, gcexpo, gccoeff, aux) &
                 bind(c, name="cuest_init_basis")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_double, c_bool
         implicit none
         integer(c_int64_t), intent(in) :: ncenter(*)
         integer(c_int64_t), intent(in) :: first_basis_function(*)
         integer(c_int64_t), intent(in) :: last_basis_function(*)
         integer(c_int64_t), intent(in) :: katom(*)
         integer(c_int64_t), intent(in) :: ktype(*)
         integer(c_int64_t), intent(in) :: kprim(*)
         real(c_double), intent(in) :: gcexpo(*)
         real(c_double), intent(in) :: gccoeff(*)
         logical(c_bool), intent(in), value :: aux
      end subroutine cuest_init_basis
   end interface

   interface
      subroutine cuest_init_oei_plan(cutoff) bind(c, name="cuest_init_oei_plan")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(in), value :: cutoff
      end subroutine cuest_init_oei_plan
   end interface

   interface
      subroutine cuest_get_oei_S(o) bind(c, name="cuest_get_oei_S")
         use, intrinsic::iso_c_binding, only: c_ptr
         type(c_ptr), intent(in), value :: o ! double; modified
      end subroutine cuest_get_oei_S
   end interface

   interface
      subroutine cuest_get_oei_T(o) bind(c, name="cuest_get_oei_T")
         use, intrinsic::iso_c_binding, only: c_ptr
         type(c_ptr), intent(in), value :: o ! double; modified
      end subroutine cuest_get_oei_T
   end interface

   interface
      subroutine cuest_get_oei_V(o) bind(c, name="cuest_get_oei_V")
         use, intrinsic::iso_c_binding, only: c_ptr
         type(c_ptr), intent(in), value :: o ! double; modified
      end subroutine cuest_get_oei_V
   end interface
end module quick_cuest_module
