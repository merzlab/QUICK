module quick_cuest_module
   !
   ! This module contains the fortran bindings for C functions that call cuEST
   !
   use, intrinsic::iso_c_binding, only: c_int64_t, c_ptr
   implicit none

   ! unless commented otherwise, C will not modify the memory a pointer points to

   interface
      subroutine cuest_init(natom, nshell, nbasis, nauxshell, maxcontract, maxcontract_aux, xyz, chg, nextatom, extxyz, extchg) &
         bind(c, name="cuest_init")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_double, c_ptr
         implicit none
         integer(c_int64_t), intent(in), value :: natom
         integer(c_int64_t), intent(in), value :: nshell
         integer(c_int64_t), intent(in), value :: nbasis
         integer(c_int64_t), intent(in), value :: nauxshell
         integer(c_int64_t), intent(in), value :: maxcontract
         integer(c_int64_t), intent(in), value :: maxcontract_aux
         type(c_ptr), intent(in), value :: xyz ! double
         real(c_double), intent(in) :: chg(*)
         integer(c_int64_t), intent(in), value :: nextatom
         real(c_double), intent(in) :: extxyz(*)
         real(c_double), intent(in) :: extchg(*)
      end subroutine cuest_init
   end interface

   interface
      subroutine cuest_deinit_oei_plan() bind(c, name="cuest_deinit_oei_plan")
      end subroutine
   end interface

   interface
      subroutine cuest_deinit() bind(c, name="cuest_deinit")
      end subroutine
   end interface

   interface
      subroutine cuest_init_basis(ncenter, katom, ktype, kprim, gcexpo, gccoeff, aux) &
         bind(c, name="cuest_init_basis")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_double, c_bool
         implicit none
         integer(c_int64_t), intent(in) :: ncenter(*)
         integer(c_int64_t), intent(in) :: katom(*)
         integer(c_int64_t), intent(in) :: ktype(*)
         integer(c_int64_t), intent(in) :: kprim(*)
         real(c_double), intent(in) :: gcexpo(*)
         real(c_double), intent(in) :: gccoeff(*)
         logical(c_bool), intent(in), value :: aux
      end subroutine cuest_init_basis
   end interface

   interface
      subroutine cuest_init_oei_plan() bind(c, name="cuest_init_oei_plan")
      end subroutine cuest_init_oei_plan
   end interface

   interface
      subroutine cuest_init_pair_list(cutoff) bind(c, name="cuest_init_pair_list")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(in), value :: cutoff
      end subroutine cuest_init_pair_list
   end interface

   interface
      subroutine cuest_init_dfint_plan() bind(c, name="cuest_init_dfint_plan")
      end subroutine cuest_init_dfint_plan
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

   interface
      subroutine cuest_get_eri_J(o, dense) bind(c, name="cuest_get_eri_J")
         use, intrinsic::iso_c_binding, only: c_ptr, c_double
         type(c_ptr), intent(in), value :: o ! double; modified
         real(c_double), intent(in) :: dense(*)
      end subroutine cuest_get_eri_J
   end interface

   interface
      subroutine cuest_get_eri_K(o, C, nocc) bind(c, name="cuest_get_eri_K")
         use, intrinsic::iso_c_binding, only: c_ptr, c_double, c_int64_t
         type(c_ptr), intent(in), value :: o ! double; modified
         real(c_double), intent(in) :: C(*)
         integer(c_int64_t), intent(in), value :: nocc
      end subroutine cuest_get_eri_K
   end interface
end module quick_cuest_module
