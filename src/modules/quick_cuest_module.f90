#if defined(CUDA) && defined(CUEST)

module quick_cuest_module
   !
   ! This module contains the fortran bindings for C functions that call cuEST
   !
   use, intrinsic::iso_c_binding, only: c_int8_t
   implicit none

   integer(c_int8_t), protected :: CUEST_CORRECT_REORDER = 1
   integer(c_int8_t), protected :: CUEST_CORRECT_NORM_CUEST_TO_QUICK = 2
   integer(c_int8_t), protected :: CUEST_CORRECT_NORM_QUICK_TO_CUEST = 6
   integer(c_int8_t), protected :: CUEST_CORRECT_NORM_INV = 8
   integer(c_int8_t), protected :: CUEST_CORRECT_REORDER_AND_NORM_CUEST_TO_QUICK = 3
   integer(c_int8_t), protected :: CUEST_CORRECT_REORDER_AND_NORM_QUICK_TO_CUEST = 7

   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_HF     = 0
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_B3LYP  = 1
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_B97    = 2
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_BLYP   = 3
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_M06L   = 4 ! QUICK unsupported
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_PBE    = 5
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_PBE0   = 6
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_R2SCAN = 7 ! QUICK unsupported
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_SVWN5  = 8 ! QUICK unsupported
   integer(c_int8_t), protected :: CUEST_FUNCTIONAL_B97MV  = 9 ! QUICK unsupported

   ! unless commented otherwise, C will not modify the memory a pointer points to

   interface
      subroutine cuest_init(natom, nshell, nbasis, nocc, nauxshell, maxcontract, maxcontract_aux, &
                            iattype, xyz, chg, nextatom, extxyz, extchg) &
         bind(c, name="cuest_init")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_int8_t, c_double, c_ptr
         implicit none
         integer(c_int64_t), intent(in), value :: natom
         integer(c_int64_t), intent(in), value :: nshell
         integer(c_int64_t), intent(in), value :: nbasis
         integer(c_int64_t), intent(in), value :: nocc
         integer(c_int64_t), intent(in), value :: nauxshell
         integer(c_int64_t), intent(in), value :: maxcontract
         integer(c_int64_t), intent(in), value :: maxcontract_aux
         integer(c_int8_t), intent(in) :: iattype(*)
         type(c_ptr), intent(in), value :: xyz ! double
         real(c_double), intent(in) :: chg(*)
         integer(c_int64_t), intent(in), value :: nextatom
         real(c_double), intent(in) :: extxyz(*)
         real(c_double), intent(in) :: extchg(*)
      end subroutine cuest_init
   end interface

   interface
      subroutine cuest_init_correct() bind (c, name="cuest_init_correct")
      end subroutine
   end interface

   interface
      subroutine cuest_deinit_correct() bind (c, name="cuest_deinit_correct")
      end subroutine
   end interface

   interface
      subroutine cuest_correct_o(o, qspec) bind(c, name="cuest_correct_o")
         use, intrinsic :: iso_c_binding, only: c_double, c_int8_t
         implicit none
         real(c_double), intent(inout) :: o(*)
         integer(c_int8_t), intent(in), value :: qspec
      end subroutine
   end interface

   interface
      subroutine cuest_correct_P(o, qspec) bind(c, name="cuest_correct_P")
         use, intrinsic :: iso_c_binding, only: c_double, c_int8_t
         implicit none
         real(c_double), intent(inout) :: o(*)
         integer(c_int8_t), intent(in), value :: qspec
      end subroutine
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
      subroutine cuest_init_dfint_plan(hyb_coeff) bind(c, name="cuest_init_dfint_plan")
         use, intrinsic :: iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(in), value :: hyb_coeff
      end subroutine cuest_init_dfint_plan
   end interface

   interface
      subroutine cuest_get_oei_S(o) bind(c, name="cuest_get_oei_S")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(out) :: o(*)
      end subroutine cuest_get_oei_S
   end interface

   interface
      subroutine cuest_get_oei_T(o) bind(c, name="cuest_get_oei_T")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(out) :: o(*)
      end subroutine cuest_get_oei_T
   end interface

   interface
      subroutine cuest_get_oei_V(o) bind(c, name="cuest_get_oei_V")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(out) :: o(*)
      end subroutine cuest_get_oei_V
   end interface

   interface
      subroutine cuest_get_eri_J(o, dense) bind(c, name="cuest_get_eri_J")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(out) :: o(*)
         real(c_double), intent(in) :: dense(*)
      end subroutine cuest_get_eri_J
   end interface

   interface
      subroutine cuest_get_eri_K(o, C) bind(c, name="cuest_get_eri_K")
         use, intrinsic::iso_c_binding, only: c_double, c_int64_t
         implicit none
         real(c_double), intent(out) :: o(*)
         real(c_double), intent(in) :: C(*)
      end subroutine cuest_get_eri_K
   end interface

   interface
      subroutine cuest_create_atom_grid_setup() bind(c, name="cuest_create_atom_grid_setup")
      end subroutine cuest_create_atom_grid_setup
   end interface

   interface
      subroutine cuest_create_atom_grid (nrad, r, w, nang) bind(c, name="cuest_create_atom_grid")
         use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
         implicit none
         integer(c_int64_t), intent(in), value :: nrad
         real(c_double), intent(in) :: r(*)
         real(c_double), intent(in) :: w(*)
         integer(c_int64_t), intent(in) :: nang(*)
      end subroutine cuest_create_atom_grid
   end interface

   interface
      subroutine cuest_destroy_atom_grid() bind(c, name="cuest_destroy_atom_grid")
      end subroutine cuest_destroy_atom_grid
   end interface

   interface
      subroutine cuest_init_xc (fnl) bind(c, name="cuest_init_xc")
         use, intrinsic :: iso_c_binding, only: c_int8_t
         implicit none
         integer(c_int8_t), intent(in), value :: fnl
      end subroutine
   end interface

   interface
      subroutine cuest_get_Vxc (Vxc, Exc, C) bind(c, name="cuest_get_Vxc")
         use, intrinsic :: iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(out) :: Vxc(*)
         real(c_double), intent(out) :: Exc
         real(c_double), intent(in) :: C(*)
      end subroutine
   end interface

   interface
      subroutine cuest_debuglog_inner(str) bind(c, name="cuest_debuglog")
         use, intrinsic :: iso_c_binding, only: c_char
         implicit none
         character(kind=c_char), intent(in) :: str(*)
      end subroutine cuest_debuglog_inner
   end interface

   interface
      subroutine cuest_get_memtrace(hostmax, hosttotal, hostallocs, devmax, devtotal, devallocs) bind(c, name="cuest_get_memtrace")
         use, intrinsic :: iso_c_binding, only: c_int64_t
         implicit none
         integer(c_int64_t), intent(out) :: hostmax
         integer(c_int64_t), intent(out) :: hosttotal
         integer(c_int64_t), intent(out) :: hostallocs
         integer(c_int64_t), intent(out) :: devmax
         integer(c_int64_t), intent(out) :: devtotal
         integer(c_int64_t), intent(out) :: devallocs
      end subroutine cuest_get_memtrace
   end interface

contains

   subroutine cuest_debuglog(str)
      use, intrinsic :: iso_c_binding, only: c_null_char
      implicit none
      character(len=*), intent(in) :: str

      call cuest_debuglog_inner(str // c_null_char)
   end subroutine cuest_debuglog

   subroutine cuest_print_memtrace(io)
      use, intrinsic :: iso_c_binding, only: c_int64_t
      implicit none
      integer, intent(in) :: io
      integer(c_int64_t) :: hostmax, hosttotal, hostallocs, devmax, devtotal, devallocs

      call cuest_get_memtrace(hostmax, hosttotal, hostallocs, devmax, devtotal, devallocs)

      call PrtAct(iOutFile, "Output cuEST Memory Footprint")
      write(io, *) "| Maximum Host Footprint:   ", hostmax
      write(io, *) "| Maximum Host Footprint:   ", hostmax
      write(io, *) "| Total Host Footprint:     ", hosttotal
      write(io, *) "| Host Allocations:         ", hostallocs
      write(io, *) "| Maximum Device Footprint: ", devmax
      write(io, *) "| Total Device Footprint:   ", devtotal
      write(io, *) "| Device Allocations:       ", devallocs
      call PrtAct(iOutFile, "Finish Output cuEST Memory Footprint")
   end subroutine cuest_print_memtrace

end module quick_cuest_module

#endif
