!---------------------------------------------------------------------!
! Updated by Madu Manathunga on 06/09/2020                            !
!                                                                     !
! Previous contributors: Yipu Miao                                    !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!  Grid Points Module
module quick_gridpoints_module

! The gridpoints arrays are fairly simple: XANG, YANG, and ZANG hold
! the angular grid points location for a unit sphere, and WTANG
! is the weights of those points.  RGRID and RWEIGHT are the positions
! and weights of the radial grid points, which in use are sclaed by the
! radii and radii^3 of the atoms.

    use quick_size_module
    use quick_MPI_module
    implicit double precision(a-h,o-z)

    type quick_xc_grid_type

    !Binned grid point coordinates
    double precision,dimension(:), allocatable   :: gridxb

    double precision,dimension(:), allocatable   :: gridyb

    double precision,dimension(:), allocatable   :: gridzb

    !Binned sswt & weight
    double precision,dimension(:), allocatable   :: gridb_sswt

    double precision,dimension(:), allocatable   :: gridb_weight

    !Parent atom index
    integer,dimension(:), allocatable   :: gridb_atm

    !array of basis functions belonging to each bin
    integer,dimension(:), allocatable   :: basf

    !array of primitive functions beloning to binned basis functions
    integer,dimension(:), allocatable   :: primf

    !a counter to keep track of which basis functions belong to which bin
    integer,dimension(:), allocatable   :: basf_counter

    !a counter to keep track of which primitive functions belong to which basis
    !function
    integer,dimension(:), allocatable   :: primf_counter

#if defined CUDA || defined CUDA_MPIV
    !an array indicating bin of a grid point
    integer,dimension(:), allocatable   :: bin_locator
#endif
    !This array keeps track of the size of each bin
    integer,dimension(:), allocatable   :: bin_counter

    !length of binned grid arrays
    integer :: gridb_count

    !number of bins
    integer :: nbins

    !total number of basis functions
    integer :: nbtotbf

    !total number of primitive functions
    integer :: nbtotpf

    !save the number of initial grid pts for printing purposes
    integer :: init_ngpts

#ifdef MPIV
    integer, dimension(:), allocatable :: igridptul
    integer, dimension(:), allocatable :: igridptll
#endif
    end type quick_xc_grid_type


    type quick_xcg_tmp_type

    integer, dimension(:), allocatable :: init_grid_atm

    double precision,  dimension(:), allocatable :: init_grid_ptx

    double precision,  dimension(:), allocatable :: init_grid_pty

    double precision,  dimension(:), allocatable :: init_grid_ptz

    double precision,  dimension(:), allocatable :: arr_wtang

    double precision,  dimension(:), allocatable :: arr_rwt

    double precision,  dimension(:), allocatable :: arr_rad3

    double precision,  dimension(:), allocatable :: sswt

    double precision,  dimension(:), allocatable :: weight

#ifdef MPIV
    double precision,  dimension(:), allocatable :: tmp_sswt

    double precision, dimension(:), allocatable :: tmp_weight
#endif

    integer :: rad_gps = 50

    integer :: ang_gps = 194

    end type quick_xcg_tmp_type



    type(quick_xc_grid_type), save :: quick_dft_grid
    type(quick_xcg_tmp_type), save :: quick_xcg_tmp

    double precision ::  XANG(MAXANGGRID),YANG(MAXANGGRID), &
    ZANG(MAXANGGRID),WTANG(MAXANGGRID),RGRID(MAXRADGRID), &
    RWT(MAXRADGRID)
    double precision,  dimension(:), allocatable :: sigrad2
    integer :: iradial(0:10), iangular(10),iregion


    interface form_dft_grid
        module procedure form_xc_quadrature
    end interface form_dft_grid

    interface deform_dft_grid
        module procedure dealloc_grid_variables
    end interface deform_dft_grid

    interface print_grid_info
        module procedure print_grid_information
    end interface print_grid_info

    contains

    subroutine form_xc_quadrature(self, xcg_tmp)
    use quick_method_module
    use quick_molspec_module
    use quick_basis_module
    use quick_timer_module

    implicit double precision(a-h,o-z)
    type(quick_xc_grid_type) self
    type(quick_xcg_tmp_type) xcg_tmp
    double precision :: t_octree, t_prscrn

    !Form the quadrature and store coordinates and other information
    !Measure the time to form grid

    call alloc_xcg_tmp_variables(xcg_tmp)

#ifdef MPIV
  if(bMPI) then
    call alloc_mpi_grid_variables(self)
  endif

   if(master) then
#endif
    call cpu_time(timer_begin%TDFTGrdGen)

    idx_grid = 0
    do Iatm=1,natom
        if(quick_method%iSG.eq.1)then
            Iradtemp=50
        else
            if(quick_molspec%iattype(iatm).le.10)then
                Iradtemp=23
            else
                Iradtemp=26
            endif
        endif
        do Irad = 1, Iradtemp
            if(quick_method%iSG.eq.1)then
                call gridformnew(iatm,RGRID(Irad),iiangt)
                rad = radii(quick_molspec%iattype(iatm))
            else
                call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
                rad = radii2(quick_molspec%iattype(iatm))
            endif
            rad3 = rad*rad*rad
            do Iang=1,iiangt
                idx_grid=idx_grid+1
                xcg_tmp%init_grid_ptx(idx_grid)=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
                xcg_tmp%init_grid_pty(idx_grid)=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
                xcg_tmp%init_grid_ptz(idx_grid)=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)
                xcg_tmp%init_grid_atm(idx_grid)=Iatm
                xcg_tmp%arr_wtang(idx_grid) = WTANG(Iang)
                xcg_tmp%arr_rwt(idx_grid) = RWT(Irad)
                xcg_tmp%arr_rad3(idx_grid) = rad3
            enddo

        enddo
    enddo

    self%init_ngpts  = idx_grid

    call cpu_time(timer_end%TDFTGrdGen)

    timer_cumer%TDFTGrdGen = timer_cumer%TDFTGrdGen + timer_end%TDFTGrdGen - timer_begin%TDFTGrdGen

    !Measure time to compute grid weights
    call cpu_time(timer_begin%TDFTGrdWt)

#ifdef MPIV
   endif
#endif

    !Calculate the grid weights and store them
#if defined CUDA || defined CUDA_MPIV

    call gpu_get_ssw(xcg_tmp%init_grid_ptx, xcg_tmp%init_grid_pty, xcg_tmp%init_grid_ptz, &
    xcg_tmp%arr_wtang, xcg_tmp%arr_rwt, xcg_tmp%arr_rad3, &
    xcg_tmp%sswt, xcg_tmp%weight, xcg_tmp%init_grid_atm, self%init_ngpts)

#else

#if defined MPIV && !defined CUDA_MPIV

   if(bMPI) then

      call setup_ssw_mpi

      ist=self%igridptll(mpirank+1)
      iend=self%igridptul(mpirank+1)
   else
      ist=1
      iend = idx_grid
   endif

   do idx=ist, iend
#else
   do idx=1, idx_grid
#endif
        xcg_tmp%sswt(idx)=SSW(xcg_tmp%init_grid_ptx(idx), xcg_tmp%init_grid_pty(idx), xcg_tmp%init_grid_ptz(idx), &
        xcg_tmp%init_grid_atm(idx))
        xcg_tmp%weight(idx)=xcg_tmp%sswt(idx)*xcg_tmp%arr_wtang(idx)*xcg_tmp%arr_rwt(idx)*xcg_tmp%arr_rad3(idx)
    enddo

#if defined MPIV && !defined CUDA_MPIV
   if(bMPI) then
      call get_mpi_ssw
   endif

#endif

#endif

#ifdef MPIV
   if(master) then
#endif

    call cpu_time(timer_end%TDFTGrdWt)

    timer_cumer%TDFTGrdWt = timer_cumer%TDFTGrdWt + timer_end%TDFTGrdWt - timer_begin%TDFTGrdWt

    !Measure time to pack grid points
    call cpu_time(timer_begin%TDFTGrdPck)

#ifdef MPIV
   endif
#endif

    ! octree run and grid point packing are currently done using a single gpu
#ifdef CUDA_MPIV
   if(master) then
#endif

    ! initialize cpp data structure for octree and grid point packing
    call gpack_initialize()

    ! run octree, pack grid points and get the array sizes for f90 memory allocation
    call gpack_pack_pts(xcg_tmp%init_grid_ptx, xcg_tmp%init_grid_pty, xcg_tmp%init_grid_ptz, &
    xcg_tmp%init_grid_atm, xcg_tmp%sswt, xcg_tmp%weight, self%init_ngpts, natom, &
    nbasis, maxcontract, quick_method%DMCutoff, sigrad2, ncontract, aexp, dcoeff, quick_basis%ncenter, itype, xyz, &
    self%gridb_count, self%nbins, self%nbtotbf, self%nbtotpf, t_octree, t_prscrn)

    timer_cumer%TDFTGrdOct = timer_cumer%TDFTGrdOct + t_octree
    timer_cumer%TDFTPrscrn = timer_cumer%TDFTPrscrn + t_prscrn

#ifdef CUDA_MPIV
    endif
#endif

#ifdef MPIV

    if(master) then
#endif
!    write(*,*) "quick_grid_point_module: Total grid pts", self%gridb_count,"bin count:", self%nbins, "total bfs:", self%nbtotbf, &
!    "total pfs:", self%nbtotpf
#ifdef MPIV
    endif
#endif


#ifdef MPIV
   call setup_xc_mpi_1
#endif
   ! allocate f90 memory for pruned grid info from cpp side
   call alloc_grid_variables(self)

#ifdef MPIV
    if(master) then
#endif

#if defined CUDA || defined CUDA_MPIV

#ifdef CUDA_MPIV
   if(master) then
#endif

    ! save packed grid information into f90 data structures
     call get_gpu_grid_info(self%gridxb, self%gridyb, self%gridzb, self%gridb_sswt, self%gridb_weight, self%gridb_atm, &
     self%bin_locator, self%basf, self%primf, self%basf_counter, self%primf_counter, self%bin_counter)

#ifdef CUDA_MPIV
    endif
#endif

#else
    ! save packed grid information into f90 data structures
    call get_cpu_grid_info(self%gridxb, self%gridyb, self%gridzb, self%gridb_sswt, self%gridb_weight, self%gridb_atm, &
    self%basf, self%primf, self%basf_counter, self%primf_counter, self%bin_counter)

#endif

#ifdef MPIV
    endif
    if(bMPI) then
       call setup_xc_mpi_new_imp
    endif
#endif

#ifndef CUDA
!    do i=1, self%nbins
!        nid = self%bin_counter(i+1)-self%bin_counter(i)

!        write(*,*) "test_bin_counter_array:",nid, self%bin_counter(i)

!            j = self%bin_counter(i)+1

!            do while(j < self%bin_counter(i+1)+1)
!                write(*,*) "test cpu arrays:", i, nid, self%gridxb(j), self%gridyb(j), self%gridzb(j), self%gridb_sswt(j), &
!                self%gridb_weight(j), self%gridb_atm(j)
!                j = j+1
!            enddo
!    enddo
#endif


!    do i=1, self%nbins
!        nid=self%basf_counter(i+1)-self%basf_counter(i)
!            j=self%basf_counter(i)+1
!            do while (j<self%basf_counter(i+1)+1)
!                write(*,*) "test_dft_f90_bf i:",i,nid,self%basf_counter(i),self%basf_counter(i+1),j,self%basf(j)
!                k=self%primf_counter(j)+1
!                do while(k<self%primf_counter(j+1)+1)
!                    write(*,*) "test_dft_f90 i:",i,nid,self%basf_counter(i)+1,self%basf_counter(i+1)+1,j,self%basf(j), &
!                    k, self%primf(k)
!                    k=k+1
!                end do
!                j=j+1
!            end do
!    enddo

    ! relinquish memory allocated for octree and grid point packing
#ifdef CUDA_MPIV
   if(master) then
#endif
    call gpack_finalize()
#ifdef CUDA_MPIV
    endif
#endif

    ! relinquish memory allocated for temporary f90 variables
    call dealloc_xcg_tmp_variables(xcg_tmp)

#ifdef MPIV
    if(master) then
#endif

    call cpu_time(timer_end%TDFTGrdPck)

    timer_cumer%TDFTGrdPck = timer_cumer%TDFTGrdPck + timer_end%TDFTGrdPck - timer_begin%TDFTGrdPck - t_octree &
                           - t_prscrn

!    write(*,*) "DFT grid timings: Grid form:", timer_cumer%TDFTGrdGen, "Compute grid weights:", timer_cumer%TDFTGrdWt, &
!    "Octree:",timer_cumer%TDFTGrdOct,"Prescreening:",timer_cumer%TDFTPrscrn, "Pack points:",timer_cumer%TDFTGrdPck


#ifdef MPIV
    endif
#endif

    end subroutine

    ! allocate gridpoints
    subroutine allocate_quick_gridpoints(nbasis)
        implicit double precision(a-h,o-z)
        integer nbasis
        if (.not. allocated(sigrad2)) allocate(sigrad2(nbasis))
    end subroutine allocate_quick_gridpoints

    ! deallocate
    subroutine deallocate_quick_gridpoints
        implicit double precision(a-h,o-z)
        if (allocated(sigrad2)) deallocate(sigrad2)
    end subroutine deallocate_quick_gridpoints

    ! Allocate memory for dft grid variables
    subroutine alloc_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        if (.not. allocated(self%gridxb)) allocate(self%gridxb(self%gridb_count))
        if (.not. allocated(self%gridyb)) allocate(self%gridyb(self%gridb_count))
        if (.not. allocated(self%gridzb)) allocate(self%gridzb(self%gridb_count))
        if (.not. allocated(self%gridb_sswt)) allocate(self%gridb_sswt(self%gridb_count))
        if (.not. allocated(self%gridb_weight)) allocate(self%gridb_weight(self%gridb_count))
        if (.not. allocated(self%gridb_atm)) allocate(self%gridb_atm(self%gridb_count))
        if (.not. allocated(self%basf)) allocate(self%basf(self%nbtotbf))
        if (.not. allocated(self%primf)) allocate(self%primf(self%nbtotpf))
        if (.not. allocated(self%basf_counter)) allocate(self%basf_counter(self%nbins + 1))
        if (.not. allocated(self%primf_counter)) allocate(self%primf_counter(self%nbtotbf + 1))

#if defined CUDA || defined CUDA_MPIV
        if (.not. allocated(self%bin_locator)) allocate(self%bin_locator(self%gridb_count))
#endif
        if (.not. allocated(self%bin_counter)) allocate(self%bin_counter(self%nbins+1))

    end subroutine

    subroutine alloc_xcg_tmp_variables(xcg_tmp)
        use quick_molspec_module
        implicit none
        type(quick_xcg_tmp_type) xcg_tmp
        integer :: tot_gps

        tot_gps = natom*xcg_tmp%rad_gps*xcg_tmp%ang_gps

        if (.not. allocated(xcg_tmp%init_grid_atm)) allocate(xcg_tmp%init_grid_atm(tot_gps))
        if (.not. allocated(xcg_tmp%init_grid_ptx)) allocate(xcg_tmp%init_grid_ptx(tot_gps))
        if (.not. allocated(xcg_tmp%init_grid_pty)) allocate(xcg_tmp%init_grid_pty(tot_gps))
        if (.not. allocated(xcg_tmp%init_grid_ptz)) allocate(xcg_tmp%init_grid_ptz(tot_gps))
        if (.not. allocated(xcg_tmp%arr_wtang)) allocate(xcg_tmp%arr_wtang(tot_gps))
        if (.not. allocated(xcg_tmp%arr_rwt)) allocate(xcg_tmp%arr_rwt(tot_gps))
        if (.not. allocated(xcg_tmp%arr_rad3)) allocate(xcg_tmp%arr_rad3(tot_gps))
        if (.not. allocated(xcg_tmp%sswt)) allocate(xcg_tmp%sswt(tot_gps))
        if (.not. allocated(xcg_tmp%weight)) allocate(xcg_tmp%weight(tot_gps))
#ifdef MPIV
        if (.not. allocated(xcg_tmp%tmp_sswt)) allocate(xcg_tmp%tmp_sswt(tot_gps))
        if (.not. allocated(xcg_tmp%tmp_weight)) allocate(xcg_tmp%tmp_weight(tot_gps))
#endif
    end subroutine

#ifdef MPIV
    subroutine alloc_mpi_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        if (.not. allocated(self%igridptul)) allocate(self%igridptul(mpisize))
        if (.not. allocated(self%igridptll)) allocate(self%igridptll(mpisize))

   end subroutine
#endif

    ! Deallocate memory reserved for dft grid variables
    subroutine dealloc_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        if (allocated(self%gridxb)) deallocate(self%gridxb)
        if (allocated(self%gridyb)) deallocate(self%gridyb)
        if (allocated(self%gridzb)) deallocate(self%gridzb)
        if (allocated(self%gridb_sswt)) deallocate(self%gridb_sswt)
        if (allocated(self%gridb_weight)) deallocate(self%gridb_weight)
        if (allocated(self%gridb_atm)) deallocate(self%gridb_atm)
        if (allocated(self%basf)) deallocate(self%basf)
        if (allocated(self%primf)) deallocate(self%primf)
        if (allocated(self%basf_counter)) deallocate(self%basf_counter)
        if (allocated(self%primf_counter)) deallocate(self%primf_counter)
#if defined CUDA || defined CUDA_MPIV
        if (allocated(self%bin_locator)) deallocate(self%bin_locator)
#endif
        if (allocated(self%bin_counter)) deallocate(self%bin_counter)

#ifdef MPIV
        if(bMPI) then
                call dealloc_mpi_grid_variables(self)
        endif
#endif
    end subroutine

    subroutine dealloc_xcg_tmp_variables(xcg_tmp)
        use quick_molspec_module
        implicit none
        type(quick_xcg_tmp_type) xcg_tmp

        if (allocated(xcg_tmp%init_grid_atm)) deallocate(xcg_tmp%init_grid_atm)
        if (allocated(xcg_tmp%init_grid_ptx)) deallocate(xcg_tmp%init_grid_ptx)
        if (allocated(xcg_tmp%init_grid_pty)) deallocate(xcg_tmp%init_grid_pty)
        if (allocated(xcg_tmp%init_grid_ptz)) deallocate(xcg_tmp%init_grid_ptz)
        if (allocated(xcg_tmp%arr_wtang)) deallocate(xcg_tmp%arr_wtang)
        if (allocated(xcg_tmp%arr_rwt)) deallocate(xcg_tmp%arr_rwt)
        if (allocated(xcg_tmp%arr_rad3)) deallocate(xcg_tmp%arr_rad3)
        if (allocated(xcg_tmp%sswt)) deallocate(xcg_tmp%sswt)
        if (allocated(xcg_tmp%weight)) deallocate(xcg_tmp%weight)
#ifdef MPIV
        if (allocated(xcg_tmp%tmp_sswt)) deallocate(xcg_tmp%tmp_sswt)
        if (allocated(xcg_tmp%tmp_weight)) deallocate(xcg_tmp%tmp_weight)
#endif
    end subroutine

#ifdef MPIV
    subroutine dealloc_mpi_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        if (allocated(self%igridptul)) deallocate(self%igridptul)
        if (allocated(self%igridptll)) deallocate(self%igridptll)
   end subroutine
#endif

   subroutine print_grid_information(self)
     use quick_files_module
     use quick_method_module
     use quick_molspec_module
     use quick_basis_module
     implicit none
     type(quick_xc_grid_type) self

     write (ioutfile,'(" OCTAGO: OCTree Algorithm for Grid Operations ")')
     write (ioutfile,'("   PRUNING CUTOFF       =",E10.3)') quick_method%DMCutoff
     write (ioutfile,'("   INITIAL GRID POINTS  =",I12)') self%init_ngpts
     write (ioutfile,'("|   FINAL GRID POINTS    =",I12)') self%gridb_count
     write (ioutfile,'("|   SIGNIFICANT NUMBER OF BASIS FUNCTIONS     =",I12)') self%nbtotbf
     write (ioutfile,'("|   SIGNIFICANT NUMBER OF PRIMITIVE FUNCTIONS =",I12)') self%nbtotpf

   end subroutine print_grid_information

end module quick_gridpoints_module
