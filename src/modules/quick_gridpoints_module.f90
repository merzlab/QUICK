#include "../config.h"
!	quick_gridpoints_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!       
!       Madu Manathunga updated this module on 04/17/2017
!
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

#ifdef MPIV
!   include "mpif.h"
#endif

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

    !an array indicating if a binned grid point is true or a dummy grid point
    integer,dimension(:), allocatable   :: dweight

    !in cpu case, we will have bins with different number of points. This array keeps track
    !of the size of each bin
    integer,dimension(:), allocatable   :: bin_counter

    !length of binned grid arrays
    integer :: gridb_count

    !number of bins 
    integer :: nbins

    !total number of basis functions
    integer :: nbtotbf

    !total number of primitive functions
    integer :: nbtotpf

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

    integer :: idx_grid = 0

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

    contains

    subroutine form_xc_quadrature(self, xcg_tmp)
    use quick_method_module
    use quick_molspec_module
    use quick_basis_module    
    use quick_timer_module

    implicit double precision(a-h,o-z) 
    type(quick_xc_grid_type) self
    type(quick_xcg_tmp_type) xcg_tmp
    !Form the quadrature and store coordinates and other information
    !Measure the time to form grid

    call alloc_xcg_tmp_variables(xcg_tmp)    

#ifdef MPIV
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

    xcg_tmp%idx_grid = idx_grid    

    call cpu_time(timer_end%TDFTGrdGen)

    timer_cumer%TDFTGrdGen = timer_end%TDFTGrdGen - timer_begin%TDFTGrdGen

    !Measure time to compute grid weights
    call cpu_time(timer_begin%TDFTGrdWt)

#ifdef MPIV
   endif
#endif

    !Calculate the grid weights and store them
#ifdef CUDA

    call gpu_get_ssw(xcg_tmp%init_grid_ptx, xcg_tmp%init_grid_pty, xcg_tmp%init_grid_ptz, &
    xcg_tmp%arr_wtang, xcg_tmp%arr_rwt, xcg_tmp%arr_rad3, &
    xcg_tmp%sswt, xcg_tmp%weight, xcg_tmp%init_grid_atm, xcg_tmp%idx_grid)

#else

#ifdef MPIV

   if(bMPI) then

      call alloc_mpi_grid_variables(self)

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

#ifdef MPIV
   if(bMPI) then
      call get_mpi_ssw 
   endif

#endif

#endif

#ifdef MPIV
   if(master) then
#endif

    call cpu_time(timer_end%TDFTGrdWt)

    timer_cumer%TDFTGrdWt = timer_end%TDFTGrdWt - timer_begin%TDFTGrdWt

    !Measure time to pack grid points
    call cpu_time(timer_begin%TDFTGrdPck)

#ifdef MPIV
   endif
#endif
    call pack_grid_pts_f90(xcg_tmp%init_grid_ptx, xcg_tmp%init_grid_pty, xcg_tmp%init_grid_ptz, &
    xcg_tmp%init_grid_atm, xcg_tmp%sswt, xcg_tmp%weight, xcg_tmp%idx_grid, &
    nbasis, maxcontract, quick_method%DMCutoff, sigrad2, ncontract, aexp, dcoeff, quick_basis%ncenter, itype, xyz, & 
    self%gridb_count, self%nbins, self%nbtotbf, self%nbtotpf, timer_cumer%TDFTGrdOct, timer_cumer%TDFTPrscrn) 

#ifdef MPIV

    if(master) then
#endif
    write(*,*) "quick_grid_point_module: Total grid pts", self%gridb_count,"bin count:", self%nbins, "total bfs:", self%nbtotbf, &
    "total pfs:", self%nbtotpf
#ifdef MPIV
    endif
#endif


#ifdef MPIV
   call setup_xc_mpi_1
#endif

   call alloc_grid_variables(self)

#ifdef MPIV
    if(master) then
#endif
    call save_dft_grid_info(self%gridxb, self%gridyb, self%gridzb, self%gridb_sswt, self%gridb_weight, self%gridb_atm, &
    self%dweight, self%basf, self%primf, self%basf_counter, self%primf_counter, self%bin_counter)

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

    call dealloc_xcg_tmp_variables(xcg_tmp)

#ifdef MPIV
    if(master) then
#endif

    call cpu_time(timer_end%TDFTGrdPck)

    timer_cumer%TDFTGrdPck = timer_end%TDFTGrdPck - timer_begin%TDFTGrdPck - timer_cumer%TDFTGrdOct - timer_cumer%TDFTPrscrn

    write(*,*) "DFT grid timings: Grid form:", timer_cumer%TDFTGrdGen, "Compute grid weights:", timer_cumer%TDFTGrdWt, &
    "Octree:",timer_cumer%TDFTGrdOct,"Prescreening:",timer_cumer%TDFTPrscrn, "Pack points:",timer_cumer%TDFTGrdPck

#ifdef MPIV
    endif
#endif

    end subroutine    

    ! allocate gridpoints
    subroutine allocate_quick_gridpoints(nbasis)
        implicit double precision(a-h,o-z)
        integer nbasis
        allocate(sigrad2(nbasis))
    end subroutine allocate_quick_gridpoints

    ! deallocate
    subroutine deallocate_quick_gridpoints
        implicit double precision(a-h,o-z)
        deallocate(sigrad2)
    end subroutine deallocate_quick_gridpoints

    ! Allocate memory for dft grid variables    
    subroutine alloc_grid_variables(self)
        use quick_MPI_module 
        implicit none
        type(quick_xc_grid_type) self

        allocate(self%gridxb(self%gridb_count))
        allocate(self%gridyb(self%gridb_count))
        allocate(self%gridzb(self%gridb_count))
        allocate(self%gridb_sswt(self%gridb_count))
        allocate(self%gridb_weight(self%gridb_count))
        allocate(self%gridb_atm(self%gridb_count))
        allocate(self%dweight(self%gridb_count))
        allocate(self%basf(self%nbtotbf))
        allocate(self%primf(self%nbtotpf))
        allocate(self%basf_counter(self%nbins + 1))
        allocate(self%primf_counter(self%nbtotbf + 1))
        allocate(self%bin_counter(self%nbins+1))
    end subroutine

    subroutine alloc_xcg_tmp_variables(xcg_tmp)
        use quick_molspec_module
        implicit none
        type(quick_xcg_tmp_type) xcg_tmp
        integer :: tot_gps
        
        tot_gps = natom*xcg_tmp%rad_gps*xcg_tmp%ang_gps

        allocate(xcg_tmp%init_grid_atm(tot_gps))
        allocate(xcg_tmp%init_grid_ptx(tot_gps))
        allocate(xcg_tmp%init_grid_pty(tot_gps))
        allocate(xcg_tmp%init_grid_ptz(tot_gps))
        allocate(xcg_tmp%arr_wtang(tot_gps))
        allocate(xcg_tmp%arr_rwt(tot_gps))
        allocate(xcg_tmp%arr_rad3(tot_gps))
        allocate(xcg_tmp%sswt(tot_gps))
        allocate(xcg_tmp%weight(tot_gps))
#ifdef MPIV
        allocate(xcg_tmp%tmp_sswt(tot_gps))
        allocate(xcg_tmp%tmp_weight(tot_gps))        
#endif
    end subroutine

#ifdef MPIV
    subroutine alloc_mpi_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        allocate(self%igridptul(mpisize))
        allocate(self%igridptll(mpisize))

   end subroutine
#endif    

    ! Deallocate memory reserved for dft grid variables
    subroutine dealloc_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        deallocate(self%gridxb)
        deallocate(self%gridyb)
        deallocate(self%gridzb)
        deallocate(self%gridb_sswt)
        deallocate(self%gridb_weight)
        deallocate(self%gridb_atm)
        deallocate(self%dweight)
        deallocate(self%basf)        
        deallocate(self%primf)
        deallocate(self%basf_counter)
        deallocate(self%primf_counter)
        deallocate(self%bin_counter)
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

        deallocate(xcg_tmp%init_grid_atm)
        deallocate(xcg_tmp%init_grid_ptx)
        deallocate(xcg_tmp%init_grid_pty)
        deallocate(xcg_tmp%init_grid_ptz)
        deallocate(xcg_tmp%arr_wtang)
        deallocate(xcg_tmp%arr_rwt)
        deallocate(xcg_tmp%arr_rad3)
        deallocate(xcg_tmp%sswt)
        deallocate(xcg_tmp%weight)
#ifdef MPIV
        deallocate(xcg_tmp%tmp_sswt)
        deallocate(xcg_tmp%tmp_weight)
#endif
    end subroutine

#ifdef MPIV
    subroutine dealloc_mpi_grid_variables(self)
        use quick_MPI_module
        implicit none
        type(quick_xc_grid_type) self

        deallocate(self%igridptul)
        deallocate(self%igridptll)
   end subroutine
#endif

end module quick_gridpoints_module
