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

#include "util.fh"

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

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
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
   if(master) then
#endif
   
   if (quick_method%iSG.eq.1) call gridformSG1() 

#ifdef MPIV
  endif

   call alloc_mpi_grid_variables(self)

   call mpi_bcast_grid_vars()

   if(master) then
#endif
    RECORD_TIME(timer_begin%TDFTGrdGen)

    ! form SG1 grid
    !if(quick_method%iSG.eq.1) call gridformSG1()

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

    RECORD_TIME(timer_end%TDFTGrdGen)

    timer_cumer%TDFTGrdGen = timer_cumer%TDFTGrdGen + timer_end%TDFTGrdGen - timer_begin%TDFTGrdGen

    !Measure time to compute grid weights
    RECORD_TIME(timer_begin%TDFTGrdWt)

#ifdef MPIV
   endif
#endif

    ! allocate memory for data structures holding radius of significance, phi,
    ! dphi and etc. 
    call allocate_sigrad_phi()

    ! compute the radius of significance for basis functions on each center
    call get_sigrad()

    !Calculate the grid weights and store them
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

    call gpu_get_ssw(xcg_tmp%init_grid_ptx, xcg_tmp%init_grid_pty, xcg_tmp%init_grid_ptz, &
    xcg_tmp%arr_wtang, xcg_tmp%arr_rwt, xcg_tmp%arr_rad3, &
    xcg_tmp%sswt, xcg_tmp%weight, xcg_tmp%init_grid_atm, self%init_ngpts)

#else

#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV

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

#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
   if(bMPI) then
      call get_mpi_ssw
   endif

#endif

#endif

#ifdef MPIV
   if(master) then
#endif

    RECORD_TIME(timer_end%TDFTGrdWt)

    timer_cumer%TDFTGrdWt = timer_cumer%TDFTGrdWt + timer_end%TDFTGrdWt - timer_begin%TDFTGrdWt

    !Measure time to pack grid points
    RECORD_TIME(timer_begin%TDFTGrdPck)

#ifdef MPIV
   endif
#endif

    ! octree run and grid point packing are currently done using a single gpu
#if defined CUDA_MPIV || defined HIP_MPIV
   if(master) then
#endif

      
    ! initialize cpp data structure for octree and grid point packing
    call gpack_initialize()


    
    ! run octree, pack grid points and get the array sizes for f90 memory allocation
    call gpack_pack_pts(xcg_tmp%init_grid_ptx, xcg_tmp%init_grid_pty, xcg_tmp%init_grid_ptz, &
    xcg_tmp%init_grid_atm, xcg_tmp%sswt, xcg_tmp%weight, self%init_ngpts, natom, &
    nbasis, maxcontract, quick_method%DMCutoff, quick_method%XCCutoff, sigrad2, ncontract, &
    aexp, dcoeff, quick_basis%ncenter, itype, xyz, &
    self%gridb_count, self%nbins, self%nbtotbf, self%nbtotpf, t_octree, t_prscrn)



    
    timer_cumer%TDFTGrdOct = timer_cumer%TDFTGrdOct + t_octree
    timer_cumer%TDFTPrscrn = timer_cumer%TDFTPrscrn + t_prscrn

#if defined CUDA_MPIV || defined HIP_MPIV
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

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

#if defined CUDA_MPIV || defined HIP_MPIV
   if(master) then
#endif

    ! save packed grid information into f90 data structures
     call get_gpu_grid_info(self%gridxb, self%gridyb, self%gridzb, self%gridb_sswt, self%gridb_weight, self%gridb_atm, &
     self%bin_locator, self%basf, self%primf, self%basf_counter, self%primf_counter, self%bin_counter)

#if defined CUDA_MPIV || defined HIP_MPIV
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

#if !defined CUDA && !defined HIP
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
#if defined CUDA_MPIV || defined HIP_MPIV
   if(master) then
#endif
    call gpack_finalize()
#if defined CUDA_MPIV || defined HIP_MPIV
    endif
#endif

    ! relinquish memory allocated for temporary f90 variables
    call dealloc_xcg_tmp_variables(xcg_tmp)

#ifdef MPIV
    if(master) then
#endif

    RECORD_TIME(timer_end%TDFTGrdPck)

    timer_cumer%TDFTGrdPck = timer_cumer%TDFTGrdPck + timer_end%TDFTGrdPck - timer_begin%TDFTGrdPck - t_octree &
                           - t_prscrn

!    write(*,*) "DFT grid timings: Grid form:", timer_cumer%TDFTGrdGen, "Compute grid weights:", timer_cumer%TDFTGrdWt, &
!    "Octree:",timer_cumer%TDFTGrdOct,"Prescreening:",timer_cumer%TDFTPrscrn, "Pack points:",timer_cumer%TDFTGrdPck


#ifdef MPIV
    endif
#endif

    end subroutine

    ! allocate memory for radius of significance, phi and dphi for host xc
    ! version
    subroutine allocate_sigrad_phi

        use quick_basis_module, only: nbasis, quick_basis, alloc
        implicit double precision(a-h,o-z)
        logical :: isDFT                 

        if (.not. allocated(sigrad2)) allocate(sigrad2(nbasis))

#if !defined CUDA || !defined CUDA_MPIV || !defined HIP || !defined HIP_MPIV
        isDFT = .true.
        call alloc(quick_basis, isDFT)
#endif

    end subroutine allocate_sigrad_phi

    ! deallocate sigrad2, phi, dphi
    subroutine deallocate_sigrad_phi

        use quick_basis_module, only: quick_basis, dealloc
        implicit double precision(a-h,o-z)
        logical :: isDFT

        if (allocated(sigrad2)) deallocate(sigrad2)

#if !defined CUDA || !defined CUDA_MPIV || !defined HIP || !defined HIP_MPIV
        isDFT = .true.
        call dealloc(quick_basis, isDFT)
#endif

    end subroutine deallocate_sigrad_phi

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

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
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

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
        if (allocated(self%bin_locator)) deallocate(self%bin_locator)
#endif
        if (allocated(self%bin_counter)) deallocate(self%bin_counter)

#ifdef MPIV
        if(bMPI) then
                call dealloc_mpi_grid_variables(self)
        endif
#endif
        ! deallocate sigrad2, phi, dphi and etc. 
        call deallocate_sigrad_phi()

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
     write (ioutfile,'("   PRUNING CUTOFF       =",E10.3)') quick_method%XCCutoff
     write (ioutfile,'("   INITIAL GRID POINTS  =",I12)') self%init_ngpts
     write (ioutfile,'("|   FINAL GRID POINTS    =",I12)') self%gridb_count
     write (ioutfile,'("|   SIGNIFICANT NUMBER OF BASIS FUNCTIONS     =",I12)') self%nbtotbf
     write (ioutfile,'("|   SIGNIFICANT NUMBER OF PRIMITIVE FUNCTIONS =",I12)') self%nbtotpf

   end subroutine print_grid_information

   subroutine get_sigrad
   
      ! calculate the radius of the sphere of basis function signifigance.
      ! (See Stratmann,Scuseria,and Frisch, Chem. Phys. Lett., 257, 1996, page 213-223 Section 5.)
      ! Also, the radius of the sphere comes from the spherical average of
      ! the basis function, from Perez-Jorda and Yang, Chem. Phys. Lett., 241,
      ! 1995, pg 469-76.
      ! The spherical average of a gaussian function is:
   
      ! (1 + 2 L)/4  (3 + 2 L)/4  L
      ! 2            a            r
      ! ave  = ---------------------------------
      ! 2
      ! a r                      3
      ! E     Sqrt[Pi] Sqrt[Gamma[- + L]]
      ! 2
      ! where a is the most diffuse (smallest) orbital exponent and L is the
      ! sum of the angular momentum exponents.  This code finds the r value where
      ! the average is the signifigance threshold (signif) and this r value is
      ! called the target below. Rearranging gives us:
   
      ! -(1 + 2 L)/4   -(3 + 2 L)/4                           3
      !r^L E^-ar^2= 2               a           Sqrt[Pi] signif Sqrt[Gamma[- + L]]
      ! 2
      use allmod
#ifdef MPIV
      use mpi
#endif
      implicit double precision(a-h,o-z)
   
#ifdef MPIV
      if(master) then
#endif
        if (quick_method%debug) write (iOutFile,'(/"RADII OF SIGNIFICANCE FOR THE BASIS FUNCTIONS")')
#ifdef MPIV
      endif
#endif
   
      do Ibas=1,nbasis
   
         ! Find the minimum gaussian exponent.
   
         amin=10.D10
         do Icon=1,ncontract(Ibas)
            amin=min(amin,aexp(Icon,Ibas))
         enddo
   
         ! calculate L.
   
         L = itype(1,Ibas)+ itype(2,Ibas)+ itype(3,Ibas)
   
         ! calculate 2 Pi Gamma[L+3/2]
         ! Remember that Gamma[i+1/2]=(i-1+1/2) Gamma[i-1+1/2] until you get to
         ! Gamma[1/2] = Sqrt[Pi]
         gamma=1.d0
         do i=1,L+1
            gamma = gamma * (dble(L+1-i) + .5d0)
         enddo
         gamma2pi=gamma*11.13665599366341569
   
         ! Now put it all together to get the target value.
   
         target = quick_method%basisCutoff* &
   
               (((2.d0*amin)**(dble(L)+1.5))/gamma2pi)**(-.5d0)
   
         ! Now search to find the correct radial value.
   
         stepsize=1.d0
         radial=0.d0
   
         do WHILE (stepsize.gt.1.d-4)
            radial=radial+stepsize
            current=Dexp(-amin*radial*radial)*radial**(dble(L))
            if (current < target) then
               radial=radial-stepsize
               stepsize=stepsize/10.d0
            endif
         enddo
   
         ! Store the square of the radii of signifigance as this is what the
         ! denisty calculator works in.
   
         sigrad2(Ibas)=radial*radial
   
#ifdef MPIV
         if(master) then
#endif
           if (quick_method%debug) write (iOutFile,'(I4,7x,F12.6)') Ibas,radial
#ifdef MPIV
         endif
#endif
   
      enddo
   
   end subroutine get_sigrad
   
   
   ! Xiao HE 2/9/07
   ! SG-0 standard grid CHIEN SH and Gill PMW,JCC 27,730,2006
   ! Gill PMW and CHIEN SH,JCC 24,732,2003
   ! EL-SHERBINY A and POIRIER RA JCC 25,1378,2004
   
   subroutine gridformSG0(iitype,ILEB,iiang,RGRIDt,RWTt)
      use allmod
      implicit double precision(a-h,o-z)
      parameter(MAXGNUMBER=30)
      double precision RGRIDt(MAXGNUMBER),RWTt(MAXGNUMBER)
   
      !      double precision :: ratomic(18)
      !      data ratomic &
            !      /1.30d0,0.0d0,1.95d0,2.20d0,1.45d0, 1.20d0,1.10d0 &
            !      ,1.10d0,1.20d0,0.0d0,2.30d0,2.20d0,2.10d0,1.30d0,1.30d0,1.10d0,
            !      &
            !      1.45d0,0.0d0/
   
      double precision :: aa46(46),aa52(52)
      data aa46 &
            /0.001505892474584d0,0.19397997519818d0, &
            0.009949112846861d0,0.262363963659648d0, &
            0.026212787562514d0,0.267428126763970d0, &
            0.050215014094684d0,0.248662024728409d0, &
            0.081660512821456d0,0.219828495609352d0, &
            0.120092694277137d0,0.187495823926848d0, &
            0.164915797166005d0,0.155234272915304d0, &
            0.215411397226014d0,0.125066174707640d0, &
            0.270754073298502d0,0.098100418872551d0, &
            0.330027592716307d0,0.074860364849399d0, &
            0.392241993524124d0,0.055477247414685d0, &
            0.456351572232027d0,0.039817159733692d0, &
            0.521273620526389d0,0.027571576765108d0, &
            0.585907670629421d0,0.018325604666402d0, &
            0.649154963911799d0,0.011611019656801d0, &
            0.709937833943106d0,0.006947749930384d0, &
            0.767218686284763d0,0.003875771461057d0, &
            0.820018260579743d0,0.001978563506850d0, &
            0.867432877982759d0,0.000898892530630d0, &
            0.908650423041717d0,0.000347557889758d0, &
            0.942964955771737d0,0.000105728680567d0, &
            0.969790557585917d0,0.000021565953939d0, &
            0.988681214124179d0,0.000001920578888d0/
   
      data aa52 &
            /0.00121189595314421d0,0.165909597900741d0, &
            0.00795083508342117d0,0.229969050948748d0, &
            0.0209119707010145d0,0.24044079379164d0, &
            0.040062890303616d0,0.229758620259279d0, &
            0.065227756094948d0,0.209301634430997d0, &
            0.0961248739663807d0,0.184563873212425d0, &
            0.132381145124222d0,0.158605827354705d0, &
            0.173541771179187d0,0.133246033424505d0, &
            0.219078862911660d0,0.109577661754471d0, &
            0.268400049319685d0,0.0882300767430008d0, &
            0.320857458070521d0,0.0695182568002067d0, &
            0.375757171448330d0,0.0535371508529839d0, &
            0.432369144701918d0,0.0402264558385049d0, &
            0.489937515066220d0,0.0294181396250844d0, &
            0.547691197495453d0,0.0208730234574819d0, &
            0.604854644468783d0,0.0143097962966441d0, &
            0.660658636490247d0,0.00942831962752039d0, &
            0.714350964474867d0,0.00592827752359897d0, &
            0.765206863851496d0,0.00352379596677165d0, &
            0.812539062502577d0,0.0019544295871423d0, &
            0.855707311124978d0,0.000992804711502569d0, &
            0.894127278139190d0,0.000449166032352755d0, &
            0.927278723938938d0,0.000173072023044939d0, &
            0.954712978447157d0,0.0000525042596156813d0, &
            0.976060296284965d0,0.0000106871851953229d0, &
            0.991042553891775d0,0.000000950391838932671d0/
   
      do i=1,MAXGNUMBER
         RGRIDt(i)=0.0d0
         RWTt(i)=0.0d0
      enddo
   
      if(quick_molspec%iattype(iitype).le.10)then
         do i=1,23
            RGRIDt(i)=-dlog(aa46(2*i-1))
            RWTt(i)=aa46(2*i)/aa46(2*i-1)
         enddo
      else
         do i=1,26
            RGRIDt(i)=-dlog(aa52(2*i-1))
            RWTt(i)=aa52(2*i)/aa52(2*i-1)
         enddo
      endif
   
      !  This subroutine calculates the angular and radial grid points
      !  and weights.  The current implementation presupposes Lebedev angular
      !  and Euler-Maclaurin radial grids.
   
      !  First, calculate the angular points and weights.
   
      if(quick_molspec%iattype(iitype).eq.1.or.quick_molspec%iattype(iitype).eq.3)then
         if(ILEB.le.6)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.9)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.10)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.11)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.12)then
            CALl LD0074(XANG,YANG,ZANG,WTANG,N)
            iiang=74
            elseif(ILEB.le.13)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.19)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.20)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.21)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.22)then
            CALl LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.23)then
            CALl LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
         endif
      endif
      if(quick_molspec%iattype(iitype).eq.6)then
         if(ILEB.le.6)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.8)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.9)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.11)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.13)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.14)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.15)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.16)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.18)then
            CALl LD0170(XANG,YANG,ZANG,WTANG,N)
            iiang=170
            elseif(ILEB.le.20)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.21)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.22)then
            CALl LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.23)then
            CALl LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
         endif
      endif
      if(quick_molspec%iattype(iitype).eq.7)then
         if(ILEB.le.6)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.9)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.10)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.12)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.14)then
            CALl LD0074(XANG,YANG,ZANG,WTANG,N)
            iiang=74
            elseif(ILEB.le.15)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.17)then
            CALl LD0170(XANG,YANG,ZANG,WTANG,N)
            iiang=170
            elseif(ILEB.le.20)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.21)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.23)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
         endif
      endif
      if(quick_molspec%iattype(iitype).eq.8)then
         if(ILEB.le.5)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.6)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.8)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.9)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.13)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.14)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.19)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.20)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.21)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.22)then
            CALl LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.23)then
            CALl LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
         endif
      endif
      if(quick_molspec%iattype(iitype).eq.9)then
         if(ILEB.le.4)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.6)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.10)then
            CALL LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.12)then
            CALL LD0074(XANG,YANG,ZANG,WTANG,N)
            iiang=74
            elseif(ILEB.le.14)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.16)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.18)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.21)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.22)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.23)then
            CALl LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
         endif
      endif
   
      if(quick_molspec%iattype(iitype).eq.15)then
         if(ILEB.le.5)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.9)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.13)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.16)then
            CALL LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.17)then
            CALl LD0074(XANG,YANG,ZANG,WTANG,N)
            iiang=74
            elseif(ILEB.le.19)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.20)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.23)then
            CALl LD0170(XANG,YANG,ZANG,WTANG,N)
            iiang=170
            elseif(ILEB.le.24)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.25)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.26)then
            CALl LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
         endif
      endif
   
      if(quick_molspec%iattype(iitype).eq.16)then
         if(ILEB.le.4)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.5)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.13)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.15)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.16)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.18)then
            CALl LD0074(XANG,YANG,ZANG,WTANG,N)
            iiang=74
            elseif(ILEB.le.19)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.22)then
            CALl LD0170(XANG,YANG,ZANG,WTANG,N)
            iiang=170
            elseif(ILEB.le.23)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.24)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.25)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.26)then
            CALl LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
         endif
      endif
   
      if(quick_molspec%iattype(iitype).eq.17)then
         if(ILEB.le.4)then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(ILEB.le.11)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.13)then
            CALL LD0026(XANG,YANG,ZANG,WTANG,N)
            iiang=26
            elseif(ILEB.le.15)then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(ILEB.le.16)then
            CALl LD0050(XANG,YANG,ZANG,WTANG,N)
            iiang=50
            elseif(ILEB.le.17)then
            CALl LD0074(XANG,YANG,ZANG,WTANG,N)
            iiang=74
            elseif(ILEB.le.19)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.22)then
            CALl LD0170(XANG,YANG,ZANG,WTANG,N)
            iiang=170
            elseif(ILEB.le.23)then
            CALl LD0146(XANG,YANG,ZANG,WTANG,N)
            iiang=146
            elseif(ILEB.le.24)then
            CALl LD0110(XANG,YANG,ZANG,WTANG,N)
            iiang=110
            elseif(ILEB.le.25)then
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(ILEB.le.26)then
            CALl LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
         endif
      endif
   
      !  The Lebedev weights are returned normalized to 1.  Multiply them by
      !  4Pi to get the correct value.
   
      do I=1,iiang
         wtang(I)=wtang(I)*12.56637061435917295385d0
      enddo
   
   end subroutine gridformSG0
   
   ! Xiao HE 1/9/07
   ! SG-1 standard grid Peter MWG, Benny GJ and Pople JA, CPL 209,506,1993,
   subroutine gridformnew(iitype,distance,iiang)
      use allmod
      implicit double precision(a-h,o-z)
   
      double precision :: hpartpara(4),lpartpara(4),npartpara(4)
   
      data hpartpara /0.2500d0,0.5000d0,1.0000d0,4.5000d0/
      data lpartpara /0.1667d0,0.5000d0,0.9000d0,3.5000d0/
      data npartpara /0.1000d0,0.4000d0,0.8000d0,2.5000d0/
   
      !  This subroutine calculates the angular and radial grid points
      !  and weights.  The current implementation presupposes Lebedev angular
      !  and Euler-Maclaurin radial grids.
   
      !  First, calculate the angular points and weights.
   
      if(quick_molspec%iattype(iitype).ge.1.and.quick_molspec%iattype(iitype).le.2)then
         if(distance.lt.hpartpara(1))then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(distance.lt.hpartpara(2))then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(distance.lt.hpartpara(3))then
            CALL LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(distance.lt.hpartpara(4))then
            CALL LD0194(XANG,YANG,ZANG,WTANG,N)
            iiang=194
         else
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
         endif
      else if(quick_molspec%iattype(iitype).ge.3.and.quick_molspec%iattype(iitype).le.10)then
         if(distance.lt.lpartpara(1))then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(distance.lt.lpartpara(2))then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(distance.lt.lpartpara(3))then
            CALL LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(distance.lt.lpartpara(4))then
            CALL LD0194(XANG,YANG,ZANG,WTANG,N)
            iiang=194
         else
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
         endif
      else if(quick_molspec%iattype(iitype).ge.11.and.quick_molspec%iattype(iitype).le.18)then
         if(distance.lt.npartpara(1))then
            CALL LD0006(XANG,YANG,ZANG,WTANG,N)
            iiang=6
            elseif(distance.lt.npartpara(2))then
            CALL LD0038(XANG,YANG,ZANG,WTANG,N)
            iiang=38
            elseif(distance.lt.npartpara(3))then
            CALL LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
            elseif(distance.lt.npartpara(4))then
            CALL LD0194(XANG,YANG,ZANG,WTANG,N)
            iiang=194
         else
            CALl LD0086(XANG,YANG,ZANG,WTANG,N)
            iiang=86
         endif
      else
         CALL LD0194(XANG,YANG,ZANG,WTANG,N)
         iiang=194
      endif
   
   
      !  The Lebedev weights are returned normalized to 1.  Multiply them by
      !  4Pi to get the correct value.
   
      do I=1,iiang
         wtang(I)=wtang(I)*12.56637061435917295385d0
      enddo
   
   end subroutine gridformnew

   subroutine gridformSG1
      use allmod
      implicit none
      integer itemp,i
      itemp=50
      do I=1,itemp
         RGRID(I)=(I**2.d0)/dble((itemp+1-I)*(itemp+1-I))
         RWT(I)=2.d0*dble(itemp+1)*(dble(I)**5.d0) &
               *dble(itemp+1-I)**(-7.d0)
      enddo
   end subroutine gridformSG1
   
#include "./include/labedev.fh"
   
end module quick_gridpoints_module
