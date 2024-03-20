#include "util.fh"
#ifdef MPIV
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup MPI environment
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine initialize_quick_mpi()
    use allmod
    use mpi
    implicit none
    logical mpi_initialized_flag

    ! Initinalize MPI evironment, and determind master node
    if (bMPI) then

      if(.not. libMPIMode) then
        call MPI_INIT(mpierror)
        call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
      endif

      call MPI_GET_PROCESSOR_NAME(pname,namelen,mpierror)
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    
      if(.not. allocated(MPI_STATUS)) allocate(MPI_STATUS(MPI_STATUS_SIZE))
    
      if (mpirank.eq.0) then
        master=.true.
      else
        master=.false.
      endif
    else
      master=.true.
    endif
    
    call mpi_setup_timer
    
    end subroutine
    
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Job specs   
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_setup_job(ierr)
    use allmod
    use mpi
    implicit none
    integer, intent(inout) :: ierr   
    
    call Broadcast(quick_method,ierr)
    call MPI_BCAST(natom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    if (quick_method%ecp) then
        call MPI_BCAST(tolecp,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(thrshecp,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    endif

    call MPI_BCAST(quick_molspec%nextatom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    end

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Mol specs part 1    
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_setup_mol1(ierr)
    use allmod
    use quick_gridpoints_module
    use mpi
    implicit none

    integer :: i    
    integer, intent(inout) :: ierr

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
   
! mols specs
    call Broadcast(quick_molspec,ierr)
    call MPI_BCAST(natom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(xyz,natom*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    
    end


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Mol specs part 2
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_setup_mol2(ierr)

    use allmod
    use quick_gridpoints_module
    use mpi
    implicit none
    integer, intent(inout) :: ierr

    call Broadcast(quick_molspec,ierr)

    call MPI_BCAST(dcoeff,nbasis*maxcontract,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    if (quick_method%ecp) then
      call MPI_BCAST(eta,nprim,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    endif

! SEDFT Parameters  
    if (quick_method%SEDFT) then
      call MPI_BCAST(At1prm,3*3*3*84,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(bndprm,3*3*3*84,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    endif

    end

    subroutine mpi_bcast_grid_vars

      use quick_gridpoints_module, only: RGRID, RWT 
      use quick_size_module, only: MAXRADGRID
      use quick_mpi_module, only: mpierror
      use mpi
      implicit none

      call MPI_BCAST(RGRID,MAXRADGRID,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(RWT,MAXRADGRID,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

    end subroutine mpi_bcast_grid_vars   

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Mol Basis
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    subroutine mpi_setup_basis
    use allmod
    use mpi
    implicit none
    
    integer :: i, j

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(jshell,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(jbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(nshell,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(nprim,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    
    call MPI_BCAST(quick_basis%kshell,93,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%ktype,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%katom,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%kstart,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%kprim,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    
    call MPI_BCAST(quick_basis%Qnumber,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%Qstart,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%Qfinal,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%Qsbasis,nshell*4,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%Qfbasis,nshell*4,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%ksumtype,nshell+1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%cons,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    
    
    if (quick_method%ecp) then
        call MPI_BCAST(kmin,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(kmax,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(ktypecp,nshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    endif
    
    
    
!    call MPI_BCAST(aexp,nprim,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!    call MPI_BCAST(gcs,nprim,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!    call MPI_BCAST(quick_basis%gccoeff,6*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!    call MPI_BCAST(quick_basis%gcexpo,6*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!    call MPI_BCAST(quick_basis%gcexpomin,nshell,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

    !Madu: 05/01/2019
    call MPI_BCAST(quick_basis%gccoeff,size(quick_basis%gccoeff),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%gcexpo,size(quick_basis%gcexpo),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_molspec%chg,size(quick_molspec%chg),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_method%iopt,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    call MPI_BCAST(quick_basis%KLMN,3*nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(itype,3*nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%ncenter,nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(ncontract,nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    
!    call MPI_BCAST(maxcontract,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(aexp,maxcontract*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcoeff,maxcontract*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    
    call MPI_BCAST(quick_basis%first_basis_function,natom,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%last_basis_function,natom,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)   

    end

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup iniDivCon specs 
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_setup_inidivcon(natomt)
    use allmod
    use mpi
    implicit none
    integer natomt,i,k1,k2,j,k,tempinteger,tempinteger2
    
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)


    call MPI_BCAST(kshells,natomt,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(kshellf,natomt,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcconnect,jshell*jshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    
    
    call MPI_BCAST(np,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(npsaved,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(NNmax,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    call MPI_BCAST(dccore,npsaved*500,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dccoren,npsaved,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcbuffer1,npsaved*500,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcbuffer2,npsaved*500,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcbuffer1n,npsaved,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcbuffer2n,npsaved,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcsub,npsaved*500,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcsubn,npsaved,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    
    call MPI_BCAST(dclogic,npsaved*natomt*natomt,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    call MPI_BCAST(nbasisdc,npsaved,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(nelecdcsub,npsaved,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(invdcoverlap,natomt*natomt,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(dcoverlap,natomt*natomt,mpi_integer,0,MPI_COMM_WORLD,mpierror)


    
    !-------------------MPI/MASTER ------------------------------------
    ! Distrubute fragment to nodes for calculation
    if (master) then
        ! re-initial mpi dc frag infos
        do i=0,mpisize-1
            mpi_dc_fragn(i)=0
            mpi_dc_nbasis=0
            do j=1,np
                mpi_dc_frag(i,j)=0
            enddo
        enddo
        
        write(iOutfile,'("-----------------------------------------------------------")')
        write(iOutfile,'("         MPI INFORMATION")')
        write(iOutfile,'("-----------------------------------------------------------")')
        write(iOutfile,'(" MPI SIZE =",i4)') mpisize
        write(iOutfile,'(" NODE",2x,"TotFrag",5x,"NBasis",2x,"Frag")')
        
        ! use greedy algrithm to obtain the distrubution
        call greedy_distrubute(nbasisdc,np,mpisize,mpi_dc_fragn,mpi_dc_frag)    
        
        do i=0,mpisize-1
            do j=1,mpi_dc_fragn(i)
                mpi_dc_nbasis(i)=mpi_dc_nbasis(I)+nbasisdc(mpi_dc_frag(i,j))
            enddo
        enddo
                        
        do i=0,mpisize-1
            write(iOutfile,'(i4,4x,i4,5x,i4,2x)',advance="no") i,mpi_dc_fragn(i),mpi_dc_nbasis(i)
            do j=1,mpi_dc_fragn(i)
                write(iOutfile,'(i4)',advance="no") mpi_dc_frag(i,j)
            enddo
            write(iOutfile,*)
        enddo
        write(iOutfile,'("-----------------------------------------------------------")')

        call flush(iOutfile)
    endif
    !-------------------END MPI/MASTER --------------------------------
            
    ! Broadcast mpi_dc variables
    call MPI_BCAST(mpi_dc_fragn,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(mpi_dc_frag,mpisize*np,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(mpi_dc_nbasis,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
                
    end subroutine mpi_setup_inidivcon
    
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup operator duties
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    subroutine MPI_setup_hfoperator()
    use allmod
    use mpi
    implicit none
    integer i,k1,k2,j,k,tempinteger,tempinteger2
    integer temp1d(nbasis)
    
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    

    if (MASTER) then
        
        ! The first part is to distrubute jshell
        ! re-initial mpi jshell info
        do i=0,mpisize-1
            mpi_jshelln(i)=0
            do j=1,jshell
                mpi_jshell(i,j)=0
            enddo
        enddo


        do i=1,jshell
            temp1d(i)=jshell-i
        enddo
        ! here we use greed method to obtain the optimized distrubution
        ! please note the algrithm is not the most optimized but is almost is
        call greedy_distrubute(temp1d(1:jshell),jshell,mpisize,mpi_jshelln,mpi_jshell)
        
        ! The second part is to distrubute nbasis       
        ! re-initial mpi nbasis info
        do i=0,mpisize-1
            mpi_nbasisn(i)=0
            do j=1,nbasis
                mpi_nbasis(i,j)=0
            enddo
        enddo

        do i=1,nbasis
            temp1d(i)=nbasis-i
        enddo
        ! here we use greed method to obtain the optimized distrubution
        ! please note the algrithm is not the most optimized but is almost is
        call greedy_distrubute(temp1d,nbasis,mpisize,mpi_nbasisn,mpi_nbasis)
    endif
    
    if (bMPI) then
        call MPI_BCAST(mpi_jshelln,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(mpi_jshell,mpisize*jshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        
        call MPI_BCAST(mpi_nbasisn,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(mpi_nbasis,mpisize*nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

    endif

    call mpi_distribute_atoms(quick_molspec%natom, quick_molspec%nextatom)
    
    end subroutine MPI_setup_hfoperator

#if defined CUDA_MPIV || defined HIP_MPIV

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup multi GPUs
! Madu Manathunga 07/22/2020
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    subroutine mgpu_setup()

      use quick_mpi_module
      use mpi
      implicit none
      integer :: i, IERROR

      ! allocate memory for device ids
      if(master) call allocate_mgpu

      ! get slave device ids and save them for printing device info
      if(.not. master) then
        call MPI_SEND(mgpu_id,1,mpi_integer,0,mpirank,MPI_COMM_WORLD,IERROR)
      else
        mgpu_ids(1)=mgpu_id

        do i=1,mpisize-1
          call MPI_RECV(mgpu_ids(i+1),1,mpi_integer,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
        enddo
        
      endif

    end subroutine mgpu_setup


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Delete multi GPU setup
! Madu Manathunga 07/22/2020
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine delete_mgpu_setup(ierr)

      use quick_mpi_module
      implicit none
      integer, intent(inout) :: ierr

      call deallocate_mgpu()

    end subroutine delete_mgpu_setup

#endif


 subroutine setup_xc_mpi_1
   use allmod
   use quick_gridpoints_module
   use mpi
   implicit double precision(a-h,o-z)

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridb_count,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%nbtotbf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%nbtotpf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%nbins,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_method%nof_functionals,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
 
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
 end subroutine setup_xc_mpi_1

 subroutine setup_xc_mpi_new_imp
!-----------------------------------------------------------------------------
!  This subroutine sets the mpi environment required for exchange correlation 
!  energy/gradient computations (i.e. get_xc & get_xc_grad methods).
!  Madu Manathunga 01/03/2020
!-----------------------------------------------------------------------------
   use allmod
   use quick_gridpoints_module
   use mpi
   implicit double precision(a-h,o-z)

   integer, dimension(1:mpisize) :: itotgridspn
!   integer, dimension(0:mpisize-1) :: igridptul
!   integer, dimension(0:mpisize-1) :: igridptll
 
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

#if !(defined CUDA_MPIV) || !(defined HIP_MPIV)

   if(master) then
      do impi=1, mpisize
         itotgridspn(impi)=0
         quick_dft_grid%igridptul(impi)=0
         quick_dft_grid%igridptll(impi)=0
      enddo

      itmpgriddist=quick_dft_grid%nbins
      do while(itmpgriddist .gt. 0)
         do impi=1, mpisize
            itotgridspn(impi)=itotgridspn(impi)+1
            itmpgriddist=itmpgriddist-1
            if (itmpgriddist .lt. 1) exit
         enddo
      enddo

      itmpgridptul=0
      do impi=1, mpisize
         itmpgridptul=itmpgridptul+itotgridspn(impi)
         quick_dft_grid%igridptul(impi)=itmpgridptul
         if(impi .eq. 1) then
            quick_dft_grid%igridptll(impi)=1
         else
            quick_dft_grid%igridptll(impi)=quick_dft_grid%igridptul(impi-1)+1
         endif
      enddo

   endif

#endif


   if(bMPI) then

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(quick_basis%gccoeff,size(quick_basis%gccoeff),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_basis%gcexpo,size(quick_basis%gcexpo),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_molspec%chg,size(quick_molspec%chg),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

#if defined CUDA_MPIV || defined HIP_MPIV
      call MPI_BCAST(quick_dft_grid%bin_locator,quick_dft_grid%gridb_count,mpi_integer,0,MPI_COMM_WORLD,mpierror)
#else
      call MPI_BCAST(quick_dft_grid%igridptll,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%igridptul,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
#endif
      call MPI_BCAST(quick_dft_grid%bin_counter,quick_dft_grid%nbins+1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%basf_counter,quick_dft_grid%nbins+1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%primf_counter,quick_dft_grid%nbtotbf+1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%basf,quick_dft_grid%nbtotbf,mpi_integer,0,MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(quick_dft_grid%primf,quick_dft_grid%nbtotpf,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridxb,quick_dft_grid%gridb_count,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridyb,quick_dft_grid%gridb_count,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridzb,quick_dft_grid%gridb_count,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridb_sswt,quick_dft_grid%gridb_count,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridb_weight,quick_dft_grid%gridb_count,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridb_atm,quick_dft_grid%gridb_count,mpi_integer,0,MPI_COMM_WORLD,mpierror)

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

   endif

   return
 end subroutine setup_xc_mpi_new_imp

   subroutine setup_ssw_mpi

   use allmod
   use quick_gridpoints_module
   use mpi
   implicit double precision(a-h,o-z)

   integer, dimension(1:mpisize) :: itotgridspn
   
   if(master) then

      do impi=1, mpisize
         itotgridspn(impi)=0
         quick_dft_grid%igridptul(impi)=0
         quick_dft_grid%igridptll(impi)=0
      enddo

      itmpgriddist=quick_dft_grid%init_ngpts

      do while(itmpgriddist .gt. 0)
         do impi=1, mpisize
            itotgridspn(impi)=itotgridspn(impi)+1
            itmpgriddist=itmpgriddist-1
            if (itmpgriddist .lt. 1) exit
         enddo
      enddo

      itmpgridptul=0
      do impi=1, mpisize
         itmpgridptul=itmpgridptul+itotgridspn(impi)
         quick_dft_grid%igridptul(impi)=itmpgridptul
         if(impi .eq. 1) then
            quick_dft_grid%igridptll(impi)=1
         else
            quick_dft_grid%igridptll(impi)=quick_dft_grid%igridptul(impi-1)+1
         endif
      enddo

   endif  

    do j=1,quick_dft_grid%init_ngpts
        quick_xcg_tmp%sswt(j) = 0.0d0
        quick_xcg_tmp%tmp_sswt(j) = 0.0d0
        quick_xcg_tmp%weight(j) = 0.0d0
        quick_xcg_tmp%tmp_weight(j) = 0.0d0
    enddo


   if(bMPI) then

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(quick_dft_grid%init_ngpts, 1, mpi_integer, 0, MPI_COMM_WORLD,mpierror)      
      call MPI_BCAST(quick_dft_grid%igridptll,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%igridptul,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_ptx,quick_dft_grid%init_ngpts,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_pty,quick_dft_grid%init_ngpts,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_ptz,quick_dft_grid%init_ngpts,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_atm,quick_dft_grid%init_ngpts,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%arr_wtang,quick_dft_grid%init_ngpts,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%arr_rwt,quick_dft_grid%init_ngpts,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%arr_rad3,quick_dft_grid%init_ngpts,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)




call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
   endif 

   end subroutine setup_ssw_mpi

   subroutine get_mpi_ssw

   use allmod
   use quick_gridpoints_module
   use mpi
   implicit double precision(a-h,o-z)

   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

   if(.not. master) then
      call MPI_SEND(quick_xcg_tmp%sswt,quick_dft_grid%init_ngpts,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
      call MPI_SEND(quick_xcg_tmp%weight,quick_dft_grid%init_ngpts,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
   else

      do i=1,mpisize-1
         call MPI_RECV(quick_xcg_tmp%tmp_sswt,quick_dft_grid%init_ngpts,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
         call MPI_RECV(quick_xcg_tmp%tmp_weight,quick_dft_grid%init_ngpts,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)

         do j=1,quick_dft_grid%init_ngpts
            quick_xcg_tmp%sswt(j)=quick_xcg_tmp%sswt(j)+quick_xcg_tmp%tmp_sswt(j)
            quick_xcg_tmp%weight(j)=quick_xcg_tmp%weight(j)+quick_xcg_tmp%tmp_weight(j)
         enddo
      enddo

   endif

 
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  
   end subroutine get_mpi_ssw

#endif
