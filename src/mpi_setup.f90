#include "config.h"
#ifdef MPIV
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup MPI environment
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine initialize_quick_mpi()
    use allmod
    implicit none
    logical mpi_initialized_flag
    
    include 'mpif.h'

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
    subroutine mpi_setup_job()
    use allmod
    implicit none
    
    include "mpif.h"
    
    call Broadcast(quick_method)
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
    subroutine mpi_setup_mol1()
    use allmod
    implicit none

    integer :: i    
    include 'mpif.h'

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
   
! mols specs
    call Broadcast(quick_molspec)
    call MPI_BCAST(natom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(xyz,natom*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
! DFT and SEDFT specs
    call MPI_BCAST(RGRID,MAXRADGRID,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(RWT,MAXRADGRID,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    
    end


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Mol specs part 2
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_setup_mol2()
    use allmod
    implicit none
    
    include 'mpif.h'

    call Broadcast(quick_molspec)
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

    call MPI_BCAST(dcoeff,nbasis*maxcontract,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    if (quick_method%ecp) then
      call MPI_BCAST(eta,nprim,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    endif

! DFT Parameter
    if (quick_method%DFT.or.quick_method%SEDFT) then  
      call MPI_BCAST(sigrad2,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    endif

! SEDFT Parameters  
    if (quick_method%SEDFT) then
      call MPI_BCAST(At1prm,3*3*3*84,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(bndprm,3*3*3*84,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    endif
    
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

    end

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Mol Basis
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    subroutine mpi_setup_basis
    use allmod
    implicit none
    
    include 'mpif.h'
    

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
    implicit none
    integer natomt,i,k1,k2,j,k,tempinteger,tempinteger2
    
    include 'mpif.h'    
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
    implicit none
    integer i,k1,k2,j,k,tempinteger,tempinteger2
    integer temp1d(nbasis)
    
    include 'mpif.h'
    
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

#ifdef CUDA_MPIV
!        call mgpu_upload_basis_setup(mpi_jshelln,mpi_jshell,mpi_nbasisn,mpi_nbasis)
#endif

    endif

    
    end subroutine MPI_setup_hfoperator

#ifdef CUDA_MPIV

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup eri calculation on multi GPUs
! Madu Manathunga 05/08/2020
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine MPI_setup_mgpu_eri()
    use allmod
    implicit none

    include 'mpif.h'

    if (master) then
        call mgpu_distribute_qshell(quick_basis%mpi_qshell,quick_basis%mpi_qshelln)
    endif

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%mpi_qshell,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%mpi_qshelln,mpisize+1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    call mgpu_upload_qshell(quick_basis%mpi_qshell, quick_basis%mpi_qshelln)

    end subroutine MPI_setup_mgpu_eri

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup eri calculation on multi GPUs
! Madu Manathunga 05/08/2020
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine MPI_setup_arr_bsd_mgpu_eri()
    use allmod
    implicit none
    integer :: nqshell      ! Number of sorted shells
    integer :: remainder, idx, impi, icount 
    integer, allocatable, dimension(:)  :: mpi_qshell
    integer, allocatable, dimension(:)  :: mpi_qshelln

    include 'mpif.h'

    if (master) then
        call mgpu_get_nqshell(nqshell)
    endif

    call MPI_BCAST(nqshell,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
 
    ! All nodes allocate memory and initialize values to zero
    if(.not. allocated(mpi_qshell)) allocate(mpi_qshell(mpisize))
    if(.not. allocated(mpi_qshelln)) allocate(mpi_qshelln(mpisize*nqshell))
    call zeroiVec(mpi_qshell,mpisize)
    call zeroiVec(mpi_qshelln,mpisize*nqshell) 

    ! Now master will distribute the qshells 
    if(master) then
       icount=1
       remainder = nqshell
       do while(remainder .gt. 0)
           do impi=1, mpisize
              idx=nqshell-remainder
              mpi_qshell(impi) = mpi_qshell(impi)+1
              mpi_qshelln((impi-1)*nqshell+icount)=idx   
              remainder=remainder-1
              if (remainder .lt. 1) exit
           enddo
           icount=icount+1
       enddo    
    endif    

    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(mpi_qshell,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(mpi_qshelln,mpisize*nqshell,mpi_integer,0,MPI_COMM_WORLD,mpierror)

    call mgpu_upload_arr_bsd_qshell(mpi_qshell,mpi_qshelln)

    end subroutine MPI_setup_arr_bsd_mgpu_eri

#endif

! subroutine setup_xc_mpi(itotgridspn, igridptul, igridptll, Iradtemp)
!-----------------------------------------------------------------------------
!  This subroutine sets the mpi environment required for exchange correlation 
!  energy/gradient computations (i.e. get_xc & get_xc_grad methods).
!  Madu Manathunga 08/15/2019
!-----------------------------------------------------------------------------
!   use allmod
!   implicit double precision(a-h,o-z)

!   integer, dimension(0:mpisize-1) :: itotgridspn
!   integer, dimension(0:mpisize-1) :: igridptul
!   integer, dimension(0:mpisize-1) :: igridptll

!   include 'mpif.h'

!   if(master) then
!      do impi=0, mpisize-1
!         itotgridspn(impi)=0
!         igridptul(impi)=0
!         igridptll(impi)=0
!      enddo
!
!      itmpgriddist=Iradtemp
!      do while(itmpgriddist .gt. 1)
!         do impi=0, mpisize-1
!            itotgridspn(impi)=itotgridspn(impi)+1
!            itmpgriddist=itmpgriddist-1
!            if (itmpgriddist .lt. 1) exit
!         enddo
!      enddo
!
!      itmpgridptul=0
!      do impi=0, mpisize-1
!         itmpgridptul=itmpgridptul+itotgridspn(impi)
!         igridptul(impi)=itmpgridptul
!         if(impi .eq. 0) then
!            igridptll(impi)=1
!         else
!            igridptll(impi)=igridptul(impi-1)+1
!         endif
!      enddo
!   endif

!   if(bMPI) then

!      call MPI_BCAST(quick_basis%gccoeff,size(quick_basis%gccoeff),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(quick_basis%gcexpo,size(quick_basis%gcexpo),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(quick_molspec%chg,size(quick_molspec%chg),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      !call MPI_BCAST(quick_method%nof_functionals,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      !call MPI_BCAST(quick_method%functional_id,size(quick_method%functional_id),mpi_integer,0,MPI_COMM_WORLD,mpierror)
      !call MPI_BCAST(quick_method%xc_polarization,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(igridptll,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(igridptul,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
!   endif      

!   return
! end subroutine setup_xc_mpi 

 subroutine setup_xc_mpi_1
   use allmod
   implicit double precision(a-h,o-z)

   include 'mpif.h'

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%gridb_count,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%nbtotbf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%nbtotpf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%nbins,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
 
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
 end subroutine setup_xc_mpi_1

 subroutine setup_xc_mpi_new_imp
!-----------------------------------------------------------------------------
!  This subroutine sets the mpi environment required for exchange correlation 
!  energy/gradient computations (i.e. get_xc & get_xc_grad methods).
!  Madu Manathunga 01/03/2020
!-----------------------------------------------------------------------------
   use allmod
   implicit double precision(a-h,o-z)

   integer, dimension(1:mpisize) :: itotgridspn
!   integer, dimension(0:mpisize-1) :: igridptul
!   integer, dimension(0:mpisize-1) :: igridptll

   include 'mpif.h'
 
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

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

   if(bMPI) then

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(quick_basis%gccoeff,size(quick_basis%gccoeff),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_basis%gcexpo,size(quick_basis%gcexpo),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_molspec%chg,size(quick_molspec%chg),mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      !call
      !MPI_BCAST(quick_method%nof_functionals,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      !call
      !MPI_BCAST(quick_method%functional_id,size(quick_method%functional_id),mpi_integer,0,MPI_COMM_WORLD,mpierror)
      !call
      !MPI_BCAST(quick_method%xc_polarization,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%igridptll,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%igridptul,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(quick_dft_grid%gridb_count,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(quick_dft_grid%nbtotbf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(quick_dft_grid%nbtotpf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
!      call MPI_BCAST(quick_dft_grid%nbins,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

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
   implicit double precision(a-h,o-z)

   integer, dimension(1:mpisize) :: itotgridspn
   include 'mpif.h'
   
   if(master) then

      do impi=1, mpisize
         itotgridspn(impi)=0
         quick_dft_grid%igridptul(impi)=0
         quick_dft_grid%igridptll(impi)=0
      enddo

      itmpgriddist=quick_xcg_tmp%idx_grid

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

    do j=1,quick_xcg_tmp%idx_grid
        quick_xcg_tmp%sswt(j) = 0.0d0
        quick_xcg_tmp%tmp_sswt(j) = 0.0d0
        quick_xcg_tmp%weight(j) = 0.0d0
        quick_xcg_tmp%tmp_weight(j) = 0.0d0
    enddo


   if(bMPI) then

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(quick_xcg_tmp%idx_grid, 1, mpi_integer, 0, MPI_COMM_WORLD,mpierror)      
      call MPI_BCAST(quick_dft_grid%igridptll,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_dft_grid%igridptul,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_ptx,quick_xcg_tmp%idx_grid,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_pty,quick_xcg_tmp%idx_grid,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_ptz,quick_xcg_tmp%idx_grid,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%init_grid_atm,quick_xcg_tmp%idx_grid,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%arr_wtang,quick_xcg_tmp%idx_grid,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%arr_rwt,quick_xcg_tmp%idx_grid,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_xcg_tmp%arr_rad3,quick_xcg_tmp%idx_grid,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)



   !if(.not. master) then
   !   do idx=1, quick_xcg_tmp%idx_grid
   !      write(*,*) "ssw bounds:",quick_xcg_tmp%init_grid_ptx(idx),quick_xcg_tmp%init_grid_pty(idx), &
   !      quick_xcg_tmp%init_grid_ptz(idx), quick_xcg_tmp%init_grid_atm(idx)
   !   enddo
   !endif

call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
   endif 

   end subroutine setup_ssw_mpi

   subroutine get_mpi_ssw

   use allmod
   implicit double precision(a-h,o-z)

   include 'mpif.h'

   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

   if(.not. master) then
      call MPI_SEND(quick_xcg_tmp%sswt,quick_xcg_tmp%idx_grid,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
      call MPI_SEND(quick_xcg_tmp%weight,quick_xcg_tmp%idx_grid,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
   else

      do i=1,mpisize-1
         call MPI_RECV(quick_xcg_tmp%tmp_sswt,quick_xcg_tmp%idx_grid,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
         call MPI_RECV(quick_xcg_tmp%tmp_weight,quick_xcg_tmp%idx_grid,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)

         do j=1,quick_xcg_tmp%idx_grid
            quick_xcg_tmp%sswt(j)=quick_xcg_tmp%sswt(j)+quick_xcg_tmp%tmp_sswt(j)
            quick_xcg_tmp%weight(j)=quick_xcg_tmp%weight(j)+quick_xcg_tmp%tmp_weight(j)
         enddo
      enddo

   endif

 
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  
   end subroutine get_mpi_ssw

#endif
