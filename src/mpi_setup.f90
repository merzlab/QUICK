#include "config.h"
#ifdef MPI
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup MPI environment
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_initialize()
    use allmod
    implicit none
    logical mpi_initialized_flag
    
    include 'mpif.h'

    ! Initinalize MPI evironment, and determind master node
    if (bMPI) then
      call MPI_INIT(mpierror)
      call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
      call MPI_GET_PROCESSOR_NAME(pname,namelen,mpierror)
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    
      allocate(MPI_STATUS(MPI_STATUS_SIZE))
    
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
    end

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Setup Mol specs part 1    
! Yipu Miao 08/03/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
    subroutine mpi_setup_mol1()
    use allmod
    implicit none
    
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
    call MPI_BCAST(quick_basis%gccoeff,6*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(quick_basis%gcexpo,6*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!    call MPI_BCAST(quick_basis%gcexpomin,nshell,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

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
    endif

    
    end subroutine MPI_setup_hfoperator

#endif
