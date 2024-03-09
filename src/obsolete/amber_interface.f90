#include "util.fh"
!
!	amber_interface.f90
!	amber_interface
!
!	Created by Yipu Miao on 1/19/11.
!	Copyright 2011 University of Florida. All rights reserved.
!


subroutine qm2_quick_energy(escf,scf_mchg)

! Calculates the SCF, DFT or MP2 energy by calling quick
! The energy is returned in 'escf'.
!
!
!     Variables for qm-mm:
!
!     qmmm_struct%nquant_nlink    - Total number of qm atoms. (Real + link)
!
! This routine is called from qm2_energies.f
  
!In parallel is implimented from quick

   use qmmm_module, only : qmmm_nml, qmmm_struct, qm2_struct, element_sym, qmmm_mpi
   
! quick mod
   use allmod
   use mpi
   implicit none

   double precision, intent(out)   :: escf
   double precision, intent(inout) :: scf_mchg(qmmm_struct%nquant_nlink)

   double precision total_e, geseatom
   
   double precision, parameter :: AU_TO_EV = 27.21d0, EV_TO_KCAL = 23.060362D0
   double precision, parameter :: AU_TO_KCAL = AU_TO_EV*EV_TO_KCAL

   integer k, i, j
   integer mm_link_atom
   logical :: failed=.false.
   
   integer :: ierror, natomsave
   character(len=80) :: keyWD

   ! BLOCK MPI QM CALCULATION FOR TEST PHASE
   bMPI=.false.
   
   ! Enable some funcion hidding in quick designing for AMBER interface
   AMBER_interface_logic=.true.
   
   !==============================
   ! Print basic information 
   !==============================
   if (qmmm_mpi%commqmmm_master) then   

     ! The classical atoms from nonbond list will treat as point charge
     if (qmmm_nml%verbosity > 0) then
        if (qmmm_struct%qm_mm_pairs > 0) then
           write(6,*) "QMMM QUICK: Classical atoms not in QM Region will taken as point charges."
        else
           write(6,*) "QMMM QUICK: No external charges."
        end if
     end if

     !     Print molecule information
     if (qmmm_nml%verbosity > 3) then
        
        ! QM region information
        write(6,'(" QMMM QUICK: QM Region Input Cartesian Coordinates ")')
        write(6,'(" QMMM QUICK: ",4X,"NO.",2X,"ATOM",9X,"X",16X,"Y",16X,"Z",13X,"CHARGE")')
        do i = 1, qmmm_struct%nquant_nlink
           write(6,'(" QMMM QUICK: ",I6,2X,i2,1X,A2,2X,4F16.10)') i, &
                 qmmm_struct%iqm_atomic_numbers(i), &
                 element_sym(qmmm_struct%iqm_atomic_numbers(i)), &
                 (qmmm_struct%qm_coords(j,i), j=1,3), scf_mchg(i)
        end do

        ! MM region information
        if (qmmm_struct%qm_mm_pairs > 0) then
           write(6,'(" QMMM QUICK: number of external charges",I6)') qmmm_struct%qm_mm_pairs
           write(6,'(" QMMM QUICK: MM Region Input Cartesian Coordinates")')
           write(6,'(" QMMM QUICK: ",4X,"NO.",16X,"X",16X,"Y",16X,"Z",13X,"CHARGE")')
           do i=1,qmmm_struct%qm_mm_pairs
              write(6,'(" QMMM QUICK: ",i6,4(2x,f16.10))') i,(qmmm_struct%qm_xcrd(j,i),j=1,4)
           end do
        end if
     end if
   end if
   
!*****************************************************************
! Begin of initialization
!-----------------------------------------------------------------
!
    if (quick_first_call) then
        !--------------------MPI/ALL NODES--------------------------------
        ! MPI Initializer
        call MPI_initialize()
        !------------------- End MPI  -----------------------------------

        ! Initial neccessary variables 
        call initialize1(ierror)
    endif
    
    call cpu_time(timer_begin%TTotal) ! Trigger time counter    
    
    !--------------------MPI/MASTER----------------------------------
    !   
    masterwork_initial: if (qmmm_mpi%commqmmm_master) then ! Master Work: to initial the work
    !----------------End MPI/MASTER----------------------------------

      ! Read enviromental variables: QUICK_BASIS and ECPs
      ! those can be defined in ~/.bashrc
      call getenv("QUICK_BASIS",basisdir)
      call getenv("ECPs",ecpdir)

    !--------------------MPI/MASTER-----------------------------------
    endif masterwork_initial
    !--------------- End MPI/MASTER-----------------------------------


!*****************************************************************
! 1. The first thing that must be done is reading in the molecule atoms
!------------------------------------------------------------------
!

    !-------------------MPI/MASTER---------------------------------------
    masterwork_readInput: if (qmmm_mpi%commqmmm_master) then

      natomsave=natom
      
      ! forword output to amber output file
      ioutfile=6
      
      ! total number of QM region, including link atoms
      natom=qmmm_struct%nquant_nlink
      
      ! atom type
      iatomtype=qmmm_struct%qm_ntypes
      
      ! symbol of atom type
      do i=1,iatomtype
        atom_type_sym(i)=symbol(qmmm_struct%qm_type_id(i))
      enddo
            
    
    endif masterwork_readInput ! master
    !--------------------End MPI/MASTER----------------------------------

    !-------------------MPI/ALL NODES------------------------------------    
    if (bMPI) then
      call MPI_BCAST(natomsave,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(iatomtype,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(natom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    endif
    !------------------END MPI/ALL NODES---------------------------------    

    !allocate essential variables
    if (quick_first_call) call alloc(quick_molspec)

!*****************************************************************
! 2. Next step is to read job and initial guess
!------------------------------------------------------------------
!
    !read job spec
    call read_AMBER_Job
        
    ! Then do inital guess
    call cpu_time(timer_begin%TIniGuess)

    ! SAD method is default and the only option now available for AMBER interface
    if (iSAD.eq.0 .and. quick_first_call) then
        call getMolSad()
    endif
    
    ! pass mol info from AMBER to quick
    if (quick_first_call) then
        call getMol(qmmm_struct%nquant_nlink)
    else
        call initialGuess
        call read_AMBER_interface
        call read_AMBER_charge
    endif
    
    quick_first_call=.false.
    
    call g2eshell
    call schwarzoff
    call getEnergy(failed)
    
    ! Output Energy information
    
    if (qmmm_mpi%commqmmm_master .and. qmmm_nml%verbosity > 1) then
        write(6,'(" QMMM QUICK:")')
        write(6,'(" QMMM QUICK:    Electronic Energy       = ",f20.12," eV (", f20.12," kcal)")') &
                                quick_qm_struct%Eel * AU_TO_EV, quick_qm_struct%Eel * AU_TO_KCAL
        write(6,'(" QMMM QUICK:    Repulsive Energy        = ",f20.12," eV (", f20.12," kcal)")') &
                                quick_qm_struct%Ecore * AU_TO_EV, quick_qm_struct%Ecore * AU_TO_KCAL
        write(6,'(" QMMM QUICK:    Total Energy            = ",f20.12," eV (", f20.12," kcal)")') &
                                quick_qm_struct%Etot * AU_TO_EV, quick_qm_struct%Etot * AU_TO_KCAL

    end if
    
    escf=quick_qm_struct%Etot * AU_TO_KCAL
    qmmm_struct%elec_eng = quick_qm_struct%Eel * AU_TO_EV
    qmmm_struct%enuclr_qmqm = quick_qm_struct%Ecore * AU_TO_EV 
            
    call dipole
    
    call zmake
    
    do i=1,natom
        scf_mchg(i)=quick_qm_struct%Mulliken(I)
    enddo
    
    
    ! now calculate gradient
    
    do j=1,natom
        do k=1,3
            quick_qm_struct%gradient((j-1)*3+K)=0d0
        enddo
    enddo
  
    if (analgrad) then
           !            if (UNRST) then
           !                if (HF) call uhfgrad
           !                if (DFT) call uDFTgrad
           !                if (SEDFT) call uSEDFTgrad
           !            else
           if (HF) then
                if (bMPI) then
                    call mpi_hfgrad
                else
                    call hfgrad
                endif
           endif
           !                if (DFT) call DFTgrad
           !                if (SEDFT) call SEDFTgrad
           !            endif

    endif
        
    
    natom=natomsave

    call cpu_time(timer_end%TIniGuess)

   return
end subroutine qm2_quick_energy

!-------------------------------
! Read mol info from Amber
!-------------------------------
subroutine read_AMBER_interface
!
! convert atom information from AMBER to quick including atom coordinates and atom number.
!
! This routine is called from qm2_quick_energy

    use allmod
    use qmmm_module, only :qmmm_struct
    implicit none
    
    integer i,j
        
    ! Mol coordinates
    do j=1,qmmm_struct%nquant_nlink
        do i=1,3
            xyz(i,j)=qmmm_struct%qm_coords(i,j)/bohr
        enddo
    enddo
    
    if (quick_first_call) then
    ! atom charge, type and total electon number
    do i=1,qmmm_struct%nquant_nlink
        chg(i)=qmmm_struct%iqm_atomic_numbers(i)
        iattype(i)=qmmm_struct%iqm_atomic_numbers(i)
        nelec=nelec+chg(i)
    enddo
    endif
end subroutine read_AMBER_interface

!--------------------------------
! read charge info from AMBER to quick
!--------------------------------
subroutine read_AMBER_charge
!
! read MM atom from AMBER and treat them as external point charge in QM calculation
!

    use allmod
    use qmmm_module, only: qmmm_struct
    
    implicit none
    
    integer i,j
    
    if (quick_first_call) then
        nextatom=qmmm_struct%qm_mm_pairs
        allocate(extchg(nextatom),extxyz(3,nextatom))
    endif
    
    do j=1,nextatom
        do i=1,3
            extxyz(i,j) = qmmm_struct%qm_xcrd(i,j)/bohr
        enddo
    enddo

    do j=1,nextatom
        extchg(j) = qmmm_struct%qm_xcrd(4,j)
    enddo    
    
    call flush(6)
end subroutine read_AMBER_charge


!--------------------------------------------
! connect AMBER namelist with quick
!--------------------------------------------
subroutine read_AMBER_job
!
! read AMBER job specification and pass them to quick
! and set some default value if not specified.
!
    use allmod
    use qmmm_module, only : qmmm_nml, qmmm_struct, qmmm_mpi
    
    implicit none
    integer ibasisstart,ibasisend
    character(len=10):: basisname

    if (qmmm_mpi%commqmmm_master) then
    
    ibasisstart = 1
    ibasisend = 80
    call rdword(basisdir,ibasisstart,ibasisend)
    
    ecp       = .false.
    custecp   = .false.

    acutoff   = 1.0d0/(10.0d0**7.0d0)
    MAXDIISSCF= 10        ! Max DIIS SCF cycles
    NCYC      = 1000      ! Max delta matrix method
    signif    = 1.d-10
    tol       = 1.d-10
    iscf      = 30
    UNRST     = .false. ! restricted or unrestricted
    debug     = .false. ! if debug?
    readdmx   = .false. ! read density matrix
    zmat      = .false. ! if do zmat conversion?
    writepmat = .false. ! write density matrix
! AG 03/05/2007
    itolecp   = 0

    ! Turn off some advanced option for quick
    bPDB      = .false.
    iMFCC     = 1
    SEDFT     = .false.
    PBSOL     = .false.
    core      = .false.
    opt       = .false.
    iFMM      = 1

    ! keep this two option on
    printEnergy = .true. ! print energy for every cycle
    isad      = 0        ! SAD initial guess
    
    basisname=qmmm_nml%basis
    
    call upcase(basisname,10)
    basisfilename=basisdir(1:ibasisend) // '/' // trim(basisname)
    call flush(6)
    
    
    ! Now connect qmmm namelist from AMBER
    ! Theory options. now it takes MP2 HF and DFT job
    MP2 = qmmm_nml%qmtheory%MP2
    HF  = qmmm_nml%qmtheory%HF
    DFT = qmmm_nml%qmtheory%DFT

    ! fetch molecule infomation from AMBER namelist
    molchg = qmmm_nml%qmcharge  ! molecule charge
    imult = qmmm_nml%spin       ! spin
    if (imult/=1) UNRST=.true.
    iscf = qmmm_nml%itrmax      ! Max scf cycles
    analgrad = qmmm_nml%qmqm_analyt     ! flag of analytical gradient
    
    ! SCF convergence criteria
    if (qmmm_nml%tight_p_conv) then
        pmaxrms = qmmm_nml%scfconv
    else
        pmaxrms = 0.05*sqrt(qmmm_nml%scfconv)
    endif
    
    ! flag to indicate external charges
    if (qmmm_struct%qm_mm_pairs > 0) extcharges = .true.
    
    if(pmaxrms.lt.0.0001d0)then
      integralCutoff=1.0d0/(10.0d0**7.0d0)
      quick_method%primLimit=1.0d0/(10.0d0**7.0d0)
    endif
    
    endif

    !-------------------MPI/ALL NODES---------------------------------------
    if (bMPI) then
      ! communicates nodes and pass job specs to all nodes
      call mpi_setup_job()
    endif
    !-------------------MPI/ALL NODES---------------------------------------

    if (qmmm_mpi%commqmmm_master .and. qmmm_nml%verbosity > 3) then
        write(6,'(" QMMM QUICK: QM CHARGE = ",i4," QM SPIN = ",i4," DM CONV = ",f16.9," SCF CYC = ",i6)') &
            molchg,imult,pmaxrms,iscf
        write(6,'(" QMMM QUICK: BASIS SET = ",a)') basisfilename
    endif
    
    
end subroutine read_AMBER_job

!
subroutine AMBER_interface_get_qm_forces(dxyzqm)
      use qmmm_module, only : qmmm_struct, qmmm_nml, qmmm_mpi
      use allmod
      
      implicit none
      double precision, intent(out) :: dxyzqm(3, qmmm_struct%nquant_nlink)
      double precision, parameter :: BOHRS_TO_A = 0.529177249D0
      double precision, parameter :: A_TO_BOHRS = 1.0d0 / BOHRS_TO_A
      double precision, parameter :: AU_TO_EV = 27.21d0, EV_TO_KCAL = 23.060362D0
      double precision, parameter :: AU_TO_KCAL = AU_TO_EV*EV_TO_KCAL
      
      integer i,j
      
      if (qmmm_mpi%commqmmm_master) then
        do i=1,qmmm_struct%nquant_nlink
           do j = 1,3
              dxyzqm(j,i) = quick_qm_struct%gradient((i-1)*3+j) * AU_TO_KCAL * A_TO_BOHRS
           end do
        end do
      endif
    
end subroutine AMBER_interface_get_qm_forces
