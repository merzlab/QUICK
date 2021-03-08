!
!	quick_method_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

module quick_method_module
    use quick_constants_module
    implicit none

    type quick_method_type

        ! the first section includes some elements of namelist is for the QM method that is going to use
        logical :: HF =  .false.       ! HF
        logical :: DFT =  .false.      ! DFT
        logical :: MP2 =  .false.      ! MP2

        !Madu Manathunga 05/30/2019 We should get rid of these functional
        !variables in future. Instead, we call funcationals from libxc
        logical :: B3LYP = .false.     ! B3LYP
        logical :: BLYP = .false.      ! BLYP
        logical :: BPW91 = .false.     ! BPW91
        logical :: MPW91LYP = .false.  ! MPW91LYP
        logical :: MPW91PW91 = .false. ! MPW91PW91
        logical :: SEDFT = .false.     ! Semi-Empirical DFT

        logical :: PBSOL = .false.     ! PB Solvent
        logical :: UNRST =  .false.    ! Unrestricted

        ! the second section includes some advanced option
        logical :: debug =  .false.    ! debug mode
        logical :: nodirect = .false.  ! conventional scf
        logical :: readDMX =  .false.  ! flag to read density matrix
        logical :: readSAD = .false.   ! flag to read SAD guess
        logical :: writePMat = .false. ! flag to write density matrix
        logical :: diisSCF =  .false.  ! DIIS SCF
        logical :: prtGap =  .false.   ! flag to print HOMO-LUMO gap
        logical :: opt =  .false.      ! optimization
        logical :: grad = .false.      ! if calculate gradient
        logical :: analGrad =  .false. ! Analytical Gradient
        logical :: analHess =  .false. ! Analytical Hessian Matrix
        logical :: diisOpt =  .false.  ! DIIS Optimization
        logical :: core =  .false.     ! Add core
        logical :: annil =  .false.    ! Annil Spin Contamination
        logical :: freq =  .false.     ! Frenquency calculation
        logical :: zmat = .false.      ! Z-matrix
        logical :: dipole = .false.    ! Dipole Momenta
        logical :: printEnergy = .true.! Print Energy each cycle, since it's cheap but useful, set it's true for default.
        logical :: fFunXiao            ! If f orbitial is contained
        logical :: calcDens = .false.  ! calculate density
        logical :: calcDensLap = .false.
                                       ! calculate density lap
        double precision :: gridSpacing = 0.1d0
                                       ! Density file gridspacing
        double precision :: lapGridSpacing = 0.1d0
                                       ! Density lapcacian file gridspacing

        logical :: PDB = .false.       ! PDB input
        logical :: extCharges = .false.! external charge


        ! those methods are mostly for research use
        logical :: FMM = .false.       ! Fast Multipole
        logical :: DIVCON = .false.    ! Div&Con
        integer :: ifragbasis = 1      ! =2.residue basis,=1.atom basis(DEFUALT),=3 non-h atom basis

        ! this is DFT grid
        integer :: iSG = 1             ! =0. SG0, =1. SG1(DEFAULT)

        ! Initial guess part
        logical :: SAD = .true.        ! SAD initial guess(defualt
        logical :: MFCC = .false.      ! MFCC

        ! this part is about ECP
        logical :: ecp                 ! ECP
        logical :: custECP             ! Custom ECP

        ! max cycles for scf or opt
        integer :: iscf = 200          ! max scf cycles
        integer :: iopt = 0            ! max opt cycles

        ! Maximum number of DIIS error vectors for scf convergence.
        integer :: maxdiisscf = 10

        ! start cycle for delta density cycle
        integer :: ncyc =1000

        ! following are some cutoff criteria
        double precision :: coreIntegralCutoff = 1.0d-12 ! cutoff for 1e integral prescreening
        double precision :: integralCutoff = 1.0d-7   ! integral cutoff
        double precision :: leastIntegralCutoff = LEASTCUTOFF  ! the smallest cutoff
        double precision :: maxIntegralCutoff = 1.0d-12
        double precision :: primLimit      = 1.0d-7   ! prime cutoff
        double precision :: gradCutoff     = 1.0d-7   ! gradient cutoff
        double precision :: DMCutoff       = 1.0d-10  ! density matrix cutoff
        !tol
        double precision :: pmaxrms        = 1.0d-4   ! density matrix convergence criteria
        double precision :: aCutoff        = 1.0d-7   ! 2e cutoff
        double precision :: basisCufoff    = 1.0d-10  ! basis set cutoff
        !signif

        ! following are some gradient cutoff criteria
        double precision :: stepMax        = .1d0/0.529177249d0
                                                      ! max change of one step
        double precision :: geoMaxCrt      = .0018d0  ! max geometry change
        double precision :: gRMSCrt        = .0012d0  ! geometry rms change
        double precision :: gradMaxCrt     = .001d0 ! max gradient change
        double precision :: gNormCrt       = .00030d0 ! gradient normalization
        double precision :: EChange        = 1.0d-6   ! Energy change

        !Madu Manathunga 05/30/2019
        !Following variables facilitates the use of libxc functionals
        logical :: uselibxc = .false.
        integer :: xc_polarization = 0
        !Following holds functional ids. Currently only holds two functionals.
        integer, dimension(10) :: functional_id
        double precision :: x_hybrid_coeff  = 1.0d0 !Amount of exchange contribution. 1.0 for HF.
        integer :: nof_functionals = 0

#if defined CUDA || defined CUDA_MPIV
        logical :: bCUDA                ! if CUDA is used here
#endif

    end type quick_method_type

    type (quick_method_type),save :: quick_method

    interface print
        module procedure print_quick_method
    end interface print

    interface read
        module procedure read_quick_method
    end interface read

#ifdef MPIV
    interface broadcast
        module procedure broadcast_quick_method
    end interface Broadcast
#endif

    interface init
        module procedure init_quick_method
    end interface init

    interface check
        module procedure check_quick_method
    end interface check

    contains
#ifdef MPIV
        !------------------------
        ! Broadcast quick_method
        !------------------------
        subroutine broadcast_quick_method(self)
            use quick_MPI_module
            implicit none

            type(quick_method_type) self

            include 'mpif.h'

            call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%HF,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%DFT,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%MP2,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%B3LYP,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%BLYP,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%BPW91,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%MPW91LYP,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%MPW91PW91,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%PBSOL,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%UNRST,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%debug,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%nodirect,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%readDMX,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%diisSCF,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%prtGap,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%opt,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%grad,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%analGrad,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%analHess,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%diisOpt,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%core,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%annil,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%freq,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%SEDFT,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%Zmat,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%dipole,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%ecp,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%custECP,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%printEnergy,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%fFunXiao,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%calcDens,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%calcDensLap,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%gridspacing,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%lapGridSpacing,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%writePMat,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%extCharges,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%PDB,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%SAD,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%FMM,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%DIVCON,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%MFCC,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%ifragbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%iSG,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%iscf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%iopt,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%maxdiisscf,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%ncyc,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%integralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%leastIntegralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%maxIntegralCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%primLimit,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%gradCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%DMCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%pmaxrms,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%aCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%basisCufoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%stepMax,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%geoMaxCrt,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%gRMSCrt,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%gradMaxCrt,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%gNormCrt,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

            !mpi variables for libxc implementation
            call MPI_BCAST(self%uselibxc,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%functional_id,shape(self%functional_id),mpi_integer,0,MPI_COMM_WORLD,mpierror)
            call MPI_BCAST(self%x_hybrid_coeff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

        end subroutine broadcast_quick_method


#endif
        !------------------------
        ! print quick_method
        !------------------------
        subroutine print_quick_method(self,io)
            use xc_f90_types_m
            use xc_f90_lib_m
            implicit none
            integer io
            type(quick_method_type) self

            !libxc variables
            type(xc_f90_pointer_t) :: xc_func
            type(xc_f90_pointer_t) :: xc_info
            character(len=120) :: f_name, f_kind, f_family
            integer :: vmajor, vminor, vmicro, f_id

            if (io.ne.0) then
            write(io,'(" ============== JOB CARD =============")')
            if (self%HF) then
                write(io,'(" METHOD = HATREE FOCK")')
            else if (self%MP2) then
                write(io,'(" METHOD = SECOND ORDER PERTURBATION THEORY")')
            else if (self%DFT) then
                write(io,'(" METHOD = DENSITY FUNCTIONAL THEORY")')

                if(self%uselibxc) then
                     call xc_f90_version(vmajor, vminor, vmicro)
                     write(io,'(" USING LIBXC VERSION: ",I1,".",I1,".",I1)') vmajor, &
                     vminor, vmicro
                     write(io,'(" FUNCTIONAL INFORMATION:")')
                !Get functional information from libxc and print

                  do f_id=1, self%nof_functionals
                     !Initiate libx function; but we only care unpolarized at
                     !this point
                     if(self%xc_polarization >0 ) then
                        call xc_f90_func_init(xc_func, xc_info, self%functional_id(f_id),XC_POLARIZED)
                     else
                        call xc_f90_func_init(xc_func, xc_info,self%functional_id(f_id),XC_UNPOLARIZED)
                     endif
                     call xc_f90_info_name(xc_info, f_name)

                     select case(xc_f90_info_kind(xc_info))
                        case (XC_EXCHANGE)
                          write(f_kind, '(a)') 'EXCHANGE'
                        case (XC_CORRELATION)
                          write(f_kind, '(a)') 'CORRELATION'
                        case (XC_EXCHANGE_CORRELATION)
                          write(f_kind, '(a)') 'EXCHANGE CORRELATION'
                        case (XC_KINETIC)
                          write(f_kind, '(a)') 'KINETIC'
                        case default
                          write (f_kind, '(a)') 'UNKNOWN'
                     end select

                    select case (xc_f90_info_family(xc_info))
                        case (XC_FAMILY_LDA);
                          write(f_family,'(a)') "LDA"
                        case (XC_FAMILY_GGA);
                          write(f_family,'(a)') "GGA"
                        case (XC_FAMILY_HYB_GGA);
                          write(f_family,'(a)') "HYBRID GGA"
                        case (XC_FAMILY_MGGA);
                          write(f_family,'(a)') "MGGA"
                        case (XC_FAMILY_HYB_MGGA);
                          write(f_family,'(a)') "HYBRID MGGA"
                        case default;
                          write(f_family,'(a)') "UNKNOWN"
                        end select

                     write(io,'(" NAME = ",a, " FAMILY = ",a, " KIND = ",a)') &
                     trim(f_name), trim(f_family), trim(f_kind)
                     call xc_f90_func_end(xc_func)
                  enddo

                elseif (self%B3LYP) then
                    write(io,'(" DENSITY FUNCTIONAL = B3LYP")')
                elseif(self%BLYP) then
                    write(io,'(" DENSITY FUNCTIONAL = BLYP")')
                elseif(self%BPW91) then
                    write(io,'(" DENSITY FUNCTIONAL = BPW91")')
                elseif(self%MPW91LYP) then
                    write(io,'(" DENSITY FUNCTIONAL = MPW91LYP")')
                elseif(self%MPW91PW91) then
                    write(io,'(" DENSITY FUNCTIONAL = MPW91PW91")')
                endif
            else if (self%SEDFT) then
                write(io,'(" METHOD = SEMI-EMPIRICAL DENSTITY FUNCTIONAL THEORY")')
            endif

if (self%nodirect) then
write(io,'(" SAVE 2E INT TO DISK ")')
else
write(io,'(" DIRECT SCF ")')
endif

            if (self%PDB) write(io,'(" PDB INPUT ")')
            if (self%MFCC) write(io,'(" MFCC INITIAL GUESS ")')
            if (self%SAD)  write(io,'(" SAD INITAL GUESS ")')

            if (self%FMM)  write(io,'(" FAST MULTIPOLE METHOD = TRUE ")')

            if (self%UNRST)     write(io,'(" UNRESTRICTED SYSTEM")')
            if (self%annil)     write(io,'(" ANNIHILATE SPIN CONTAMINAT")')

            if (self%PBSOL)     write(io,'(" SOLVATION MODEL = PB")')
            if (self%diisSCF)   write(io,'(" USE DIIS SCF")')
            if (self%prtGap)    write(io,'(" PRINT HOMO-LUMO GAP")')
            if (self%printEnergy) write(io,'(" PRINT ENERGY EVERY CYCLE")')

            if (self%readDMX)   write(io,'(" READ DENSITY MATRIX FROM FILE")')
            if (self%readSAD)   write(io,'(" READ SAD GUESS FROM FILE")')
            if (self%writePMat) write(io,'(" WRITE DENSITY MATRIX TO FILE")')

            if (self%zmat)      write(io,'(" Z-MATRIX CONSTRUCTION")')
            if (self%dipole)    write(io,'(" DIPOLE")')
            if (self%ecp)       write(io,'(" ECP BASIS SET")')
            if (self%custECP)   write(io,'(" CUSTOM ECP BASIS SET")')

            if (self%extCharges)write(io,'(" EXTERNAL CHARGES")')
            if (self%core)      write(io,'(" SUM INNER ELECTRONS INTO CORE")')
            if (self%debug)     write(io,'(" DEBUG MODE")')

            if (self%DFT) then
                if (self%iSG .eq. 0) write(io,'(" STANDARD GRID = SG0")')
                if (self%iSG .eq. 1) write(io,'(" STANDARD GRID = SG1")')
            endif

            if (self%opt) then
                write(io,'(" GEOMETRY OPTIMIZATION")',advance="no")
                if (self%diisOpt)   write(io,'(" USE DIIS FOR GEOMETRY OPTIMIZATION")')
                if (self%analGrad)  then
                    write(io,'(" ANALYTICAL GRADIENT")')
                else
                    write(io,'(" NUMERICAL GRADIENT")')
                endif

                if (self%freq) then
                    write(io,'(" FREQENCY CALCULATION")',advance="no")
                    if (self%analHess)  then
                        write(io,'(" ANALYTICAL HESSIAN MATRIX")')
                    else
                        write(io,'(" NUMERICAL HESSIAN MATRIX")')
                    endif
                endif
                write (io,'(" REQUEST CRITERIA FOR GEOMETRY CONVERGENCE: ")')
                write (io,'("      MAX ALLOWED GEO CHANGE  = ",E10.3)') self%stepMax
                write (io,'("      MAX GEOMETRY CHANGE     = ",E10.3)') self%geoMaxCrt
                write (io,'("      GEOMETRY CHANGE RMS     = ",E10.3)') self%gRMSCrt
                write (io,'("      MAX GRADIENT CHANGE     = ",E10.3)') self%gradMaxCrt
                write (io,'("      GRADIENT NORMALIZATION  = ",E10.3)') self%gNormCrt
            endif

            if (self%grad)      write(io,'(" GRADIENT CALCULATION")')

            if (self%DIVCON) then
                write(io,'(" DIV & CON METHOD")',advance="no")
                if (self%ifragbasis .eq. 1) then
                    write(io,'(" = DIV AND CON ON ATOM BASIS")')
                elseif (self%ifragbasis .eq. 2) then
                    write(iO,'(" = DIV AND CON ON RESIDUE BASIS")')
                elseif (self%ifragbasis .eq. 3) then
                    write(io,'(" = DIV AND CON ON NON HYDROGEN ATOM BASIS")')
                else
                    write(io,'(" = DIV AND CON ON ATOM BASIS (BY DEFAULT)")')
                endif
            endif

            ! computing cycles
            write(io,'(" MAX SCF CYCLES = ",i6)') self%iscf
            if (self%diisSCF) write (io,'(" MAX DIIS CYCLES = ",I4)') self%maxdiisscf
            write (io,'(" DELTA DENSITY START CYCLE = ",I4)') self%ncyc

            ! cutoff size
            write (io,'(" COMPUTATIONAL CUTOFF: ")')
            write (io,'("      TWO-e INTEGRAL   = ",E10.3)') self%acutoff
            write (io,'("      BASIS SET PRIME  = ",E10.3)') self%primLimit
            write (io,'("      MATRIX ELEMENTS  = ",E10.3)') self%DMCutoff
            write (io,'("      BASIS FUNCTION   = ",E10.3)') self%basisCufoff
            if (self%grad) then
                write (io,'("      GRADIENT CUTOFF  = ",E10.3)') self%gradCutoff
            endif
            write (io,'(" DENSITY MATRIX MAXIMUM RMS FOR CONVERGENCE  = ",E10.3)') self%pmaxrms

            if (self%calcDens) write (io,'(" GENERATE ELECTRON DENSITY FILE WITH GRIDSPACING ",E12.6, "A")') &
                                self%gridspacing
            if (self%calcDensLap) write (io,'(" GENERATE ELECTRON DENSITY LAPLACIAN FILE WITH GRIDSPACING ", &
                                E12.6, "A")') self%lapgridspacing
            endif !io.ne.0

        end subroutine print_quick_method



        !------------------------
        ! read quick_method
        !------------------------
        subroutine read_quick_method(self,keywd)
            implicit none
            character(len=200) :: keyWD
            character(len=200) :: tempstring
            integer :: itemp,rdinml,i,j
            double precision :: rdnml
            type (quick_method_type) self

            call upcase(keyWD,200)
            if (index(keyWD,'PDB').ne. 0)       self%PDB=.true.
            if (index(keyWD,'MFCC').ne.0)       self%MFCC=.true.
            if (index(keyWD,'FMM').ne.0)        self%FMM=.true.
            if (index(keyWD,'MP2').ne.0)        self%MP2=.true.
            if (index(keyWD,'HF').ne.0)         self%HF=.true.
            if (index(keyWD,'DFT').ne.0)        self%DFT=.true.
            if (index(keyWD,'SEDFT').ne.0)      self%SEDFT=.true.
            if (index(keyWD,'PBSOL').ne.0)      self%PBSOL=.true.
            if (index(keyWD,'ANNIHILATE').ne.0) self%annil=.true.
            if (index(keyWD,'BPW91').ne.0)      self%BPW91=.true.
            if (index(keyWD,'MPW91LYP').ne.0)   self%MPW91LYP=.true.
            if (index(keyWD,'MPW91PW91').ne.0)  self%MPW91PW91=.true.

            if (index(keyWD,'CORE').ne.0)       self%CORE=.true.
            if (index(keyWD,'OPT').ne.0) then
                self%opt=.true.
                self%grad=.true.
            endif
            if (index(keyWD,'GRADIENT').ne.0) self%grad=.true.

            !Read dft functional keywords and set variable values
            if (index(keyWD,'LIBXC').ne.0) then
                self%uselibxc=.true.
                call set_libxc_func_info(keyWD, self)
            elseif(index(keyWD,'B3LYP').ne.0) then
                self%B3LYP=.true.
                self%x_hybrid_coeff =0.2d0
            elseif(index(keyWD,'BLYP').ne.0) then
                self%BLYP=.true.
                self%x_hybrid_coeff =0.0d0
            endif

            if(self%B3LYP .or. self%BLYP .or. self%BPW91 .or. self%MPW91PW91 .or. &
                self%MPW91LYP .or. self%uselibxc) self%DFT=.true.

            if (index(keyWD,'DIIS-OPTIMIZE').ne.0)self%diisOpt=.true.
            if (index(keyWD,'GAP').ne.0)        self%prtGap=.true.
            if (index(keyWD,'GRAD').ne.0)       self%analGrad=.true.
            if (index(keyWD,'HESSIAN').ne.0)    self%analHess=.true.
            if (index(keyWD,'FREQ').ne.0)       self%freq=.true.
            if (index(keywd,'DEBUG').ne.0)      self%debug=.true.
            if (index(keyWD,'READ').ne.0)       self%readDMX=.true.
            if (index(keyWD,'READSAD').ne.0)    self%readSAD=.true.
            if (index(keyWD,'ZMAKE').ne.0)      self%zmat=.true.
            if (index(keyWD,'DIPOLE').ne.0)      self%dipole=.true.
            if (index(keyWD,'WRITE').ne.0)      self%writePMat=.true.
            if (index(keyWD,'EXTCHARGES').ne.0) self%EXTCHARGES=.true.
            if (index(keyWD,'FORCE').ne.0)      self%grad=.true.

            if (index(keyWD,'NODIRECT').ne.0)      self%NODIRECT=.true.

            if (index(keywd,'DIVCON') .ne. 0) then
                self%divcon = .true.
                if (index(keywd,'ATOMBASIS') /= 0) then
                    self%ifragbasis=1
                else if (index(keywd,'RESIDUEBASIS') /= 0) then
                    self%ifragbasis=2
                else if (index(keywd,'NHAB') /= 0) then
                    self%ifragbasis=3
                else
                    self%ifragbasis=1
                endif
            endif

            if (self%DFT) then
                if (index(keyWD,'SG0').ne.0) then
                    self%iSG=0
                else
                    self%iSG=1
                endif
            endif

            self%printEnergy=.true.
            self%sad=.true.
            self%diisSCF=.true.

            if (index(keyWD,'ECP').ne.0)  then
                self%ECP=.true.
                i=index(keywd,'ECP=')
                call rdword(keywd,i,j)
                if (keywd(i+4:j).eq.'CUSTOM')  self%custECP=.true.
            endif

            if (index(keyWD,'USEDFT').ne.0) then
                self%SEDFT=.true.
                self%UNRST=.true.
            endif

            if (index(keyWD,'UHF').ne.0) then
                self%HF=.true.
                self%UNRST=.true.
            endif

            if (index(keyWD,'UDFT').ne.0) then
                self%DFT=.true.
                self%UNRST=.true.
            endif

            ! Density map
            ! JOHN FAVER 12/2008
            if (index(keywd,'DENSITYMAP') /= 0) then
                self%gridspacing = rdnml(keywd,'DENSITYMAP')
                self%calcdens = .true.
            endif

            ! Density lapmap
            if (index(keywd,'DENSITYLAPMAP') /= 0) then
                self%lapgridspacing= rdnml(keywd,'DENSITYLAPMAP')
                self%calcdenslap = .true.
            endif

            ! opt cycls
            if (index(keywd,'OPTIMIZE=') /= 0) self%iopt = rdinml(keywd,'OPTIMIZE')

            ! scf cycles
            if (index(keywd,'SCF=') /= 0) self%iscf = rdinml(keywd,'SCF')

            ! DM Max RMS
            if (index(keywd,'DENSERMS=') /= 0) self%pmaxrms = rdnml(keywd,'DENSERMS')

            ! 2e-cutoff
            if (index(keywd,'CUTOFF=') /= 0) then
                self%acutoff = rdnml(keywd,'CUTOFF')
                self%integralCutoff=self%acutoff !min(self%integralCutoff,self%acutoff)
                self%primLimit=1E-20 !self%acutoff*0.001 !min(self%integralCutoff,self%acutoff)
                self%gradCutoff=self%acutoff
            endif

            ! Max DIIS cycles
            if (index(keywd,'MAXDIIS=') /= 0) self%maxdiisscf=rdinml(keywd,'MAXDIIS')

            ! Delta DM Cycle Start
            if (index(keywd,'NCYC=') /= 0) self%ncyc = rdinml(keywd,'NCYC')

            ! DM cutoff
            if (index(keywd,'MATRIXZERO=') /= 0) self%DMCutoff = rdnml(keywd,'MAXTRIXZERO')

            ! Basis cutoff
            if (index(keywd,'BASISZERO=') /= 0) then
                itemp=rdinml(keywd,'BASISZERO')
                self%basisCufoff=10.d0**(-1.d0*itemp)
            endif

        end subroutine read_quick_method

        !------------------------
        ! initial quick_method
        !------------------------
        subroutine init_quick_method(self)

            implicit none
            type(quick_method_type) self

            self%HF =  .false.       ! HF
            self%DFT =  .false.      ! DFT
            self%MP2 =  .false.      ! MP2
            self%B3LYP = .false.     ! B3LYP
            self%BLYP = .false.      ! BLYP
            self%BPW91 = .false.     ! BPW91
            self%MPW91LYP = .false.  ! MPW91LYP
            self%MPW91PW91 = .false. ! MPW91PW91
            self%SEDFT = .false.     ! Semi-Empirical DFT
            self%PBSOL = .false.     ! PB Solvent
            self%UNRST =  .false.    ! Unrestricted

            self%debug =  .false.    ! debug mode
            self%nodirect = .false.  ! conventional SCF
            self%readDMX =  .false.  ! flag to read density matrix
            self%readSAD =  .false.  ! flag to read sad guess
            self%diisSCF =  .false.  ! DIIS SCF
            self%prtGap =  .false.   ! flag to print HOMO-LUMO gap
            self%opt =  .false.      ! optimization
            self%grad =  .false.     ! gradient
            self%analGrad =  .false. ! Analytical Gradient
            self%analHess =  .false. ! Analytical Hessian Matrix

            self%diisOpt =  .false.  ! DIIS Optimization
            self%core =  .false.     !
            self%annil =  .false.    !
            self%freq =  .false.     ! Frenquency calculation
            self%zmat = .false.      ! Z-matrix
            self%dipole = .false.    ! Dipole
            self%ecp = .false.       ! ECP
            self%custECP = .false.   ! Custom ECP
            self%printEnergy = .true.! Print Energy each cycle
            self%fFunXiao = .false.            ! If f orbitial is contained
            self%calcDens = .false.    ! calculate density
            self%calcDensLap = .false. ! calculate density lap
            self%writePMat = .false.   ! Output density matrix
            self%extCharges = .false.  ! external charge
            self%PDB = .false.         ! PDB input
            self%SAD = .true.          ! SAD initial guess
            self%FMM = .false.         ! Fast Multipole
            self%DIVCON = .false.      ! Div&Con

            self%ifragbasis = 1        ! =2.residue basis,=1.atom basis(DEFUALT),=3 non-h atom basis
            self%iSG = 1               ! =0. SG0, =1. SG1(DEFAULT)
            self%MFCC = .false.        ! MFCC

            self%iscf = 200
            self%maxdiisscf = 10
            self%iopt = 0
            self%ncyc = 1000

            self%integralCutoff = 1.0d-7   ! integral cutoff
            self%leastIntegralCutoff = LEASTCUTOFF
                                           ! smallest integral cutoff, used in conventional SCF
            self%maxIntegralCutoff = 1.0d-12
                                           ! smallest integral cutoff, used in conventional SCF
            self%primLimit      = 1.0d-7   ! prime cutoff
            self%gradCutoff     = 1.0d-7   ! gradient cutoff
            self%DMCutoff       = 1.0d-10  ! density matrix cutoff

            self%pmaxrms        = 1.0d-4   ! density matrix convergence criteria
            self%aCutoff        = 1.0d-7   ! 2e cutoff
            self%basisCufoff    = 1.0d-10  ! basis set cutoff

            self%stepMax        = .1d0/0.529177249d0
                                           ! max change of one step
            self%geoMaxCrt      = .0018d0  ! max geometry change
            self%gRMSCrt        = .0012d0  ! geometry rms change
            self%gradMaxCrt     = .00100d0 ! max gradient change
            self%gNormCrt       = .00030d0 ! gradient normalization
            self%EChange        = 1.0d-6
            self%gridSpacing    = 0.1
            self%lapgridspacing = 0.1

            !Initialize libxc variables
            self%x_hybrid_coeff = 1.0d0 !Set the default exchange value
            self%uselibxc = .false.
            self%xc_polarization = 0
            self%nof_functionals = 0

#if defined CUDA || defined CUDA_MPIV
            self%bCUDA  = .false.
#endif

        end subroutine init_quick_method

        !------------------------
        ! check quick_method
        !------------------------
        subroutine check_quick_method(self,io)
            implicit none
            type(quick_method_type) self
            integer io

            ! If MP2, then set HF as default
            if (self%MP2) then
                self%HF = .true.
                self%DFT = .false.
            endif

            if (self%opt) then
                self%grad = .true.
            endif

            if(self%pmaxrms.lt.0.0001d0)then
                !self%integralCutoff=min(1.0d-7,self%integralCutoff)
                !self%primLimit=min(1.0d-7,self%primLimit)
            endif

            ! OPT not available for MP2
            if (self%MP2 .and. self%OPT) then
                call PrtWrn(io,"GEOMETRY OPTIMIZAION IS NOT AVAILABLE WITH MP2, WILL DO MP2 SINGLE POINT ONLY")
                self%OPT = .false.
            endif

            ! OPT not available for BLYP and B3LYP DFT methods
            if(self%DFT.and. self%OPT .and. (.not. (self%BLYP .or. self%B3LYP) .and. .not.(self%uselibxc)))then
                call PrtWrn(io,"GEOMETRY OPTIMIZATION is only available with HF, DFT/BLYP, DFT/B3LYP" )
                self%OPT = .false.
            endif

        end subroutine check_quick_method

        subroutine obtain_leastIntCutoff(self)
            use quick_constants_module
            implicit none
            type(quick_method_type) self


            self%leastIntegralCutoff = LEASTCUTOFF

            if (self%pmaxrms .gt. 1.0d0/10.0d0**9.5) self%leastIntegralCutoff = TEN_TO_MINUS5
            if (self%pmaxrms .gt. 1.0d0/10.0d0**8.5) self%leastIntegralCutoff = TEN_TO_MINUS4
            if (self%pmaxrms .gt. 1.0d0/10.0d0**7.5) self%leastIntegralCutoff = TEN_TO_MINUS3

            self%maxIntegralCutoff  = 1.0d-12


            if (self%pmaxrms .gt. 1.0d0/10.0d0**9.5) self%maxIntegralCutoff = TEN_TO_MINUS10
            if (self%pmaxrms .gt. 1.0d0/10.0d0**8.5) self%maxIntegralCutoff = TEN_TO_MINUS9
            if (self%pmaxrms .gt. 1.0d0/10.0d0**7.5) self%maxIntegralCutoff = TEN_TO_MINUS8
            if (self%integralCutoff .le. self%maxIntegralCutoff) self%maxIntegralCutoff=self%integralCutoff

        end subroutine obtain_leastIntCutoff


        subroutine adjust_Cutoff(PRMS,PCHANGE,self)
            use quick_constants_module
            implicit none
            double precision prms,pchange
            type(quick_method_type) self

             if(PRMS.le.TEN_TO_MINUS5 .and. self%integralCutoff.gt.1.0d0/(10.0d0**8.5d0))then
                self%integralCutoff=TEN_TO_MINUS9
                self%primLimit=min(self%integralCutoff,self%primLimit)
             endif

            if(PRMS.le.TEN_TO_MINUS6 .and. self%integralCutoff.gt.1.0d0/(10.0d0**9.5d0))then
                self%integralCutoff=TEN_TO_MINUS10
                self%primLimit=min(self%integralCutoff,self%primLimit)
            endif

            if(PRMS.le.TEN_TO_MINUS7 .and.quick_method%integralCutoff.gt.1.0d0/(10.0d0**10.5d0))then
            quick_method%integralCutoff=TEN_TO_MINUS11
            quick_method%primLimit=min(quick_method%integralCutoff,self%primLimit)
            endif

        end subroutine adjust_Cutoff

        !Madu Manathunga 05/31/2019
        !This subroutine set the functional id and  x_hybrid_coeff
        subroutine set_libxc_func_info(f_keywd, self)
           use xc_f90_types_m
           use xc_f90_lib_m
           implicit none
           character(len=200) :: f_keywd
           type(quick_method_type) self
           integer :: f_id, nof_f, istart, iend, imid, f_nlen, usf1_nlen, usf2_nlen
           type(xc_f90_pointer_t) :: xc_func
           type(xc_f90_pointer_t) :: xc_info
           double precision :: x_hyb_coeff
           character(len=256) :: functional_name

        !We now set the functional ids corresponding to each functional.
        !Note that these ids are coming from libxc. One should obtain them
        !by looking at the functional name (eg: PBE0) in libxc manual and
        !using xc_f90_functional_get_number() function in libxc library.

        imid=0
        if (index(f_keywd,'LIBXC=') /= 0) then
           istart = index(f_keywd,'LIBXC=')
           call rdword(f_keywd,istart,iend)
           imid=index(f_keywd(istart+6:iend),',')

           if(imid>0) then
              usf1_nlen=imid-1
              usf2_nlen = iend-(istart+6+usf1_nlen)
           else
              usf1_nlen=iend-(istart+6)+1
           endif
           !write(*,*) "Reading LIBXC key words: ",f_keywd(istart+6:iend), imid, usf1_nlen, usf2_nlen
        endif

        nof_f=0
        do f_id=0,1000
           call xc_f90_functional_get_name(f_id,functional_name)
           if((index(functional_name,'unknown') .eq. 0) &
            .and. (index(functional_name,'mgga') .eq. 0))  then
                functional_name=trim(functional_name)
                f_nlen=len(trim(functional_name))

                call upcase(functional_name,200)

                if((index(f_keywd,trim(functional_name)) .ne. 0) .and. ((usf1_nlen .eq. f_nlen) &
                .or. (usf2_nlen .eq. f_nlen))) then

                        nof_f=nof_f+1
                        if(self%xc_polarization > 0) then
                                call xc_f90_func_init(xc_func, xc_info, f_id, XC_POLARIZED)
                        else
                                call xc_f90_func_init(xc_func, xc_info, f_id, XC_UNPOLARIZED)
                        endif

                        self%functional_id(nof_f)=xc_f90_functional_get_number(functional_name)
                        call xc_f90_hyb_exx_coef(xc_func, self%x_hybrid_coeff)
                        call xc_f90_func_end(xc_func)
                endif
           endif
        enddo

        self%nof_functionals=nof_f

        end subroutine set_libxc_func_info
end module quick_method_module
