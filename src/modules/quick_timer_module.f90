!
!
!	quick_timer_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

module quick_timer_module
    implicit none

    integer TIMER_SIZE,TIMER_CUMER_SIZE

    ! MPI timer data type
    integer MPI_timer_cumer_type,MPI_timer_type
    parameter(TIMER_SIZE=33,TIMER_CUMER_SIZE=36)

    !timer type
    type quick_timer
        double precision:: TInitialize=0.0d0
        double precision:: TIniGuess=0.0
        double precision:: TTotal=0.0d0
        double precision:: TDiag=0.0d0
        double precision:: TMP2=0.0d0
        double precision:: TDII=0.0d0
        double precision:: TSCF=0.0d0
        double precision:: TOp=0.0d0
        double precision:: T1e=0.0d0
        double precision:: T1eS=0.0d0
        double precision:: T1eSD=0.0d0
        double precision:: T1eT=0.0d0
        double precision:: T1eTGrad=0.0d0
        double precision:: T1eV=0.0d0
        double precision:: T1eVGrad=0.0d0
        double precision:: T2e=0.0d0
        double precision:: T2eAll=0.0d0
        double precision:: TDip=0.0d0
        double precision:: TE=0.0d0
        double precision:: TEx=0.0d0
        double precision:: TGrad=0.0d0
        double precision:: TNucGrad=0.0d0
        double precision:: T1eGrad=0.0d0
        double precision:: T2eGrad=0.0d0
        double precision:: TExGrad=0.0d0
        double precision:: TDFTGrdGen=0.0d0 !Time to generate dft grid
        double precision:: TDFTGrdWt=0.0d0  !Time to compute grid weights
        double precision:: TDFTGrdOct=0.0d0 !Time to run octree algorithm
        double precision:: TDFTPrscrn=0.0d0 !Time to prescreen basis & primitive funtions
        double precision:: TDFTGrdPck=0.0d0 !Time to pack grid points
        double precision:: T2elb=0.0d0      !Time for eri load balancing in mgpu version
        double precision:: TEred=0.0d0      !Time for operator reduction in mpi/mgpu versions 
        double precision:: TGradred=0.0d0   !Time for gradient reductin in mpi/mgpu versions
        double precision:: Tcew=0.0d0       !Total time for cew potential contributions
        double precision:: TcewLri=0.0d0    !Time for computing long range integrals in cew
        double precision:: TcewLriGrad=0.0d0 !Time for computing long range integral gradients in cew 
        double precision:: TcewLriQuad=0.0d0 !Time for computing quadrature contribution in cew
        double precision:: TcewLriGradQuad=0.0d0 !Time for computing quadrature gradient contribution in cew
        double precision:: Tdisp=0.0d0      ! Time for computing dispersion correction
    end type quick_timer

    type quick_timer_cumer
        double precision:: TInitialize=0.0d0
        double precision:: TTotal=0.0d0
        double precision:: TDiag=0.0d0
        double precision:: TMP2=0.0d0
        double precision:: TDII=0.0d0
        double precision:: TSCF=0.0d0
        double precision:: TOp=0.0d0
        double precision:: T1e=0.0d0
        double precision:: T1eS=0.0d0
        double precision:: T1eSD=0.0d0
        double precision:: T1eT=0.0d0
        double precision:: T1eTGrad=0.0d0
        double precision:: T1eV=0.0d0
        double precision:: T1eVGrad=0.0d0
        double precision:: T2e=0.0d0
        double precision:: T2eAll=0.0d0
        double precision:: TE=0.0d0
        double precision:: TEx=0.0d0
        double precision:: TGrad=0.0d0
        double precision:: TNucGrad=0.0d0
        double precision:: T1eGrad=0.0d0
        double precision:: T2eGrad=0.0d0
        double precision:: TExGrad=0.0d0
        double precision:: TIniGuess=0.0d0  !Time for initial guess
        double precision:: TDFTGrdGen=0.0d0 !Time to generate dft grid
        double precision:: TDFTGrdWt=0.0d0  !Time to compute grid weights
        double precision:: TDFTGrdOct=0.0d0 !Time to run octree algorithm
        double precision:: TDFTPrscrn=0.0d0 !Time to prescreen basis & primitive funtions
        double precision:: TDFTGrdPck=0.0d0 !Time to pack grid points
        double precision:: TDip=0.0d0       !Time for calculating dipoles
        double precision:: TDFTlb=0.0d0     !Time for xc load balancing in mgpu version
        double precision:: TDFTrb=0.0d0     !Time for xc load re-balancing in mgpu version
        double precision:: TDFTpg=0.0d0     !Time for XC grid pruning
        double precision:: T2elb=0.0d0      !Time for eri load balancing in mgpu version
        double precision:: TEred=0.0d0      !Time for operator reduction in mpi/mgpu versions 
        double precision:: TGradred=0.0d0   !Time for gradient reductin in mpi/mgpu versions
        double precision:: Tcew=0.0d0       !Total time for cew potential contributions
        double precision:: TcewLri=0.0d0    !Time for computing long range integrals in cew
        double precision:: TcewLriGrad=0.0d0 !Time for computing long range integral gradients in cew 
        double precision:: TcewLriQuad=0.0d0 !Time for computing quadrature contribution in cew
        double precision:: TcewLriGradQuad=0.0d0 !Time for computing quadrature gradient contribution in cew
        double precision:: Tdisp=0.0d0      ! Time for computing dispersion correction
    end type quick_timer_cumer

    type (quick_timer),save:: timer_begin
    type (quick_timer),save:: timer_end
    type (quick_timer_cumer),save:: timer_cumer
    type (quick_timer_cumer),save:: MPI_timer_cumer

    contains

    !-----------------------
    ! Output time infos
    !-----------------------
    subroutine timer_output(io)
        use quick_mpi_module
        use quick_method_module
#ifdef MPIV
        use mpi
#endif
        implicit none
        integer i,IERROR,io
        double precision :: t_tot_dftop, t_tot_lb
        
#ifdef MPIV
        double precision :: tst2e(mpisize), tstxc(mpisize), tst2egrad(mpisize), tstxcgrad(mpisize)
        double precision :: tend2e(mpisize), tendxc(mpisize), tend2egrad(mpisize), tendxcgrad(mpisize)
        double precision :: t2e(mpisize), txc(mpisize), t2egrad(mpisize), txcgrad(mpisize)
#endif
        type (quick_timer) tmp_timer
        type (quick_timer_cumer) tmp_timer_cumer,max_timer_cumer

        !----------------------------------------------------
        ! For Master nodes or single process timing infomations
        !----------------------------------------------------

#if defined CUDA_MPIV || defined HIP_MPIV
    call get_mgpu_time()
#endif

        if (master) then
            call PrtAct(io,"Output Timing Information")
            write (io,'("------------- TIMING ---------------")')
            ! Initial Guess Timing
#ifdef DEBUGTIME
            write (io,'("| INITIALIZATION TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%TInitialize, &
                timer_cumer%TInitialize/(timer_end%TTotal-timer_begin%TTotal)*100
#endif
            write (io,'("| INITIAL GUESS TIME  =",F16.9,"( ",F5.2,"%)")') timer_cumer%TIniGuess, &
                timer_cumer%TIniGuess/(timer_end%TTotal-timer_begin%TTotal)*100

#ifdef DEBUGTIME
            ! Time to evaluate overlap 1e integrals
            write (io,'("| OVERLAP 1e INTEGRALS TIME      =",F16.9,"(",F5.2,"%)")') timer_cumer%T1eS, &
                timer_cumer%T1eS/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to diagnalize overlap 1e matrix
            write (io,'("| OVERLAP 1e DIAGONALIZATION TIME =",F16.9,"(",F5.2,"%)")') timer_cumer%T1eSD, &
                timer_cumer%T1eSD/(timer_end%TTotal-timer_begin%TTotal)*100
#endif

            if(quick_method%DFT) then
                t_tot_dftop = timer_cumer%TDFTGrdGen + timer_cumer%TDFTGrdWt + timer_cumer%TDFTGrdOct + timer_cumer%TDFTPrscrn &
                            + timer_cumer%TDFTGrdPck
                ! Total time for dft grid formation, pruning and prescreening
                write (io,'("| DFT GRID OPERATIONS =",F16.9,"( ",F5.2,"%)")') t_tot_dftop, &
                (t_tot_dftop)/(timer_end%TTotal-timer_begin%TTotal)*100
#ifdef DEBUGTIME
                ! Time for grid formation
                write (io,'("| ",6x,"TOTAL GRID FORMATION TIME   =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDFTGrdGen, &
                timer_cumer%TDFTGrdGen/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for computing grid weight
                write (io,'("| ",6x,"TOTAL GRID WEIGHT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDFTGrdWt, &
                timer_cumer%TDFTGrdWt/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for running octree algorithm
                write (io,'("| ",6x,"TOTAL OCTREE RUN TIME       =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDFTGrdOct, &
                timer_cumer%TDFTGrdOct/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for prescreening basis and primitive functions
                write (io,'("| ",6x,"TOTAL PRESCREENING TIME     =",F16.9,"( ",F5.2,"%)")')timer_cumer%TDFTPrscrn, &
                timer_cumer%TDFTPrscrn/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for packing points
                write (io,'("| ",6x,"TOTAL DATA PACKING TIME     =",F16.9,"( ",F5.2,"%)")')timer_cumer%TDFTGrdPck, &
                timer_cumer%TDFTGrdPck/(timer_end%TTotal-timer_begin%TTotal)*100
#endif
            endif
#if defined CUDA_MPIV || defined HIP_MPIV && DEBUGTIME
            t_tot_lb=timer_cumer%t2elb+timer_cumer%tdftlb+timer_cumer%tdftrb
            write (io,'("| TOTAL LOAD BALANCING TIME =",F16.9,"( ",F5.2,"%)")') t_tot_lb, &
            t_tot_lb/(timer_end%TTotal-timer_begin%TTotal)*100
#ifdef DEBUGTIME
            write (io,'("| ",6x,"2E LOAD BALANCING TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%t2elb, &
            timer_cumer%t2elb/(timer_end%TTotal-timer_begin%TTotal)*100
            if(quick_method%DFT) then
                write (io,'("| ",6x,"DFT LOAD BALANCING TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%tdftlb, &
                timer_cumer%tdftlb/(timer_end%TTotal-timer_begin%TTotal)*100
                if (quick_method%opt .or. quick_method%grad ) then
!                   write (io,'("| ",6x,"DFT GRID REPRUNING TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%tdftpg, &
!                   timer_cumer%tdftpg/(timer_end%TTotal-timer_begin%TTotal)*100
                   write (io,'("| ",6x,"DFT LOAD REBALANCING TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%tdftrb, &
                   timer_cumer%tdftrb/(timer_end%TTotal-timer_begin%TTotal)*100
                endif
            endif
#endif
#endif

            if(quick_method%edisp) then
                write (io,'("| DISPERSION CORRECTION TIME  =",F16.9,"( ",F5.2,"%)")') timer_cumer%Tdisp, &
                timer_cumer%Tdisp/(timer_end%TTotal-timer_begin%TTotal)*100
            endif

            if (quick_method%nodirect) &
            write (io,'("| 2E EVALUATION TIME =",F16.9,"( ",F5.2,"%)")') timer_end%T2eAll-timer_begin%T2eAll, &
                (timer_end%T2eAll-timer_begin%T2eAll)/(timer_end%TTotal-timer_begin%TTotal)*100
            ! SCF Timing
            write (io,'("| TOTAL SCF TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TSCF, &
                timer_cumer%TSCF/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to evaluate operator
            write (io,'("| ",6x,"TOTAL OP TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TOp, &
                timer_cumer%TOp/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to evaluate 1e integrals
            write (io,'("| ",12x,"TOTAL 1e TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1e, &
                timer_cumer%T1e/(timer_end%TTotal-timer_begin%TTotal)*100
#ifdef DEBUGTIME
            ! Time to evaluate kinetic 1e integrals
            write (io,'("| ",15x,"KINETIC 1e INTEGRALS TIME    =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1eT, &
                timer_cumer%T1eT/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to evaluate attraction 1e integrals
            write (io,'("| ",15x,"ATTRACTION 1e INTEGRALS TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1eV, &
                timer_cumer%T1eV/(timer_end%TTotal-timer_begin%TTotal)*100
#endif
            ! Time to evaluate 2e integrals
            write (io,'("| ",12x,"TOTAL 2e TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T2e, &
                timer_cumer%T2e/(timer_end%TTotal-timer_begin%TTotal)*100
            if(quick_method%DFT) then
                ! Time to evaluate exchange energy
                write (io,'("| ",12x,"TOTAL EXC TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TEx, &
                    timer_cumer%TEx/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
#if defined MPIV && DEBUGTIME
            ! time to reduce operator
            write (io,'("| ",12x,"TOTAL OPERATOR REDUCTION TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TEred, &
                    timer_cumer%TEred/(timer_end%TTotal-timer_begin%TTotal)*100
#endif
#ifdef DEBUGTIME
            write (io,'("| ",12x,"TOTAL ENERGY TIME  =",F16.9,"( ",F5.2,"%)")') timer_cumer%TE, &
                timer_cumer%TE/(timer_end%TTotal-timer_begin%TTotal)*100
#endif
            ! DII Time
            write (io,'("| ",6x,"TOTAL DII TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDII, &
                timer_cumer%TDII/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to diag
            if(quick_method%DIVCON) then
                write (io,'("| ",12x,"TOTAL DC DIAG TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDiag, &
                    timer_cumer%TDiag/(timer_end%TTotal-timer_begin%TTotal)*100
            else
                write (io,'("| ",12x,"TOTAL DIAG TIME    =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDiag, &
                    timer_cumer%TDiag/(timer_end%TTotal-timer_begin%TTotal)*100
            endif

            ! Grad Timing
            if (quick_method%opt .or. quick_method%grad ) then
                write (io,'("| TOTAL GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TGrad, &
                    timer_cumer%TGrad/(timer_end%TTotal-timer_begin%TTotal)*100
#ifdef DEBUGTIME
                write (io,'("| ",6x,"TOTAL NUCLEAR GRADIENT TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%TNucGrad, &
                        timer_cumer%TNucGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                write (io,'("| ",6x,"TOTAL 1e GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') &
                timer_cumer%T1eTGrad,timer_cumer%T1eGrad/(timer_end%TTotal-timer_begin%TTotal)*100
#else
                write (io,'("| ",6x,"TOTAL 1e GRADIENT TIME      =",F16.9,"(",F5.2,"%)")') &
                        timer_cumer%T1eTGrad,timer_cumer%T1eGrad/(timer_end%TTotal-timer_begin%TTotal)*100
#endif

#ifdef DEBUGTIME
                write (io,'("| ",9x,"KINETIC 1e GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1eTGrad, &
                        timer_cumer%T1eTGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                write (io,'("| ",9x,"ATTRACTION 1e GRADIENT TIME   =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1eVGrad, &
                        timer_cumer%T1eVGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                write (io,'("| ",6x,"TOTAL 2e GRADIENT TIME      =",F16.9,"(",F5.2,"%)")') &
                timer_cumer%T2eGrad,timer_cumer%T2eGrad/(timer_end%TTotal-timer_begin%TTotal)*100                
#else
                write (io,'("| ",6x,"TOTAL 2e GRADIENT TIME      =",F16.9,"(",F5.2,"%)")') &
                timer_cumer%T2eGrad,timer_cumer%T2eGrad/(timer_end%TTotal-timer_begin%TTotal)*100
#endif

                if(quick_method%DFT) then
                   write (io,'("| ",6x,"TOTAL EXC GRADIENT TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TExGrad, &
                           timer_cumer%TExGrad/(timer_end%TTotal-timer_begin%TTotal)*100
                endif
#if defined MPIV && DEBUGTIME
                write (io,'("| ",6x,"TOTAL GRAD REDUCTION TIME   =",F16.9,"( ",F5.2,"%)")') timer_cumer%TGradred, &
                        timer_cumer%TGradred/(timer_end%TTotal-timer_begin%TTotal)*100
#endif
            endif

            ! MP2 Time
            if(quick_method%MP2)then
                write (io,'("| TOTAL MP2 TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TMP2, &
                    timer_cumer%TMP2/(timer_end%TTotal-timer_begin%TTotal)*100
            endif

#ifdef DEBUGTIME
            if (quick_method%dipole) then
            ! Dipole Timing
                write (io,'("| DIPOLE TIME        =",F16.9,"(",F5.2,"%)")') timer_cumer%TDip, &
                    timer_cumer%TDip/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
#endif

            ! Most Important, total time
            write (io,'("| TOTAL TIME          =",F16.9)') timer_end%TTotal-timer_begin%TTotal
        endif

        timer_cumer%TTotal=timer_end%TTotal-timer_begin%TTotal

#ifdef MPIV

  call MPI_GATHER(timer_begin%T2e,1,mpi_double_precision,tst2e,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_begin%TEx,1,mpi_double_precision,tstxc,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_begin%T2eGrad,1,mpi_double_precision,tst2egrad,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_begin%TExGrad,1,mpi_double_precision,tstxcgrad,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

  call MPI_GATHER(timer_end%T2e,1,mpi_double_precision,tend2e,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_end%TEx,1,mpi_double_precision,tendxc,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_end%T2eGrad,1,mpi_double_precision,tend2egrad,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_end%TExGrad,1,mpi_double_precision,tendxcgrad,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

  call MPI_GATHER(timer_cumer%T2e,1,mpi_double_precision,t2e,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_cumer%TEx,1,mpi_double_precision,txc,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_GATHER(timer_cumer%T2eGrad,1,mpi_double_precision,t2egrad,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)  
  call MPI_GATHER(timer_cumer%TExGrad,1,mpi_double_precision,txcgrad,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

#ifdef DEBUGTIME
  if(master) then
    write (io,'(" ")')
    write (io,'("----------- MPI TIMING -------------")')

    do i=1, mpisize
     write (io,'("| Rank =", I4,2x,"2e START =",F16.9,2x,"XC START =",F16.9,2x,"2e GRAD START =",F16.9,2x, &
     "XC GRAD START =",F16.9)') i, tst2e(i), tstxc(i), tst2egrad(i), tstxcgrad(i)
    enddo

    do i=1, mpisize
     write (io,'("| Rank =", I4,2x,"2e END   =",F16.9,2x,"XC END   =",F16.9,2x,"2e GRAD END   =",F16.9,2x, &
     "XC GRAD END   =",F16.9)') i, tend2e(i), tendxc(i), tend2egrad(i), tendxcgrad(i)
    enddo

    do i=1, mpisize
     write (io,'("| Rank =", I4,2x,"2e TIME  =",F16.9,2x,"XC TIME  =",F16.9,2x,"2e GRAD TIME  =",F16.9,2x, &
     "XC GRAD TIME  =",F16.9)') i, t2e(i), txc(i), t2egrad(i), txcgrad(i) 
    enddo
  endif
#endif
#endif


! Madu Manathunga blocked following block on 03/10/2020
#ifdef MPIV
!        !----------------------------------------------------
!        ! For MPI timing
!        !----------------------------------------------------
!        if (bMPI) then
!            if (.not.master) then
!                tmp_timer_cumer=timer_cumer
!                call MPI_SEND(tmp_timer_cumer,1,mpi_timer_cumer_type,0,mpirank,MPI_COMM_WORLD,IERROR)
!            else
!                MPI_timer_cumer=timer_cumer
!                max_timer_cumer=timer_cumer
!                do i=1,mpisize-1
!                    call MPI_RECV(tmp_timer_cumer,1,mpi_timer_cumer_type,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
!                    MPI_timer_cumer%TTotal=MPI_timer_cumer%TTotal+tmp_timer_cumer%T2e+tmp_timer_cumer%TMP2+ &
!                        tmp_timer_cumer%T1e+ tmp_timer_cumer%TDiag+tmp_timer_cumer%TGrad
!                    MPI_timer_cumer%TTotal=MPI_timer_cumer%TTotal+tmp_timer_cumer%TDiag
!                    MPI_timer_cumer%T2e=MPI_timer_cumer%T2e+tmp_timer_cumer%T2e
!                    MPI_timer_cumer%T1e=MPI_timer_cumer%T1e+tmp_timer_cumer%T1e
!                    MPI_timer_cumer%TDiag=MPI_timer_cumer%TDiag+tmp_timer_cumer%TDiag
!                    MPI_timer_cumer%TSCF=MPI_timer_cumer%TSCF+tmp_timer_cumer%TDiag+tmp_timer_cumer%T2e+ &
!                        tmp_timer_cumer%T1e
!                    MPI_timer_cumer%TDII=MPI_timer_cumer%TDII+tmp_timer_cumer%TDII
!                    MPI_timer_cumer%TMP2=MPI_timer_cumer%TMP2+tmp_timer_cumer%TMP2
!                    MPI_timer_cumer%TOp=MPI_timer_cumer%TOp+tmp_timer_cumer%T2e+tmp_timer_cumer%T1e
!                    MPI_timer_cumer%TGrad=MPI_timer_cumer%TGrad+tmp_timer_cumer%TGrad
!
!                    if (tmp_timer_cumer%T2e.ge.max_timer_cumer%T2e) max_timer_cumer%T2e=tmp_timer_cumer%T2e
!                    if (tmp_timer_cumer%T1e.ge.max_timer_cumer%T1e) max_timer_cumer%T1e=tmp_timer_cumer%T1e
!                    if (tmp_timer_cumer%TDiag.ge.max_timer_cumer%TDiag) max_timer_cumer%TDiag=tmp_timer_cumer%TDiag
!                    if (tmp_timer_cumer%TMP2.ge.max_timer_cumer%TMP2) max_timer_cumer%TMP2=tmp_timer_cumer%TMP2
!                    if (tmp_timer_cumer%TGrad.ge.max_timer_cumer%TGrad) max_timer_cumer%TGrad=tmp_timer_cumer%TGrad
!                enddo
!            endif
!
!            if (master) then
!                write (io,'("----------- MPI TIMING -------------")')
!                ! SCF Time
!                write (io,'("MPI SCF TIME               =",F16.9," (MASTER=",F6.2,"%)")') MPI_timer_cumer%TSCF, &
!                    timer_cumer%TSCF/MPI_timer_cumer%TSCF*100
!                ! Op Time
!                write (io,'(6x,"MPI Op TIME                =",F16.9," (MASTER=",F6.2,"%)")') MPI_timer_cumer%TOp, &
!                    timer_cumer%TOp/MPI_timer_cumer%TOp*100
!                ! 1e Time
!                write (io,'(12x,"MPI 1e TIME                =",F16.9," (MASTER=",F6.2,"%, MAX=",F6.2,"%)")') MPI_timer_cumer%T1e, &
!                    timer_cumer%T1e/MPI_timer_cumer%T1e*100,max_timer_cumer%T1e/MPI_timer_cumer%T1e*100
!                ! 2e Time
!                write (io,'(12x,"MPI 2e TIME                =",F16.9," (MASTER=",F6.2,"%, MAX=",F6.2,"%)")') MPI_timer_cumer%T2e, &
!                    timer_cumer%T2e/MPI_timer_cumer%T2e*100,max_timer_cumer%T2e/MPI_timer_cumer%T2e*100
!                ! DIIS Time
!                write (io,'(6x,"MPI DIIS TIME              =",F16.9," (MASTER=",F6.2,"%)")') MPI_timer_cumer%TDII, &
!                    timer_cumer%TDII/MPI_timer_cumer%TDII*100
!                ! Diag Time
!                write (io,'(12x,"MPI DIAG TIME              =",F16.9," (MASTER=",F6.2,"%, MAX=",F6.2,"%)")') MPI_timer_cumer%TDiag,&
!                    timer_cumer%TDiag/MPI_timer_cumer%TDiag*100,max_timer_cumer%TDiag/MPI_timer_cumer%TDiag*100
!                ! MP2 Time
!                if (quick_method%MP2) then
!                    write (io,'("MPI MP2 TIME               =",F16.9," (MASTER=",F6.2,"%, MAX=",F6.2,"%)")') MPI_timer_cumer%TMP2, &
!                        timer_cumer%TMP2/MPI_timer_cumer%TMP2*100,max_timer_cumer%TMP2/MPI_timer_cumer%TMP2*100
!                endif
!                ! Gradient Time
!                if (quick_method%opt) then
!                    write (io,'("MPI GRAD TIME              =",F16.9," (MASTER=",F6.2,"%, MAX=",F6.2,"%)")') MPI_timer_cumer%TGrad,&
!                        timer_cumer%TGrad/MPI_timer_cumer%TGrad*100,max_timer_cumer%TGrad/MPI_timer_cumer%TGrad*100
!                endif
!
!                ! Most Important, total time
!                write (io,'("MPI TOTAL CPU TIME         =",F16.9," (MASTER=",F6.2,"%)")') MPI_timer_cumer%TTotal, &
!                    timer_cumer%TTotal/MPI_timer_cumer%TTotal*100
!            endif
!        endif
#endif
        if (master) then
            write (io,'("------------------------------------")')
            call PrtTim(io,timer_end%TTotal-timer_begin%TTotal)
            call PrtAct(io,"Finish Output Timing Information")
        endif
    end subroutine timer_output


#ifdef MPIV
    !-----------------------
    ! mpi timer setup
    !-----------------------
    subroutine mpi_setup_timer

        use quick_mpi_module
        use mpi
        implicit none

        ! declaim mpi timer
        if (bMPI) then
            call MPI_TYPE_CONTIGUOUS(TIMER_SIZE, mpi_double_precision,MPI_timer_type,mpierror)
            call MPI_TYPE_COMMIT(MPI_timer_type,mpierror)

            call MPI_TYPE_CONTIGUOUS(TIMER_CUMER_SIZE, mpi_double_precision,MPI_timer_cumer_type,mpierror)
            call MPI_TYPE_COMMIT(MPI_timer_cumer_type,mpierror)
        endif

    end subroutine mpi_setup_timer
#endif

#if defined CUDA_MPIV || defined HIP_MPIV
    subroutine get_mgpu_time
        use quick_mpi_module
        use mpi
        implicit none
        integer :: IERROR
        double precision :: tsum_2elb, tsum_xclb, tsum_xcrb, tsum_xcpg

        call MPI_REDUCE(timer_cumer%T2elb, tsum_2elb, 1, mpi_double_precision, MPI_MAX, 0, MPI_COMM_WORLD, IERROR)
        call MPI_REDUCE(timer_cumer%TDFTlb, tsum_xclb, 1, mpi_double_precision, MPI_MAX, 0, MPI_COMM_WORLD, IERROR)
        call MPI_REDUCE(timer_cumer%TDFTrb, tsum_xcrb, 1, mpi_double_precision, MPI_MAX, 0, MPI_COMM_WORLD, IERROR)
        call MPI_REDUCE(timer_cumer%TDFTpg, tsum_xcpg, 1, mpi_double_precision, MPI_MAX, 0, MPI_COMM_WORLD, IERROR)

        if(master) then
          timer_cumer%T2elb=tsum_2elb
          timer_cumer%TDFTlb=tsum_xclb
          timer_cumer%TDFTrb=tsum_xcrb
          timer_cumer%TDFTpg=tsum_xcpg
        endif
    end subroutine get_mgpu_time

#endif

end module quick_timer_module
