#   include "../config.h"
!
!
!	quick_timer_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
module quick_timer_module
    implicit none
    
    integer TIMER_SIZE,TIMER_CUMER_SIZE
    
    ! MPI timer data type
    integer MPI_timer_cumer_type,MPI_timer_type
    parameter(TIMER_SIZE=12,TIMER_CUMER_SIZE=10)
    
    !timer type
    type quick_timer
        double precision:: TIniGuess=0.0
        double precision:: TTotal=0.0d0
        double precision:: TDiag=0.0d0
        double precision:: TMP2=0.0d0
        double precision:: TDII=0.0d0
        double precision:: TSCF=0.0d0
        double precision:: TOp=0.0d0
        double precision:: T1e=0.0d0
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
    end type quick_timer
    
    type quick_timer_cumer
        double precision:: TTotal=0.0d0
        double precision:: TDiag=0.0d0
        double precision:: TMP2=0.0d0
        double precision:: TDII=0.0d0
        double precision:: TSCF=0.0d0
        double precision:: TOp=0.0d0
        double precision:: T1e=0.0d0
        double precision:: T2e=0.0d0
        double precision:: T2eAll=0.0d0
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
        implicit none
        integer i,IERROR,io
        double precision t_pure_init_guess,t_tot_dftop
#ifdef MPIV
        include "mpif.h"
#endif
        type (quick_timer) tmp_timer
        type (quick_timer_cumer) tmp_timer_cumer,max_timer_cumer

        !----------------------------------------------------
        ! For Master nodes or single process timing infomations
        !----------------------------------------------------
        if (master) then
            call PrtAct(io,"Output Timing Information")
            write (io,'("------------- TIMING ---------------")')
            ! Initial Guess Timing
            if(quick_method%DFT) then
                t_tot_dftop = timer_cumer%TDFTGrdGen + timer_cumer%TDFTGrdWt + timer_cumer%TDFTGrdOct + timer_cumer%TDFTPrscrn &
                          +timer_cumer%TDFTGrdPck
            else
                t_tot_dftop=0.0d0
            endif
            t_pure_init_guess = timer_end%TIniGuess-timer_begin%TIniGuess-t_tot_dftop 
            write (io,'("INITIAL GUESS TIME  =",F16.9,"( ",F5.2,"%)")') t_pure_init_guess, &
                t_pure_init_guess/(timer_end%TTotal-timer_begin%TTotal)*100
            if(quick_method%DFT) then
                ! Total time for dft grid formation, pruning and prescreening
                write (io,'("DFT GRID OPERATIONS =",F16.9,"( ",F5.2,"%)")') t_tot_dftop, &
                (t_tot_dftop)/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for grid formation
                write (io,'(6x,"TOTAL GRID FORMATION TIME   =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDFTGrdGen, &
                timer_cumer%TDFTGrdGen/(timer_end%TTotal-timer_begin%TTotal)*100                
                ! Time for computing grid weight
                write (io,'(6x,"TOTAL GRID WEIGHT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDFTGrdWt, &
                timer_cumer%TDFTGrdWt/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for running octree algorithm
                write (io,'(6x,"TOTAL OCTREE RUN TIME       =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDFTGrdOct, &
                timer_cumer%TDFTGrdOct/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for prescreening basis and primitive functions
                write (io,'(6x,"TOTAL PRESCREENING TIME     =",F16.9,"( ",F5.2,"%)")')timer_cumer%TDFTPrscrn, &
                timer_cumer%TDFTPrscrn/(timer_end%TTotal-timer_begin%TTotal)*100
                ! Time for packing points
                write (io,'(6x,"TOTAL DATA PACKING TIME     =",F16.9,"( ",F5.2,"%)")')timer_cumer%TDFTGrdPck, &
                timer_cumer%TDFTGrdPck/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
            if (quick_method%nodirect) &
            write (io,'("2E EVALUATION TIME =",F16.9,"( ",F5.2,"%)")') timer_end%T2eAll-timer_begin%T2eAll, &
                (timer_end%T2eAll-timer_begin%T2eAll)/(timer_end%TTotal-timer_begin%TTotal)*100

            ! SCF Timing
            write (io,'("TOTAL SCF TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TSCF, &
                timer_cumer%TSCF/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to evaluate operator
            write (io,'(6x,"TOTAL OP TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TOp, &
                timer_cumer%TOp/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to evaluate 1e integrals
            write (io,'(12x,"TOTAL 1e TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1e, &
                timer_cumer%T1e/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to evaluate 2e integrals
            write (io,'(12x,"TOTAL 2e TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T2e, &
                timer_cumer%T2e/(timer_end%TTotal-timer_begin%TTotal)*100
            if(quick_method%DFT) then
                ! Time to evaluate exchange energy
                write (io,'(12x,"TOTAL EXC TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TEx, &
                    timer_cumer%TEx/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
            write (io,'(12x,"TOTAL ENERGY TIME  =",F16.9,"( ",F5.2,"%)")') timer_cumer%TE, &
                timer_cumer%TE/(timer_end%TTotal-timer_begin%TTotal)*100                              
            ! DII Time
            write (io,'(6x,"TOTAL DII TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDII, &
                timer_cumer%TDII/(timer_end%TTotal-timer_begin%TTotal)*100
            ! Time to diag
            if(quick_method%DIVCON) then
                write (io,'(12x,"TOTAL DC DIAG TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDiag, &
                    timer_cumer%TDiag/(timer_end%TTotal-timer_begin%TTotal)*100
            else
                write (io,'(12x,"TOTAL DIAG TIME    =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDiag, &
                    timer_cumer%TDiag/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
            
            if (quick_method%dipole) then
            ! Dipole Timing
                write (io,'(6x,"DIPOLE TIME        =",F16.9,"( ",F5.2,"%)")') timer_end%TDip-timer_begin%TDip, &
                    (timer_end%TDip-timer_begin%TDip)/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
            ! Grad Timing
            if (quick_method%opt .or. quick_method%grad ) then
                write (io,'("TOTAL GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TGrad, &
                    timer_cumer%TGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                write (io,'(6x,"TOTAL NUCLEAR GRADIENT TIME =",F16.9,"( ",F5.2,"%)")') timer_cumer%TNucGrad, &
                        timer_cumer%TNucGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                write (io,'(6x,"TOTAL 1e GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T1eGrad, &  
                        timer_cumer%T1eGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                write (io,'(6x,"TOTAL 2e GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%T2eGrad, & 
                        timer_cumer%T2eGrad/(timer_end%TTotal-timer_begin%TTotal)*100

                if(quick_method%DFT) then
                   write (io,'(6x,"TOTAL EXC GRADIENT TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TExGrad, &     
                           timer_cumer%TExGrad/(timer_end%TTotal-timer_begin%TTotal)*100
                endif
            endif

            ! MP2 Time
            if(quick_method%MP2)then
                write (io,'("TOTAL MP2 TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TMP2, &
                    timer_cumer%TMP2/(timer_end%TTotal-timer_begin%TTotal)*100
            endif

            ! Most Important, total time
            write (io,'("TOTAL TIME          =",F16.9)') timer_end%TTotal-timer_begin%TTotal
        endif

        timer_cumer%TTotal=timer_end%TTotal-timer_begin%TTotal

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
        implicit none
        include "mpif.h"    
        
        ! declaim mpi timer
        if (bMPI) then
            call MPI_TYPE_CONTIGUOUS(TIMER_SIZE, mpi_double_precision,MPI_timer_type,mpierror) 
            call MPI_TYPE_COMMIT(MPI_timer_type,mpierror)
        
            call MPI_TYPE_CONTIGUOUS(TIMER_CUMER_SIZE, mpi_double_precision,MPI_timer_cumer_type,mpierror) 
            call MPI_TYPE_COMMIT(MPI_timer_cumer_type,mpierror)
        endif
        
    end subroutine mpi_setup_timer    
#endif
    
end module quick_timer_module
