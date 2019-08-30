










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
        type (quick_timer) tmp_timer
        type (quick_timer_cumer) tmp_timer_cumer,max_timer_cumer

        !----------------------------------------------------
        ! For Master nodes or single process timing infomations
        !----------------------------------------------------
        if (master) then
            call PrtAct(io,"Output Timing Infomration")
            write (io,'("------------- TIMING ---------------")')
            ! Initial Guess Timing
            write (io,'("INITIAL GUESS TIME =",F16.9,"( ",F5.2,"%)")') timer_end%TIniGuess-timer_begin%TIniGuess, &
                (timer_end%TIniGuess-timer_begin%TIniGuess)/(timer_end%TTotal-timer_begin%TTotal)*100
            if (quick_method%nodirect) &
            write (io,'("2E EVALUATION TIME =",F16.9,"( ",F5.2,"%)")') timer_end%T2eAll-timer_begin%T2eAll, &
                (timer_end%T2eAll-timer_begin%T2eAll)/(timer_end%TTotal-timer_begin%TTotal)*100

            ! SCF Timing
            write (io,'("TOTAL SCF TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TSCF, &
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
                write (io,'(12x,"TOTAL Exchange TIME=",F16.9,"( ",F5.2,"%)")') timer_cumer%TEx, &
                    timer_cumer%TEx/(timer_end%TTotal-timer_begin%TTotal)*100
            endif
            write (io,'(12x,"TOTAL ENERGY TIME  =",F16.9,"( ",F5.2,"%)")') timer_cumer%TE, &
                timer_cumer%TE/(timer_end%TTotal-timer_begin%TTotal)*100                              
            ! DII Time
            write (io,'(6x,"TOTAL DII TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TDII, &
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
            if (quick_method%opt) then
                write (io,'("GRADIENT TIME      =",F16.9,"( ",F5.2,"%)")') timer_cumer%TGrad, &
                    timer_cumer%TGrad/(timer_end%TTotal-timer_begin%TTotal)*100
            endif

            ! MP2 Time
            if(quick_method%MP2)then
                write (io,'("TOTAL MP2 TIME     =",F16.9,"( ",F5.2,"%)")') timer_cumer%TMP2, &
                    timer_cumer%TMP2/(timer_end%TTotal-timer_begin%TTotal)*100
            endif

            ! Most Important, total time
            write (io,'("TOTAL TIME         =",F16.9)') timer_end%TTotal-timer_begin%TTotal
        endif

        timer_cumer%TTotal=timer_end%TTotal-timer_begin%TTotal

        if (master) then
            write (io,'("------------------------------------")')
            call PrtTim(io,timer_end%TTotal-timer_begin%TTotal)
            call PrtAct(io,"Finish Output Timing Infomration")
        endif
    end subroutine timer_output
    
    
    
end module quick_timer_module
