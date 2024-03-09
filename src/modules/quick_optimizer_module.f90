#include "util.fh"

!---------------------------------------------------------------------!
! Created by Madu Manathunga on 03/24/2021                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     !
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

module quick_optimizer_module

  implicit double precision(a-h,o-z)
  private
  public :: lopt

  interface lopt
        module procedure optimize
  end interface lopt

contains

! IOPT to control the cycles
! Ed Brothers. August 18,2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
  subroutine optimize(ierr)
     use allmod
     use quick_gridpoints_module
     use quick_cutoff_module, only: schwarzoff
     use quick_cshell_eri_module, only: getEriPrecomputables
     use quick_cshell_gradient_module, only: scf_gradient
     use quick_oshell_gradient_module, only: uscf_gradient
     use quick_exception_module
     use quick_molden_module, only: quick_molden
#ifdef MPIV
     use mpi
#endif
     implicit double precision(a-h,o-z)

     logical :: done,diagco
     character(len=1) cartsym(3)
     dimension W(3*natom*(2*MLBFGS+1)+2*MLBFGS)
     dimension coordsnew(natom*3),hdiag(natom*3),iprint(2)
     EXTERNAL LB2
     COMMON /LB3/MP,LP,GTOL,STPMIN,STPMAX

     logical lsearch,diis
     integer IMCSRCH,nstor,ndiis
     double precision gnorm,dnorm,diagter,safeDX,gntest,gtest,sqnpar,accls,oldGrad(3*natom),coordsold(natom*3)
     double precision EChg
     integer, intent(inout) :: ierr

     !---------------------------------------------------------
     ! This subroutine optimizes the geometry of the molecule. It has a
     ! variety of options that are enumerated in the text.  Please note
     ! that all of the methods in this subroutine presuppose the use of
     ! cartesian space for optimization.
     !---------------------------------------------------------

     cartsym(1) = 'X'
     cartsym(2) = 'Y'
     cartsym(3) = 'Z'
     done=.false.      ! flag to show opt is done
     diagco=.false.
     iprint(1)=-1
     iprint(2)=0
     EPS=1.d-9
     XTOL=1.d-11
     EChg=0.0

     ! For right now, there is no way to adjust these and only analytical gradients
     ! are available.  This should be changed later.
     quick_method%analgrad=.true.

     ! Some varibles to determine the geometry optimization
     IFLAG=0
     I=0

     do j=1,natom
        do k=1,3
           quick_qm_struct%gradient((j-1)*3+K)=0d0
        enddo
     enddo

     !------------- MPI/MASTER --------------------------------
     if (master) then
        call PrtAct(ioutfile,"Begin Optimization Job")

        ! At the start of this routine we have a converged density matrix.
        ! Check to be sure you should be here.
        if (natom == 1) then
           write (ioutfile,'(" ONE ATOM = NO OPTIMIZATION ")')
           return
        endif

        if (quick_method%iopt < 0) then
           error=0.d0
           do I=1,natom*3
              temp = quick_qm_struct%gradient(I)/5.D-4
              error = temp*temp+error
           enddo
           Write (ioutfile,'(" GRADIENT BASED ERROR =",F20.10)') error
        endif
     endif
     
     !------------- END MPI/MASTER ----------------------------

     do WHILE (I.lt.quick_method%iopt.and..not.done)
        I=I+1

        if (master) then
           call PrtAct(ioutfile,"Optimize for New Step")

           ! First let's print input geometry
           write(ioutfile,*)
           write(ioutfile,'(12("="))',advance="no")
           write(ioutfile,'(2x,"GEOMETRY FOR OPTIMIZATION STEP",I4," OUT OF ",I4,2x)',advance="no") I,quick_method%iopt
           write(ioutfile,'(12("="))')
           write(ioutfile,*)
           write(ioutfile,'("GEOMETRY INPUT")')
           write(ioutfile,'("ELEMENT",6x,"X",14x,"Y",14x,"Z")')
           do J=1,natom
              Write (ioutfile,'(2x,A2,6x,F12.6,3x,F12.6,3x,F12.6)') &
                    symbol(quick_molspec%iattype(J)),xyz(1,J)*0.529177249d0, &
                    xyz(2,J)*0.529177249d0,xyz(3,J)*0.529177249d0
           enddo

           !        Block temperorly, this is integralcutoff for opt step,
           !        should be larger than single point to save time
           quick_method%integralCutoff=1.0d0/(10.0d0**6.0d0)
           quick_method%Primlimit=1.0d0/(10.0d0**6.0d0)


           ! Save grad and coordinate for further use
           do j=1,natom
              do k=1,3
                 oldGrad((j-1)*3+K)=quick_qm_struct%gradient((j-1)*3+K)
                 coordsold((j-1)*3+K)=xyz(k,j)
              enddo
           enddo
        endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
        call gpu_setup(natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &
              quick_molspec%molchg, quick_molspec%iAtomType)
        call gpu_upload_xyz(xyz)
        call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
#endif

        ! calculate energy first
        call getEriPrecomputables
        call schwarzoff

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV 
        call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
              ncontract, itype, aexp, dcoeff, &
              quick_basis%first_basis_function, quick_basis%last_basis_function, &
              quick_basis%first_shell_basis_function,quick_basis%last_shell_basis_function, &
              quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
              quick_basis%ktype, quick_basis%kprim, quick_basis%kshell,quick_basis%Ksumtype, &
              quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal,quick_basis%Qsbasis, quick_basis%Qfbasis, &
              quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)

        call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)

        call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, quick_molspec%extchg, ierr)

#if defined CUDA_MPIV || defined HIP_MPIV
      timer_begin%T2elb = timer_end%T2elb
      call mgpu_get_2elb_time(timer_end%T2elb)
      timer_cumer%T2elb = timer_cumer%T2elb+timer_end%T2elb-timer_begin%T2elb
#endif

#endif

        call getEnergy(.false., ierr)

        !   This line is for test only
        !   quick_method%bCUDA = .false.
        ! Now we have several scheme to obtain gradient. For now,
        ! only analytical gradient is available

        ! 11/19/2010 YIPU MIAO BLOCKED SOME SUBS.
        if (quick_method%analgrad) then
           if (quick_method%UNRST) then
             if (.not. quick_method%uscf_conv .and. .not. quick_method%allow_bad_scf) then
                ierr=33
                return
             endif 
             CALL uscf_gradient
           else
             if (.not. quick_method%scf_conv .and. .not. quick_method%allow_bad_scf) then
                ierr=33
                return
             endif
             CALL scf_gradient
           endif
        endif

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
        if (quick_method%bCUDA) then
          call gpu_cleanup()
        endif
#endif
          !quick_method%bCUDA=.true.
        if (master) then

           !-----------------------------------------------------------------------
           ! Copy current geometry into coordsnew. Fill the rest of the array w/ zeros.
           ! Also fill the rest of gradient with zeros.
           !-----------------------------------------------------------------------
           do J=1,natom
              do K=1,3
                 coordsnew((J-1)*3 + K) = xyz(K,J)
              enddo
           enddo

           ! Now let's call LBFGS.
           call LBFGS(natom*3,MLBFGS,coordsnew,quick_qm_struct%Etot,quick_qm_struct%gradient,DIAGCO,HDIAG,IPRINT,EPS,XTOL,W,IFLAG)

           lsearch=.false.
           diis=.true.
           imcsrch=0
           nstor=10
           dnorm=1.0d0
           diagterm=1.0e-4
           safeDX=0.10d0

           sqnpar=dsqrt(natom*3.0d0)
           gtest=0.5d0
           gntest = max(gtest*sqnpar*0.25d0,gtest)
           accls=0.0d0
           !-----------------------------------------------------------------------
           ! We have a new set of coordinates, copy it onto the xyz array,
           ! and get a new energy and set of gradients. Be sure to check stepsize.
           ! Also, assemble some of the test criterion.
           !-----------------------------------------------------------------------

           ! First store coordinates for Molden so we don't store coordinates of next step
           if(write_molden) then
               quick_molden%xyz_snapshots(:,:,quick_molden%iexport_snapshot)=xyz(:,:)
               quick_molden%iexport_snapshot = quick_molden%iexport_snapshot + 1
           endif

           geomax = -1.d0
           georms = 0.d0
           do J=1,natom
              do K=1,3
                 tempgeo =dabs(xyz(K,J)- coordsnew((J-1)*3 + K))

                 ! If the change is too much, then we have to have small change to avoid error
                 if (tempgeo > quick_method%stepMax) then
                    xyz(K,J) =  xyz(K,J)+(coordsnew((J-1)*3+K)-xyz(K,J))*quick_method%stepMax/tempgeo
                    tempgeo = quick_method%stepMax*0.529177249d0
                 !else if (abs(quick_qm_struct%gradient((J-1)*3 + K))>0.001) then
                 !  xyz(K,J) = (coordsnew((J-1)*3 + K)-XYZ(K,J))*3+XYZ(K,J)
                 else
                    tempgeo = tempgeo*0.529177249d0
                    xyz(K,J) = coordsnew((J-1)*3 + K)
                 endif

                 ! Max geometry change
                 geomax = max(geomax,tempgeo)
                 ! geometry RMS
                 georms = georms + tempgeo**2.d0
              enddo
           enddo

           gradmax = -1.d0
           gradnorm = 0.d0
           write (ioutfile,'(/," ANALYTICAL GRADIENT: ")')
           write (ioutfile,'(76("-"))')
           write (ioutfile,'(" VARIBLES",4x,"OLD_X",12x,"OLD_GRAD",8x,"NEW_GRAD",10x,"NEW_X")')
           write (ioutfile,'(76("-"))')
           do Iatm=1,natom
              do Imomentum=1,3
                 ! Max gradient change
                 gradmax = max(gradmax,dabs(quick_qm_struct%gradient((Iatm-1)*3+Imomentum)))
                 ! Grad change normalization
                 gradnorm = gradnorm + quick_qm_struct%gradient((Iatm-1)*3+Imomentum)**2.d0
                 write (ioutfile,'(I5,A1,3x,F14.10,3x,F14.10,3x,F14.10,3x,F14.10)')Iatm,cartsym(imomentum), &
                       coordsold((Iatm-1)*3+Imomentum)*0.529177249d0,oldGrad((Iatm-1)*3+Imomentum), &
                       quick_qm_struct%gradient((Iatm-1)*3+Imomentum),xyz(Imomentum,Iatm)*0.529177249d0
              enddo
           enddo
           write(ioutfile,'(76("-"))')
           gradnorm = (gradnorm/dble(natom*3))**.5d0

           ! geometry RMS
           georms = (georms/dble(natom*3))**.5d0

           if (i.gt.1) then
              Write (ioutfile,'(" OPTIMIZATION STATISTICS:")')
              Write (ioutfile,'(" ENERGY CHANGE           = ",E20.10," (REQUEST= ",E12.5" )")') quick_qm_struct%Etot-Elast, &
                                                                                          quick_method%EChange
              Write (ioutfile,'(" MAXIMUM GEOMETRY CHANGE = ",E20.10," (REQUEST= ",E12.5" )")') geomax,quick_method%geoMaxCrt
              Write (ioutfile,'(" GEOMETRY CHANGE RMS     = ",E20.10," (REQUEST= ",E12.5" )")') georms,quick_method%gRMSCrt
              !Write (ioutfile,'(" MAXIMUM GRADIENT ELEMENT= ",E20.10," (REQUEST= ",E12.5" )")') gradmax,quick_method%gradMaxCrt
              Write (ioutfile,'(" GRADIENT NORM           = ",E20.10," (REQUEST= ",E12.5" )")') gradnorm,quick_method%gNormCrt

              EChg = quick_qm_struct%Etot-Elast
              done = quick_method%geoMaxCrt.gt.geomax
              done = done.and.(quick_method%EChange.gt.abs(EChg))
!              done = done.and.(quick_method%gRMSCrt.gt.georms)
              !done = done.and.(quick_method%gradMaxCrt.gt.gradmax * 10 .or. (EChg.gt.0 .and. i.gt.5))
              !done = done.and.quick_method%gNormCrt.gt.gradnorm
           else
              Write (ioutfile,'(" OPTIMZATION STATISTICS:")')
              Write (ioutfile,'(" MAXIMUM GRADIENT ELEMENT = ",E20.10," (REQUEST = ",E20.10" )")') gradmax,quick_method%gradMaxCrt
              done = quick_method%gradMaxCrt.gt.gradmax
              done = done.and.quick_method%gNormCrt.gt.gradnorm
              if (done) then
                 Write (ioutfile,'(" NO SIGNIFICANT CHANGE, NO NEED TO OPTIMIZE. USE INITIAL GEOMETRY.")')
                 do j=1,natom
                    do k=1,3
                       xyz(k,j)=coordsold((j-1)*3+K)
                    enddo
                 enddo
              endif
           endif

           if (done)  Write (ioutfile,'(/" GEOMETRY OPTIMIZED AFTER",i5," CYCLES")') i
           call PrtAct(ioutfile,"Finish Optimization for This Step")
           Elast = quick_qm_struct%Etot

           ! If read is on, write out a restart file.
           if (quick_method%readdmx) call wrtrestart
        endif

        !-------------- END MPI/MASTER --------------------
#ifdef MPIV
        ! we now have new geometry, and let other nodes know the new geometry
        if (bMPI)call MPI_BCAST(xyz,natom*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)


        ! Notify every nodes if opt is done
        if (bMPI)call MPI_BCAST(done,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
#endif

        !For DFT geometry optimization, we should delete the grid variables here
        !and
        !reinitiate them in getEnergy method.
        if (quick_method%DFT) then
             if(I.le.quick_method%iopt.and..not.done) then
                  call deform_dft_grid(quick_dft_grid)
             endif
        endif

     enddo


     if (master) then
        if (done) then
           Write (ioutfile,'("================ OPTIMIZED GEOMETRY INFORMATION ==============")')
        else
           write (ioutfile,*) "WARNING: REACHED MAX OPT CYCLES. THE GEOMETRY IS NOT OPTIMIZED."
           write (ioutfile,*) "         PRINTING THE GEOMETRY FROM LAST STEP."
           Write (ioutfile,'("============= GEOMETRY INFORMATION (NOT OPTIMIZED) ===========")')
        endif
        write (ioutfile,*)
        Write (ioutfile,'(" OPTIMIZED GEOMETRY IN CARTESIAN")')
        write (ioutfile,'(" ELEMENT",6x,"X",9x,"Y",9x,"Z")')

        do I=1,natom
           Write (ioutfile,'(2x,A2,6x,F7.4,3x,F7.4,3x,F7.4)') &
                 symbol(quick_molspec%iattype(I)),xyz(1,I)*0.529177249d0, &
                 xyz(2,I)*0.529177249d0,xyz(3,I)*0.529177249d0
        enddo

        write(ioutfile,*)
        write (ioutfile,'(" FORCE")')
        write (ioutfile,'(" ELEMENT",6x, "X",9x,"Y",9x,"Z")')
        do i=1,natom
           write(ioutfile,'(2x,A2,6x,F7.4,3x,F7.4,3x,F7.4)') &
                 symbol(quick_molspec%iattype(I)),-quick_qm_struct%gradient((i-1)*3+1)*0.529177249d0, &
                 -quick_qm_struct%gradient((i-1)*3+2)*0.529177249d0,-quick_qm_struct%gradient((i-1)*3+3)*0.529177249d0
        enddo

        write (ioutfile,*)
        write (ioutfile,'(" MINIMIZED ENERGY = ",F16.9)') quick_qm_struct%Etot
        Write (ioutfile,'("===============================================================")')


        call PrtAct(ioutfile,"Finish Optimization Job")

     endif

     return
  end subroutine optimize

end module quick_optimizer_module
