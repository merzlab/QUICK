#include "util.fh"


!*******************************************************
! electdiisdc
!-------------------------------------------------------
! 10/25/2010 YIPU MIAO Successful run on mpi, begin to test
! 10/20/2010 YIPU MIAO Add MPI option to DIIS div&con
! this is dii for div & con
subroutine electdiisdc(jscf,PRMS)
   use allmod
#ifdef MPIV
   use mpi
#endif
   implicit double precision(a-h,o-z)

   logical :: diisdone
   dimension :: B(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1),BSAVE(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1)
   dimension :: BCOPY(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1),W(quick_method%maxdiisscf+1)
   dimension :: COEFF(quick_method%maxdiisscf+1),RHS(quick_method%maxdiisscf+1)
   double precision,allocatable :: dcco(:,:)

   logical templog1,templog2
   integer elecs,itemp
   double precision efermi(10),oneElecO(nbasis,nbasis)
   integer :: lsolerr = 0

   !===========================================================
   ! The purpose of this subroutine is to utilize Pulay's accelerated
   ! scf convergence as detailed in J. Comp. Chem, Vol 3, #4, pg 566-60, 1982.
   ! At the beginning of this process, their is an approximate density
   ! matrix.
   ! The step in the procedure are:
   ! 1)  Form the operator matrix for step i, O(i).
   ! 2)  Form error matrix for step i.
   ! e(i) = ODS - Sdo
   ! 3)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
   ! 4)  Store the e'(I) and O(i)
   ! 5)  Form matrix B, which is:
   ! _                                                 _
   ! |                                                   |
   ! |  B(1,1)      B(1,2)     . . .     B(1,J)      -1  |
   ! |  B(2,1)      B(2,2)     . . .     B(2,J)      -1  |
   ! |  .            .                     .          .  |
   ! B = |  .            .                     .          .  |
   ! |  .            .                     .          .  |
   ! |  B(I,1)      B(I,2)     . . .     B(I,J)      -1  |
   ! | -1            -1        . . .      -1          0  |
   ! |_                                                 _|
   ! Where B(i,j) = Trace(e(i) Transpose(e(j)) )
   ! 6)  Solve B*COEFF = RHS which is:
   ! _                                             _  _  _     _  _
   ! |                                               ||    |   |    |
   ! |  B(1,1)      B(1,2)     . . .     B(1,J)  -1  ||  C1|   |  0 |
   ! |  B(2,1)      B(2,2)     . . .     B(2,J)  -1  ||  C2|   |  0 |
   ! |  .            .                     .      .  ||  . |   |  0 |
   ! |  .            .                     .      .  ||  . | = |  0 |
   ! |  .            .                     .      .  ||  . |   |  0 |
   ! |  B(I,1)      B(I,2)     . . .     B(I,J)  -1  ||  Ci|   |  0 |
   ! | -1            -1        . . .      -1      0  || -L |   | -1 |
   ! |_                                             _||_  _|   |_  _|
   ! 7) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
   ! 8) Diagonalize the operator matrix to form a new density matrix.
   ! As in scf.F, each step wil be reviewed as we pass through the code.
   !===========================================================

   !--------------------------------------------
   ! Form the operator matrix for step i, O(i).
   !--------------------------------------------
   diisdone=.false.
   idiis=0
   jscf=0
   if (bMPI) TdcDiagMPI=0.0d0

#ifdef MPIV
   ! Setup MPI integral configuration
   if (bMPI)   call MPI_setup_hfoperator
#endif

   ! First, let's get 1e opertor which only need 1-time calculation
   ! and store them in oneElecO and fetch it every scf time.
   call get1e(oneElecO)

   do while (.not.diisdone)

      ! Tragger timer
      RECORD_TIME(timer_begin%TSCF)
      ! Now Get the Operator

      RECORD_TIME(timer_begin%TOp)
      if (quick_method%HF) then
#ifdef MPIV
         if (bMPI) then
            call MPI_hfoperatordc(oneElecO)
         else
            call hfoperatordc(oneElecO)
         endif
#else
         call hfoperatordc(oneElecO)
#endif
      endif
      if (quick_method%DFT)   call dftoperator
      if (quick_method%SEDFT) call sedftoperator
      RECORD_TIME(timer_end%TOp)

      jscf=jscf+1            ! Cycle time
      idiis=idiis+1          ! DIIS time

      !-------- MPI/MASTER----------------
      if (master) then

         if (quick_method%debug) call debugDivconNorm()

         !--------------------------------------------
         ! Now begin to implement delta increase
         !--------------------------------------------
         ! 10/20/10 YIPU MIAO Rewrite everything, you can't image how mess and urgly it was.
         ! 07/07/07 Xiao HE   Delta density matrix increase is implemented here.
         !--------------------------------------------
         if(jscf.ge.quick_method%ncyc)then

            RECORD_TIME(timer_begin%TDII)
            !--------------------------------------------
            ! Before doing everything, we may save Density Matrix and Operator matrix first.
            ! Note try not to modify Osave and DENSAVE unless you know what you are doing
            !--------------------------------------------
            call CopyDMat(quick_qm_struct%oSave,quick_qm_struct%o,nbasis)            ! recover Operator first
            call CopyDMat(quick_qm_struct%dense,quick_qm_struct%denseSave,nbasis)    ! save density matrix

            do I=1,nbasis
               do J=1,nbasis
                  quick_qm_struct%dense(I,J)=quick_qm_struct%dense(I,J)-quick_qm_struct%denseOld(I,J)
               enddo
            enddo

            !--------------------------------------------
            ! obtain opertor now
            !--------------------------------------------
            if (quick_method%HF) call hfoperatordeltadc
            if (quick_method%DFT) call dftoperator(.true.)


            !--------------------------------------------
            ! recover density matrix
            !--------------------------------------------
            call CopyDMat(quick_qm_struct%denseSave,quick_qm_struct%dense,nbasis)
            RECORD_TIME(timer_end%TDII)
         endif

         !--------------------------------------------
         ! We have modified O and density matrix. And we need to save
         ! Operator for next cycle
         !--------------------------------------------
         call CopyDMat(quick_qm_struct%o,quick_qm_struct%oSave,nbasis)
         call CopyDMat(quick_qm_struct%dense,quick_qm_struct%denseOld,nbasis)


         if(quick_method%debug) call debugElecdii(jscf)


         !--------------------------------------------
         ! Form extract subsystem operator from full system
         !--------------------------------------------
         call Odivided

         !--------------------------------------------
         ! recover basis number
         !--------------------------------------------
         nbasissave=nbasis
      endif
      !------ END MPI/MASTER ----------------------

#ifdef MPIV
      !------ MPI/ALL NODES -----------------------
      ! Broadcast the new density and operator
      if (bMPI) then
         call MPI_BCAST(nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(NNmax,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(np,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(Odcsub,np*NNmax*NNmax,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(Xdcsub,np*NNmax*NNmax,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      endif
      !------ END MPI/ALL NODES -------------------
#endif

      !============================================
      ! Now begin to diag O of subsystems:
      ! which is to do DIV part
      !============================================

      !------ MPI/ALL NODES -----------------------

      Ttmp=0.0d0

#ifdef MPIV
      do Ittt=1,mpi_dc_fragn(mpirank)
         itt=mpi_dc_frag(mpirank,ittt)   ! aimed fragment
#else
      do itt = 1, np
#endif

         ! pass value for convience reason
         nbasis=nbasisdc(itt)    ! basis set for fragment

         allocate(Odcsubtemp(nbasisdc(itt),nbasisdc(itt)))
         allocate(VECtemp(nbasisdc(itt),nbasisdc(itt)))
         allocate(Vtemp(3,nbasisdc(itt)))
         allocate(EVAL1temp(nbasisdc(itt)))
         allocate(IDEGEN1temp(nbasisdc(itt)))
         allocate(dcco(nbasisdc(itt),nbasisdc(itt)))


         ! O'=XOX, the operators are all for subsystem,
         ! transform the operator into canonial ones
         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               quick_scratch%hold(I,J)=0.0D0
               HOLDIJ = 0.0D0
               do K=1,nbasisdc(itt)
                  HOLDIJ = HOLDIJ + Odcsub(itt,I,K)*Xdcsub(itt,K,J)
               enddo
               quick_scratch%hold(I,J) = HOLDIJ
            enddo
         enddo

         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               OIJ = 0.0D0
               do K=1,nbasisdc(itt)
                  OIJ = OIJ + Xdcsub(itt,K,I)*quick_scratch%hold(K,J)
               enddo
               Odcsubtemp(I,J)=OIJ
            enddo
         enddo
         NtempN=nbasisdc(itt)

         !--------------------------------------------
         ! Diag Operator of Subsystem, to solve the eigen equation (F-eS)C=0
         ! Get the eigen energy and coeffecient. Transfer them and store them into
         ! evaldsub(:,:) and codcsub(:,:)
         ! Odcsubtemp:           Opertor need to be diag
         ! NtempN=nbasisdc(itt): dimension
         ! evaldcsub:            energy for subsystems
         ! codcsub:              cooefficient for subsystem
         !--------------------------------------------
         RECORD_TIME(timer_begin%TDiag) ! Trigger the dc timer for subsytem

         call DIAG(NtempN,Odcsubtemp,NtempN,quick_method%DMCutoff,Vtemp,EVAL1temp,IDEGEN1temp,VECtemp,IERROR)

         RECORD_TIME(timer_end%TDiag)  ! Stop the timer

         Ttmp=timer_end%TDiag-timer_begin%TDiag
         timer_cumer%TDiag=timer_cumer%TDiag+timer_end%TDiag-timer_begin%TDiag   ! Global dc diag time

         do i=1,NtempN
            evaldcsub(itt,i)=EVAL1temp(i)
         enddo


         !---------------------------------------------
         ! Calculate C = XC' and form a new density matrix.
         ! The C' is from the above diagonalization.
         !---------------------------------------------
         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               CIJ = 0.0D0
               do K=1,nbasisdc(itt)
                  CIJ = CIJ + Xdcsub(itt,I,K)*VECtemp(K,J)
               enddo
               dcco(I,J) = CIJ
            enddo
         enddo

         ! save coeffiecent to dcsub
         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               codcsub(I,J,itt)=dcco(I,J)
               codcsubtran(I,J,itt)=dcco(I,J)
            enddo
         enddo

         if (allocated(Odcsubtemp)) deallocate(Odcsubtemp)
         if (allocated(VECtemp)) deallocate(VECtemp)
         if (allocated(Vtemp)) deallocate(Vtemp)
         if (allocated(EVAL1temp)) deallocate(EVAL1temp)
         if (allocated(IDEGEN1temp)) deallocate(IDEGEN1temp)
         if (allocated(dcco)) deallocate(dcco)
      enddo

#ifdef MPIV
      !--------------------------------------------
      ! Communicate with master nodes, send energy, density matrix and other infomation
      ! to master nodes. Master nodes will integrate them with its job to complete div-con
      ! calculation
      if (bMPI) then
         if (master) write(ioutfile,'(" ")')
         ! send coefficient to master node
         if (.not.master) then
            do Ittt=1,mpi_dc_fragn(mpirank)
               itt=mpi_dc_frag(mpirank,ittt)
               call MPI_SEND(codcsub(1:NNmax,1:NNmax,itt),NNmax*NNmax, &
                     mpi_double_precision,0,itt,MPI_COMM_WORLD,IERROR)
               call MPI_SEND(codcsubtran(1:NNmax,1:NNmax,itt),NNmax*NNmax, &
                     mpi_double_precision,0,itt,MPI_COMM_WORLD,IERROR)
               call MPI_SEND(evaldcsub(itt,1:NNmax),NNmax,mpi_double_precision, &
                     0,itt,MPI_COMM_WORLD,IERROR)
            enddo

         else
            ! receive data from other node
            do ittt=1,mpisize-1
               do i=1,mpi_dc_fragn(ittt)
                  itt=mpi_dc_frag(ittt,i)
                  call MPI_RECV(codcsub(1:NNmax,1:NNmax,itt),NNmax*NNmax, &
                        mpi_double_precision,ittt,itt,MPI_COMM_WORLD,MPI_STATUS,IERROR)
                  call MPI_RECV(codcsubtran(1:NNmax,1:NNmax,itt),NNmax*NNmax, &
                        mpi_double_precision,ittt,itt,MPI_COMM_WORLD,MPI_STATUS,IERROR)
                  call MPI_RECV(evaldcsub(itt,1:NNmax),NNmax,mpi_double_precision, &
                        ittt,itt,MPI_COMM_WORLD,MPI_STATUS,IERROR)
               enddo
            enddo
         endif
         call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      endif
#endif
      !--------------------------------------------
      ! End of diag for subs
      !--------------------------------------------
      !------ END MPI/ALL NODES -------------------

      !------ MPI/MASTER --------------------------
      if (master) then
         diisdone=.false.

         ! Recover basis set number
         nbasis=nbasissave
         write(ioutfile,*)'-------------------------------'
         write(ioutfile,'("SCF CYCLE=",i5)')jscf

         ! save density
         call CopyDMat(quick_qm_struct%dense,quick_scratch%hold,nbasis)

         !--------------------------------------------
         ! Next step is to calculate fermi energy and renormalize
         ! the density matrix
         !--------------------------------------------
         call fermiSCF(efermi,jscf)

         ! Now check for convergence.
         nsubtemp=0
         PRMS=0.d0
         PCHANGE=0.d0
         do I=1,nbasis
            do J=1,nbasis
               if(dabs(quick_scratch%hold(J,I)).ge.0.00000001d0)then
                  nsubtemp=nsubtemp+1
                  PRMS=PRMS+(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I))**2.d0
                  PCHANGE=max(PCHANGE,abs(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I)))
               endif
            enddo
         enddo
         PRMS = (PRMS/nsubtemp)**0.5d0
         itemp=mod(idiis,quick_method%maxdiisscf)
         if(itemp.eq.0) itemp=quick_method%maxdiisscf
      endif
      !------ END MPI/MASTER ----------------------

      !------ MPI/ALL NODES ------------------------
      RECORD_TIME(timer_end%TSCF)
      timer_cumer%TSCF=timer_cumer%TSCF+timer_end%TSCF-timer_begin%TSCF
      timer_cumer%TDII=timer_cumer%TDII+timer_end%TDII-timer_begin%TDII
      timer_cumer%TOp=timer_cumer%TOp+timer_end%TOp-timer_begin%TOp

      !------ END MPI/ALL NODES --------------------

      if(quick_method%MP2)then
         do itt=1,np
            natt=0
            do k=1,nbasisdc(itt)
               if (evaldcsub(itt,k).lt.efermi(1)) then
                  natt=natt+1
               endif
            enddo
            nelecmp2sub(itt)=natt*2
         enddo
      endif

      !---------- MPI/MASTER ----------------------
      if (master) then
         ! Write infos for this cycle
         write (*,'("SCF TIME = ",F12.2)') timer_end%TSCF-timer_begin%TSCF
         write (*,'("DC DIAG TIME = ",F12.2)')  Ttmp                                  ! Max CPU Time in every node
         write (ioutfile,'("DIIS CYCLE     = ",I8)') itemp
         write (ioutfile,'("DIIS TIME = ",F12.2)') timer_end%TDII-timer_begin%TDII
         write (ioutfile,'("RMS CHANGE     = ",E12.6, "  MAX CHANGE= ",E12.6)') PRMS,PCHANGE
         if(quick_method%printEnergy) write (*,'("TOTAL ENERGY=",F16.9)') quick_qm_struct%Eel+quick_qm_struct%Ecore

         if(PRMS.le.0.00001d0.and.quick_method%integralCutoff.gt.1.0d0/(10.0d0**7.5d0))then
            quick_method%integralCutoff=1.0d0/(10.0d0**8.0d0)
            quick_method%primLimit=1.0d0/(10.0d0**8.0d0)
         endif
         if(PRMS.le.0.000001d0.and.quick_method%integralCutoff.gt.1.0d0/(10.0d0**8.5d0))then
            quick_method%integralCutoff=1.0d0/(10.0d0**9.0d0)
            quick_method%primLimit=1.0d0/(10.0d0**9.0d0)
         endif
         if(PRMS.le.0.0000001d0.and.quick_method%integralCutoff.gt.1.0d0/(10.0d0**9.5d0))then
            quick_method%integralCutoff=1.0d0/(10.0d0**10.0d0)
            quick_method%primLimit=1.0d0/(10.0d0**10.0d0)
         endif

         ! density for dft
         if (quick_method%DFT .or. quick_method%SEDFT) then
            write (ioutfile,'("ALPHA ELECTRON DENSITY    =",F16.10)')    quick_qm_struct%aelec
            write (ioutfile,'("BETA ELECTRON DENSITY     =",F16.10)')    quick_qm_struct%belec
         endif

         ! lsolerr.ne.0 indicates dii failure
         if (lsolerr /= 0) write (ioutfile,'("DIIS FAILED !! PERFORM NORMAL SCF. (NOT FATAL.)")')

         ! Print homo-lumo gap
         if (quick_method%prtgap) then
            write (ioutfile,'("HOMO-LUMO GAP (EV) =",11x,F12.6)')  &
                  (quick_qm_struct%E((quick_molspec%nelec/2)+1) - quick_qm_struct%E(quick_molspec%nelec/2))*27.2116d0
         endif

         ! check convengency
         if (PRMS < quick_method%pmaxrms .and. pchange < quick_method%pmaxrms*100.d0)then
            write (ioutfile,'("REACH CONVERGENCE AFTER ",i4," CYCLES")') jscf
            diisdone=.true.
         endif

         if(jscf >= quick_method%iscf-1) then
            write (ioutfile,'("RAN OUT OF CYCLES.  NO CONVERGENCE.")')
            write (ioutfile,'("PERFORM FINAL NO INTERPOLATION ITERATION")')
            diisdone=.true.
         endif

         diisdone = idiis.eq.30*quick_method%maxdiisscf.or.diisdone
         call flush(ioutfile)
      endif
      !-------- END MPI/MASTER----------------

#ifdef MPIV
      if (bMPI) then
         call MPI_BCAST(diisdone,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
         !            call MPI_BCAST(DENSE,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      endif
#endif
   enddo


   return
end subroutine electdiisdc



!*******************************************************
! fermiSCF(fermidone
!-------------------------------------------------------
! Use iteriation method to calculate fermi energy
!
!
subroutine fermiSCF(efermi,jscf)
   use allmod
   implicit double precision(a-h,o-z)
   logical fermidone
   integer jscf
   double precision :: efermi(10)

   ! Boltzmann constant in eV/Kelvin:
   boltz = 8.617335408d0*0.00001d0/27.2116d0
   tempk=1000.0d0
   betah = 1.0d0/(boltz*tempk)
   etoler = 1.0d-8
   maxit = 100
   elecs=quick_molspec%nelec


   ! Determine the observed range of eigenvalues for all subsystems.
   emin = 10000000.0d0
   emax =-10000000.0d0
   if(np > 1)then
      do itt=1,np
         imin = nelecdcsub(itt)/2
         imax = nelecdcsub(itt)/2+1
         if(mod(nelecdcsub(itt),2).eq.1)imax = nelecdcsub(itt)/2+2
         emin = min(emin,evaldcsub(itt,imin))
         emax = max(emax,evaldcsub(itt,imax))
      enddo
   endif

   ! Use a bisection technique to determine the fermi energy.
   !        emax =emax+2.0d0
   elecs = quick_molspec%nelec
   niter = 0
   fermidone = .false.

   ! Use iteriation method to calculate fermi energy
   do while ((niter .le. maxit).and.(.not.fermidone))


      efermi(1) = (emax + emin)/2.0d0

      ! Set fermi occupation numbers and use the sum of weighted squared
      ! eigenvector coefficients evecsq to get the number of electrons.
      niter = niter + 1
      etotal = 0.0d0

      do ii=1,nbasis
         do jj=1,nbasis
            quick_qm_struct%dense(ii,jj)=0.0d0
         enddo
      enddo

      !--------------------------------------------
      ! Now Begin to transfer DENSE of subsystem to full system
      ! Puv(a)=2puv*sigma(fb(Ef-ei)*Cui(sub a)*Cvi(sub a)
      ! where puv is partion matrix=1,1/2 or 0
      ! fb is Fermi function
      ! Ef is the eigen energy
      ! Ei is Fermi energy which is unknow but can be get from normlization constraint
      !--------------------------------------------

      do itt=1,np
         do I=1,nbasisdc(itt)
            do J=1,nbasisdc(itt)
               Pdcsubtran(I,J,itt)=0.0d0
            enddo
         enddo
      enddo
      do itt=1,np
         do K=1,nbasisdc(itt)
            deltae = evaldcsub(itt,K) - efermi(1)
            arg = betah*deltae
            ! Fermi Function part
            if(arg > 25.0d0)then
               fermitemp = 0.0d0
               elseif(arg < -25.0d0)then
               fermitemp = 1.0d0
            else
               fermitemp = 1.0d0/(1.0d0 + exp(arg))
            endif

            if(dabs(fermitemp).ge.0.00000000001d0)then
               do I=1,nbasisdc(itt)
                  temp1=COdcsub(I,K,itt)*fermitemp*2.0d0
                  do J=I,nbasisdc(itt)
                     temp2=COdcsubtran(J,K,itt)
                     DENSEJI=temp1*temp2
                     Pdcsubtran(J,I,itt)=Pdcsubtran(J,I,itt)+DENSEJI
                  enddo
               enddo
            endif
         enddo
      enddo

      ! And Puv are symmatry
      do itt=1,np
         do I=1,nbasisdc(itt)
            do J=I,nbasisdc(itt)
               Pdcsub(itt,I,J)=Pdcsubtran(J,I,itt)
               Pdcsub(itt,J,I)=Pdcsub(itt,I,J)
            enddo
         enddo
      enddo

      !------------------------------------------------
      ! At this step, now we have Density matrix for subsystems for this cycle
      ! What we want to do is to transfer them to full-system density matrix
      !------------------------------------------------
      ! Xiao HE reconsider buffer area 07/17/2008
      do itt=1,np
         Kstart1=0
         do jtt=1,dcsubn(itt)
            Iblockatom=dcsub(itt,jtt)
            do itemp=quick_basis%first_basis_function(Iblockatom),quick_basis%last_basis_function(Iblockatom)
               Kstart2=0
               do jtt2=1,dcsubn(itt)
                  Jblockatom=dcsub(itt,jtt2)
                  if(dclogic(itt,Iblockatom,Jblockatom).eq.0)then
                     do jtemp=quick_basis%first_basis_function(Jblockatom),quick_basis%last_basis_function(Jblockatom)
                        quick_qm_struct%dense(itemp,jtemp)= quick_qm_struct%dense(itemp,jtemp)+ &
                              invdcoverlap(Iblockatom,Jblockatom)* &
                              Pdcsub(itt,Kstart1+itemp-quick_basis%first_basis_function(Iblockatom)+1, &
                              Kstart2+jtemp-quick_basis%first_basis_function(Jblockatom)+1)
                     enddo
                  endif
                  Kstart2=Kstart2+quick_basis%last_basis_function(Jblockatom)-quick_basis%first_basis_function(Jblockatom)+1
               enddo
            enddo
            Kstart1=Kstart1+quick_basis%last_basis_function(Iblockatom)-quick_basis%first_basis_function(Iblockatom)+1
         enddo
      enddo

      !------------------------------------------------
      ! The accuracy of this step is depend on the fermi energy with under normalization constraint
      ! Now calculate normalization and compare with no. of elections to rule the accurate of fermi energy
      !------------------------------------------------

      temp=0.0d0
      do i=1,nbasis
         do j=1,nbasis
            temp=temp+quick_qm_struct%dense(j,i)*quick_qm_struct%s(j,i)
         enddo
      enddo

      diff = abs(temp - elecs)

      !------------------------------------------------
      ! Throw away lower or upper half of interval, depending upon whether
      ! the computed number of electrons is too low or too high.  Note that
      ! raising the fermi energy increases the number of electrons because
      ! more mo's become fully occupied.
      !------------------------------------------------

      if (temp < elecs) then
         emin = efermi(1)
      else
         emax = efermi(1)
      endif


      if(diff < etoler) fermidone=.true.
   enddo

   ! If can't converge, it will lead to fatal error at late step, but quite common
   ! for first or second iteration
   if (.not.fermidone) then
      write(ioutfile,*) "Exceed the maximum interations"
      write(ioutfile,*) "IF IT APPEARS AT LATE STEP MAY LEAD TO WRONG RESULT!"
   endif

   ! At this point we get the fermi energy (Ei) and store for next cycle.
   write (ioutfile,'("E(Fermi)    = ",f12.7,"  AFTER ",i3, " N.C. Cycles")') efermi(1),niter

   call flush(ioutfile)

end subroutine fermiSCF
