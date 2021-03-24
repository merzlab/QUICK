#include "util.fh"
! Ed Brothers. May 14, 2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine uelectdiis(jscf,PRMS,verbose)
   use allmod
   implicit none

   integer :: nelec,nelecb
   logical :: diisdone
   double precision :: PRMS
   integer :: jscf
   logical, intent(in) :: verbose

   double precision :: BIJ,CIJ,DENSEIJ,errormax,HOLDIJ,LSOLERR,OIJ,OJK,OLDPRMS,PCHANGE,PRMS2
   double precision :: t1,t2, tempij,DENSEJI
   integer i,j,idiis,IERROR,k

   double precision :: alloperatorB(quick_method%maxdiisscf,nbasis,nbasis)
   double precision :: B(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1)
   double precision :: BCOPY(quick_method%maxdiisscf+1,quick_method%maxdiisscf+1)
   double precision :: W(quick_method%maxdiisscf+1), V2(3,nbasis)
   double precision :: COEFF(quick_method%maxdiisscf+1),RHS(quick_method%maxdiisscf+1)

   double precision :: allerror(quick_method%maxdiisscf,nbasis,nbasis)
   double precision :: alloperator(quick_method%maxdiisscf,nbasis,nbasis)

   nelec = quick_molspec%nelec
   nelecb = quick_molspec%nelecb

   ! The purpose of this subroutine is to utilize Pulay's accelerated
   ! scf convergence as detailed in J. Comp. Chem, Vol 3, #4, pg 566-60, 1982.
   ! At the beginning of this process, their is an approximate density
   ! matrix.
   ! This is the unrestricted (Pople-Neesbitt) version of the code.
   ! The step in the procedure are:
   ! 1)  Form the alpha operator matrix for step i, O(i).  (Store in
   ! alloperator array.)
   ! 2)  Form alpha error matrix for step i.
   ! e(i) = Oa Da S - S Da Oa
   ! 3)  Form the beta operator matrix for step i, O(i).  (Store in
   ! alloperatorb array.)
   ! 4)  Form beta error matrix for step i.
   ! e(i) = e(i,alpha part)+Ob Db S - S Db Ob
   ! 5)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
   ! 6)  Store the e'(I) in allerror.
   ! 7)  Form matrix B, which is:

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

   ! 8)  Solve B*COEFF = RHS which is:

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


   ! 9) Form a new alpha operator matrix based on
   ! O(new) = [Sum over i] c(i)O(i)

   ! 10) Diagonalize the operator matrix to form a new density matrix.

   ! 11) Form a new beta operator matrix based on
   ! O(new) = [Sum over i] c(i)O(i)

   ! 12) Diagonalize the operator matrix to form a new density matrix.

   ! As in scf.F, each step wil be reviewed as we pass through the code.

   ! 1)  Form the alpha operator matrix for step i, O(i).  (Store in
   ! alloperator array.)

   diisdone=.false.
   idiis=0
   do WHILE (.not.diisdone)
      idiis=idiis+1
      jscf=jscf+1
      call cpu_time(t1)
      if (quick_method%HF) call uhfoperatora
      if (quick_method%DFT) call udftoperatora
      if (quick_method%SEDFT) call usedftoperatora

      do I=1,nbasis
         do J=1,nbasis
            alloperator(idiis,J,I) = quick_qm_struct%o(J,I)
         enddo
      enddo

      ! 2)  Form alpha error matrix for step i.
      ! e(i) = Oa Da S - S Da Oa

      ! The matrix multiplier comes from Steve Dixon. It calculates
      ! C = Transpose(A) B.  Thus to utilize this we have to make sure that the
      ! A matrix is symetric. First, calculate DENSE*S and store in the scratch
      ! matrix hold.

      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + quick_qm_struct%dense(K,I)*quick_qm_struct%s(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      ! Now calculate O*(DENSE*S).  As the operator matrix is symmetric, the
      ! above code can be used. Store this (the ODS term) in the all error
      ! matrix.

      do I=1,nbasis
         do J=1,nbasis
            TEMPIJ = 0.0D0
            do K=1,nbasis
               TEMPIJ = TEMPIJ + quick_qm_struct%o(K,I)*quick_scratch%hold(K,J)
            enddo
            allerror(idiis,I,J) = TEMPIJ
         enddo
      enddo


      ! Calculate D O.

      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + quick_qm_struct%dense(K,I)*quick_qm_struct%o(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      ! Now calculate S (do) and subtract that from the allerror matrix.
      ! This means we now have the e(i) matrix.

      do I=1,nbasis
         do J=1,nbasis
            TEMPIJ = 0.0D0
            do K=1,nbasis
               TEMPIJ = TEMPIJ + quick_qm_struct%s(K,I)*quick_scratch%hold(K,J)
            enddo
            allerror(idiis,I,J) = allerror(idiis,I,J) - TEMPIJ
         enddo
      enddo

      ! 3)  Form the beta operator matrix for step i, O(i).  (Store in
      ! alloperatorb array.)

      if (quick_method%HF) call uhfoperatorb
      if (quick_method%DFT) call udftoperatorb
      if (quick_method%SEDFT) call usedftoperatorb

      do I=1,nbasis
         do J=1,nbasis
            alloperatorB(idiis,J,I) = quick_qm_struct%o(J,I)
         enddo
      enddo

      ! 4)  Form beta error matrix for step i.
      ! e(i) = e(i,alpha part)+Ob Db S - S Db Ob

      ! First, calculate quick_qm_struct%denseb*S and store in the scratch
      ! matrix hold.

      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + quick_qm_struct%denseb(K,I)*quick_qm_struct%s(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      ! Now calculate O*(quick_qm_struct%denseb*S).  As the operator matrix is symmetric, the
      ! above code can be used. Add this (the ODS term) into the allerror
      ! matrix.

      do I=1,nbasis
         do J=1,nbasis
            TEMPIJ = 0.0D0
            do K=1,nbasis
               TEMPIJ = TEMPIJ + quick_qm_struct%o(K,I)*quick_scratch%hold(K,J)
            enddo
            allerror(idiis,I,J) = allerror(idiis,I,J)+TEMPIJ
         enddo
      enddo

      ! Calculate Db O.

      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + quick_qm_struct%denseb(K,I)*quick_qm_struct%o(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      ! Now calculate S (DbO) and subtract that from the allerror matrix.
      ! This means we now have the complete e(i) matrix.

      errormax = 0.d0
      do I=1,nbasis
         do J=1,nbasis
            TEMPIJ = 0.0D0
            do K=1,nbasis
               TEMPIJ = TEMPIJ + quick_qm_struct%s(K,I)*quick_scratch%hold(K,J)
            enddo
            allerror(idiis,I,J) = allerror(idiis,I,J) - TEMPIJ
            errormax = max(allerror(idiis,I,J),errormax)
         enddo
      enddo

      ! 5)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
      ! X is symmetric, but we do not know anything about the symmetry of e.
      ! The easiest way to do this is to calculate e(i) . X , store
      ! this in HOLD, and then calculate Transpose[X] (.e(i) . X)
      ! This also takes care of:
      ! 6) Store e'(i) in allerror.

      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + allerror(idiis,K,I) *quick_qm_struct%x(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            TEMPIJ = 0.0D0
            do K=1,nbasis
               TEMPIJ = TEMPIJ + quick_qm_struct%x(K,I)*quick_scratch%hold(K,J)
            enddo
            allerror(idiis,I,J) = TEMPIJ
         enddo
      enddo

      ! 7)  Form matrix B, which is:

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


      ! Where B(i,j) = Trace(e(i) Transpose(e(j)))
      ! According to an example done in mathematica, B12 = B21.  Note that
      ! the rigorous proof of this phenomenon is left as an exercise for the
      ! reader.  Thus the first step is copying BCOPY to B.  In this way we
      ! only have to recalculate the new elements.

      do I=1,idiis
         do J=1,idiis
            B(J,I) = BCOPY(J,I)
         enddo
      enddo

      ! Now copy the current matrix into HOLD2 transposed.  This will be the
      ! Transpose[ej] used in B(i,j) = Trace(e(i) Transpose(e(j)))

      do J=1,nbasis
         do K=1,nbasis
            quick_scratch%hold2(K,J) = allerror(idiis,J,K)
         enddo
      enddo

      do I=1,idiis

         ! Copy the transpose of error matrix I into HOLD.

         do J=1,nbasis
            do K=1,nbasis
               quick_scratch%hold(K,J) = allerror(I,J,K)
            enddo
         enddo

         ! Calculate and sum together the diagonal elements of e(i) Transpose(e(j))).

         BIJ = 0.d0
         do J=1,nbasis
            do K=1,nbasis
               BIJ = BIJ + quick_scratch%hold2(K,J)*quick_scratch%hold(K,J)
            enddo
         enddo

         ! Now place this in the B matrix.

         B(idiis,I) = BIJ
         B(I,idiis) = BIJ
      enddo

      ! Now that all the BIJ elements are in place, fill in all the column
      ! and row ending -1, and fill up the rhs matrix.

      do I=1,idiis
         B(I,idiis+1) = -1.d0
         B(idiis+1,I) = -1.d0
      enddo
      do I=1,idiis
         RHS(I) = 0.d0
      enddo
      RHS(idiis+1) = -1.d0
      B(idiis+1,idiis+1) = 0.d0

      ! Now save the B matrix in Bcopy so it is available for subsequent
      ! iterations.

      do I=1,idiis
         do J=1,idiis
            BCOPY(J,I)=B(J,I)
         enddo
      enddo

      ! 8)  Solve B*COEFF = RHS which is:

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

      LSOLERR=0
      call LSOLVE(idiis+1,quick_method%maxdiisscf+1,B,RHS,W,quick_method%DMCutoff,COEFF,LSOLERR)

      ! 9) Form a new alpha operator matrix based on
      ! O(new) = [Sum over i] c(i)O(i)
      ! If the solution to step eight failed, skip this step and revert
      ! to a standard scf cycle.

      if (LSOLERR == 0) then
         do J=1,nbasis
            do K=1,nbasis
               OJK=0.d0
               do I=1,idiis
                  OJK = OJK + COEFF(I) * alloperator(I,K,J)
               enddo
               quick_qm_struct%o(J,K) = OJK
            enddo
         enddo
      else
         if(verbose) write (ioutfile,'(" DIIS FAILED !! RETURN TO ", &
               & "NORMAL SCF. (NOT FATAL.)")')
         jscf=jscf-1
         diisdone=.true.
         goto 50
      endif

      ! 10) Diagonalize the alpha operator matrix to form a new alpha
      ! density matrix.

      ! First you have to transpose this into an orthogonal basis, which
      ! is accomplished by calculating Transpose[X] . O . X.


      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + quick_qm_struct%o(K,I)*quick_qm_struct%x(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            OIJ = 0.0D0
            do K=1,nbasis
               OIJ = OIJ + quick_qm_struct%x(K,J)*quick_scratch%hold(K,I)
            enddo
            quick_qm_struct%o(I,J) = OIJ
         enddo
      enddo

      ! Now diagonalize the operator matrix.

      CALL DIAG(nbasis,quick_qm_struct%o,nbasis,quick_method%DMCutoff,V2,quick_qm_struct%E, &
            quick_qm_struct%idegen,quick_qm_struct%vec, &
            IERROR)



      ! Calculate C = XC' and form a new density matrix.
      ! The C' is from the above diagonalization.  Also, save the previous
      ! Density matrix to check for convergence.

      do I=1,nbasis
         do J=1,nbasis
            CIJ = 0.0D0
            do K=1,nbasis
               CIJ = CIJ + quick_qm_struct%x(K,I)*quick_qm_struct%vec(K,J)
            enddo
            quick_qm_struct%co(I,J) = CIJ
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            quick_scratch%hold(J,I) = quick_qm_struct%dense(J,I)
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            DENSEJI = 0.d0
            do K=1,nelec
               DENSEJI = DENSEJI + (quick_qm_struct%co(J,K)*quick_qm_struct%co(I,K))
            enddo
            quick_qm_struct%dense(J,I) = DENSEJI
         enddo
      enddo

      ! Now check for convergence of the alpha matrix.

      OLDPRMS = PRMS
      PRMS=0.d0
      PCHANGE=0.d0
      do I=1,nbasis
         do J=1,nbasis
            PRMS=PRMS+(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I))**2.d0
            PCHANGE=MAX(PCHANGE,ABS(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I)))
         enddo
      enddo
      PRMS = (PRMS/nbasis**2.d0)**0.5d0

      ! 11) Form a new BETA operator matrix based on
      ! O(new) = [Sum over i] c(i)O(i)
      ! If the solution to step eight failed, skip this step and revert
      ! to a standard scf cycle.

      if (LSOLERR == 0) then
         do J=1,nbasis
            do K=1,nbasis
               OJK=0.d0
               do I=1,idiis
                  OJK = OJK + COEFF(I) * alloperatorb(I,K,J)
               enddo
               quick_qm_struct%o(J,K) = OJK
            enddo
         enddo
      endif

      ! 8) Diagonalize the beta operator matrix to form a new beta
      ! density matrix.

      ! First you have to transpose this into an orthogonal basis, which
      ! is accomplished by calculating Transpose[X] . O . X.


      do I=1,nbasis
         do J=1,nbasis
            HOLDIJ = 0.0D0
            do K=1,nbasis
               HOLDIJ = HOLDIJ + quick_qm_struct%o(K,I)*quick_qm_struct%x(K,J)
            enddo
            quick_scratch%hold(I,J) = HOLDIJ
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            OIJ = 0.0D0
            do K=1,nbasis
               OIJ = OIJ + quick_qm_struct%x(K,J)*quick_scratch%hold(K,I)
            enddo
            quick_qm_struct%o(I,J) = OIJ
         enddo
      enddo

      ! Now diagonalize the operator matrix.

      CALL DIAG(nbasis,quick_qm_struct%o,nbasis,quick_method%DMCutoff,V2, &
        quick_qm_struct%EB,quick_qm_struct%idegen,quick_qm_struct%vec, IERROR)



      ! Calculate C = XC' and form a new density matrix.
      ! The C' is from the above diagonalization.  Also, save the previous
      ! Density matrix to check for convergence.

      do I=1,nbasis
         do J=1,nbasis
            CIJ = 0.0D0
            do K=1,nbasis
               CIJ = CIJ + quick_qm_struct%x(K,I)*quick_qm_struct%vec(K,J)
            enddo
            quick_qm_struct%cob(I,J) = CIJ
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            quick_scratch%hold(J,I) = quick_qm_struct%denseb(J,I)
         enddo
      enddo

      do I=1,nbasis
         do J=1,nbasis
            DENSEJI = 0.d0
            do K=1,nelecb
               DENSEJI = DENSEJI + (quick_qm_struct%cob(J,K)*quick_qm_struct%cob(I,K))
            enddo
            quick_qm_struct%denseb(J,I) = DENSEJI
         enddo
      enddo

      ! Now check for convergence.

      PRMS2=0.d0
      do I=1,nbasis
         do J=1,nbasis
            PRMS2=PRMS2+(quick_qm_struct%denseb(J,I)-quick_scratch%hold(J,I))**2.d0
            PCHANGE=MAX(PCHANGE,ABS(quick_qm_struct%denseb(J,I)-quick_scratch%hold(J,I)))
         enddo
      enddo
      PRMS2 = (PRMS2/nbasis**2.d0)**0.5d0
      PRMS = MAX(PRMS,PRMS2)

      call cpu_time(t2)

      if(verbose) write (ioutfile,'(/,"| SCF CYCLE      = ",I8, &
            & "      TIME      = ",F6.2)') &
            jscf,T2-T1
      if(verbose) write (ioutfile,'("| DIIS CYCLE     = ",I8, &
            & "      MAX ERROR = ",E12.6)') &
            idiis,errormax
      if(verbose) write (ioutfile,'("| RMS CHANGE     = ",E12.6, &
            & "  MAX CHANGE= ",E12.6)') &
            PRMS,PCHANGE

      if (quick_method%DFT .OR. quick_method%SEDFT) then
         if(verbose) write (ioutfile,'(" ALPHA ELECTRON DENSITY    =",F16.10)') &
               quick_qm_struct%aelec
         if(verbose) write (ioutfile,'(" BETA ELECTRON DENSITY     =",F16.10)') &
               quick_qm_struct%belec
      endif

      if (quick_method%prtgap) then
         if(verbose) write (ioutfile,'(" ALPHA HOMO-LUMO GAP (EV) =", &
               & 11x,F12.6)') (quick_qm_struct%E(nelec+1) - quick_qm_struct%E(nelec))*27.2116d0
         if(verbose) write (ioutfile,'(" BETA HOMO-LUMO GAP (EV)  =", &
               & 11x,F12.6)') (quick_qm_struct%EB(nelecb+1) - quick_qm_struct%EB(nelecb))*27.2116d0
      endif

      if (PRMS < quick_method%pmaxrms .and. pchange < quick_method%pmaxrms*100.d0)then
         if(verbose) write (ioutfile,' &
               & (" PREPARING FOR FINAL NO INTERPOLATION ITERATION")')
         diisdone=.true.
         elseif(OLDPRMS <= PRMS) then
         if(verbose) write (ioutfile,' &
               & (" DIIS NOT IMPROVING. RETURN TO TRADITIONAL SCF.")')
         diisdone=.true.
      endif
      if(jscf >= quick_method%iscf-1) then
         if(verbose) write (ioutfile,'(" RAN OUT OF CYCLES.  NO CONVERGENCE.")')
         if(verbose) write (ioutfile,' &
               & (" PERFORM FINAL NO INTERPOLATION ITERATION")')
         diisdone=.true.
      endif
      50 continue
      diisdone = idiis.eq.quick_method%maxdiisscf.or.diisdone
   enddo

end subroutine uelectdiis
