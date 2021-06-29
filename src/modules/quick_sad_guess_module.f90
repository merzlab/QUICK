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

!---------------------------------------------------------------------!
! This module contains subroutines related to sad guess.              ! 
!---------------------------------------------------------------------!

module quick_sad_guess_module

  implicit double precision(a-h,o-z)
  private

  public :: getSadGuess, getSadDense

interface getSadGuess
  module procedure getmolsad
end interface getSadGuess

interface getSadDense
  module procedure get_sad_density_matrix
end interface getSadDense

contains

  subroutine getmolsad(ierr)
     use allmod
     use quick_gridpoints_module
     use quick_files_module
     use quick_exception_module

#ifdef CEW 
     use quick_cew_module, only : quick_cew
#endif
  
     implicit double precision(a-h,o-z)
  
     logical :: present, MPIsaved, readSAD
     double precision:: xyzsaved(3,natom)
     character(len=80) :: keywd
     character(len=20) :: tempstring
     character(len=340) :: sadfile
     integer natomsaved
     type(quick_method_type) quick_method_save
     type(quick_molspec_type) quick_molspec_save
     integer, intent(inout) :: ierr
     logical :: use_cew_save
  
     ! first save some important value
     quick_method_save=quick_method
     quick_molspec_save=quick_molspec
     ! quick_molspec_type has pointers which may lead to memory leaks
     ! therefore, assign individual variable values
  !   quick_molspec_save%imult = quick_molspec%imult
  !   quick_molspec_save%nelec = quick_molspec%nelec
  
     natomsaved=natom
     xyzsaved=xyz
     MPIsaved=bMPI
  
     istart = 1
     ifinal = 80
     ibasisstart = 1
     ibasisend = 80
  
     ! Then give them new value
     bMPI=.false.
     quick_molspec%imult=0
     quick_method%HF=.true.
     quick_method%DFT=.false.
     quick_method%UNRST=.true.
     quick_method%ZMAT=.false.
     quick_method%divcon=.false.
     quick_method%nodirect=.false.
     call allocate_mol_sad(quick_molspec%iatomtype)
  
  
     if (master) then
        call PrtAct(ioutfile,"Begin SAD initial guess")
        !-------------------------------------------
        ! First, find atom type and initialize
        !-------------------------------------------
  
        natom=1
        do I=1,3
           xyz(I,1) = 0.0d0
        enddo
  
        do iitemp=1,quick_molspec%iatomtype
           write(ioutfile,'(" For Atom Kind = ",i4)') iitemp
  
           ! if quick is called through api multiple times, this is necessary
           if(wrtStep .gt. 1) then
             call deallocate_calculated
           endif
  
           do i=1,90
              if(symbol(i).eq.quick_molspec%atom_type_sym(iitemp))then
                 quick_molspec%imult = spinmult(i)
                 quick_molspec%chg(1)=i
                 quick_molspec%iattype(1)=i
                 write(ioutfile,'(" ELEMENT = ",a)') symbol(i)
              endif
           enddo
           if (quick_molspec%imult /= 1) quick_method%UNRST= .TRUE.
           quick_molspec%nelec = quick_molspec%iattype(1)
           if ((quick_method%DFT .OR. quick_method%SEDFT).and.quick_method%isg.eq.1) &
                 call gridformSG1()
           call check_quick_method_and_molspec(ioutfile,quick_molspec,quick_method)
  
           !-------------------------------------------
           ! At this point we have the positions and identities of the atoms. We also
           ! have the number of electrons. Now we must assign basis functions. This
           ! is done in a subroutine.
           !-------------------------------------------
           nsenhai=1
           call readbasis(nsenhai,0,0,0,0,ierr)
           CHECK_ERROR(ierr)
           
           atombasis(iitemp)=nbasis
           write (ioutfile,'(" BASIS FUNCTIONS = ",I4)') nbasis
  
           if(nbasis < 1) then
                  call PrtErr(iOutFile,'Unable to find basis set information for this atom.')
                  call PrtMsg(iOutFile,'Update the corresponding basis set file or use a different basis set.')
                  call quick_exit(iOutFile,1)
           endif
  
           ! if quick is called through api multiple times, this is necessary
           if(wrtStep .gt. 1) then
             call dealloc(quick_qm_struct)
           endif
  
           quick_qm_struct%nbasis => nbasis
           call alloc(quick_qm_struct)
           call init(quick_qm_struct)
  
           ! this following subroutine is as same as normal basis set normlization
           call normalize_basis()
           if (quick_method%ecp) call store_basis_to_ecp()
           if (quick_method%DFT .OR. quick_method%SEDFT) call get_sigrad
  
           ! Initialize Density arrays. Create initial density matrix guess.
           present = .false.
           if (quick_method%readdmx) inquire (file=dmxfilename,exist=present)
           if (present) then
              return
           else
              ! Initial Guess
              diagelement=dble(quick_molspec%nelec)/dble(nbasis)
              diagelementb=dble(quick_molspec%nelecb)/dble(nbasis)+1.d-8
              do I=1,nbasis
                 quick_qm_struct%dense(I,I)=diagelement
                 quick_qm_struct%denseb(I,I)=diagelementb
              enddo
           endif
  
           ! AWG Check if SAD file is present when requesting readSAD
           ! AWG If not present fall back to computing SAD guess
           ! AWG note the whole structure of this routine should be improved
           readSAD = quick_method%readSAD
           !readSAD = .false.
           if (readSAD) then
              sadfile = trim(sadGuessDir) // '/' // &
                             trim(quick_molspec%atom_type_sym(iitemp))
              inquire (file=sadfile, exist=present)
              if (.not. present) readSAD = .false.
           end if
 
           ! From SCF calculation to get initial density guess
           if(readSAD) then
  
              open(212,file=sadfile)  !Read from sadfile
              do i=1,nbasis
                 do j=1,nbasis
                    read(212,*) ii,jj,temp
                    atomdens(iitemp,ii,jj)=temp
                 enddo
              enddo
              close(212)
  
           else

#ifdef CEW
              use_cew_save = quick_cew%use_cew
              quick_cew%use_cew = .false.
#endif
              
              ! Compute SAD guess
              call sad_uscf(.true., ierr)
              do i=1,nbasis
                 do j=1,nbasis
                    atomdens(iitemp,i,j)=quick_qm_struct%dense(i,j)+quick_qm_struct%denseb(i,j)
                 enddo
              enddo
#ifdef CEW
              quick_cew%use_cew = use_cew_save
#endif              
              ! write SAD guess if requested
              if(quick_method%writeSAD) then
                 sadfile = trim(quick_molspec%atom_type_sym(iitemp))
                 open(213,file=sadfile)
                 do i=1,nbasis
                    do j=1,nbasis
                       write(213,*) i,j,atomdens(iitemp,i,j)
                    enddo
                 enddo
                 close(213)             
              endif           
  
           endif
  
           call deallocate_calculated
           call dealloc(quick_qm_struct)
        enddo
        call PrtAct(ioutfile,"Finish SAD initial guess")
     endif
  
     natom=natomsaved
     xyz=xyzsaved
  
     quick_method=quick_method_save
     quick_molspec=quick_molspec_save
  !   quick_molspec%imult = quick_molspec_save%imult
  !   quick_molspec%nelec = quick_molspec_save%nelec
  
     bMPI=MPIsaved
  
     return
  
  end subroutine getmolsad

  subroutine get_sad_density_matrix

    use quick_constants_module, only: symbol
    use quick_basis_module, only: atombasis, atomdens 
    use quick_molspec_module, only: natom
    use quick_molspec_module, only: quick_molspec
    use quick_calculated_module, only: quick_qm_struct

    implicit none 
    integer :: n, Iatm, sadAtom, i, j

    n=0
    do Iatm=1,natom
       do sadAtom=1,10

          if(symbol(quick_molspec%iattype(Iatm)).eq. &
                quick_molspec%atom_type_sym(sadAtom))then
             do i=1,atombasis(sadAtom)
                do j=1,atombasis(sadAtom)

                   quick_qm_struct%dense(i+n,j+n)=atomdens(sadAtom,i,j)
                enddo
             enddo
             n=n+atombasis(sadAtom)
          endif
       enddo
    enddo   
 
    call deallocate_mol_sad

  end subroutine get_sad_density_matrix
  
  subroutine allocate_mol_sad(n)
     use quick_basis_module
     integer n
  
     if(.not. allocated(atomDens))  allocate(atomDens(n,100,100))
     if(.not. allocated(atomBasis)) allocate(atomBasis(n))
  
  end subroutine allocate_mol_sad
  
  subroutine deallocate_mol_sad()
     use quick_basis_module
  
     if (allocated(atomDens)) deallocate(atomDens)
     if (allocated(atomBasis)) deallocate(atomBasis)
  
  end subroutine deallocate_mol_sad


  subroutine sad_uscf(verbose,ierr)
     !-------------------------------------------------------
     ! this subroutine is to do scf job for restricted system
     !-------------------------------------------------------
     use allmod
     use quick_overlap_module, only: fullx
     implicit none

     logical :: done
     logical, intent(in) :: verbose
     integer, intent(inout) :: ierr
     integer :: jscf
     done=.false.

     !-----------------------------------------------------------------
     ! The purpose of this subroutine is to perform scf cycles.  At this
     ! point, X has been formed. The remaining steps are:
     ! 1)  Form operator matrix.
     ! 2)  Calculate O' = Transpose[X] O X
     ! 3)  Diagonalize O' to obtain C' and eigenvalues.
     ! 4)  Calculate C = XC'
     ! 5)  Form new density matrix.
     ! 6)  Check for convergence.
     !-----------------------------------------------------------------

     ! Each location in the code that the step is occurring will be marked.
     ! The cycles stop when prms  is less than pmaxrms or when the maximum
     ! number of scfcycles has been reached.
     jscf=0

     if (master) then
       if (verbose) call PrtAct(ioutfile,"Begin SAD USCF")     

       ! calculate the overlap matrix, form transformation matrix
       call fullX
     
       ! Classical Nuclear-Nuclear interaction energy
       quick_qm_struct%Ecore=0.d0      ! atom-extcharge and atom-atom replusion
       quick_qm_struct%ECharge=0d0     ! extcharge-extcharge interaction

       if (quick_method%diisscf .and. .not. quick_method%divcon) call sad_uelectdiis(jscf,verbose,ierr)

       if (verbose) call PrtAct(ioutfile,"End SAD USCF")   
     endif

     jscf=jscf+1

     return

  end subroutine sad_uscf


  subroutine sad_uelectdiis(jscf,verbose,ierr)

     use allmod
     use quick_gridpoints_module
     use quick_scf_module
     use quick_oei_module, only: bCalc1e
     use quick_uscf_module, only: allocate_quick_uscf,deallocate_quick_uscf,alloperatorB

     implicit none

     ! variable inputed to return
     integer :: jscf                ! scf interation
     logical, intent(in) :: verbose
     integer, intent(inout) :: ierr
     logical :: diisdone = .false.  ! flag to indicate if diis is done
     integer :: idiis = 0           ! diis iteration
     integer :: IDIISfinal,iidiis,current_diis
     integer :: lsolerr = 0
     integer :: IDIIS_Error_Start, IDIIS_Error_End
     double precision :: BIJ,DENSEJI,errormax,OJK,temp
     double precision :: Sum2Mat,rms
     integer :: I,J,K,L,IERROR
  
     double precision :: oldEnergy=0.0d0,E1e ! energy for last iteriation, and 1e-energy
     double precision :: PRMS,PRMS2,PCHANGE, tmp
  
     !---------------------------------------------------------------------------
     ! The purpose of this subroutine is to utilize Pulay's accelerated
     ! scf convergence as detailed in J. Comp. Chem, Vol 3, #4, pg 566-60, 1982.
     ! At the beginning of this process, their is an approximate density
     ! matrix.
     ! The step in the procedure are:
     ! 1)  Form the operator matrix for step i, O(i).
     ! 2)  Form error matrix for step i.
     ! e(i) = ODS - SDO
     ! 3)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
     ! 4)  Store the e'(I) and O(i)
     ! 5)  Form matrix B, which is:
     !      _                                                 _
     !     |                                                   |
     !     |  B(1,1)      B(1,2)     . . .     B(1,J)      -1  |
     !     |  B(2,1)      B(2,2)     . . .     B(2,J)      -1  |
     !     |  .            .                     .          .  |
     ! B = |  .            .                     .          .  |
     !     |  .            .                     .          .  |
     !     |  B(I,1)      B(I,2)     . . .     B(I,J)      -1  |
     !     | -1            -1        . . .      -1          0  |
     !     |_                                                 _|
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
     !---------------------------------------------------------------------------
  
     call allocate_quick_uscf(ierr)
  
     if(verbose) write(ioutfile,'(30x," USCF ENERGY")')
     if (quick_method%printEnergy) then
        if(verbose) write(ioutfile,'("| ",72("-"))')
     else
        if(verbose) write(ioutfile,'("| ",42("-"))')
     endif
     if(verbose) write(ioutfile,'("| ","NCYC",6x)',advance="no")
     if(verbose) write(ioutfile,'(" ENERGY ",8x,"DELTA_E",5x)',advance="no")
     if(verbose) write(ioutfile,'(" MAX_ERR",4x,"RMS_CHG",4x,"MAX_CHG")')
     if (quick_method%printEnergy) then
        if(verbose) write(ioutfile,'("| ",72("-"))')
     else
        if(verbose) write(ioutfile,'("| ",42("-"))')
     endif
  
  
     bCalc1e = .true.
     diisdone = .false.
     idiis = 0
     ! Now Begin DIIS
     do while (.not.diisdone)
  
  
        !--------------------------------------------
        ! 1)  Form the operator matrix for step i, O(i).
        !--------------------------------------------
  
        ! Determine dii cycle and scf cycle
        idiis=idiis+1
        jscf=jscf+1
  
        if(idiis.le.quick_method%maxdiisscf)then
           IDIISfinal=idiis; iidiis=idiis
        else
           IDIISfinal=quick_method%maxdiisscf; iidiis=1
        endif
        !-----------------------------------------------
        ! Before Delta Densitry Matrix, normal operator is implemented here
        !-----------------------------------------------
  
        call sad_uscf_operator
  
  
        quick_qm_struct%oSave(:,:) = quick_qm_struct%o(:,:)
        quick_qm_struct%denseOld(:,:) = quick_qm_struct%dense(:,:)
  
        !-----------------------------------------------
        ! 2)  Form error matrix for step i.
        ! e(i) = ODS - SDO
        !-----------------------------------------------
        ! The matrix multiplier comes from Steve Dixon. It calculates
        ! C = Transpose(A) B.  Thus to utilize this we have to make sure that the
        ! A matrix is symetric. First, calculate DENSE*S and store in the scratch
        ! matrix hold.Then calculate O*(DENSE*S).  As the operator matrix is symmetric, the
        ! above code can be used. Store this (the ODS term) in the all error
        ! matrix.
  
        ! The first part is ODS

        quick_scratch%hold=matmul(quick_qm_struct%dense,quick_qm_struct%s)
        quick_scratch%hold2=matmul(quick_qm_struct%o,quick_scratch%hold)
  
        allerror(:,:,iidiis) = quick_scratch%hold2(:,:)
  
        ! Calculate D O. then calculate S (do) and subtract that from the allerror matrix.
        ! This means we now have the e(i) matrix.
        ! allerror=ODS-SDO

        quick_scratch%hold=matmul(quick_qm_struct%dense,quick_qm_struct%o)
        quick_scratch%hold2=matmul(quick_qm_struct%s,quick_scratch%hold)

  
        do I=1,nbasis
           do J=1,nbasis
              allerror(J,I,iidiis) = allerror(J,I,iidiis) - quick_scratch%hold2(J,I) !e=ODS=SDO
           enddo
        enddo

        ! 3)  Form the beta operator matrix for step i, O(i).  (Store in alloperatorb array.)
        quick_qm_struct%obSave(:,:) = quick_qm_struct%ob(:,:)
        quick_qm_struct%densebOld(:,:) = quick_qm_struct%denseb(:,:)
        
        ! 4)  Form beta error matrix for step i.
        ! e(i) = e(i,alpha part)+Ob Db S - S Db Ob
  
        ! First, calculate quick_qm_struct%denseb*S and store in the scratch
        ! matrix hold. Then calculate O*(quick_qm_struct%denseb*S).  As the operator matrix is
        ! symmetric, the above code can be used. Add this (the ODS term) into the allerror
        ! matrix.

        quick_scratch%hold=matmul(quick_qm_struct%denseb,quick_qm_struct%s)
        quick_scratch%hold2=matmul(quick_qm_struct%ob,quick_scratch%hold)

        do I=1,nbasis
           do J=1,nbasis
              allerror(J,I,iidiis) = allerror(J,I,iidiis) + quick_scratch%hold2(J,I) !e=ODS=SDO
           enddo
        enddo

        ! Calculate Db O.Then calculate S (DbO) and subtract that from the allerror matrix.
        ! This means we now have the complete e(i) matrix.

        quick_scratch%hold=matmul(quick_qm_struct%denseb,quick_qm_struct%ob)
        quick_scratch%hold2=matmul(quick_qm_struct%s,quick_scratch%hold)

        errormax = 0.d0
        do I=1,nbasis
           do J=1,nbasis
              allerror(J,I,iidiis) = allerror(J,I,iidiis) - quick_scratch%hold2(J,I) !e=ODS=SDO
              errormax = max(allerror(J,I,iidiis),errormax)
           enddo
        enddo
  
        !-----------------------------------------------
        ! 5)  Move e to an orthogonal basis.  e'(i) = Transpose[X] .e(i). X
        ! X is symmetric, but we do not know anything about the symmetry of e.
        ! The easiest way to do this is to calculate e(i) . X , store
        ! this in HOLD, and then calculate Transpose[X] (.e(i) . X)
        !-----------------------------------------------
        quick_scratch%hold2(:,:) = allerror(:,:,iidiis)

        quick_scratch%hold=matmul(quick_scratch%hold2,quick_qm_struct%x)
        quick_scratch%hold2=matmul(quick_qm_struct%x,quick_scratch%hold)

        allerror(:,:,iidiis) = quick_scratch%hold2(:,:)
        !-----------------------------------------------
        ! 6)  Store the e'(I) and O(i).
        ! e'(i) is already stored.  Simply store the operator matrix in
        ! all operator.
        !-----------------------------------------------
  
        if(idiis.le.quick_method%maxdiisscf)then
           alloperator(:,:,iidiis) = quick_qm_struct%o(:,:)
           alloperatorB(:,:,iidiis) = quick_qm_struct%ob(:,:)
        else
           do K=1,quick_method%maxdiisscf-1
              alloperator(:,:,K) = alloperator(:,:,K+1)
              alloperatorB(:,:,K) = alloperatorB(:,:,K+1)
           enddo
           alloperator(:,:,quick_method%maxdiisscf) = quick_qm_struct%o(:,:)
           alloperatorB(:,:,quick_method%maxdiisscf) = quick_qm_struct%ob(:,:)
        endif
  
        !-----------------------------------------------
        ! 7)  Form matrix B, which is:
        !       _                                                 _
        !       |                                                   |
        !       |  B(1,1)      B(1,2)     . . .     B(1,J)      -1  |
        !       |  B(2,1)      B(2,2)     . . .     B(2,J)      -1  |
        !       |  .            .                     .          .  |
        ! B =   |  .            .                     .          .  |
        !       |  .            .                     .          .  |
        !       |  .            .                     .          .  |
        !       |  B(I,1)      B(I,2)     . . .     B(I,J)      -1  |
        !       | -1            -1        . . .      -1          0  |
        !       |_                                                 _|
  
        ! Where B(i,j) = Trace(e(i) Transpose(e(j)))
        ! According to an example done in mathematica, B12 = B21.  Note that
        ! the rigorous proof of this phenomenon is left as an exercise for the
        ! reader.  Thus the first step is copying BCOPY to B.  In this way we
        ! only have to recalculate the new elements.
        !-----------------------------------------------
        do I=1,IDIISfinal
           do J=1,IDIISfinal
              B(J,I) = BCOPY(J,I)
           enddo
        enddo
  
        if(IDIIS.gt.quick_method%maxdiisscf)then
           do I=1,IDIISfinal-1
              do J=1,IDIISfinal-1
                 B(J,I) = BCOPY(J+1,I+1)
              enddo
           enddo
        endif
  
        ! Now copy the current matrix into HOLD2 transposed.  This will be the
        ! Transpose[ej] used in B(i,j) = Trace(e(i) Transpose(e(j)))
        quick_scratch%hold2(:,:) = allerror(:,:,iidiis)
  
        do I=1,IDIISfinal
           ! Copy the transpose of error matrix I into HOLD.
           quick_scratch%hold(:,:) = allerror(:,:,I) 
  
           ! Calculate and sum together the diagonal elements of e(i) Transpose(e(j))).
           BIJ=Sum2Mat(quick_scratch%hold2,quick_scratch%hold,nbasis)
           
           ! Now place this in the B matrix.
           if(idiis.le.quick_method%maxdiisscf)then
              B(iidiis,I) = BIJ
              B(I,iidiis) = BIJ
           else
              if(I.gt.1)then
                 B(quick_method%maxdiisscf,I-1)=BIJ
                 B(I-1,quick_method%maxdiisscf)=BIJ
              else
                 B(quick_method%maxdiisscf,quick_method%maxdiisscf)=BIJ
              endif
           endif
        enddo
  
        if(idiis.gt.quick_method%maxdiisscf)then
           quick_scratch%hold(:,:) = allerror(:,:,1)
           do J=1,quick_method%maxdiisscf-1
              allerror(:,:,J) = allerror(:,:,J+1)
           enddo
           allerror(:,:,quick_method%maxdiisscf) = quick_scratch%hold(:,:)
        endif
  
        ! Now that all the BIJ elements are in place, fill in all the column
        ! and row ending -1, and fill up the rhs matrix.
  
        do I=1,IDIISfinal
           B(I,IDIISfinal+1) = -1.d0
           B(IDIISfinal+1,I) = -1.d0
        enddo
        do I=1,IDIISfinal
           RHS(I) = 0.d0
        enddo
        RHS(IDIISfinal+1) = -1.d0
        B(IDIISfinal+1,IDIISfinal+1) = 0.d0
  
        ! Now save the B matrix in Bcopy so it is available for subsequent
        ! iterations.
        do I=1,IDIISfinal
           do J=1,IDIISfinal
              BCOPY(J,I)=B(J,I)
           enddo
        enddo
  
        !-----------------------------------------------
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
        !
        !-----------------------------------------------
  
        BSAVE(:,:) = B(:,:)
        call LSOLVE(IDIISfinal+1,quick_method%maxdiisscf+1,B,RHS,W,quick_method%DMCutoff,COEFF,LSOLERR)
  
        IDIIS_Error_Start = 1
        IDIIS_Error_End   = IDIISfinal
        111     IF (LSOLERR.ne.0 .and. IDIISfinal > 0)then
           IDIISfinal=Idiisfinal-1
           do I=1,IDIISfinal+1
              do J=1,IDIISfinal+1
                 B(I,J)=BSAVE(I+IDIIS_Error_Start,J+IDIIS_Error_Start)
              enddo
           enddo
           IDIIS_Error_Start = IDIIS_Error_Start + 1
  
           do i=1,IDIISfinal
              RHS(i)=0.0d0
           enddo
  
           RHS(IDIISfinal+1)=-1.0d0
  
  
           call LSOLVE(IDIISfinal+1,quick_method%maxdiisscf+1,B,RHS,W,quick_method%DMCutoff,COEFF,LSOLERR)
  
           goto 111
        endif
  
        !-----------------------------------------------
        ! 9) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
        ! If the solution to step eight failed, skip this step and revert
        ! to a standard scf cycle.
        !-----------------------------------------------
        ! Xiao HE 07/20/2007,if the B matrix is ill-conditioned, remove the first,second... error vector
        if (LSOLERR == 0) then
           do J=1,nbasis
              do K=1,nbasis
                 OJK=0.d0
                 do I=IDIIS_Error_Start, IDIIS_Error_End
                    OJK = OJK + COEFF(I-IDIIS_Error_Start+1) * alloperator(K,J,I)
                 enddo
                 quick_qm_struct%o(J,K) = OJK
              enddo
           enddo
           
        endif
        !-----------------------------------------------
        ! 10) Diagonalize the operator matrix to form a new density matrix.
        ! First you have to transpose this into an orthogonal basis, which
        ! is accomplished by calculating Transpose[X] . O . X.
        !-----------------------------------------------

        quick_scratch%hold=matmul(quick_qm_struct%o,quick_qm_struct%x)
        quick_qm_struct%o=matmul(quick_qm_struct%x,quick_scratch%hold)
  
        ! Now diagonalize the operator matrix.
  
        call DIAG(nbasis,quick_qm_struct%o,nbasis,quick_method%DMCutoff,V2,quick_qm_struct%E,&
              quick_qm_struct%idegen,quick_qm_struct%vec,IERROR)

  
        ! Calculate C = XC' and form a new density matrix.
        ! The C' is from the above diagonalization.  Also, save the previous
        ! Density matrix to check for convergence.

        quick_qm_struct%co=matmul(quick_qm_struct%x,quick_qm_struct%vec)
  
        quick_scratch%hold(:,:) = quick_qm_struct%dense(:,:) 
  
        ! Form new density matrix using MO coefficients
        do I=1,nbasis
            do J=1,nbasis
                quick_qm_struct%dense(J,I) = 0.d0
                do K=1,quick_molspec%nelec
                    quick_qm_struct%dense(J,I) = quick_qm_struct%dense(J,I) + (quick_qm_struct%co(J,K)* &
                    quick_qm_struct%co(I,K))
                enddo
            enddo
        enddo

        ! Now check for convergence. pchange is the max change
        ! and prms is the rms
        PCHANGE=0.d0
        do I=1,nbasis
           do J=1,nbasis
              PCHANGE=max(PCHANGE,abs(quick_qm_struct%dense(J,I)-quick_scratch%hold(J,I)))
           enddo
        enddo
        PRMS = rms(quick_qm_struct%dense,quick_scratch%hold,nbasis)


        !-----------------------------------------------
        ! 11) Form a new operator matrix based on O(new) = [Sum over i] c(i)O(i)
        ! If the solution to step eight failed, skip this step and revert
        ! to a standard scf cycle.
        !-----------------------------------------------
        ! Xiao HE 07/20/2007,if the B matrix is ill-conditioned, remove the first,second... error vector
        if (LSOLERR == 0) then
           do J=1,nbasis
              do K=1,nbasis
                 OJK=0.d0
                 do I=IDIIS_Error_Start, IDIIS_Error_End
                    OJK = OJK + COEFF(I-IDIIS_Error_Start+1) * alloperatorB(K,J,I)
                 enddo
                 quick_qm_struct%ob(J,K) = OJK
              enddo
           enddo

        endif

        !-----------------------------------------------
        ! 12) Diagonalize the beta operator matrix to form a new beta density matrix.
        ! First you have to transpose this into an orthogonal basis, which
        ! is accomplished by calculating Transpose[X] . O . X.
        !-----------------------------------------------
        quick_scratch%hold=matmul(quick_qm_struct%ob,quick_qm_struct%x)
        quick_qm_struct%ob=matmul(quick_qm_struct%x,quick_scratch%hold)

        ! Now diagonalize the operator matrix.

        call DIAG(nbasis,quick_qm_struct%ob,nbasis,quick_method%DMCutoff,V2,quick_qm_struct%EB,&
              quick_qm_struct%idegen,quick_qm_struct%vec,IERROR)

        ! Calculate C = XC' and form a new density matrix.
        ! The C' is from the above diagonalization.  Also, save the previous
        ! Density matrix to check for convergence.
        !        call DMatMul(nbasis,X,VEC,CO)    ! C=XC'

        quick_qm_struct%cob=matmul(quick_qm_struct%x,quick_qm_struct%vec)

        quick_scratch%hold(:,:) = quick_qm_struct%denseb(:,:)

        ! Form new density matrix using MO coefficients
        do I=1,nbasis
            do J=1,nbasis
                quick_qm_struct%denseb(J,I) = 0.d0
                do K=1,quick_molspec%nelecb
                    quick_qm_struct%denseb(J,I) = quick_qm_struct%denseb(J,I) + (quick_qm_struct%cob(J,K)* &
                    quick_qm_struct%cob(I,K))
                enddo
            enddo
        enddo

        ! Now check for convergence. pchange is the max change
        ! and prms is the rms
        do I=1,nbasis
           do J=1,nbasis
              PCHANGE=max(PCHANGE,abs(quick_qm_struct%denseb(J,I)-quick_scratch%hold(J,I)))
           enddo
        enddo
        PRMS2 = rms(quick_qm_struct%denseb,quick_scratch%hold,nbasis)
        PRMS = MAX(PRMS,PRMS2)


        tmp = quick_method%integralCutoff
        call adjust_cutoff(PRMS,PCHANGE,quick_method,ierr)  !from quick_method_module

        !do I=1,nbasis; do J=1,nbasis
        !  write(*,*) jscf,i,j,quick_qm_struct%dense(j,i),quick_qm_struct%denseb(j,i),&
        !  quick_qm_struct%co(j,i), quick_qm_struct%cob(j,i)
        !enddo; enddo
  
        current_diis=mod(idiis-1,quick_method%maxdiisscf)
        current_diis=current_diis+1
  
        if(verbose) write (ioutfile,'("|",I3,1x)',advance="no") jscf
        if(quick_method%printEnergy)then
           if(verbose) write (ioutfile,'(F16.9,2x)',advance="no") quick_qm_struct%Eel+quick_qm_struct%Ecore
           if (jscf.ne.1) then
              if(verbose) write(ioutFile,'(E12.6,2x)',advance="no") oldEnergy-quick_qm_struct%Eel-quick_qm_struct%Ecore
           else
              if(verbose) write(ioutFile,'(4x,"------",4x)',advance="no")
           endif
           oldEnergy=quick_qm_struct%Eel+quick_qm_struct%Ecore
        endif
        if(verbose) write (ioutfile,'(E10.4,2x)',advance="no") errormax
        if(verbose) write (ioutfile,'(E10.4,2x,E10.4)')  PRMS,PCHANGE
  
        if(verbose .and. lsolerr /= 0) write (ioutfile,'(" DIIS FAILED !!", &
              " PERFORM NORMAL SCF. (NOT FATAL.)")')
  
        if (PRMS < quick_method%pmaxrms .and. pchange < quick_method%pmaxrms*100.d0 .and. jscf.gt.MIN_SCF)then
           if (quick_method%printEnergy) then
              if(verbose) write(ioutfile,'("| ",72("-"))')
           else
              if(verbose) write(ioutfile,'("| ",42("-"))')
           endif
           if(verbose) write (ioutfile,'("| REACH CONVERGENCE AFTER ",i3," CYLCES")') jscf
           if(verbose) write (ioutfile,'("| MAX ERROR = ",E12.6,2x," RMS CHANGE = ",E12.6,2x," MAX CHANGE = ",E12.6)') &
                 errormax,prms,pchange
           if(verbose) write (ioutfile,'("| -----------------------------------------------")')
  
           diisdone=.true.
  
  
        endif
        if(jscf >= quick_method%iscf-1) then
           if(verbose) write (ioutfile,'(" RAN OUT OF CYCLES.  NO CONVERGENCE.")')
           if(verbose) write (ioutfile,'(" PERFORM FINAL NO INTERPOLATION ITERATION")')
           diisdone=.true.
        endif
        diisdone = idiis.gt.MAX_DII_CYCLE_TIME*quick_method%maxdiisscf .or. diisdone
  
        if((tmp .ne. quick_method%integralCutoff).and. .not.diisdone) then
           if(verbose) write(ioutfile, '("| -------------- 2E-INT CUTOFF CHANGE TO ", E10.4, " ------------")') &
           quick_method%integralCutoff
        endif
  
        if(verbose) flush(ioutfile)
  
        !if (quick_method%debug)  call debug_SCF(jscf)
     enddo
  
  
     call deallocate_quick_uscf(ierr)
  
     return
  end subroutine sad_uelectdiis


  subroutine sad_uscf_operator
  !-------------------------------------------------------
  !  The purpose of this subroutine is to form the operator matrix
  !  for a full Hartree-Fock/DFT calculation, i.e. the Fock matrix.  The
  !  Fock matrix is as follows:  O(I,J) =  F(I,J) = KE(I,J) + IJ attraction
  !  to each atom + repulsion_prim
  !  with each possible basis  - 1/2 exchange with each
  !  possible basis. Note that the Fock matrix is symmetric.
  !  This code now also does all the HF energy calculation. Ed.
  !-------------------------------------------------------
     use allmod
     use quick_cutoff_module, only: oshell_density_cutoff
     use quick_oshell_eri_module, only: getOshellEri, getOshellEriEnergy 
     use quick_oei_module, only:get1eEnergy, get1e
  
     implicit none
  
     integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, I, J
     common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
     double precision tst, te, tred
  
     quick_qm_struct%o  = 0.0d0
     quick_qm_struct%ob = 0.0d0
     quick_qm_struct%Eel=0.0d0
  
  !-----------------------------------------------------------------
  !  Step 1. evaluate 1e integrals
  !-----------------------------------------------------------------
  
     call get1e()

     quick_qm_struct%ob(:,:) = quick_qm_struct%o(:,:)
  
     if(quick_method%printEnergy) call get1eEnergy()
  
  !  Delta density matrix cutoff
     call oshell_density_cutoff
  
  !-----------------------------------------------------------------
  ! Step 2. evaluate 2e integrals
  !-----------------------------------------------------------------
  !
  ! The previous two terms are the one electron part of the Fock matrix.
  ! The next two terms define the two electron part.
  !-----------------------------------------------------------------
  !  Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
  !  Reference: Strout DL and Scuseria JCP 102(1995),8448.
  
        do II=1,jshell
           call getOshellEri(II)
        enddo
  
  
  !  Remember the operator is symmetric
     call copySym(quick_qm_struct%o,nbasis)
     call copySym(quick_qm_struct%ob,nbasis)
  
  !  Give the energy, E=1/2*sigma[i,j](Pij*(Fji+Hcoreji))
     if(quick_method%printEnergy) call getOshellEriEnergy
  
  return
  
  end subroutine sad_uscf_operator

end module quick_sad_guess_module
