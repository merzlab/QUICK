!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk), Alex Turner, Salomon Billeter,
!!  Stephan Thiel, Max-Planck Institut fuer Kohlenforshung, Muelheim, 
!!  Germany.
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as 
!!  published by the Free Software Foundation, either version 3 of the 
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public 
!!  License along with DL-FIND.  If not, see 
!!  <http://www.gnu.org/licenses/>.
!!

MODULE dlfhdlc_constraint
!  USE global
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout!,pi
  use dlf_allocate, only: allocate, deallocate
  USE dlfhdlc_matrixlib
  IMPLICIT NONE

!------------------------------------------------------------------------------
! General:
!
! - Reference:
!   Implements J. Baker, J. Chem. Phys. 105 (1996) 192.
!
! - Procedure:
!   1. (done in <interface>.f90:hdlc_get_params)
!      Read in constraints (search.f90:search)
!   2. assign_cons
!      Assign constraints to particular residues and renumber atoms
!      (search.f90:search, after dlc_connect -> assign_cons)
!   3. ci_cons
!      Check in required connections (hdlclib.f90:hdlc_create after
!      hdlc_con_checkin -> ci_cons)
!   4. ck_angle_cons
!      Check in required bends and rots (hdlclib.f90:hdlc_create before
!      end if ! (associated(con)) -> ck_angle_cons)
!   5. ortho_cons
!      Project and orthogonalise active space to constraints
!      (dlc_manager.f:dlc_manager after dlc_make_ut -> ortho_cons)
!   6. split_cons
!      Separate between active and constrained coordinate values
!      (prfossa.f: optef_main after dlc_get_dlc_values and
!      dlc_get_dlc_gradients -> split_cons)
!   7. rest_cons
!      Restore Ut from constrained Ut and unprojected constraints
!      (prfossa.f: optef_main -> rest_cons)
!
! - Data to be passed between steps:
!        context scope  form symbol name description
!   1-2: global  local  a)               constraints specification
!   2-3: residue arg    b)               constraints specification
!   3-5: residue object c)               int. coord. seq. nbrs. of constr.
!   4-5: residue object c)               do.
!   5-6: residue object --   Ut,ncons    transposed V matrix, nbrs. of cns.
!   6-7: residue object d)               constraints values, number
!
! - Form specification:
!   a) iconstr(4,nconstr): i, j, k, l
!      vconstr(nconstr): value
!      atom sequence numbers global
!      k=l=0 -> bond, l=0 -> bond angle, all >0 -> torsion
!      type: array, double(5,ncnstr)
!   b) cns(7,ncons): i, j, k, l, itype, iseq, value; matrix
!      itype = 1: bond; i, j specified
!      itype = 2: bond angle; i, j, k specified
!      itype = 3: torsion; i, j, k, l specified
!      iseq is undefined: see c)
!      value is defined by (see d):
!        sequence number (mcnstr.eq.0)
!        target value (mcnstr.eq.1)
!      i, j, k, l are relative to the residue, ncons is stored per residue
!      object, as is icons(7,ncons) as well
!      type: matrix, (7,N), double
!   c) see b) but iseq points to the internal coordinate to be constrained
!      types: icons: array, (6,N), integer, vcons: array, (N), double
!   d) see b) but ivalue points to the target value in constraints(ivalue)
!      if extended Lagrangean is chosen, ivalue is kept from before
!   constraints(nconsvals) is stored globally
!
! - Notes:
!   * U/natoms <-> V/natoms/ncons/ (C/C_proj)?
!   * ncons has to be subtracted from n6 at each occurrence of if(internal)
!   * Correspondence between 'rows' and 'columns' by Alex Turner, Fortran
!     and mathematics is straightforward: the place of the index is always
!     the same, no matter whether the faster index or not.
!     First index: row, second index: column; a(row,column)
!
! changes by Johannes Kaestner (JK): bondlength-difference implemented (Jan 2005)
!  this is implemented with some drawbacks: the input is { diffbond at1 at2 at3 -1 }
!  -1 is required as no information on the keyword is transferred from hdlcopt2.tcl
!  to constraint.f90. Additionally, the bond indices are transferred in icons(4,:) and 
!  icons(6,:). Usually, icons(6,:) should be used, but two indices are required here.
!------------------------------------------------------------------------------

CONTAINS

!------------------------------------------------------------------------------
! subroutine get_cons_regions
!
! Arguments:
! nconstr:           number of constraints provided (in)
! iconstr(4,ncnstr): constraints as read in, format a (in)
! nat:               number of atoms (in)
! spec(nat):         residue/frozen atom specification (in)
! micspec(nat):      microiterative specification (in)
! nincons:           number of constraints in inner region (out) 
! noutcons:          number of constraints in outer region (out)
!
! Description: determine whether constraints apply to the inner or outer 
!              microiterative region for the purpose of counting the number
!              of internal coordinates in each region.
!              Also checks that no constraints have been set on frozen atoms
!------------------------------------------------------------------------------

  subroutine get_cons_regions(nconstr, iconstr, nat, spec, micspec, nincons, noutcons)
    ! arguments
    integer, intent(in) :: nconstr
    integer, dimension (5,nconstr), intent(in) :: iconstr ! type, atom1, atom2, atom3, atom4
    integer, intent(in) :: nat
    integer, intent(in) :: spec(nat)
    integer, intent(in) :: micspec(nat)
    integer, intent(out):: nincons
    integer, intent(out):: noutcons
    ! local vars
    integer :: kc, ncat, i, icat, iregion

    nincons = 0
    noutcons = 0

    do kc = 1, nconstr
       ! determine number of atoms to check
       select case (iconstr(1,kc))
       case (1)
          ncat = 2 ! Bond
       case (2)
          ncat = 3 ! Angle
       case (3)
          ncat = 4 ! Torsion
       case (4) 
          ncat = 1 ! Cartesian
       case (5)
          ncat = 3 ! Bond difference
       case default
          WRITE (stdout,'(A,I5,A,I5,A)') 'Constraint ', kc, ' type ', iconstr(1,kc), &
               ' not recognised!'
          CALL hdlc_errflag('Constraints error','stop')
       end select

       ! Determine whether constraint is in inner or outer region
       iregion = -1
       do i = 2, ncat + 1
          icat = iconstr(i,kc)
          if (spec(icat) == -1) then
             write(stdout,'(A,I5,A,I5)') 'Constraint ', kc, ' includes frozen atom ', icat
             write(stdout,'(A,5I5)') 'iconstr=', iconstr(1:5,kc)
             call hdlc_errflag('Constraints error','stop')
          end if
          if (iregion == -1) then
             iregion = micspec(icat)
          else if (iregion /= micspec(icat)) then
             write(stdout,'(A,I5,A)') 'Constraint ', kc, ' crosses inner/outer boundary!'
             write(stdout,'(A,5I5)') 'iconstr=', iconstr(1:5,kc)
             call hdlc_errflag('Constraints error','stop')
          end if
       end do

       if (iregion == 1) then
          nincons = nincons + 1
       else if (iregion == 0) then
          noutcons = noutcons + 1
       else
          write(stdout,'(A,I5,A,I5)') 'Constraint ', kc, ' iregion=', iregion
          write(stdout,'(A,5I5)') 'iconstr=', iconstr(1:5,kc)          
          call hdlc_errflag('Constraints error','stop')
       end if
    end do
     
   end subroutine get_cons_regions

!------------------------------------------------------------------------------
! subroutine assign_cons
!
! Arguments:
! cns(7,ncns):       matrix of constraints of one residue, format b (out)
! iconstr(4,ncnstr): constraints as read in, format a (in)
! nconstr:           number of constraints provided (in)
! mconstr:           type of constraints (in)
! vconstr(nconstr):  target or actual values (in)
! nat:               number of atoms (in)
! at:                array of atom indices (in)
! ncns:              number of constraints assigned in this residue (out)
!
! Description: step 2 of the procedure described above
!------------------------------------------------------------------------------

  SUBROUTINE assign_cons(cns,iconstr,nconstr,vconstr,mconstr,nat,at, &
      internal,ncns)

! args
    LOGICAL internal
    INTEGER nconstr, mconstr
    INTEGER, DIMENSION (5,nconstr) :: iconstr ! type, atom1, atom2, atom3, atom4
    REAL (rk), DIMENSION (nconstr) :: vconstr
    TYPE (matrix) :: cns
    integer, intent(in) :: nat
    integer, intent(in) :: at(nat)
    integer, intent(out):: ncns

! local vars
    LOGICAL li, lj, lk, ll
    INTEGER i, iatom, j, k, kc, l, m, mcomp, type,iat
    REAL (rk), DIMENSION (:,:), ALLOCATABLE :: icns_tmp
    integer :: ni,nj,nk,nl

! begin, allocate three times more space for the Cartesian components
    IF (printl>=5) WRITE (stdout,'(3X,A,/)') 'Looking for constraints'
    ncns = 0
    call allocate (icns_tmp,7,nconstr*3)

! loop over constraints and check if they affect the current residue
    DO kc = 1, nconstr
      i = iconstr(2,kc)
      j = iconstr(3,kc)
      k = iconstr(4,kc)
      l = iconstr(5,kc)

      li=.false.
      lj=.false.
      lk=.false.
      ll=.false.
      ni=0; nj=0; nk=0; nl=0;
      do iat=1,nat
        if(at(iat)==i) then
          li=.true.
          ni=iat
        end if
        if(at(iat)==j) then
          lj=.true.
          nj=iat
        end if
        if(at(iat)==k) then
          lk=.true.
          nk=iat
        end if
        if(at(iat)==l) then
          ll=.true.
          nl=iat
        end if
      end do
      IF (li .OR. lj .OR. lk .OR. ll) THEN
        IF (iconstr(1,kc)==3 .and. li .AND. lj .AND. lk .AND. ll) THEN
          type = 3 ! torsion
        ELSE IF (iconstr(1,kc)==2 .and. li .AND. lj .AND. lk) THEN
          type = 2 ! angle (bend)
        ELSE IF (iconstr(1,kc)==1 .AND. li .AND. lj) THEN
          type = 1 ! bond
        ELSE IF (iconstr(1,kc)==4 .and. (l<=0) .AND. (k<=0) &
            .AND. (j<=0) .AND. li) THEN
          type = 4 ! cartesian
! type 5 included by JK
        ELSE IF (iconstr(1,kc)==5 .and. li .AND. lj .AND. lk) THEN
          type = 5 ! bond difference
        ELSE
          WRITE (stdout,'(A,I5,A,I5,A,I5,A,I5,A)') 'Constraint ', i, ' - ', j, &
            ' - ', k, ' - ', l, ' crosses residue boundary!'
          CALL hdlc_errflag('Constraints error','stop')
        END IF

! found an internal constraint
        IF (type/=4) THEN
          ncns = ncns + 1
          IF (printl>=2) THEN
            IF (type<5) THEN ! included by JK
              WRITE (stdout,'(3X,A,4I7,/)') 'Found constraint between atoms ', &
                (iconstr(m+1,kc),m=1,type+1)
            else
              WRITE (stdout,'(3X,A,4I7,/)') 'Found constraint between atoms ', &
                (iconstr(m+1,kc),m=1,3)
            END IF
          END IF
          icns_tmp(1,ncns)=real(ni,rk)
          icns_tmp(2,ncns)=real(nj,rk)
          icns_tmp(3,ncns)=real(nk,rk)
          icns_tmp(4,ncns)=real(nl,rk)
!!$          DO m = 1, 4
!!$            IF (iconstr(m+1,kc)/=0) THEN
!!$              icns_tmp(m,ncns) = real((iconstr(m+1,kc)-(start-1)),rk)
!!$            ELSE
!!$              icns_tmp(m,ncns) = 0.0D0
!!$            END IF
!!$          END DO
! JK
          IF (type==5) icns_tmp(4,ncns) = 0.0D0
          icns_tmp(5,ncns) = real(type,rk)
          icns_tmp(6,ncns) = 0.0D0
          IF (mconstr>=1) THEN
            icns_tmp(7,ncns) = vconstr(kc)
          ELSE
            icns_tmp(7,ncns) = 0.0D0
          END IF

! found Cartesian constraints
        ELSE ! (type .ne. 4)
          IF (internal) THEN
            WRITE (stdout,'(A,I5,A)') 'Constraint ', i, &
              ': no Cartesian components can be constrained with DLC'
            CALL hdlc_errflag('Constraints error','stop')
          END IF
          ncns = ncns + 1
          iatom = iconstr(2,kc) !1 + int((iconstr(2,kc)-1)/3)
          mcomp = -iconstr(4,kc) !iconstr(2,kc) - 3*(iatom-1)
          IF (printl>=2) THEN
            WRITE (stdout,'(3X,A,4I5,/)') 'Found constrained Cartesian, atom ' &
              , iatom, ', component ', mcomp
          END IF
          icns_tmp(1,ncns) = real(mcomp+ni,rk)
          DO m = 2, 4
            icns_tmp(m,ncns) = 0.0D0
          END DO
          icns_tmp(5,ncns) = real(type,rk)
          icns_tmp(6,ncns) = 0.0D0
          IF (mconstr>=1) THEN
            icns_tmp(7,ncns) = vconstr(kc)
          ELSE
            icns_tmp(7,ncns) = 0.0D0
          END IF
        END IF ! (type .le. 3) ... else
      END IF ! (li .or. lj .or. lk .or. ll)
    END DO ! kc = 1,nconstr
    cns = matrix_create(7,ncns,'constraints')
    DO kc = 1, ncns
      i = matrix_set_column(cns,7,icns_tmp(1:7,kc),kc)
    END DO
    call deallocate (icns_tmp)
  END SUBROUTINE assign_cons

!------------------------------------------------------------------------------
! subroutine ci_cons
!
! Description:
! Step 3 of the procedure described above. The constraints provided in cns are
! copied to icons, and all connections not yet occurring in iconn are added.
! For bonds, the bond sequence number is stored in icons(iseq,*).
! The target value of the constraints is copied to vcons. This only matters if
! the extended Lagrangean method is chosen.
!
! Arguments:
! cns   matrix(7,ncons) (in)  constraint specification for the residue
! ncons                 (in)  number of constraints provided in cns
! icons (6,ncons)       (out) constraint specification copied from cns
! vcons (ncons)         (out) constraint values copied from cns
! nconn                 (i/o) number of connections between atoms in the res.
! iconn (2,nconn)       (i/o) connections list, new pointer returned
!------------------------------------------------------------------------------

  SUBROUTINE ci_cons(cns,ncons,icons,vcons,nconn,iconn)

! args
    TYPE (matrix) :: cns
    INTEGER ncons, nconn
    INTEGER, POINTER, DIMENSION (:,:) :: icons, iconn
    REAL (rk), POINTER, DIMENSION (:) :: vcons

! local vars
    INTEGER, allocatable, DIMENSION (:,:) :: iconn_tmp
    INTEGER, pointer, DIMENSION (:,:) :: iconn_new
    INTEGER el, k, kc
    INTEGER ibnd, jbnd, kconn, kibnd, kjbnd, ktype, nconn_new

! begin, allocate space to hold potential new connections
    nconn_new = nconn
    ! 3*ncons is the maximum number of required additional bonds
    !  (if all constraints are torsions and no bonds defined ...)
    call allocate (iconn_tmp,2,3*ncons)

! copy the constraints
    DO kc = 1, ncons
      DO el = 1, 6
        icons(el,kc) = int(cns%data(el,kc))
      END DO
      vcons(kc) = cns%data(7,kc)

! check all constraints including angles and torsions
      ktype = icons(5,kc)
      jbnd = icons(1,kc)
      IF (printl>=4) THEN
        IF (ktype<5) THEN
          WRITE (stdout,'(5X,A,I1,A,4I5)') 'Considering constraint (type ', &
            ktype, '): ', (icons(k,kc),k=1,ktype+1)
        ELSE
          WRITE (stdout,'(5X,A,I1,A,4I5)') 'Considering constraint (type ', &
            ktype, '): ', (icons(k,kc),k=1,3)
        END IF
      END IF
      DO k = 1, ktype
        IF (ktype==5 .AND. k>=3) EXIT ! catch bond difference
        IF (ktype==4) exit ! no bonds needed for cartesian constraints
        ibnd = jbnd
        jbnd = icons(1+k,kc)

! check if the connection ibnd - jbnd already occurs
        DO kconn = 1, nconn
          kibnd = iconn(1,kconn)
          IF (ibnd==kibnd .OR. jbnd==kibnd) THEN
            kjbnd = iconn(2,kconn)
            IF (ibnd==kjbnd .OR. jbnd==kjbnd) THEN

! connection is already defined - store its sequence number and break the loop
              IF (ktype==1) THEN
                icons(6,kc) = kconn
                IF (printl>=4) THEN
                  WRITE (stdout,'(7X,A,I4)') '... constraining bond ', kconn
                END IF
              END IF
              IF (ktype==5) THEN
! jk: take care of bond difference
! the positive contribution is stored in icons(6,ic)
! the negative contribution is stored in icons(4,ic)      
                IF (k==1) THEN
                  icons(6,kc) = kconn
                  IF (printl>=4) THEN
                    WRITE (stdout,'(7X,A,I4)') '... positive part of diffbond: ', kconn
                  END IF
                ELSE ! the case if k=2
                  icons(4,kc) = kconn
                  IF (printl>=4) THEN
                    WRITE (stdout,'(7X,A,I4)') '... negative part of diffbond: ', kconn
                  END IF
                END IF
              END IF
              GO TO 10
            END IF
          END IF
        END DO ! kconn = 1,nconn

! connection is not yet defined - do it now
        nconn_new = nconn_new + 1
        iconn_tmp(1,nconn_new-nconn) = ibnd
        iconn_tmp(2,nconn_new-nconn) = jbnd
        IF (ktype==1) THEN
          icons(6,kc) = nconn_new
          IF (printl>=4) THEN
            WRITE (stdout,'(9X,A,I4,A,I4,A,I4,/)') '... adding stretch ', &
              ibnd, ' - ', jbnd, ' to constrain it as bond ', nconn_new
          END IF
        ELSE
          IF (printl>=4) THEN
            WRITE (stdout,'(9X,A,I4,A,I4,/)') '... adding stretch ', ibnd, &
              ' - ', jbnd
          END IF
        END IF
        IF (ktype==5) THEN
! jk: take care of bond difference
! the positive contribution is stored in icons(6,ic)
! the negative contribution is stored in icons(4,ic)      
          IF (k==1) THEN
            icons(6,kc) = nconn_new
          ELSE ! the case if k=2
            icons(4,kc) = nconn_new
          END IF
        END IF

! jump here if the connection is already defined
10      CONTINUE
      END DO ! do k = 1,ktype
    END DO ! kc = 1,ncons
! add new connections to the old ones and reset pointers
    IF (nconn_new>nconn) THEN
      ALLOCATE (iconn_new(2,nconn_new))
      DO k = 1, nconn
        iconn_new(1,k) = iconn(1,k)
        iconn_new(2,k) = iconn(2,k)
      END DO
      DO k = 1, nconn_new - nconn
        iconn_new(1,k+nconn) = iconn_tmp(1,k)
        iconn_new(2,k+nconn) = iconn_tmp(2,k)
      END DO
      IF (associated(iconn)) DEALLOCATE (iconn)
      iconn => iconn_new
    END IF
    call deallocate (iconn_tmp)
    nconn = nconn_new
  END SUBROUTINE ci_cons

!------------------------------------------------------------------------------
! subroutine ck_angle_cons
!
! Description:
! Step 4 of the procedure described above. The angles to be constrained must
! have been found already. Check their sequence numbers and store them.
! Only bond angles are scanned (ibend(4,*).eq.0), no impropers or linears.
!
! Arguments:
! ncons           (in)  number of constraints
! icons (6,ncons) (i/o) constraint specification
! nconn           (in)  number of connections between atoms in the res.
! nbend           (i/o) number of bends
! ibend (4,nbend) (i/o) bends list
! nrots           (i/o) number of torsions
! irots (4,nrots) (i/o) torsions list
!------------------------------------------------------------------------------

  SUBROUTINE ck_angle_cons(ncons,icons,nconn,ibend,nbend,irots,nrots)

! to allocate additional memory for missing torsions and bond angles
    use dlfhdlc_primitive, only : rots_grow

! args
    INTEGER ncons, nconn, nbend, nrots
    INTEGER, pointer, dimension (:,:) :: ibend, irots
    INTEGER, POINTER, DIMENSION (:,:) :: icons

! local vars
    INTEGER k, kangle, kc, ktype, ni
    integer maxbend, maxrots

! begin
    DO kc = 1, ncons
      ktype = icons(5,kc)

! bends
      IF (ktype==2) THEN

        DO kangle = 1, nbend
          IF (ibend(4,kangle)==0) THEN
            IF (ibend(2,kangle)==icons(2,kc)) THEN
              IF ((ibend(1,kangle)==icons(1,kc) .AND. ibend(3, &
                  kangle)==icons(3,kc)) .OR. (ibend(1,kangle)==icons(3, &
                  kc) .AND. ibend(3,kangle)==icons(1,kc))) THEN
                GO TO 10
              END IF
            END IF
          END IF
        END DO

! angle not found - add it to the list of primitive coordinates
        IF (printl>=4) WRITE (stdout,'(A,I5,A,I5,A,I5,A)') &
          'Constrained angle ', icons(1,kc), &
          ' - ', icons(2,kc), ' - ', icons(3,kc), &
          ' not found - adding to primitive list'
        ! Allocate additional space for the new bend if necessary
        maxbend = size(ibend,2)
        if (maxbend == nbend) then
           ! We can reuse rots_grow here because ibend has the same
           ! first dimension size as irots (4).
           call rots_grow(ibend,maxbend,maxbend+1,maxbend)
        endif
        nbend = nbend + 1
        kangle = nbend
        ibend(1:3, kangle) = icons(1:3, kc)
        ibend(4, kangle) = 0

! angle found
10      CONTINUE
        ni = kangle
        DO k = 1, kangle
          IF (ibend(4,k)<0 .AND. ibend(1,k)/=0) ni = ni + 1
        END DO
        IF (printl>=4) WRITE (stdout,'(5X,A,I3,A,4I5)') &
          'Constraining angle ', ni, ': ', (icons(k,kc),k=1,3)
        icons(6,kc) = ni + nconn

! torsions
      ELSE IF (ktype==3) THEN

        DO kangle = 1, nrots

          IF (irots(2,kangle)==icons(2,kc)) THEN
            IF (irots(1,kangle)==icons(1,kc) .AND. irots(3,kangle)==icons(3,kc &
              ) .AND. irots(4,kangle)==icons(4,kc)) GO TO 20
          ELSE IF (irots(2,kangle)==icons(3,kc)) THEN
            IF (irots(1,kangle)==icons(4,kc) .AND. irots(3,kangle)==icons(2,kc &
              ) .AND. irots(4,kangle)==icons(1,kc)) GO TO 20
          END IF
        END DO

! torsion not found - add it to the list of primitive coordinates
        IF (printl>=4) WRITE (stdout,'(A,I5,A,I5,A,I5,A,I5,A)') &
          'Constrained torsion ', &
          icons(1,kc), ' - ', icons(2,kc), ' - ', icons(3,kc), ' - ', &
          icons(4,kc), ' not found - adding to primitive list'
        ! Allocate additional space for the new torsion if necessary
        maxrots = size(irots,2)
        if (maxrots == nrots) then
          call rots_grow(irots,maxrots,maxrots+1,maxrots)
        endif
        nrots = nrots + 1
        kangle = nrots
        irots(1:4,kangle) = icons(1:4,kc)

! torsion found - add stretches, bends and linears to the sequence number
20      CONTINUE
        ni = kangle
        DO k = 1, nbend
          IF (ibend(4,k)<0 .AND. ibend(1,k)/=0) ni = ni + 1
        END DO
        IF (printl>=4) WRITE (stdout,'(5X,A,I3,A,4I5)') &
          'Constraining torsion ', kangle, ': ', (icons(k,kc),k=1,4)
        icons(6,kc) = ni + nconn + nbend

! Cartesians - add stretches, bends and linears to the sequence number
      ELSE IF (ktype==4) THEN ! (ktype.eq.2) .. elseif (ktype.eq.3)
        ni = icons(1,kc)
        kangle = ni
        DO k = 1, nbend
          IF (ibend(4,k)<0 .AND. ibend(1,k)/=0) ni = ni + 1
        END DO
        IF (printl>=4) WRITE (stdout,'(5X,A,I3)') &
          'Constraining Cartesian ', kangle
        icons(6,kc) = ni + nconn + nbend
      END IF ! (ktype.eq.2) .. elseif (ktype.eq.4)
    END DO ! kc = 1,ncons
  END SUBROUTINE ck_angle_cons

!------------------------------------------------------------------------------
! subroutine ortho_cons
!
! Implements the method for adding constraints to natural DLC
! described by Baker et al.
! based on Schmidt orthogonalisation of the column vectors of U to the
! projected constraint vectors C
!
! The np-dimensional space of the internal, primitive, local coordinates is
! represented as in 
!
! Input:
!   Ut(m,np): Transpose of the nonredundant eigenspace of G
!     m rows: nonredundant eigenvectors of G represented using
!     np columns: primitive, local, internal coordinates
!   nc: number of internal coordinates to be constrained
!
! Output:
!   Ut(m-nc,np): Transpose of the active, nonredundant eigenspace of G
!   Ut(m-nc..m,np): Transpose of the constrained eigenspace of G
!     m rows: see above, orthogonal to the space {C(k), k=1..nc} projected
!
! Temporarily used:
!   C(np,nc): Constraints
!     np rows: primitive, local, internal coordinates
!     nc columns: constraint vectors represented using
!   np: number of primitive internal coordinates
!   m: dimension of the nonredundant eigenspace of G
!
! Description: step 5 of the procedure described above
!------------------------------------------------------------------------------


  SUBROUTINE ortho_cons(ut_mat,nc,icons)

! args
    INTEGER       :: nc, icons(6,nc)
    TYPE (matrix) :: ut_mat

! local vars
    INTEGER j, m, np
    TYPE (matrix) :: c_mat, v_mat
    REAL (rk), DIMENSION (:), ALLOCATABLE :: work

! begin, find dimensions, allocate space, np is always .gt. m
    m = matrix_dimension(ut_mat,1)
    np = matrix_dimension(ut_mat,2)
    c_mat = matrix_create(np,nc,'C matrix')
    v_mat = matrix_create(np,m+nc,'V matrix')
    call allocate (work,np+m)

! set constraints matrix and project constrained coordinates
    CALL gen_cons(c_mat%data,icons,nc,np)
    CALL proj_cons(c_mat%data,ut_mat%data,work,m,nc,np)

! now we have Ut and Cp, start orthogonalisation
    CALL merge_cons(v_mat,c_mat,ut_mat,work,m,nc,np)
    CALL ortho_mat(v_mat%data,work,m,nc,np)

! move the constraints to the end of the non-zero vectors and transpose
    CALL move_cons(v_mat,ut_mat,work,m,nc,np)

! replace projected by unprojected constraints
    CALL gen_cons(c_mat%data,icons,nc,np)
    CALL unproj_cons(ut_mat,c_mat,work,m,nc,np)
    j = matrix_destroy(c_mat)
    j = matrix_destroy(v_mat)
    call deallocate (work)

  END SUBROUTINE ortho_cons

!------------------------------------------------------------------------------
! subroutine split_cons
!
! Is called by the 'method routine' hdlc_split_cons of hdlc_manager, but is
! implemented separately.
! matrix_set copies the data rather than setting the handle to the data
!
! Arguments:
! hdlc:    matrix of full HDLC coordinates (in)
!          matrix of active HDLC coordinates (out)
! lstore:  constrained HDLC coordinates are stored to vcons if (lstore) (in)
! natom:   number of atoms in the residue (in)
! ncons:   number of constraints in the residue (in)
! vcons:   (ncons) values of the constrained coordinates (out)
!
! Description: step 6 of the procedure described above
!------------------------------------------------------------------------------

  SUBROUTINE split_cons(hdlc,lstore,linternal,natom,ncons,vcons)

! args
    LOGICAL lstore, linternal
    INTEGER natom, ncons
    REAL (rk), DIMENSION (ncons) :: vcons
    TYPE (matrix) :: hdlc

! local vars
    INTEGER idum, ncoords, nactive
    REAL (rk), DIMENSION (:), ALLOCATABLE :: dhdlc

! begin
    ncoords = 3*natom
    if(linternal) ncoords = ncoords - 6
    nactive = ncoords - ncons

! move all data to temp array
    call allocate (dhdlc,ncoords)
    idum = matrix_get(hdlc,ncoords,dhdlc)
    idum = matrix_destroy(hdlc)

! store data to new HDLC matrix and residue%vcons if requested
    hdlc = matrix_create(nactive,1,'HDLC active')
    idum = matrix_set(hdlc,ncoords,dhdlc)
    IF (lstore) THEN
      CALL copy_coords(vcons,dhdlc(nactive+1),ncons)
    END IF
    call deallocate (dhdlc)

! end split_cons
  END SUBROUTINE split_cons

!------------------------------------------------------------------------------
! subroutine rest_cons
!
! Is called by the 'method routine' hdlc_rest_cons of hdlc_manager, but is
! implemented separately.
! matrix_set copies the data rather than setting the handle to the data
!
! Arguments:
! hdlc:    matrix of active HDLC coordinates (in)
!          matrix of full HDLC coordinates (out)
! natom:   number of atoms in the residue (in)
! ncons:   number of constraints in the residue (in)
! vcons:   (ncons) values of the constrained coordinates (in)
!
! Description: step 7 of the procedure described above
!------------------------------------------------------------------------------

  SUBROUTINE rest_cons(hdlc,linternal,natom,ncons,vcons)

! args
    INTEGER natom, ncons
    logical linternal
    REAL (rk), DIMENSION (ncons) :: vcons
    TYPE (matrix) :: hdlc

! local vars
    INTEGER idum, ncoords, nactive
    REAL (rk), DIMENSION (:), ALLOCATABLE :: dhdlc

! begin
    ncoords = 3*natom
    if(linternal) ncoords = ncoords - 6
    nactive = ncoords - ncons

! move all data from old HDLC matrix and vcons to temp array
    call allocate (dhdlc,ncoords)
    idum = matrix_get(hdlc,ncoords,dhdlc)
    idum = matrix_destroy(hdlc)
    CALL copy_coords(dhdlc(nactive+1),vcons,ncons)

! store data to new HDLC
    hdlc = matrix_create(ncoords,1,'HDLC all')
    idum = matrix_set(hdlc,ncoords,dhdlc)
    call deallocate (dhdlc)

! end split_cons
  END SUBROUTINE rest_cons

!------------------------------------------------------------------------------
! Helpers
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
! This tiny helper is self explaining
!------------------------------------------------------------------------------

  SUBROUTINE copy_coords(target,source,length)

! args
    INTEGER length
    REAL (rk), DIMENSION (length) :: target, source

! local vars
    INTEGER i

! begin
    DO i = 1, length
      target(i) = source(i)
    END DO

! end copy_coords
  END SUBROUTINE copy_coords

!------------------------------------------------------------------------------
! subroutine gen_cons
!
! Generates a constraints matrix in the primitive internal basis from the
! specification in icons(iseq,*)
!
! Arguments:
! cdat(np,nc): constraints matrix (out)
! icons(6,nc): constraints specification (in)
! nc:          number of constraints (in)
! np:          dimension of the primitive space
!------------------------------------------------------------------------------

  SUBROUTINE gen_cons(cdat,icons,nc,np)

! args
    INTEGER nc, np, icons(6,nc)
    REAL (rk), DIMENSION (np,nc) :: cdat

! local vars
    INTEGER ip, ic
! begin
    DO ic = 1, nc
      DO ip = 1, np
        cdat(ip,ic) = 0.0D0
      END DO
      cdat(icons(6,ic),ic) = 1.0D0
! jk: take care of bond difference
! the positive contribution is stored in icons(6,ic)
! the negative contribution is stored in icons(4,ic)      
      IF (icons(5,ic)==5) THEN
        cdat(icons(4,ic),ic) = -dsqrt(2.D0)
        cdat(icons(6,ic),ic) = dsqrt(2.D0)
      END IF
    END DO

! end gen_cons
  END SUBROUTINE gen_cons

!------------------------------------------------------------------------------
! subroutine proj_cons
!
! Projects the vectors in the matrix c into the space spanned by utmat
!
! Arguments:
! cdat(np,nc): unprojected constraints matrix (in) / projected matrix (out)
! utdat(m,np): m-dimensional space of vectors of dimension np (in)
! work(m):     scratch array, used for dp_{ic,j} (in)
! m:           dimension of the space spanned by utmat (nonredundant) (in)
! nc:          number of constraints (in)
! np:          dimension of the space in which utmat is represented (in)
!------------------------------------------------------------------------------

  SUBROUTINE proj_cons(cdat,utdat,work,m,nc,np)

! args
    INTEGER nc, np, m
    REAL (rk), DIMENSION (m) :: work
    REAL (rk), DIMENSION (np,nc) :: cdat
    REAL (rk), DIMENSION (m,np) :: utdat

! local vars
    INTEGER ic, ip, j

! begin, dp_{ic,j} = <C_ic|U_j>
    DO ic = 1, nc
      DO j = 1, m
        work(j) = 0.0D0
        DO ip = 1, np
          work(j) = work(j) + utdat(j,ip)*cdat(ip,ic)
        END DO
      END DO

! C_ic = 0
      DO ip = 1, np
        cdat(ip,ic) = 0.0D0
      END DO

! C_ic = sum_j dp_{ic,j}*U_j
      DO j = 1, m
        DO ip = 1, np
          cdat(ip,ic) = cdat(ip,ic) + work(j)*utdat(j,ip)
        END DO
      END DO
    END DO ! ic = 1,nc

! end proj_cons
  END SUBROUTINE proj_cons

!------------------------------------------------------------------------------
! subroutine merge_cons
!
! Composes a new matrix V out of [C,Ut]
!
! Arguments:
! v_mat:  matrix (np,m+nc) of [cmat,utmat transposed] (out)
! c_mat:  matrix (np,nc) of projected constraints (in)
! ut_mat: matrix (m,np) of U transposed (in)
! work:   work array(np) (in)
! m:      dimension of the space spanned by utmat (nonredundant) (in)
! nc:     number of constraints (in)
! np:     dimension of the space in which utmat is represented (in)
!------------------------------------------------------------------------------

  SUBROUTINE merge_cons(v_mat,c_mat,ut_mat,work,m,nc,np)

! args
    INTEGER m, nc, np
    TYPE (matrix) :: v_mat, c_mat, ut_mat
    REAL (rk), DIMENSION (np) :: work

! local vars
    INTEGER i, idum, j

! begin
    DO i = 1, nc
      idum = matrix_get_column(c_mat,size(work),work,i)
      idum = matrix_set_column(v_mat,size(work),work,i)
    END DO
    j = nc
    DO i = 1, m
      j = j + 1
      idum = matrix_get_row(ut_mat,size(work),work,i)
      idum = matrix_set_column(v_mat,size(work),work,j)
    END DO

! end merge_cons
  END SUBROUTINE merge_cons

!------------------------------------------------------------------------------
! subroutine ortho_mat
!
! Applies Schmidt orthogonalisation to the columns of vmat
! Taken from mankyopt: orthog.F
!
! Arguments:
! v_dat(np,nc+m): [cmat,utmat transposed] (in), orthogonalised (out)
! work(np):       work array (in)
! m, nc, np:      see above (in)
!------------------------------------------------------------------------------

  SUBROUTINE ortho_mat(v_dat,work,m,nc,np)

! args
    INTEGER m, nc, np
    REAL (rk), DIMENSION (np,nc+m) :: v_dat
    REAL (rk), DIMENSION (np) :: work

! local vars
    INTEGER i, j, k, nelem, nvec
    REAL (rk) dnorm, scapro, tol

! data
    DATA tol/1.0D-10/

! begin, orthogonalise vectors i = 2,nvec
    nvec = m + nc
    nelem = np
    DO i = 1, nvec
      DO j = 1, nelem
        work(j) = 0.0D0
      END DO

! make vector i orthogonal to vectors  k = 1,i-1
      DO k = 1, i - 1
        scapro = 0.0D0
        DO j = 1, nelem
          scapro = scapro + v_dat(j,i)*v_dat(j,k)
        END DO
        DO j = 1, nelem
          work(j) = work(j) + v_dat(j,k)*scapro
        END DO
      END DO

! subtract the collinear vector to make vector i orthogonal to k = 1,i-1
      DO j = 1, nelem
        v_dat(j,i) = v_dat(j,i) - work(j)
      END DO

! normalise vector i
      dnorm = 0.0D0
      DO j = 1, nelem
        dnorm = dnorm + v_dat(j,i)*v_dat(j,i)
      END DO
      IF (abs(dnorm)<tol) THEN
        dnorm = 0.0D0
      ELSE
        dnorm = 1.0D0/dsqrt(dnorm)
      END IF
      DO j = 1, nelem
        v_dat(j,i) = v_dat(j,i)*dnorm
      END DO
    END DO

! report resulting matrix
!   do i = 1,nvec
!      work(i) = 0.0D0
!      do j = 1,nelem
!         work(i) = work(i) + v_dat(j,i)*v_dat(j,i)
!      end do
!   end do

! end ortho_mat
  END SUBROUTINE ortho_mat

!------------------------------------------------------------------------------
! subroutine move_cons
!
! Moves the constraints behind the active space in the V matrix, transposes
! V and stores it in Ut
!
! Arguments: see merge_cons
!------------------------------------------------------------------------------

  SUBROUTINE move_cons(v_mat,ut_mat,work,m,nc,np)

! args
    INTEGER m, nc, np
    TYPE (matrix) :: v_mat, ut_mat
    REAL (rk), DIMENSION (np) :: work

! local vars
    INTEGER is, it, i
    REAL (rk) dnorm, tol

! data
    DATA tol/1.0D-10/

! begin
    DO is = 1, nc
      i = matrix_get_column(v_mat,size(work),work,is)
      i = matrix_set_row(ut_mat,size(work),work,m-nc+is)
    END DO
    it = 0
    DO is = nc + 1, nc + m
      i = matrix_get_column(v_mat,size(work),work,is)
      dnorm = 0.0D0
      DO i = 1, np
        dnorm = dnorm + work(i)*work(i)
      END DO
      IF (dnorm>tol) THEN
!         write(stdout,'("JK dnorm ",g15.7," tol ",g15.7)') dnorm,tol
        it = it + 1
        IF (it>m-nc) THEN
          WRITE (stdout,'(A,I5)') 'Too many active vectors, required: ', &
            m - nc
!            write(stdout,'("JK dnorm ",g15.7," tol ",g15.7)') dnorm,tol
          CALL hdlc_errflag('Constraints error','abort')
        ELSE
          i = matrix_set_row(ut_mat,size(work),work,it)
        END IF
      END IF
    END DO

! end move_cons
  END SUBROUTINE move_cons

!------------------------------------------------------------------------------
! subroutine unproj_cons
!
! Replaces projected constraints by the unprojected ones in the matrix
! U transposed
!
! Arguments: see merge_cons
!------------------------------------------------------------------------------

  SUBROUTINE unproj_cons(ut_mat,c_mat,work,m,nc,np)

! args
    INTEGER m, nc, np
    TYPE (matrix) :: ut_mat, c_mat
    REAL (rk), DIMENSION (np) :: work

! local vars
    INTEGER is, i

! begin
    DO is = 1, nc
      i = matrix_get_column(c_mat,size(work),work,is)
      i = matrix_set_row(ut_mat,size(work),work,m-nc+is)
    END DO

! end unproj_cons
  END SUBROUTINE unproj_cons

END MODULE dlfhdlc_constraint
