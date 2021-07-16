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
MODULE dlfhdlc_primitive
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout,pi
  USE dlfhdlc_matrixlib
  IMPLICIT NONE

CONTAINS

!------------------------------------------------------------------------------
! subroutine connect_prim
!
! Automatically constructs a connection pattern for the system
!
! Method
! ======
!
! This method goes in two stages.  First a simple covalent radius method is
! employed, second the shortest branched path that internconnects every atom
! is found. The result is the fusion of these two connection patterns.
!
! The covalent interconnection method uses a fixed radius for each row of the
! periodic table, this is a bit crude. A connection is made when the square of
! the sum of the atomic radii is greater than 0.8 times the square of the
! inter atomic distance.
!
! The minimum distance method can be described as a two bucket algorithm.
! The is a 'known' bucket and an 'unknown' bucket.
!
! Initially, one atom is placed in the known and the rest in unknown.
! The algorithm cycles by finding the mimium distance between any atom in
! the known and unknown buckets.  For each cycle, a connection is made
! between the atoms found to have this shortest distance, and the atom
! of the connection that is in the unknows, and the unknown atom is placed
! into the known bucket.
!
! Once all the atoms are in the known bucket, the connection pattern is 
! finished.
!
! The routine can be called multiple times with different structures. The final
! connection pattern will then be the union of the connection patterns of all
! structures
!
! Parameters
! =========
!
! natom               (in)  number of atoms, integer
! types   (natom)     (in)  integer array of nuclear charges
! xyz matrix(3,natom) (in)  cartesian coordiantes in AU
! nconn               (out) number of connections found
! iconn   (2,nconn)   (out) pointer to connections list
!------------------------------------------------------------------------------
! the pointer here can only be resolved by a dry routine that
! first calculates nconn and then by the same way iconn

  SUBROUTINE connect_prim(natom,maxcon,types,nconn,iconn,xyz)

    IMPLICIT NONE
! args
    INTEGER natom, nconn
    INTEGER, DIMENSION (natom) :: types
    INTEGER, POINTER, DIMENSION (:,:) :: iconn
    TYPE (matrix) :: xyz

! local vars
    LOGICAL, ALLOCATABLE, DIMENSION (:) :: lconn
    INTEGER i, ii, idum, j, jj, k, maxcon, nico
    REAL (rk) dx, dy, dz, rad, radk, rr, s1, x, y, z
    REAL (rk), ALLOCATABLE, DIMENSION (:,:) :: xcoords
    LOGICAL tmp

! begin, get coordinates and allocate temporary space
    call allocate (xcoords,3,natom)
    idum = matrix_get(xyz,3*natom,xcoords)
    !nconn = 0
    !maxcon = natom
    !allocate (iconn(2,maxcon)) ! pointer allocation done in the calling routine

! loop over primary atom i, get covalent radius and coordinates
    DO i = 1, natom
      rad = covrad(types(i))
      IF (printl>=5) WRITE (stdout,'(5x,a,i4,a,i2,a,f10.4)') 'Atom ', i, &
        ', type: ', types(i), ', radius: ', rad
      x = xcoords(1,i)
      y = xcoords(2,i)
      z = xcoords(3,i)

! loop over secondary atom k, get threshold distance
      DO k = 1, i - 1
        tmp = not_connected(i,k,iconn,nconn)
        if(.not.tmp) cycle ! they are already connected

        radk = rad + covrad(types(k))
        dx = x - xcoords(1,k)
        dy = y - xcoords(2,k)
        dz = z - xcoords(3,k)
        rr = 0.8D0*(dx*dx+dy*dy+dz*dz)

! atoms i and k are covalently bound, connect them
        IF (rr<radk*radk) THEN
          nconn = nconn + 1

! allocate more space if required
          IF (nconn>maxcon) THEN
            CALL conn_grow(iconn,maxcon,maxcon+natom,maxcon)
            maxcon = maxcon + natom
          END IF
          iconn(1,nconn) = i
          iconn(2,nconn) = k
          IF (printl>=5) WRITE (stdout,'(5x,a,i5,a,i5,a,f10.4,a,f10.4)') &
            'Covalent bond ', i, ' - ', k, ': ', sqrt(rr/0.8D0), &
            ', cov. distance: ', radk
        END IF
      END DO ! k = 1,i-1
    END DO ! i = 1,natom

! done covalent connections
! now insert connections from the branched minimum distance path

! allocate space to hold information which atom is already connected
    nico = 0
    call allocate (lconn,natom)
    DO i = 2, natom
      lconn(i) = .FALSE.
    END DO
    lconn(1) = .TRUE.

! jump here if more connections need to be made
10  CONTINUE
    s1 = -1.0D0
    DO i = 1, natom
      IF (lconn(i)) THEN
        DO j = 1, natom
          IF ( .NOT. lconn(j)) THEN
            dx = xcoords(1,i) - xcoords(1,j)
            dy = xcoords(2,i) - xcoords(2,j)
            dz = xcoords(3,i) - xcoords(3,j)
            rr = dx*dx + dy*dy + dz*dz
            IF (rr<s1 .OR. s1==-1.0D0) THEN
              s1 = rr
              jj = j
              ii = i
            END IF
          END IF ! if (.not. lconn(j))
        END DO ! j = 1,natom
      END IF ! if (lconn(i))
    END DO ! i = 1,natom

! check if shortest distance is not yet included in the connections list
    lconn(jj) = .TRUE.
    nico = nico + 1
    tmp = not_connected(ii,jj,iconn,nconn)
    IF (tmp) THEN
      IF (printl>=4) THEN
        WRITE (stdout,'(5x,a,i5,i5)') 'Adding inter fragment connction ', jj, &
          ii
      END IF
      nconn = nconn + 1

! allocate more space if required
      IF (nconn>maxcon) THEN
        CALL conn_grow(iconn,maxcon,maxcon+natom,maxcon)
        maxcon = maxcon + natom
      END IF
      iconn(1,nconn) = jj
      iconn(2,nconn) = ii
    END IF ! (not_connected (ii, jj, iconn, nconn))

! check if all done
    IF (nico/=natom-1) GO TO 10

! free flags array and tidy up size of returned memory
    call deallocate (lconn)
    CALL conn_grow(iconn,maxcon,nconn,nconn)
! maxcon reset to ensure that it is consistent with the new size of iconn
    maxcon = nconn
    call deallocate (xcoords)

  END SUBROUTINE connect_prim

!//////////////////////////////////////////////////////////////////////////////
! Used by connect_prim only: Returns covalent radius from nuclear charge
! Currently: fixed radius for all atoms of a row
!//////////////////////////////////////////////////////////////////////////////

  FUNCTION covrad(nuccg)
    IMPLICIT NONE
    REAL (rk) covrad
! args
    INTEGER nuccg
! local vars
    INTEGER i, irelem
    REAL (rk) radius
! begin
    irelem = 0
!!$    DO i = 1, ctrl%nrad
!!$      IF (ctrl%irad(i)==nuccg) irelem = i
!!$    END DO
    IF (irelem==0) THEN
      IF (nuccg<=2) THEN
        radius = 1.0D0 ! H..He
      ELSE IF (nuccg<=10) THEN
        radius = 1.6D0 ! Li..Ne
      ELSE IF (nuccg<=18) THEN
        radius = 2.5D0 ! Na..Ar
      ELSE IF (nuccg<=36) THEN
        radius = 3.0D0 ! K..Kr
      ELSE IF (nuccg<=54) THEN
        radius = 4.0D0 ! Rb..Xe
      ELSE
        radius = 5.0D0 ! Cs....
      END IF
    ELSE
!      radius = ctrl%vrad(irelem)
    END IF
    covrad = radius
  END FUNCTION covrad

!//////////////////////////////////////////////////////////////////////////////
! Used by connect_prim only: Check if ii - jj is not contained in iconn
!//////////////////////////////////////////////////////////////////////////////

  FUNCTION not_connected(ii,jj,iconn,nconn)
    IMPLICIT NONE
! retval
    LOGICAL not_connected
! args
    INTEGER ii, jj, nconn
    INTEGER iconn(2,nconn)
! local vars
    INTEGER i
! begin
    not_connected = .TRUE.
    DO i = 1, nconn
      IF ((iconn(1,i)==ii .AND. iconn(2,i)==jj) .OR. (iconn(1, &
          i)==jj .AND. iconn(2,i)==ii)) THEN
        not_connected = .FALSE.
        RETURN
      END IF
    END DO
  END FUNCTION not_connected

!//////////////////////////////////////////////////////////////////////////////
! Used by connect_prim only: Copy the connectivity into a larger list
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE conn_grow(iconn,old_size,new_size,copy_size)
    IMPLICIT NONE
! args
    INTEGER old_size, new_size, copy_size
    INTEGER, POINTER, DIMENSION (:,:) :: iconn
! local vars
    INTEGER, POINTER, DIMENSION (:,:) :: iconn_new
    INTEGER i
! begin
    allocate (iconn_new(2,new_size)) !pointer
    DO i = 1, copy_size
      iconn_new(1,i) = iconn(1,i)
      iconn_new(2,i) = iconn(2,i)
    END DO
    DEallocate (iconn) ! pointer
    iconn => iconn_new
  END SUBROUTINE conn_grow

!//////////////////////////////////////////////////////////////////////////////
! end routines used by connect_prim
!//////////////////////////////////////////////////////////////////////////////

!------------------------------------------------------------------------------
! subroutine connect_all
!
! Alternative to connect_prim and valcoor. Every atom is connected to every
! atom. No angular coordinates required.
!
! Arguments
! natom             (in)  number of atoms, integer
! nconn             (out) number of connections found
! iconn   (2,nconn) (out) pointer to connections list
!------------------------------------------------------------------------------

  SUBROUTINE connect_all(natom,nconn,iconn)
    IMPLICIT NONE

! args
    INTEGER natom, nconn
    INTEGER, POINTER, DIMENSION (:,:) :: iconn

! local vars
    INTEGER i, j

! begin
    allocate (iconn(2,natom*(natom-1)/2)) !pointer
    nconn = 0
    DO i = 1, natom
      DO j = 1, i - 1
        nconn = nconn + 1
        iconn(1,nconn) = i
        iconn(2,nconn) = j
      END DO
    END DO
    IF (printl>=2) THEN
      WRITE (stdout,'(/,a,/)') 'Generating total connection'
      WRITE (stdout,'(5x,a,i5,a,/)') 'System has ', nconn, ' connections'
    END IF
  END SUBROUTINE connect_all

!------------------------------------------------------------------------------
! subroutine ci_conn
!
! Check in user connections and create a connections (integer) matrix. Note
! that all sequence numbers are relative to the start of the residue at exit.
!
! Arguments:
! con    int_matrix(2,*) (out) connections matrix, relative
! nconn                  (in)  number of calculated connections
! iconn  (2,nconn)       (in)  calculated connections, relative
! nincon                 (in)  number of user connections
! incon  (2,nincon)      (in)  user connections, absolute
! nat                    (in)  number of atoms in the residue
! at                     (in)  array of atom indices
!------------------------------------------------------------------------------

  SUBROUTINE ci_conn(con,nconn,iconn,nincon,incon,nat,at)
    IMPLICIT NONE

! args
    INTEGER nconn, nincon, start, finish
    INTEGER, DIMENSION (2,nconn) :: iconn
    INTEGER, DIMENSION (2,nincon) :: incon
    TYPE (int_matrix) :: con
    integer  :: nat
    integer  :: at(nat)

! local vars
    INTEGER i, j, ninres,iat
    INTEGER, DIMENSION (:,:), ALLOCATABLE :: incon_tmp
    integer :: in1,in2

! begin, allocate scratch
    call allocate (incon_tmp,2,nincon)

! check if input connections are in this residue
    ninres = 0
    DO j = 1, nincon
      in1=0
      do iat=1,nat
        if(incon(1,j)==at(iat)) in1=iat
      end do
      IF (in1>0) then
        in2=0
        do iat=1,nat
          if(incon(2,j)==at(iat)) in2=iat
        end do
        IF (in2>0) then
!!$          IF (printl>=2) THEN
!!$            WRITE (stdout,'(5x,a,i4,a,i4)') 'Adding user stretch', &
!!$              ctrl%incon(1,j), ' - ', ctrl%incon(2,j)
!!$          END IF
          ninres = ninres + 1
          incon_tmp(1,ninres) = in1
          incon_tmp(2,ninres) = in2
        ELSE
          IF (printl>=6) WRITE (stdout,'(a,i4,a,i4,a,/)') &
              'Warning: user stretch ', incon(1,j), ' - ', incon(2,j), &
              'crosses the residue boundary'
        END IF
      END IF
    END DO

! create the matrix and check in the calculated connections
    con = int_matrix_create(2,nconn+ninres,'connections')
    DO i = 1, nconn
      j = int_matrix_set_column(con,2,iconn(1:2,i),i)
    END DO

! check in the user connections
    DO i = 1, ninres
      j = int_matrix_set_column(con,2,incon_tmp(1:2,i),i+nconn)
    END DO
    call deallocate (incon_tmp)
  END SUBROUTINE ci_conn

!------------------------------------------------------------------------------
! subroutine valcoor
!
! Alexander J Turner Oct 1997
!
! This algorithm will make a fully redundant set of interanal coordinates.
! It defines rotation about bonds but not dihedral angles.
!
! The rotational orientation of atoms should be handled using
! rotation coordinates along the connection axes.
!
! Oct. 1997
! Changed to handle Quasi style matrix objects and to generate
! redundant co-ordinates. This is a derivative of valcoor.f in
! GRACE - I.H.Williams and A.J.Turner - University of Bath
! U.K. 1997
!
! June 1999
! Rewritten object and memory management - Zurich, SB
!
! On input
! ========
! natom : number of atoms
! nconn : Number of connections
! x,y,z : simple cartesian arrays
! iconn : 2 by n array - integer: column one is connected to column two
!
! On output
! ========
! ibend : 4 by nbend  - pointer to integer array
! irots : 3 by nnrots - pointer to integer array
! x,y,z,iconn - unchanged
!
! Definition of ibend and irots
! =============================
! ibend : bends are between ibend(1,i) and ibend(2,i) about ibend(3,1)
!     ibend(1,i) is zero - no bend defined
!       : linear bend ibend(1,i)-ibend(2,i)-ibend(3,i) 
!     ibend(4,i)=-2 => relative to xy plane
!     ibend(4,i)=-1 => relative to yz plane
!
! irots : Diherdal angle 1-2-3 w.r.t 2-3-4 , auto generated dihedrals
!     will either be proper or inproper 
!     *** FOR HISTORICAL REASONS - THESE ARE OFTEN REFERRED ***
!               TO AS ROTATIONS IN THIS CODE
!  
! nrots : Number of auto-dihedrals
! nbend : Number of bends
!
! Comments:
! =========
! The algorithm uses dynamic storage for valence information
! The scratch storage is in four dynamically allocated parts
!   ivale - an array giving the valence of each atom
!   ipvle - an array of pointers into ipple
!      ipple(ipvle+i-1) points to the start of the record of
!      atom numbers in the valence of atom i
!   ipple  - the array used to store
!      the atoms in an atoms valence shell, the pointers
!      stored from ipvle point into this array 
!   ipcvl  - counters for atomic valence collection
!
! Arguments:
! ==========
! See the comments to the fields of hdlc_obj (hdlclib.f90)
! in:  nconn, iconn, natom, x, y, z
! out: nbend, ibend, nrots, irots
!------------------------------------------------------------------------------

  SUBROUTINE valcoor(nconn,iconn,natom,x,y,z,nbend,ibend,nrots,irots)

    IMPLICIT NONE

! args
    INTEGER nconn, natom, nbend, nrots
    INTEGER, POINTER, DIMENSION (:,:) :: iconn
    INTEGER, pointer, DIMENSION (:,:) :: ibend
    INTEGER, POINTER, DIMENSION (:,:) :: irots
    REAL (rk), pointer, DIMENSION (:) :: x, y, z

! local vars
    LOGICAL notline,tmp
    INTEGER i, ib, ib1, ib2, ib3, icent, ileft, ip, ibp, j, k, l,jj
    INTEGER maxbend, maxrots
    INTEGER, DIMENSION (:), ALLOCATABLE :: ivale, ipvle, ipcvl, ipple, index
    REAL (rk) :: angle, tx, ty, tz, ax, ay, az, ex, ey, ez, dx, dy, dz, r

! begin
    IF (printl>=5) THEN
      WRITE (stdout,'(5X,A,/)') 'Generating primitive internal coordinates'
    END IF

! get some scratch space
    call allocate (ivale,natom)
    call allocate (ipvle,natom)
    call allocate (ipcvl,natom)

!//////////////////////////////////////////////////////////////////////////////
! compute the valence of each atom
!
! ivale stores the total valence of each atom
! ipvle(i) points to the start of the valence of atom i in ipcvl(*)
!//////////////////////////////////////////////////////////////////////////////

    DO i = 1, natom
      ivale(i) = 0
    END DO
    DO i = 1, nconn
      j = iconn(1,i)
      if (j<=0 .OR. j>natom) then
        print*,"Connection no",i
        print*,"Connection between atoms",iconn(1,i),iconn(2,i)
        print*,"Number of atoms",natom
        CALL hdlc_errflag('Error: atom of connection out of range','stop')
      end if
      ivale(j) = ivale(j) + 1
      j = iconn(2,i)
      ivale(j) = ivale(j) + 1
    END DO

! get the number of bends, ib counts the total number of valence records
    nbend = 0
    ib = 0
    DO i = 1, natom
      ib = ib + ivale(i)
      DO j = 1, ivale(i) - 1
        nbend = nbend + j
      END DO
    END DO
    maxbend = nbend
    call allocate (ipple,ib)
    allocate (ibend(4,max(maxbend,1))) ! pointer
    ipvle(1) = 1

! set pointers for valence records (formerly valcoor2)
    nbend = 0
    DO i = 2, natom
      ipvle(i) = ipvle(i-1) + ivale(i-1)
    END DO

! set counters for atomic valence collection to -1
    DO i = 1, natom
      ipcvl(i) = -1
    END DO

! record the valence atoms around each atom from connection list
    DO i = 1, nconn
      j = iconn(1,i)
      k = iconn(2,i)

! increment record counter for atom j
      ipcvl(j) = ipcvl(j) + 1

! acquire pointer to records for j
      l = ipvle(j) + ipcvl(j)

! set that element to the atom to which it is connected
      ipple(l) = k

! repeat for other atom
      j = iconn(2,i)
      k = iconn(1,i)

! increment record counter for j
      ipcvl(j) = ipcvl(j) + 1

! acquire pointer to records for j
      l = ipvle(j) + ipcvl(j)

! set that element to the atom to which it is connected
      ipple(l) = k
    END DO

! print the valence
    IF (printl>=5) THEN
      WRITE (stdout,'(5X,A,/)') 'Valence of atomic centres'
      DO i = 1, natom
        j = ipvle(i)
        k = ivale(i)
        WRITE (stdout,'(5x,A,I5,A,8(1X,I5))') 'Atom no ', i, ' has valence =', &
          (ipple(j+l),l=0,k-1)
      END DO
      WRITE (stdout,'(/)')
    END IF

!//////////////////////////////////////////////////////////////////////////////
! Generate bends
!
! Loop scanning atomic centres. It looks for bends and also puts in linear
! bends and wags etc.
!
! I scans each atom
! J scans the valence atoms around the I'th atom
! K scans the 'other' valence atoms around the I'th atom so bends are made
!
! Other variables used
!
! ib     : The number of bends allocated about a given centre
! ib1    : The valence atom from which bends are coming
! ib2    : The valence atom to which bends are going
! ib3    : The waging atom ib3 wags in plane of ib1-i-ib2
! ip     : Pointer to valence records for I'th atom
! ibp    : Pointer to the bend record that is being written 
! nbend  : Number of allocated bends in total
! maxbend: Ammount of space allocated for bends
!//////////////////////////////////////////////////////////////////////////////

    DO i = 1, natom
      ib = 0
      IF (printl>=5) THEN
        WRITE (stdout,'(7X,A,I5)') 'Scanning bends around ', i
      END IF

! pointer to start of valence info of atom i and loop through atoms j
      ip = ipvle(i)
      DO j = 1, ivale(i) - 1

! set ib1 to the atom connected to atom i from which bends are to be made
        ib1 = ipple(ip+j-1)
        IF (printl>=5) THEN
          WRITE (stdout,'(7X,A,I5)') 'Making bends from atom ', ib1
        END IF

! loop through all atoms higher in order than atom j and make bends
        DO k = j + 1, ivale(i)
          nbend = nbend + 1

! detect a potential problem in the code
          IF (nbend>maxbend) THEN
            CALL hdlc_errflag( &
              'Insufficient bends allocated in valcoor - code error','abort')
          END IF

! increment bond counter
          ib = ib + 1

! set ib2 to the atom connected to atom i to which bends are to be made
          ib2 = ipple(ip+k-1)

! ibp is a pointer to the present bend record: make bend
          ibp = nbend
          ibend(1,ibp) = ib1
          ibend(2,ibp) = i
          ibend(3,ibp) = ib2
          ibend(4,ibp) = 0

!//////////////////////////////////////////////////////////////////////////////
! check for linearity
!//////////////////////////////////////////////////////////////////////////////

          angle = vangled(x(ib1),x(i),x(ib2),y(ib1),y(i),y(ib2),z(ib1),z(i), &
            z(ib2))
          IF (printl>=5) THEN
            WRITE (stdout,'(7X,A5,I5,A2,3I5,A8,F12.7)') 'Bend ', nbend, ': ', &
              ib1, i, ib2, ' angle: ', angle
          END IF

! make an l_function if the centre is bi-valent & > 179 deg
          IF (angle>=175.0D0) THEN
            IF (ivale(i)==2) THEN
              IF (printl>=4) THEN
                WRITE (stdout,'(7X,A,I5)') 'Making linear about ', i
              END IF

! test linear function for being perp to xy plane:
! translate one point in z
! take angle between point and other two points
! must be 5<angle<175 or yz plane signaled (istra(nline,4)=-1)

              angle = vangled(x(ib1),x(ib2),x(ib2),y(ib1),y(ib2),y(ib2), &
                z(ib1),z(ib2),z(ib2)+1.0D0)
              IF (printl>=5) THEN
                WRITE (stdout,'(7X,A,F8.3)') 'Tester angle = ', angle
              END IF
              IF (angle<=5.0D0 .OR. angle>=175.0D0) THEN
                ibend(4,ibp) = -2
              ELSE
                ibend(4,ibp) = -1
              END IF

! note: removed if (.not.internal) then (remove the bend) endif
            ELSE ! if (ivale(i).eq.2) then ...
              IF (printl>=5) WRITE (stdout,'(7X,A)') &
                'Bend ignored, linear non bivalent'
              ibend(1,ibp) = 0
              nbend = nbend - 1
            END IF ! if (ivale(i).eq.2) then ... else
          END IF ! if (angle.ge.175.0D0)
        END DO ! do k=j+1,ivale(i)
      END DO ! do j = 1,ivale(i)-1

!//////////////////////////////////////////////////////////////////////////////
! check for planar systems
!//////////////////////////////////////////////////////////////////////////////

      IF (ivale(i)>2) THEN

! acquire pointer to start of valence info for this atom
        ip = ipvle(i)
        tx = 1.0D0
        ty = 1.0D0
        tz = 1.0D0

! loop through all valence atoms dotting the perps to the normalised vectors
        DO j = 1, ivale(i) - 1

! set ib1 to the atom connected to atom i from which bends are to be made
          ib1 = ipple(ip+j-1)
          ax = x(ib1) - x(i)
          ay = y(ib1) - y(i)
          az = z(ib1) - z(i)
          r = sqrt(ax*ax+ay*ay+az*az)
          ax = ax/r
          ay = ay/r
          az = az/r
          ib1 = ipple(ip+j)
          ex = x(ib1) - x(i)
          ey = y(ib1) - y(i)
          ez = z(ib1) - z(i)
          r = sqrt(ex*ex+ey*ey+ez*ez)
          ex = ex/r
          ey = ey/r
          ez = ez/r
          dx = (ay*ez-az*ey)
          dy = (az*ex-ax*ez)
          dz = (ax*ey-ay*ex)
          r = sqrt(dx*dx+dy*dy+dz*dz)
          tx = tx*dx/r
          ty = ty*dy/r
          tz = tz*dz/r
        END DO
        IF (printl>=5) THEN
          WRITE (stdout,'(7X,A,I5,A,F12.8)') 'Dot-product about ', i, ' is ', &
            tx + ty + tz
        END IF

! if dot greater than 5% off 1 put in dihedral instead of bend
        IF (abs(tx+ty+tz)>0.95D0) THEN
          IF (printl>=4) THEN
            WRITE (stdout,'(7X,A)') 'System near planar - making improper'
          END IF

! construct improper from last bend - i.e. nbend
          k = -1
          DO j = 1, ivale(i)
            ib1 = ipple(ip+j-1)
            IF (ib1/=ibend(1,nbend) .AND. ib1/=ibend(3,nbend)) k = ib1
          END DO
          IF (k==-1) THEN
            CALL hdlc_errflag('Failed to contruct internals','abort')
          END IF
          ibend(4,nbend) = k

! swap 1 and 3 if 2-3-4 is co-linear
          dx = x(ibend(2,nbend)) - x(ibend(3,nbend))
          dy = y(ibend(2,nbend)) - y(ibend(3,nbend))
          dz = z(ibend(2,nbend)) - z(ibend(3,nbend))
          r = sqrt(dx*dx+dy*dy+dz*dz)
          tx = dx/r
          ty = dy/r
          tz = dz/r
          dx = x(ibend(3,nbend)) - x(ibend(4,nbend))
          dy = y(ibend(3,nbend)) - y(ibend(4,nbend))
          dz = z(ibend(3,nbend)) - z(ibend(4,nbend))
          r = sqrt(dx*dx+dy*dy+dz*dz)
          tx = tx*dx/r
          ty = ty*dy/r
          tz = tz*dz/r
          IF (tx+ty+tz>0.095D0) THEN
            k = ibend(3,nbend)
            ibend(3,nbend) = ibend(1,nbend)
            ibend(1,nbend) = k
          END IF
        END IF ! (abs(tx+ty+tz).gt.0.95D0)
      END IF ! (ivale(i).gt.2)
    END DO ! i = 1,natom (end of valcoor2)

!//////////////////////////////////////////////////////////////////////////////
! generate dihedrals and improper dihedrals
!//////////////////////////////////////////////////////////////////////////////

! initialise (formerly valcoor3)
    call deallocate (ipple)
    call allocate (index,natom)
    maxrots = nbend*3
    allocate (irots(4,max(1,maxrots))) !pointer
    IF (printl>=6) WRITE (stdout,'(7X,A,I5)') 'Guessing maxrots at ', &
      maxrots
    DO i = 1, natom
      index(i) = -1
    END DO

! set index(i) to point to records of bends about i
    k = 0
    DO i = 1, nbend
      IF (ibend(2,i)/=k) THEN
        k = ibend(2,i)
        IF (ibend(1,i)/=0) index(k) = i
      END IF
    END DO

! sort all bends so if one end is univalent and the other bivalent, the
! bivalent is the far end
    DO i = 1, nbend
      IF (index(ibend(3,i))==-1 .AND. ibend(4,i)>=0) THEN
        k = ibend(3,i)
        ibend(3,i) = ibend(1,i)
        ibend(1,i) = k
      END IF
    END DO
    DO i = 1, natom
      index(i) = -1
    END DO
    k = -1
    DO i = 1, nbend
      IF (ibend(2,i)/=k) THEN
        k = ibend(2,i)
        IF (ibend(1,i)/=0) index(k) = i
      END IF
    END DO

! main loop through bends constructing dihedrals
    nrots = 0
    DO i = 1, nbend
      IF (ibend(4,i)<0) GO TO 300 ! skip if linear
!WT write (stdout,'(1X,A,2I5)') ' nbend,i ', nbend,i

! flag this is a new bend and not due to expansion of a dihedral
      notline = .TRUE.

! k points to the bend record of the far end of bend i
      ileft = ibend(2,i)
      icent = ibend(3,i)
      k = index(icent)

! follow linears until a real bend (loop point for a 'do until' type loop)
100   CONTINUE

! if the atom ibend(3,i) is univalent, you cannot make a dihedral -> break
      IF (k==-1) GO TO 300

! expand the dihedral (follow the bond) if atom is linear bivalent
      IF (ibend(4,k)<0) THEN

! test for three membered ring, break if found
        IF ((ibend(3,i)==ibend(3,k) .OR. ibend(1,i)==ibend(1,k) .OR. ibend(1, &
          i)==ibend(3,k) .OR. ibend(3,i)==ibend(1,k)) .AND. notline) GO TO 300
        notline = .FALSE.

! test which way around the bend is
        ileft = ibend(2,k)
        IF (ileft/=max(ibend(1,k),ibend(3,k))) THEN
          icent = max(ibend(1,k),ibend(3,k))
          k = index(icent)
          GO TO 100
        ELSE
          icent = min(ibend(1,k),ibend(3,k))
          k = index(icent)
          GO TO 100
        END IF
      END IF ! (ibend(4,k).lt.0)

! cycle over all bends around icent
      DO WHILE (ibend(2,k)==icent)

! test for three membered ring, break if found
        IF ((ibend(3,i)==ibend(3,k) .OR. ibend(1,i)==ibend(1,k) .OR. ibend(1, &
          i)==ibend(3,k) .OR. ibend(3,i)==ibend(1,k)) .AND. notline) GO TO 300

! test added by Martin Graf
! test for identical central atom on creating dihedral MG
        IF (ibend(2,k)==ibend(2,i)) GO TO 300

        ! test added by Johannes Kaestner >>>>
        ! Do not include dihedral if one of its center atoms is part of an
        ! improper
        ! Have to remove it as it leads to an underdetermined system if
        ! not all atoms next to the planar atom are monovalent.
!!$        tmp=.false.
!!$        do jj=1,nbend
!!$          if(ibend(4,jj)<=0) cycle
!!$          if(ibend(2,jj)==ibend(3,i).or. &
!!$              ibend(2,jj)==ibend(2,i)) then
!!$            tmp=.true.
!!$            if(printl>=6) write(stdout,'("Deleting Dihedral due to &
!!$                &overlay with improper")')
!!$          end if
!!$        end do
!!$        if(tmp) go to 300
        ! <<<< JK

! test if bend k is valid and matching bend i
        IF (ibend(1,k)/=0 .AND. ibend(4,k)==0 .AND. (ibend(1, &
            k)==ileft .OR. ibend(3,k)==ileft)) THEN
          nrots = nrots + 1

! allocate more space if required
          IF (nrots>maxrots) THEN
            CALL rots_grow(irots,maxrots,maxrots+nbend*9,maxrots)
            maxrots = maxrots + nbend*9
          END IF

! store the dihedral (formerly valcoor4)
          IF (ibend(1,k)==ileft) THEN
            irots(4,nrots) = ibend(3,k)
          ELSE
            irots(4,nrots) = ibend(1,k)
          END IF
          DO j = 1, 3
            irots(j,nrots) = ibend(j,i)
          END DO

! to ensure identical dihedrals are not included
          DO j = 1, nrots - 1
            IF (irots(1,j)==irots(4,nrots)) THEN
              IF (irots(2,j)==irots(3,nrots) .AND. irots(3,j)==irots(2,nrots)) &
                  THEN
                nrots = nrots - 1
              END IF
            END IF
          END DO
        END IF ! valid and matching bend

! end cycle over all bends around icent
        k = k + 1
        IF (k>maxbend) EXIT
      END DO ! WHILE (ibend(2,k)==icent)

! Added by Johannes Kaestner (JK):
! try the first atom of ibend(:,i). This is necessary in special cases, like
! \|   |/
! -4   3-
!   \ /
!    1
!    |
!    2
!   / \
! -5   6-
! /|   |\
! Try it out with linear molecules! It may cause problems there.
      if(icent==ibend(3,i)) then
        ileft=ibend(2,i)
        icent=ibend(1,i)
        k = index(icent)
        go to 100
      end if
! end of addtion by JK

! loss of indentation due to end of loop to 200
300   CONTINUE
    END DO ! do i = 1,nbend (end of valcoor3)

! deallocate scratch space
    call deallocate (ivale)
    call deallocate (ipvle)
    call deallocate (ipcvl)
    call deallocate (index)

  END SUBROUTINE valcoor

!//////////////////////////////////////////////////////////////////////////////
! Copy the torsions list into a larger list
! Also used by ck_angle_cons if extra torsion/bond angle primitives are needed
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE rots_grow(irots,old_size,new_size,copy_size)
    IMPLICIT NONE
! args
    INTEGER old_size, new_size, copy_size
    INTEGER, POINTER, DIMENSION (:,:) :: irots
! local vars
    INTEGER, POINTER, DIMENSION (:,:) :: irots_new
    INTEGER i
! begin
    allocate (irots_new(4,new_size)) !pointer
    DO i = 1, copy_size
      irots_new(1,i) = irots(1,i)
      irots_new(2,i) = irots(2,i)
      irots_new(3,i) = irots(3,i)
      irots_new(4,i) = irots(4,i)
    END DO
    deallocate (irots) !pointer
    irots => irots_new
  END SUBROUTINE rots_grow

!//////////////////////////////////////////////////////////////////////////////
! end routines used by valcoor
!//////////////////////////////////////////////////////////////////////////////

!------------------------------------------------------------------------------
! subroutine valcoor_print
!
! Arguments:
! See the comments to the fields of hdlc_obj (hdlclib.f90)
! in:  nconn, iconn, natom, x, y, z, nbend, ibend, nrots, irots
!------------------------------------------------------------------------------

  SUBROUTINE valcoor_print(nconn,iconn,nbend,ibend,nrots,irots,x,y,z,natom)
    IMPLICIT NONE

! args
    INTEGER nconn, nbend, nrots, natom
    INTEGER iconn(2,nconn), ibend(4,nbend), irots(4,nrots)
    REAL (rk) :: x(natom), y(natom), z(natom)

! local vars
    INTEGER i, j, k
    REAL (rk) :: dx, dy, dz, r

! begin
    IF (printl>=4) THEN
      WRITE (stdout,'(/,3X,A)') 'The Cartesian coordinates'
      DO i = 1, natom
        WRITE (stdout,'(5X,I5,1X,F20.14,1X,F20.14,1X,F20.14)') i, x(i), y(i), &
          z(i)
      END DO
    END IF

    IF (printl>=4) THEN
      write (stdout,'(/,3X,A)') 'The primitive internal coordinates'
      k = 0

      IF (printl>=4) WRITE (stdout,'(/)')
      write (stdout,'(5X,A,I5,A)') 'The system has ', nconn, ' stretches'
      IF (printl>=4) THEN
        DO i = 1, nconn
          k = k + 1
          dx = x(iconn(1,i)) - x(iconn(2,i))
          dy = y(iconn(1,i)) - y(iconn(2,i))
          dz = z(iconn(1,i)) - z(iconn(2,i))
          r = sqrt(dx*dx+dy*dy+dz*dz)
          write (stdout,'(5X,i5,A,2(I5,1X),12X,A,F13.8)') k, ' Stre ', &
              (iconn(j,i), j=1,2), ' = ', r
        END DO
        WRITE (stdout,'(/)')
      END IF ! (printl.ge.2)

      write (stdout,'(5X,A,I5,A)') 'The system has ', nbend, &
          ' bends and impropers'
      IF (printl>=4) THEN
        DO i = 1, nbend
          k = k + 1
          IF (ibend(4,i)==0) THEN
            r = vangled(x(ibend(1,i)),x(ibend(2,i)),x(ibend(3,i)),y(ibend(1, &
              i)),y(ibend(2,i)),y(ibend(3,i)),z(ibend(1,i)),z(ibend(2, &
              i)),z(ibend(3,i)))
            write (stdout,'(5X,I5,A,3(I5,1X),6X,A,F13.8,a,f13.8)') k, ' Bend ', &
                (ibend(j,i), j=1,3), ' = ', r,' deg = ',r*pi/180.D0
          ELSE IF (ibend(4,i)==-1) THEN
            write (stdout,'(5X,I5,A,3(I5,1X),A)') k, ' Line ', &
                (ibend(j,i), j=1,3), ' wrt xy'
          ELSE IF (ibend(4,i)==-2) THEN
            write (stdout,'(5X,I5,A,3(I5,1X),A)') k, ' Line ', &
                (ibend(j,i), j=1,3), ' wrt yz'
          ELSE
            r = vdihedrald(x(ibend(1,i)),y(ibend(1,i)),z(ibend(1, &
              i)),x(ibend(2,i)),y(ibend(2,i)),z(ibend(2,i)),x(ibend(3, &
              i)),y(ibend(3,i)),z(ibend(3,i)),x(ibend(4,i)),y(ibend(4, &
              i)),z(ibend(4,i)))
            WRITE (stdout,'(5X,I5,A,4(I5,1X),A,F13.8,a,f13.8)') k, ' Impr ', &
              (ibend(j,i),j=1,4), ' = ', r,' deg = ',r*pi/180.D0
          END IF ! (ibend(4,i).eq.0)
        END DO ! i = 1,nbend
        WRITE (stdout,'(/)')
      END IF ! (printl.ge.2)

      write (stdout,'(5X,A,I5,A)') 'The system has ', nrots, ' dihedrals'
      IF (printl>=4) THEN
        DO i = 1, nrots
          k = k + 1
          r = vdihedrald(x(irots(1,i)),y(irots(1,i)),z(irots(1,i)),x(irots(2, &
            i)),y(irots(2,i)),z(irots(2,i)),x(irots(3,i)),y(irots(3, &
            i)),z(irots(3,i)),x(irots(4,i)),y(irots(4,i)),z(irots(4,i)))
          write (stdout,'(5X,I5,A,4(I5,1X),A,F13.8,a,f13.8)') k, ' Dihe ', &
              (irots(j,i), j=1,4), ' = ', r,' deg = ',r*pi/180.D0
        END DO
        WRITE (stdout,'(/)')
      END IF ! (printl.ge.2)
    END IF ! (printl.gt.1)
    CALL dlf_flushout()
  END SUBROUTINE valcoor_print

!------------------------------------------------------------------------------
! Utilities by Alex Turner (originally dlc_angle_utils)
!------------------------------------------------------------------------------

  SUBROUTINE rots_dlc(noint,ii,b,ib,c)
!
! Alexander J Turner
!
! Diagram:
!
!   p
!   |\ A
! C | \
!   s--q-----r
!    D   E
!
!  What it does:
!
! Computes the cartesian force on  A
! given a unit moment about axis   E
!
! It is intended for the generation of B matrix elements
!
! Method:
!
! Force on p is perpendicular to plane r,q,p
! Force on p is proportional to |C|
!
! C = A - (A.E x E/|E|)
!
! Direction perpendicular to p = ((AyEz-AzEy),(AzEx-AxEz),(AxEy-AyEx))
! Call this vector Z
!
! P.S. The use of case in this subroutine is just for human reading
!      Uppercase => vector , lowercase => scalar or point
!
    IMPLICIT NONE
    INTEGER i, ib(4,*), iaind, jaind, kaind, m
    INTEGER noint, ii
!
    REAL (rk) :: b(3,4,*), c(*), dzero, done, eps
    REAL (rk) :: dx, dy, dz
    REAL (rk) :: ex, ey, ez
    REAL (rk) :: tx, ty, tz
    REAL (rk) :: ax, ay, az
    REAL (rk) :: cx, cy, cz
    REAL (rk) :: zx, zy, zz
    REAL (rk) :: magae, mage, fact, magc
    PARAMETER (eps=1.0D-6)
!
    DATA dzero/0.0D0/, done/1.0D0/
!
! Set up position in B matrix pointers
! Zero bmat for premeture return
!
    ib(1,noint) = ii
    ib(2,noint) = 0
    ib(3,noint) = 0
    ib(4,noint) = 0
    b(1,1,noint) = dzero
    b(2,1,noint) = dzero
    b(3,1,noint) = dzero
!
! Get Vectors
!
    ex = c(4) - c(7)
    ey = c(5) - c(8)
    ez = c(6) - c(9)

    ax = c(1) - c(4)
    ay = c(2) - c(5)
    az = c(3) - c(6)
!
! Get A.E / |E|
!
    mage = sqrt(ex*ex+ey*ey+ez*ez)
    magae = ax*ex + ay*ey + az*ez
    fact = -1.0D0*magae/mage
!
! If E is small - return nothing
!
    IF (abs(mage)<eps) RETURN
!
! Scale E by factor
!
    tx = ex*fact
    ty = ey*fact
    tz = ez*fact
!
! Get C
!
    cx = ax - tx
    cy = ay - ty
    cz = az - tz
!
! Get magnitude of C - and thus force
!
    magc = sqrt(cx*cx+cy*cy+cz*cz)
!
! If C is small - angle linear - zero force, or numerical problems can occur
!
    IF (magc<eps) RETURN
!
! Get Z
!
    zx = (ay*ez-az*ey)
    zy = (az*ex-ax*ez)
    zz = (ax*ey-ay*ex)
!
! Normalise and times by magC
!
    fact = magc/sqrt(zx*zx+zy*zy+zz*zz)
    zx = zx*fact
    zy = zy*fact
    zz = zz*fact
!
! Put it into B
!
    b(1,1,noint) = zx
    b(2,1,noint) = zy
    b(3,1,noint) = zz
!
! All done
!
    RETURN
  END SUBROUTINE rots_dlc
!
! ------------------------------------------------------------
!
  FUNCTION dlc_free_rot(pxa,pxb,pxc,pya,pyb,pyc,pza,pzb,pzc,poxa,poxb,poxc, &
      poya,poyb,poyc,poza,pozb,pozc)
    IMPLICIT NONE
    REAL (rk) :: dlc_free_rot
!
! Alexander J Turner
! Feb 1998
!
! Compute the magnitude of the rotation of an atom about the axis of a bond,
! given that the vector of the bond is constant
!
    REAL (rk) :: xa, xb, xc, ya, yb, yc, za, zb, zc, oxa, oxb, oxc, oya, oyb, &
      oyc, oza, ozb, ozc, dx, dy, dz, s, pxa, pya, pza, pxb, pyb, pzb, pxc, &
      pyc, pzc, poxa, poya, poza, poxb, poyb, pozb, poxc, poyc, pozc
    REAL (rk) :: dzero, theta1, theta2, theta3, theta4
    PARAMETER (dzero=0.0D0)

!
! Move by refernece pxa ... to by value oxa ....
!
    xa = pxa
    xb = pxb
    xc = pxc
    ya = pya
    yb = pyb
    yc = pyc
    za = pza
    zb = pzb
    zc = pzc
    oxa = poxa
    oxb = poxb
    oxc = poxc
    oya = poya
    oyb = poyb
    oyc = poyc
    oza = poza
    ozb = pozb
    ozc = pozc
!
! move A,B,C by B->origin
!
    dx = -oxb
    dy = -oyb
    dz = -ozb

    oxa = oxa - dx
    oya = oya - dy
    oza = oza - dz

    oxc = oxc - dx
    oyc = oyc - dy
    ozc = ozc - dz

    oxb = dzero
    oyb = dzero
    oyb = dzero

    xa = xa - dx
    ya = ya - dy
    za = za - dz

    xc = xc - dx
    yc = yc - dy
    zc = zc - dz

    xb = dzero
    yb = dzero
    yb = dzero
!
! Translate A' B'->C'
!
! oxa=oxa-xc
! oya=oya-yc
! oza=oza-zc
!
    dlc_free_rot = -vdihedral(xa,ya,za,oxc,oyc,ozc,oxb,oyb,ozb,oxa,oya,oza)
!      if(abs(dlc_free_rot).gt.1.0d-1) then
!         write(stdout,*)
!         write(stdout,*)'Dihed ',dlc_free_rot
!         write(stdout,*)'XY Theta: ',theta1
!         write(stdout,*)'XZ Theta: ',theta2
!         write(stdout,*)'Old c ',oxc,oyc,ozc
!         write(stdout,*)'New c',xc,yc,zc
!         write(stdout,*)'Old b ',oxb,oyb,ozb
!         write(stdout,*)'New b',xb,yb,zb
!         write(stdout,*)'Old a ',oxa,oya,oza
!         write(stdout,*)'New a',xa,ya,za
!      endif
!
    RETURN
  END FUNCTION dlc_free_rot
!
! ------------------------------------------------------------
!
  FUNCTION vdihedral(xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl)
    IMPLICIT NONE
    REAL (rk) :: vdihedral
    REAL (rk) :: xi, yi, zi, xj, yj, zj, xk, yk, zk, xl, yl, zl
    REAL (rk) :: fx, fy, fz, fr2, fr, hx, hy, hz, hr2, hr, ex, ey, ez, er2, er
    REAL (rk) :: rik, th1, th2, th3, phi, gx, gy, gz, gr2, gr, grr, gxr, gyr, &
      gzr
    REAL (rk) :: frr, fxr, fyr, fzr, cst, hrr, hxr, hyr, hzr, err, exr, eyr, &
      ezr
    REAL (rk) :: ax, ay, az, bx, by, bz, ra2, rb2, ra, rb, rar, rbr, axr, ayr, &
      azr
    REAL (rk) :: bxr, byr, bzr, cx, cy, cz
    REAL (rk) :: cut, one
    PARAMETER (cut=1.0D-8)
    PARAMETER (one=1.0D0)

    vdihedral = 0.0D0
!
! BOND DISTANCES
!
    fx = xi - xj
    fy = yi - yj
    fz = zi - zj
    fr2 = fx*fx + fy*fy + fz*fz
    IF (fr2<cut) RETURN
!
    hx = xl - xk
    hy = yl - yk
    hz = zl - zk
    hr2 = hx*hx + hy*hy + hz*hz
    IF (hr2<cut) RETURN
!
    ex = xi - xk
    ey = yi - yk
    ez = zi - zk
    er2 = ex*ex + ey*ey + ez*ez
    IF (er2<cut) RETURN
!
    gx = xj - xk
    gy = yj - yk
    gz = zj - zk
    gr2 = gx*gx + gy*gy + gz*gz
    IF (gr2<cut) RETURN
!
! DIHEDRAL VALUES
!
! AX perp to F-G
! BX perp to H-G
!
    ax = fy*gz - fz*gy
    ay = fz*gx - fx*gz
    az = fx*gy - fy*gx
    bx = hy*gz - hz*gy
    by = hz*gx - hx*gz
    bz = hx*gy - hy*gx
!
    ra2 = ax*ax + ay*ay + az*az
    rb2 = bx*bx + by*by + bz*bz
    ra = sqrt(ra2)
    rb = sqrt(rb2)
    IF (ra<=0.01D0) THEN
!      write(stdout,'(a)')'Warning: Dihedral near linear'
!      write(stdout,'(4(x,g13.5))')xi,yi,zi
!      write(stdout,'(4(x,g13.5))')xj,yj,zj
!      write(stdout,'(4(x,g13.5))')xk,yk,zk
!      write(stdout,'(4(x,g13.5))')xl,yl,zl
!      RA=0.01D0
      RETURN
    END IF
    rar = 1.0D0/ra
    IF (rb<=0.01D0) THEN
!       write(stdout,'(a)')'Warning: Dihedral near linear'
!       RB=0.01D0
      RETURN
    END IF
    rbr = 1.0D0/rb
!
! Normalise
!
    axr = ax*rar
    ayr = ay*rar
    azr = az*rar
    bxr = bx*rbr
    byr = by*rbr
    bzr = bz*rbr
!
    cst = axr*bxr + ayr*byr + azr*bzr
    IF (abs(cst)>=1.0D0) cst = sign(one,cst)
    phi = acos(cst)
    cx = ayr*bzr - azr*byr
    cy = azr*bxr - axr*bzr
    cz = axr*byr - ayr*bxr
    IF (gx*cx+gy*cy+gz*cz>0.0D0) phi = -phi
    vdihedral = phi
!   vdihedral=0.0D0
!   write(stdout,*)phi
  END FUNCTION vdihedral
!
! ------------------------------------------------------------
!
  FUNCTION vangled(x1,x2,x3,y1,y2,y3,z1,z2,z3)
    IMPLICIT NONE
    REAL (rk) :: vangled
    REAL (rk) :: d21, d23, d13
    REAL (rk) :: x1, x2, x3, y1, y2, y3, z1, z2, z3
    REAL (rk) :: two, eps, sd21, sd23, sd13

    PARAMETER (two=2.0D0)
    PARAMETER (eps=1.0D-6)
!
! Alexander J Turner - June 1997
! Used by valence, it returns the angle 1 and 3 about 2
! by cosine law in degrees
!
    d21 = (x1-x2)**two + (y1-y2)**two + (z1-z2)**two
    d23 = (x3-x2)**two + (y3-y2)**two + (z3-z2)**two
    d13 = (x1-x3)**two + (y1-y3)**two + (z1-z3)**two
!
! Allow for angle near 0 deg not causing numerical problems
!
    IF (d21<eps .OR. d23<eps .OR. d21<eps) THEN
      vangled = 0.0D0
      RETURN
    END IF
!
! Allow for angle near 180 deg not causing numberical problems
!
    sd21 = sqrt(d21)
    sd23 = sqrt(d23)
    sd13 = sqrt(d13)
!
    IF (abs(sd13-sd21-sd23)<eps) THEN
      vangled = 180.0D0
    ELSE
      vangled = 180.D0/pi*acos((d21+d23-d13)/(2.0D0*sd21*sd23))
    END IF
!
    RETURN
  END FUNCTION vangled
!
! ------------------------------------------------------------
!
  FUNCTION vangle(x1,x2,x3,y1,y2,y3,z1,z2,z3)
    IMPLICIT NONE
    REAL (rk) :: vangle
    REAL (rk) :: x1, x2, x3, y1, y2, y3, z1, z2, z3
!
    vangle = vangled(x1,x2,x3,y1,y2,y3,z1,z2,z3)
    vangle = vangle*pi/180.D0
    RETURN
  END FUNCTION vangle
!
! ------------------------------------------------------------
!
  FUNCTION vdihedrald(xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl)
    IMPLICIT NONE
    REAL (rk) vdihedrald
    REAL (rk) :: xi, yi, zi, xj, yj, zj, xk, yk, zk, xl, yl, zl

    vdihedrald = vdihedral(xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl)
    vdihedrald = vdihedrald*180.D0/pi
    RETURN
  END FUNCTION vdihedrald
END MODULE dlfhdlc_primitive
