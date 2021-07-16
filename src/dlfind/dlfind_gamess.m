c
c        Interface between GAMESS-UK and DL-FIND
c
c
c
c
c   Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
c   Tom Keal (thomas.keal@stfc.ac.uk)
c 
c   This file is part of DL-FIND.
c 
c   DL-FIND is free software: you can redistribute it and/or modify
c   it under the terms of the GNU Lesser General Public License as 
c   published by the Free Software Foundation, either version 3 of the 
c   License, or (at your option) any later version.
c 
c   DL-FIND is distributed in the hope that it will be useful,
c   but WITHOUT ANY WARRANTY; without even the implied warranty of
c   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c   GNU Lesser General Public License for more details.
c 
c   You should have received a copy of the GNU Lesser General Public 
c   License along with DL-FIND.  If not, see 
c   <http://www.gnu.org/licenses/>.
c
c     ..................................................................
      subroutine dlfind_gamess(core)
      implicit none
      REAL :: core(*)
c     local vars
      integer  :: nvar2, nprint_save,opg_root_int
      logical,external::  opg_root
INCLUDE(../m4/common/sizes)
INCLUDE(../m4/common/infoa)
INCLUDE(../m4/common/dlfind)
INCLUDE(../m4/common/seerch)
c     runlab for zruntp so we call the wavefunction analysis
INCLUDE(../m4/common/runlab)
c     restar for nprint (needed by optend)
INCLUDE(../m4/common/restar)
c     prnprn for print level (oprn)
INCLUDE(../m4/common/prnprn)
c     iofile for iwr
INCLUDE(../m4/common/iofile)
c     integer global_nodeid
c     external global_nodeid

      ! ****************************************************************
c      write(6,*)'calling dl-find icoord=',icoord
c      write(6,*)'calling dl-find ncons=',ncons_co
c      write(6,*)'calling dl-find nat=',nat
c      write(6,*)'calling dl-find nat=',nat

c     nprint is fiddled about with in the code to change the output
c     of the scf, so that there is less output after the first geometry
c     optimisation cycle. We therefore need to save the original nprint
      nprint_save=nprint
      npts = -1
      ncoord=3*nat
      nvar2=nat
      if(opg_root()) then
        opg_root_int=1
      else
        opg_root_int=0
      end if
      if(geom2.ne."") nvar2= nvar2 + 3*nat
c      write(6,*)'calling dl-find nvar=',3*nat
c      write(6,*)'calling dl-find nvar2=',nvar2
c      write(6,*)'calling dl-find nspec=',2*nat+5*ncons_co
      call dl_find(3*nat,nvar2,2*nat+5*ncons_co,opg_root_int,core)

c     write(6,*)'DL-FIND COMPLETED',global_nodeid()

c     restore print level
      nprint=nprint_save

c     header
      write(iwr,'(//1x,104("=")/)')

c     analysis of moment of inertia etc
      if(oprn(28)) call anamom(core)
c
c     analysis of bond lengths etc
      call intr(core)
c
c     Write out the orbitals
      call optend(core,nprint)
c
c     Set runtype to calculate wavefunction properties
      zruntp='prop'

      write(6,*)'DL-FIND RETURNING'

      return
      end
c
c     ..................................................................
      subroutine dlf_get_params(nvar,nvar2,nspec,
     +     coords,coords2,spec,ierr,
     +     tolerance,printl,maxcycle,maxene,
     +     tatoms,icoord_,iopt,iline,maxstep,
     +     scalestep,lbfgs_mem,nimage,nebk,
     +     dump,restart,nz,ncons,nconn,
     +     update,maxupd,delta,soft,inithessian,
     +     carthessian,tsrel,maxrot,tolrot,nframe,nmass,nweight,
     +     timestep,fric0,fricfac,fricp,
     +     imultistate,state_i,state_j,
     +     pf_c1,pf_c2,gp_c3,gp_c4,ln_t1,ln_t2,
     +     printf,tolerance_e,distort,massweight,minstep,maxdump,
     +     task, temperature, po_pop_size, po_radius,po_contraction,
     +     po_tolerance_r, po_tolerance_g, po_distribution,
     +     po_maxcycle, po_init_pop_size, po_reset, po_mutation_rate,
     +     po_death_rate, po_scalefac, po_nsave, ntasks, tdlf_farm, 
     +     n_po_scaling,
     +     neb_climb_test, neb_freeze_test,nzero, coupled_states, 
     +     qtsflag, imicroiter, maxmicrocycle, micro_esp_fit)
      implicit none
      integer   ,intent(in)      :: nvar 
      integer   ,intent(in)      :: nvar2
      integer   ,intent(in)      :: nspec
      REAL      ,intent(inout)   :: coords(nvar) ! start coordinates
      REAL      ,intent(inout)   :: coords2(nvar2) ! a real array that can be used
                                ! depending on the calculation
                                ! e.g. a second set of coordinates
      integer   ,intent(inout)   :: spec(nspec) ! specifications like fragment or frozen
      integer   ,intent(out)     :: ierr
      REAL      ,intent(inout)   :: tolerance
      integer   ,intent(inout)   :: printl
      integer   ,intent(inout)   :: maxcycle
      integer   ,intent(inout)   :: maxene
      integer   ,intent(inout)   :: tatoms
      integer   ,intent(inout)   :: icoord_
      integer   ,intent(inout)   :: iopt
      integer   ,intent(inout)   :: iline
      REAL      ,intent(inout)   :: maxstep
      REAL      ,intent(inout)   :: scalestep
      integer   ,intent(inout)   :: lbfgs_mem
      integer   ,intent(inout)   :: nimage
      REAL      ,intent(inout)   :: nebk
      integer   ,intent(inout)   :: dump
      integer   ,intent(inout)   :: restart
      integer   ,intent(inout)   :: nz
      integer   ,intent(inout)   :: ncons
      integer   ,intent(inout)   :: nconn
      integer   ,intent(inout)   :: update
      integer   ,intent(inout)   :: maxupd
      REAL      ,intent(inout)   :: delta
      REAL      ,intent(inout)   :: soft
      integer   ,intent(inout)   :: inithessian
      integer   ,intent(inout)   :: carthessian
      integer   ,intent(inout)   :: tsrel
      integer   ,intent(inout)   :: maxrot
      REAL      ,intent(inout)   :: tolrot
      integer   ,intent(inout)   :: nframe
      integer   ,intent(inout)   :: nmass
      integer   ,intent(inout)   :: nweight
      REAL      ,intent(inout)   :: timestep
      REAL      ,intent(inout)   :: fric0
      REAL      ,intent(inout)   :: fricfac
      REAL      ,intent(inout)   :: fricp
      integer   ,intent(inout)   :: imultistate
      integer   ,intent(inout)   :: state_i
      integer   ,intent(inout)   :: state_j
      REAL      ,intent(inout)   :: pf_c1
      REAL      ,intent(inout)   :: pf_c2
      REAL      ,intent(inout)   :: gp_c3
      REAL      ,intent(inout)   :: gp_c4
      REAL      ,intent(inout)   :: ln_t1
      REAL      ,intent(inout)   :: ln_t2
      integer   ,intent(inout)   :: printf
      REAL      ,intent(inout)   :: tolerance_e
      REAL      ,intent(inout)   :: distort
      integer   ,intent(inout)   :: massweight
      REAL      ,intent(inout)   :: minstep
      integer   ,intent(inout)   :: maxdump
c JMC new arguments
      integer   ,intent(inout)   :: task
      REAL      ,intent(inout)   :: temperature
      integer   ,intent(inout)   :: po_pop_size
      REAL      ,intent(inout)   :: po_radius
      REAL      ,intent(inout)   :: po_contraction
      REAL      ,intent(inout)   :: po_tolerance_r
      REAL      ,intent(inout)   :: po_tolerance_g
      integer   ,intent(inout)   :: po_distribution
      integer   ,intent(inout)   :: po_maxcycle
      integer   ,intent(inout)   :: po_init_pop_size
      integer   ,intent(inout)   :: po_reset
      REAL      ,intent(inout)   :: po_mutation_rate
      REAL      ,intent(inout)   :: po_death_rate
      REAL      ,intent(inout)   :: po_scalefac
      integer   ,intent(inout)   :: po_nsave
      integer   ,intent(inout)   :: ntasks
      integer   ,intent(inout)   :: tdlf_farm
      integer   ,intent(inout)   :: n_po_scaling
      REAL      ,intent(inout)   :: neb_climb_test
      REAL      ,intent(inout)   :: neb_freeze_test
      integer   ,intent(inout)   :: nzero
      integer   ,intent(inout)   :: coupled_states
      integer   ,intent(inout)   :: qtsflag
      integer   ,intent(inout)   :: imicroiter
      integer   ,intent(inout)   :: maxmicrocycle
      integer   ,intent(inout)   :: micro_esp_fit
c     local vars
      integer                    :: iat
      integer, external          :: jsubst
      REAL , external            :: amass_get
      character(8)               :: ztag_(nvar/3)
c     GAMESS common blocks
INCLUDE(../m4/common/sizes)
c infoa contains c(3,maxat)
INCLUDE(../m4/common/infoa)
INCLUDE(../m4/common/runlab)
INCLUDE(../m4/common/dlfind)
      ! ****************************************************************
c      write(6,*)'in get_params: nvar,nvar2,nspec',nvar,nvar2,nspec
      ierr=0
      if(nvar.ne.3*nat) call caserr2('nvar is not 3*nat in '//
     &     'dlf_get_params')
      coords(:)=reshape(c(1:3,1:nat),(/nvar/))
      nmass=nat
c     spec should contain: 
c       nvar spec (1 or constraint)
c       nvar atomic number (Z)
c       constraint data
      do iat=1,nat
        spec(nat+iat)=jsubst(ztag(iat))
      end do
c      print*,"spec: ",spec
      delta=delta_co
      icoord_=icoord
      if(icoord>=100.and.icoord<200) then
c       NEB
        nimage=nimage_co
        nebk=nebk_co
c       New options, not currently implemented in GAMESS
        neb_climb_test=-1.0d0
        neb_freeze_test=-1.0d0
      end if
c     Include all atoms into the first residue in case of internals
c     At the moment, input of HDLC residues is not possible
      if(mod(icoord,10)>0.and.mod(icoord,10)<5) then
        spec(1:nat)=1
      end if
c     Optimiser
      iopt=iopt_co
      lbfgs_mem=mem_co
      timestep=time_co
      fric0=fri0_co
      fricfac=frif_co
      fricp=frip_co
      if((iopt.eq.3.or.iopt.lt.0).and.icoord.lt.100) then
        iline=1
      end if
c     Update
      update=upd_co
      maxupd=rec_co
CTWK  This is correct if fd_co=1 -> one point, =2 -> two point FD:
      inithessian = fd_co
      soft=soft_co
      tolerance=4.D0/9.D0*tol_co
      tolerance_e=1.D0
      if(rst_co) restart=1
      dump=dump_co
c
      maxcycle=maxc_co
      maxstep=maxs_co
      printl=4
      nz=nat
      ncons=ncons_co
      printf=6
c     coords2
      if(geom2.ne."") then
        call read_xyz(geom2,nat,ztag_,coords2(1:nvar),iat)
        if(iat.ne.nat) call caserr2('number of atoms in second set of'//
     &       ' coordinates not equal to first')
        nframe=1
c       the masses
        do iat=1,nat
          coords2(iat+3*nat)=amass_get(1,iat)
        end do
      else
        nframe=0
c       the masses
        do iat=1,nat
          coords2(iat)=amass_get(1,iat)
        end do
      endif
      nweight=0
c      print*,"coords2",coords2
c     miscellaneous
      task=task_co
      temperature=temperature_co
c     parallel optimization
      po_pop_size=po_pop_size_co
      po_radius=po_radius_co
      po_contraction=po_contraction_co
      po_tolerance_r=po_tolerance_r_co
      po_tolerance_g=po_tolerance_g_co
      po_distribution=po_distribution_co
      po_maxcycle=po_maxcycle_co
      po_init_pop_size=po_init_pop_size_co
      po_reset=po_reset_co
      po_mutation_rate=po_mutation_rate_co
      po_death_rate=po_death_rate_co
      po_scalefac=po_scalefac_co
      po_nsave=po_nsave_co

      n_po_scaling=0 ! ??? for testing

c     taskfarming
      ntasks=ntasks_co

c JMC hardwire the tdlf_farm=0 for now -- is there any need for 
c tdlf_farm to be settable from within gamess?
c this will cause dl-find to load the split commss stuff from gamess

      tdlf_farm=0

      end 
c
c     ..................................................................
      subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,
     +     core,status)
      implicit none
      integer   ,intent(in)    :: nvar
      REAL      ,intent(in)    :: coords(nvar)
      REAL      ,intent(out)   :: energy
      REAL      ,intent(out)   :: gradient(nvar)
      integer   ,intent(in)    :: iimage
      REAL                     :: core(*)
      integer   ,intent(out)   :: status
c     local vars
      REAL     :: co,g,dx,func
      integer,external :: lensec
      integer          :: isize,m17=17,natdlf
INCLUDE(../m4/common/dlfind)
INCLUDE(../m4/common/sizes)
INCLUDE(../m4/common/machin)
INCLUDE(../m4/common/dump3)
INCLUDE(../m4/common/restri)
INCLUDE(../m4/common/funct)
INCLUDE(../m4/common/infoa)
INCLUDE(../m4/common/restar)
c     it seems we have to rely on this common block from valopt (m4/optim.m)
      common/miscop/co(maxat*3),g(maxat*3),dx(maxat*3),func
c
c ----- allocate gradient section on dumpfile
c
      isize = lensec(nvar*nvar) + lensec(mach(7))
      call secput(isect(495),m17,isize,ibl3g)
      ibl3hs = ibl3g + lensec(mach(7))
c
c ----- set coordinates and calculate energy and gradient
c
      natdlf=nvar/3
      c(1:3,1:natdlf)=reshape(coords(:),(/3,natdlf/))

c     calculate energy and gradient in gamess
      call valopt(core)

c     get back the resulting energy and gradient
      energy=func
      gradient(:)=egrad(1:nvar)

      status=irest
      end
c
c     ..................................................................
      subroutine dlf_get_hessian(nvar,coords,hessian,status)
c     subroutine dlf_get_hessian(nvar,coords,hessian,status,core)
c                               !  get the hessian at a given geometry
c     This routine does not work for the moment, as core is not passed to it
      implicit none
      integer   ,intent(in)    :: nvar
      REAL      ,intent(in)    :: coords(nvar)
      REAL      ,intent(out)   :: hessian(nvar,nvar)
      integer   ,intent(out)   :: status
c     REAL                     :: core(*)
      REAL                     :: core
c     local vars
      integer          :: isize,nc2,len3,lenc,l3,m17=17
      integer,external :: lensec
INCLUDE(../m4/common/sizes)
INCLUDE(../m4/common/infoa)
INCLUDE(../m4/common/misc)
INCLUDE(../m4/common/cndx41)
INCLUDE(../m4/common/dlfind)
INCLUDE(../m4/common/machin)
INCLUDE(../m4/common/dump3)
INCLUDE(../m4/common/restri)
INCLUDE(../m4/common/funct)
INCLUDE(../m4/common/restar)
c     ******************************************************************

      status=1
      return

      call flushout()
c     
c ----- allocate gradient section on dumpfile
c
      l3 = num*num
      len3 = lensec(l3)
      lenc = lensec(mach(9))
      nc2 = nvar**2
      isize = lensec(nc2) + lensec(mach(7))
      call secput(isect(495),m17,isize,ibl3g)

      c(1:3,1:nvar)=reshape(coords(:),(/3,nvar/))
      call flushout()
      if (omp2) then
         call mp2dd(core)
      else if( mp3) then
         call caserr2
     +   ('correlated second derivatives - only mp2 allowed')
      else
        call flushout()
        call scfdd(core)
        call flushout()
        call flushout()
        call anairr(core)
        call flushout()
        call flushout()
      end if
c
c     read cartesian force constant matrix
c
c      call rdfcm(vec,'iranal')
      call rdfcm(hessian,'dl-find')

c      hessian(:,:)=0.D0
      status=0
      end

c
c     ..................................................................
      subroutine dlf_get_multistate_gradients(nvar,coords,energy,
     +     gradient,coupling,needcoupling,iimage,status)
c     Dummy routine for multistate gradients. This option is 
c     not supported in Gamess
      implicit none
      integer     :: nvar
      REAL        :: coords
      REAL        :: energy
      REAL        :: gradient
      REAL        :: coupling
      integer     :: needcoupling
      integer     :: iimage
      integer     :: status
      end
c
c     ..................................................................
      subroutine dlf_put_coords(nvar,mode,energy,coords,iam)
      implicit none
      integer   ,intent(in)    :: nvar
      integer   ,intent(in)    :: mode
      integer   ,intent(in)    :: iam
      REAL      ,intent(in)    :: energy
      REAL      ,intent(in)    :: coords(nvar)
      end

c
c     ..................................................................
      subroutine dlf_update()
      implicit none
c     dummy routine here
      end

      subroutine dlf_error()
      implicit none
INCLUDE(../m4/common/errcodes)
      call gamerr(
     &     'DL-FIND error',
     &     ERR_NO_CODE, ERR_UNLUCKY, ERR_SYNC, ERR_NO_SYS)
      end
c
c     ..................................................................
      subroutine dlf_geom(maxat,ztag,cat,nat)
c     Read in the geometrie(s)
c     at the moment, the file name is hardcoded to geom.xyz !!
c     a second geometry may be read in from a file specified after the keyword geom
      implicit none
      integer      ,intent(in) :: maxat
      character(8) ,intent(out):: ztag(maxat)
      REAL         ,intent(out):: cat(3,maxat)
      integer      ,intent(out):: nat
c     ******************************************************************
      call read_xyz("geom.xyz",maxat,ztag,cat,nat)
      end
c

      logical function isalpha(c)
c     true for ascii [A-Z,a-z]
      implicit none
      character(1), intent(in)::c
      character(1)            ::a,z
      logical                 ::alpha
      a='a'
      z='z'
      alpha=((iachar(c).ge.iachar(a)).and.(iachar(c).le.iachar(z)))
      a='A'
      z='Z'
      alpha=alpha.or.
     &((iachar(c).ge.iachar(a)).and.(iachar(c).le.iachar(z)))
      isalpha=alpha
      end function isalpha

c     ..................................................................
*     read_xyz now accepts both
*     neb .xyz output files (cartesian angstrom)
*         header           : # of atoms
*         with atom records: tag x y z
*     and gamess 'nuclear coordinates' blocks (cartesian,au or angstrom)
*         header           : <au/an> # of atoms
*         with atom records: x y z Q tag
*     fname no longer a dummy.

      subroutine read_xyz(fname,maxat,ztag,cat,nat)
      implicit none
      integer      ,intent(in) :: maxat
      character(8) ,intent(out):: ztag(maxat)
      REAL         ,intent(out):: cat(3,maxat)
      integer      ,intent(out):: nat
c     local vars
      logical       :: tchk,guessunit,isalpha
      logical       :: dlfformat,gamessformat
      integer       :: iunit=5111,iat,wantnat
      character(2)  :: ch2
      REAL          :: ang=0.529177249d0,scalecoor,fdum
      character(32) :: theunits
      character(256):: buffer
c     old string type
      character*(*) fname
c     ******************************************************************
      guessunit=.false.
      dlfformat=.false.
      gamessformat=.false.
      inquire(FILE=fname,EXIST=tchk)
      if(.not.tchk) then
c       print*,"Input geometry NOT read from xyz file"
        nat=0
        return
      end if
c     print*,'dl-find: reading second geometry from xyz file'
c     print*,'maxat = ',maxat
      open (unit=iunit,file=fname,err=201)
c     possible header: 'angstrom/au #atoms'
      read(iunit,*,err=100,end=200) theunits,wantnat
      goto 150
100   continue
c     print*,'error on read 1 theunits,nat'
      guessunit=.true.
      rewind(iunit)
c     possible header:  '#atoms'
      read(iunit,*,err=110,end=200) wantnat
      goto 150
110   continue
c     no header a all...
c     print*,'error on read 2 nat'
      rewind(iunit)

150   continue
      if (.not.guessunit) then
*        print*,'dl-find: read_xyz, file uses ',theunits
*        print*,'dl-find: read_xyz, file has ',wantnat,' atoms'
         if (theunits(1:2).eq.'an'.or.theunits(1:2).eq.'AN') then
            scalecoor=ang
         else
            scalecoor=1d0
         endif
      else
*        original default: angstrom
         scalecoor=ang
      endif

      nat=0
      iat=1
      do while (.true.)
151      continue
         if (gamessformat) then
            read(iunit,'(A256)',end=161,err=200) buffer
            do while (len(trim(buffer)).lt.6) 
*              print*,'dl-find: read_xyz skipping line ',buffer
               read(iunit,'(A256)',end=161,err=200) buffer
            end do
            read(buffer,*,err=200) cat(:,iat),fdum,ch2
*           print*,'ch2 is ',ch2,' cat= ',cat(:,iat)
         else if (dlfformat) then
            read(iunit,'(A256)',end=161,err=200) buffer
            do while (len(trim(buffer)).lt.6) 
*              print*,'dl-find: read_xyz skipping line ',buffer
               read(iunit,'(A256)',end=161,err=200) buffer
            end do
            read(buffer,*,err=200) ch2,cat(:,iat)
*           print*,'ch2 is ',ch2,' cat= ',cat(:,iat)
         else
            read(iunit,*,end=161,err=200) ch2
            backspace(iunit)
            if (isalpha(ch2(1:1))) then
               dlfformat=.true.
*           print*,'dl-find: read_xyz, file uses default format'
            else
               gamessformat=.true.
*           print*,'dl-find: read_xyz, file uses gamess format'
            endif
            goto 151
         end if
         ztag(iat)=ch2
         iat=iat+1
      end do
161   continue
*     normal exit
      nat=iat-1
*     print*,'found ',nat,' atoms'
      close(iunit)
      cat=cat/scalecoor
      return

*     (premature) EOF
 201  print*,"dl-find read_xyz Error: end of input file reached"
      go to 203
*     read error on input record
 200  print*,"dl-find read_xyz Error reading input record"
 203  close(iunit)
      go to 205
*     open failed
 204  print*,"dl-find read_xyz Error opening input file"
 205  nat=0
      return
      end
_IF(splitcomm)
c
c     ..................................................................
      subroutine dlf_put_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)
      implicit none
      integer, intent(in) :: dlf_nprocs ! total number of processors
      integer, intent(in) :: dlf_iam    ! my rank in mpi_comm_world
      integer, intent(in) :: dlf_global_comm ! world-wide communicator
c     ******************************************************************

ccc variable in the calling program = corresponding dummy argument

      end subroutine dlf_put_procinfo
c
c     ..................................................................
      subroutine dlf_get_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)
      implicit none
      integer :: dlf_nprocs ! total number of processors
      integer :: dlf_iam ! my rank, from 0, in mpi_comm_world
      integer :: dlf_global_comm ! world-wide communicator
INCLUDE(../m4/common/sizes)
INCLUDE(../m4/common/nodinf)
INCLUDE(../m4/common/mpidata)
c     ******************************************************************
_IF(mpi)
      include 'mpif.h'

ccc dummy argument = corresponding variable in the calling program
 
      dlf_nprocs = nnodes
      dlf_iam = minode
      dlf_global_comm = MPI_COMM_WORLD
_ENDIF

      end subroutine dlf_get_procinfo
c
c     ..................................................................
c
c  Not Currently Used
c

      subroutine dlf_put_taskfarm(dlf_ntasks, dlf_nprocs_per_task, 
     +    dlf_iam_in_task, dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)
      implicit none
      integer, intent(in) :: dlf_ntasks          ! number of taskfarms
      integer, intent(in) :: dlf_nprocs_per_task ! no of procs per farm
      integer, intent(in) :: dlf_iam_in_task     ! my rank in my farm
      integer, intent(in) :: dlf_mytask          ! rank of my farm
      integer, intent(in) :: dlf_task_comm       ! comm within each farm
      integer, intent(in) :: dlf_ax_tasks_comm   ! comm involving the 
                                              ! i-th proc from each farm
INCLUDE(../m4/common/sizes)
INCLUDE(../m4/common/nodinf)
INCLUDE(../m4/common/mpidata)
c     ******************************************************************

ccc variable in the calling program = corresponding dummy argument

      minode = dlf_iam_in_task
      nnodes = dlf_nprocs_per_task
      MPI_COMM_GAMESS = dlf_task_comm
      MPI_COMM_WORKERS = dlf_task_comm

      end subroutine dlf_put_taskfarm
c
c     ..................................................................
      subroutine dlf_get_taskfarm(dlf_ntasks, dlf_nprocs_per_task, 
     +    dlf_iam_in_task, dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)
      implicit none

INCLUDE(../m4/common/parcntl)

      integer :: dlf_ntasks          ! number of taskfarms
      integer :: dlf_nprocs_per_task ! no of procs per farm
      integer :: dlf_iam_in_task     ! my rank, from 0, in my farm
      integer :: dlf_mytask          ! rank of my farm, from 0
      integer :: dlf_task_comm       ! communicator within each farm
      integer :: dlf_ax_tasks_comm   ! communicator involving the
                                     ! i-th proc from each farm
c     ******************************************************************

      dlf_ntasks  = ntasks    
      dlf_nprocs_per_task  = nprocs_per_task
      dlf_iam_in_task = iam_in_task
      dlf_mytask = mytask    
      dlf_task_comm  = task_comm
      dlf_ax_tasks_comm  = ax_tasks_comm
      
      return
      end subroutine dlf_get_taskfarm
_ENDIF
c
c     ..................................................................
      subroutine dlf_output(guk_stdout,guk_stderr)
      use dlf_global, only: glob,stderr,stdout,keep_alloutput
      implicit none
      integer :: guk_stdout
      integer :: guk_stderr
c     ******************************************************************

      if (guk_stdout >= 0) stdout = guk_stdout
      if (guk_stderr >= 0) stderr = guk_stderr

      if (glob%nprocs > 1) then
        ! write some info on the parallelization
        write(stdout,'(1x,a,i10,a)')"I have rank ",glob%iam,
     &    " in mpi_comm_world"
        write(stdout,'(1x,a,i10)')"Total number of processors = ",
     &    glob%nprocs
        if (keep_alloutput) then
           write(stdout,'(1x,a)')"Keeping output from all processors"
        else
           write(stdout,'(1x,a)')
     &    "Not keeping output from processors /= 0"
        end if
      end if

      return
      end subroutine dlf_output
