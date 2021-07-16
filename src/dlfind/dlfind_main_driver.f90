! **********************************************************************
! **        Driver file for dl_find                                   **
! ** isystem:                                                         **
! **  1 - Mueller-Brown potential (2 Dim)                             **
! **  2 - 3 Lennard-Jones Atoms (9 Dim)                               **
! **  3 - 100 Lennard-Jones Atoms (300 Dim)                           **
! **  4 - Eckart Barrier (1 Dim)                                      **
! **  5 - Quartic potential (1 Dim)                                   **
! **  6 - Vibrational sufaces by molpro                               **
! **  7 - A potential for which the MEP splits in two (2 Dim)         **
! **        (by Judith Rommel)                                        **
! **  8 - A polynomial potential in more dimensions                   **
! **                                                                  **
! **********************************************************************

!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
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

module driver_module
  use dlf_parameter_module, only: rk
 ! use vib_pot !uncommented to have a running system
  integer,parameter :: isystem=9! decide which system to run on

! variables for non-continuous differentiable MEP potential
  real(rk) :: smallbarvar !parameter to control the hight of the small barrier
end module driver_module

module driver_parameter_module
  use dlf_parameter_module
  ! variables for Mueller-Brown potential
  real(rk) :: acappar(4),apar(4),bpar(4),cpar(4),x0par(4),y0par(4)
  ! variables for Lennard-Jones potentials
  real(rk),parameter :: epsilon=1.D-1
  real(rk),parameter :: sigma=1.D0
!!$  ! variables for the Eckart potential (1D), taken from Andri's thesis
  real(rk),parameter :: Va= -0.191D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
  real(rk),parameter :: Vb=  1.343D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
  real(rk),parameter :: alpha= 5.762D0/1.889725989D0 !2nd factor is for conversion from Angstrom to Bohr (a.u.)

  ! modified Eckart potential to have a comparison
!!$  REAL(8), PARAMETER :: Va= -0.091D0*0.0367493254D0  !2nd factor is conversion to E_h from eV
!!$  REAL(8), PARAMETER :: Vb=  1.343D0*0.0367493254D0  !2nd factor is conversion to E_h from eV
!!$  REAL(8), PARAMETER :: alpha =20.762D0/1.889725989D0 
!!$  real(rk),parameter :: Va= -0.191D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
!!$  real(rk),parameter :: Vb=  1.343D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
!!$  real(rk),parameter :: alpha= 20.762D0/1.889725989D0 !2nd factor is for conversion from Angstrom to Bohr (a.u.)

  ! "fitted" to Glum SN-1-Glu, E->D
!  real(rk),parameter :: Va= 0.D0 ! symmetric (real: almost symmetric)
!  real(rk),parameter :: Vb= 4.D0*74.81D0/2625.5D0 ! 4*barrier for symmetric
!  real(rk),parameter :: alpha= 2.38228D0 

!!$  ! Eckart for comparison with scattering
!!$  real(rk),parameter :: Va= -100.D0
!!$  real(rk),parameter :: Vb= 373.205080756888D0
!!$  real(rk),parameter :: alpha= 7.07106781186547D0

  real(rk) :: xvar,yvar
  ! quartic potential
  real(rk),parameter :: V0=1.D0
  real(rk),parameter :: X0=5.D0
  ! polymul
  integer ,parameter :: num_dim=3 ! should be a multiple of 3
  real(rk),parameter :: Ebarr=80.D0/2526.6D0  ! >0
  real(rk) :: dpar(num_dim)
!!$  real(rk),parameter :: vdamp=15.D0 ! change of frequencies towards TS. +inf -> no change
end module driver_parameter_module

!program main
!  use driver_module
!  use driver_parameter_module
!  use dlfind_quick, only: quick_opt
  !use vib_pot
!  implicit none
!  integer :: ivar

!  call dlf_mpi_initialize() ! only necessary for a parallel build; 
                            ! can be present for a serial build
!  call dlf_output(6,0)

!  call driver_init

!  select case (isystem)
!  case (1)
    !call system ("rm -f dimer.xy")
    !do ivar=1,100
!      call dl_find(9,3,6,1) ! no frame in coords2
!      call dl_find(3,4,3,1) ! one frame in coords2
!      call dl_find(3,16,1,1) ! 5 frames in coords2
!      call dl_find(3,58,1,1) ! 19 frames in coords2
    !  call system ("echo '' >> dimer.xy")
    !end do
!  case (2)
   ! call dl_find(12,1,4)
!    call dl_find(12,16,8,1) ! 1 frame + masses
!  case (3)
    !call dl_find(99,1,33)
    !call dl_find(30,1,15)

    ! LJ-particle surface with one atom hopping on it
!    call dl_find(9,3,6,1) ! 1 frame + weigths + masses

 ! case (4,5)
!    call dl_find(3,4,2,1)

!  case (6)
!    call dl_find(12,16,8,1) ! 1 frame + masses
    !call read_pot_destroy
 ! case (7)
!    call dl_find(3,4,1,1) ! one frame in coords2

!  case (8)
!    call dl_find(num_dim,num_dim+num_dim/3,num_dim/3,1) ! one frame + masses in coords2

!  case (9)
!    call quick_opt()

 ! case default
  !  call dlf_mpi_finalize() ! only necessary for a parallel build;
                            ! can be present for a serial build

!!    stop "Wrong isystem"
!  end select

!  call dlf_mpi_finalize() ! only necessary for a parallel build;
                          ! can be present for a serial build


!end program main

! **********************************************************************
! subroutines that have to be provided to dl_find from outside
! **********************************************************************

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_params(nvar,nvar2,nspec,coords,coords2,spec,ierr, &
    tolerance,printl,maxcycle,maxene,tatoms,icoord, &
    iopt,iline,maxstep,scalestep,lbfgs_mem,nimage,nebk,dump,restart,&
    nz,ncons,nconn,update,maxupd,delta,soft,inithessian,carthessian,tsrel, &
    maxrot,tolrot,nframe,nmass,nweight,timestep,fric0,fricfac,fricp, &
    imultistate, state_i,state_j,pf_c1,pf_c2,gp_c3,gp_c4,ln_t1,ln_t2, &
    printf,tolerance_e,distort,massweight,minstep,maxdump,task,temperature, &
    po_pop_size,po_radius,po_contraction,po_tolerance_r,po_tolerance_g, &
    po_distribution,po_maxcycle,po_init_pop_size,po_reset,po_mutation_rate, &
    po_death_rate,po_scalefac,po_nsave,ntasks,tdlf_farm,n_po_scaling, &
    neb_climb_test, neb_freeze_test, &
    nzero, coupled_states, qtsflag,&
    imicroiter, maxmicrocycle, micro_esp_fit)
  use dlf_parameter_module, only: rk
  use driver_parameter_module
  use driver_module
  !use vib_pot
  implicit none
  integer   ,intent(in)      :: nvar 
  integer   ,intent(in)      :: nvar2
  integer   ,intent(in)      :: nspec
  real(rk)  ,intent(inout)   :: coords(nvar) ! start coordinates
  real(rk)  ,intent(inout)   :: coords2(nvar2) ! a real array that can be used
                                               ! depending on the calculation
                                               ! e.g. a second set of coordinates
  integer   ,intent(inout)   :: spec(nspec)  ! specifications like fragment or frozen
  integer   ,intent(out)     :: ierr
  real(rk)  ,intent(inout)   :: tolerance
  real(rk)  ,intent(inout)   :: tolerance_e
  integer   ,intent(inout)   :: printl
  integer   ,intent(inout)   :: maxcycle
  integer   ,intent(inout)   :: maxene
  integer   ,intent(inout)   :: tatoms
  integer   ,intent(inout)   :: icoord
  integer   ,intent(inout)   :: iopt
  integer   ,intent(inout)   :: iline
  real(rk)  ,intent(inout)   :: maxstep
  real(rk)  ,intent(inout)   :: scalestep
  integer   ,intent(inout)   :: lbfgs_mem
  integer   ,intent(inout)   :: nimage
  real(rk)  ,intent(inout)   :: nebk
  integer   ,intent(inout)   :: dump
  integer   ,intent(inout)   :: restart
  integer   ,intent(inout)   :: nz
  integer   ,intent(inout)   :: ncons
  integer   ,intent(inout)   :: nconn
  integer   ,intent(inout)   :: update
  integer   ,intent(inout)   :: maxupd
  real(rk)  ,intent(inout)   :: delta
  real(rk)  ,intent(inout)   :: soft
  integer   ,intent(inout)   :: inithessian
  integer   ,intent(inout)   :: carthessian
  integer   ,intent(inout)   :: tsrel
  integer   ,intent(inout)   :: maxrot
  real(rk)  ,intent(inout)   :: tolrot
  integer   ,intent(inout)   :: nframe
  integer   ,intent(inout)   :: nmass
  integer   ,intent(inout)   :: nweight
  real(rk)  ,intent(inout)   :: timestep
  real(rk)  ,intent(inout)   :: fric0
  real(rk)  ,intent(inout)   :: fricfac
  real(rk)  ,intent(inout)   :: fricp
  integer   ,intent(inout)   :: imultistate
  integer   ,intent(inout)   :: state_i
  integer   ,intent(inout)   :: state_j
  real(rk)  ,intent(inout)   :: pf_c1  
  real(rk)  ,intent(inout)   :: pf_c2  
  real(rk)  ,intent(inout)   :: gp_c3  
  real(rk)  ,intent(inout)   :: gp_c4
  real(rk)  ,intent(inout)   :: ln_t1  
  real(rk)  ,intent(inout)   :: ln_t2  
  integer   ,intent(inout)   :: printf
  real(rk)  ,intent(inout)   :: distort
  integer   ,intent(inout)   :: massweight
  real(rk)  ,intent(inout)   :: minstep
  integer   ,intent(inout)   :: maxdump
  integer   ,intent(inout)   :: task
  real(rk)  ,intent(inout)   :: temperature
  integer   ,intent(inout)   :: po_pop_size
  real(rk)  ,intent(inout)   :: po_radius
  real(rk)  ,intent(inout)   :: po_contraction
  real(rk)  ,intent(inout)   :: po_tolerance_r
  real(rk)  ,intent(inout)   :: po_tolerance_g
  integer   ,intent(inout)   :: po_distribution
  integer   ,intent(inout)   :: po_maxcycle
  integer   ,intent(inout)   :: po_init_pop_size
  integer   ,intent(inout)   :: po_reset
  real(rk)  ,intent(inout)   :: po_mutation_rate
  real(rk)  ,intent(inout)   :: po_death_rate
  real(rk)  ,intent(inout)   :: po_scalefac
  integer   ,intent(inout)   :: po_nsave
  integer   ,intent(inout)   :: ntasks
  integer   ,intent(inout)   :: tdlf_farm
  integer   ,intent(inout)   :: n_po_scaling
  real(rk)  ,intent(inout)   :: neb_climb_test
  real(rk)  ,intent(inout)   :: neb_freeze_test
  integer   ,intent(inout)   :: nzero
  integer   ,intent(inout)   :: coupled_states
  integer   ,intent(inout)   :: qtsflag
  integer   ,intent(inout)   :: imicroiter
  integer   ,intent(inout)   :: maxmicrocycle
  integer   ,intent(inout)   :: micro_esp_fit
  ! local variables
  real(rk)                   :: svar
  integer                    :: iat,jat
  interface
    subroutine read_rand(arr)
      use dlf_parameter_module, only: rk
      use dlf_global, only : glob
      real(rk)  :: arr(:)
    end subroutine read_rand
  end interface
! **********************************************************************
  ierr=0
  tsrel=1

  print*,"External sizes:",nvar,nvar2,nspec
  ! Minima of Mueller-Brown potential:
  ! left:   -0.5582236346340204   1.4417258418038705   energy:  -0.6919788547639341
  ! middle: -0.050010822944531706 0.4666941048659066   energy:  -0.38098027419650493
  ! right:   0.623499404927291    0.028037758526434815 energy:  -0.5102203967776056
  ! TS:
  ! main:   -0.8220015634663578   0.624312796202859    energy:  -0.19181529956913862
  ! other:   0.21248658127591744  0.2929883285085865   energy:  -0.34079688732228874

!!$  spec(1)=-1
!!$  spec(2)=-34
!!$  spec(3)=-4
!!$  spec(:)=1
!!$  spec(3)=-34
!!$  spec(1:3)=2
!!$  spec(6)=2
!!$  spec(9:10)=0
  ncons=1
  ! cosntraint
!!$  spec(11)=1
!!$  spec(12)=1
!!$  spec(13)=2
!!$  spec(14:15)=0

  ncons=0
  nconn=0

  coords(:)=-1.D0
  spec(:)=0
  
!!$  ! Near Saddle point:
! coords=(/ -0.6D0, 0.6D0 /) 

  coords2(:)=-1.D0
  if(isystem==1) then

    nzero=0
    open(unit=10, file="water.in")
    do iat= 0,2
        read(10,*)(coords(iat+jat), jat=1,3)
    end do
    close (10)
!    coords=(/ -0.8220015634663578D0,   0.624312796202859D0, 0.D0 /)! main saddle point

!    coords=(/ -0.050010822944531706D0, 0.4666941048659066D0 , 0.D0 /)! middle minimum
!!$!    coords=(/ 0.623499404927291D0, 0.028037758526434815D0 , 0.D0 /)! right minimum
!!$!    coords=(/ 0.05D0,   0.7D0, 0.D0 /)
!!$    ! Main TS and direction
!!$    nframe=1
!!$    coords2(1:3)=(/ 0.8D0 ,  -0.5D0,  0.D0 /)

    ! Main minima
!    coords=(/ -0.5582236346340204D0,   1.4417258418038705D0, 0.D0 /)
    nframe=0
!    coords2(1:3)=(/ 0.623499404927291D0,    0.028037758526434815D0, 0.D0 /)

!!$    ! 6 points converged at 0.8 T_c, but with non-doubled path
!!$    nframe=5
    ! pairwise identical
!!$    coords=(/  -0.4493311064542917D0 , 0.5419479158349173D0 , 0.D0 /)
!!$    coords2(1:nframe*3)=(/ &
!!$        -0.5952499491610727D0 , 0.7072076082065596D0 , 0.D0 , &
!!$        -0.74556821224306D0 , 1.1707853456208879D0 , 0.D0 , &
!!$        -0.74556821224306D0 , 1.1707853456208879D0 , 0.D0 , &
!!$        -0.5952499491610727D0 , 0.7072076082065596D0 , 0.D0 , &
!!$        -0.4493311064542917D0 , 0.5419479158349173D0 , 0.D0 /)
!!$    ! turning points non-doubled
!!$    coords=(/ -0.3578971447649985D0 , 0.501608351688023D0 , 0.D0 /)
!!$    coords2(1:nframe*3)=(/ &
!!$        -0.43170926134006843D0 , 0.5293693376680748D0 , 0.D0 , &
!!$        -0.5875518956609882D0 , 0.6297820581123532D0 , 0.D0 ,&
!!$        -0.7976650261570393D0 , 0.8923927204798402D0 , 0.D0 ,&
!!$        -0.5875518956609882D0 , 0.6297820581123532D0 , 0.D0 ,&
!!$        -0.43170926134006843D0 , 0.5293693376680748D0 , 0.D0 /)


!!$    ! 20 points approx. converged at 7.D4 K This was used for the P-RFO convergence tests
!!$    coords=(/    -0.8426643D0 , 0.9151530D0 , 0.D0 /)
!!$    nframe=19
!!$    coords2(1:nframe*3)=(/ &
!!$        -0.8341730D0 , 0.9016234D0 , 0.D0, &
!!$        -0.8167761D0 , 0.8750182D0 , 0.D0, &
!!$        -0.7902620D0 , 0.8368417D0 , 0.D0, &
!!$        -0.7552811D0 , 0.7902725D0 , 0.D0, &
!!$        -0.7140001D0 , 0.7401491D0 , 0.D0, &
!!$        -0.6694554D0 , 0.6916154D0 , 0.D0, &
!!$        -0.6245622D0 , 0.6483821D0 , 0.D0, &
!!$        -0.5814185D0 , 0.6121439D0 , 0.D0, &
!!$        -0.5412253D0 , 0.5829760D0 , 0.D0, &
!!$        -0.5046521D0 , 0.5601122D0 , 0.D0, &
!!$        -0.4719413D0 , 0.5424788D0 , 0.D0, &
!!$        -0.4431635D0 , 0.5290462D0 , 0.D0, &
!!$        -0.4182906D0 , 0.5188846D0 , 0.D0, &
!!$        -0.3972339D0 , 0.5112838D0 , 0.D0, &
!!$        -0.3798929D0 , 0.5056672D0 , 0.D0, &
!!$        -0.3661631D0 , 0.5016168D0 , 0.D0, &
!!$        -0.3559567D0 , 0.4988193D0 , 0.D0, &
!!$        -0.3491923D0 , 0.4970686D0 , 0.D0, &
!!$        -0.3458243D0 , 0.4962209D0 , 0.D0 &
!!$        /)

    ! right minimum for rates
!    coords=(/0.623499404927291D0,    0.028037758526434815D0, 0.D0 /)

!!$    ! 20 points approx. converged at 9.D4 K
!!$    coords=(/    -0.8426643D0 , 0.9151530D0 , 0.D0 /)
!!$    nframe=19
!!$    coords2(1:nframe*3)=(/ &
!!$        -0.8862898D0 , 0.7529666D0 , 0.D0, &
!!$        -0.8830748D0 , 0.7491050D0 , 0.D0, &
!!$        -0.8767162D0 , 0.7415692D0 , 0.D0, &
!!$        -0.8673471D0 , 0.7307005D0 , 0.D0, &
!!$        -0.8551798D0 , 0.7169933D0 , 0.D0, &
!!$        -0.8405764D0 , 0.7011254D0 , 0.D0, &
!!$        -0.8239636D0 , 0.6838182D0 , 0.D0, &
!!$        -0.8059024D0 , 0.6658717D0 , 0.D0, &
!!$        -0.7869926D0 , 0.6480244D0 , 0.D0, &
!!$        -0.7678376D0 , 0.6309045D0 , 0.D0, &
!!$        -0.7490010D0 , 0.6149905D0 , 0.D0, &
!!$        -0.7309768D0 , 0.6006029D0 , 0.D0, &
!!$        -0.7141770D0 , 0.5879203D0 , 0.D0, &
!!$        -0.6989314D0 , 0.5770092D0 , 0.D0, &
!!$        -0.6854953D0 , 0.5678584D0 , 0.D0, &
!!$        -0.6740516D0 , 0.5604040D0 , 0.D0, &
!!$        -0.6647626D0 , 0.5545798D0 , 0.D0, &
!!$        -0.6577119D0 , 0.5502927D0 , 0.D0, &
!!$        -0.6529801D0 , 0.5474803D0 , 0.D0, &
!!$        -0.6506084D0 , 0.5460899D0 , 0.D0 &
!!$        /)

    nmass=3
!    coords2=(/ 12.011, 1.00794, 1.00794, 1.00794 /)
    coords2=(/ 15.999, 1.00794, 1.00794 /)
    coords2(nvar2)=1.0078250321D0 ! mass: last entry in coords2, here using mass of protium
!    coords2(nvar2)=10.00 ! mass: last entry in coords2, here using mass of more 


!    print*,"nspec",nspec
!    spec(1)=-4

  end if
  if(isystem==2) then
    ! minimum distance is 1.1224620
!!$    coords(:)=0.D0
!!$    coords(4)=-1.D0
!!$    coords(7)=0.6D0
!!$    coords(8)=1.D0
!!$
!!$    coords2(:)=0.D0
!!$    coords2(4)=-1.D0
!!$    coords2(7)=0.6D0
!!$    coords2(8)=-1.D0

    ! Arrangements of 4 LJ-Particles:
    ! Square with a=1.1126198 is a 2nd order Saddle point
    ! Rombus (60.27 deg) with a=1.1202310 (1.124800) is a 1st order Saddle point

    ! This is a rombus quite near to the TS
    svar=1.1224620D0 ! very good
    svar=1.0D0 
    coords(:)=0.D0
    coords(4:6)  =(/svar,0.D0,0.D0/)
    coords(7:9)  =(/svar*0.5D0, svar*0.5D0*dsqrt(3.D0), 0.D0/)
    coords(10:12)=(/svar*1.5D0, svar*0.5D0*dsqrt(3.D0), 0.D0/)

    coords(12)=1.D0

    ! Two points quite near to a TS:
    svar=1.1224620D0*0.9D0
!!$    coords(:)=0.D0
!!$    coords(4:6)  =(/svar,0.D0,0.D0/)
!!$    coords(7:9)  =(/svar*0.5D0, svar*0.5D0*dsqrt(3.D0), 0.D0/)
!!$    coords(10:12)=(/svar*1.5D0, svar*0.5D0*dsqrt(3.D0), 0.1D0/)
    coords2(:)=0.D0
    coords2(4:6)  =(/svar,0.D0,0.D0/)
    coords2(7:9)  =(/svar*0.5D0, svar*0.5D0*dsqrt(3.D0), 0.D0/)
    coords2(10:12)=(/svar*1.5D0, svar*0.5D0*dsqrt(3.D0), -0.3D0/)
    nframe=1

!!$    coords2(:)=0.D0
!!$    coords2(1:2)=(/-svar,svar*sqrt(3.D0)/)
!!$    coords2(7:8)=(/svar,svar*sqrt(3.D0)/)

!!$    ! four atoms in nearly a square
!!$    coords(:)=0.D0
!!$    coords(3)=0.d0
!!$    coords(4)=1.0D0
!!$    coords(8)=1.0D0
!!$    coords(10)=1.0D0
!!$    coords(11)=1.0D0
!!$    coords(12)=0.610D0

    ! masses
    nmass=4
    coords2(13:14)=1.D0
    coords2(15:16)=10.D0

  end if
  if(isystem==3) then
    ! one hopping atom on a 2*3 surface
!    svar=1.1224620D0
!    do iat=0,1
!      do jat=0,2

!        coords(3+iat*9+jat*3+1)=dble(jat)*svar
!        coords(3+iat*9+jat*3+2)=dble(iat)*svar
!        coords(3+iat*9+jat*3+3)=0.D0
!      end do
!    end do
    ! TS energy with 4 frozen atoms: -0.01156076358016981
    ! hopping atom (atom 1) 
!    coords(1)=svar*0.7D0
!    coords(2)=svar*0.5D0
!    coords(3)=svar*0.7D0
    !coords(9)=-0.3D0
    !coords(18)=-0.3D0

    ! stationary point with all atoms free:
!!$    coords(:) = (/&
!!$        1.1848983D0,   0.5065666D0,   0.5943077D0,&
!!$        0.2522962D0,   0.1414045D0,   0.0995254D0,&
!!$        1.2142701D0,   0.1259186D0,  -0.4664030D0,&
!!$        2.0938589D0,  -0.0116502D0,   0.2075193D0,&
!!$        0.2244601D0,   1.0502398D0,   0.7566075D0,&
!!$        0.7204213D0,   1.1023797D0,  -0.2424304D0,&
!!$        1.8302904D0,   1.0137577D0,  -0.1634031D0 /)

    nzero=0

   ! read coordinates of water

    coords=(/ -0.06756756,        -0.31531531,      0.00000000, &
               0.89243244,        -0.31531531,      0.00000000, &
              -0.38802215,         0.58962052,      0.00000000 /)

    nframe=0

    nmass=3
    coords2=(/ 15.999, 1.00794, 1.00794 /)

    spec(:)=0

!    nmass=
!    nframe=0
!    coords2(:)=1.D0
!    coords2(1:nvar)=coords(:)
    !coords2(nvar+1:2*nvar)=coords(:)
    ! hop atom other minimum
!    coords2(1)=coords2(1)+svar
    !coords2(9)=coords2(9)+0.1D0
    !coords2(3)=coords2(3)+0.2D0
    ! for dimer:
    !coords2(1)=coords2(1)+0.6D0*svar
    !coords2(3)=coords2(3)+0.2D0
    !coords2(nvar+1)=coords2(nvar+1)+svar
!    spec(:)=-1 ! NEB: -1
!    spec(1)=1
!    spec(3)=1
!    spec(6)=1
!    spec(:)=0 ! all free for now

    ! microiterative: last two are inner region
!    spec(13:14)=1

    ! weights
!    nweight=7
    !coords2(22:28) are weights
!    coords2(22:28)=1.0d0
!    coords2(24)=1.d0

    ! masses
    !coords2(29:35) are masses
!    coords2(29:35)=10.D0
!    coords2(29:29)=1.D0

  end if

  ! Eckart barrier
  if(isystem==4) then

    coords=0.D0
    coords2=0.D0

   ! coords(1)=-2.0D0

    nframe=1
    coords2(1:3)=(/ 1.D0 ,  0.D0,  0.D0 /)

    nzero=0
    nmass=1
    coords2(nvar2)=1.0078250321D0 ! mass: last entry in coords2, here using mass of protium
  !  coords2(nvar2)=2.013553212D0 ! mass: last entry in coords2, here using mass of deuterium
  !  coords2(nvar2)=3.D0
!    coords2(nvar2)=0.5D0/1.8228884842645E+03 ! mass: leads to a mass of 0.5 in atomic units

    spec(1)=-34
  end if

  ! Quartic potential
  if(isystem==5) then

    coords=0.D0
    coords2=0.D0

    ! Saddle point:
  !  coords(1)=2.D0/3.D0

    nframe=1
    coords2(1:3)=(/ 1.D0 ,  0.D0,  0.D0 /)

    nzero=0
    nmass=1
   ! coords2(nvar2)=1.0078250321D0 ! mass: last entry in coords2, here using mass of protium
    coords2(nvar2)=0.5D0/1.8228884842645E+03 ! mass: leads to a mass of 1 in atomic units
    spec(1)=-34

  end if

  ! Vibrational potential
  if(isystem==6) then

!N          0.0000000000        0.0000000000       -0.0000061916
!H          0.0000000000        0.0000000000        1.8816453413
!H          0.0000000000        1.6294938353       -0.9407796566
!H          0.0000000000       -1.6294938353       -0.9407796566

    ! reference geometry
    coords=(/ 0.0000000000D0,        0.0000000000D0,       -0.0000061916D0, &
              0.0000000000D0,        0.0000000000D0,        1.8816453413D0, &
              0.0000000000D0,        1.6294938353D0,       -0.9407796566D0, &
              0.0000000000D0,       -1.6294938353D0,       -0.9407796566D0 /)
 
!!$    ! distorted, but probably consistent with the reference:
!!$    coords=(/  0.0080835D0,   0.0000000D0,  -0.0000062D0,&
!!$        -0.5176180D0,   0.0000000D0,   1.8816453D0,&
!!$        -0.5181795D0,   1.6294938D0,  -0.9407797D0,&
!!$        -0.5181795D0,  -1.6294938D0,  -0.9407797D0 /)

!!$    !minimum geometry - 1D
!!$    coords=(/  -0.0934927D0,   0.0000000D0,  -0.0000000D0, &
!!$        0.4308739D0,  -0.0000000D0,   1.8815559D0,&
!!$        0.4313413D0,   1.6294751D0,  -0.9407780D0,&
!!$        0.4313413D0,  -1.6294751D0,  -0.9407780D0 /)


    

    nframe=1
    coords2(1:nvar)=coords

    nzero=6
    nmass=4
!    coords2(nvar+1)=14.0030740048D0
    !OH3+
    coords2(nvar+1)=15.99491461956D0
    coords2(nvar+2:nvar2)=1.0078250321D0
    spec(1:4)=0 ! all atoms active
    nz=4
    spec(5)=7 ! Nuclear charge
    spec(6:8)=1 ! Nuclear charge

    printl=4
    ! read in potential data
    !call read_pot_init(nvar/3)

  end if

  if(isystem==7) then

    nzero=0
    coords=(/ -1.D0,   0.D0, 0.D0 /)! main saddle point
!    coords=(/ -1.D0,   0.01D0, 0.D0 /)!  main saddle point distorted startvalue

!   coords=(/ -2.D0, 0.D0 , 0.D0 /)! left minimum
!    coords=(/ -2.D0, 0.2D0 , 0.D0 /)! left minimum distorted startvalue
!    coords=(/ 1.D0, 0.D0, 0.D0 /)! second saddle point
!    coords=(/ 1.D0, 1.D0 , 0.D0 /)! right minimum, perpendicular to second saddle point
!!$    coords=(/ 1.D0, 0.3670625D0, 0.D0 /)! right minimum, perpendicular to second saddle point

   ! Main TS and direction
    nframe=1
    coords2(1:3)=(/ -0.8D0 ,  0.1D0,  0.D0 /)

!More than one frame
!!$    coords=(/ -2.D0, 0.D0 , 0.D0 /)! left minimum
!!$    nframe=19
!!$    coords2(1:nframe*3)=(/ &
!!$         -1.7D0, 0.D0 , 0.D0, &
!!$         -1.4D0, 0.D0 , 0.D0, &
!!$         -1.2D0, 0.D0 , 0.D0, &
!!$         -1.0D0, 0.D0 , 0.D0, & ! main saddle point
!!$         -0.8D0, 0.D0 , 0.D0, &       
!!$         -0.6D0, 0.D0 , 0.D0, &
!!$         -0.4D0, 0.D0 , 0.D0, &
!!$         -0.2D0, 0.D0 , 0.D0, &
!!$          0.0D0, 0.D0 , 0.D0, &
!!$          0.2D0, 0.D0 , 0.D0, &
!!$          0.4D0, 0.D0 , 0.D0, &
!!$          0.6D0, 0.D0 , 0.D0, &
!!$          0.8D0, 0.D0 , 0.D0, &
!!$          1.D0, 0.0D0 , 0.D0, & ! second saddle point
!!$          1.D0, 0.2D0 , 0.D0, &
!!$          1.D0, 0.4D0 , 0.D0, &
!!$          1.D0, 0.6D0 , 0.D0, &
!!$          1.D0, 0.8D0 , 0.D0, &
!!$          1.D0, 1.0D0 , 0.D0  &
!!$        /)

!More than one frame
!!$    coords=(/ -2.D0, 0.D0 , 0.D0 /)! left minimum
!!$    nframe=10
!!$    coords2(1:nframe*3)=(/ &
!!$        -1.4D0, 0.D0 , 0.D0, &
!!$         -1.2D0, 0.D0 , 0.D0, &
!!$         -1.0D0, 0.D0 , 0.D0, & ! main saddle point
!!$         -0.8D0, 0.D0 , 0.D0, &       
!!$         -0.6D0, 0.D0 , 0.D0, &
!!$         -0.4D0, 0.2D0 , 0.D0, &
!!$         -0.2D0, 0.D0 , 0.D0, &
!!$          0.0D0, 0.D0 , 0.D0, &
!!$          0.2D0, 0.D0 , 0.D0, &
!!$          0.2D0, 0.2D0 , 0.D0 &
!!$          /)


    nmass=1
    coords2(nvar2)=1.0078250321D0 ! mass: last entry in coords2, here using mass of protium
!    coords2(nvar2)=10.00 ! mass: last entry in coords2, here using mass of more 


    print*,"nspec",nspec
    spec(1)=-4
  end if

  ! polymul
  if(isystem==8) then

    coords=0.D0
    coords2=0.D0
    coords(1)=1.D0
!    call random_number(coords)

  !  coords(1)=-2.0D0

    nframe=1
    !coords2(1:num_dim)=(/ 1.D0 ,  0.D0,  0.D0 /)

    nzero=0
    nmass=num_dim/3
    coords2(num_dim/3+1:num_dim+num_dim/3)=1.0078250321D0 ! mass: last entry in coords2, here using mass of protium
  !  coords2(nvar2)=2.013553212D0 ! mass: last entry in coords2, here using mass of deuterium
  !  coords2(nvar2)=3.D0

    !spec(1)=-34
    spec(:)=0

  end if


  !quick_opt
  if(isystem==9) then
    
    nzero=0
    
   ! read coordinates of water

    coords=(/ 0.0, 0.0, 0.0, &
              1.88972613, 0.0, 0.0, &
              0.0, 1.88972613, 0.0 /)

!    coords=(/ -1.401288613,        3.311045835,      5.473175981, &
!              -2.0997710735,        1.8514950758,    6.334909992, &
!              -1.7394929,        3.848332767,        3.749292221 /)

    
    do iat=0,6,3
      print*, (coords(iat+jat), jat=1, 3)
    end do

    nframe=0

    nmass=3
    coords2=(/ 15.999, 1.00794, 1.00794 /)

    spec(:)=0
!    spec((nvar/3)+1:(nvar/3)+nzero)=(/ 8, 1, 1 /) 
    nz=3
    spec(4)=8
    spec(5:6)=1
  end if

!*************END case of isystem checking******************

  tolerance=4.5D-5 ! negative: default settings
  printl=4
  printf=4
  maxcycle=100 !200
  maxene=100000

  tolrot=1.D2
!  tolrot=0.1D0
!  maxrot=100 !was 100

  task=0 !1011

  distort=0.D0 !0.4 
  tatoms=0
  icoord=0 !0 cartesian coord !210 Dimer !190 qts search !120 NEB frozen endpoint
  massweight=0

! TO DO (urgent): better dtau for endpoints (interpolate energy) when reading dist

  imicroiter=0 ! means: use microiterative optimization

  iopt=3 ! 20 !3 or 20 later change to 12
  temperature = 2206.66844626D0*0.99D0 ! K ! T_c for the MB-potential for hydrogen is 2206.668 K
  !temperature = 0.8D0*333.40829D0  ! K ! T_c 20102.83815 K
  !temperature = 500.D0
  ! Cubic potential (isystem=5: Crossover temperature for tunnelling  1846563.59447 K)
  temperature = 0.99D0 * 1846563.59447D0
  ! eckart potential (scatter) Tc: 3186172.17493753 K
  temperature = 0.9D0*3186172.17493753D0
  temperature= 0.7D0 * 275.13301D0 
! T_c for 162 DOF: 511.06618

  iline=0
  maxstep=0.1D0
  scalestep=1.0D0
  lbfgs_mem=100
  nimage=39 !*k-k+1
  nebk=1.0D0 ! for QTS calculations with variable tau, nebk transports the parameter alpha (0=equidist. in tau)
  qtsflag=0 ! 1: Tunneling splittings, 11/10: read image hessians

  !"accurate" etunnel for 0.4 Tc: 0.3117823831234522

  ! Hessian
  delta=1.D-2
  soft=-6.D-4
  update=2
  maxupd=0
  inithessian=0
  minstep=1.D0 **2 ! 1.D-5
  minstep=1.D-5

!!$  ! variables for exchange by sed
!!$  iopt=IOPT_VAR
!!$  nimage=NIMAGE_VAR
!!$  temperature = TFAC ! K ! T_c for the MB-potential for hydrogen is 2206.668 K
!!$  nebk= NEBK_VAR


  ! damped dynamics
  fric0=0.1D0
  fricfac=1.0D0
  fricp=0.1D0

  dump=0
  restart=0

  ! Parallel optimization

  po_pop_size=25
  po_radius=0.5D0
  po_init_pop_size=50
  po_contraction=0.95D0
  po_tolerance_r=1.0D-8
  po_tolerance_g=1.0D-6
  po_distribution=3
  po_maxcycle=100000
  po_reset=500
  po_mutation_rate=0.15D0
  po_death_rate=0.5D0
  po_scalefac=10.0D0
  po_nsave=10
  n_po_scaling=0 ! meaning that the base radii values for the sampling and tolerance 
                 ! are used for all components of the working coordinate vector.
                 ! Remember to change the second arg in the call to dl_find if a 
                 ! non-zero value for n_po_scaling is desired, and also to add the 
                 ! necessary values to the coords2 array...

  ! Taskfarming 
  ! (the two lines below, with any values assigned, 
  ! may be safely left in place for a serial build) 
  ntasks = 1
  tdlf_farm = 1

  tatoms=1
  
 ! call test_ene

end subroutine dlf_get_params

subroutine test_update
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  use dlf_global, only: glob,printl
  implicit none
  integer(4) :: varperimage,nimage,iimage,ivar4
  real(rk), allocatable:: coords(:,:),grad(:,:) ! varperimage,nimage
  real(rk), allocatable:: hess(:,:,:) ! varperimage,varperimage,nimage
  real(rk), allocatable:: fhess(:,:) ! 2*varperimage*nimage,2*varperimage*nimage
  real(rk), allocatable:: eigval(:),eigvec(:,:)
  real(rk), allocatable:: vec0(:)
  real(rk), allocatable:: determinant(:)
  real(rk), allocatable:: tmphess(:,:)
  real(rk) :: svar
  integer :: ivar,jvar,vp8,target_image,step,lastimage,jimage,turnimg
  logical :: havehessian,fracrecalc

  open(unit=102,file="grad_coor.bin",form="unformatted")
  read(102) varperimage,nimage
  print*,"varperimage,nimage",varperimage,nimage
  vp8=varperimage

!!$  call allocate(coords,int(varperimage,kind=8),int(nimage,kind=8))
!!$  call allocate(grad,int(varperimage,kind=8),int(nimage,kind=8))
!!$  call allocate(determinant,int(nimage,kind=8))

  do iimage=1,nimage
    read(102) ivar4
    print*,"Reading coords/grad of image",ivar4
    read(102) grad(:,iimage)
    read(102) coords(:,iimage)
  end do
  close(102)
  print*,"Coords sucessfully read"

  ! print coords and grad
  iimage=2
  print*,"Coords and grad for image ",iimage
  do ivar=1,vp8
    write(6,"(i6,1x,2es18.9)") &
        ivar,coords(ivar,iimage),grad(ivar,iimage)
  end do

  open(unit=101,file="hessian.bin",form="unformatted")
  ! Hessian in mass-weighted coordinates on the diagnal blocks - everything else should be zero
  read(101) iimage,ivar4!neb%varperimage,neb%nimage
  if(iimage/=varperimage.or.ivar4/=nimage) then
    print*,"Dimensions read",iimage,ivar4
    call dlf_fail("ERROR: wrong dimensions in hessian.bin!")
  end if
  ivar=2*varperimage*nimage
  print*,"File Hessian size",ivar
  call allocate(fhess,ivar,ivar)
  read(101) fhess
  close(101)
  print*,"Hessian sucessfully read"

  ! map hessian to different array:
!!$  call allocate(hess,int(varperimage,kind=8),int(varperimage,kind=8),int(nimage,kind=8))
  do iimage=1,nimage
    print*,"Image",iimage,"Hessian positions",(iimage-1)*varperimage+1,iimage*varperimage
    hess(:,:,iimage)=fhess((iimage-1)*varperimage+1:iimage*varperimage,(iimage-1)*varperimage+1:iimage*varperimage)
  end do
  call deallocate(fhess)

  call allocate(eigval,vp8)
  call allocate(eigvec,vp8,vp8)
  ! now we have all we need

  print*,"# Distance from previous image"
  do iimage=2,nimage
    print*,iimage,sqrt(sum( (coords(:,iimage)-coords(:,iimage-1))**2))
  end do

  print*,"# Determinant of Hessian"
  do iimage=1,nimage
    do ivar=1,vp8
      do jvar=ivar+1,vp8
        if(abs(hess(ivar,jvar,iimage)-hess(jvar,ivar,iimage))>1.D-20) &
            print*,"Unsymmetric:",ivar,jvar,iimage,hess(ivar,jvar,iimage),hess(jvar,ivar,iimage)
      end do
    end do
    call dlf_matrix_diagonalise(vp8,hess(:,:,iimage),eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do
    determinant(iimage)=product(eigval)
    write(6,"(i6,1x,9es12.3)") iimage,product(eigval),eigval(1:8)
  end do
  
  print*,"maxval(hess(:,:,nimage-1)-hess(:,:,nimage))",maxval(hess(:,:,nimage-1)-hess(:,:,nimage))

!!$  ! Richtungsableitung
!!$  call allocate(vec0,vp8)!int(varperimage,kind=8))
!!$  do iimage=2,nimage-1
!!$    ! eigval is Vector along which the derivative is taken
!!$    eigval=coords(:,iimage+1)-coords(:,iimage-1)
!!$    svar=sqrt(sum(eigval**2))
!!$    eigval=eigval/svar
!!$    vec0=matmul(hess(:,:,iimage),eigval)
!!$    do ivar=1,vp8
!!$      write(6,"(2i6,1x,2es18.9,1x,f10.5)") &
!!$          iimage,ivar,(grad(ivar,iimage+1)-grad(ivar,iimage-1))/svar,vec0(ivar),&
!!$          vec0(ivar)/((grad(ivar,iimage+1)-grad(ivar,iimage-1))/svar)
!!$    end do
!!$  end do

  !
  ! now test updates
  !
  call allocate(tmphess,vp8,vp8)
  havehessian=.true.
  fracrecalc=.false.
  printl=2
  glob%maxupd=30000

  target_image=1 !nimage
  ! update hessians to the one of the first image
  print*,"Updating Hessians to that of image",target_image
  print*,"Sum-of-squares difference"
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    call dlf_hessian_update(vp8, &
        coords(:,target_image),coords(:,iimage),&
        grad(:,target_image),grad(:,iimage), &
        tmphess, havehessian, fracrecalc)
    if(.not.havehessian) then
      print*,"Problem with hessian update, image",iimage
      havehessian=.true.
    end if
    print*,iimage,sum( (tmphess-hess(:,:,target_image))**2),&
        sum( (hess(:,:,iimage)-hess(:,:,target_image))**2)
  end do

  print*,"Minstep",glob%minstep
  print*,"Updating Hessians to that of image",target_image
  print*,"Determinant"
  open(file="determinant",unit=10)
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    call dlf_hessian_update(vp8, &
        coords(:,target_image),coords(:,iimage),&
        grad(:,target_image),grad(:,iimage), &
        tmphess, havehessian, fracrecalc)
    if(.not.havehessian) then
      print*,"Problem with hessian update, image",iimage
      havehessian=.true.
    end if
    call dlf_matrix_diagonalise(vp8,tmphess,eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do

    print*,iimage,product(eigval),determinant(iimage),determinant(target_image)
    write(10,*) iimage,product(eigval),determinant(iimage),determinant(target_image)
  end do
  close(10)

  print*,"Updating Hessians to that of image",target_image
  print*,"Determinant - incremental"
  open(file="determinant_incr",unit=10)
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    step=1
    if(iimage>target_image) step=-1
    lastimage=iimage
    do jimage=iimage+step,target_image,step
      !print*,"updating",lastimage," to ",jimage
      call dlf_hessian_update(vp8, &
          coords(:,jimage),coords(:,lastimage),&
          grad(:,jimage),grad(:,lastimage), &
          tmphess, havehessian, fracrecalc)

      if(.not.havehessian) then
        print*,"Problem with hessian update, image",iimage
        havehessian=.true.
      end if
      lastimage=jimage
    end do
    call dlf_matrix_diagonalise(vp8,tmphess,eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do

    print*,iimage,product(eigval),determinant(iimage),determinant(target_image)
    write(10,*) iimage,product(eigval),determinant(iimage),determinant(target_image)
  end do
  close(10)

  print*,"Updating Hessians to that of image",target_image
  print*,"Determinant - incremental turning around"
  turnimg=20
  open(file="determinant_turn",unit=10)
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    lastimage=iimage
    ! first upwards to turnimg
    do jimage=iimage+1,turnimg
      print*,"updating",lastimage," to ",jimage
      call dlf_hessian_update(vp8, &
          coords(:,jimage),coords(:,lastimage),&
          grad(:,jimage),grad(:,lastimage), &
          tmphess, havehessian, fracrecalc)

      if(.not.havehessian) then
        print*,"Problem with hessian update, image",iimage
        havehessian=.true.
      end if
      lastimage=jimage
    end do
    step=1
    if(lastimage>target_image) step=-1
    do jimage=lastimage+step,target_image,step
      print*,"updating",lastimage," to ",jimage
      call dlf_hessian_update(vp8, &
          coords(:,jimage),coords(:,lastimage),&
          grad(:,jimage),grad(:,lastimage), &
          tmphess, havehessian, fracrecalc)

      if(.not.havehessian) then
        print*,"Problem with hessian update, image",iimage
        havehessian=.true.
      end if
      lastimage=jimage
    end do
    call dlf_matrix_diagonalise(vp8,tmphess,eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do

    print*,iimage,product(eigval),determinant(iimage),determinant(target_image)
    write(10,*) iimage,product(eigval),determinant(iimage),determinant(target_image)
  end do
  close(10)

!!$  do ivar=1,vp8
!!$    WRITE(*,'(33f10.5)') hess(ivar,:,1)*1.D6
!!$  end do
!!$
!!$  do ivar=1,vp8
!!$    write(6,"(i6,1x,es18.9)") &
!!$            ivar,eigval(ivar)
!!$  end do

  call deallocate(coords)
  call deallocate(grad)
  
  call dlf_fail("stop in test_update")
end subroutine test_update

! just a test routine
subroutine test_ene
  use dlf_parameter_module, only: rk
  implicit none
  integer :: ivar,ivar2,status
  integer :: samples
  real(rk) :: halfSamples
  real(rk) :: coords(3),grad(3),hess(3,3),ene
  coords(:)=0.D0
!  open(file="energy",unit=13)
  open(file="energy-2d.dat",unit=13)
  samples = 100
  halfSamples = dble(samples) * 0.25D0
  do ivar2=1,samples
    do ivar=1,samples
      coords(1)=dble(ivar-samples/2)/halfSamples
      coords(2)=dble(ivar2-samples/2)/halfSamples
      call dlf_get_gradient(3,coords,ene,grad,1,-1,status)
      call dlf_get_hessian(3,coords,hess,status)
     write(13,*) coords(1),coords(2),ene,grad(1),grad(2),hess(1,1),hess(2,2),hess(2,1)
!     write(13,*) coords(1),coords(2),ene,grad(1)
    end do
    write(13,*) ""
  end do
  close(13)
  call dlf_fail("stop in test_ene")
end subroutine test_ene

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,kiter,status)
  !  Mueller-Brown Potential
  !  see K Mueller and L. D. Brown, Theor. Chem. Acta 53, 75 (1979)
  !  taken from JCP 111, 9475 (1999)
  use dlf_parameter_module, only: rk
  use driver_module
  use driver_parameter_module
  !use vib_pot
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: kiter
  integer   ,intent(out)   :: status
  !
  ! variables for Mueller-Brown potential
  real(rk) :: x,y,svar,svar2
  integer  :: icount
  ! variables for Lennard-Jones potentials
  real(rk) :: acoords(3,nvar/3)
  real(rk) :: agrad(3,nvar/3)
  real(rk) :: r
  integer  :: nat,iat,jat,scrt
  ! additional variables non-cont. diff MEP potential
  real(rk) :: t ! variation of the depth of the flater saddle point

! **********************************************************************
!  call test_update
  status=1
  select case (isystem)
  case (1)
    !print*,"coords in energy eval",coords
    x =  coords(1)
    y =  coords(2)

    energy=0.D0
    gradient=0.D0
    do icount=1,4
      svar= apar(icount)*(x-x0par(icount))**2 + &
          bpar(icount)*(x-x0par(icount))*(y-y0par(icount)) + &
          cpar(icount)*(y-y0par(icount))**2 
      svar2= acappar(icount) * dexp(svar)
      energy=energy+ svar2
      gradient(1)=gradient(1) + svar2 * &
          (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))
      gradient(2)=gradient(2) + svar2 * &
          (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))
    end do
    energy=energy+0.692D0
    ! write(*,'("x,y,func",2f10.5,es15.7)') x,y,energy
  case (2,3)

    ! one could use a Lennard-Jones particle with slightly different
    ! parameters as "Excited state": epsilon=0.9D-3, sigma=1.1D0 to search for
    ! conical intersections

    acoords=reshape(coords,(/3,nvar/3/))
    energy=0.D0
    agrad(:,:)=0.D0
    do iat=1,nvar/3
      do jat=iat+1,nvar/3
        r=sum((acoords(:,iat)-acoords(:,jat))**2)
        ! Lennard-Jones Potential
        energy=energy+4.D0*epsilon * ((sigma**2/r)**6-(sigma**2/r)**3)
        svar = -4.D0*epsilon * (12.D0*sigma**12/r**7-6.D0*sigma**6/r**4)
        agrad(1,iat)=agrad(1,iat)+ svar * (acoords(1,iat)-acoords(1,jat))
        agrad(2,iat)=agrad(2,iat)+ svar * (acoords(2,iat)-acoords(2,jat))
        agrad(3,iat)=agrad(3,iat)+ svar * (acoords(3,iat)-acoords(3,jat))
        agrad(1,jat)=agrad(1,jat)- svar * (acoords(1,iat)-acoords(1,jat))
        agrad(2,jat)=agrad(2,jat)- svar * (acoords(2,iat)-acoords(2,jat))
        agrad(3,jat)=agrad(3,jat)- svar * (acoords(3,iat)-acoords(3,jat))
      end do
    end do
    gradient=reshape(agrad,(/nvar/))
    print*, energy


  case(4)

    energy=0.D0
    gradient=0.D0
    xvar= coords(1)
    yvar= (Vb+Va)/(Vb-Va)*exp(alpha*xvar)
    
    energy=Va*yvar/(1.D0+yvar)+Vb*yvar/(1.D0+yvar)**2
    svar=1.D0+yvar
    gradient(1)=alpha*yvar * ( &
        Va/svar + (Vb-Va*yvar)/svar**2 - 2.D0*Vb*yvar/svar**3)
    
  case(5)
!!$    !assign parameters for Eckart potential (see Andri's thesis) ???
!!$    xvar= coords(1)
!!$    energy=V0*(xvar**2/x0**2-1.D0)**2
!!$    gradient=0.D0
!!$    gradient(1)= V0 * 4.D0 / x0**2 * (xvar**3/x0**2-xvar)
    
    ! 3rd order polynomial as also used in the 1D-scattering code
    xvar=coords(1)*1.5D0
    energy=50.D0*(-2.D0*xvar**3+3.D0*xvar**2)
    gradient=0.D0
    gradient(1)= 50.D0*(-6.D0*xvar**2+6.D0*xvar) * 1.5D0

    
  case(6)
    
    !Vibrational potential
    !call vib_get_gradient(coords,energy,gradient)
    call dlf_fail("Vibrational calculations not linked")

  case(7)
    !print*,"coords in energy eval",coords
    x =  coords(1)
    y =  coords(2)
    t =  smallbarvar 
    
    energy=0.D0
    gradient=0.D0
    
    energy = 10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)+(dexp(-2.D0*(y+1.D0)**2)))- &
        3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2)) - 1.6D0*dexp(-2.D0*(x-1.D0)**2)* &
        (dexp(-2.D0*(y+t)**2)+dexp(-2.D0*(y-t)**2))
    gradient(1)= -4.D0*(10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)+ & 
        (dexp(-2.D0*(y+1.D0)**2)))*(x+1.D0)-3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(x+2.D0)- &
        1.6D0*dexp(-2.D0*(x-1.D0)**2)*(dexp(-2.D0*(y+t)**2)+ &
        dexp(-2.D0*(y-t)**2))*(x-1.D0))
    gradient(2)= -4.D0*(10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)* &
        (y-1.D0)+(dexp(-2.D0*(y+1.D0)**2)*(y+1.D0)))-3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*y- &
        1.6D0*dexp(-2*(x-1)**2)*(dexp(-2*(y+t)**2)*(y+t)+ &
        dexp(-2.D0*(y-t)**2)*(y-t)))
    
  case(8)
    ! polymul (Polynomial in multiple dimensions)
    ! E=E1+E2+...+En
    ! E1= (-2Ebarr) x1**3 + (3Ebarr) x1**2
    ! Ei=xi**2 * 1/2 * d_i (1-x1/5)  | i>1
    gradient=0.D0
    energy= -2.D0 * Ebarr * coords(1)**3 + 3.D0 * Ebarr * coords(1)**2
    gradient(1)=-6.D0 * Ebarr * coords(1)**2 + 6.D0 * Ebarr * coords(1)
    do icount=2,num_dim
!!$      ! finite vdamp
!!$      energy=energy+coords(icount)**2*0.5D0 * dpar(icount) * (1.D0-coords(1)/vdamp)
!!$      gradient(1)=gradient(1) -0.5D0*dpar(icount)*coords(icount)**2/vdamp
!!$      gradient(icount)=coords(icount) * dpar(icount) * (1.D0-coords(1)/vdamp)
      ! vdamp=infinity
      energy=energy+coords(icount)**2*0.5D0 * dpar(icount) 
      gradient(icount)=coords(icount) * dpar(icount)
    end do

  case(9)

    read*,scrt
     
    ! Read gradients from Quick gradient calculation for water
    gradient=0.D0
    energy=0.D0
    
    open(unit=12, file="water_grad.in")
    read(12,*)
    read(12,*)energy
!    print*, "The energy is:", energy
    read(12,*)
    print*, "The gradients are:"
    do iat=1, nvar
       read(12,*) gradient(iat)
       print*, gradient(iat)
    end do
    
  end select
  status=0
end subroutine dlf_get_gradient

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_hessian(nvar,coords,hessian,status)
  !  get the hessian at a given geometry
  use dlf_parameter_module
  use driver_module
  use driver_parameter_module
  !use vib_pot
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
  integer   ,intent(out)   :: status
  real(rk) :: acoords(3,nvar/3),r,svar,svar2
  integer  :: posi,posj,iat,jat,m,n
  ! variables for Mueller-Brown potential
  real(rk) :: x,y
  integer  :: icount
  ! variables non-cont. diff. potential
  real(rk) :: t
! **********************************************************************
  hessian(:,:)=0.D0
  status=1
  select case (isystem)
  case(1)

    x =  coords(1)
    y =  coords(2)

    hessian=0.D0
    do icount=1,4
      svar= apar(icount)*(x-x0par(icount))**2 + &
          bpar(icount)*(x-x0par(icount))*(y-y0par(icount)) + &
          cpar(icount)*(y-y0par(icount))**2 
      svar2= acappar(icount) * dexp(svar)
      !energy=energy+ svar2
!!$      gradient(1)=gradient(1) + svar2 * &
!!$          (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))
!!$      gradient(2)=gradient(2) + svar2 * &
!!$          (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))
      hessian(1,1)=hessian(1,1)+svar2 * &
          (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))**2 + &
          svar2 * 2.D0 * apar(icount)
      hessian(2,2)=hessian(2,2)+svar2 * &
          (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))**2 + &
          svar2 * 2.D0 * cpar(icount)
      hessian(1,2)=hessian(1,2) + svar2 * &
          (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))* &
          (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount))) + &
          svar2 * bpar(icount)
    end do
    hessian(2,1)=hessian(1,2)
    status=0

  case(2,3)
    acoords=reshape(coords,(/3,nvar/3/))
    do iat=1,nvar/3
      do jat=iat+1,nvar/3
        r=sum((acoords(:,iat)-acoords(:,jat))**2)
        ! Lennard-Jones Potential
        svar = 96.D0*epsilon * (7.D0*sigma**12/r**8-2.D0*sigma**6/r**5) ! coeff of x1x2
        svar2= epsilon * (-2.D0*sigma**12/r**7+sigma**6/r**4) ! for x1x1
        posi=(iat-1)*3+1
        posj=(jat-1)*3+1
        ! off-diag
        hessian(posi,posi+1)  =hessian(posi,posi+1)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(2,iat)-acoords(2,jat))
        hessian(posi,posi+2)  =hessian(posi,posi+2)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posi+1,posi)  =hessian(posi+1,posi)  + svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posi+1,posi+2)=hessian(posi+1,posi+2)+ svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posi+2,posi)  =hessian(posi+2,posi)  + svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posi+2,posi+1)=hessian(posi+2,posi+1)+ svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(2,iat)-acoords(2,jat))

        do m=0,2
          do n=0,2
            if(m==n) cycle
  hessian(posi+m,posj+n)=hessian(posi+m,posj+n)- svar * (acoords(M+1,iat)-acoords(M+1,jat)) * (acoords(N+1,iat)-acoords(N+1,jat))
  hessian(posj+m,posi+n)=hessian(posj+m,posi+n)- svar * (acoords(M+1,iat)-acoords(M+1,jat)) * (acoords(N+1,iat)-acoords(N+1,jat))
          end do
        end do
        ! Diag for different atoms ...
        do m=0,2
          hessian(posi+m,posj+m)=hessian(posi+m,posj+m) - 24.D0*(svar2+1.D0/24.D0*svar* (acoords(m+1,iat)-acoords(M+1,jat))**2)
          hessian(posj+m,posi+m)=hessian(posj+m,posi+m) - 24.D0*(svar2+1.D0/24.D0*svar* (acoords(m+1,iat)-acoords(M+1,jat))**2)
        end do

        hessian(posj,posj+1)  =hessian(posj,posj+1)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(2,iat)-acoords(2,jat))
        hessian(posj,posj+2)  =hessian(posj,posj+2)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posj+1,posj)  =hessian(posj+1,posj)  + svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posj+1,posj+2)=hessian(posj+1,posj+2)+ svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posj+2,posj)  =hessian(posj+2,posj)  + svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posj+2,posj+1)=hessian(posj+2,posj+1)+ svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(2,iat)-acoords(2,jat))
        ! diag
        hessian(posi,posi)    =hessian(posi,posi)    + 24.D0*(svar2+1.D0/24.D0*svar* (acoords(1,iat)-acoords(1,jat))**2)
        hessian(posi+1,posi+1)=hessian(posi+1,posi+1)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(2,iat)-acoords(2,jat))**2)
        hessian(posi+2,posi+2)=hessian(posi+2,posi+2)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(3,iat)-acoords(3,jat))**2)

        hessian(posj,posj)    =hessian(posj,posj)    + 24.D0*(svar2+1.D0/24.D0*svar* (acoords(1,iat)-acoords(1,jat))**2)
        hessian(posj+1,posj+1)=hessian(posj+1,posj+1)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(2,iat)-acoords(2,jat))**2)
        hessian(posj+2,posj+2)=hessian(posj+2,posj+2)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(3,iat)-acoords(3,jat))**2)
      end do
    end do
    status=0
    status=1 ! 1= not possible
  case(4)
    ! Eckart
    xvar= coords(1)
    yvar= (Vb+Va)/(Vb-Va)*exp(alpha*xvar)
    svar=1.D0+yvar
    hessian(1,1)=alpha**2*yvar * ( Va/svar + (Vb-Va*yvar)/svar**2 - 2.D0*Vb*yvar/svar**3) + &
        alpha**2 * yvar**2 / svar**2 * ( -2.D0*Va + (-4.D0*Vb+2.D0*Va*yvar)/svar +6.D0*Vb*yvar/svar**2)
!!$    ! debug output - test of hessian-at-maximum-equations.
!!$    print*,"Hessian calculated, 1,1:",hessian(1,1)+alpha**2*vb/8.D0,hessian(1,1),-alpha**2*vb/8.D0
!!$    print*,"Hessian short           ",hessian(1,1)+alpha**2*(Va+Vb)**2*(Vb-Va)/8.D0/Vb**3    ,&
!!$        hessian(1,1),-alpha**2*(Va+Vb)**3*(Vb-Va)/8.D0/Vb**3
!!$    print*,va,hessian(1,1)+0.08085432314993411D0,"leading"
!!$    print*,"should be zero",Va/svar + (Vb-Va*yvar)/svar**2 - 2.D0*Vb*yvar/svar**3
    status=0 ! 0=OK 1=not available 
  case(5)
!!$    ! Quartic
!!$    xvar= coords(1)
!!$    hessian(1,1)=V0 * 4.D0 / x0**2 * (3.D0*xvar**2/x0**2-1.D0)
!!$    status=0 
    
    ! 3rd order polynomial as also used in the 1D-scattering code
    xvar=coords(1)*1.5D0
    !gradient(1)= 50.D0*(-6.D0*xvar**2+6.D0*xvar) * 1.5D0
    hessian(1,1)=50.D0*(-12.D0*xvar+6.D0) * 1.5D0**2
    status=0 

  case(6)
    !Vibrational potential
    !call vib_get_hessian(coords,hessian)
    !status=0 
  case(7)

    x =  coords(1)
    y =  coords(2)
    t =  smallbarvar 
    hessian=0.D0


!!$    energy = 10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)+(dexp(-2.D0*(y+1.D0)**2)))- &
!!$       dexp(-2.D0*(y**2+(x+2.D0)**2)) - 1.6D0*dexp(-2.D0*(x-1.D0)**2)* &
!!$       (dexp(-2.D0*(y+t)**2)+dexp(-2.D0*(y-t)**2))

    hessian(1,1)=-4.D0*(10.D0*(dexp(-2.D0*(y-1.D0)**2)+dexp(-2.D0*(y+1.D0)**2))*(dexp(-2.D0*(x+1.D0)**2)*&
        (-4.D0*(x+1.D0)**2+1.D0))-3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(-4.D0*(x+2.D0)**2+1.D0)-1.6D0* &
        (dexp(-2.D0*(x-1.D0)**2)*(-4.D0*(x-1.D0)**2+1.D0))*(dexp(-2.D0*(y+t)**2)+dexp(-2.D0*(y-t)**2)))
    
    hessian(2,2)=-4.D0*(10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)*(-4.D0*(y-1.D0)**2+1.D0))- &
        3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(-4.D0*y**2+1.D0)-1.6D0*dexp(-2.D0*(x-1.D0)**2)* &
        (dexp(-2.D0*(y+t)**2)*(-4.D0*(y+t)**2+1.D0)+dexp(-2.D0*(y-t)**2)*(-4.D0*(y-t)**2+1.D0)))
    
    hessian(1,2)= 8.D0*(10.D0*dexp(-2.D0*(x+1.D0)**2)*(x+1.D0)*(dexp(-2.D0*(y-1.D0)**2)*(y-1.D0)+ &
        (dexp(-2.D0*(y+1.D0)**2)*(y+1.D0)))-3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(x+2.D0)*y- &
        1.6D0*dexp(-2.D0*(x-1.D0)**2)*(dexp(-2.D0*(y+t)**2)*(y+t)+dexp(-2.D0*(y-t)**2)*(y-t))*(x-1.D0))

    hessian(2,1)=hessian(1,2)

    status=0

  case(8)
    ! polymul (Polynomial in multiple dimensions)
    ! E=E1+E2+...+En
    ! E1= (-2Ebarr) x1**3 + (3Ebarr) x1**2
    ! Ei=xi**2 * 1/2 * d_i (1-x1/5)  | i>1
    hessian=0.D0
    hessian(1,1)=-12.D0 * Ebarr * coords(1) + 6.D0 * Ebarr
!!$    ! finite vdamp
!!$    do iat=2,num_dim
!!$      hessian(1,iat)=-2.D0/vdamp * coords(iat)
!!$      hessian(iat,1)=-2.D0/vdamp * coords(iat)
!!$    end do
    do iat=2,num_dim
!!$      ! finite vdamp
!!$      hessian(iat,iat)=dpar(iat) * (1.D0-coords(1)/vdamp)
      ! vdamp = infinity
      hessian(iat,iat)=dpar(iat)
    end do
    
    status=0

  end select

end subroutine dlf_get_hessian

! initialize parameters for the test potentials
subroutine driver_init
  use driver_parameter_module
  implicit none
  real(rk) :: ebarr_
  integer :: icount
  ! assign parameters for MB potential
  ebarr_=0.5D0
  acappar(1)=-200.D0*ebarr_/106.D0
  acappar(2)=-100.D0*ebarr_/106.D0
  acappar(3)=-170.D0*ebarr_/106.D0
  acappar(4)=  15.D0*ebarr_/106.D0
  apar(1)=-1.D0
  apar(2)=-1.D0
  apar(3)=-6.5D0
  apar(4)=0.7D0
  bpar(1)=0.D0
  bpar(2)=0.D0
  bpar(3)=11.D0
  bpar(4)=0.6D0
  cpar(1)=-10.D0
  cpar(2)=-10.D0
  cpar(3)=-6.5D0
  cpar(4)=0.7D0
  x0par(1)=1.D0
  x0par(2)=0.D0
  x0par(3)=-0.5D0
  x0par(4)=-1.D0
  y0par(1)=0.D0
  y0par(2)=0.5D0
  y0par(3)=1.5D0
  y0par(4)=1.D0
  ! parameters for polymul
  dpar(1)=0.D0
  do icount=2,num_dim
    dpar(icount)=1.D-4+(dble(icount-2)/dble(num_dim-2))**2*0.415D0
  end do
end subroutine driver_init

! classical flux of the Eckart potential 
! returns log(flux)
subroutine driver_clrate(beta_hbar,flux)
  use driver_parameter_module
  implicit none
  real(rk),intent(in) :: beta_hbar
  real(rk),intent(out):: flux
  real(rk) :: vmax,pi
  pi=4.D0*atan(1.D0)
  vmax=(Va+Vb)**2/(4.D0*Vb)
  print*,"V_max=",vmax
  flux=-beta_hbar*vmax-log(2.D0*pi*beta_hbar)
  
end subroutine driver_clrate

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_coords(nvar,mode,energy,coords,iam)
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  integer   ,intent(in)    :: mode
  integer   ,intent(in)    :: iam
  real(rk)  ,intent(in)    :: energy
  real(rk)  ,intent(in)    :: coords(nvar)
  integer                  :: iat
! **********************************************************************

! Only do this writing of files if I am the rank-zero processor
  if (iam /= 0) return

  if(mod(nvar,3)==0) then
    !assume coords are atoms
    if(mode==2) then
      open(unit=20,file="tsmode.xyz")
    else
      open(unit=20,file="coords.xyz")
    end if
    write(20,*) nvar/3
    write(20,*) 
    do iat=1,nvar/3
      write(20,'("H ",3f12.7)') coords((iat-1)*3+1:(iat-1)*3+3)
    end do
    close(20)
  else
    !print*,"Coords in put_coords: ",coords
  end if
end subroutine dlf_put_coords

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_error()
  implicit none
! **********************************************************************
  call dlf_mpi_abort() ! only necessary for a parallel build;
                       ! can be present for a serial build
  call exit(1)
end subroutine dlf_error

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_update()
  implicit none
! **********************************************************************
  ! only a dummy routine here.
end subroutine dlf_update


subroutine dlf_get_multistate_gradients(nvar,coords,energy,gradient,iimage,status)
  ! only a dummy routine up to now
  ! for conical intersection search
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  integer   ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(in)    :: energy(2)
  real(rk)  ,intent(in)    :: gradient(nvar,2)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: status
end subroutine dlf_get_multistate_gradients


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)

  implicit none

  integer, intent(in) :: dlf_nprocs ! total number of processors
  integer, intent(in) :: dlf_iam ! my rank, from 0, in mpi_comm_world
  integer, intent(in) :: dlf_global_comm ! world-wide communicator
! **********************************************************************

!!! variable in the calling program = corresponding dummy argument

end subroutine dlf_put_procinfo


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)

  implicit none

  integer :: dlf_nprocs ! total number of processors
  integer :: dlf_iam ! my rank, from 0, in mpi_comm_world
  integer :: dlf_global_comm ! world-wide communicator
! **********************************************************************

!!! dummy argument = corresponding variable in the calling program

end subroutine dlf_get_procinfo


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_taskfarm(dlf_ntasks, dlf_nprocs_per_task, dlf_iam_in_task, &
                        dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)

  implicit none

  integer, intent(in) :: dlf_ntasks          ! number of taskfarms
  integer, intent(in) :: dlf_nprocs_per_task ! no of procs per farm
  integer, intent(in) :: dlf_iam_in_task     ! my rank, from 0, in my farm
  integer, intent(in) :: dlf_mytask          ! rank of my farm, from 0
  integer, intent(in) :: dlf_task_comm       ! communicator within each farm
  integer, intent(in) :: dlf_ax_tasks_comm   ! communicator involving the 
                                             ! i-th proc from each farm
! **********************************************************************

!!! variable in the calling program = corresponding dummy argument 

end subroutine dlf_put_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_taskfarm(dlf_ntasks, dlf_nprocs_per_task, dlf_iam_in_task, &
                        dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)

  implicit none

  integer :: dlf_ntasks          ! number of taskfarms
  integer :: dlf_nprocs_per_task ! no of procs per farm
  integer :: dlf_iam_in_task     ! my rank, from 0, in my farm
  integer :: dlf_mytask          ! rank of my farm, from 0
  integer :: dlf_task_comm       ! communicator within each farm
  integer :: dlf_ax_tasks_comm   ! communicator involving the
                                 ! i-th proc from each farm
! **********************************************************************

!!! dummy argument = corresponding variable in the calling program

end subroutine dlf_get_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_output(dum_stdout, dum_stderr)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,keep_alloutput
  implicit none
  integer :: dum_stdout
  integer :: dum_stderr
  integer :: ierr
  logical :: topened
  character(len=10) :: suffix

! sort out output units; particularly important on multiple processors
 
! set unit numbers for main output and error messages
  if (dum_stdout >= 0) stdout = dum_stdout 
  if (dum_stderr >= 0) stderr = dum_stderr

  if (glob%iam /= 0) then
     inquire(unit=stdout, opened=topened, iostat=ierr)
     if (topened .and. ierr == 0) close(stdout)
     if (keep_alloutput) then ! hardwired in dlf_global_module.f90
        write(suffix,'(i10)') glob%iam
        open(unit=stdout,file='output.proc'//trim(adjustl(suffix)))
     else
        open(unit=stdout,file='/dev/null')
     end if
  endif

  if (glob%nprocs > 1) then
     ! write some info on the parallelization
     write(stdout,'(1x,a,i10,a)')"I have rank ",glob%iam," in mpi_comm_world"
     write(stdout,'(1x,a,i10)')"Total number of processors = ",glob%nprocs
     if (keep_alloutput) then
        write(stdout,'(1x,a)')"Keeping output from all processors"
     else
        write(stdout,'(1x,a)')"Not keeping output from processors /= 0"
     end if
  end if

end subroutine dlf_output


! **********************************************************************
! **********************************************************************
! The following routine either writes random numbers to a file, or reads
! them. This is to have equal starting conditions for different compilers
subroutine read_rand(arr)
  use dlf_parameter_module, only: rk
  use dlf_global, only : glob
  real(rk)  :: arr(:)
  integer, parameter :: si1=12
  integer, parameter :: si2=3000
  logical,parameter :: readf=.true.
  real(rk) :: ar(si2)
  integer :: l(1),length
  l=ubound(arr)
  length=l(1)
  if(readf) then
    if(length<=si1) then
      open(unit=201,file="random1.bin",form="unformatted")
    else if(length<=si2) then
      open(unit=201,file="random2.bin",form="unformatted")
    else
      call dlf_mpi_finalize() ! only necessary for a parallel build;
                              ! can be present for a serial build
      stop "Too many coordinates to be read from random.bin file"
    end if
    read(201) ar(1:length)
    close(201)
    arr=ar(1:length)
  else
    if (glob%iam == 0) then
       call random_number(ar)
       open(unit=201,file="random1.bin",form="unformatted")
       write(201) ar(1:si1)
       close(201)
       open(unit=201,file="random2.bin",form="unformatted")
       write(201) ar(1:si2)
       close(201)
    end if
  end if
end subroutine read_rand

subroutine driver_eck(vb_,alpha_)
  use dlf_parameter_module
  use driver_parameter_module
  real(rk),intent(out) :: vb_,alpha_
  vb_=vb
  alpha_=alpha
end subroutine driver_eck
