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

  public :: optimize

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
     use quick_dlfind_module, only: dlfind_interface
!     use quick_dlfind_module, only: dlfind_init, dlfind_run, dlfind_final 
     use quick_exception_module
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

#ifdef MPIV
   include "mpif.h"
#endif

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
     
!     if (master) call dlfind_init
     !------------- END MPI/MASTER ----------------------------

     call dlfind_interface(ierr)

     return
  end subroutine optimize

end module quick_optimizer_module
