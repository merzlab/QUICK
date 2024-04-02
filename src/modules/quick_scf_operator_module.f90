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

module quick_scf_operator_module

  implicit none
  private 

  public :: scf_operator
  

contains
  
  subroutine scf_operator(deltaO)
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
     use quick_cutoff_module, only: cshell_density_cutoff
     use quick_cshell_eri_module, only: getCshellEri, getCshellEriEnergy 
     use quick_oei_module, only:get1eEnergy,get1e
#ifdef MPIV
     use mpi
#endif
  
     implicit none
  !   double precision oneElecO(nbasis,nbasis)
     logical :: deltaO
     integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, I, J
     common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
     double precision tst, te, tred
#ifdef MPIV
     integer ierror
     double precision :: Eelsum, Excsum, aelec, belec
  
     quick_scratch%osum=0.0d0
     Eelsum=0.0d0
     Excsum=0.0d0
     aelec=0.0d0
     belec=0.0d0
#endif
  
     quick_qm_struct%o = 0.0d0
     quick_qm_struct%Eel=0.0d0
  
  !-----------------------------------------------------------------
  !  Step 1. evaluate 1e integrals
  !-----------------------------------------------------------------
  
  !  if only calculate operation difference
     if (deltaO) then
  !     save density matrix
        quick_qm_struct%denseSave(:,:) = quick_qm_struct%dense(:,:)
        quick_qm_struct%dense=quick_qm_struct%dense-quick_qm_struct%denseOld

        if(quick_method%dft) then
          quick_qm_struct%o = quick_qm_struct%oSave-quick_qm_struct%oxc
        else
          quick_qm_struct%o(:,:) = quick_qm_struct%oSave(:,:)
        endif
     endif
  
  !  Delta density matrix cutoff
     call cshell_density_cutoff
  
#ifdef MPIV
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif
  
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
     if (quick_method%bCUDA) then
  
        call gpu_upload_calculated(quick_qm_struct%o,quick_qm_struct%co, &
        quick_qm_struct%vec,quick_qm_struct%dense)
        call gpu_upload_cutoff(cutmatrix,quick_method%integralCutoff,quick_method%primLimit,quick_method%DMCutoff, &
                                quick_method%coreIntegralCutoff)
  
     endif
#endif
 
     call get1e(deltaO)

     if(quick_method%printEnergy) call get1eEnergy(deltaO)


!     if (quick_method%nodirect) then
!#ifdef CUDA
!        call gpu_addint(quick_qm_struct%o, intindex, intFileName)
!#else
!#ifndef MPI
!        call addInt
!#endif
!#endif
!     else
  !-----------------------------------------------------------------
  ! Step 2. evaluate 2e integrals
  !-----------------------------------------------------------------
  !
  ! The previous two terms are the one electron part of the Fock matrix.
  ! The next two terms define the two electron part.
  !-----------------------------------------------------------------
  !  Start the timer for 2e-integrals
     RECORD_TIME(timer_begin%T2e)

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
        if (quick_method%bCUDA) then          
           call gpu_get_cshell_eri(deltaO, quick_qm_struct%o)  
        else                                  
#endif
  !  Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
  !  Reference: Strout DL and Scuseria JCP 102(1995),8448.
  
#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
  !  Every nodes will take about jshell/nodes shells integrals such as 1 water, which has 
  !  4 jshell, and 2 nodes will take 2 jshell respectively.
     if(bMPI) then
        do i=1,mpi_jshelln(mpirank)
           ii=mpi_jshell(mpirank,i)
           call getCshellEri(II)
        enddo
     else
        do II=1,jshell
           call getCshellEri(II)
        enddo
     endif        
#else
        do II=1,jshell
           call getCshellEri(II)
        enddo
#endif
  
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
        endif                             
#endif
 !    endif
  
  !  Remember the operator is symmetric
     call copySym(quick_qm_struct%o,nbasis)
  
  !  recover density if calculate difference
     if (deltaO) quick_qm_struct%dense(:,:) = quick_qm_struct%denseSave(:,:)

  !  Give the energy, E=1/2*sigma[i,j](Pij*(Fji+Hcoreji))
     if(quick_method%printEnergy) call getCshellEriEnergy



#ifdef MPIV
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif
  
  !  Terminate the timer for 2e-integrals
     RECORD_TIME(timer_end%T2e)
  
  !  add the time to cumer
     timer_cumer%T2e=timer_cumer%T2e+timer_end%T2e-timer_begin%T2e
  
  !-----------------------------------------------------------------
  !  Step 3. If DFT, evaluate the exchange/correlation contribution 
  !          to the operator
  !-----------------------------------------------------------------
  
#ifdef MPIV
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif
  
     if (quick_method%DFT) then
  
  !  Start the timer for exchange correlation calculation
        RECORD_TIME(timer_begin%TEx)

  !  Calculate exchange correlation contribution & add to operator    
        call get_xc(deltaO)

  !  Remember the operator is symmetric
        call copySym(quick_qm_struct%o,nbasis)
 
#ifdef MPIV
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif
  
  !  Stop the exchange correlation timer
        RECORD_TIME(timer_end%TEx)
  
  !  Add time total time
        timer_cumer%TEx=timer_cumer%TEx+timer_end%TEx-timer_begin%TEx
     endif

     quick_qm_struct%oSave(:,:) = quick_qm_struct%o(:,:) 
 
#ifdef MPIV
  !  MPI reduction operations
  
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  
     RECORD_TIME(timer_begin%TEred)
  
     if (quick_method%DFT) then
     call MPI_REDUCE(quick_qm_struct%Exc, Excsum, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
     call MPI_REDUCE(quick_qm_struct%aelec, aelec, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
     call MPI_REDUCE(quick_qm_struct%belec, belec, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
  
     if(master) then
       quick_qm_struct%Exc = Excsum
       quick_qm_struct%aelec  = aelec
       quick_qm_struct%belec  = belec
     endif
     endif
  
     call MPI_REDUCE(quick_qm_struct%o, quick_scratch%osum, nbasis*nbasis, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
     call MPI_REDUCE(quick_qm_struct%Eel, Eelsum, 1, mpi_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, IERROR)
  
     if(master) then
       quick_qm_struct%o(:,:) = quick_scratch%osum(:,:)
       quick_qm_struct%Eel    = Eelsum

!do i=1, nbasis
!  do j=1, nbasis
!    write(*,*) j, i, quick_qm_struct%o(j,i)
!  enddo
!enddo
     endif

     RECORD_TIME(timer_end%TEred)
     timer_cumer%TEred=timer_cumer%TEred+timer_end%TEred-timer_begin%TEred
  
#endif

  return
  
  end subroutine scf_operator
  
  subroutine get_xc(deltaO)
  !----------------------------------------------------------------
  !  The purpose of this subroutine is to calculate the exchange
  !  correlation contribution to the Fock operator. 
  !  The angular grid code came from CCL.net.  The radial grid
  !  formulas (position and wieghts) is from Gill, Johnson and Pople,
  !  Chem. Phys. Lett. v209, n 5+6, 1993, pg 506-512.  The weighting scheme
  !  is from Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
  !  1996, pg 213-223.
  !
  !  The actual element is:
  !  F alpha mu nu = Integral((df/drhoa Phimu Phinu)+
  !  (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
  !  where F alpha mu nu is the the alpha spin portion of the operator matrix
  !  element mu, nu,
  !  df/drhoa is the derivative of the functional by the alpha density,
  !  df/dgaa is the derivative of the functional by the alpha gradient
  !  invariant, i.e. the dot product of the gradient of the alpha
  !  density with itself.
  !  df/dgab is the derivative of the functional by the dot product of
  !  the gradient of the alpha density with the beta density.
  !  Grad(Phimu Phinu) is the gradient of Phimu times Phinu. 
  !----------------------------------------------------------------
     use allmod
     use quick_gridpoints_module
     use quick_dft_module, only: b3lypf, b3lyp_e, becke, becke_e, lyp, lyp_e
     use xc_f90_types_m
     use xc_f90_lib_m
#ifdef MPIV
     use mpi
#endif
     implicit none
  
     !integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, I, J
     !common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
  
     logical, intent(in) :: deltaO
     double precision, dimension(1) :: libxc_rho
     double precision, dimension(1) :: libxc_sigma
     double precision, dimension(1) :: libxc_exc
     double precision, dimension(1) :: libxc_vrhoa
     double precision, dimension(1) :: libxc_vsigmaa
     type(xc_f90_pointer_t), dimension(quick_method%nof_functionals) :: xc_func
     type(xc_f90_pointer_t), dimension(quick_method%nof_functionals) :: xc_info   
     integer :: iatm, ibas, ibin, icount, ifunc, igp, jbas, jcount, ierror 
     double precision :: density, densityb, densitysum, dfdgaa, dfdgaa2, dfdgab, &
     dfdgab2, dfdr, dfdr2, dphi2dx, dphi2dy, dphi2dz, dphidx, dphidy, dphidz, &
     gax, gay, gaz, gbx, gby, gbz, gridx, gridy, gridz, phi, phi2, quicktest, &
     sigma, sswt, temp, tempgx, tempgy, tempgz, tsttmp_exc, tsttmp_vrhoa, &
     tsttmp_vsigmaa, weight, xdot, ydot, zdot, xiaodot, zkec, Ex, Ec, Eelxc
  
#ifdef MPIV
     integer :: i, ii, irad_end, irad_init, jj
#endif
 
     quick_qm_struct%Exc=0.0d0
     quick_qm_struct%aelec=0.d0
     quick_qm_struct%belec=0.d0

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  
     if(quick_method%bCUDA) then

        if(deltaO) call gpu_upload_density_matrix(quick_qm_struct%dense)

        quick_qm_struct%oxc=quick_qm_struct%o 
 
        call gpu_get_cshell_xc(quick_qm_struct%Exc, quick_qm_struct%aelec, quick_qm_struct%belec, quick_qm_struct%o)

        quick_qm_struct%oxc=quick_qm_struct%o-quick_qm_struct%oxc  

     endif
#else

     quick_qm_struct%oxc=0.0d0 
 
     if(quick_method%uselibxc) then
  !  Initiate the libxc functionals
        do ifunc=1, quick_method%nof_functionals
           if(quick_method%xc_polarization > 0 ) then
              call xc_f90_func_init(xc_func(ifunc), xc_info(ifunc), &
                    quick_method%functional_id(ifunc),XC_POLARIZED)
           else
              call xc_f90_func_init(xc_func(ifunc), &
                    xc_info(ifunc),quick_method%functional_id(ifunc),XC_UNPOLARIZED)
           endif
        enddo
     endif
  
  
#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
        if(bMPI) then
           irad_init = quick_dft_grid%igridptll(mpirank+1)
           irad_end = quick_dft_grid%igridptul(mpirank+1)
        else
           irad_init = 1
           irad_end = quick_dft_grid%nbins
        endif
     do Ibin=irad_init, irad_end
     
#else
      do Ibin=1, quick_dft_grid%nbins
#endif
  
          Igp=quick_dft_grid%bin_counter(Ibin)+1
  
          do while(Igp < quick_dft_grid%bin_counter(Ibin+1)+1)
  
             gridx=quick_dft_grid%gridxb(Igp)
             gridy=quick_dft_grid%gridyb(Igp)
             gridz=quick_dft_grid%gridzb(Igp)
  
             sswt=quick_dft_grid%gridb_sswt(Igp)
             weight=quick_dft_grid%gridb_weight(Igp)
             Iatm=quick_dft_grid%gridb_atm(Igp)
  
              if (weight < quick_method%DMCutoff ) then
                 continue
              else
  
                 icount=quick_dft_grid%basf_counter(Ibin)+1
                 do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
                 Ibas=quick_dft_grid%basf(icount)+1
                    call pteval_new_imp(gridx,gridy,gridz,phi,dphidx,dphidy, &
                    dphidz,Ibas,icount)
                    phixiao(Ibas)=phi
                    dphidxxiao(Ibas)=dphidx
                    dphidyxiao(Ibas)=dphidy
                    dphidzxiao(Ibas)=dphidz
  
                    icount=icount+1
                 enddo
  
  !  Next, evaluate the densities at the grid point and the gradient
  !  at that grid point.
  
  !               call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
  !               gbx,gby,gbz)
  
                 call denspt_new_imp(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                 gbx,gby,gbz,Ibin)
  
                 if (density < quick_method%DMCutoff ) then
                    continue
                 else
  
  !  This allows the calculation of the derivative of the functional with regard to the 
  !  density (dfdr), with regard to the alpha-alpha density invariant (df/dgaa), and the
  !  alpha-beta density invariant.
  
                    densitysum=2.0d0*density
                    sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)
  
                    libxc_rho(1)=densitysum
                    libxc_sigma(1)=sigma
  
                    tsttmp_exc=0.0d0
                    tsttmp_vrhoa=0.0d0
                    tsttmp_vsigmaa=0.0d0
  
                    if(quick_method%uselibxc) then
                       do ifunc=1, quick_method%nof_functionals
                          select case(xc_f90_info_family(xc_info(ifunc)))
                             case(XC_FAMILY_LDA)
                                call xc_f90_lda_exc_vxc(xc_func(ifunc),1,libxc_rho(1), &
                                libxc_exc(1), libxc_vrhoa(1))
                                libxc_vsigmaa(1) = 0.0d0
                             case(XC_FAMILY_GGA, XC_FAMILY_HYB_GGA)
                                call xc_f90_gga_exc_vxc(xc_func(ifunc),1,libxc_rho(1), libxc_sigma(1), &
                                libxc_exc(1), libxc_vrhoa(1), libxc_vsigmaa(1))
                          end select
  
                          tsttmp_exc=tsttmp_exc+libxc_exc(1)
                          tsttmp_vrhoa=tsttmp_vrhoa+libxc_vrhoa(1)
                          tsttmp_vsigmaa=tsttmp_vsigmaa+libxc_vsigmaa(1)
                       enddo
  
                       zkec=densitysum*tsttmp_exc
                       dfdr=tsttmp_vrhoa
                       xiaodot=tsttmp_vsigmaa*4
  
  !  Calculate the first term in the dot product shown above,
  !  i.e.: (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                       xdot=xiaodot*gax
                       ydot=xiaodot*gay
                       zdot=xiaodot*gaz
  
                    elseif(quick_method%BLYP) then
  
                       call becke_E(density, densityb, gax, gay, gaz, gbx, gby,gbz, Ex)
                       call lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz,Ec)
  
                       zkec=Ex+Ec
                       
                       call becke(density, gax, gay, gaz, gbx, gby, gbz, dfdr, dfdgaa, dfdgab)
                       call lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, dfdr2, dfdgaa2, dfdgab2)
  
                       dfdr = dfdr + dfdr2
                       dfdgaa = dfdgaa + dfdgaa2
                       dfdgab = dfdgab + dfdgab2
  
                       xdot = 2.d0*dfdgaa*gax + dfdgab*gbx
                       ydot = 2.d0*dfdgaa*gay + dfdgab*gby
                       zdot = 2.d0*dfdgaa*gaz + dfdgab*gbz
  
                    elseif(quick_method%B3LYP) then
  
                       call b3lyp_e(densitysum, sigma, zkec)
                       call b3lypf(densitysum, sigma, dfdr, xiaodot)
  
                       xdot=xiaodot*gax
                       ydot=xiaodot*gay
                       zdot=xiaodot*gaz
  
                    endif
  
                    quick_qm_struct%Exc = quick_qm_struct%Exc + zkec*weight
  
                    quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                    quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec
  
  !  Now loop over basis functions and compute the addition to the matrix element.
  !                  do Ibas=1,nbasis
                    icount=quick_dft_grid%basf_counter(Ibin)+1
                    do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
                    Ibas=quick_dft_grid%basf(icount)+1
  
                       phi=phixiao(Ibas)
                       dphidx=dphidxxiao(Ibas)
                       dphidy=dphidyxiao(Ibas)
                       dphidz=dphidzxiao(Ibas)
                       quicktest = DABS(dphidx+dphidy+dphidz+phi)
  
                       if (quicktest < quick_method%XCCutoff ) then
                          continue
                       else
                          jcount=icount
                          do while(jcount<quick_dft_grid%basf_counter(Ibin+1)+1)
                          Jbas = quick_dft_grid%basf(jcount)+1
                          !do Jbas=Ibas,nbasis
                             phi2=phixiao(Jbas)
                             dphi2dx=dphidxxiao(Jbas)
                             dphi2dy=dphidyxiao(Jbas)
                             dphi2dz=dphidzxiao(Jbas)
                             temp = phi*phi2
                             tempgx = phi*dphi2dx + phi2*dphidx
                             tempgy = phi*dphi2dy + phi2*dphidy
                             tempgz = phi*dphi2dz + phi2*dphidz
                             quick_qm_struct%oxc(Jbas,Ibas)=quick_qm_struct%oxc(Jbas,Ibas)+(temp*dfdr+&
                             xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                             jcount=jcount+1
                          enddo
                       endif
                       icount=icount+1
                    enddo
                 endif
              endif
           !enddo
  
           Igp=Igp+1
        enddo
     enddo
  
     if(quick_method%uselibxc) then
  !  Uninitilize libxc functionals
        do ifunc=1, quick_method%nof_functionals
           call xc_f90_func_end(xc_func(ifunc))
        enddo
     endif

  !  Update KS operators
     quick_qm_struct%o=quick_qm_struct%o+quick_qm_struct%oxc

#endif

  !  Add the exchange correlation energy to total electronic energy
     quick_qm_struct%Eel    = quick_qm_struct%Eel+quick_qm_struct%Exc

     return
  
  end subroutine get_xc

end module quick_scf_operator_module
