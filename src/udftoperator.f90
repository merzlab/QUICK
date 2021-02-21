! Ed Brothers. Febuary 8,2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine udftoperatora
    use allmod
    implicit double precision(a-h,o-z)
    double precision g_table(200)
    integer i,j,k,ii,jj,kk,g_count

! The purpose of this subroutine is to form the operator matrix
! for the alpha Density Functional calculation, i.e. the alpha KS
! matrix.  The KS  matrix is as follows:

! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
! with each possible basis  + Exchange/Correlation functional.

! Note that the KS operator matrix is symmetric.

    do Ibas=1,nbasis
        do Jbas=Ibas,nbasis
            Ax = xyz(1,quick_basis%ncenter(Jbas))
            Bx = xyz(1,quick_basis%ncenter(Ibas))
            Ay = xyz(2,quick_basis%ncenter(Jbas))
            By = xyz(2,quick_basis%ncenter(Ibas))
            Az = xyz(3,quick_basis%ncenter(Jbas))
            Bz = xyz(3,quick_basis%ncenter(Ibas))
            
            ii = itype(1,Ibas)
            jj = itype(2,Ibas)
            kk = itype(3,Ibas)
            i = itype(1,Jbas)
            j = itype(2,Jbas)
            k = itype(3,Jbas)
            g_count = i+ii+j+jj+k+kk+2

            quick_qm_struct%o(Jbas,Ibas) = 0.d0
            do Icon=1,ncontract(ibas)
                do Jcon=1,ncontract(jbas)

                   b = aexp(Icon,Ibas)
                   a = aexp(Jcon,Jbas)
                   call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)     

                ! The first part is the kinetic energy.

                    quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+ &
                    dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                      ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                    ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                    itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                    xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                    xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                    xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

                ! Next is a loop over atoms to contruct the attraction terms.

                    do iatom = 1,natom
                        quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+ &
                        dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                        attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                        itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                        xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                        xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                        quick_molspec%chg(iatom))
                    enddo
                enddo
            enddo
        enddo
    enddo

!
! Alessandro GENONI 03/21/2007
! Sum the ECP integrals to the partial Fock matrix
!
    if (quick_method%ecp) then
      call ecpoperator
    end if

! The previous two terms are the one electron part of the Fock matrix.
! The next term defines the electron repulsion_prim.


    do I=1,nbasis
    ! Set some variables to reduce access time for some of the more
    ! used quantities.

        DENSEII=quick_qm_struct%dense(I,I)+quick_qm_struct%denseb(I,I)

    ! do all the (ii|ii) integrals.

        repint = repulsion(gauss(I),gauss(I),gauss(I),gauss(I), &
        xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
        xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)))
        quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+DENSEII*repint

        do J=I+1,nbasis
        ! Set some variables to reduce access time for some of the more
        ! used quantities. (AGAIN)

            DENSEJI=quick_qm_struct%dense(J,I)+quick_qm_struct%denseb(J,I)
            DENSEJJ=quick_qm_struct%dense(J,J)+quick_qm_struct%denseb(J,J)

        ! Find  all the (ii|jj) integrals.

            repint = repulsion(gauss(I),gauss(I),gauss(J),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
            xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+DENSEJJ*repint
            quick_qm_struct%o(J,J) = quick_qm_struct%o(J,J)+DENSEII*repint

        ! Find  all the (ij|jj) integrals.

            repint = repulsion(gauss(I),gauss(J),gauss(J),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
            xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+DENSEJJ*repint
            quick_qm_struct%o(J,J) = quick_qm_struct%o(J,J)+2.d0*DENSEJI*repint

        ! Find  all the (ii|ij) integrals.

            repint = repulsion(gauss(I),gauss(I),gauss(I),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+DENSEII*repint
            quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+2.d0*DENSEJI*repint

        ! Find all the (ij|ij) integrals
            repint = repulsion(gauss(I),gauss(J),gauss(I),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+2.d0*DENSEJI*repint

            do K=J+1,nbasis
            ! Set some variables to reduce access time for some of the more
            ! used quantities. (AGAIN)

                DENSEKI=quick_qm_struct%dense(K,I)+quick_qm_struct%denseb(K,I)
                DENSEKJ=quick_qm_struct%dense(K,J)+quick_qm_struct%denseb(K,J)
                DENSEKK=quick_qm_struct%dense(K,K)+quick_qm_struct%denseb(K,K)

            ! Find all the (ij|ik) integrals where j>i,k>j
                repint = repulsion(gauss(I),gauss(J),gauss(I),gauss(K), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(K)))
                quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+2.d0*DENSEKI*repint
                quick_qm_struct%o(K,I) = quick_qm_struct%o(K,I)+2.d0*DENSEJI*repint

            ! Find all the (ij|kk) integrals where j>i, k>j.

                repint = repulsion(gauss(I),gauss(J),gauss(K),gauss(K), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
                xyz(:,quick_basis%ncenter(K)),xyz(:,quick_basis%ncenter(K)))
                quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+DENSEKK*repint
                quick_qm_struct%o(K,K) = quick_qm_struct%o(K,K)+2.d0*DENSEJI*repint

            ! Find all the (ik|jj) integrals where j>i, k>j.

                repint = repulsion(gauss(I),gauss(K),gauss(J),gauss(J), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(K)), &
                xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(J)))
                quick_qm_struct%o(K,I) = quick_qm_struct%o(K,I)+DENSEJJ*repint
                quick_qm_struct%o(J,J) = quick_qm_struct%o(J,J)+2.d0*DENSEKI*repint

            ! Find all the (ii|jk) integrals where j>i, k>j.

                repint = repulsion(gauss(I),gauss(I),gauss(J),gauss(K), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
                xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(K)))
                quick_qm_struct%o(K,J) = quick_qm_struct%o(K,J)+DENSEII*repint
                quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+2.d0*DENSEKJ*repint
            enddo

            do K=I+1,nbasis-1

                do L=K+1,nbasis
                    DENSELK=quick_qm_struct%dense(L,K)+quick_qm_struct%denseb(L,K)

                ! Find the (ij|kl) integrals where j>i,k>i,l>k.

                    repint = repulsion(gauss(I),gauss(J),gauss(K),gauss(L), &
                    xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
                    xyz(:,quick_basis%ncenter(K)),xyz(:,quick_basis%ncenter(L)))
                    quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+2.d0*DENSELK*repint
                    quick_qm_struct%o(L,K) = quick_qm_struct%o(L,K)+2.d0*DENSEJI*repint
                enddo
            enddo
        enddo
    enddo

! The next portion is the exchange/correlation functional.
! The angular grid code came from CCL.net.  The radial grid
! formulas (position and wieghts) is from Gill, Johnson and Pople,
! Chem. Phys. Lett. v209, n 5+6, 1993, pg 506-512.  The weighting scheme
! is from Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
! 1996, pg 213-223.

! The actual element is:
! F alpha mu nu = Integral((df/drhoa Phimu Phinu)+
! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

! where F alpha mu nu is the the alpha spin portion of the operator matrix
! element mu, nu,
! df/drhoa is the derivative of the functional by the alpha density,
! df/dgaa is the derivative of the functional by the alpha gradient
! invariant, i.e. the dot product of the gradient of the alpha
! density with itself.
! df/dgab is the derivative of the functional by the dot product of
! the gradient of the alpha density with the beta density.
! Grad(Phimu Phinu) is the gradient of Phimu times Phinu.

! First, find the grid point.

    quick_qm_struct%aelec=0.d0
    quick_qm_struct%belec=0.d0

    do Ireg=1,iregion
        call gridform(iangular(Ireg))
        do Irad=iradial(Ireg-1)+1,iradial(Ireg)
            do Iatm=1,natom
                rad = radii(quick_molspec%iattype(iatm))
                rad3 = rad*rad*rad
                do Iang=1,iangular(Ireg)
                    gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
                    gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
                    gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

                ! Next, calculate the weight of the grid point in the SSW scheme.  If
                ! the grid point has a zero weight, we can skip it.

                    weight=SSW(gridx,gridy,gridz,Iatm) &
                    *WTANG(Iang)*RWT(Irad)*rad3

                    if (weight < quick_method%DMCutoff ) then
                        continue
                    else

                    ! Next, evaluate the densities at the grid point and the gradient
                    ! at that grid point.

                        call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                        gbx,gby,gbz)

                        quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                        quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec
                        if (density < quick_method%DMCutoff ) then
                            continue
                        else

                        ! This allows the calculation of the derivative of the functional
                        ! with regard to the density (dfdr), with regard to the alpha-alpha
                        ! density invariant (df/dgaa), and the alpha-beta density invariant.

                            call becke(density,gax,gay,gaz,gbx,gby,gbz, &
                            dfdr,dfdgaa,dfdgab)

                        ! Calculate the first term in the dot product shown above,i.e.:
                        ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

                            xdot=2.d0*dfdgaa*gax+dfdgab*gbx
                            ydot=2.d0*dfdgaa*gay+dfdgab*gby
                            zdot=2.d0*dfdgaa*gaz+dfdgab*gbz

                        ! Now loop over basis functions and compute the addition to the matrix
                        ! element.

                            do Ibas=1,nbasis
                                call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                                dphidz,Ibas)
                                quicktest = DABS(dphidx)+DABS(dphidy)+DABS(dphidz) &
                                +DABS(phi)
                                if (quicktest < quick_method%DMCutoff ) then
                                    continue
                                else
                                    do Jbas=Ibas,nbasis
                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                        dphi2dz,Jbas)
                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                        +DABS(phi2)
                                        if (quicktest < quick_method%DMCutoff ) then
                                            continue
                                        else
                                            temp = phi*phi2
                                            tempgx = phi*dphi2dx + phi2*dphidx
                                            tempgy = phi*dphi2dy + phi2*dphidy
                                            tempgz = phi*dphi2dz + phi2*dphidz
                                            quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                            xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                                        endif
                                    enddo
                                endif
                            enddo
                        endif
                    endif
                enddo
            enddo
        enddo
    enddo
! Finally, copy lower diagonal to upper diagonal.

    do Ibas=1,nbasis
        do Jbas=Ibas+1,nbasis
            quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
        enddo
    enddo

    end subroutine udftoperatora

! Ed Brothers. Febuary 8,2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine udftoperatorb
    use allmod
    implicit double precision(a-h,o-z)
    double precision g_table(200)
    integer i,j,k,ii,jj,kk,g_count

! The purpose of this subroutine is to form the operator matrix
! for the beta Density Functional calculation, i.e. the beta KS
! matrix.  The KS  matrix is as follows:

! quick_qm_struct%o(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
! with each possible basis  + Exchange/Correlation functional.

! Noteethat the KS operator matrix is symmetric.
! enddo

    do Ibas=1,nbasis
        do Jbas=Ibas,nbasis

            Ax = xyz(1,quick_basis%ncenter(Jbas))
            Bx = xyz(1,quick_basis%ncenter(Ibas))
            Ay = xyz(2,quick_basis%ncenter(Jbas))
            By = xyz(2,quick_basis%ncenter(Ibas))
            Az = xyz(3,quick_basis%ncenter(Jbas))
            Bz = xyz(3,quick_basis%ncenter(Ibas))
            
            ii = itype(1,Ibas)
            jj = itype(2,Ibas)
            kk = itype(3,Ibas)
            i = itype(1,Jbas)
            j = itype(2,Jbas)
            k = itype(3,Jbas)
            g_count = i+ii+j+jj+k+kk+2

            quick_qm_struct%o(Jbas,Ibas) = 0.d0
            do Icon=1,ncontract(ibas)
                do Jcon=1,ncontract(jbas)

                   b = aexp(Icon,Ibas)
                   a = aexp(Jcon,Jbas)
                   call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table) 

               ! The first part is the kinetic energy.

                    quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+ &
                    dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                      ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                    ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                    itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                    xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter!(Jbas)), &
!                    xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                    xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

                ! Next is a loop over atoms to contruct the attraction terms.

                    do iatom = 1,natom
                        quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+ &
                        dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                        attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                        itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                        xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                        xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                        quick_molspec%chg(iatom))
                    enddo
                enddo
            enddo
        enddo
    enddo

!
! Alessandro GENONI 03/21/2007
! Sum the ECP integrals to the partial Fock matrix
!
    if (quick_method%ecp) then
      call ecpoperator
    end if

! The previous two terms are the one electron part of the Fock matrix.
! The next term defines the electron repulsion_prim.


    do I=1,nbasis
    ! Set some variables to reduce access time for some of the more
    ! used quantities.

        DENSEII=quick_qm_struct%dense(I,I)+quick_qm_struct%denseb(I,I)

    ! do all the (ii|ii) integrals.

        repint = repulsion(gauss(I),gauss(I),gauss(I),gauss(I), &
        xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
        xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)))
        quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+DENSEII*repint

        do J=I+1,nbasis
        ! Set some variables to reduce access time for some of the more
        ! used quantities. (AGAIN)

            DENSEJI=quick_qm_struct%dense(J,I)+quick_qm_struct%denseb(J,I)
            DENSEJJ=quick_qm_struct%dense(J,J)+quick_qm_struct%denseb(J,J)

        ! Find  all the (ii|jj) integrals.

            repint = repulsion(gauss(I),gauss(I),gauss(J),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
            xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+DENSEJJ*repint
            quick_qm_struct%o(J,J) = quick_qm_struct%o(J,J)+DENSEII*repint

        ! Find  all the (ij|jj) integrals.

            repint = repulsion(gauss(I),gauss(J),gauss(J),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
            xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+DENSEJJ*repint
            quick_qm_struct%o(J,J) = quick_qm_struct%o(J,J)+2.d0*DENSEJI*repint

        ! Find  all the (ii|ij) integrals.

            repint = repulsion(gauss(I),gauss(I),gauss(I),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+DENSEII*repint
            quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+2.d0*DENSEJI*repint

        ! Find all the (ij|ij) integrals

            repint = repulsion(gauss(I),gauss(J),gauss(I),gauss(J), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
            xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)))
            quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+2.d0*DENSEJI*repint

            do K=J+1,nbasis
            ! Set some variables to reduce access time for some of the more
            ! used quantities. (AGAIN)

                DENSEKI=quick_qm_struct%dense(K,I)+quick_qm_struct%denseb(K,I)
                DENSEKJ=quick_qm_struct%dense(K,J)+quick_qm_struct%denseb(K,J)
                DENSEKK=quick_qm_struct%dense(K,K)+quick_qm_struct%denseb(K,K)

            ! Find all the (ij|ik) integrals where j>i,k>j

                repint = repulsion(gauss(I),gauss(J),gauss(I),gauss(K), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(K)))
                quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+2.d0*DENSEKI*repint
                quick_qm_struct%o(K,I) = quick_qm_struct%o(K,I)+2.d0*DENSEJI*repint

            ! Find all the (ij|kk) integrals where j>i, k>j.

                repint = repulsion(gauss(I),gauss(J),gauss(K),gauss(K), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
                xyz(:,quick_basis%ncenter(K)),xyz(:,quick_basis%ncenter(K)))
                quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+DENSEKK*repint
                quick_qm_struct%o(K,K) = quick_qm_struct%o(K,K)+2.d0*DENSEJI*repint

            ! Find all the (ik|jj) integrals where j>i, k>j.

                repint = repulsion(gauss(I),gauss(K),gauss(J),gauss(J), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(K)), &
                xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(J)))
                quick_qm_struct%o(K,I) = quick_qm_struct%o(K,I)+DENSEJJ*repint
                quick_qm_struct%o(J,J) = quick_qm_struct%o(J,J)+2.d0*DENSEKI*repint

            ! Find all the (ii|jk) integrals where j>i, k>j.

                repint = repulsion(gauss(I),gauss(I),gauss(J),gauss(K), &
                xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(I)), &
                xyz(:,quick_basis%ncenter(J)),xyz(:,quick_basis%ncenter(K)))
                quick_qm_struct%o(K,J) = quick_qm_struct%o(K,J)+DENSEII*repint
                quick_qm_struct%o(I,I) = quick_qm_struct%o(I,I)+2.d0*DENSEKJ*repint
            enddo

            do K=I+1,nbasis-1

                do L=K+1,nbasis
                    DENSELK=quick_qm_struct%dense(L,K)+quick_qm_struct%denseb(L,K)

                ! Find the (ij|kl) integrals where j>i,k>i,l>k.

                    repint = repulsion(gauss(I),gauss(J),gauss(K),gauss(L), &
                    xyz(:,quick_basis%ncenter(I)),xyz(:,quick_basis%ncenter(J)), &
                    xyz(:,quick_basis%ncenter(K)),xyz(:,quick_basis%ncenter(L)))
                    quick_qm_struct%o(J,I) = quick_qm_struct%o(J,I)+2.d0*DENSELK*repint
                    quick_qm_struct%o(L,K) = quick_qm_struct%o(L,K)+2.d0*DENSEJI*repint
                enddo
            enddo
        enddo
    enddo

! The next portion is the exchange/correlation functional.
! The angular grid code came from CCL.net.  The radial grid
! formulas (position and wieghts) is from Gill, Johnson and Pople,
! Chem. Phys. Lett. v209, n 5+6, 1993, pg 506-512.  The weighting scheme
! is from Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
! 1996, pg 213-223.

! The actual element is:
! F alpha mu nu = Integral((df/drhoa Phimu Phinu)+
! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

! where F alpha mu nu is the the alpha spin portion of the operator matrix
! element mu, nu,
! df/drhoa is the derivative of the functional by the alpha density,
! df/dgaa is the derivative of the functional by the alpha gradient
! invariant, i.e. the dot product of the gradient of the alpha
! density with itself.
! df/dgab is the derivative of the functional by the dot product of
! the gradient of the alpha density with the beta density.
! Grad(Phimu Phinu) is the gradient of Phimu times Phinu.

! First, find the grid point.

    quick_qm_struct%aelec=0.d0
    quick_qm_struct%belec=0.d0

    do Ireg=1,iregion
        call gridform(iangular(Ireg))
        do Irad=iradial(Ireg-1)+1,iradial(Ireg)
            do Iatm=1,natom
                rad = radii(quick_molspec%iattype(iatm))
                rad3 = rad*rad*rad
                do Iang=1,iangular(Ireg)
                    gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
                    gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
                    gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

                ! Next, calculate the weight of the grid point in the SSW scheme.  If
                ! the grid point has a zero weight, we can skip it.

                    weight=SSW(gridx,gridy,gridz,Iatm) &
                    *WTANG(Iang)*RWT(Irad)*rad3

                    if (weight < quick_method%DMCutoff ) then
                        continue
                    else

                    ! Next, evaluate the densities at the grid point and the gradient
                    ! at that grid point.

                        call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                        gbx,gby,gbz)

                        quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                        quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec
                        if (densityb < quick_method%DMCutoff ) then
                            continue
                        else

                        ! This allows the calculation of the derivative of the functional
                        ! with regard to the density (dfdr), with regard to the alpha-alpha
                        ! density invariant (df/dgaa), and the alpha-beta density invariant.

                            call becke(densityb,gbx,gby,gbz,gax,gay,gaz, &
                            dfdr,dfdgbb,dfdgab)

                        ! Calculate the first term in the dot product shown above,i.e.:
                        ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

                            xdot=2.d0*dfdgbb*gbx+dfdgab*gax
                            ydot=2.d0*dfdgbb*gby+dfdgab*gay
                            zdot=2.d0*dfdgbb*gbz+dfdgab*gaz

                        ! Now loop over basis functions and compute the addition to the matrix
                        ! element.

                            do Ibas=1,nbasis
                                call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                                dphidz,Ibas)
                                quicktest = DABS(dphidx)+DABS(dphidy)+DABS(dphidz) &
                                +DABS(phi)
                                if (quicktest < quick_method%DMCutoff ) then
                                    continue
                                else
                                    do Jbas=Ibas,nbasis
                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                        dphi2dz,Jbas)
                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                        +DABS(phi2)
                                        if (quicktest < quick_method%DMCutoff ) then
                                            continue
                                        else
                                            temp = phi*phi2
                                            tempgx = phi*dphi2dx + phi2*dphidx
                                            tempgy = phi*dphi2dy + phi2*dphidy
                                            tempgz = phi*dphi2dz + phi2*dphidz
                                            quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                            xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                                        endif
                                    enddo
                                endif
                            enddo
                        endif
                    endif
                enddo
            enddo
        enddo
    enddo
! Finally, copy lower diagonal to upper diagonal.

    do Ibas=1,nbasis
        do Jbas=Ibas+1,nbasis
            quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
        enddo
    enddo

    end subroutine udftoperatorb

