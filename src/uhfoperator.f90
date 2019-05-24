! Ed Brothers. December 20, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine uhfoperatora
       use allmod
       implicit double precision(a - h, o - z)

! The purpose of this subroutine is to form the operator matrix
! for a full Hartree-Fock calculation, i.e. the Fock matrix.  The
! Fock matrix is as follows:

! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
! with each possible basis  - 1/2 exchange with each
! possible basis.

! Note that the Fock matrix is symmetric.

       do Ibas = 1, nbasis
          do Jbas = Ibas, nbasis
             quick_qm_struct%o(Jbas, Ibas) = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)

                   ! The first part is the kinetic energy.

                   quick_qm_struct%o(Jbas, Ibas) = quick_qm_struct%o(Jbas, Ibas) + &
                                                   dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas)* &
                                                   ekinetic(aexp(Jcon, Jbas), aexp(Icon, Ibas), &
                                                            itype(1, Jbas), itype(2, Jbas), itype(3, Jbas), &
                                                            itype(1, Ibas), itype(2, Ibas), itype(3, Ibas), &
                                                            xyz(1, quick_basis%ncenter(Jbas)), xyz(2, quick_basis%ncenter(Jbas)), &
                                                            xyz(3, quick_basis%ncenter(Jbas)), xyz(1, quick_basis%ncenter(Ibas)), &
                                                            xyz(2, quick_basis%ncenter(Ibas)), xyz(3, quick_basis%ncenter(Ibas)))

                   ! Next is a loop over atoms to contruct the attraction terms.

                   do iatom = 1, natom
                      quick_qm_struct%o(Jbas, Ibas) = quick_qm_struct%o(Jbas, Ibas) + &
                                                      dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas)* &
                                                      attraction(aexp(Jcon, Jbas), aexp(Icon, Ibas), &
                                                                 itype(1, Jbas), itype(2, Jbas), itype(3, Jbas), &
                                                                 itype(1, Ibas), itype(2, Ibas), itype(3, Ibas), &
                                                             xyz(1, quick_basis%ncenter(Jbas)), xyz(2, quick_basis%ncenter(Jbas)), &
                                                             xyz(3, quick_basis%ncenter(Jbas)), xyz(1, quick_basis%ncenter(Ibas)), &
                                                             xyz(2, quick_basis%ncenter(Ibas)), xyz(3, quick_basis%ncenter(Ibas)), &
                                                                 xyz(1, iatom), xyz(2, iatom), xyz(3, iatom), &
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
! The next two terms define the two electron part.
       do I = 1, nbasis
          ! Set some variables to reduce access time for some of the more
          ! used quantities.

          xI = xyz(1, quick_basis%ncenter(I))
          yI = xyz(2, quick_basis%ncenter(I))
          zI = xyz(3, quick_basis%ncenter(I))
          itype1I = itype(1, I)
          itype2I = itype(2, I)
          itype3I = itype(3, I)
          DENSEII = quick_qm_struct%dense(I, I) + quick_qm_struct%denseb(I, I)
          DENSEIIX = quick_qm_struct%dense(I, I)

          ! do all the (ii|ii) integrals.
          Ibas = I
          Jbas = I
          IIbas = I
          JJbas = I
          repint = 0.d0
          do Icon = 1, ncontract(ibas)
             do Jcon = 1, ncontract(jbas)
                do IIcon = 1, ncontract(iibas)
                   do JJcon = 1, ncontract(jjbas)
                      repint = repint + &
                               dcoeff(Icon, Ibas)*dcoeff(Jcon, Jbas) &
                               *dcoeff(IIcon, IIbas)*dcoeff(JJcon, JJbas)* &
                               (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                               aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                               itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                               itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                               xI, yI, zI, xI, yI, zI, xI, yI, zI, xI, yI, zI))
                   enddo
                enddo
             enddo
          enddo
          quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + DENSEII*repint
          quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - DENSEIIX*repint

          do J = I + 1, nbasis
             ! Set some variables to reduce access time for some of the more
             ! used quantities. (AGAIN)

             xJ = xyz(1, quick_basis%ncenter(J))
             yJ = xyz(2, quick_basis%ncenter(J))
             zJ = xyz(3, quick_basis%ncenter(J))
             itype1J = itype(1, J)
             itype2J = itype(2, J)
             itype3J = itype(3, J)
             DENSEJI = quick_qm_struct%dense(J, I) + quick_qm_struct%denseb(J, I)
             DENSEJJ = quick_qm_struct%dense(J, J) + quick_qm_struct%denseb(J, J)
             DENSEJIX = quick_qm_struct%dense(J, I)
             DENSEJJX = quick_qm_struct%dense(J, J)

             ! Find  all the (ii|jj) integrals.
             Ibas = I
             Jbas = I
             IIbas = J
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Icon, Ibas)*dcoeff(Jcon, Jbas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                                  itype1J, itype2J, itype3J, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xI, yI, zI, xJ, yJ, zJ, xJ, yJ, zJ))
                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + DENSEJJ*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) + DENSEII*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEJIX*repint

             ! Find  all the (ij|jj) integrals.
             Ibas = I
             Jbas = J
             IIbas = J
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  itype1J, itype2J, itype3J, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xJ, yJ, zJ, xJ, yJ, zJ, xJ, yJ, zJ))

                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + DENSEJJ*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) + 2.d0*DENSEJI*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEJJX*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) - 2.d0*DENSEJIX*repint

             ! Find  all the (ii|ij) integrals.
             Ibas = I
             Jbas = I
             iiBAS = i
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Icon, Ibas)*dcoeff(Jcon, Jbas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xI, yI, zI, xI, yI, zI, xJ, yJ, zJ))

                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + DENSEII*repint
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + 2.d0*DENSEJI*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEIIX*repint
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - 2.d0*DENSEJIX*repint
             ! Find all the (ij|ij) integrals
             Ibas = I
             Jbas = J
             IIbas = I
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xJ, yJ, zJ, xI, yI, zI, xJ, yJ, zJ))

                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + 2.d0*DENSEJI*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) - DENSEIIX*repint
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - DENSEJJX*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEJIX*repint

             do K = J + 1, nbasis
                ! Set some variables to reduce access time for some of the more
                ! used quantities. (AGAIN)

                xK = xyz(1, quick_basis%ncenter(K))
                yK = xyz(2, quick_basis%ncenter(K))
                zK = xyz(3, quick_basis%ncenter(K))
                itype1K = itype(1, K)
                itype2K = itype(2, K)
                itype3K = itype(3, K)
                DENSEKI = quick_qm_struct%dense(K, I) + quick_qm_struct%denseb(K, I)
                DENSEKJ = quick_qm_struct%dense(K, J) + quick_qm_struct%denseb(K, J)
                DENSEKK = quick_qm_struct%dense(K, K) + quick_qm_struct%denseb(K, K)
                DENSEKIX = quick_qm_struct%dense(K, I)
                DENSEKJX = quick_qm_struct%dense(K, J)
                DENSEKKX = quick_qm_struct%dense(K, K)

                ! Find all the (ij|ik) integrals where j>i,k>j
                Ibas = I
                Jbas = J
                IIbas = I
                JJbas = K
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                     itype1I, itype2I, itype3I, itype1K, itype2K, itype3K, &
                                                     xI, yI, zI, xJ, yJ, zJ, xI, yI, zI, xK, yK, zK))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + 2.d0*DENSEKI*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) + 2.d0*DENSEJI*repint
                quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - 2.d0*DENSEKJX*repint
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEKIX*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSEJIX*repint
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSEIIX*repint

                ! Find all the (ij|kk) integrals where j>i, k>j.
                Ibas = I
                Jbas = J
                IIbas = K
                JJbas = K
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                     itype1K, itype2K, itype3K, itype1K, itype2K, itype3K, &
                                                     xI, yI, zI, xJ, yJ, zJ, xK, yK, zK, xK, yK, zK))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + DENSEKK*repint
                quick_qm_struct%o(K, K) = quick_qm_struct%o(K, K) + 2.d0*DENSEJI*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSEKJX*repint
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSEKIX*repint

                ! Find all the (ik|jj) integrals where j>i, k>j.
                Ibas = I
                Jbas = K
                IIbas = J
                JJbas = J
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1K, itype2K, itype3K, &
                                                     itype1J, itype2J, itype3J, itype1J, itype2J, itype3J, &
                                                     xI, yI, zI, xK, yK, zK, xJ, yJ, zJ, xJ, yJ, zJ))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) + DENSEJJ*repint
                quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) + 2.d0*DENSEKI*repint
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSEJIX*repint
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEKJX*repint

                ! Find all the (ii|jk) integrals where j>i, k>j.
                Ibas = I
                Jbas = I
                IIbas = J
                JJbas = K
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                                     itype1J, itype2J, itype3J, itype1K, itype2K, itype3K, &
                                                     xI, yI, zI, xI, yI, zI, xJ, yJ, zJ, xK, yK, zK))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) + DENSEII*repint
                quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + 2.d0*DENSEKJ*repint
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEKIX*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSEJIX*repint
             enddo

             do K = I + 1, nbasis - 1
                xK = xyz(1, quick_basis%ncenter(K))
                yK = xyz(2, quick_basis%ncenter(K))
                zK = xyz(3, quick_basis%ncenter(K))
                itype1K = itype(1, K)
                itype2K = itype(2, K)
                itype3K = itype(3, K)
                DENSEKI = quick_qm_struct%dense(K, I) + quick_qm_struct%denseb(K, I)
                DENSEKJ = quick_qm_struct%dense(K, J) + quick_qm_struct%denseb(K, J)
                DENSEKK = quick_qm_struct%dense(K, K) + quick_qm_struct%denseb(K, K)
                DENSEKIX = quick_qm_struct%dense(K, I)
                DENSEKJX = quick_qm_struct%dense(K, J)
                DENSEKKX = quick_qm_struct%dense(K, K)

                do L = K + 1, nbasis
                   xL = xyz(1, quick_basis%ncenter(L))
                   yL = xyz(2, quick_basis%ncenter(L))
                   zL = xyz(3, quick_basis%ncenter(L))
                   itype1L = itype(1, L)
                   itype2L = itype(2, L)
                   itype3L = itype(3, L)
                   DENSELJ = quick_qm_struct%dense(L, J) + quick_qm_struct%denseb(L, J)
                   DENSELI = quick_qm_struct%dense(L, I) + quick_qm_struct%denseb(L, I)
                   DENSELK = quick_qm_struct%dense(L, K) + quick_qm_struct%denseb(L, K)
                   DENSELJX = quick_qm_struct%dense(L, J)
                   DENSELIX = quick_qm_struct%dense(L, I)
                   DENSELKX = quick_qm_struct%dense(L, K)

                   ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                   ! can be equal.

                   Ibas = I
                   Jbas = J
                   IIbas = K
                   JJbas = L
                   repint = 0.d0
                   do Icon = 1, ncontract(ibas)
                      do Jcon = 1, ncontract(jbas)
                         do IIcon = 1, ncontract(iibas)
                            do JJcon = 1, ncontract(jjbas)
                               repint = repint + &
                                        dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                        *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                        (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                        aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                        itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                        itype1K, itype2K, itype3K, itype1L, itype2L, itype3L, &
                                                        xI, yI, zI, xJ, yJ, zJ, xK, yK, zK, xL, yL, zL))

                            enddo
                         enddo
                      enddo
                   enddo
                   quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + 2.d0*DENSELK*repint
                   quick_qm_struct%o(L, K) = quick_qm_struct%o(L, K) + 2.d0*DENSEJI*repint
                   quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSELJX*repint
                   quick_qm_struct%o(L, I) = quick_qm_struct%o(L, I) - DENSEKJX*repint
                   if (J == K) then
                      quick_qm_struct%o(J, K) = quick_qm_struct%o(J, K) - 2.d0*DENSELIX*repint
                      quick_qm_struct%o(J, L) = quick_qm_struct%o(J, L) - DENSEKIX*repint
                      quick_qm_struct%o(L, J) = quick_qm_struct%o(L, J) - DENSEKIX*repint
                   elseif (J == L) then
                      quick_qm_struct%o(J, L) = quick_qm_struct%o(J, L) - 2.d0*DENSEKIX*repint
                      quick_qm_struct%o(J, K) = quick_qm_struct%o(J, K) - DENSELIX*repint
                      quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSELIX*repint
                   else
                      quick_qm_struct%o(J, K) = quick_qm_struct%o(J, K) - DENSELIX*repint
                      quick_qm_struct%o(J, L) = quick_qm_struct%o(J, L) - DENSEKIX*repint
                      quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSELIX*repint
                      quick_qm_struct%o(L, J) = quick_qm_struct%o(L, J) - DENSEKIX*repint
                   endif
                enddo
             enddo
          enddo
       enddo

       do Ibas = 1, nbasis
          do Jbas = Ibas + 1, nbasis
             quick_qm_struct%o(Ibas, Jbas) = quick_qm_struct%o(Jbas, Ibas)
          enddo
       enddo

    end subroutine uhfoperatora

! CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
! Ed Brothers. December 20, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine uhfoperatorb
       use allmod
       implicit double precision(a - h, o - z)

! The purpose of this subroutine is to form the operator matrix
! for the beta portion of the unrestricted HF erquation.
! Fock matrix is as follows:

! quick_qm_struct%o(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
! with each possible basis  - 1/2 exchange with each
! possible basis.

! Note that the Fock matrix is symmetric.

       do Ibas = 1, nbasis
          do Jbas = Ibas, nbasis
             quick_qm_struct%o(Jbas, Ibas) = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)

                   ! The first part is the kinetic energy.

                   quick_qm_struct%o(Jbas, Ibas) = quick_qm_struct%o(Jbas, Ibas) + &
                                                   dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas)* &
                                                   ekinetic(aexp(Jcon, Jbas), aexp(Icon, Ibas), &
                                                            itype(1, Jbas), itype(2, Jbas), itype(3, Jbas), &
                                                            itype(1, Ibas), itype(2, Ibas), itype(3, Ibas), &
                                                            xyz(1, quick_basis%ncenter(Jbas)), xyz(2, quick_basis%ncenter(Jbas)), &
                                                            xyz(3, quick_basis%ncenter(Jbas)), xyz(1, quick_basis%ncenter(Ibas)), &
                                                            xyz(2, quick_basis%ncenter(Ibas)), xyz(3, quick_basis%ncenter(Ibas)))

                   ! Next is a loop over atoms to contruct the attraction terms.

                   do iatom = 1, natom
                      quick_qm_struct%o(Jbas, Ibas) = quick_qm_struct%o(Jbas, Ibas) + &
                                                      dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas)* &
                                                      attraction(aexp(Jcon, Jbas), aexp(Icon, Ibas), &
                                                                 itype(1, Jbas), itype(2, Jbas), itype(3, Jbas), &
                                                                 itype(1, Ibas), itype(2, Ibas), itype(3, Ibas), &
                                                             xyz(1, quick_basis%ncenter(Jbas)), xyz(2, quick_basis%ncenter(Jbas)), &
                                                             xyz(3, quick_basis%ncenter(Jbas)), xyz(1, quick_basis%ncenter(Ibas)), &
                                                             xyz(2, quick_basis%ncenter(Ibas)), xyz(3, quick_basis%ncenter(Ibas)), &
                                                                 xyz(1, iatom), xyz(2, iatom), xyz(3, iatom), &
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
! The next two terms define the two electron part.

       do I = 1, nbasis

          ! Set some variables to reduce access time for some of the more
          ! used quantities.

          xI = xyz(1, quick_basis%ncenter(I))
          yI = xyz(2, quick_basis%ncenter(I))
          zI = xyz(3, quick_basis%ncenter(I))
          itype1I = itype(1, I)
          itype2I = itype(2, I)
          itype3I = itype(3, I)
          DENSEII = quick_qm_struct%dense(I, I) + quick_qm_struct%denseb(I, I)
          DENSEIIX = quick_qm_struct%denseb(I, I)

          ! do all the (ii|ii) integrals.
          Ibas = I
          Jbas = I
          IIbas = I
          JJbas = I
          repint = 0.d0
          do Icon = 1, ncontract(ibas)
             do Jcon = 1, ncontract(jbas)
                do IIcon = 1, ncontract(iibas)
                   do JJcon = 1, ncontract(jjbas)
                      repint = repint + &
                               dcoeff(Icon, Ibas)*dcoeff(Jcon, Jbas) &
                               *dcoeff(IIcon, IIbas)*dcoeff(JJcon, JJbas)* &
                               (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                               aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                               itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                               itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                               xI, yI, zI, xI, yI, zI, xI, yI, zI, xI, yI, zI))
                   enddo
                enddo
             enddo
          enddo
          quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + DENSEII*repint
          quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - DENSEIIX*repint

          do J = I + 1, nbasis
             ! Set some variables to reduce access time for some of the more
             ! used quantities. (AGAIN)

             xJ = xyz(1, quick_basis%ncenter(J))
             yJ = xyz(2, quick_basis%ncenter(J))
             zJ = xyz(3, quick_basis%ncenter(J))
             itype1J = itype(1, J)
             itype2J = itype(2, J)
             itype3J = itype(3, J)
             DENSEJI = quick_qm_struct%dense(J, I) + quick_qm_struct%denseb(J, I)
             DENSEJJ = quick_qm_struct%dense(J, J) + quick_qm_struct%denseb(J, J)
             DENSEJIX = quick_qm_struct%denseb(J, I)
             DENSEJJX = quick_qm_struct%denseb(J, J)

             ! Find  all the (ii|jj) integrals.
             Ibas = I
             Jbas = I
             IIbas = J
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Icon, Ibas)*dcoeff(Jcon, Jbas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                                  itype1J, itype2J, itype3J, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xI, yI, zI, xJ, yJ, zJ, xJ, yJ, zJ))
                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + DENSEJJ*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) + DENSEII*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEJIX*repint

             ! Find  all the (ij|jj) integrals.
             Ibas = I
             Jbas = J
             IIbas = J
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  itype1J, itype2J, itype3J, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xJ, yJ, zJ, xJ, yJ, zJ, xJ, yJ, zJ))

                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + DENSEJJ*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) + 2.d0*DENSEJI*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEJJX*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) - 2.d0*DENSEJIX*repint

             ! Find  all the (ii|ij) integrals.
             Ibas = I
             Jbas = I
             iiBAS = i
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Icon, Ibas)*dcoeff(Jcon, Jbas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xI, yI, zI, xI, yI, zI, xJ, yJ, zJ))

                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + DENSEII*repint
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + 2.d0*DENSEJI*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEIIX*repint
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - 2.d0*DENSEJIX*repint
             ! Find all the (ij|ij) integrals
             Ibas = I
             Jbas = J
             IIbas = I
             JJbas = J
             repint = 0.d0
             do Icon = 1, ncontract(ibas)
                do Jcon = 1, ncontract(jbas)
                   do IIcon = 1, ncontract(iibas)
                      do JJcon = 1, ncontract(jjbas)
                         repint = repint + &
                                  dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                  *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                  (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                  aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                  xI, yI, zI, xJ, yJ, zJ, xI, yI, zI, xJ, yJ, zJ))

                      enddo
                   enddo
                enddo
             enddo
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + 2.d0*DENSEJI*repint
             quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) - DENSEIIX*repint
             quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - DENSEJJX*repint
             quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEJIX*repint

             do K = J + 1, nbasis
                ! Set some variables to reduce access time for some of the more
                ! used quantities. (AGAIN)

                xK = xyz(1, quick_basis%ncenter(K))
                yK = xyz(2, quick_basis%ncenter(K))
                zK = xyz(3, quick_basis%ncenter(K))
                itype1K = itype(1, K)
                itype2K = itype(2, K)
                itype3K = itype(3, K)
                DENSEKI = quick_qm_struct%dense(K, I) + quick_qm_struct%denseb(K, I)
                DENSEKJ = quick_qm_struct%dense(K, J) + quick_qm_struct%denseb(K, J)
                DENSEKK = quick_qm_struct%dense(K, K) + quick_qm_struct%denseb(K, K)
                DENSEKIX = quick_qm_struct%denseb(K, I)
                DENSEKJX = quick_qm_struct%denseb(K, J)
                DENSEKKX = quick_qm_struct%denseb(K, K)

                ! Find all the (ij|ik) integrals where j>i,k>j
                Ibas = I
                Jbas = J
                IIbas = I
                JJbas = K
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                     itype1I, itype2I, itype3I, itype1K, itype2K, itype3K, &
                                                     xI, yI, zI, xJ, yJ, zJ, xI, yI, zI, xK, yK, zK))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + 2.d0*DENSEKI*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) + 2.d0*DENSEJI*repint
                quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) - 2.d0*DENSEKJX*repint
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEKIX*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSEJIX*repint
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSEIIX*repint

                ! Find all the (ij|kk) integrals where j>i, k>j.
                Ibas = I
                Jbas = J
                IIbas = K
                JJbas = K
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                     itype1K, itype2K, itype3K, itype1K, itype2K, itype3K, &
                                                     xI, yI, zI, xJ, yJ, zJ, xK, yK, zK, xK, yK, zK))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + DENSEKK*repint
                quick_qm_struct%o(K, K) = quick_qm_struct%o(K, K) + 2.d0*DENSEJI*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSEKJX*repint
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSEKIX*repint

                ! Find all the (ik|jj) integrals where j>i, k>j.
                Ibas = I
                Jbas = K
                IIbas = J
                JJbas = J
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1K, itype2K, itype3K, &
                                                     itype1J, itype2J, itype3J, itype1J, itype2J, itype3J, &
                                                     xI, yI, zI, xK, yK, zK, xJ, yJ, zJ, xJ, yJ, zJ))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) + DENSEJJ*repint
                quick_qm_struct%o(J, J) = quick_qm_struct%o(J, J) + 2.d0*DENSEKI*repint
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSEJIX*repint
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEKJX*repint

                ! Find all the (ii|jk) integrals where j>i, k>j.
                Ibas = I
                Jbas = I
                IIbas = J
                JJbas = K
                repint = 0.d0
                do Icon = 1, ncontract(ibas)
                   do Jcon = 1, ncontract(jbas)
                      do IIcon = 1, ncontract(iibas)
                         do JJcon = 1, ncontract(jjbas)
                            repint = repint + &
                                     dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                     *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                     (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                     aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                     itype1I, itype2I, itype3I, itype1I, itype2I, itype3I, &
                                                     itype1J, itype2J, itype3J, itype1K, itype2K, itype3K, &
                                                     xI, yI, zI, xI, yI, zI, xJ, yJ, zJ, xK, yK, zK))

                         enddo
                      enddo
                   enddo
                enddo
                quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) + DENSEII*repint
                quick_qm_struct%o(I, I) = quick_qm_struct%o(I, I) + 2.d0*DENSEKJ*repint
                quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) - DENSEKIX*repint
                quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSEJIX*repint
             enddo

             do K = I + 1, nbasis - 1
                xK = xyz(1, quick_basis%ncenter(K))
                yK = xyz(2, quick_basis%ncenter(K))
                zK = xyz(3, quick_basis%ncenter(K))
                itype1K = itype(1, K)
                itype2K = itype(2, K)
                itype3K = itype(3, K)
                DENSEKI = quick_qm_struct%dense(K, I) + quick_qm_struct%denseb(K, I)
                DENSEKJ = quick_qm_struct%dense(K, J) + quick_qm_struct%denseb(K, J)
                DENSEKK = quick_qm_struct%dense(K, K) + quick_qm_struct%denseb(K, K)
                DENSEKIX = quick_qm_struct%denseb(K, I)
                DENSEKJX = quick_qm_struct%denseb(K, J)
                DENSEKKX = quick_qm_struct%denseb(K, K)

                do L = K + 1, nbasis
                   xL = xyz(1, quick_basis%ncenter(L))
                   yL = xyz(2, quick_basis%ncenter(L))
                   zL = xyz(3, quick_basis%ncenter(L))
                   itype1L = itype(1, L)
                   itype2L = itype(2, L)
                   itype3L = itype(3, L)
                   DENSELJ = quick_qm_struct%dense(L, J) + quick_qm_struct%denseb(L, J)
                   DENSELI = quick_qm_struct%dense(L, I) + quick_qm_struct%denseb(L, I)
                   DENSELK = quick_qm_struct%dense(L, K) + quick_qm_struct%denseb(L, K)
                   DENSELJX = quick_qm_struct%denseb(L, J)
                   DENSELIX = quick_qm_struct%denseb(L, I)
                   DENSELKX = quick_qm_struct%denseb(L, K)

                   ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                   ! can be equal.

                   Ibas = I
                   Jbas = J
                   IIbas = K
                   JJbas = L
                   repint = 0.d0
                   do Icon = 1, ncontract(ibas)
                      do Jcon = 1, ncontract(jbas)
                         do IIcon = 1, ncontract(iibas)
                            do JJcon = 1, ncontract(jjbas)
                               repint = repint + &
                                        dcoeff(Jcon, Jbas)*dcoeff(Icon, Ibas) &
                                        *dcoeff(JJcon, JJbas)*dcoeff(IIcon, IIbas)* &
                                        (repulsion_prim(aexp(Icon, Ibas), aexp(Jcon, Jbas), &
                                                        aexp(IIcon, IIbas), aexp(JJcon, JJbas), &
                                                        itype1I, itype2I, itype3I, itype1J, itype2J, itype3J, &
                                                        itype1K, itype2K, itype3K, itype1L, itype2L, itype3L, &
                                                        xI, yI, zI, xJ, yJ, zJ, xK, yK, zK, xL, yL, zL))

                            enddo
                         enddo
                      enddo
                   enddo
                   quick_qm_struct%o(J, I) = quick_qm_struct%o(J, I) + 2.d0*DENSELK*repint
                   quick_qm_struct%o(L, K) = quick_qm_struct%o(L, K) + 2.d0*DENSEJI*repint
                   quick_qm_struct%o(K, I) = quick_qm_struct%o(K, I) - DENSELJX*repint
                   quick_qm_struct%o(L, I) = quick_qm_struct%o(L, I) - DENSEKJX*repint
                   if (J == K) then
                      quick_qm_struct%o(J, K) = quick_qm_struct%o(J, K) - 2.d0*DENSELIX*repint
                      quick_qm_struct%o(J, L) = quick_qm_struct%o(J, L) - DENSEKIX*repint
                      quick_qm_struct%o(L, J) = quick_qm_struct%o(L, J) - DENSEKIX*repint
                   elseif (J == L) then
                      quick_qm_struct%o(J, L) = quick_qm_struct%o(J, L) - 2.d0*DENSEKIX*repint
                      quick_qm_struct%o(J, K) = quick_qm_struct%o(J, K) - DENSELIX*repint
                      quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSELIX*repint
                   else
                      quick_qm_struct%o(J, K) = quick_qm_struct%o(J, K) - DENSELIX*repint
                      quick_qm_struct%o(J, L) = quick_qm_struct%o(J, L) - DENSEKIX*repint
                      quick_qm_struct%o(K, J) = quick_qm_struct%o(K, J) - DENSELIX*repint
                      quick_qm_struct%o(L, J) = quick_qm_struct%o(L, J) - DENSEKIX*repint
                   endif
                enddo
             enddo
          enddo
       enddo

       do Ibas = 1, nbasis
          do Jbas = Ibas + 1, nbasis
             quick_qm_struct%o(Ibas, Jbas) = quick_qm_struct%o(Jbas, Ibas)
          enddo
       enddo

    end subroutine uhfoperatorb

