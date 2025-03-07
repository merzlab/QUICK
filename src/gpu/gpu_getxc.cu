#if defined(CUDA) || defined(CUDA_MPIV)
  #include "cuda/gpu.h"
#elif defined(HIP) || defined(HIP_MPIV)
  #include "hip/gpu.h"
#endif

#if defined(CEW)
  #include "iface.hpp"
#endif

#include "libxc_dev_funcs.h"

#include "gpu_work_gga_x.cu"
#include "gpu_work_gga_c.cu"
#include "gpu_work_lda.cu"

#if defined(CUDA) || defined(CUDA_MPIV)
static __constant__ gpu_simulation_type devSim_dft;
#elif defined(HIP) || defined(HIP_MPIV)
__constant__ gpu_simulation_type devSim_dft;
#endif

#if defined(DEBUG) || defined(DEBUGTIME)
static float totTime;
#endif

#define LOCUWSSD(A, i1, i2, d1, d2) A[((i1) + (i2) * (d1)) * gridDim.x * blockDim.x]

#define SSW_NUMGRAD_DELTA (2.5E-5)
#define SSW_NUMGRAD_DELTA2 (5.0E-5)
#define RECIP_SSW_NUMGRAD_DELTA2 (20000.0)

#define SSW_POLYFAC1 (3.4179687500)
#define SSW_POLYFAC2 (8.344650268554688)
#define SSW_POLYFAC3 (12.223608791828156)
#define SSW_POLYFAC4 (7.105427357601002)


/*
   Given a point in space, this function calculates the value of basis
   function I and the value of its cartesian derivatives in all three
   derivatives.
*/
__device__ static inline void pteval_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz,
        QUICKDouble * const phi, QUICKDouble * const dphidx, QUICKDouble * const dphidy, QUICKDouble * const dphidz,
        int const * const primf, int const * const primf_counter, int ibas, int ibasp)
{
    // relative coordinates between grid point and basis function I.
    QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
    QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
    QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
    QUICKDouble dist = x1 * x1 + y1 * y1 + z1 * z1;

    QUICKDouble x1i, y1i, z1i;
    QUICKDouble x1imin1, y1imin1, z1imin1;
    QUICKDouble x1iplus1, y1iplus1, z1iplus1;

    int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
    int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
    int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

//    if ( dist <= devSim_dft.sigrad2[ibas]) {
    if (itypex != 0) {
        x1imin1 = pow(x1, itypex - 1);
        x1i = x1imin1 * x1;
        x1iplus1 = x1i * x1;
    } else {
        x1imin1 = 0.0;
        x1i = 1.0;
        x1iplus1 = x1;
    }

    if (itypey != 0) {
        y1imin1 = pow(y1, itypey - 1);
        y1i = y1imin1 * y1;
        y1iplus1 = y1i * y1;
    } else {
        y1imin1 = 0.0;
        y1i = 1.0;
        y1iplus1 = y1;
    }

    if (itypez != 0) {
        z1imin1 = pow(z1, itypez - 1);
        z1i = z1imin1 * z1;
        z1iplus1 = z1i * z1;
    } else {
        z1imin1 = 0.0;
        z1i = 1.0;
        z1iplus1 = z1;
    }

    *phi = 0.0;
    *dphidx = 0.0;
    *dphidy = 0.0;
    *dphidz = 0.0;

//    for (int i = 0; i < devSim_dft.ncontract[ibas - 1]; i++) {
    for (int i = primf_counter[ibasp]; i < primf_counter[ibasp + 1]; i++) {
        int kprim = (int) primf[i];
        QUICKDouble aexp = LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis);
        QUICKDouble tmp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)
            * exp(-aexp * dist);
        aexp *= -2.0;
        *phi += tmp;
        *dphidx += tmp * (aexp * x1iplus1 + (QUICKDouble) itypex * x1imin1);
        *dphidy += tmp * (aexp * y1iplus1 + (QUICKDouble) itypey * y1imin1);
        *dphidz += tmp * (aexp * z1iplus1 + (QUICKDouble) itypez * z1imin1);
    }

    *phi *= x1i * y1i * z1i;
    *dphidx *= y1i * z1i;
    *dphidy *= x1i * z1i;
    *dphidz *= x1i * y1i;
//    }
}


/*  Madu Manathunga 09/10/2019
*/
__device__ static inline QUICKDouble get_unnormalized_weight(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, int iatm)
{
    QUICKDouble uw = 1.0;
    QUICKDouble xiatm, yiatm, ziatm, xjatm, yjatm, zjatm;
    QUICKDouble rig, rjg, rij;

    xiatm = LOC2(devSim_dft.xyz, 0, iatm, 3, devSim_dft.natom);
    yiatm = LOC2(devSim_dft.xyz, 1, iatm, 3, devSim_dft.natom);
    ziatm = LOC2(devSim_dft.xyz, 2, iatm, 3, devSim_dft.natom);

    rig = sqrt(SQR(gridx - xiatm) + SQR(gridy - yiatm) + SQR(gridz - ziatm));

    for (int jatm = 0; jatm < devSim_dft.natom; jatm++) {
        if (jatm != iatm) {
            xjatm = LOC2(devSim_dft.xyz, 0, jatm, 3, devSim_dft.natom);
            yjatm = LOC2(devSim_dft.xyz, 1, jatm, 3, devSim_dft.natom);
            zjatm = LOC2(devSim_dft.xyz, 2, jatm, 3, devSim_dft.natom);

            rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));
            rij = sqrt(SQR(xiatm - xjatm) + SQR(yiatm - yjatm) + SQR(ziatm - zjatm));

            QUICKDouble confocal = (rig - rjg) / rij;

            if (confocal >= 0.64) {
                uw = 0.0;
            } else if (confocal >= -0.64) {
                QUICKDouble frctn = confocal / 0.64;
                QUICKDouble gofconfocal = (35.0 * frctn - 35.0 * CUBE(frctn)
                        + 21.0 * pow(frctn, 5.0) - 5.0 * pow(frctn, 7.0)) / 16.0;
                uw = uw * 0.5 * (1.0 - gofconfocal);
            }
        }
    }

    return uw;
}


__device__ static inline QUICKDouble SSW(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz,
        QUICKDouble* xyz, int atm)
{
    /*
       This subroutie calculates the Scuseria-Stratmann wieghts.  There are
       two conditions that cause the weights to be unity: If there is only
       one atom:
    */
    QUICKDouble ssw;

    if (devSim_dft.natom == 1) {
        ssw = 1.0e0;
        return ssw;
    }

    /*
       Another time the weight is unity is r(iparent,g)<.5*(1-a)*R(i,n)
       where r(iparent,g) is the distance from the parent atom to the grid
       point, a is a parameter (=.64) and R(i,n) is the distance from the
       parent atom to it's nearest neighbor.
    */
    QUICKDouble xparent = LOC2(xyz, 0, atm - 1, 3, devSim_dft.natom);
    QUICKDouble yparent = LOC2(xyz, 1, atm - 1, 3, devSim_dft.natom);
    QUICKDouble zparent = LOC2(xyz, 2, atm - 1, 3, devSim_dft.natom);

    QUICKDouble rig = sqrt(SQR(gridx - xparent) + SQR(gridy - yparent) + SQR(gridz - zparent));

    /* !!!! this part can be done in CPU */
    QUICKDouble distnbor = 1e3;
    for (int i = 0; i < devSim_dft.natom; i++) {
        if (i != atm - 1) {
            QUICKDouble distance = sqrt(SQR(xparent - LOC2(xyz, 0, i, 3, devSim_dft.natom))
                    + SQR(yparent - LOC2(xyz, 1, i, 3, devSim_dft.natom))
                    + SQR(zparent - LOC2(xyz, 2, i, 3, devSim_dft.natom)));
            distnbor = (distnbor<distance)? distnbor: distance;
        }
    }

    if (rig < 0.18 * distnbor) {
        ssw = 1.0e0;
        return ssw;
    }

    /*
       If neither of those are the case, we have to actually calculate the
       weight.  First we must calculate the unnormalized wieght of the grid point
       with respect to the parent atom.

       Step one of calculating the unnormalized weight is finding the confocal
       elliptical coordinate between each cell.  This it the mu with subscripted
       i and j in the paper:
       Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
       1996, pg 213-223.
    */
    QUICKDouble wofparent = 1.0e0; // weight of parents

    for (int jatm = 1; jatm <= devSim_dft.natom; jatm ++) {
        if (jatm != atm && wofparent != 0.0e0) {
            QUICKDouble xjatm = LOC2(xyz, 0, jatm - 1, 3, devSim_dft.natom);
            QUICKDouble yjatm = LOC2(xyz, 1, jatm - 1, 3, devSim_dft.natom);
            QUICKDouble zjatm = LOC2(xyz, 2, jatm - 1, 3, devSim_dft.natom);

            QUICKDouble rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));
            QUICKDouble rij = sqrt(SQR(xparent - xjatm) + SQR(yparent - yjatm) + SQR(zparent - zjatm));
            QUICKDouble confocal = (rig - rjg) / rij;

            if (confocal >= 0.64) {
                wofparent = 0.0e0;
            } else if (confocal >= -0.64e0) {
                QUICKDouble frctn = confocal / 0.64;
                QUICKDouble gofconfocal = (35.0 * frctn - 35.0 * CUBE(frctn)
                        + 21.0 * pow(frctn, 5.0) - 5.0 * pow(frctn, 7.0)) / 16.0;
                wofparent = wofparent * 0.5 * (1.0 - gofconfocal);
            }
        }
    }

    QUICKDouble totalw = wofparent;
    if (wofparent == 0.0e0) {
        ssw = 0.0e0;
        return ssw;
    }

    /*
       Now we have the unnormalized weight of the grid point with regard to the
       parent atom.  Now we have to do this for all other atom pairs to
       normalize the grid weight.
    */

    // !!!! this part should be rewrite
    for (int i = 0; i < devSim_dft.natom; i++) {
        if (i != atm - 1) {
            QUICKDouble xiatm = LOC2(xyz, 0, i, 3, devSim_dft.natom);
            QUICKDouble yiatm = LOC2(xyz, 1, i, 3, devSim_dft.natom);
            QUICKDouble ziatm = LOC2(xyz, 2, i, 3, devSim_dft.natom);

            rig = sqrt(SQR(gridx - xiatm) + SQR(gridy - yiatm) + SQR(gridz - ziatm));
            QUICKDouble wofiatom = 1.0;
            for (int jatm = 1; jatm <= devSim_dft.natom; jatm++) {
                if (jatm != i + 1 && wofiatom != 0.0e0) {
                    QUICKDouble xjatm = LOC2(xyz, 0, jatm - 1, 3, devSim_dft.natom);
                    QUICKDouble yjatm = LOC2(xyz, 1, jatm - 1, 3, devSim_dft.natom);
                    QUICKDouble zjatm = LOC2(xyz, 2, jatm - 1, 3, devSim_dft.natom);

                    QUICKDouble rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));
                    QUICKDouble rij = sqrt(SQR(xiatm - xjatm) + SQR(yiatm - yjatm) + SQR(ziatm - zjatm));
                    QUICKDouble confocal = (rig - rjg) / rij;
                    if (confocal >= 0.64) {
                        wofiatom = 0.0e0;
                    } else if (confocal >= -0.64e0) {
                        QUICKDouble frctn = confocal / 0.64;
                        QUICKDouble gofconfocal = (35.0 * frctn - 35.0 * CUBE(frctn)
                                + 21.0 * pow(frctn, 5.0) - 5.0 * pow(frctn, 7.0)) / 16.0;
                        wofiatom = wofiatom * 0.5 * (1.0 - gofconfocal);
                    }
                }
            }
            totalw = totalw + wofiatom;
        }
    }
    ssw = wofparent / totalw;

    return ssw;
}


__device__ static inline QUICKDouble SSW(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz,
        QUICKDouble *xyz, QUICKDouble xparent, QUICKDouble yparent, QUICKDouble zparent,
        QUICKDouble xatom, QUICKDouble yatom, QUICKDouble zatom,
        int iatom, int iparent, unsigned int natom)
{
    /*
       This subroutie calculates the Scuseria-Stratmann wieghts.  There are
       two conditions that cause the weights to be unity: If there is only
       one atom:
    */
    QUICKDouble ssw;
    if (natom == 1) {
        ssw = 1.0e0;
        return ssw;
    }

    /*
       Another time the weight is unity is r(iparent,g)<.5*(1-a)*R(i,n)
       where r(iparent,g) is the distance from the parent atom to the grid
       point, a is a parameter (=.64) and R(i,n) is the distance from the
       parent atom to it's nearest neighbor.
    */
    QUICKDouble rig = sqrt(SQR(gridx - xparent) + SQR(gridy - yparent) + SQR(gridz - zparent));

    // compute the ptr to reduce pointer arithmatic inside loops
    QUICKDouble *distance_ptr = devSim_dft.distance + (iparent - 1) * natom;

    /* !!!! this part can be done in CPU*/
    QUICKDouble distnbor = 1e3;
    for (int i = 0; i < natom; i++) {
        if (i != iparent - 1) {
            QUICKDouble distance = distance_ptr[i];

            if (i == iatom) {
                distance = sqrt(SQR(xparent - xatom) + SQR(yparent - yatom) + SQR(zparent - zatom));
            }

            distnbor = (distnbor < distance) ? distnbor: distance;
        }
    }

    if (rig < 0.18 * distnbor) {
        ssw = 1.0e0;
        return ssw;
    }

    /*
       If neither of those are the case, we have to actually calculate the
       weight.  First we must calculate the unnormalized wieght of the grid point
       with respect to the parent atom.

       Step one of calculating the unnormalized weight is finding the confocal
       elliptical coordinate between each cell.  This it the mu with subscripted
       i and j in the paper:
       Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
       1996, pg 213-223.

       Here reduce the MUL operations by precomputing polynomial constants in eqn 14.
       constant of the first term, 3.4179687500 = 35.0 * (1/0.64) * (1/16)
       constant of the second term, 8.344650268554688 = 35.0 * (1/0.64)^3 * (1/16)
       constant of the third term, 12.223608791828156 = 21.0 * (1/0.64)^5 * (1/16)
       constant of the fourth term, 7.105427357601002 = 5.0 * (1/0.64)^7 * (1/16)
    */

    bool bComputeDistance = false;

    if (iatom == iparent - 1) {
        bComputeDistance = true;
    }

    QUICKDouble wofparent = 1.0e0; // weight of parents

    for (int jatm = 0; jatm < natom; jatm++)
    {
        if (jatm != iparent - 1 && wofparent != 0.0e0) {
            QUICKDouble xjatm = LOC2(xyz, 0, jatm, 3, natom);
            QUICKDouble yjatm = LOC2(xyz, 1, jatm, 3, natom);
            QUICKDouble zjatm = LOC2(xyz, 2, jatm, 3, natom);

            QUICKDouble rij = distance_ptr[jatm];

            if (jatm == iatom) {
                xjatm = xatom;
                yjatm = yatom;
                zjatm = zatom;

                bComputeDistance = true;
            }

            QUICKDouble rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));

            if (bComputeDistance) {
                rij = sqrt(SQR(xparent - xjatm) + SQR(yparent - yjatm) + SQR(zparent - zjatm));
            }

            QUICKDouble confocal = (rig - rjg) * (1.0 / rij);

            if (confocal >= 0.64) {
                wofparent = 0.0e0;
            } else if (confocal >= -0.64e0) {
                //QUICKDouble frctn = confocal * 1.5625;
                //QUICKDouble frctn3 = frctn * frctn * frctn;
                //QUICKDouble gofconfocal = (35.0 * frctn - 35.0 * frctn3 +21.0 * frctn * frctn * frctn3 - 5.0 * frctn * frctn3 * frctn3) * 0.062500;
                QUICKDouble confocal3 = CUBE(confocal);
                QUICKDouble gofconfocal = SSW_POLYFAC1 * confocal - SSW_POLYFAC2 * confocal3
                    + SSW_POLYFAC3 * confocal * confocal * confocal3
                    - SSW_POLYFAC4 * confocal * confocal3 * confocal3;

                wofparent = wofparent * 0.5 * (1.0 - gofconfocal);
            }
        }
    }

    QUICKDouble totalw = wofparent;
    if (wofparent == 0.0e0) {
        ssw = 0.0e0;
        return ssw;
    }

    /*
       Now we have the unnormalized weight of the grid point with regard to the
       parent atom.  Now we have to do this for all other atom pairs to
       normalize the grid weight.
    */
    // !!!! this part should be rewrite
    for (int i = 0; i < natom; i++) {
        if (i != iparent - 1) {
            QUICKDouble xiatm = LOC2(xyz, 0, i, 3, natom);
            QUICKDouble yiatm = LOC2(xyz, 1, i, 3, natom);
            QUICKDouble ziatm = LOC2(xyz, 2, i, 3, natom);

            bComputeDistance = false;
            distance_ptr = devSim_dft.distance + i * natom;

            if (i == iatom) {
                xiatm = xatom;
                yiatm = yatom;
                ziatm = zatom;
                bComputeDistance = true;
            }

            rig = sqrt(SQR(gridx - xiatm) + SQR(gridy - yiatm) + SQR(gridz - ziatm));

            QUICKDouble wofiatom = 1.0;
            for (int jatm = 0; jatm < natom; jatm++) {
                if (jatm != i && wofiatom != 0.0e0) {
                    QUICKDouble xjatm = LOC2(xyz, 0, jatm, 3, natom);
                    QUICKDouble yjatm = LOC2(xyz, 1, jatm, 3, natom);
                    QUICKDouble zjatm = LOC2(xyz, 2, jatm, 3, natom);

                    bool bComputeDistance2 = false;

                    QUICKDouble rij = distance_ptr[jatm];

                    if (jatm == iatom) {
                        xjatm = xatom;
                        yjatm = yatom;
                        zjatm = zatom;

                        bComputeDistance2 = true;
                    }

                    QUICKDouble rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));

                    if (bComputeDistance || bComputeDistance2) {
                        rij = sqrt(SQR(xiatm - xjatm) + SQR(yiatm - yjatm) + SQR(ziatm - zjatm));
                    }

                    QUICKDouble confocal = (rig - rjg) * (1.0 / rij);

                    if (confocal >= 0.64) {
                        wofiatom = 0.0;
                    } else if (confocal >= -0.64) {
                        //QUICKDouble frctn = confocal * 1.5625;
                        //QUICKDouble frctn3 = frctn * frctn * frctn;
                        //QUICKDouble gofconfocal = (35.0 * frctn - 35.0 * frctn3 + 21.0 * frctn * frctn * frctn3 - 5.0 * frctn * frctn3 * frctn3) * 0.062500;
                        QUICKDouble confocal3 = CUBE(confocal);
                        QUICKDouble gofconfocal = SSW_POLYFAC1 * confocal - SSW_POLYFAC2 * confocal3
                            + SSW_POLYFAC3 * confocal * confocal * confocal3
                            - SSW_POLYFAC4 * confocal * confocal3 * confocal3;

                        wofiatom = wofiatom * 0.5 * (1.0 - gofconfocal);
                    }
                }
            }
            totalw = totalw + wofiatom;
        }
    }
    ssw = wofparent / totalw;

    return ssw;
}


__device__ static inline QUICKDouble get_uw_ssd(const QUICKDouble gridx, const QUICKDouble gridy, const QUICKDouble gridz,
        QUICKDouble* const uw_ssd, const int iparent, const int iatm, const int natom, const bool grad=true)
{
    QUICKDouble xiatm = LOC2(devSim_dft.xyz, 0, iatm, 3, natom);
    QUICKDouble yiatm = LOC2(devSim_dft.xyz, 1, iatm, 3, natom);
    QUICKDouble ziatm = LOC2(devSim_dft.xyz, 2, iatm, 3, natom);

    QUICKDouble rig = sqrt(SQR(gridx - xiatm) + SQR(gridy - yiatm) + SQR(gridz - ziatm));

    QUICKDouble uw = 1.0;

    for (int jatm=0; jatm < natom; jatm++) {
        if (jatm != iatm) {
            QUICKDouble xjatm = LOC2(devSim_dft.xyz, 0, jatm, 3, natom);
            QUICKDouble yjatm = LOC2(devSim_dft.xyz, 1, jatm, 3, natom);
            QUICKDouble zjatm = LOC2(devSim_dft.xyz, 2, jatm, 3, natom);

            QUICKDouble rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));
            QUICKDouble rij = sqrt(SQR(xiatm - xjatm) + SQR(yiatm - yjatm) + SQR(ziatm - zjatm));

            QUICKDouble mu = (rig - rjg) * (1.0 / rij);


            if (mu <= -0.64) {

            } else if (mu >= 0.64) {
                uw = 0.0;
            } else {
                QUICKDouble mu3 = CUBE(mu);
                QUICKDouble g = SSW_POLYFAC1 * mu - SSW_POLYFAC2 * mu3
                    + SSW_POLYFAC3 * mu3 * mu * mu - SSW_POLYFAC4 * mu3 * mu3 * mu;
                QUICKDouble s = 0.5 * (1.0 - g);
                uw = uw * s;

                if (grad) {
                    QUICKDouble dsdmu = -0.5 * (SSW_POLYFAC1 - 3.0 * SSW_POLYFAC2 * mu * mu
                            + 5.0 * SSW_POLYFAC3 * mu3 * mu - 7.0 * SSW_POLYFAC4 * mu3 * mu3);

                    QUICKDouble recip_rij3 = 1.0 / (rij * rij * rij);
                    QUICKDouble dmudi = (xiatm - gridx) * (1.0 / (rig * rij))
                        - (rig - rjg) * recip_rij3 * (xiatm - xjatm);
                    QUICKDouble dmudj = (gridx - xjatm) * (1.0 / (rjg * rij))
                        + (rig - rjg) * recip_rij3 * (xiatm - xjatm);
                    QUICKDouble dmudg = ((gridx - xiatm) * (1.0 / rig)
                            + (xjatm - gridx) * (1.0 / rjg)) * (1.0 / rij);

                    LOCUWSSD(uw_ssd, 0, iatm, 3, natom) += dsdmu * dmudi / s;
                    LOCUWSSD(uw_ssd, 0, jatm, 3, natom) += dsdmu * dmudj / s;
                    LOCUWSSD(uw_ssd, 0, iparent - 1, 3, natom) += dsdmu * dmudg / s;

                    dmudi = (yiatm - gridy) * (1.0 / (rig * rij))
                        - (rig - rjg) * recip_rij3 * (yiatm - yjatm);
                    dmudj = (gridy - yjatm) * (1.0 / (rjg * rij))
                        + (rig - rjg) * recip_rij3 * (yiatm - yjatm);
                    dmudg = ((gridy - yiatm) * (1.0 / rig)
                            + (yjatm - gridy) * (1.0 / rjg)) * (1.0 / rij);

                    LOCUWSSD(uw_ssd, 1, iatm, 3, natom) += dsdmu * dmudi / s;
                    LOCUWSSD(uw_ssd, 1, jatm, 3, natom) += dsdmu * dmudj / s;
                    LOCUWSSD(uw_ssd, 1, iparent - 1, 3, natom) += dsdmu * dmudg / s;

                    dmudi = (ziatm - gridz) * (1.0 / (rig * rij))
                        - (rig - rjg) * recip_rij3 * (ziatm - zjatm);
                    dmudj = (gridz - zjatm) * (1.0 / (rjg * rij))
                        + (rig - rjg) * recip_rij3 * (ziatm - zjatm);
                    dmudg = ((gridz - ziatm) * (1.0 / rig)
                            + (zjatm - gridz) * (1.0 / rjg)) * (1.0 / rij);

                    LOCUWSSD(uw_ssd, 2, iatm, 3, natom) += dsdmu * dmudi / s;
                    LOCUWSSD(uw_ssd, 2, jatm, 3, natom) += dsdmu * dmudj / s;
                    LOCUWSSD(uw_ssd, 2, iparent - 1, 3, natom) += dsdmu * dmudg / s;
                }
            }
        }
    }

    return uw;
}


__device__ static inline void sswanader(const QUICKDouble gridx, const QUICKDouble gridy, const QUICKDouble gridz,
        const QUICKDouble Exc, const QUICKDouble quadwt, 
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL* const smemGrad,
#else
        QUICKDouble* const smemGrad,
#endif
        QUICKDouble* const uw_ssd, const int iparent, const int natom)
{
    QUICKDouble sumUW = 0.0;
    QUICKDouble parent_uw = 0.0;

    for (int iatm = 0; iatm < natom; iatm++) {
        QUICKDouble uw = get_uw_ssd(gridx, gridy, gridz, uw_ssd, iparent, iatm, natom, false);
        sumUW += uw;

        if (iatm == iparent - 1) {
            parent_uw = uw;
        }
    }

    QUICKDouble p = parent_uw / sumUW;

    for (int iatm = 0; iatm < natom; iatm++) {
        for (int i = 0; i < natom; i++) {
            for (int j = 0; j < 3; j++) {
                LOCUWSSD(uw_ssd, j, i, 3, natom) = 0.0;
            }
        }

        QUICKDouble uw = get_uw_ssd(gridx, gridy, gridz, uw_ssd, iparent, iatm, natom);

        for (int i = 0; i < natom; i++) {
            for (int j = 0; j < 3; j++) {
                if (abs(uw) > devSim_dft.DMCutoff) {
                    GPUATOMICADD(&smemGrad[i * 3 + j],
                            LOCUWSSD(uw_ssd, j, i, 3, natom) * Exc * quadwt * uw * (-p / sumUW), GRADSCALE);
                }
            }
        }
    }

    for (int i = 0; i < natom; i++) {
        for (int j = 0; j < 3; j++) {
            LOCUWSSD(uw_ssd, j, i, 3, natom) = 0.0;
        }
    }

    get_uw_ssd(gridx, gridy, gridz, uw_ssd, iparent, iparent - 1, natom);

    for (int i = 0; i < natom; i++) {
        for (int j = 0; j < 3; j++) {
            if (abs(parent_uw) > devSim_dft.DMCutoff) {
                GPUATOMICADD(&smemGrad[i * 3 + j],
                        LOCUWSSD(uw_ssd, j, i, 3, natom) * (1.0 / sumUW) * Exc * quadwt * parent_uw, GRADSCALE);
            }
        }
    }
}


__device__ static inline void sswder(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz,
        QUICKDouble Exc, QUICKDouble quadwt, 
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL* smemGrad,
#else
        QUICKDouble* smemGrad,
#endif
        int iparent, int gid)
{
    /*
       This subroutine calculates the derivatives of weight found in
       Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
       1996, pg 213-223.  The mathematical development is similar to
       the developement found in the papers of Gill and Pople, but uses
       SSW method rather than Becke's weights.

       The derivative of the weight with repsect to the displacement of the
       parent atom is mathematically quite complicated, thus rotational
       invariance is used.  Rotational invariance simply is the condition that
       the sum of all derivatives must be zero.

       Certain things will be needed again and again in this subroutine.  We
       are therefore goint to store them.  They are the unnormalized weights.
       The array is called UW for obvious reasons.
    */
#if defined(DEBUG)
//    printf("gridx: %f  gridy: %f  gridz: %f, exc: %.10e, quadwt: %.10e, iparent: %i \n", gridx, gridy, gridz, Exc, quadwt, iparent);
#endif

    QUICKDouble sumUW = 0.0;
    QUICKDouble xiatm, yiatm, ziatm, xjatm, yjatm, zjatm, xlatm, ylatm, zlatm;
    QUICKDouble rig, rjg, rlg, rij, rjl;

    for (int iatm = 0; iatm < devSim_dft.natom; iatm++) {
        QUICKDouble tmp_uw = get_unnormalized_weight(gridx, gridy, gridz, iatm);
        devSim_dft.uw_ssd[gid * devSim_dft.natom + iatm] = tmp_uw;
        sumUW = sumUW + tmp_uw;
    }

#if defined(DEBUG)
//    printf("gridx: %f  gridy: %f  gridz: %f, exc: %.10e, quadwt: %.10e, iparent: %i, sumUW: %.10e \n",
//            gridx, gridy, gridz, Exc, quadwt, iparent, sumUW);
#endif

    /*
       At this point we now have the unnormalized weight and the sum of same.
       Calculate the parent atom - grid point distance, and then start the loop.
    */
    xiatm = LOC2(devSim_dft.xyz, 0, iparent - 1, 3, devSim_dft.natom);
    yiatm = LOC2(devSim_dft.xyz, 1, iparent - 1, 3, devSim_dft.natom);
    ziatm = LOC2(devSim_dft.xyz, 2, iparent - 1, 3, devSim_dft.natom);

    rig = sqrt(SQR(gridx - xiatm) + SQR(gridy - yiatm) + SQR(gridz - ziatm));

    QUICKDouble a = 0.64;

    int istart = (iparent - 1) * 3;

    QUICKDouble wtgradix = 0.0;
    QUICKDouble wtgradiy = 0.0;
    QUICKDouble wtgradiz = 0.0;

    for (int jatm = 0; jatm < devSim_dft.natom; jatm++) {
        int jstart = jatm * 3;

        QUICKDouble wtgradjx = 0.0;
        QUICKDouble wtgradjy = 0.0;
        QUICKDouble wtgradjz = 0.0;

        if (jatm != iparent - 1) {
            xjatm = LOC2(devSim_dft.xyz, 0, jatm, 3, devSim_dft.natom);
            yjatm = LOC2(devSim_dft.xyz, 1, jatm, 3, devSim_dft.natom);
            zjatm = LOC2(devSim_dft.xyz, 2, jatm, 3, devSim_dft.natom);

            rjg = sqrt(SQR(gridx - xjatm) + SQR(gridy - yjatm) + SQR(gridz - zjatm));
            rij = sqrt(SQR(xiatm - xjatm) + SQR(yiatm - yjatm) + SQR(ziatm - zjatm));

            QUICKDouble dmudx = (-1.0 / rij) * (1.0 / rjg) * (xjatm - gridx) + ((rig - rjg) / CUBE(rij)) * (xiatm - xjatm);
            QUICKDouble dmudy = (-1.0 / rij) * (1.0 / rjg) * (yjatm - gridy) + ((rig - rjg) / CUBE(rij)) * (yiatm - yjatm);
            QUICKDouble dmudz = (-1.0 / rij) * (1.0 / rjg) * (zjatm - gridz) + ((rig - rjg) / CUBE(rij)) * (ziatm - zjatm);

            QUICKDouble u = (rig - rjg) / rij;
            QUICKDouble t = (-35.0 * CUBE(a + u)) / ((a - u) * (16.0 * CUBE(a)
                        + 29.0 * SQR(a) * u + 20.0 * a * SQR(u) + 5.0 * CUBE(u)));
            //QUICKDouble uw_iparent = get_unnormalized_weight(gridx, gridy, gridz, iparent - 1);
            QUICKDouble uw_iparent = devSim_dft.uw_ssd[gid * devSim_dft.natom + (iparent - 1)];
            //QUICKDouble uw_jatm = get_unnormalized_weight(gridx, gridy, gridz, jatm);
            QUICKDouble uw_jatm = devSim_dft.uw_ssd[gid * devSim_dft.natom + jatm];

            wtgradjx += uw_iparent * dmudx * t / sumUW;
            wtgradjy += uw_iparent * dmudy * t / sumUW;
            wtgradjz += uw_iparent * dmudz * t / sumUW;

#if defined(DEBUG)
//            printf("gridx: %f  gridy: %f  gridz: %f \n", wtgradjx, wtgradjy, wtgradjz);
#endif

            for (int latm = 0; latm < devSim_dft.natom; latm++) {
                if (latm != jatm) {
                    xlatm = LOC2(devSim_dft.xyz, 0, latm, 3, devSim_dft.natom);
                    ylatm = LOC2(devSim_dft.xyz, 1, latm, 3, devSim_dft.natom);
                    zlatm = LOC2(devSim_dft.xyz, 2, latm, 3, devSim_dft.natom);

                    rlg = sqrt(SQR(gridx - xlatm) + SQR(gridy - ylatm) + SQR(gridz - zlatm));
                    rjl = sqrt(SQR(xjatm - xlatm) + SQR(yjatm - ylatm) + SQR(zjatm - zlatm));

                    dmudx = (-1.0 / rjl) * (1.0 / rjg) * (xjatm - gridx) + ((rlg - rjg) / CUBE(rjl)) * (xlatm - xjatm);
                    dmudy = (-1.0 / rjl) * (1.0 / rjg) * (yjatm - gridy) + ((rlg - rjg) / CUBE(rjl)) * (ylatm - yjatm);
                    dmudz = (-1.0 / rjl) * (1.0 / rjg) * (zjatm - gridz) + ((rlg - rjg) / CUBE(rjl)) * (zlatm - zjatm);

                    u = (rlg - rjg) / rjl;
                    t = (-35.0 * CUBE(a + u)) / ((a - u) * (16.0 * CUBE(a) + 29.0 * SQR(a) * u + 20.0 * a * SQR(u) + 5.0 * CUBE(u)));
                    //QUICKDouble uw_latm = get_unnormalized_weight(gridx, gridy, gridz, latm);
                    QUICKDouble uw_latm = devSim_dft.uw_ssd[gid * devSim_dft.natom + latm];
                    wtgradjx -= uw_latm * uw_iparent * dmudx * t / SQR(sumUW);
                    wtgradjy -= uw_latm * uw_iparent * dmudy * t / SQR(sumUW);
                    wtgradjz -= uw_latm * uw_iparent * dmudz * t / SQR(sumUW);
                }
            }

#if defined(DEBUG)
//            printf("gridx: %f  gridy: %f  gridz: %f \n", wtgradjx, wtgradjy, wtgradjz);

#endif
            for (int latm = 0; latm < devSim_dft.natom; latm++) {
                if (latm != jatm) {
                    xlatm = LOC2(devSim_dft.xyz, 0, latm, 3, devSim_dft.natom);
                    ylatm = LOC2(devSim_dft.xyz, 1, latm, 3, devSim_dft.natom);
                    zlatm = LOC2(devSim_dft.xyz, 2, latm, 3, devSim_dft.natom);

                    rlg = sqrt(SQR(gridx - xlatm) + SQR(gridy - ylatm) + SQR(gridz - zlatm));
                    rjl = sqrt(SQR(xjatm - xlatm) + SQR(yjatm - ylatm) + SQR(zjatm - zlatm));

                    dmudx = (-1.0 / rjl) * (1.0 / rlg) * (xlatm - gridx) + ((rjg - rlg) / CUBE(rjl)) * (xjatm - xlatm);
                    dmudy = (-1.0 / rjl) * (1.0 / rlg) * (ylatm - gridy) + ((rjg - rlg) / CUBE(rjl)) * (yjatm - ylatm);
                    dmudz = (-1.0 / rjl) * (1.0 / rlg) * (zlatm - gridz) + ((rjg - rlg) / CUBE(rjl)) * (zjatm - zlatm);

                    u = (rjg - rlg) / rjl;
                    t = (-35.0 * CUBE(a + u)) / ((a - u) * (16.0 * CUBE(a) + 29.0 * SQR(a) * u + 20.0 * a * SQR(u) + 5.0 * CUBE(u)));

                    wtgradjx += uw_jatm * uw_iparent * dmudx * t / SQR(sumUW);
                    wtgradjy += uw_jatm * uw_iparent * dmudy * t / SQR(sumUW);
                    wtgradjz += uw_jatm * uw_iparent * dmudz * t / SQR(sumUW);
                }
            }

#if defined(DEBUG)
//            printf("gridx: %f  gridy: %f  gridz: %f \n", wtgradjx, wtgradjy, wtgradjz);
#endif

            // Now do the rotational invariance part of the derivatives.
            wtgradix -= wtgradjx;
            wtgradiy -= wtgradjy;
            wtgradiz -= wtgradjz;

#if defined(DEBUG)
//            printf("gridx: %f  gridy: %f  gridz: %f Exc: %e quadwt: %e\n", wtgradjx, wtgradjy, wtgradjz, Exc, quadwt);
#endif

            // We should now have the derivatives of the SS weights.  Now just add it to the temporary gradient vector in shared memory.
            GPUATOMICADD(&smemGrad[jstart], wtgradjx * Exc * quadwt, GRADSCALE);
            GPUATOMICADD(&smemGrad[jstart + 1], wtgradjy * Exc * quadwt, GRADSCALE);
            GPUATOMICADD(&smemGrad[jstart + 2], wtgradjz * Exc * quadwt, GRADSCALE);
        }
    }

#if defined(DEBUG)
//    printf("istart: %i  gridx: %f  gridy: %f  gridz: %f Exc: %e quadwt: %e\n", istart, wtgradix, wtgradiy, wtgradiz, Exc, quadwt);
#endif

    // update the temporary gradient vector
    GPUATOMICADD(&smemGrad[istart], wtgradix * Exc * quadwt, GRADSCALE);
    GPUATOMICADD(&smemGrad[istart + 1], wtgradiy * Exc * quadwt, GRADSCALE);
    GPUATOMICADD(&smemGrad[istart + 2], wtgradiz * Exc * quadwt, GRADSCALE);
}


__device__ static inline void pt2der_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz,
        QUICKDouble* dxdx, QUICKDouble* dxdy, QUICKDouble* dxdz,
        QUICKDouble* dydy, QUICKDouble* dydz, QUICKDouble* dzdz,
        int *primf, int *primf_counter, int ibas, int ibasp)
{
    /* Given a point in space, this function calculates the value of basis
     * function I and the value of its cartesian derivatives in all three derivatives.
     * */

    // relative coordinates between grid point and basis function I.
    QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
    QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
    QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
    QUICKDouble dist = x1 * x1 + y1 * y1 + z1 * z1;

    QUICKDouble x1i, y1i, z1i;
    QUICKDouble x1imin1, y1imin1, z1imin1, x1imin2, y1imin2, z1imin2;
    QUICKDouble x1iplus1, y1iplus1, z1iplus1, x1iplus2, y1iplus2, z1iplus2;

    *dxdx = 0.0;
    *dxdy = 0.0;
    *dxdz = 0.0;
    *dydy = 0.0;
    *dydz = 0.0;
    *dzdz = 0.0;

    int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
    int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
    int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

//    if (dist <= devSim_dft.sigrad2[ibas]) {
    if (itypex == 1) {
        x1imin1 = 1.0;
        x1imin2 = 0.0;
        x1i = x1;
        x1iplus1 = x1 * x1;
        x1iplus2 = x1iplus1 * x1;
    } else if (itypex > 1) {
        x1imin2 = pow(x1, itypex - 2);
        x1imin1 = x1imin2 * x1;
        x1i = x1imin1 * x1;
        x1iplus1 = x1i * x1;
        x1iplus2 = x1iplus1 * x1;
    } else {
        x1imin1 = 0.0;
        x1imin2 = 0.0;
        x1i = 1.0;
        x1iplus1 = x1;
        x1iplus2 = x1 * x1;
    }

    if (itypey == 1) {
        y1imin1 = 1.0;
        y1imin2 = 0.0;
        y1i = y1;
        y1iplus1 = y1 * y1;
        y1iplus2 = y1iplus1 * y1;
    } else if (itypey > 1) {
        y1imin2 = pow(y1, itypey - 2);
        y1imin1 = y1imin2 * y1;
        y1i = y1imin1 * y1;
        y1iplus1 = y1i * y1;
        y1iplus2 = y1iplus1 * y1;
    } else {
        y1imin1 = 0.0;
        y1imin2 = 0.0;
        y1i = 1.0;
        y1iplus1 = y1;
        y1iplus2 = y1 * y1;
    }

    if (itypez == 1) {
        z1imin1 = 1.0;
        z1imin2 = 0.0;
        z1i = z1;
        z1iplus1 = z1 * z1;
        z1iplus2 = z1iplus1 * z1;
    } else if (itypez > 1) {
        z1imin2 = pow(z1, itypez - 2);
        z1imin1 = z1imin2 * z1;
        z1i = z1imin1 * z1;
        z1iplus1 = z1i * z1;
        z1iplus2 = z1iplus1 * z1;
    } else {
        z1imin1 = 0.0;
        z1imin2 = 0.0;
        z1i = 1.0;
        z1iplus1 = z1;
        z1iplus2 = z1 * z1;
    }

//    for (int i = 0; i < devSim_dft.ncontract[ibas - 1]; i++) {
    for (int i = primf_counter[ibasp]; i < primf_counter[ibasp + 1]; i++) {
        int kprim = (int) primf[i];
        QUICKDouble aexp = LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis);
        QUICKDouble temp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)
            * exp(-aexp * dist);
        QUICKDouble twoA = 2.0 * aexp;
        QUICKDouble fourAsqr = twoA * twoA;

        *dxdx += temp * ((QUICKDouble) itypex * ((QUICKDouble) itypex - 1.0) * x1imin2
                - twoA * (2.0 * (QUICKDouble) itypex + 1.0) * x1i
                + fourAsqr * x1iplus2);

        *dydy += temp * ((QUICKDouble) itypey * ((QUICKDouble) itypey - 1.0) * y1imin2
                - twoA * (2.0 * (QUICKDouble) itypey + 1.0) * y1i
                + fourAsqr * y1iplus2);

        *dzdz += temp * ((QUICKDouble) itypez * ((QUICKDouble) itypez - 1.0) * z1imin2
                - twoA * (2.0 * (QUICKDouble) itypez + 1.0) * z1i
                + fourAsqr * z1iplus2);

        *dxdy += temp * ((QUICKDouble) itypex * (QUICKDouble) itypey * x1imin1 * y1imin1
                - twoA * (QUICKDouble) itypex * x1imin1 * y1iplus1
                - twoA * (QUICKDouble) itypey * x1iplus1 * y1imin1
                + fourAsqr * x1iplus1 * y1iplus1);

        *dxdz += temp * ((QUICKDouble) itypex * (QUICKDouble) itypez * x1imin1 * z1imin1
                - twoA * (QUICKDouble) itypex * x1imin1 * z1iplus1
                - twoA * (QUICKDouble) itypez * x1iplus1 * z1imin1
                + fourAsqr * x1iplus1 * z1iplus1);

        *dydz += temp * ((QUICKDouble) itypey * (QUICKDouble) itypez * y1imin1 * z1imin1
                - twoA * (QUICKDouble) itypey * y1imin1 * z1iplus1
                - twoA * (QUICKDouble) itypez * y1iplus1 * z1imin1
                + fourAsqr * y1iplus1 * z1iplus1);
    }

    *dxdx *= y1i * z1i;
    *dydy *= x1i * z1i;
    *dzdz *= x1i * y1i;
    *dxdy *= z1i;
    *dxdz *= y1i;
    *dydz *= x1i;
//    }
}


// Given the densities and the two gradients, return the energy.
__device__ static inline QUICKDouble becke_e(QUICKDouble density, QUICKDouble densityb,
        QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
        QUICKDouble gbx, QUICKDouble gby, QUICKDouble gbz)
{
    /*
       Becke exchange functional
       __  ->
       |\/ rho|
       s =    ---------
       rou^(4/3)int jstart = jatm*3;
       -
       Ex = E(LDA)x[rou(r)] -  |  Fx[s]rou^(4/3) dr
       -

       1             s^2
       Fx[s] = b 2^(- -) ------------------------
       3      1 + 6bs sinh^(-1) s
    */
    QUICKDouble const b = 0.0042;

    /*
       __  ->
       |\/ rho|
       s =    ---------
       rou^(4/3)
    */
    QUICKDouble x = sqrt(gax * gax + gay * gay + gaz * gaz) / pow(density, 4.0 / 3.0);

    QUICKDouble e = pow(density, 4.0 / 3.0) * (-0.93052573634910002500 - b * x * x
            / (1.0 + 0.0252 * x * log(x + sqrt(x * x + 1.0))));

    if (densityb != 0.0) {
        QUICKDouble rhob4thirds = pow(densityb, 4.0 / 3.0);
        QUICKDouble xb = sqrt(gbx * gbx + gby * gby + gbz * gbz) / rhob4thirds;
        QUICKDouble gofxb = -0.93052573634910002500 - b * xb * xb / (1.0 + 6.0 * 0.0042 * xb * log(xb + sqrt(xb * xb + 1.0)));
        e += rhob4thirds * gofxb;
    }

    return e;
}


// Given either density and the two gradients, (with gother being for
// the spin that density is not, i.e. beta if density is alpha) return
// the derivative of beckes 1988 functional with regard to the density
// and the derivatives with regard to the gradient invariants.
//
// Example:  If becke() is passed the alpha density and the alpha and beta
// density gradients, return the derivative of f with regard to the alpha
//denisty, the alpha-alpha gradient invariant, and the alpha beta gradient
// invariant.
__device__ static inline void becke(QUICKDouble density,
        QUICKDouble gx, QUICKDouble gy, QUICKDouble gz,
        QUICKDouble gotherx, QUICKDouble gothery, QUICKDouble gotherz,
        QUICKDouble* dfdr, QUICKDouble* dfdgg, QUICKDouble* dfdggo)
{
    /*
       Becke exchange functional
       __  ->
       |\/ rho|
       s = ---------
       rou^(4/3)
       -
       Ex = E(LDA)x[rou(r)] -  |  Fx[s]rou^(4/3) dr
       -

       1             s^2
       Fx[s] = b 2^(- -) ------------------------
       3      1 + 6bs sinh^(-1) s
    */

    //QUICKDouble fourPi= 4 * PI;
    QUICKDouble b = 0.0042;
    *dfdggo = 0.0;

    QUICKDouble rhothirds = cbrt(density);
    QUICKDouble rho4thirds = SQR(SQR(rhothirds));
    QUICKDouble x = sqrt(gx * gx + gy * gy + gz * gz) / rho4thirds;
    QUICKDouble arcsinhx = log(x + sqrt(x * x + 1.0));
    QUICKDouble denom = 1.0 + 6.0 * b * x * arcsinhx;
    QUICKDouble gofx = -0.930525736349100025000 - b * x * x / denom;
    QUICKDouble gprimeofx = (6.0 * b * b * x * x * (x / sqrt(x * x + 1.0) - arcsinhx) - 2.0 * b * x) / (denom * denom);

    *dfdr = (4.0 / 3.0) * rhothirds * (gofx - x * gprimeofx);
    *dfdgg = 0.5 * gprimeofx / (x * rho4thirds);
}


// Given the densites and the two gradients, (with gother being for
// the spin that density is not, i.e. beta if density is alpha) return
// the derivative of the lyp correlation functional with regard to the density
// and the derivatives with regard to the gradient invariants.
__device__ static inline void lyp(QUICKDouble pa, QUICKDouble pb,
        QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
        QUICKDouble gbx, QUICKDouble gby, QUICKDouble gbz,
        QUICKDouble* dfdr, QUICKDouble* dfdgg, QUICKDouble* dfdggo)
{
    QUICKDouble a = 0.049180;
    QUICKDouble b = 0.1320;
    QUICKDouble c = 0.25330;
    QUICKDouble d = 0.3490;
    QUICKDouble CF = 0.30 * pow(3.0 * PI * PI, 2.0 / 3.0);

    // And some required quanitities.

    QUICKDouble gaa = gax * gax + gay * gay + gaz * gaz;
    QUICKDouble gbb = gbx * gbx + gby * gby + gbz * gbz;
    QUICKDouble gab = gax * gbx + gay * gby + gaz * gbz;
    QUICKDouble ptot = pa + pb;
    QUICKDouble ptone3rd = cbrt(ptot);
    QUICKDouble third2 = 2.0 / 3.0;

    QUICKDouble w = exp(-c / ptone3rd) * pow(ptone3rd, -11.0) / (1.0 + d / ptone3rd);

    QUICKDouble abw = a * b * w;
    QUICKDouble abwpapb = abw * pa * pb;
    QUICKDouble dabw = (a * b * exp(-c / ptone3rd) * (c * d * SQR(ptone3rd) + (c - 10.0 * d) * ptot - 11.0 * SQR(SQR(ptone3rd))))
        / (3.0 * pow(ptone3rd, 16.0) * SQR(d + ptone3rd));

    QUICKDouble dabwpapb = (a * b * exp(-c / ptone3rd) * pb * (c * d * pa * SQR(ptone3rd)
                + (SQR(SQR(ptone3rd)) * (-8.0 * pa + 3.0 * pb)
                    + ptot * (c * pa - 7.0 * d * pa + 3.0 * d * pb))))
        / (3.0 * (pow(ptone3rd, 16.0)) * SQR(d + ptone3rd));

    QUICKDouble delta = c / ptone3rd + (d / ptone3rd)/ (1.0 + d / ptone3rd);

    *dfdr = - dabwpapb * CF * 12.6992084 * (pow(pa, 8.0 / 3.0) + pow(pb, 8.0 / 3.0))
        - dabwpapb * (47.0 / 18.0 - 7.0 * delta / 18.0) * (gaa + gbb + 2.0 * gab)
        + dabwpapb * (2.50 - delta / 18.0) * (gaa + gbb)
        + dabwpapb * ((delta - 11.0) / 9.0) * (pa * gaa / ptot + pb * gbb / ptot)
        + dabw * (third2 * ptot * ptot * (2.0 * gab) + pa * pa * gbb + pb * pb * gaa)
        + (-4.0 * a * pb * (3.0 * pb * cbrt(ptot) + d * (pa + 3.0 * pb))) / \
        //--------------------------------------------------------------------
        (3.0 * pow(ptot, 5.0 / 3.0) * SQR(d + cbrt(ptot))) \
        + (-64.0 * pow(2.0, third2) * abwpapb * CF * pow(pa, 5.0 / 3.0)) / 3.0
        - (7.0 * abwpapb * (gaa + 2.0 * gab + gbb) * (c + (d * pow(ptot, third2)) / SQR(d + cbrt(ptot)))) / \
        //-----------------------------------------------------------------------------------------------------
        (54.0 * pow(ptot, 4.0 / 3.0)) \
        + (abwpapb * (gaa + gbb) * (c + (d * pow(ptot, third2)) / SQR(d + cbrt((ptot))))) / \
        //-----------------------------------------------------------------------------------------------------
        (54.0 * pow(ptot, 4.0 / 3.0))
        + (abwpapb * (-30.0 * d * d * (gaa - gbb) * pb * cbrt(ptot)
                    -33.0 * (gaa - gbb) * pb * ptot
                    - c * (gaa * (pa - 3.0 * pb) + 4.0 * gbb * pb) * SQR(d + cbrt(ptot))
                    - d * pow((pa + pb), third2) * (-62.0 * gbb * pb + gaa * (pa + 63.0 * pb)))) / \
        //-----------------------------------------------------------------------------------------------------
        (27.0 * pow(ptot, 7.0 / 3.0) * SQR(d + cbrt(ptot)))
        + (4.0 * abw * (gaa + 2.0 * gab + gbb) * (ptot)) / 3.0
        + (2.0 * abw * gbb * (pa - 2.0 * pb)) / 3.0
        + (-4.0 * abw * gaa * (ptot)) / 3.0;

    *dfdgg = 0.0 - abwpapb * (47.0 / 18.0 - 7.0 * delta / 18.0) + abwpapb * (2.50 - delta / 18.0)
        + abwpapb * ((delta - 11.0) / 9.0) * (pa / ptot) + abw * (2.0 / 3.0) * ptot * ptot - abw * ((2.0 / 3.0) * ptot * ptot - pb * pb);

    *dfdggo= 0.0 - abwpapb * (47.0 / 18.0 - 7.0 * delta / 18.0) * 2.0 + abw * (2.0 / 3.0) * ptot * ptot * 2.0;
}


// Given the densities and the two gradients, return the energy, return
// the LYP correlation energy.
__device__ static inline QUICKDouble lyp_e(QUICKDouble pa, QUICKDouble pb,
        QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
        QUICKDouble gbx, QUICKDouble gby, QUICKDouble gbz)
{
    // Note the code is kind of garbled, as Mathematic was used to write it.

    /*
       Lee-Yang-Parr correction functional
       _        -a
       E(LYP) = |  rou [ -----[1+b*Cf*exp(-cx)]]dr
       -         1+dx
       _    __            exp(-cx)   1        7        dx
       + |  ab|\/rou|^2*x^5 ---------- --- [ 1 + -(cx + ------)]dr
       -                      1+dx     24       3       1+dx
       3           2
       where Cf = --(3 PI^2)^(--)
       10           3
       a  = 0.04918
       b  = 0.132
       c  = 0.2533
       d  = 0.349
       x  = rou^(1/3)
    */
    QUICKDouble const a = 0.04918;
    QUICKDouble const b = 0.132;
    QUICKDouble const c = 0.2533;
    QUICKDouble const d = 0.349;
    QUICKDouble const CF = 0.3 * pow((3.0 * PI * PI), 2.0 / 3.0);

    // And some required quanitities.
    QUICKDouble gaa = gax * gax + gay * gay + gaz * gaz;
    QUICKDouble gbb = gbx * gbx + gby * gby + gbz * gbz;
    QUICKDouble gab = gax * gbx + gay * gby + gaz * gbz;
    QUICKDouble ptot = pa + pb;
    QUICKDouble ptone3rd = cbrt(ptot);

    QUICKDouble t1 = d / ptone3rd;
    QUICKDouble t2 = 1.0 + t1;
    QUICKDouble w = exp(-c / ptone3rd) * pow(ptone3rd, -11.0) / t2;
    QUICKDouble abw = a * b * w;
    QUICKDouble abwpapb = abw * pa * pb;
    QUICKDouble delta = c / ptone3rd + t1 / (1.0 + t1);
    QUICKDouble const c1 = pow(2.0, 11.0 / 3.0);

    QUICKDouble e = - 4.0 * a * pa * pb / (ptot * t2)
        - abwpapb * CF * c1 * (pow(pa, 8.0 / 3.0) + pow(pb, 8.0 / 3.0))
        - abwpapb * (47.0 / 18.0 - 7.0 * delta / 18.0) * (gaa + gbb + 2.0 * gab)
        + abwpapb * (2.50 - delta/18.0) * (gaa + gbb)
        + abwpapb * ((delta - 11.0) / 9.0) * (pa * gaa / ptot + pb * gbb / ptot)
        + abw * (2.0 / 3.0) * ptot * ptot * (gaa + gbb + 2.0 * gab)
        - abw * ((2.0 / 3.0) * ptot * ptot - pa * pa) * gbb
        - abw * ((2.0 / 3.0) * ptot * ptot - pb * pb) * gaa;

    return e;
}


/*
   P.J. Stephens, F.J. Devlin, C.F. Chabalowski, M.J. Frisch
   Ab initio calculation of vibrational absorption and circular
   dichroism spectra using density functional force fields
   J. Phys. Chem. 98 (1994) 11623-11627

   CITATION:
   Functionals were obtained from the Density Functional Repository
   as developed and distributed by the Quantum Chemistry Group,
   CCLRC Daresbury Laboratory, Daresbury, Cheshire, WA4 4AD
   United Kingdom. Contact Huub van Dam (h.j.j.vandam@dl.ac.uk) or
   Paul Sherwood for further information.

   COPYRIGHT:

   Users may incorporate the source code into software packages and
   redistribute the source code provided the source code is not
   changed in anyway and is properly cited in any documentation or
   publication related to its use.

   ACKNOWLEDGEMENT:

   The source code was generated using Maple 8 through a modified
   version of the dfauto script published in:

   R. Strange, F.R. Manby, P.J. Knowles
   Automatic code generation in density functional theory
   Comp. Phys. Comm. 136 (2001) 310-318.
*/
__device__ static inline QUICKDouble b3lyp_e(QUICKDouble rho, QUICKDouble sigma)
{
    QUICKDouble Eelxc;
    QUICKDouble t2 = cbrt(rho);
    QUICKDouble t3 = t2 * rho;
    QUICKDouble t5 = 1.0 / t3;
    QUICKDouble t7 = sqrt(sigma);
    QUICKDouble t8 = t7 * t5;
    QUICKDouble t10 = log(0.1259921049894873e1 * t8 + sqrt(1.0 + 0.1587401051968199e1 * t8 * t8));
    QUICKDouble t17 = 1.0 / t2;
    QUICKDouble t20 = 1.0 / (1.e0 + 0.349e0 * t17);
    QUICKDouble t23 = 0.2533 * t17;
    QUICKDouble t24 = exp(-t23);
    QUICKDouble t26 = rho * rho;
    QUICKDouble t28 = t2 * t2;
    QUICKDouble t34 = t17 * t20;
    QUICKDouble t56 = 1.0 / rho;
    QUICKDouble t57 = cbrt(t56);
    QUICKDouble t59 = pow(t56, 1.0 / 6.0);
    QUICKDouble t62 = 1.0 / (0.6203504908994 * t57 + 0.1029581201158544e2 * t59 + 0.427198e2);
    QUICKDouble t65 = log(0.6203504908994 * t57 * t62);
    QUICKDouble t71 = atan(0.448998886412873e-1 / (0.1575246635799487e1 * t59 + 0.13072e2));
    QUICKDouble t75 = SQR(0.7876233178997433e0 * t59 + 0.409286e0);
    QUICKDouble t77 = log(t75 * t62);
    Eelxc = -0.5908470131056179e0 * t3
        - 0.3810001254882096e-2 * t5 * sigma / (1.e0 + 0.317500104573508e-1 * t8 * t10)
        - 0.398358e-1 * t20 * rho
        - 0.52583256e-2 * t24 * t20 / (t28 * t26 * rho) * (0.25e0 * t26 * (0.1148493600075277e2 * t28 * t26
                    + (0.2611111111111111e1 - 0.9850555555555556e-1 * t17 - 0.1357222222222222e0 * t34) * sigma
                    - 0.5 * (0.25e1 - 0.1407222222222222e-1 * t17 - 0.1938888888888889e-1 * t34) * sigma
                    - 0.2777777777777778e-1 * (t23 + 0.349 * t34 - 11.0) * sigma) - 0.4583333333333333e0 * t26 * sigma)
        + 0.19 * rho * (0.310907e-1 * t65 + 0.205219729378375e2 * t71 + 0.4431373767749538e-2 * t77);

    return Eelxc;
}


/*
   P.J. Stephens, F.J. Devlin, C.F. Chabalowski, M.J. Frisch
   Ab initio calculation of vibrational absorption and circular
   dichroism spectra using density functional force fields
   J. Phys. Chem. 98 (1994) 11623-11627

   CITATION:
   Functionals were obtained from the Density Functional Repository
   as developed and distributed by the Quantum Chemistry Group,
   CCLRC Daresbury Laboratory, Daresbury, Cheshire, WA4 4AD
   United Kingdom. Contact Huub van Dam (h.j.j.vandam@dl.ac.uk) or
   Paul Sherwood for further information.

   COPYRIGHT:

   Users may incorporate the source code into software packages and
   redistribute the source code provided the source code is not
   changed in anyway and is properly cited in any documentation or
   publication related to its use.

   ACKNOWLEDGEMENT:

   The source code was generated using Maple 8 through a modified
   version of the dfauto script published in:

   R. Strange, F.R. Manby, P.J. Knowles
   Automatic code generation in density functional theory
   Comp. Phys. Comm. 136 (2001) 310-318.
*/
__device__ static inline QUICKDouble b3lypf(QUICKDouble rho, QUICKDouble sigma, QUICKDouble* dfdr)
{
    QUICKDouble dot;
    QUICKDouble t2 = cbrt(rho);
    QUICKDouble t3 = t2 * rho;
    QUICKDouble t5 = 1.0 / t3;
    QUICKDouble t6 = t5 * sigma;
    QUICKDouble t7 = sqrt(sigma);
    QUICKDouble t8 = t7 * t5;
    QUICKDouble t10 = log(0.1259921049894873e1 * t8 + sqrt(1.0 + 0.1587401051968199e1 * t8 * t8));
    QUICKDouble t13 = 1.0e0 + 0.317500104573508e-1 * t8 * t10;
    QUICKDouble t14 = 1.0e0 / t13;
    QUICKDouble t17 = 1.0 / t2;
    QUICKDouble t19 = 1.e0 + 0.349e0 * t17;
    QUICKDouble t20 = 1.0 / t19;
    QUICKDouble t23 = 0.2533e0 * t17;
    QUICKDouble t24 = exp(-t23);
    QUICKDouble t25 = t24 * t20;
    QUICKDouble t26 = rho * rho;
    QUICKDouble t28 = t2 * t2;
    QUICKDouble t30 = 1.0 / (t28 * t26 * rho);
    QUICKDouble t31 = t28 * t26;
    QUICKDouble t34 = t17 * t20;

    QUICKDouble t36 = 0.2611111111111111e1 - 0.9850555555555556e-1 * t17 - 0.1357222222222222e0 * t34;
    QUICKDouble t44 = t23 + 0.349e0 * t34 - 11.e0;
    QUICKDouble t47 = 0.1148493600075277e2 * t31 + t36 * sigma - 0.5e0 * (0.25e1 - 0.1407222222222222e-1 * t17
            - 0.1938888888888889e-1 * t34) * sigma - 0.2777777777777778e-1 * t44 * sigma;

    QUICKDouble t52 = 0.25e0 * t26 * t47 - 0.4583333333333333e0 * t26 * sigma;
    QUICKDouble t56 = 1.0 / rho;
    QUICKDouble t57 = cbrt(t56);
    QUICKDouble t59 = pow(t56, 1.0 / 6.0);
    QUICKDouble t61 = 0.6203504908994e0 * t57 + 0.1029581201158544e2 * t59 + 0.427198e2;
    QUICKDouble t62 = 1.0 / t61;
    QUICKDouble t65 = log(0.6203504908994e0 * t57 * t62);
    QUICKDouble t68 = 0.1575246635799487e1 * t59 + 0.13072e2;
    QUICKDouble t71 = atan(0.448998886412873e-1 / t68);
    QUICKDouble t74 = 0.7876233178997433e0 * t59 + 0.409286e0;
    QUICKDouble t75 = t74 * t74;
    QUICKDouble t77 = log(t75 * t62);
    QUICKDouble t84 = 1.0 / (t2 * t26);
    QUICKDouble t88 = t13 * t13;
    QUICKDouble t89 = 1.0 / t88;
    QUICKDouble t94 = 1.0 / t31;
    QUICKDouble t98 = sqrt(1.e0 + 0.1587401051968199e1 * sigma * t94);
    QUICKDouble t99 = 1.0 / t98;
    QUICKDouble t109 = t28 * rho;
    QUICKDouble t112 = t44 * t56 * sigma;
    QUICKDouble t117 = rho * sigma;
    QUICKDouble t123 = t19 * t19;
    QUICKDouble t124 = 1.0 / t123;
    QUICKDouble t127 = t26 * t26;
    QUICKDouble t129 = 1.0 / (t127 * rho);
    QUICKDouble t144 = t5 * t20;
    QUICKDouble t146 = 1.0 / t109;
    QUICKDouble t147 = t146 * t124;
    QUICKDouble t175 = t57 * t57;
    QUICKDouble t176 = 1.0 / t175;
    QUICKDouble t178 = 1.0 / t26;
    QUICKDouble t181 = t61 * t61;
    QUICKDouble t182 = 1.0 / t181;
    QUICKDouble t186 = t59 * t59;
    QUICKDouble t187 = t186 * t186;
    QUICKDouble t189 = 1.0 / (t187 * t59);
    QUICKDouble t190 = t189 * t178;
    QUICKDouble t192 = -0.2067834969664667e0 * t176 * t178 - 0.1715968668597574e1 * t190;
    QUICKDouble t200 = t68 * t68;
    QUICKDouble t201 = 1.0 / t200;

    *dfdr = -0.7877960174741572e0 * t2
        + 0.5080001673176129e-2 * t84 * sigma * t14
        + 0.1905000627441048e-2 * t6 * t89 * (-0.8466669455293548e-1 * t7*t84 * t10 - 0.106673350692263e0 * sigma * t30 * t99)
        - 0.398358e-1 * t20
        - 0.52583256e-2 * t25 * t30 * (0.5e0 * rho * t47 + 0.25e0 * t26 * (0.3062649600200738e2 * t109 - 0.2777777777777778e-1 * t112)
                - 0.25e0 * t117)
        - 0.46342314e-2 * t124 * t17
        - 0.44397795816e-3 * t129 * t24 * t20 * t52;

    *dfdr += -0.6117185448e-3 * t24 * t124 * t129 * t52
        + 0.192805272e-1 * t25 / t28 / t127 * t52
        - 0.52583256e-2 * t25 * t30 * (0.25e0 * t26 * ((0.3283518518518519e-1 * t5 + 0.4524074074074074e-1 * t144
                        - 0.1578901851851852e-1 * t147) * sigma
                    - 0.5e0 * (0.4690740740740741e-2 * t5 + 0.6462962962962963e-2 * t144 - 0.2255574074074074e-2 * t147) * sigma
                - 0.2777777777777778e-1 * (-0.8443333333333333e-1 * t5 - 0.1163333333333333e0 * t144 + 0.4060033333333333e-1 * t147) * sigma
                + 0.2777777777777778e-1 * t112) - 0.6666666666666667e0 * t117);

    *dfdr += 0.5907233e-2 * t65
        + 0.3899174858189126e1 * t71
        + 0.8419610158724123e-3 * t77
        + 0.19e0 * rho * (0.5011795824473985e-1 * (-0.2067834969664667e0 * t176 * t62 * t178
                    - 0.6203504908994e0 * t57 * t182 * t192) / t57 * t61
                + 0.2419143800947354e0 * t201 * t189 * t178 / (1.e0 + 0.2016e-2 * t201)
                + 0.4431373767749538e-2 * (-0.2625411059665811e0 * t74 * t62 * t190 - 1.e0 * t75 * t182 * t192) / t75 * t61);

    dot = -0.1524000501952839e-1 * t5 * t14
        + 0.3810001254882096e-2 * t6 * t89 * (0.6350002091470161e-1 / t7 * t5 * t10 + 0.8000501301919725e-1 * t94 * t99)
        + 0.5842584e-3 * t25 * t146
        - 0.210333024e-1 * t25*t30 * (0.25e0 * t26 * t36 - 0.6666666666666667e0 * t26);

    return dot;
}


/*
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
   ! w
   ! w    Given a point on a sphere (specified by a and b), generate all
   ! w    the equivalent points under Oh symmetry, making grid points with
   ! w    weight v.
   ! w    The variable num is increased by the number of different points
   ! w    generated.
   ! w
   ! w    Depending on code, there are 6...48 different but equivalent
   ! w    points.
   ! w
   ! w    code=1:   (0,0,1) etc                                (  6 points)
   ! w    code=2:   (0,a,a) etc, a=1/sqrt(2)                   ( 12 points)
   ! w    code=3:   (a,a,a) etc, a=1/sqrt(3)                   (  8 points)
   ! w    code=4:   (a,a,b) etc, b=sqrt(1-2 a^2)               ( 24 points)
   ! w    code=5:   (a,b,0) etc, b=sqrt(1-a^2), a input        ( 24 points)
   ! w    code=6:   (a,b,c) etc, c=sqrt(1-a^2-b^2), a/b input  ( 48 points)
   ! w
*/
__device__ static inline int gen_oh(int code, int num, QUICKDouble* x, QUICKDouble* y, QUICKDouble* z,
        QUICKDouble* w, QUICKDouble a, QUICKDouble b, QUICKDouble v)
{
    QUICKDouble c;
    switch (code) {
        case 1:
            {
                a = 1.0e0;
                x[0 + num] =      a; y[0 + num] =  0.0e0; z[0 + num] =  0.0e0; w[0 + num] =  v;
                x[1 + num] =     -a; y[1 + num] =  0.0e0; z[1 + num] =  0.0e0; w[1 + num] =  v;
                x[2 + num] =  0.0e0; y[2 + num] =      a; z[2 + num] =  0.0e0; w[2 + num] =  v;
                x[3 + num] =  0.0e0; y[3 + num] =     -a; z[3 + num] =  0.0e0; w[3 + num] =  v;
                x[4 + num] =  0.0e0; y[4 + num] =  0.0e0; z[4 + num] =      a; w[4 + num] =  v;
                x[5 + num] =  0.0e0; y[5 + num] =  0.0e0; z[5 + num] =     -a; w[5 + num] =  v;
                num = num + 6;
                break;
            }
        case 2:
            {
                a = sqrt(0.5);
                x[0  + num] =  0e0;  y[0  + num] =    a;  z[0  + num] =    a;  w[0  + num] =  v;
                x[1  + num] =  0e0;  y[1  + num] =   -a;  z[1  + num] =    a;  w[1  + num] =  v;
                x[2  + num] =  0e0;  y[2  + num] =    a;  z[2  + num] =   -a;  w[2  + num] =  v;
                x[3  + num] =  0e0;  y[3  + num] =   -a;  z[3  + num] =   -a;  w[3  + num] =  v;
                x[4  + num] =    a;  y[4  + num] =  0e0;  z[4  + num] =    a;  w[4  + num] =  v;
                x[5  + num] =   -a;  y[5  + num] =  0e0;  z[5  + num] =    a;  w[5  + num] =  v;
                x[6  + num] =    a;  y[6  + num] =  0e0;  z[6  + num] =   -a;  w[6  + num] =  v;
                x[7  + num] =   -a;  y[7  + num] =  0e0;  z[7  + num] =   -a;  w[7  + num] =  v;
                x[8  + num] =    a;  y[8  + num] =    a;  z[8  + num] =  0e0;  w[8  + num] =  v;
                x[9  + num] =   -a;  y[9  + num] =    a;  z[9  + num] =  0e0;  w[9  + num] =  v;
                x[10 + num] =    a;  y[10 + num] =   -a;  z[10 + num] =  0e0;  w[10 + num] =  v;
                x[11 + num] =   -a;  y[11 + num] =   -a;  z[11 + num] =  0e0;  w[11 + num] =  v;
                num = num + 12;
                break;
            }
        case 3:
            {
                a = sqrt(1.0 / 3.0);
                x[0 + num] =  a;  y[0 + num] =  a;  z[0 + num] =  a;  w[0 + num] =  v;
                x[1 + num] = -a;  y[1 + num] =  a;  z[1 + num] =  a;  w[1 + num] =  v;
                x[2 + num] =  a;  y[2 + num] = -a;  z[2 + num] =  a;  w[2 + num] =  v;
                x[3 + num] = -a;  y[3 + num] = -a;  z[3 + num] =  a;  w[3 + num] =  v;
                x[4 + num] =  a;  y[4 + num] =  a;  z[4 + num] = -a;  w[4 + num] =  v;
                x[5 + num] = -a;  y[5 + num] =  a;  z[5 + num] = -a;  w[5 + num] =  v;
                x[6 + num] =  a;  y[6 + num] = -a;  z[6 + num] = -a;  w[6 + num] =  v;
                x[7 + num] = -a;  y[7 + num] = -a;  z[7 + num] = -a;  w[7 + num] =  v;
                num = num + 8;
                break;
            }
        case 4:
            {
                b = sqrt(1.0 - 2.0 * a * a);
                x[0  + num] =  a;  y[0  + num] =  a;  z[0  + num] =  b;  w[0  + num] =  v;
                x[1  + num] = -a;  y[1  + num] =  a;  z[1  + num] =  b;  w[1  + num] =  v;
                x[2  + num] =  a;  y[2  + num] = -a;  z[2  + num] =  b;  w[2  + num] =  v;
                x[3  + num] = -a;  y[3  + num] = -a;  z[3  + num] =  b;  w[3  + num] =  v;
                x[4  + num] =  a;  y[4  + num] =  a;  z[4  + num] = -b;  w[4  + num] =  v;
                x[5  + num] = -a;  y[5  + num] =  a;  z[5  + num] = -b;  w[5  + num] =  v;
                x[6  + num] =  a;  y[6  + num] = -a;  z[6  + num] = -b;  w[6  + num] =  v;
                x[7  + num] = -a;  y[7  + num] = -a;  z[7  + num] = -b;  w[7  + num] =  v;
                x[8  + num] =  a;  y[8  + num] =  b;  z[8  + num] =  a;  w[8  + num] =  v;
                x[9  + num] = -a;  y[9  + num] =  b;  z[9  + num] =  a;  w[9  + num] =  v;
                x[10 + num] =  a;  y[10 + num] = -b;  z[10 + num] =  a;  w[10 + num] =  v;
                x[11 + num] = -a;  y[11 + num] = -b;  z[11 + num] =  a;  w[11 + num] =  v;
                x[12 + num] =  a;  y[12 + num] =  b;  z[12 + num] = -a;  w[12 + num] =  v;
                x[13 + num] = -a;  y[13 + num] =  b;  z[13 + num] = -a;  w[13 + num] =  v;
                x[14 + num] =  a;  y[14 + num] = -b;  z[14 + num] = -a;  w[14 + num] =  v;
                x[15 + num] = -a;  y[15 + num] = -b;  z[15 + num] = -a;  w[15 + num] =  v;
                x[16 + num] =  b;  y[16 + num] =  a;  z[16 + num] =  a;  w[16 + num] =  v;
                x[17 + num] = -b;  y[17 + num] =  a;  z[17 + num] =  a;  w[17 + num] =  v;
                x[18 + num] =  b;  y[18 + num] = -a;  z[18 + num] =  a;  w[18 + num] =  v;
                x[19 + num] = -b;  y[19 + num] = -a;  z[19 + num] =  a;  w[19 + num] =  v;
                x[20 + num] =  b;  y[20 + num] =  a;  z[20 + num] = -a;  w[20 + num] =  v;
                x[21 + num] = -b;  y[21 + num] =  a;  z[21 + num] = -a;  w[21 + num] =  v;
                x[22 + num] =  b;  y[22 + num] = -a;  z[22 + num] = -a;  w[22 + num] =  v;
                x[23 + num] = -b;  y[23 + num] = -a;  z[23 + num] = -a;  w[23 + num] =  v;
                num = num + 24;
                break;
            }
        case 5:
            {
                b=sqrt(1.0 - a * a);
                x[0  + num] =    a;  y[0  + num] =    b;  z[0  + num] =  0e0;  w[0  + num] =  v;
                x[1  + num] =   -a;  y[1  + num] =    b;  z[1  + num] =  0e0;  w[1  + num] =  v;
                x[2  + num] =    a;  y[2  + num] =   -b;  z[2  + num] =  0e0;  w[2  + num] =  v;
                x[3  + num] =   -a;  y[3  + num] =   -b;  z[3  + num] =  0e0;  w[3  + num] =  v;
                x[4  + num] =    b;  y[4  + num] =    a;  z[4  + num] =  0e0;  w[4  + num] =  v;
                x[5  + num] =   -b;  y[5  + num] =    a;  z[5  + num] =  0e0;  w[5  + num] =  v;
                x[6  + num] =    b;  y[6  + num] =   -a;  z[6  + num] =  0e0;  w[6  + num] =  v;
                x[7  + num] =   -b;  y[7  + num] =   -a;  z[7  + num] =  0e0;  w[7  + num] =  v;
                x[8  + num] =    a;  y[8  + num] =  0e0;  z[8  + num] =    b;  w[8  + num] =  v;
                x[9  + num] =   -a;  y[9  + num] =  0e0;  z[9  + num] =    b;  w[9  + num] =  v;
                x[10 + num] =    a;  y[10 + num] =  0e0;  z[10 + num] =   -b;  w[10 + num] =  v;
                x[11 + num] =   -a;  y[11 + num] =  0e0;  z[11 + num] =   -b;  w[11 + num] =  v;
                x[12 + num] =    b;  y[12 + num] =  0e0;  z[12 + num] =    a;  w[12 + num] =  v;
                x[13 + num] =   -b;  y[13 + num] =  0e0;  z[13 + num] =    a;  w[13 + num] =  v;
                x[14 + num] =    b;  y[14 + num] =  0e0;  z[14 + num] =   -a;  w[14 + num] =  v;
                x[15 + num] =   -b;  y[15 + num] =  0e0;  z[15 + num] =   -a;  w[15 + num] =  v;
                x[16 + num] =  0e0;  y[16 + num] =    a;  z[16 + num] =    b;  w[16 + num] =  v;
                x[17 + num] =  0e0;  y[17 + num] =   -a;  z[17 + num] =    b;  w[17 + num] =  v;
                x[18 + num] =  0e0;  y[18 + num] =    a;  z[18 + num] =   -b;  w[18 + num] =  v;
                x[19 + num] =  0e0;  y[19 + num] =   -a;  z[19 + num] =   -b;  w[19 + num] =  v;
                x[20 + num] =  0e0;  y[20 + num] =    b;  z[20 + num] =    a;  w[20 + num] =  v;
                x[21 + num] =  0e0;  y[21 + num] =   -b;  z[21 + num] =    a;  w[21 + num] =  v;
                x[22 + num] =  0e0;  y[22 + num] =    b;  z[22 + num] =   -a;  w[22 + num] =  v;
                x[23 + num] =  0e0;  y[23 + num] =   -b;  z[23 + num] =   -a;  w[23 + num] =  v;
                num = num + 24;
                break;
            }
        case 6:
            {
                c = sqrt(1.0 - a * a - b * b);
                x[0  + num] =  a;  y[0  + num] =  b;  z[0  + num] =  c;  w[0  + num] =  v;
                x[1  + num] = -a;  y[1  + num] =  b;  z[1  + num] =  c;  w[1  + num] =  v;
                x[2  + num] =  a;  y[2  + num] = -b;  z[2  + num] =  c;  w[2  + num] =  v;
                x[3  + num] = -a;  y[3  + num] = -b;  z[3  + num] =  c;  w[3  + num] =  v;
                x[4  + num] =  a;  y[4  + num] =  b;  z[4  + num] = -c;  w[4  + num] =  v;
                x[5  + num] = -a;  y[5  + num] =  b;  z[5  + num] = -c;  w[5  + num] =  v;
                x[6  + num] =  a;  y[6  + num] = -b;  z[6  + num] = -c;  w[6  + num] =  v;
                x[7  + num] = -a;  y[7  + num] = -b;  z[7  + num] = -c;  w[7  + num] =  v;
                x[8  + num] =  a;  y[8  + num] =  c;  z[8  + num] =  b;  w[8  + num] =  v;
                x[9  + num] = -a;  y[9  + num] =  c;  z[9  + num] =  b;  w[9  + num] =  v;
                x[10 + num] =  a;  y[10 + num] = -c;  z[10 + num] =  b;  w[10 + num] =  v;
                x[11 + num] = -a;  y[11 + num] = -c;  z[11 + num] =  b;  w[11 + num] =  v;
                x[12 + num] =  a;  y[12 + num] =  c;  z[12 + num] = -b;  w[12 + num] =  v;
                x[13 + num] = -a;  y[13 + num] =  c;  z[13 + num] = -b;  w[13 + num] =  v;
                x[14 + num] =  a;  y[14 + num] = -c;  z[14 + num] = -b;  w[14 + num] =  v;
                x[15 + num] = -a;  y[15 + num] = -c;  z[15 + num] = -b;  w[15 + num] =  v;
                x[16 + num] =  b;  y[16 + num] =  a;  z[16 + num] =  c;  w[16 + num] =  v;
                x[17 + num] = -b;  y[17 + num] =  a;  z[17 + num] =  c;  w[17 + num] =  v;
                x[18 + num] =  b;  y[18 + num] = -a;  z[18 + num] =  c;  w[18 + num] =  v;
                x[19 + num] = -b;  y[19 + num] = -a;  z[19 + num] =  c;  w[19 + num] =  v;
                x[20 + num] =  b;  y[20 + num] =  a;  z[20 + num] = -c;  w[20 + num] =  v;
                x[21 + num] = -b;  y[21 + num] =  a;  z[21 + num] = -c;  w[21 + num] =  v;
                x[22 + num] =  b;  y[22 + num] = -a;  z[22 + num] = -c;  w[22 + num] =  v;
                x[23 + num] = -b;  y[23 + num] = -a;  z[23 + num] = -c;  w[23 + num] =  v;
                x[24 + num] =  b;  y[24 + num] =  c;  z[24 + num] =  a;  w[24 + num] =  v;
                x[25 + num] = -b;  y[25 + num] =  c;  z[25 + num] =  a;  w[25 + num] =  v;
                x[26 + num] =  b;  y[26 + num] = -c;  z[26 + num] =  a;  w[26 + num] =  v;
                x[27 + num] = -b;  y[27 + num] = -c;  z[27 + num] =  a;  w[27 + num] =  v;
                x[28 + num] =  b;  y[28 + num] =  c;  z[28 + num] = -a;  w[28 + num] =  v;
                x[29 + num] = -b;  y[29 + num] =  c;  z[29 + num] = -a;  w[29 + num] =  v;
                x[30 + num] =  b;  y[30 + num] = -c;  z[30 + num] = -a;  w[30 + num] =  v;
                x[31 + num] = -b;  y[31 + num] = -c;  z[31 + num] = -a;  w[31 + num] =  v;
                x[32 + num] =  c;  y[32 + num] =  a;  z[32 + num] =  b;  w[32 + num] =  v;
                x[33 + num] = -c;  y[33 + num] =  a;  z[33 + num] =  b;  w[33 + num] =  v;
                x[34 + num] =  c;  y[34 + num] = -a;  z[34 + num] =  b;  w[34 + num] =  v;
                x[35 + num] = -c;  y[35 + num] = -a;  z[35 + num] =  b;  w[35 + num] =  v;
                x[36 + num] =  c;  y[36 + num] =  a;  z[36 + num] = -b;  w[36 + num] =  v;
                x[37 + num] = -c;  y[37 + num] =  a;  z[37 + num] = -b;  w[37 + num] =  v;
                x[38 + num] =  c;  y[38 + num] = -a;  z[38 + num] = -b;  w[38 + num] =  v;
                x[39 + num] = -c;  y[39 + num] = -a;  z[39 + num] = -b;  w[39 + num] =  v;
                x[40 + num] =  c;  y[40 + num] =  b;  z[40 + num] =  a;  w[40 + num] =  v;
                x[41 + num] = -c;  y[41 + num] =  b;  z[41 + num] =  a;  w[41 + num] =  v;
                x[42 + num] =  c;  y[42 + num] = -b;  z[42 + num] =  a;  w[42 + num] =  v;
                x[43 + num] = -c;  y[43 + num] = -b;  z[43 + num] =  a;  w[43 + num] =  v;
                x[44 + num] =  c;  y[44 + num] =  b;  z[44 + num] = -a;  w[44 + num] =  v;
                x[45 + num] = -c;  y[45 + num] =  b;  z[45 + num] = -a;  w[45 + num] =  v;
                x[46 + num] =  c;  y[46 + num] = -b;  z[46 + num] = -a;  w[46 + num] =  v;
                x[47 + num] = -c;  y[47 + num] = -b;  z[47 + num] = -a;  w[47 + num] =  v;
                num = num + 48;
                break;
            }
    default:
        break;
    }

    return num;
}


/*
   ! W
   ! W    LEBEDEV    6-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0006(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.1666666666666667);
}


/*
   ! W
   ! W    LEBEDEV    14-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0014(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.6666666666666667e-1);
    gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.7500000000000000e-1);
}


/*
   ! W
   ! W    LEBEDEV    26-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0026(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.4761904761904762e-1);
    N = gen_oh(2, N, x, y, z, w, 0.0, 0.0, 0.3809523809523810e-1);
    gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.3214285714285714e-1);
}


/*
   ! W
   ! W    LEBEDEV    38-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0038(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.9523809523809524e-2);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.3214285714285714e-1);
    gen_oh(5, N, x, y, z, w, 0.4597008433809831e+0, 0.0, 0.2857142857142857e-1);
}


/*
   ! W
   ! W    LEBEDEV    50-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0050(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.1269841269841270e-1);
    N = gen_oh(2, N, x, y, z, w, 0.0, 0.0, 0.2257495590828924e-1);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.2109375000000000e-1);
    gen_oh(4, N, x, y, z, w, 0.3015113445777636e+0, 0.0, 0.2017333553791887e-1);
}


/*
   ! W
   ! W    LEBEDEV    74-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0074(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.5130671797338464e-3);
    N = gen_oh(2, N, x, y, z, w, 0.0, 0.0, 0.1660406956574204e-1);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, -0.2958603896103896e-1);
    N = gen_oh(4, N, x, y, z, w, 0.4803844614152614, 0.0, 0.2657620708215946e-1);
    gen_oh(5, N, x, y, z, w, 0.3207726489807764, 0.0, 0.1652217099371571e-1);
}


/*
   ! W
   ! W    LEBEDEV    86-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0086(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.1154401154401154e-1);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.1194390908585628e-1);
    N = gen_oh(4, N, x, y, z, w, 0.3696028464541502, 0.0, 0.1111055571060340e-1);
    N = gen_oh(4, N, x, y, z, w, 0.6943540066026664, 0.0, 0.1187650129453714e-1);
    gen_oh(5, N, x, y, z, w, 0.3742430390903412, 0.0, 0.1181230374690448e-1);
}


/*
   ! W
   ! W    LEBEDEV    110-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0110(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.3828270494937162e-2);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.9793737512487512e-2);
    N = gen_oh(4, N, x, y, z, w, 0.1851156353447362, 0.0, 0.8211737283191111e-2);
    N = gen_oh(4, N, x, y, z, w, 0.6904210483822922, 0.0, 0.9942814891178103e-2);
    N = gen_oh(4, N, x, y, z, w, 0.3956894730559419, 0.0, 0.9595471336070963e-2);
    gen_oh(5, N, x, y, z, w, 0.4783690288121502, 0.0, 0.9694996361663028e-2);
}


/*
   ! W
   ! W    LEBEDEV    146-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0146(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.5996313688621381e-3);
    N = gen_oh(2, N, x, y, z, w, 0.0, 0.0, 0.7372999718620756e-2);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.7210515360144488e-2);
    N = gen_oh(4, N, x, y, z, w, 0.6764410400114264e+0, 0.0, 0.7116355493117555e-2);
    N = gen_oh(4, N, x, y, z, w, 0.4174961227965453e+0, 0.0, 0.6753829486314477e-2);
    N = gen_oh(4, N, x, y, z, w, 0.1574676672039082e+0, 0.0, 0.7574394159054034e-2);
    gen_oh(6, N, x, y, z, w, 0.1403553811713183e+0, 0.4493328323269557e+0, 0.6991087353303262e-2);
}


/*
   ! W
   ! W    LEBEDEV    170-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0170(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.5544842902037365e-2);
    N = gen_oh(2, N, x, y, z, w, 0.0, 0.0, 0.6071332770670752e-2);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.6383674773515093e-2);
    N = gen_oh(4, N, x, y, z, w, 0.2551252621114134e+0, 0.0, 0.5183387587747790e-2);
    N = gen_oh(4, N, x, y, z, w, 0.6743601460362766e+0, 0.0, 0.6317929009813725e-2);
    N = gen_oh(4, N, x, y, z, w, 0.4318910696719410e+0, 0.0, 0.6201670006589077e-2);
    N = gen_oh(5, N, x, y, z, w, 0.2613931360335988e+0, 0.0, 0.5477143385137348e-2);
    gen_oh(6, N, x, y, z, w, 0.4990453161796037e+0, 0.1446630744325115e+0, 0.5968383987681156e-2);
}


/*
   ! W
   ! W    LEBEDEV    194-POINT ANGULAR GRID
   ! W
   ! vd
   ! vd   This subroutine is part of a set of subroutines that generate
   ! vd   Lebedev grids [1-6] for integration on a sphere. The original
   ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
   ! vd   translated into fortran by Dr. Christoph van Wuellen.
   ! vd
   ! vd
   ! vd   Users of this code are asked to include reference [1] in their
   ! vd   publications, and in the user- and programmers-manuals
   ! vd   describing their codes.
   ! vd
   ! vd   This code was distributed through CCL (http://www.ccl.net/).
   ! vd
   ! vd   [1] V.I. Lebedev, and D.N. Laikov
   ! vd       "A quadrature formula for the sphere of the 131st
   ! vd        algebraic order of accuracy"
   ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
   ! vd
   ! vd   [2] V.I. Lebedev
   ! vd       "A quadrature formula for the sphere of 59th algebraic
   ! vd        order of accuracy"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
   ! vd
   ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
   ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
   ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
   ! vd
   ! vd   [4] V.I. Lebedev
   ! vd       "Spherical quadrature formulas exact to orders 25-29"
   ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
   ! vd
   ! vd   [5] V.I. Lebedev
   ! vd       "Quadratures on a sphere"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
   ! vd       1976, pp. 10-24.
   ! vd
   ! vd   [6] V.I. Lebedev
   ! vd       "Values of the nodes and weights of ninth to seventeenth
   ! vd        order Gauss-Markov quadrature formulae invariant under the
   ! vd        octahedron group with inversion"
   ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
   ! vd       1975, pp. 44-51.
   ! vd
*/
__device__ static inline void LD0194(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    N = gen_oh(1, 0, x, y, z, w, 0.0, 0.0, 0.1782340447244611e-2);
    N = gen_oh(2, N, x, y, z, w, 0.0, 0.0, 0.5716905949977102e-2);
    N = gen_oh(3, N, x, y, z, w, 0.0, 0.0, 0.5573383178848738e-2);
    N = gen_oh(4, N, x, y, z, w, 0.6712973442695226e+0, 0.0, 0.5608704082587997e-2);
    N = gen_oh(4, N, x, y, z, w, 0.2892465627575439e+0, 0.0, 0.5158237711805383e-2);
    N = gen_oh(4, N, x, y, z, w, 0.4446933178717437e+0, 0.0, 0.5518771467273614e-2);
    N = gen_oh(4, N, x, y, z, w, 0.1299335447650067e+0, 0.0, 0.4106777028169394e-2);
    N = gen_oh(5, N, x, y, z, w, 0.3457702197611283e+0, 0.0, 0.5051846064614808e-2);
    gen_oh(6, N, x, y, z, w, 0.1590417105383530e+0, 0.8360360154824589e+0, 0.5530248916233094e-2);
}


#include "gpu_getxc.h"
#if defined(CEW)
  #include "gpu_cew_quad.h"
#endif
#define OSHELL
#include "gpu_getxc.h"
#if defined(CEW)
  #include "gpu_cew_quad.h"
#endif
#undef OSHELL


__global__ void get_pteval_kernel() {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {
        int bin_id = devSim_dft.bin_locator[gid];
        unsigned int idx = devSim_dft.phi_loc[gid];

        QUICKDouble gridx = devSim_dft.gridx[gid];
        QUICKDouble gridy = devSim_dft.gridy[gid];
        QUICKDouble gridz = devSim_dft.gridz[gid];

        for (int i = devSim_dft.basf_locator[bin_id]; i < devSim_dft.basf_locator[bin_id + 1]; i++) {
            int ibas = (int) devSim_dft.basf[i];
            QUICKDouble phi, dphidx, dphidy, dphidz;

            pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz,
                    devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

            devSim_dft.phi[idx] = phi;
            devSim_dft.dphidx[idx] = dphidx;
            devSim_dft.dphidy[idx] = dphidy;
            devSim_dft.dphidz[idx] = dphidz;
            ++idx;
        }
    }
}


//device kernel to compute significant grid pts, contracted and primitive functions for octree
__global__ void get_primf_contraf_lists_kernel(unsigned char * const gpweight,
        unsigned int * const cfweight, unsigned int * const pfweight) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;

    for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {
        if (gpweight[gid] > 0) {
            int binIdx = devSim_dft.bin_locator[gid];

            QUICKDouble gridx = devSim_dft.gridx[gid];
            QUICKDouble gridy = devSim_dft.gridy[gid];
            QUICKDouble gridz = devSim_dft.gridz[gid];

            unsigned int sigcfcount = 0;

            // relative coordinates between grid point and basis function I.
            for (int ibas = 0; ibas < devSim_dft.nbasis; ibas++) {
                unsigned long cfwid = binIdx * devSim_dft.nbasis + ibas;

                QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
                QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
                QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas] - 1, 3, devSim_dft.natom);
                QUICKDouble dist = x1 * x1 + y1 * y1 + z1 *z1;

                QUICKDouble x1i, y1i, z1i;
                QUICKDouble x1imin1, y1imin1, z1imin1;
                QUICKDouble x1iplus1, y1iplus1, z1iplus1;

                QUICKDouble phi = 0.0;
                QUICKDouble dphidx = 0.0;
                QUICKDouble dphidy = 0.0;
                QUICKDouble dphidz = 0.0;

                int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
                int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
                int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

                if (dist <= devSim_dft.sigrad2[ibas]) {
                    if (itypex != 0) {
                        x1imin1 = pow(x1, itypex - 1);
                        x1i = x1imin1 * x1;
                        x1iplus1 = x1i * x1;
                    } else {
                        x1imin1 = 0.0;
                        x1i = 1.0;
                        x1iplus1 = x1;
                    }

                    if (itypey != 0) {
                        y1imin1 = pow(y1, itypey - 1);
                        y1i = y1imin1 * y1;
                        y1iplus1 = y1i * y1;
                    } else {
                        y1imin1 = 0.0;
                        y1i = 1.0;
                        y1iplus1 = y1;
                    }

                    if (itypez != 0) {
                        z1imin1 = pow(z1, itypez - 1);
                        z1i = z1imin1 * z1;
                        z1iplus1 = z1i * z1;
                    } else {
                        z1imin1 = 0.0;
                        z1i = 1.0;
                        z1iplus1 = z1;
                    }

                    for (int kprim = 0; kprim < devSim_dft.ncontract[ibas]; kprim++) {
                        unsigned long pfwid = binIdx * devSim_dft.nbasis * devSim_dft.maxcontract
                            + ibas * devSim_dft.maxcontract + kprim;
                        QUICKDouble aexp = LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis);

                        QUICKDouble tmp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)
                            * exp(-aexp * dist);
                        aexp *= -2.0;
                        QUICKDouble tmpdx = tmp * (aexp * x1iplus1 + (QUICKDouble) itypex * x1imin1);
                        QUICKDouble tmpdy = tmp * (aexp * y1iplus1 + (QUICKDouble) itypey * y1imin1);
                        QUICKDouble tmpdz = tmp * (aexp * z1iplus1 + (QUICKDouble) itypez * z1imin1);

                        phi += tmp;
                        dphidx += tmpdx;
                        dphidy += tmpdy;
                        dphidz += tmpdz;

                        //Check the significance of the primitive
                        if (abs(tmp + tmpdx + tmpdy + tmpdz) > devSim_dft.XCCutoff) {
                            atomicAdd(&pfweight[pfwid], 1);
                        }
                    }

                    phi *= x1i * y1i * z1i;
                    dphidx *= y1i * z1i;
                    dphidy *= x1i * z1i;
                    dphidz *= x1i * y1i;
                }

                if (abs(phi + dphidx + dphidy + dphidz) > devSim_dft.XCCutoff) {
                    atomicAdd(&cfweight[cfwid], 1);
                    sigcfcount++;
                }
            }

            if (sigcfcount < 1) {
                gpweight[gid] = 0;
            }
        }
    }
}


__global__ void get_ssw_kernel() {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {
        QUICKDouble gridx = devSim_dft.gridx[gid];
        QUICKDouble gridy = devSim_dft.gridy[gid];
        QUICKDouble gridz = devSim_dft.gridz[gid];
        QUICKDouble wtang = devSim_dft.wtang[gid];
        QUICKDouble rwt = devSim_dft.rwt[gid];
        QUICKDouble rad3 = devSim_dft.rad3[gid];
        int gatm = devSim_dft.gatm[gid];

        QUICKDouble sswt = SSW(gridx, gridy, gridz, devSim_dft.xyz, gatm);
        QUICKDouble weight = sswt * wtang * rwt * rad3;

        devSim_dft.sswt[gid] = sswt;
        devSim_dft.weight[gid] = weight;
    }
}


__global__ void get_sswgrad_kernel() {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
#if defined(USE_LEGACY_ATOMICS)
    extern __shared__ QUICKULL smem_buffer[];
    QUICKULL* smemGrad = (QUICKULL*) smem_buffer;

    for (int i = threadIdx.x; i < devSim_dft.natom * 3; i += blockDim.x) {
      smemGrad[i] = 0ull;
    }
#else
    extern __shared__ QUICKDouble smem_buffer[];
    QUICKDouble* smemGrad = (QUICKDouble*) smem_buffer;

    for (int i = threadIdx.x; i < devSim_dft.natom * 3; i += blockDim.x) {
        smemGrad[i] = 0.0;
    }
#endif
    __syncthreads();

    for (QUICKULL gid = offset; gid < devSim_dft.npoints_ssd; gid += totalThreads) {
        QUICKDouble gridx = devSim_dft.gridx_ssd[gid];
        QUICKDouble gridy = devSim_dft.gridy_ssd[gid];
        QUICKDouble gridz = devSim_dft.gridz_ssd[gid];
        QUICKDouble exc = devSim_dft.exc_ssd[gid];
        QUICKDouble quadwt = devSim_dft.quadwt[gid];
        int gatm = devSim_dft.gatm_ssd[gid];

        sswanader(gridx, gridy, gridz, exc, quadwt, smemGrad, devSim_dft.uw_ssd + offset, gatm, devSim_dft.natom);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < devSim_dft.natom * 3; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&devSim_dft.gradULL[i], smemGrad[i]);
#else
        atomicAdd(&devSim_dft.grad[i], smemGrad[i]);
#endif
    }

    __syncthreads();
}


/*
   compute grid weight gradients using finite difference method
   Note: this can be futher speed up by using npoints*natom*3 # of threads and
   computing x, y and z gradients separately.
*/
__global__ void get_sswnumgrad_kernel() {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    unsigned int natom = devSim_dft.natom;
    QUICKULL npoints_ssd = devSim_dft.npoints_ssd;
#if defined(USE_LEGACY_ATOMICS)
    //declare smem grad vector
    extern __shared__ QUICKULL smem_buffer[];
    QUICKULL* smemGrad = (QUICKULL*) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
        smemGrad[i] = 0ull;
    }
#else
    extern __shared__ QUICKDouble smem_buffer[];
    QUICKDouble* smemGrad = (QUICKDouble*) smem_buffer;

    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
        smemGrad[i] = 0.0;
    }
#endif

    __syncthreads();

    for (QUICKULL gid = offset; gid < npoints_ssd * natom; gid += totalThreads) {
        unsigned int iatom = (unsigned int) gid / npoints_ssd;
        QUICKULL idx = gid - iatom * npoints_ssd;

        QUICKDouble gridx = devSim_dft.gridx_ssd[idx];
        QUICKDouble gridy = devSim_dft.gridy_ssd[idx];
        QUICKDouble gridz = devSim_dft.gridz_ssd[idx];
        QUICKDouble gradfac = RECIP_SSW_NUMGRAD_DELTA2 * devSim_dft.exc_ssd[idx] * devSim_dft.quadwt[idx];
        int gatm = devSim_dft.gatm_ssd[idx];

//        QUICKDouble *tmpxyz = devSim_dft.xyz_ssd + 3 * devSim_dft.natom*offset;

        QUICKDouble xparent = LOC2(devSim_dft.xyz, 0, gatm - 1, 3, natom);
        QUICKDouble yparent = LOC2(devSim_dft.xyz, 1, gatm - 1, 3, natom);
        QUICKDouble zparent = LOC2(devSim_dft.xyz, 2, gatm - 1, 3, natom);
        QUICKDouble tx = gridx - xparent;
        QUICKDouble ty = gridy - yparent;
        QUICKDouble tz = gridz - zparent;

        //for (int i = 0; i<devSim_dft.natom; i++) {

        // compute x gradient
        QUICKDouble xatm = LOC2(devSim_dft.xyz, 0, iatom, 3, natom);
        QUICKDouble yatm = LOC2(devSim_dft.xyz, 1, iatom, 3, natom);
        QUICKDouble zatm = LOC2(devSim_dft.xyz, 2, iatom, 3, natom);

        xatm += SSW_NUMGRAD_DELTA;

        if (iatom == gatm - 1) {
            xparent = xatm;
        }

        QUICKDouble sswt1 = SSW(xparent + tx, yparent + ty, zparent + tz, devSim_dft.xyz,
                xparent, yparent, zparent, xatm, yatm, zatm, iatom, gatm, natom);

        xatm -= SSW_NUMGRAD_DELTA2;

        if (iatom == gatm - 1) {
            xparent = xatm;
        }

        QUICKDouble sswt2 = SSW(xparent + tx, yparent + ty, zparent + tz, devSim_dft.xyz,
                xparent, yparent, zparent, xatm, yatm, zatm, iatom, gatm, natom);

        QUICKDouble dpx = (sswt1 - sswt2) * gradfac;

        xatm += SSW_NUMGRAD_DELTA;

        if (iatom == gatm - 1)
        {
            xparent = xatm;
        }

        // compute y gradient
        yatm += SSW_NUMGRAD_DELTA;

        if (iatom == gatm - 1) {
            yparent = yatm;
        }

        sswt1 = SSW(xparent + tx, yparent + ty, zparent + tz, devSim_dft.xyz,
                xparent, yparent, zparent, xatm, yatm, zatm, iatom, gatm, natom);

        yatm -= SSW_NUMGRAD_DELTA2;

        if (iatom == gatm - 1) {
            yparent = yatm;
        }

        sswt2 = SSW(xparent + tx, yparent + ty, zparent + tz, devSim_dft.xyz,
                xparent, yparent, zparent, xatm, yatm, zatm, iatom, gatm, natom);

        QUICKDouble dpy = (sswt1 - sswt2) * gradfac;

        yatm += SSW_NUMGRAD_DELTA;

        if (iatom == gatm - 1) {
            yparent = yatm;
        }

        // compute z gradient
        zatm += SSW_NUMGRAD_DELTA;

        if(iatom == gatm-1) zparent = zatm;

        sswt1 = SSW(xparent + tx, yparent + ty, zparent + tz, devSim_dft.xyz,
                xparent, yparent, zparent, xatm, yatm, zatm, iatom, gatm, natom);

        zatm -= SSW_NUMGRAD_DELTA2;

        if(iatom == gatm-1) zparent = zatm;

        sswt2 = SSW(xparent + tx, yparent + ty, zparent + tz, devSim_dft.xyz,
                xparent, yparent, zparent, xatm, yatm, zatm, iatom, gatm, natom);

        QUICKDouble dpz = (sswt1-sswt2) * gradfac;

        zatm += SSW_NUMGRAD_DELTA;

        if(iatom == gatm-1) zparent = zatm;

        GPUATOMICADD(&smemGrad[iatom * 3], dpx, GRADSCALE);
        GPUATOMICADD(&smemGrad[iatom * 3 + 1], dpy, GRADSCALE);
        GPUATOMICADD(&smemGrad[iatom * 3 + 2], dpz, GRADSCALE);
        /*
           printf("sswgrad  %f %f %f %d %d %f %f %f \n", gridx, gridy, gridz, iatom, 1, dpx, devSim_dft.exc_ssd[idx], devSim_dft.quadwt[idx]);

           printf("sswgrad  %f %f %f %d %d %f %f %f \n", gridx, gridy, gridz, iatom, 2, dpy, devSim_dft.exc_ssd[idx], devSim_dft.quadwt[idx]);

           printf("sswgrad  %f %f %f %d %d %f %f %f \n", gridx, gridy, gridz, iatom, 3, dpz, devSim_dft.exc_ssd[idx], devSim_dft.quadwt[idx]);
           */
        //}


    }

    __syncthreads();

    // update gmem grad vector
    for (int i = threadIdx.x; i< natom * 3; i += blockDim.x)
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&devSim_dft.gradULL[i], smemGrad[i]);
#else
        atomicAdd(&devSim_dft.grad[i], smemGrad[i]);
#endif

    __syncthreads();

}

#if defined(CEW)
static void get_cew_accdens(_gpu_type gpu) {
    QUICKDouble *gridpt = new QUICKDouble[3];
    QUICKDouble *cewGrad= new QUICKDouble[3];

    gpu->gpu_xcq->densa->Download();

    if (gpu->gpu_sim.is_oshell == true) {
        gpu->gpu_xcq->densb->Download();
    }

    for (int i = 0; i < gpu->gpu_xcq->npoints; i++) {
        QUICKDouble weight = gpu->gpu_xcq->weight->_hostData[i];
        QUICKDouble densea = gpu->gpu_xcq->densa->_hostData[i];
        QUICKDouble denseb = densea;

        if (gpu -> gpu_sim.is_oshell == true) {
            denseb = gpu->gpu_xcq->densb->_hostData[i];
        }

        gridpt[0] = gpu->gpu_xcq->gridx->_hostData[i];
        gridpt[1] = gpu->gpu_xcq->gridy->_hostData[i];
        gridpt[2] = gpu->gpu_xcq->gridz->_hostData[i];

        const QUICKDouble charge_density = -weight * (densea + denseb);

        for (int j = 0; j < 3; j++) {
            cewGrad[j] = 0.0;
        }

        QUICKDouble const *cnst_gridpt = gridpt;

        // this function comes from cew library in amber
        cew_accdensatpt_(cnst_gridpt, &charge_density, cewGrad);

        //printf("cew_accdensatpt %f %f %f %f %f %f %f \n", gridpt[0], gridpt[1], gridpt[2], charge_density, cewGrad[0], cewGrad[1], cewGrad[2]);

        int Istart = (gpu->gpu_xcq->gatm->_hostData[i] - 1) * 3;

        for (int j = 0; j < 3; j++) {
#if defined(USE_LEGACY_ATOMICS)
            gpu->grad->_hostData[Istart + j] += cewGrad[j];
#else
            gpu->cew_grad->_hostData[Istart + j] += cewGrad[j];
#endif
        }
    }

    delete gridpt;
    delete cewGrad;
}
#endif


/*
   upload gpu simulation type to constant memory
*/
void upload_sim_to_constant_dft(_gpu_type gpu) {
    gpuMemcpyToSymbol((const void *) &devSim_dft, (const void *) &gpu->gpu_sim, sizeof(gpu_simulation_type));
}


void get_ssw(_gpu_type gpu) {
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    QUICK_SAFE_CALL((get_ssw_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));

#if defined(DEBUG)
    GPU_TIMER_STOP();
    totTime += time;
    fprintf(gpu->debugFile, "Time to compute grid weights on gpu:%f ms total time:%f ms\n", time, totTime);
    GPU_TIMER_DESTROY();
#endif
}


void get_primf_contraf_lists(_gpu_type gpu, unsigned char *gpweight,
        unsigned int *cfweight, unsigned int *pfweight) {
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    QUICK_SAFE_CALL((get_primf_contraf_lists_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>>
                (gpweight, cfweight, pfweight)));

#if defined(DEBUG)
    GPU_TIMER_STOP();
    totTime += time;
    fprintf(gpu->debugFile, "Time to compute primitive and contracted indices on gpu: %f ms total time:%f ms\n", time, totTime);
    GPU_TIMER_DESTROY();
#endif
}


void getpteval(_gpu_type gpu) {
    QUICK_SAFE_CALL((get_pteval_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
}


void getxc(_gpu_type gpu) {
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    if (gpu->gpu_sim.is_oshell == true) {
        QUICK_SAFE_CALL((get_oshell_density_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
        QUICK_SAFE_CALL((oshell_getxc_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
    } else {
        QUICK_SAFE_CALL((get_cshell_density_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
        QUICK_SAFE_CALL((cshell_getxc_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
    }

#if defined(DEBUG)
    GPU_TIMER_STOP();
    totTime += time;
    fprintf(gpu->debugFile, "this DFT cycle:%f ms total time:%f ms\n", time, totTime);
    GPU_TIMER_DESTROY();
#endif

//    nvtxRangePop();
}


void getxc_grad(_gpu_type gpu) {
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    if (gpu->gpu_sim.is_oshell == true) {
        QUICK_SAFE_CALL((get_oshell_density_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
        QUICK_SAFE_CALL((oshell_getxcgrad_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock, gpu->gpu_xcq->smem_size>>> ()));
    } else {
        QUICK_SAFE_CALL((get_cshell_density_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock>>> ()));
        QUICK_SAFE_CALL((cshell_getxcgrad_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock, gpu->gpu_xcq->smem_size>>> ()));
    }

#if defined(CEW)
    if (gpu->gpu_sim.use_cew) {
        get_cew_accdens(gpu);
    }
#endif

    prune_grid_sswgrad();

    QUICK_SAFE_CALL((get_sswgrad_kernel <<<gpu->blocks, gpu->xc_threadsPerBlock, gpu->gpu_xcq->smem_size>>> ()));
//    QUICK_SAFE_CALL((get_sswnumgrad_kernel <<<gpu->blocks, gpu->sswGradThreadsPerBlock, gpu->gpu_xcq->smem_size>>> ()));

    gpu_delete_sswgrad_vars();

#if defined(DEBUG)
    GPU_TIMER_STOP();
    totTime += time;
    fprintf(gpu->debugFile, "this DFT cycle:%f ms total time:%f ms\n", time, totTime);
    GPU_TIMER_DESTROY();
#endif

//    nvtxRangePop();
}


#if defined(CEW)
void getcew_quad(_gpu_type gpu) {
    QUICK_SAFE_CALL((getcew_quad_kernel<<< gpu->blocks, gpu->xc_threadsPerBlock>>>()));
}


void getcew_quad_grad(_gpu_type gpu) {
    if (gpu->gpu_sim.is_oshell == true) {
        QUICK_SAFE_CALL((get_oshell_density_kernel<<<gpu->blocks, gpu->xc_threadsPerBlock>>>()));
        QUICK_SAFE_CALL((oshell_getcew_quad_grad_kernel<<< gpu->blocks, gpu->xc_threadsPerBlock, gpu->gpu_xcq->smem_size>>>()));
    } else {
        QUICK_SAFE_CALL((get_cshell_density_kernel<<<gpu->blocks, gpu->xc_threadsPerBlock>>>()));
        QUICK_SAFE_CALL((cshell_getcew_quad_grad_kernel<<< gpu->blocks, gpu->xc_threadsPerBlock, gpu->gpu_xcq->smem_size>>>()));
    }

    get_cew_accdens(gpu);

    prune_grid_sswgrad();

    get_sswnumgrad_kernel<<< gpu->blocks, gpu->sswGradThreadsPerBlock, gpu->gpu_xcq->smem_size>>>();

    gpu_delete_sswgrad_vars();
}
#endif
