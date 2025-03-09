/*
   !---------------------------------------------------------------------!
   ! Written by QUICK-GenInt code generator on 08/12/2021                !
   !                                                                     !
   ! Copyright (C) 2020-2021 Merz lab                                    !
   ! Copyright (C) 2020-2021 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!
   */

__device__ static inline void oei_grad_vertical(int I, int J,
#if defined(DEBUG_OEI)
        int II, int JJ,
#endif
        QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
        QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
        QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
        QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp)
{
    /* SS integral gradient, m=0 */
    if (I == 0 && J == 0) {
        SPint_0 sp(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp);

        LOCSTORE(store, 0, 1, STOREDIM, STOREDIM) = sp.x_0_1;
        LOCSTORE(store, 0, 2, STOREDIM, STOREDIM) = sp.x_0_2;
        LOCSTORE(store, 0, 3, STOREDIM, STOREDIM) = sp.x_0_3;

        PSint_0 ps(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp);

        LOCSTORE(store, 1, 0, STOREDIM, STOREDIM) = ps.x_1_0;
        LOCSTORE(store, 2, 0, STOREDIM, STOREDIM) = ps.x_2_0;
        LOCSTORE(store, 3, 0, STOREDIM, STOREDIM) = ps.x_3_0;

#if defined(DEBUG_OEI)
        printf("II %d JJ %d SP store[0,1] = %f \n", II, JJ, LOCSTORE(store, 0, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d SP store[0,2] = %f \n", II, JJ, LOCSTORE(store, 0, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d SP store[0,3] = %f \n", II, JJ, LOCSTORE(store, 0, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[1,0] = %f \n", II, JJ, LOCSTORE(store, 1, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[2,0] = %f \n", II, JJ, LOCSTORE(store, 2, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[3,0] = %f \n", II, JJ, LOCSTORE(store, 3, 0, STOREDIM, STOREDIM));
#endif
    }

    /* SP integral gradient, m=0 */
    else if (I == 0 && J == 1) {
        LOCSTORE(store, 0, 0, STOREDIM, STOREDIM) = VY(0, 0, 0);

        PPint_0 pp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) = pp.x_1_1;
        LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) = pp.x_1_2;
        LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) = pp.x_1_3;
        LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) = pp.x_2_1;
        LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) = pp.x_2_2;
        LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) = pp.x_2_3;
        LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) = pp.x_3_1;
        LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) = pp.x_3_2;
        LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) = pp.x_3_3;

        SDint_0 sd(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 0, 4, STOREDIM, STOREDIM) = sd.x_0_4;
        LOCSTORE(store, 0, 5, STOREDIM, STOREDIM) = sd.x_0_5;
        LOCSTORE(store, 0, 6, STOREDIM, STOREDIM) = sd.x_0_6;
        LOCSTORE(store, 0, 7, STOREDIM, STOREDIM) = sd.x_0_7;
        LOCSTORE(store, 0, 8, STOREDIM, STOREDIM) = sd.x_0_8;
        LOCSTORE(store, 0, 9, STOREDIM, STOREDIM) = sd.x_0_9;

#if defined(DEBUG_OEI)
        printf("II %d JJ %d SS store[0,0] = %f \n", II, JJ, LOCSTORE(store, 0, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,1] = %f \n", II, JJ, LOCSTORE(store, 1, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,2] = %f \n", II, JJ, LOCSTORE(store, 1, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,3] = %f \n", II, JJ, LOCSTORE(store, 1, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,1] = %f \n", II, JJ, LOCSTORE(store, 2, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,2] = %f \n", II, JJ, LOCSTORE(store, 2, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,3] = %f \n", II, JJ, LOCSTORE(store, 2, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,1] = %f \n", II, JJ, LOCSTORE(store, 3, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,2] = %f \n", II, JJ, LOCSTORE(store, 3, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,3] = %f \n", II, JJ, LOCSTORE(store, 3, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,4] = %f \n", II, JJ, LOCSTORE(store, 0, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,5] = %f \n", II, JJ, LOCSTORE(store, 0, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,6] = %f \n", II, JJ, LOCSTORE(store, 0, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,7] = %f \n", II, JJ, LOCSTORE(store, 0, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,8] = %f \n", II, JJ, LOCSTORE(store, 0, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,9] = %f \n", II, JJ, LOCSTORE(store, 0, 9, STOREDIM, STOREDIM));
#endif
    }

    /* PS integral gradient, m=0 */
    else if (I == 1 && J == 0) {
        LOCSTORE(store, 0, 0, STOREDIM, STOREDIM) = VY(0, 0, 0);

        PPint_0 pp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) = pp.x_1_1;
        LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) = pp.x_1_2;
        LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) = pp.x_1_3;
        LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) = pp.x_2_1;
        LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) = pp.x_2_2;
        LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) = pp.x_2_3;
        LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) = pp.x_3_1;
        LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) = pp.x_3_2;
        LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) = pp.x_3_3;

        DSint_0 ds(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 4, 0, STOREDIM, STOREDIM) = ds.x_4_0;
        LOCSTORE(store, 5, 0, STOREDIM, STOREDIM) = ds.x_5_0;
        LOCSTORE(store, 6, 0, STOREDIM, STOREDIM) = ds.x_6_0;
        LOCSTORE(store, 7, 0, STOREDIM, STOREDIM) = ds.x_7_0;
        LOCSTORE(store, 8, 0, STOREDIM, STOREDIM) = ds.x_8_0;
        LOCSTORE(store, 9, 0, STOREDIM, STOREDIM) = ds.x_9_0;

#if defined(DEBUG_OEI)
        printf("II %d JJ %d SS store[0,0] = %f \n", II, JJ, LOCSTORE(store, 0, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,1] = %f \n", II, JJ, LOCSTORE(store, 1, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,2] = %f \n", II, JJ, LOCSTORE(store, 1, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,3] = %f \n", II, JJ, LOCSTORE(store, 1, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,1] = %f \n", II, JJ, LOCSTORE(store, 2, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,2] = %f \n", II, JJ, LOCSTORE(store, 2, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,3] = %f \n", II, JJ, LOCSTORE(store, 2, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,1] = %f \n", II, JJ, LOCSTORE(store, 3, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,2] = %f \n", II, JJ, LOCSTORE(store, 3, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,3] = %f \n", II, JJ, LOCSTORE(store, 3, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[4,0] = %f \n", II, JJ, LOCSTORE(store, 4, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[5,0] = %f \n", II, JJ, LOCSTORE(store, 5, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[6,0] = %f \n", II, JJ, LOCSTORE(store, 6, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[7,0] = %f \n", II, JJ, LOCSTORE(store, 7, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[8,0] = %f \n", II, JJ, LOCSTORE(store, 8, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[9,0] = %f \n", II, JJ, LOCSTORE(store, 9, 0, STOREDIM, STOREDIM));
#endif
    }

    /* PP integral gradient, m=0 */
    else if (I == 1 && J == 1) {
        SPint_0 sp(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp);

        LOCSTORE(store, 0, 1, STOREDIM, STOREDIM) = sp.x_0_1;
        LOCSTORE(store, 0, 2, STOREDIM, STOREDIM) = sp.x_0_2;
        LOCSTORE(store, 0, 3, STOREDIM, STOREDIM) = sp.x_0_3;

        PSint_0 ps(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp);

        LOCSTORE(store, 1, 0, STOREDIM, STOREDIM) = ps.x_1_0;
        LOCSTORE(store, 2, 0, STOREDIM, STOREDIM) = ps.x_2_0;
        LOCSTORE(store, 3, 0, STOREDIM, STOREDIM) = ps.x_3_0;

        DPint_0 dp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 4, 1, STOREDIM, STOREDIM) = dp.x_4_1;
        LOCSTORE(store, 4, 2, STOREDIM, STOREDIM) = dp.x_4_2;
        LOCSTORE(store, 4, 3, STOREDIM, STOREDIM) = dp.x_4_3;
        LOCSTORE(store, 5, 1, STOREDIM, STOREDIM) = dp.x_5_1;
        LOCSTORE(store, 5, 2, STOREDIM, STOREDIM) = dp.x_5_2;
        LOCSTORE(store, 5, 3, STOREDIM, STOREDIM) = dp.x_5_3;
        LOCSTORE(store, 6, 1, STOREDIM, STOREDIM) = dp.x_6_1;
        LOCSTORE(store, 6, 2, STOREDIM, STOREDIM) = dp.x_6_2;
        LOCSTORE(store, 6, 3, STOREDIM, STOREDIM) = dp.x_6_3;
        LOCSTORE(store, 7, 1, STOREDIM, STOREDIM) = dp.x_7_1;
        LOCSTORE(store, 7, 2, STOREDIM, STOREDIM) = dp.x_7_2;
        LOCSTORE(store, 7, 3, STOREDIM, STOREDIM) = dp.x_7_3;
        LOCSTORE(store, 8, 1, STOREDIM, STOREDIM) = dp.x_8_1;
        LOCSTORE(store, 8, 2, STOREDIM, STOREDIM) = dp.x_8_2;
        LOCSTORE(store, 8, 3, STOREDIM, STOREDIM) = dp.x_8_3;
        LOCSTORE(store, 9, 1, STOREDIM, STOREDIM) = dp.x_9_1;
        LOCSTORE(store, 9, 2, STOREDIM, STOREDIM) = dp.x_9_2;
        LOCSTORE(store, 9, 3, STOREDIM, STOREDIM) = dp.x_9_3;

        PDint_0 pd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 4, STOREDIM, STOREDIM) = pd.x_1_4;
        LOCSTORE(store, 2, 4, STOREDIM, STOREDIM) = pd.x_2_4;
        LOCSTORE(store, 3, 4, STOREDIM, STOREDIM) = pd.x_3_4;
        LOCSTORE(store, 1, 5, STOREDIM, STOREDIM) = pd.x_1_5;
        LOCSTORE(store, 2, 5, STOREDIM, STOREDIM) = pd.x_2_5;
        LOCSTORE(store, 3, 5, STOREDIM, STOREDIM) = pd.x_3_5;
        LOCSTORE(store, 1, 6, STOREDIM, STOREDIM) = pd.x_1_6;
        LOCSTORE(store, 2, 6, STOREDIM, STOREDIM) = pd.x_2_6;
        LOCSTORE(store, 3, 6, STOREDIM, STOREDIM) = pd.x_3_6;
        LOCSTORE(store, 1, 7, STOREDIM, STOREDIM) = pd.x_1_7;
        LOCSTORE(store, 2, 7, STOREDIM, STOREDIM) = pd.x_2_7;
        LOCSTORE(store, 3, 7, STOREDIM, STOREDIM) = pd.x_3_7;
        LOCSTORE(store, 1, 8, STOREDIM, STOREDIM) = pd.x_1_8;
        LOCSTORE(store, 2, 8, STOREDIM, STOREDIM) = pd.x_2_8;
        LOCSTORE(store, 3, 8, STOREDIM, STOREDIM) = pd.x_3_8;
        LOCSTORE(store, 1, 9, STOREDIM, STOREDIM) = pd.x_1_9;
        LOCSTORE(store, 2, 9, STOREDIM, STOREDIM) = pd.x_2_9;
        LOCSTORE(store, 3, 9, STOREDIM, STOREDIM) = pd.x_3_9;

#if defined(DEBUG_OEI)
        printf("II %d JJ %d SP store[0,1] = %f \n", II, JJ, LOCSTORE(store, 0, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d SP store[0,2] = %f \n", II, JJ, LOCSTORE(store, 0, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d SP store[0,3] = %f \n", II, JJ, LOCSTORE(store, 0, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[1,0] = %f \n", II, JJ, LOCSTORE(store, 1, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[2,0] = %f \n", II, JJ, LOCSTORE(store, 2, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[3,0] = %f \n", II, JJ, LOCSTORE(store, 3, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,1] = %f \n", II, JJ, LOCSTORE(store, 4, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,2] = %f \n", II, JJ, LOCSTORE(store, 4, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,3] = %f \n", II, JJ, LOCSTORE(store, 4, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,1] = %f \n", II, JJ, LOCSTORE(store, 5, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,2] = %f \n", II, JJ, LOCSTORE(store, 5, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,3] = %f \n", II, JJ, LOCSTORE(store, 5, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,1] = %f \n", II, JJ, LOCSTORE(store, 6, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,2] = %f \n", II, JJ, LOCSTORE(store, 6, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,3] = %f \n", II, JJ, LOCSTORE(store, 6, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,1] = %f \n", II, JJ, LOCSTORE(store, 7, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,2] = %f \n", II, JJ, LOCSTORE(store, 7, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,3] = %f \n", II, JJ, LOCSTORE(store, 7, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,1] = %f \n", II, JJ, LOCSTORE(store, 8, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,2] = %f \n", II, JJ, LOCSTORE(store, 8, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,3] = %f \n", II, JJ, LOCSTORE(store, 8, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,1] = %f \n", II, JJ, LOCSTORE(store, 9, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,2] = %f \n", II, JJ, LOCSTORE(store, 9, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,3] = %f \n", II, JJ, LOCSTORE(store, 9, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,4] = %f \n", II, JJ, LOCSTORE(store, 1, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,4] = %f \n", II, JJ, LOCSTORE(store, 2, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,4] = %f \n", II, JJ, LOCSTORE(store, 3, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,5] = %f \n", II, JJ, LOCSTORE(store, 1, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,5] = %f \n", II, JJ, LOCSTORE(store, 2, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,5] = %f \n", II, JJ, LOCSTORE(store, 3, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,6] = %f \n", II, JJ, LOCSTORE(store, 1, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,6] = %f \n", II, JJ, LOCSTORE(store, 2, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,6] = %f \n", II, JJ, LOCSTORE(store, 3, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,7] = %f \n", II, JJ, LOCSTORE(store, 1, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,7] = %f \n", II, JJ, LOCSTORE(store, 2, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,7] = %f \n", II, JJ, LOCSTORE(store, 3, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,8] = %f \n", II, JJ, LOCSTORE(store, 1, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,8] = %f \n", II, JJ, LOCSTORE(store, 2, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,8] = %f \n", II, JJ, LOCSTORE(store, 3, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,9] = %f \n", II, JJ, LOCSTORE(store, 1, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,9] = %f \n", II, JJ, LOCSTORE(store, 2, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,9] = %f \n", II, JJ, LOCSTORE(store, 3, 9, STOREDIM, STOREDIM));
#endif
    }

    /* SD integral gradient, m=0 */
    else if (I == 0 && J == 2) {
        SPint_0 sp(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp);

        LOCSTORE(store, 0, 1, STOREDIM, STOREDIM) = sp.x_0_1;
        LOCSTORE(store, 0, 2, STOREDIM, STOREDIM) = sp.x_0_2;
        LOCSTORE(store, 0, 3, STOREDIM, STOREDIM) = sp.x_0_3;

        PDint_0 pd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 4, STOREDIM, STOREDIM) = pd.x_1_4;
        LOCSTORE(store, 2, 4, STOREDIM, STOREDIM) = pd.x_2_4;
        LOCSTORE(store, 3, 4, STOREDIM, STOREDIM) = pd.x_3_4;
        LOCSTORE(store, 1, 5, STOREDIM, STOREDIM) = pd.x_1_5;
        LOCSTORE(store, 2, 5, STOREDIM, STOREDIM) = pd.x_2_5;
        LOCSTORE(store, 3, 5, STOREDIM, STOREDIM) = pd.x_3_5;
        LOCSTORE(store, 1, 6, STOREDIM, STOREDIM) = pd.x_1_6;
        LOCSTORE(store, 2, 6, STOREDIM, STOREDIM) = pd.x_2_6;
        LOCSTORE(store, 3, 6, STOREDIM, STOREDIM) = pd.x_3_6;
        LOCSTORE(store, 1, 7, STOREDIM, STOREDIM) = pd.x_1_7;
        LOCSTORE(store, 2, 7, STOREDIM, STOREDIM) = pd.x_2_7;
        LOCSTORE(store, 3, 7, STOREDIM, STOREDIM) = pd.x_3_7;
        LOCSTORE(store, 1, 8, STOREDIM, STOREDIM) = pd.x_1_8;
        LOCSTORE(store, 2, 8, STOREDIM, STOREDIM) = pd.x_2_8;
        LOCSTORE(store, 3, 8, STOREDIM, STOREDIM) = pd.x_3_8;
        LOCSTORE(store, 1, 9, STOREDIM, STOREDIM) = pd.x_1_9;
        LOCSTORE(store, 2, 9, STOREDIM, STOREDIM) = pd.x_2_9;
        LOCSTORE(store, 3, 9, STOREDIM, STOREDIM) = pd.x_3_9;

        SFint_0 sf(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

#if defined(REG_SF)
        LOCSTORE(store, 0, 10, STOREDIM, STOREDIM) = sf.x_0_10;
        LOCSTORE(store, 0, 11, STOREDIM, STOREDIM) = sf.x_0_11;
        LOCSTORE(store, 0, 12, STOREDIM, STOREDIM) = sf.x_0_12;
        LOCSTORE(store, 0, 13, STOREDIM, STOREDIM) = sf.x_0_13;
        LOCSTORE(store, 0, 14, STOREDIM, STOREDIM) = sf.x_0_14;
        LOCSTORE(store, 0, 15, STOREDIM, STOREDIM) = sf.x_0_15;
        LOCSTORE(store, 0, 16, STOREDIM, STOREDIM) = sf.x_0_16;
        LOCSTORE(store, 0, 17, STOREDIM, STOREDIM) = sf.x_0_17;
        LOCSTORE(store, 0, 18, STOREDIM, STOREDIM) = sf.x_0_18;
        LOCSTORE(store, 0, 19, STOREDIM, STOREDIM) = sf.x_0_19;
#endif

#if defined(DEBUG_OEI)
        printf("II %d JJ %d SP store[0,1] = %f \n", II, JJ, LOCSTORE(store, 0, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d SP store[0,2] = %f \n", II, JJ, LOCSTORE(store, 0, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d SP store[0,3] = %f \n", II, JJ, LOCSTORE(store, 0, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,4] = %f \n", II, JJ, LOCSTORE(store, 1, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,4] = %f \n", II, JJ, LOCSTORE(store, 2, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,4] = %f \n", II, JJ, LOCSTORE(store, 3, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,5] = %f \n", II, JJ, LOCSTORE(store, 1, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,5] = %f \n", II, JJ, LOCSTORE(store, 2, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,5] = %f \n", II, JJ, LOCSTORE(store, 3, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,6] = %f \n", II, JJ, LOCSTORE(store, 1, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,6] = %f \n", II, JJ, LOCSTORE(store, 2, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,6] = %f \n", II, JJ, LOCSTORE(store, 3, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,7] = %f \n", II, JJ, LOCSTORE(store, 1, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,7] = %f \n", II, JJ, LOCSTORE(store, 2, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,7] = %f \n", II, JJ, LOCSTORE(store, 3, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,8] = %f \n", II, JJ, LOCSTORE(store, 1, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,8] = %f \n", II, JJ, LOCSTORE(store, 2, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,8] = %f \n", II, JJ, LOCSTORE(store, 3, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,9] = %f \n", II, JJ, LOCSTORE(store, 1, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,9] = %f \n", II, JJ, LOCSTORE(store, 2, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,9] = %f \n", II, JJ, LOCSTORE(store, 3, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,10] = %f \n", II, JJ, LOCSTORE(store, 0, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,11] = %f \n", II, JJ, LOCSTORE(store, 0, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,12] = %f \n", II, JJ, LOCSTORE(store, 0, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,13] = %f \n", II, JJ, LOCSTORE(store, 0, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,14] = %f \n", II, JJ, LOCSTORE(store, 0, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,15] = %f \n", II, JJ, LOCSTORE(store, 0, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,16] = %f \n", II, JJ, LOCSTORE(store, 0, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,17] = %f \n", II, JJ, LOCSTORE(store, 0, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,18] = %f \n", II, JJ, LOCSTORE(store, 0, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d SF store[0,19] = %f \n", II, JJ, LOCSTORE(store, 0, 19, STOREDIM, STOREDIM));
#endif
    }

    /* DS integral gradient, m=0 */
    else if (I == 2 && J == 0) {
        PSint_0 ps(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp);

        LOCSTORE(store, 1, 0, STOREDIM, STOREDIM) = ps.x_1_0;
        LOCSTORE(store, 2, 0, STOREDIM, STOREDIM) = ps.x_2_0;
        LOCSTORE(store, 3, 0, STOREDIM, STOREDIM) = ps.x_3_0;

        DPint_0 dp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 4, 1, STOREDIM, STOREDIM) = dp.x_4_1;
        LOCSTORE(store, 4, 2, STOREDIM, STOREDIM) = dp.x_4_2;
        LOCSTORE(store, 4, 3, STOREDIM, STOREDIM) = dp.x_4_3;
        LOCSTORE(store, 5, 1, STOREDIM, STOREDIM) = dp.x_5_1;
        LOCSTORE(store, 5, 2, STOREDIM, STOREDIM) = dp.x_5_2;
        LOCSTORE(store, 5, 3, STOREDIM, STOREDIM) = dp.x_5_3;
        LOCSTORE(store, 6, 1, STOREDIM, STOREDIM) = dp.x_6_1;
        LOCSTORE(store, 6, 2, STOREDIM, STOREDIM) = dp.x_6_2;
        LOCSTORE(store, 6, 3, STOREDIM, STOREDIM) = dp.x_6_3;
        LOCSTORE(store, 7, 1, STOREDIM, STOREDIM) = dp.x_7_1;
        LOCSTORE(store, 7, 2, STOREDIM, STOREDIM) = dp.x_7_2;
        LOCSTORE(store, 7, 3, STOREDIM, STOREDIM) = dp.x_7_3;
        LOCSTORE(store, 8, 1, STOREDIM, STOREDIM) = dp.x_8_1;
        LOCSTORE(store, 8, 2, STOREDIM, STOREDIM) = dp.x_8_2;
        LOCSTORE(store, 8, 3, STOREDIM, STOREDIM) = dp.x_8_3;
        LOCSTORE(store, 9, 1, STOREDIM, STOREDIM) = dp.x_9_1;
        LOCSTORE(store, 9, 2, STOREDIM, STOREDIM) = dp.x_9_2;
        LOCSTORE(store, 9, 3, STOREDIM, STOREDIM) = dp.x_9_3;

        FSint_0 fs(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

#if defined(REG_FS)
        LOCSTORE(store, 10, 0, STOREDIM, STOREDIM) = fs.x_10_0;
        LOCSTORE(store, 11, 0, STOREDIM, STOREDIM) = fs.x_11_0;
        LOCSTORE(store, 12, 0, STOREDIM, STOREDIM) = fs.x_12_0;
        LOCSTORE(store, 13, 0, STOREDIM, STOREDIM) = fs.x_13_0;
        LOCSTORE(store, 14, 0, STOREDIM, STOREDIM) = fs.x_14_0;
        LOCSTORE(store, 15, 0, STOREDIM, STOREDIM) = fs.x_15_0;
        LOCSTORE(store, 16, 0, STOREDIM, STOREDIM) = fs.x_16_0;
        LOCSTORE(store, 17, 0, STOREDIM, STOREDIM) = fs.x_17_0;
        LOCSTORE(store, 18, 0, STOREDIM, STOREDIM) = fs.x_18_0;
        LOCSTORE(store, 19, 0, STOREDIM, STOREDIM) = fs.x_19_0;
#endif

#if defined(DEBUG_OEI)
        printf("II %d JJ %d PS store[1,0] = %f \n", II, JJ, LOCSTORE(store, 1, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[2,0] = %f \n", II, JJ, LOCSTORE(store, 2, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PS store[3,0] = %f \n", II, JJ, LOCSTORE(store, 3, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,1] = %f \n", II, JJ, LOCSTORE(store, 4, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,2] = %f \n", II, JJ, LOCSTORE(store, 4, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,3] = %f \n", II, JJ, LOCSTORE(store, 4, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,1] = %f \n", II, JJ, LOCSTORE(store, 5, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,2] = %f \n", II, JJ, LOCSTORE(store, 5, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,3] = %f \n", II, JJ, LOCSTORE(store, 5, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,1] = %f \n", II, JJ, LOCSTORE(store, 6, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,2] = %f \n", II, JJ, LOCSTORE(store, 6, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,3] = %f \n", II, JJ, LOCSTORE(store, 6, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,1] = %f \n", II, JJ, LOCSTORE(store, 7, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,2] = %f \n", II, JJ, LOCSTORE(store, 7, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,3] = %f \n", II, JJ, LOCSTORE(store, 7, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,1] = %f \n", II, JJ, LOCSTORE(store, 8, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,2] = %f \n", II, JJ, LOCSTORE(store, 8, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,3] = %f \n", II, JJ, LOCSTORE(store, 8, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,1] = %f \n", II, JJ, LOCSTORE(store, 9, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,2] = %f \n", II, JJ, LOCSTORE(store, 9, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,3] = %f \n", II, JJ, LOCSTORE(store, 9, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[10,0] = %f \n", II, JJ, LOCSTORE(store, 10, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[11,0] = %f \n", II, JJ, LOCSTORE(store, 11, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[12,0] = %f \n", II, JJ, LOCSTORE(store, 12, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[13,0] = %f \n", II, JJ, LOCSTORE(store, 13, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[14,0] = %f \n", II, JJ, LOCSTORE(store, 14, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[15,0] = %f \n", II, JJ, LOCSTORE(store, 15, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[16,0] = %f \n", II, JJ, LOCSTORE(store, 16, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[17,0] = %f \n", II, JJ, LOCSTORE(store, 17, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[18,0] = %f \n", II, JJ, LOCSTORE(store, 18, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d FS store[19,0] = %f \n", II, JJ, LOCSTORE(store, 19, 0, STOREDIM, STOREDIM));
#endif
    }

    /* PD integral gradient, m=0 */
    else if (I == 1 && J == 2) {
        SDint_0 sd(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 0, 4, STOREDIM, STOREDIM) = sd.x_0_4;
        LOCSTORE(store, 0, 5, STOREDIM, STOREDIM) = sd.x_0_5;
        LOCSTORE(store, 0, 6, STOREDIM, STOREDIM) = sd.x_0_6;
        LOCSTORE(store, 0, 7, STOREDIM, STOREDIM) = sd.x_0_7;
        LOCSTORE(store, 0, 8, STOREDIM, STOREDIM) = sd.x_0_8;
        LOCSTORE(store, 0, 9, STOREDIM, STOREDIM) = sd.x_0_9;

        PPint_0 pp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) = pp.x_1_1;
        LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) = pp.x_1_2;
        LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) = pp.x_1_3;
        LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) = pp.x_2_1;
        LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) = pp.x_2_2;
        LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) = pp.x_2_3;
        LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) = pp.x_3_1;
        LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) = pp.x_3_2;
        LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) = pp.x_3_3;

        DDint_0 dd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

#if defined(REG_DD)
        LOCSTORE(store, 4, 4, STOREDIM, STOREDIM) = dd.x_4_4;
        LOCSTORE(store, 4, 5, STOREDIM, STOREDIM) = dd.x_4_5;
        LOCSTORE(store, 4, 6, STOREDIM, STOREDIM) = dd.x_4_6;
        LOCSTORE(store, 4, 7, STOREDIM, STOREDIM) = dd.x_4_7;
        LOCSTORE(store, 4, 8, STOREDIM, STOREDIM) = dd.x_4_8;
        LOCSTORE(store, 4, 9, STOREDIM, STOREDIM) = dd.x_4_9;
        LOCSTORE(store, 5, 4, STOREDIM, STOREDIM) = dd.x_5_4;
        LOCSTORE(store, 5, 5, STOREDIM, STOREDIM) = dd.x_5_5;
        LOCSTORE(store, 5, 6, STOREDIM, STOREDIM) = dd.x_5_6;
        LOCSTORE(store, 5, 7, STOREDIM, STOREDIM) = dd.x_5_7;
        LOCSTORE(store, 5, 8, STOREDIM, STOREDIM) = dd.x_5_8;
        LOCSTORE(store, 5, 9, STOREDIM, STOREDIM) = dd.x_5_9;
        LOCSTORE(store, 6, 4, STOREDIM, STOREDIM) = dd.x_6_4;
        LOCSTORE(store, 6, 5, STOREDIM, STOREDIM) = dd.x_6_5;
        LOCSTORE(store, 6, 6, STOREDIM, STOREDIM) = dd.x_6_6;
        LOCSTORE(store, 6, 7, STOREDIM, STOREDIM) = dd.x_6_7;
        LOCSTORE(store, 6, 8, STOREDIM, STOREDIM) = dd.x_6_8;
        LOCSTORE(store, 6, 9, STOREDIM, STOREDIM) = dd.x_6_9;
        LOCSTORE(store, 7, 4, STOREDIM, STOREDIM) = dd.x_7_4;
        LOCSTORE(store, 7, 5, STOREDIM, STOREDIM) = dd.x_7_5;
        LOCSTORE(store, 7, 6, STOREDIM, STOREDIM) = dd.x_7_6;
        LOCSTORE(store, 7, 7, STOREDIM, STOREDIM) = dd.x_7_7;
        LOCSTORE(store, 7, 8, STOREDIM, STOREDIM) = dd.x_7_8;
        LOCSTORE(store, 7, 9, STOREDIM, STOREDIM) = dd.x_7_9;
        LOCSTORE(store, 8, 4, STOREDIM, STOREDIM) = dd.x_8_4;
        LOCSTORE(store, 8, 5, STOREDIM, STOREDIM) = dd.x_8_5;
        LOCSTORE(store, 8, 6, STOREDIM, STOREDIM) = dd.x_8_6;
        LOCSTORE(store, 8, 7, STOREDIM, STOREDIM) = dd.x_8_7;
        LOCSTORE(store, 8, 8, STOREDIM, STOREDIM) = dd.x_8_8;
        LOCSTORE(store, 8, 9, STOREDIM, STOREDIM) = dd.x_8_9;
        LOCSTORE(store, 9, 4, STOREDIM, STOREDIM) = dd.x_9_4;
        LOCSTORE(store, 9, 5, STOREDIM, STOREDIM) = dd.x_9_5;
        LOCSTORE(store, 9, 6, STOREDIM, STOREDIM) = dd.x_9_6;
        LOCSTORE(store, 9, 7, STOREDIM, STOREDIM) = dd.x_9_7;
        LOCSTORE(store, 9, 8, STOREDIM, STOREDIM) = dd.x_9_8;
        LOCSTORE(store, 9, 9, STOREDIM, STOREDIM) = dd.x_9_9;
#endif
#if defined(USE_PARTIAL_PF)
        {
            PFint_0_1 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 10, STOREDIM, STOREDIM) = pf.x_1_10;
            LOCSTORE(store, 2, 10, STOREDIM, STOREDIM) = pf.x_2_10;
            LOCSTORE(store, 3, 10, STOREDIM, STOREDIM) = pf.x_3_10;
        }

        {
            PFint_0_2 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 11, STOREDIM, STOREDIM) = pf.x_1_11;
            LOCSTORE(store, 2, 11, STOREDIM, STOREDIM) = pf.x_2_11;
            LOCSTORE(store, 3, 11, STOREDIM, STOREDIM) = pf.x_3_11;
        }

        {
            PFint_0_3 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 12, STOREDIM, STOREDIM) = pf.x_1_12;
            LOCSTORE(store, 2, 12, STOREDIM, STOREDIM) = pf.x_2_12;
            LOCSTORE(store, 3, 12, STOREDIM, STOREDIM) = pf.x_3_12;
        }

        {
            PFint_0_4 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 13, STOREDIM, STOREDIM) = pf.x_1_13;
            LOCSTORE(store, 2, 13, STOREDIM, STOREDIM) = pf.x_2_13;
            LOCSTORE(store, 3, 13, STOREDIM, STOREDIM) = pf.x_3_13;
        }

        {
            PFint_0_5 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 14, STOREDIM, STOREDIM) = pf.x_1_14;
            LOCSTORE(store, 2, 14, STOREDIM, STOREDIM) = pf.x_2_14;
            LOCSTORE(store, 3, 14, STOREDIM, STOREDIM) = pf.x_3_14;
        }

        {
            PFint_0_6 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 15, STOREDIM, STOREDIM) = pf.x_1_15;
            LOCSTORE(store, 2, 15, STOREDIM, STOREDIM) = pf.x_2_15;
            LOCSTORE(store, 3, 15, STOREDIM, STOREDIM) = pf.x_3_15;
        }

        {
            PFint_0_7 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 16, STOREDIM, STOREDIM) = pf.x_1_16;
            LOCSTORE(store, 2, 16, STOREDIM, STOREDIM) = pf.x_2_16;
            LOCSTORE(store, 3, 16, STOREDIM, STOREDIM) = pf.x_3_16;
        }

        {
            PFint_0_8 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 17, STOREDIM, STOREDIM) = pf.x_1_17;
            LOCSTORE(store, 2, 17, STOREDIM, STOREDIM) = pf.x_2_17;
            LOCSTORE(store, 3, 17, STOREDIM, STOREDIM) = pf.x_3_17;
        }

        {
            PFint_0_9 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 18, STOREDIM, STOREDIM) = pf.x_1_18;
            LOCSTORE(store, 2, 18, STOREDIM, STOREDIM) = pf.x_2_18;
            LOCSTORE(store, 3, 18, STOREDIM, STOREDIM) = pf.x_3_18;
        }

        {
            PFint_0_10 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 1, 19, STOREDIM, STOREDIM) = pf.x_1_19;
            LOCSTORE(store, 2, 19, STOREDIM, STOREDIM) = pf.x_2_19;
            LOCSTORE(store, 3, 19, STOREDIM, STOREDIM) = pf.x_3_19;
        }

#else
        PFint_0 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);
  #if defined(REG_PF)
        LOCSTORE(store, 1, 10, STOREDIM, STOREDIM) = pf.x_1_10;
        LOCSTORE(store, 2, 10, STOREDIM, STOREDIM) = pf.x_2_10;
        LOCSTORE(store, 3, 10, STOREDIM, STOREDIM) = pf.x_3_10;
        LOCSTORE(store, 1, 11, STOREDIM, STOREDIM) = pf.x_1_11;
        LOCSTORE(store, 2, 11, STOREDIM, STOREDIM) = pf.x_2_11;
        LOCSTORE(store, 3, 11, STOREDIM, STOREDIM) = pf.x_3_11;
        LOCSTORE(store, 1, 12, STOREDIM, STOREDIM) = pf.x_1_12;
        LOCSTORE(store, 2, 12, STOREDIM, STOREDIM) = pf.x_2_12;
        LOCSTORE(store, 3, 12, STOREDIM, STOREDIM) = pf.x_3_12;
        LOCSTORE(store, 1, 13, STOREDIM, STOREDIM) = pf.x_1_13;
        LOCSTORE(store, 2, 13, STOREDIM, STOREDIM) = pf.x_2_13;
        LOCSTORE(store, 3, 13, STOREDIM, STOREDIM) = pf.x_3_13;
        LOCSTORE(store, 1, 14, STOREDIM, STOREDIM) = pf.x_1_14;
        LOCSTORE(store, 2, 14, STOREDIM, STOREDIM) = pf.x_2_14;
        LOCSTORE(store, 3, 14, STOREDIM, STOREDIM) = pf.x_3_14;
        LOCSTORE(store, 1, 15, STOREDIM, STOREDIM) = pf.x_1_15;
        LOCSTORE(store, 2, 15, STOREDIM, STOREDIM) = pf.x_2_15;
        LOCSTORE(store, 3, 15, STOREDIM, STOREDIM) = pf.x_3_15;
        LOCSTORE(store, 1, 16, STOREDIM, STOREDIM) = pf.x_1_16;
        LOCSTORE(store, 2, 16, STOREDIM, STOREDIM) = pf.x_2_16;
        LOCSTORE(store, 3, 16, STOREDIM, STOREDIM) = pf.x_3_16;
        LOCSTORE(store, 1, 17, STOREDIM, STOREDIM) = pf.x_1_17;
        LOCSTORE(store, 2, 17, STOREDIM, STOREDIM) = pf.x_2_17;
        LOCSTORE(store, 3, 17, STOREDIM, STOREDIM) = pf.x_3_17;
        LOCSTORE(store, 1, 18, STOREDIM, STOREDIM) = pf.x_1_18;
        LOCSTORE(store, 2, 18, STOREDIM, STOREDIM) = pf.x_2_18;
        LOCSTORE(store, 3, 18, STOREDIM, STOREDIM) = pf.x_3_18;
        LOCSTORE(store, 1, 19, STOREDIM, STOREDIM) = pf.x_1_19;
        LOCSTORE(store, 2, 19, STOREDIM, STOREDIM) = pf.x_2_19;
        LOCSTORE(store, 3, 19, STOREDIM, STOREDIM) = pf.x_3_19;
  #endif
#endif

#if defined(DEBUG_OEI)
        printf("II %d JJ %d SD store[0,4] = %f \n", II, JJ, LOCSTORE(store, 0, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,5] = %f \n", II, JJ, LOCSTORE(store, 0, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,6] = %f \n", II, JJ, LOCSTORE(store, 0, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,7] = %f \n", II, JJ, LOCSTORE(store, 0, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,8] = %f \n", II, JJ, LOCSTORE(store, 0, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d SD store[0,9] = %f \n", II, JJ, LOCSTORE(store, 0, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,1] = %f \n", II, JJ, LOCSTORE(store, 1, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,2] = %f \n", II, JJ, LOCSTORE(store, 1, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,3] = %f \n", II, JJ, LOCSTORE(store, 1, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,1] = %f \n", II, JJ, LOCSTORE(store, 2, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,2] = %f \n", II, JJ, LOCSTORE(store, 2, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,3] = %f \n", II, JJ, LOCSTORE(store, 2, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,1] = %f \n", II, JJ, LOCSTORE(store, 3, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,2] = %f \n", II, JJ, LOCSTORE(store, 3, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,3] = %f \n", II, JJ, LOCSTORE(store, 3, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[4,4] = %f \n", II, JJ, LOCSTORE(store, 4, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[4,5] = %f \n", II, JJ, LOCSTORE(store, 4, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[4,6] = %f \n", II, JJ, LOCSTORE(store, 4, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[4,7] = %f \n", II, JJ, LOCSTORE(store, 4, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[4,8] = %f \n", II, JJ, LOCSTORE(store, 4, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[4,9] = %f \n", II, JJ, LOCSTORE(store, 4, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[5,4] = %f \n", II, JJ, LOCSTORE(store, 5, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[5,5] = %f \n", II, JJ, LOCSTORE(store, 5, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[5,6] = %f \n", II, JJ, LOCSTORE(store, 5, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[5,7] = %f \n", II, JJ, LOCSTORE(store, 5, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[5,8] = %f \n", II, JJ, LOCSTORE(store, 5, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[5,9] = %f \n", II, JJ, LOCSTORE(store, 5, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[6,4] = %f \n", II, JJ, LOCSTORE(store, 6, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[6,5] = %f \n", II, JJ, LOCSTORE(store, 6, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[6,6] = %f \n", II, JJ, LOCSTORE(store, 6, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[6,7] = %f \n", II, JJ, LOCSTORE(store, 6, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[6,8] = %f \n", II, JJ, LOCSTORE(store, 6, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[6,9] = %f \n", II, JJ, LOCSTORE(store, 6, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[7,4] = %f \n", II, JJ, LOCSTORE(store, 7, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[7,5] = %f \n", II, JJ, LOCSTORE(store, 7, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[7,6] = %f \n", II, JJ, LOCSTORE(store, 7, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[7,7] = %f \n", II, JJ, LOCSTORE(store, 7, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[7,8] = %f \n", II, JJ, LOCSTORE(store, 7, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[7,9] = %f \n", II, JJ, LOCSTORE(store, 7, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[8,4] = %f \n", II, JJ, LOCSTORE(store, 8, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[8,5] = %f \n", II, JJ, LOCSTORE(store, 8, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[8,6] = %f \n", II, JJ, LOCSTORE(store, 8, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[8,7] = %f \n", II, JJ, LOCSTORE(store, 8, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[8,8] = %f \n", II, JJ, LOCSTORE(store, 8, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[8,9] = %f \n", II, JJ, LOCSTORE(store, 8, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[9,4] = %f \n", II, JJ, LOCSTORE(store, 9, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[9,5] = %f \n", II, JJ, LOCSTORE(store, 9, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[9,6] = %f \n", II, JJ, LOCSTORE(store, 9, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[9,7] = %f \n", II, JJ, LOCSTORE(store, 9, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[9,8] = %f \n", II, JJ, LOCSTORE(store, 9, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store[9,9] = %f \n", II, JJ, LOCSTORE(store, 9, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,10] = %f \n", II, JJ, LOCSTORE(store, 1, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,10] = %f \n", II, JJ, LOCSTORE(store, 2, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,10] = %f \n", II, JJ, LOCSTORE(store, 3, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,11] = %f \n", II, JJ, LOCSTORE(store, 1, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,11] = %f \n", II, JJ, LOCSTORE(store, 2, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,11] = %f \n", II, JJ, LOCSTORE(store, 3, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,12] = %f \n", II, JJ, LOCSTORE(store, 1, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,12] = %f \n", II, JJ, LOCSTORE(store, 2, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,12] = %f \n", II, JJ, LOCSTORE(store, 3, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,13] = %f \n", II, JJ, LOCSTORE(store, 1, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,13] = %f \n", II, JJ, LOCSTORE(store, 2, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,13] = %f \n", II, JJ, LOCSTORE(store, 3, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,14] = %f \n", II, JJ, LOCSTORE(store, 1, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,14] = %f \n", II, JJ, LOCSTORE(store, 2, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,14] = %f \n", II, JJ, LOCSTORE(store, 3, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,15] = %f \n", II, JJ, LOCSTORE(store, 1, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,15] = %f \n", II, JJ, LOCSTORE(store, 2, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,15] = %f \n", II, JJ, LOCSTORE(store, 3, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,16] = %f \n", II, JJ, LOCSTORE(store, 1, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,16] = %f \n", II, JJ, LOCSTORE(store, 2, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,16] = %f \n", II, JJ, LOCSTORE(store, 3, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,17] = %f \n", II, JJ, LOCSTORE(store, 1, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,17] = %f \n", II, JJ, LOCSTORE(store, 2, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,17] = %f \n", II, JJ, LOCSTORE(store, 3, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,18] = %f \n", II, JJ, LOCSTORE(store, 1, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,18] = %f \n", II, JJ, LOCSTORE(store, 2, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,18] = %f \n", II, JJ, LOCSTORE(store, 3, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[1,19] = %f \n", II, JJ, LOCSTORE(store, 1, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[2,19] = %f \n", II, JJ, LOCSTORE(store, 2, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d PF store[3,19] = %f \n", II, JJ, LOCSTORE(store, 3, 19, STOREDIM, STOREDIM));
#endif
    }

    /* DP integral gradient, m=0 */
    else if (I == 2 && J == 1) {
        DSint_0 ds(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 4, 0, STOREDIM, STOREDIM) = ds.x_4_0;
        LOCSTORE(store, 5, 0, STOREDIM, STOREDIM) = ds.x_5_0;
        LOCSTORE(store, 6, 0, STOREDIM, STOREDIM) = ds.x_6_0;
        LOCSTORE(store, 7, 0, STOREDIM, STOREDIM) = ds.x_7_0;
        LOCSTORE(store, 8, 0, STOREDIM, STOREDIM) = ds.x_8_0;
        LOCSTORE(store, 9, 0, STOREDIM, STOREDIM) = ds.x_9_0;

        PPint_0 pp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) = pp.x_1_1;
        LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) = pp.x_1_2;
        LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) = pp.x_1_3;
        LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) = pp.x_2_1;
        LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) = pp.x_2_2;
        LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) = pp.x_2_3;
        LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) = pp.x_3_1;
        LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) = pp.x_3_2;
        LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) = pp.x_3_3;

        DDint_0 dd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

#if defined(REG_DD)
        LOCSTORE(store, 4, 4, STOREDIM, STOREDIM) = dd.x_4_4;
        LOCSTORE(store, 4, 5, STOREDIM, STOREDIM) = dd.x_4_5;
        LOCSTORE(store, 4, 6, STOREDIM, STOREDIM) = dd.x_4_6;
        LOCSTORE(store, 4, 7, STOREDIM, STOREDIM) = dd.x_4_7;
        LOCSTORE(store, 4, 8, STOREDIM, STOREDIM) = dd.x_4_8;
        LOCSTORE(store, 4, 9, STOREDIM, STOREDIM) = dd.x_4_9;
        LOCSTORE(store, 5, 4, STOREDIM, STOREDIM) = dd.x_5_4;
        LOCSTORE(store, 5, 5, STOREDIM, STOREDIM) = dd.x_5_5;
        LOCSTORE(store, 5, 6, STOREDIM, STOREDIM) = dd.x_5_6;
        LOCSTORE(store, 5, 7, STOREDIM, STOREDIM) = dd.x_5_7;
        LOCSTORE(store, 5, 8, STOREDIM, STOREDIM) = dd.x_5_8;
        LOCSTORE(store, 5, 9, STOREDIM, STOREDIM) = dd.x_5_9;
        LOCSTORE(store, 6, 4, STOREDIM, STOREDIM) = dd.x_6_4;
        LOCSTORE(store, 6, 5, STOREDIM, STOREDIM) = dd.x_6_5;
        LOCSTORE(store, 6, 6, STOREDIM, STOREDIM) = dd.x_6_6;
        LOCSTORE(store, 6, 7, STOREDIM, STOREDIM) = dd.x_6_7;
        LOCSTORE(store, 6, 8, STOREDIM, STOREDIM) = dd.x_6_8;
        LOCSTORE(store, 6, 9, STOREDIM, STOREDIM) = dd.x_6_9;
        LOCSTORE(store, 7, 4, STOREDIM, STOREDIM) = dd.x_7_4;
        LOCSTORE(store, 7, 5, STOREDIM, STOREDIM) = dd.x_7_5;
        LOCSTORE(store, 7, 6, STOREDIM, STOREDIM) = dd.x_7_6;
        LOCSTORE(store, 7, 7, STOREDIM, STOREDIM) = dd.x_7_7;
        LOCSTORE(store, 7, 8, STOREDIM, STOREDIM) = dd.x_7_8;
        LOCSTORE(store, 7, 9, STOREDIM, STOREDIM) = dd.x_7_9;
        LOCSTORE(store, 8, 4, STOREDIM, STOREDIM) = dd.x_8_4;
        LOCSTORE(store, 8, 5, STOREDIM, STOREDIM) = dd.x_8_5;
        LOCSTORE(store, 8, 6, STOREDIM, STOREDIM) = dd.x_8_6;
        LOCSTORE(store, 8, 7, STOREDIM, STOREDIM) = dd.x_8_7;
        LOCSTORE(store, 8, 8, STOREDIM, STOREDIM) = dd.x_8_8;
        LOCSTORE(store, 8, 9, STOREDIM, STOREDIM) = dd.x_8_9;
        LOCSTORE(store, 9, 4, STOREDIM, STOREDIM) = dd.x_9_4;
        LOCSTORE(store, 9, 5, STOREDIM, STOREDIM) = dd.x_9_5;
        LOCSTORE(store, 9, 6, STOREDIM, STOREDIM) = dd.x_9_6;
        LOCSTORE(store, 9, 7, STOREDIM, STOREDIM) = dd.x_9_7;
        LOCSTORE(store, 9, 8, STOREDIM, STOREDIM) = dd.x_9_8;
        LOCSTORE(store, 9, 9, STOREDIM, STOREDIM) = dd.x_9_9;
#endif
#if defined(USE_PARTIAL_FP)
        {
            FPint_0_1 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 10, 1, STOREDIM, STOREDIM) = fp.x_10_1;
            LOCSTORE(store, 10, 2, STOREDIM, STOREDIM) = fp.x_10_2;
            LOCSTORE(store, 10, 3, STOREDIM, STOREDIM) = fp.x_10_3;
        }

        {
            FPint_0_2 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 11, 1, STOREDIM, STOREDIM) = fp.x_11_1;
            LOCSTORE(store, 11, 2, STOREDIM, STOREDIM) = fp.x_11_2;
            LOCSTORE(store, 11, 3, STOREDIM, STOREDIM) = fp.x_11_3;
        }

        {
            FPint_0_3 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 12, 1, STOREDIM, STOREDIM) = fp.x_12_1;
            LOCSTORE(store, 12, 2, STOREDIM, STOREDIM) = fp.x_12_2;
            LOCSTORE(store, 12, 3, STOREDIM, STOREDIM) = fp.x_12_3;
        }

        {
            FPint_0_4 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 13, 1, STOREDIM, STOREDIM) = fp.x_13_1;
            LOCSTORE(store, 13, 2, STOREDIM, STOREDIM) = fp.x_13_2;
            LOCSTORE(store, 13, 3, STOREDIM, STOREDIM) = fp.x_13_3;
        }

        {
            FPint_0_5 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 14, 1, STOREDIM, STOREDIM) = fp.x_14_1;
            LOCSTORE(store, 14, 2, STOREDIM, STOREDIM) = fp.x_14_2;
            LOCSTORE(store, 14, 3, STOREDIM, STOREDIM) = fp.x_14_3;
        }

        {
            FPint_0_6 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 15, 1, STOREDIM, STOREDIM) = fp.x_15_1;
            LOCSTORE(store, 15, 2, STOREDIM, STOREDIM) = fp.x_15_2;
            LOCSTORE(store, 15, 3, STOREDIM, STOREDIM) = fp.x_15_3;
        }

        {
            FPint_0_7 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 16, 1, STOREDIM, STOREDIM) = fp.x_16_1;
            LOCSTORE(store, 16, 2, STOREDIM, STOREDIM) = fp.x_16_2;
            LOCSTORE(store, 16, 3, STOREDIM, STOREDIM) = fp.x_16_3;
        }

        {
            FPint_0_8 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 17, 1, STOREDIM, STOREDIM) = fp.x_17_1;
            LOCSTORE(store, 17, 2, STOREDIM, STOREDIM) = fp.x_17_2;
            LOCSTORE(store, 17, 3, STOREDIM, STOREDIM) = fp.x_17_3;
        }

        {
            FPint_0_9 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 18, 1, STOREDIM, STOREDIM) = fp.x_18_1;
            LOCSTORE(store, 18, 2, STOREDIM, STOREDIM) = fp.x_18_2;
            LOCSTORE(store, 18, 3, STOREDIM, STOREDIM) = fp.x_18_3;
        }

        {
            FPint_0_10 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

            LOCSTORE(store, 19, 1, STOREDIM, STOREDIM) = fp.x_19_1;
            LOCSTORE(store, 19, 2, STOREDIM, STOREDIM) = fp.x_19_2;
            LOCSTORE(store, 19, 3, STOREDIM, STOREDIM) = fp.x_19_3;
        }
#else
        FPint_0 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

  #if defined(REG_FP)
        LOCSTORE(store, 10, 1, STOREDIM, STOREDIM) = fp.x_10_1;
        LOCSTORE(store, 10, 2, STOREDIM, STOREDIM) = fp.x_10_2;
        LOCSTORE(store, 10, 3, STOREDIM, STOREDIM) = fp.x_10_3;
        LOCSTORE(store, 11, 1, STOREDIM, STOREDIM) = fp.x_11_1;
        LOCSTORE(store, 11, 2, STOREDIM, STOREDIM) = fp.x_11_2;
        LOCSTORE(store, 11, 3, STOREDIM, STOREDIM) = fp.x_11_3;
        LOCSTORE(store, 12, 1, STOREDIM, STOREDIM) = fp.x_12_1;
        LOCSTORE(store, 12, 2, STOREDIM, STOREDIM) = fp.x_12_2;
        LOCSTORE(store, 12, 3, STOREDIM, STOREDIM) = fp.x_12_3;
        LOCSTORE(store, 13, 1, STOREDIM, STOREDIM) = fp.x_13_1;
        LOCSTORE(store, 13, 2, STOREDIM, STOREDIM) = fp.x_13_2;
        LOCSTORE(store, 13, 3, STOREDIM, STOREDIM) = fp.x_13_3;
        LOCSTORE(store, 14, 1, STOREDIM, STOREDIM) = fp.x_14_1;
        LOCSTORE(store, 14, 2, STOREDIM, STOREDIM) = fp.x_14_2;
        LOCSTORE(store, 14, 3, STOREDIM, STOREDIM) = fp.x_14_3;
        LOCSTORE(store, 15, 1, STOREDIM, STOREDIM) = fp.x_15_1;
        LOCSTORE(store, 15, 2, STOREDIM, STOREDIM) = fp.x_15_2;
        LOCSTORE(store, 15, 3, STOREDIM, STOREDIM) = fp.x_15_3;
        LOCSTORE(store, 16, 1, STOREDIM, STOREDIM) = fp.x_16_1;
        LOCSTORE(store, 16, 2, STOREDIM, STOREDIM) = fp.x_16_2;
        LOCSTORE(store, 16, 3, STOREDIM, STOREDIM) = fp.x_16_3;
        LOCSTORE(store, 17, 1, STOREDIM, STOREDIM) = fp.x_17_1;
        LOCSTORE(store, 17, 2, STOREDIM, STOREDIM) = fp.x_17_2;
        LOCSTORE(store, 17, 3, STOREDIM, STOREDIM) = fp.x_17_3;
        LOCSTORE(store, 18, 1, STOREDIM, STOREDIM) = fp.x_18_1;
        LOCSTORE(store, 18, 2, STOREDIM, STOREDIM) = fp.x_18_2;
        LOCSTORE(store, 18, 3, STOREDIM, STOREDIM) = fp.x_18_3;
        LOCSTORE(store, 19, 1, STOREDIM, STOREDIM) = fp.x_19_1;
        LOCSTORE(store, 19, 2, STOREDIM, STOREDIM) = fp.x_19_2;
        LOCSTORE(store, 19, 3, STOREDIM, STOREDIM) = fp.x_19_3;
  #endif
#endif

#if defined(DEBUG_OEI)
        printf("II %d JJ %d DS store[4,0] = %f \n", II, JJ, LOCSTORE(store, 4, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[5,0] = %f \n", II, JJ, LOCSTORE(store, 5, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[6,0] = %f \n", II, JJ, LOCSTORE(store, 6, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[7,0] = %f \n", II, JJ, LOCSTORE(store, 7, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[8,0] = %f \n", II, JJ, LOCSTORE(store, 8, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d DS store[9,0] = %f \n", II, JJ, LOCSTORE(store, 9, 0, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,1] = %f \n", II, JJ, LOCSTORE(store, 1, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,2] = %f \n", II, JJ, LOCSTORE(store, 1, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[1,3] = %f \n", II, JJ, LOCSTORE(store, 1, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,1] = %f \n", II, JJ, LOCSTORE(store, 2, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,2] = %f \n", II, JJ, LOCSTORE(store, 2, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[2,3] = %f \n", II, JJ, LOCSTORE(store, 2, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,1] = %f \n", II, JJ, LOCSTORE(store, 3, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,2] = %f \n", II, JJ, LOCSTORE(store, 3, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d PP store[3,3] = %f \n", II, JJ, LOCSTORE(store, 3, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store4,4] = %f \n", II, JJ, LOCSTORE(store, 4, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store4,5] = %f \n", II, JJ, LOCSTORE(store, 4, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store4,6] = %f \n", II, JJ, LOCSTORE(store, 4, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store4,7] = %f \n", II, JJ, LOCSTORE(store, 4, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store4,8] = %f \n", II, JJ, LOCSTORE(store, 4, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store4,9] = %f \n", II, JJ, LOCSTORE(store, 4, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store5,4] = %f \n", II, JJ, LOCSTORE(store, 5, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store5,5] = %f \n", II, JJ, LOCSTORE(store, 5, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store5,6] = %f \n", II, JJ, LOCSTORE(store, 5, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store5,7] = %f \n", II, JJ, LOCSTORE(store, 5, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store5,8] = %f \n", II, JJ, LOCSTORE(store, 5, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store5,9] = %f \n", II, JJ, LOCSTORE(store, 5, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store6,4] = %f \n", II, JJ, LOCSTORE(store, 6, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store6,5] = %f \n", II, JJ, LOCSTORE(store, 6, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store6,6] = %f \n", II, JJ, LOCSTORE(store, 6, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store6,7] = %f \n", II, JJ, LOCSTORE(store, 6, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store6,8] = %f \n", II, JJ, LOCSTORE(store, 6, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store6,9] = %f \n", II, JJ, LOCSTORE(store, 6, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store7,4] = %f \n", II, JJ, LOCSTORE(store, 7, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store7,5] = %f \n", II, JJ, LOCSTORE(store, 7, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store7,6] = %f \n", II, JJ, LOCSTORE(store, 7, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store7,7] = %f \n", II, JJ, LOCSTORE(store, 7, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store7,8] = %f \n", II, JJ, LOCSTORE(store, 7, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store7,9] = %f \n", II, JJ, LOCSTORE(store, 7, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store8,4] = %f \n", II, JJ, LOCSTORE(store, 8, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store8,5] = %f \n", II, JJ, LOCSTORE(store, 8, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store8,6] = %f \n", II, JJ, LOCSTORE(store, 8, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store8,7] = %f \n", II, JJ, LOCSTORE(store, 8, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store8,8] = %f \n", II, JJ, LOCSTORE(store, 8, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store8,9] = %f \n", II, JJ, LOCSTORE(store, 8, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store9,4] = %f \n", II, JJ, LOCSTORE(store, 9, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store9,5] = %f \n", II, JJ, LOCSTORE(store, 9, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store9,6] = %f \n", II, JJ, LOCSTORE(store, 9, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store9,7] = %f \n", II, JJ, LOCSTORE(store, 9, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store9,8] = %f \n", II, JJ, LOCSTORE(store, 9, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d DD store9,9] = %f \n", II, JJ, LOCSTORE(store, 9, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[10,1] = %f \n", II, JJ, LOCSTORE(store, 10, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[10,2] = %f \n", II, JJ, LOCSTORE(store, 10, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[10,3] = %f \n", II, JJ, LOCSTORE(store, 10, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[11,1] = %f \n", II, JJ, LOCSTORE(store, 11, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[11,2] = %f \n", II, JJ, LOCSTORE(store, 11, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[11,3] = %f \n", II, JJ, LOCSTORE(store, 11, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[12,1] = %f \n", II, JJ, LOCSTORE(store, 12, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[12,2] = %f \n", II, JJ, LOCSTORE(store, 12, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[12,3] = %f \n", II, JJ, LOCSTORE(store, 12, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[13,1] = %f \n", II, JJ, LOCSTORE(store, 13, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[13,2] = %f \n", II, JJ, LOCSTORE(store, 13, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[13,3] = %f \n", II, JJ, LOCSTORE(store, 13, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[14,1] = %f \n", II, JJ, LOCSTORE(store, 14, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[14,2] = %f \n", II, JJ, LOCSTORE(store, 14, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[14,3] = %f \n", II, JJ, LOCSTORE(store, 14, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[15,1] = %f \n", II, JJ, LOCSTORE(store, 15, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[15,2] = %f \n", II, JJ, LOCSTORE(store, 15, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[15,3] = %f \n", II, JJ, LOCSTORE(store, 15, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[16,1] = %f \n", II, JJ, LOCSTORE(store, 16, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[16,2] = %f \n", II, JJ, LOCSTORE(store, 16, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[16,3] = %f \n", II, JJ, LOCSTORE(store, 16, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[17,1] = %f \n", II, JJ, LOCSTORE(store, 17, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[17,2] = %f \n", II, JJ, LOCSTORE(store, 17, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[17,3] = %f \n", II, JJ, LOCSTORE(store, 17, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[18,1] = %f \n", II, JJ, LOCSTORE(store, 18, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[18,2] = %f \n", II, JJ, LOCSTORE(store, 18, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[18,3] = %f \n", II, JJ, LOCSTORE(store, 18, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[19,1] = %f \n", II, JJ, LOCSTORE(store, 19, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[19,2] = %f \n", II, JJ, LOCSTORE(store, 19, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d FP store[19,3] = %f \n", II, JJ, LOCSTORE(store, 19, 3, STOREDIM, STOREDIM));
#endif
    }

    /* DD integral gradient, m=0 */
    else if (I == 2 && J == 2) {
        PDint_0 pd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 1, 4, STOREDIM, STOREDIM) = pd.x_1_4;
        LOCSTORE(store, 2, 4, STOREDIM, STOREDIM) = pd.x_2_4;
        LOCSTORE(store, 3, 4, STOREDIM, STOREDIM) = pd.x_3_4;
        LOCSTORE(store, 1, 5, STOREDIM, STOREDIM) = pd.x_1_5;
        LOCSTORE(store, 2, 5, STOREDIM, STOREDIM) = pd.x_2_5;
        LOCSTORE(store, 3, 5, STOREDIM, STOREDIM) = pd.x_3_5;
        LOCSTORE(store, 1, 6, STOREDIM, STOREDIM) = pd.x_1_6;
        LOCSTORE(store, 2, 6, STOREDIM, STOREDIM) = pd.x_2_6;
        LOCSTORE(store, 3, 6, STOREDIM, STOREDIM) = pd.x_3_6;
        LOCSTORE(store, 1, 7, STOREDIM, STOREDIM) = pd.x_1_7;
        LOCSTORE(store, 2, 7, STOREDIM, STOREDIM) = pd.x_2_7;
        LOCSTORE(store, 3, 7, STOREDIM, STOREDIM) = pd.x_3_7;
        LOCSTORE(store, 1, 8, STOREDIM, STOREDIM) = pd.x_1_8;
        LOCSTORE(store, 2, 8, STOREDIM, STOREDIM) = pd.x_2_8;
        LOCSTORE(store, 3, 8, STOREDIM, STOREDIM) = pd.x_3_8;
        LOCSTORE(store, 1, 9, STOREDIM, STOREDIM) = pd.x_1_9;
        LOCSTORE(store, 2, 9, STOREDIM, STOREDIM) = pd.x_2_9;
        LOCSTORE(store, 3, 9, STOREDIM, STOREDIM) = pd.x_3_9;

        DPint_0 dp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

        LOCSTORE(store, 4, 1, STOREDIM, STOREDIM) = dp.x_4_1;
        LOCSTORE(store, 4, 2, STOREDIM, STOREDIM) = dp.x_4_2;
        LOCSTORE(store, 4, 3, STOREDIM, STOREDIM) = dp.x_4_3;
        LOCSTORE(store, 5, 1, STOREDIM, STOREDIM) = dp.x_5_1;
        LOCSTORE(store, 5, 2, STOREDIM, STOREDIM) = dp.x_5_2;
        LOCSTORE(store, 5, 3, STOREDIM, STOREDIM) = dp.x_5_3;
        LOCSTORE(store, 6, 1, STOREDIM, STOREDIM) = dp.x_6_1;
        LOCSTORE(store, 6, 2, STOREDIM, STOREDIM) = dp.x_6_2;
        LOCSTORE(store, 6, 3, STOREDIM, STOREDIM) = dp.x_6_3;
        LOCSTORE(store, 7, 1, STOREDIM, STOREDIM) = dp.x_7_1;
        LOCSTORE(store, 7, 2, STOREDIM, STOREDIM) = dp.x_7_2;
        LOCSTORE(store, 7, 3, STOREDIM, STOREDIM) = dp.x_7_3;
        LOCSTORE(store, 8, 1, STOREDIM, STOREDIM) = dp.x_8_1;
        LOCSTORE(store, 8, 2, STOREDIM, STOREDIM) = dp.x_8_2;
        LOCSTORE(store, 8, 3, STOREDIM, STOREDIM) = dp.x_8_3;
        LOCSTORE(store, 9, 1, STOREDIM, STOREDIM) = dp.x_9_1;
        LOCSTORE(store, 9, 2, STOREDIM, STOREDIM) = dp.x_9_2;
        LOCSTORE(store, 9, 3, STOREDIM, STOREDIM) = dp.x_9_3;

        FDint_0 fd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

#if defined(REG_FD)
        LOCSTORE(store, 10, 4, STOREDIM, STOREDIM) = fd.x_10_4;
        LOCSTORE(store, 10, 5, STOREDIM, STOREDIM) = fd.x_10_5;
        LOCSTORE(store, 10, 6, STOREDIM, STOREDIM) = fd.x_10_6;
        LOCSTORE(store, 10, 7, STOREDIM, STOREDIM) = fd.x_10_7;
        LOCSTORE(store, 10, 8, STOREDIM, STOREDIM) = fd.x_10_8;
        LOCSTORE(store, 10, 9, STOREDIM, STOREDIM) = fd.x_10_9;
        LOCSTORE(store, 11, 4, STOREDIM, STOREDIM) = fd.x_11_4;
        LOCSTORE(store, 11, 5, STOREDIM, STOREDIM) = fd.x_11_5;
        LOCSTORE(store, 11, 6, STOREDIM, STOREDIM) = fd.x_11_6;
        LOCSTORE(store, 11, 7, STOREDIM, STOREDIM) = fd.x_11_7;
        LOCSTORE(store, 11, 8, STOREDIM, STOREDIM) = fd.x_11_8;
        LOCSTORE(store, 11, 9, STOREDIM, STOREDIM) = fd.x_11_9;
        LOCSTORE(store, 12, 4, STOREDIM, STOREDIM) = fd.x_12_4;
        LOCSTORE(store, 12, 5, STOREDIM, STOREDIM) = fd.x_12_5;
        LOCSTORE(store, 12, 6, STOREDIM, STOREDIM) = fd.x_12_6;
        LOCSTORE(store, 12, 7, STOREDIM, STOREDIM) = fd.x_12_7;
        LOCSTORE(store, 12, 8, STOREDIM, STOREDIM) = fd.x_12_8;
        LOCSTORE(store, 12, 9, STOREDIM, STOREDIM) = fd.x_12_9;
        LOCSTORE(store, 13, 4, STOREDIM, STOREDIM) = fd.x_13_4;
        LOCSTORE(store, 13, 5, STOREDIM, STOREDIM) = fd.x_13_5;
        LOCSTORE(store, 13, 6, STOREDIM, STOREDIM) = fd.x_13_6;
        LOCSTORE(store, 13, 7, STOREDIM, STOREDIM) = fd.x_13_7;
        LOCSTORE(store, 13, 8, STOREDIM, STOREDIM) = fd.x_13_8;
        LOCSTORE(store, 13, 9, STOREDIM, STOREDIM) = fd.x_13_9;
        LOCSTORE(store, 14, 4, STOREDIM, STOREDIM) = fd.x_14_4;
        LOCSTORE(store, 14, 5, STOREDIM, STOREDIM) = fd.x_14_5;
        LOCSTORE(store, 14, 6, STOREDIM, STOREDIM) = fd.x_14_6;
        LOCSTORE(store, 14, 7, STOREDIM, STOREDIM) = fd.x_14_7;
        LOCSTORE(store, 14, 8, STOREDIM, STOREDIM) = fd.x_14_8;
        LOCSTORE(store, 14, 9, STOREDIM, STOREDIM) = fd.x_14_9;
        LOCSTORE(store, 15, 4, STOREDIM, STOREDIM) = fd.x_15_4;
        LOCSTORE(store, 15, 5, STOREDIM, STOREDIM) = fd.x_15_5;
        LOCSTORE(store, 15, 6, STOREDIM, STOREDIM) = fd.x_15_6;
        LOCSTORE(store, 15, 7, STOREDIM, STOREDIM) = fd.x_15_7;
        LOCSTORE(store, 15, 8, STOREDIM, STOREDIM) = fd.x_15_8;
        LOCSTORE(store, 15, 9, STOREDIM, STOREDIM) = fd.x_15_9;
        LOCSTORE(store, 16, 4, STOREDIM, STOREDIM) = fd.x_16_4;
        LOCSTORE(store, 16, 5, STOREDIM, STOREDIM) = fd.x_16_5;
        LOCSTORE(store, 16, 6, STOREDIM, STOREDIM) = fd.x_16_6;
        LOCSTORE(store, 16, 7, STOREDIM, STOREDIM) = fd.x_16_7;
        LOCSTORE(store, 16, 8, STOREDIM, STOREDIM) = fd.x_16_8;
        LOCSTORE(store, 16, 9, STOREDIM, STOREDIM) = fd.x_16_9;
        LOCSTORE(store, 17, 4, STOREDIM, STOREDIM) = fd.x_17_4;
        LOCSTORE(store, 17, 5, STOREDIM, STOREDIM) = fd.x_17_5;
        LOCSTORE(store, 17, 6, STOREDIM, STOREDIM) = fd.x_17_6;
        LOCSTORE(store, 17, 7, STOREDIM, STOREDIM) = fd.x_17_7;
        LOCSTORE(store, 17, 8, STOREDIM, STOREDIM) = fd.x_17_8;
        LOCSTORE(store, 17, 9, STOREDIM, STOREDIM) = fd.x_17_9;
        LOCSTORE(store, 18, 4, STOREDIM, STOREDIM) = fd.x_18_4;
        LOCSTORE(store, 18, 5, STOREDIM, STOREDIM) = fd.x_18_5;
        LOCSTORE(store, 18, 6, STOREDIM, STOREDIM) = fd.x_18_6;
        LOCSTORE(store, 18, 7, STOREDIM, STOREDIM) = fd.x_18_7;
        LOCSTORE(store, 18, 8, STOREDIM, STOREDIM) = fd.x_18_8;
        LOCSTORE(store, 18, 9, STOREDIM, STOREDIM) = fd.x_18_9;
        LOCSTORE(store, 19, 4, STOREDIM, STOREDIM) = fd.x_19_4;
        LOCSTORE(store, 19, 5, STOREDIM, STOREDIM) = fd.x_19_5;
        LOCSTORE(store, 19, 6, STOREDIM, STOREDIM) = fd.x_19_6;
        LOCSTORE(store, 19, 7, STOREDIM, STOREDIM) = fd.x_19_7;
        LOCSTORE(store, 19, 8, STOREDIM, STOREDIM) = fd.x_19_8;
        LOCSTORE(store, 19, 9, STOREDIM, STOREDIM) = fd.x_19_9;
#endif

        DFint_0 df(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp);

#if defined(REG_DF)
        LOCSTORE(store, 4, 10, STOREDIM, STOREDIM) = df.x_4_10;
        LOCSTORE(store, 5, 10, STOREDIM, STOREDIM) = df.x_5_10;
        LOCSTORE(store, 6, 10, STOREDIM, STOREDIM) = df.x_6_10;
        LOCSTORE(store, 7, 10, STOREDIM, STOREDIM) = df.x_7_10;
        LOCSTORE(store, 8, 10, STOREDIM, STOREDIM) = df.x_8_10;
        LOCSTORE(store, 9, 10, STOREDIM, STOREDIM) = df.x_9_10;
        LOCSTORE(store, 4, 11, STOREDIM, STOREDIM) = df.x_4_11;
        LOCSTORE(store, 5, 11, STOREDIM, STOREDIM) = df.x_5_11;
        LOCSTORE(store, 6, 11, STOREDIM, STOREDIM) = df.x_6_11;
        LOCSTORE(store, 7, 11, STOREDIM, STOREDIM) = df.x_7_11;
        LOCSTORE(store, 8, 11, STOREDIM, STOREDIM) = df.x_8_11;
        LOCSTORE(store, 9, 11, STOREDIM, STOREDIM) = df.x_9_11;
        LOCSTORE(store, 4, 12, STOREDIM, STOREDIM) = df.x_4_12;
        LOCSTORE(store, 5, 12, STOREDIM, STOREDIM) = df.x_5_12;
        LOCSTORE(store, 6, 12, STOREDIM, STOREDIM) = df.x_6_12;
        LOCSTORE(store, 7, 12, STOREDIM, STOREDIM) = df.x_7_12;
        LOCSTORE(store, 8, 12, STOREDIM, STOREDIM) = df.x_8_12;
        LOCSTORE(store, 9, 12, STOREDIM, STOREDIM) = df.x_9_12;
        LOCSTORE(store, 4, 13, STOREDIM, STOREDIM) = df.x_4_13;
        LOCSTORE(store, 5, 13, STOREDIM, STOREDIM) = df.x_5_13;
        LOCSTORE(store, 6, 13, STOREDIM, STOREDIM) = df.x_6_13;
        LOCSTORE(store, 7, 13, STOREDIM, STOREDIM) = df.x_7_13;
        LOCSTORE(store, 8, 13, STOREDIM, STOREDIM) = df.x_8_13;
        LOCSTORE(store, 9, 13, STOREDIM, STOREDIM) = df.x_9_13;
        LOCSTORE(store, 4, 14, STOREDIM, STOREDIM) = df.x_4_14;
        LOCSTORE(store, 5, 14, STOREDIM, STOREDIM) = df.x_5_14;
        LOCSTORE(store, 6, 14, STOREDIM, STOREDIM) = df.x_6_14;
        LOCSTORE(store, 7, 14, STOREDIM, STOREDIM) = df.x_7_14;
        LOCSTORE(store, 8, 14, STOREDIM, STOREDIM) = df.x_8_14;
        LOCSTORE(store, 9, 14, STOREDIM, STOREDIM) = df.x_9_14;
        LOCSTORE(store, 4, 15, STOREDIM, STOREDIM) = df.x_4_15;
        LOCSTORE(store, 5, 15, STOREDIM, STOREDIM) = df.x_5_15;
        LOCSTORE(store, 6, 15, STOREDIM, STOREDIM) = df.x_6_15;
        LOCSTORE(store, 7, 15, STOREDIM, STOREDIM) = df.x_7_15;
        LOCSTORE(store, 8, 15, STOREDIM, STOREDIM) = df.x_8_15;
        LOCSTORE(store, 9, 15, STOREDIM, STOREDIM) = df.x_9_15;
        LOCSTORE(store, 4, 16, STOREDIM, STOREDIM) = df.x_4_16;
        LOCSTORE(store, 5, 16, STOREDIM, STOREDIM) = df.x_5_16;
        LOCSTORE(store, 6, 16, STOREDIM, STOREDIM) = df.x_6_16;
        LOCSTORE(store, 7, 16, STOREDIM, STOREDIM) = df.x_7_16;
        LOCSTORE(store, 8, 16, STOREDIM, STOREDIM) = df.x_8_16;
        LOCSTORE(store, 9, 16, STOREDIM, STOREDIM) = df.x_9_16;
        LOCSTORE(store, 4, 17, STOREDIM, STOREDIM) = df.x_4_17;
        LOCSTORE(store, 5, 17, STOREDIM, STOREDIM) = df.x_5_17;
        LOCSTORE(store, 6, 17, STOREDIM, STOREDIM) = df.x_6_17;
        LOCSTORE(store, 7, 17, STOREDIM, STOREDIM) = df.x_7_17;
        LOCSTORE(store, 8, 17, STOREDIM, STOREDIM) = df.x_8_17;
        LOCSTORE(store, 9, 17, STOREDIM, STOREDIM) = df.x_9_17;
        LOCSTORE(store, 4, 18, STOREDIM, STOREDIM) = df.x_4_18;
        LOCSTORE(store, 5, 18, STOREDIM, STOREDIM) = df.x_5_18;
        LOCSTORE(store, 6, 18, STOREDIM, STOREDIM) = df.x_6_18;
        LOCSTORE(store, 7, 18, STOREDIM, STOREDIM) = df.x_7_18;
        LOCSTORE(store, 8, 18, STOREDIM, STOREDIM) = df.x_8_18;
        LOCSTORE(store, 9, 18, STOREDIM, STOREDIM) = df.x_9_18;
        LOCSTORE(store, 4, 19, STOREDIM, STOREDIM) = df.x_4_19;
        LOCSTORE(store, 5, 19, STOREDIM, STOREDIM) = df.x_5_19;
        LOCSTORE(store, 6, 19, STOREDIM, STOREDIM) = df.x_6_19;
        LOCSTORE(store, 7, 19, STOREDIM, STOREDIM) = df.x_7_19;
        LOCSTORE(store, 8, 19, STOREDIM, STOREDIM) = df.x_8_19;
        LOCSTORE(store, 9, 19, STOREDIM, STOREDIM) = df.x_9_19;
#endif

#if defined(DEBUG_OEI)
        printf("II %d JJ %d PD store[1,4] = %f \n", II, JJ, LOCSTORE(store, 1, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,4] = %f \n", II, JJ, LOCSTORE(store, 2, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,4] = %f \n", II, JJ, LOCSTORE(store, 3, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,5] = %f \n", II, JJ, LOCSTORE(store, 1, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,5] = %f \n", II, JJ, LOCSTORE(store, 2, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,5] = %f \n", II, JJ, LOCSTORE(store, 3, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,6] = %f \n", II, JJ, LOCSTORE(store, 1, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,6] = %f \n", II, JJ, LOCSTORE(store, 2, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,6] = %f \n", II, JJ, LOCSTORE(store, 3, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,7] = %f \n", II, JJ, LOCSTORE(store, 1, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,7] = %f \n", II, JJ, LOCSTORE(store, 2, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,7] = %f \n", II, JJ, LOCSTORE(store, 3, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,8] = %f \n", II, JJ, LOCSTORE(store, 1, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,8] = %f \n", II, JJ, LOCSTORE(store, 2, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,8] = %f \n", II, JJ, LOCSTORE(store, 3, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[1,9] = %f \n", II, JJ, LOCSTORE(store, 1, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[2,9] = %f \n", II, JJ, LOCSTORE(store, 2, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d PD store[3,9] = %f \n", II, JJ, LOCSTORE(store, 3, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,1] = %f \n", II, JJ, LOCSTORE(store, 4, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,2] = %f \n", II, JJ, LOCSTORE(store, 4, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[4,3] = %f \n", II, JJ, LOCSTORE(store, 4, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,1] = %f \n", II, JJ, LOCSTORE(store, 5, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,2] = %f \n", II, JJ, LOCSTORE(store, 5, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[5,3] = %f \n", II, JJ, LOCSTORE(store, 5, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,1] = %f \n", II, JJ, LOCSTORE(store, 6, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,2] = %f \n", II, JJ, LOCSTORE(store, 6, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[6,3] = %f \n", II, JJ, LOCSTORE(store, 6, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,1] = %f \n", II, JJ, LOCSTORE(store, 7, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,2] = %f \n", II, JJ, LOCSTORE(store, 7, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[7,3] = %f \n", II, JJ, LOCSTORE(store, 7, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,1] = %f \n", II, JJ, LOCSTORE(store, 8, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,2] = %f \n", II, JJ, LOCSTORE(store, 8, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[8,3] = %f \n", II, JJ, LOCSTORE(store, 8, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,1] = %f \n", II, JJ, LOCSTORE(store, 9, 1, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,2] = %f \n", II, JJ, LOCSTORE(store, 9, 2, STOREDIM, STOREDIM));
        printf("II %d JJ %d DP store[9,3] = %f \n", II, JJ, LOCSTORE(store, 9, 3, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[10,4] = %f \n", II, JJ, LOCSTORE(store, 10, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[10,5] = %f \n", II, JJ, LOCSTORE(store, 10, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[10,6] = %f \n", II, JJ, LOCSTORE(store, 10, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[10,7] = %f \n", II, JJ, LOCSTORE(store, 10, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[10,8] = %f \n", II, JJ, LOCSTORE(store, 10, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[10,9] = %f \n", II, JJ, LOCSTORE(store, 10, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[11,4] = %f \n", II, JJ, LOCSTORE(store, 11, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[11,5] = %f \n", II, JJ, LOCSTORE(store, 11, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[11,6] = %f \n", II, JJ, LOCSTORE(store, 11, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[11,7] = %f \n", II, JJ, LOCSTORE(store, 11, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[11,8] = %f \n", II, JJ, LOCSTORE(store, 11, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[11,9] = %f \n", II, JJ, LOCSTORE(store, 11, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[12,4] = %f \n", II, JJ, LOCSTORE(store, 12, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[12,5] = %f \n", II, JJ, LOCSTORE(store, 12, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[12,6] = %f \n", II, JJ, LOCSTORE(store, 12, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[12,7] = %f \n", II, JJ, LOCSTORE(store, 12, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[12,8] = %f \n", II, JJ, LOCSTORE(store, 12, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[12,9] = %f \n", II, JJ, LOCSTORE(store, 12, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[13,4] = %f \n", II, JJ, LOCSTORE(store, 13, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[13,5] = %f \n", II, JJ, LOCSTORE(store, 13, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[13,6] = %f \n", II, JJ, LOCSTORE(store, 13, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[13,7] = %f \n", II, JJ, LOCSTORE(store, 13, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[13,8] = %f \n", II, JJ, LOCSTORE(store, 13, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[13,9] = %f \n", II, JJ, LOCSTORE(store, 13, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[14,4] = %f \n", II, JJ, LOCSTORE(store, 14, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[14,5] = %f \n", II, JJ, LOCSTORE(store, 14, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[14,6] = %f \n", II, JJ, LOCSTORE(store, 14, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[14,7] = %f \n", II, JJ, LOCSTORE(store, 14, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[14,8] = %f \n", II, JJ, LOCSTORE(store, 14, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[14,9] = %f \n", II, JJ, LOCSTORE(store, 14, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[15,4] = %f \n", II, JJ, LOCSTORE(store, 15, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[15,5] = %f \n", II, JJ, LOCSTORE(store, 15, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[15,6] = %f \n", II, JJ, LOCSTORE(store, 15, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[15,7] = %f \n", II, JJ, LOCSTORE(store, 15, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[15,8] = %f \n", II, JJ, LOCSTORE(store, 15, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[15,9] = %f \n", II, JJ, LOCSTORE(store, 15, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[16,4] = %f \n", II, JJ, LOCSTORE(store, 16, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[16,5] = %f \n", II, JJ, LOCSTORE(store, 16, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[16,6] = %f \n", II, JJ, LOCSTORE(store, 16, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[16,7] = %f \n", II, JJ, LOCSTORE(store, 16, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[16,8] = %f \n", II, JJ, LOCSTORE(store, 16, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[16,9] = %f \n", II, JJ, LOCSTORE(store, 16, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[17,4] = %f \n", II, JJ, LOCSTORE(store, 17, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[17,5] = %f \n", II, JJ, LOCSTORE(store, 17, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[17,6] = %f \n", II, JJ, LOCSTORE(store, 17, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[17,7] = %f \n", II, JJ, LOCSTORE(store, 17, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[17,8] = %f \n", II, JJ, LOCSTORE(store, 17, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[17,9] = %f \n", II, JJ, LOCSTORE(store, 17, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[18,4] = %f \n", II, JJ, LOCSTORE(store, 18, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[18,5] = %f \n", II, JJ, LOCSTORE(store, 18, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[18,6] = %f \n", II, JJ, LOCSTORE(store, 18, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[18,7] = %f \n", II, JJ, LOCSTORE(store, 18, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[18,8] = %f \n", II, JJ, LOCSTORE(store, 18, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[18,9] = %f \n", II, JJ, LOCSTORE(store, 18, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[19,4] = %f \n", II, JJ, LOCSTORE(store, 19, 4, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[19,5] = %f \n", II, JJ, LOCSTORE(store, 19, 5, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[19,6] = %f \n", II, JJ, LOCSTORE(store, 19, 6, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[19,7] = %f \n", II, JJ, LOCSTORE(store, 19, 7, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[19,8] = %f \n", II, JJ, LOCSTORE(store, 19, 8, STOREDIM, STOREDIM));
        printf("II %d JJ %d FD store[19,9] = %f \n", II, JJ, LOCSTORE(store, 19, 9, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,10] = %f \n", II, JJ, LOCSTORE(store, 4, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,10] = %f \n", II, JJ, LOCSTORE(store, 5, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,10] = %f \n", II, JJ, LOCSTORE(store, 6, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,10] = %f \n", II, JJ, LOCSTORE(store, 7, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,10] = %f \n", II, JJ, LOCSTORE(store, 8, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,10] = %f \n", II, JJ, LOCSTORE(store, 9, 10, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,11] = %f \n", II, JJ, LOCSTORE(store, 4, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,11] = %f \n", II, JJ, LOCSTORE(store, 5, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,11] = %f \n", II, JJ, LOCSTORE(store, 6, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,11] = %f \n", II, JJ, LOCSTORE(store, 7, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,11] = %f \n", II, JJ, LOCSTORE(store, 8, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,11] = %f \n", II, JJ, LOCSTORE(store, 9, 11, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,12] = %f \n", II, JJ, LOCSTORE(store, 4, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,12] = %f \n", II, JJ, LOCSTORE(store, 5, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,12] = %f \n", II, JJ, LOCSTORE(store, 6, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,12] = %f \n", II, JJ, LOCSTORE(store, 7, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,12] = %f \n", II, JJ, LOCSTORE(store, 8, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,12] = %f \n", II, JJ, LOCSTORE(store, 9, 12, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,13] = %f \n", II, JJ, LOCSTORE(store, 4, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,13] = %f \n", II, JJ, LOCSTORE(store, 5, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,13] = %f \n", II, JJ, LOCSTORE(store, 6, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,13] = %f \n", II, JJ, LOCSTORE(store, 7, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,13] = %f \n", II, JJ, LOCSTORE(store, 8, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,13] = %f \n", II, JJ, LOCSTORE(store, 9, 13, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,14] = %f \n", II, JJ, LOCSTORE(store, 4, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,14] = %f \n", II, JJ, LOCSTORE(store, 5, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,14] = %f \n", II, JJ, LOCSTORE(store, 6, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,14] = %f \n", II, JJ, LOCSTORE(store, 7, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,14] = %f \n", II, JJ, LOCSTORE(store, 8, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,14] = %f \n", II, JJ, LOCSTORE(store, 9, 14, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,15] = %f \n", II, JJ, LOCSTORE(store, 4, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,15] = %f \n", II, JJ, LOCSTORE(store, 5, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,15] = %f \n", II, JJ, LOCSTORE(store, 6, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,15] = %f \n", II, JJ, LOCSTORE(store, 7, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,15] = %f \n", II, JJ, LOCSTORE(store, 8, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,15] = %f \n", II, JJ, LOCSTORE(store, 9, 15, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,16] = %f \n", II, JJ, LOCSTORE(store, 4, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,16] = %f \n", II, JJ, LOCSTORE(store, 5, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,16] = %f \n", II, JJ, LOCSTORE(store, 6, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,16] = %f \n", II, JJ, LOCSTORE(store, 7, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,16] = %f \n", II, JJ, LOCSTORE(store, 8, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,16] = %f \n", II, JJ, LOCSTORE(store, 9, 16, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,17] = %f \n", II, JJ, LOCSTORE(store, 4, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,17] = %f \n", II, JJ, LOCSTORE(store, 5, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,17] = %f \n", II, JJ, LOCSTORE(store, 6, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,17] = %f \n", II, JJ, LOCSTORE(store, 7, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,17] = %f \n", II, JJ, LOCSTORE(store, 8, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,17] = %f \n", II, JJ, LOCSTORE(store, 9, 17, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,18] = %f \n", II, JJ, LOCSTORE(store, 4, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,18] = %f \n", II, JJ, LOCSTORE(store, 5, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,18] = %f \n", II, JJ, LOCSTORE(store, 6, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,18] = %f \n", II, JJ, LOCSTORE(store, 7, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,18] = %f \n", II, JJ, LOCSTORE(store, 8, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,18] = %f \n", II, JJ, LOCSTORE(store, 9, 18, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[24,19] = %f \n", II, JJ, LOCSTORE(store, 4, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[25,19] = %f \n", II, JJ, LOCSTORE(store, 5, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[26,19] = %f \n", II, JJ, LOCSTORE(store, 6, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[27,19] = %f \n", II, JJ, LOCSTORE(store, 7, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[28,19] = %f \n", II, JJ, LOCSTORE(store, 8, 19, STOREDIM, STOREDIM));
        printf("II %d JJ %d DF store[29,19] = %f \n", II, JJ, LOCSTORE(store, 9, 19, STOREDIM, STOREDIM));
#endif
    }
}
