/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 16/07/2021                !
 !                                                                     !
 ! Copyright (C) 2020-2021 Merz lab                                    !
 ! Copyright (C) 2020-2021 Götz lab                                    !
 !                                                                     !
 ! This Source Code Form is subject to the terms of the Mozilla Public !
 ! License, v. 2.0. If a copy of the MPL was not distributed with this !
 ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
 !_____________________________________________________________________!
*/


__device__ __inline__ void OEint_vertical(int I, int J, int II, int JJ,QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
        QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta,
        QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  /* SS integral, m=0 */ 
  if(I == 0 && J == 0){ 
    LOC2(store, 0, 0, STOREDIM, STOREDIM) += VY(0, 0, 0);

#ifdef DEBUG_OEI 
    printf("II %d JJ %d SS store[0,0] = %f \n", II, JJ, LOC2(store, 0, 0, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* PS integral, m=0 */ 
  if(I == 1 && J == 0){ 
    PSint_0 ps(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); 
    LOC2(store, 1, 0, STOREDIM, STOREDIM) += ps.x_1_0;
    LOC2(store, 2, 0, STOREDIM, STOREDIM) += ps.x_2_0;
    LOC2(store, 3, 0, STOREDIM, STOREDIM) += ps.x_3_0;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d PS store[1,0] = %f \n", II, JJ, LOC2(store, 1, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PS store[2,0] = %f \n", II, JJ, LOC2(store, 2, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PS store[3,0] = %f \n", II, JJ, LOC2(store, 3, 0, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* SP integral, m=0 */ 
  if(I == 0 && J == 1){ 
    SPint_0 sp(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); 
    LOC2(store, 0, 1, STOREDIM, STOREDIM) += sp.x_0_1;
    LOC2(store, 0, 2, STOREDIM, STOREDIM) += sp.x_0_2;
    LOC2(store, 0, 3, STOREDIM, STOREDIM) += sp.x_0_3;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d SP store[0,1] = %f \n", II, JJ, LOC2(store, 0, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SP store[0,2] = %f \n", II, JJ, LOC2(store, 0, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SP store[0,3] = %f \n", II, JJ, LOC2(store, 0, 3, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* PP integral, m=0 */ 
  if(I == 1 && J == 1){ 
    PPint_0 pp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 1, 1, STOREDIM, STOREDIM) += pp.x_1_1;
    LOC2(store, 1, 2, STOREDIM, STOREDIM) += pp.x_1_2;
    LOC2(store, 1, 3, STOREDIM, STOREDIM) += pp.x_1_3;
    LOC2(store, 2, 1, STOREDIM, STOREDIM) += pp.x_2_1;
    LOC2(store, 2, 2, STOREDIM, STOREDIM) += pp.x_2_2;
    LOC2(store, 2, 3, STOREDIM, STOREDIM) += pp.x_2_3;
    LOC2(store, 3, 1, STOREDIM, STOREDIM) += pp.x_3_1;
    LOC2(store, 3, 2, STOREDIM, STOREDIM) += pp.x_3_2;
    LOC2(store, 3, 3, STOREDIM, STOREDIM) += pp.x_3_3;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d PP store[1,1] = %f \n", II, JJ, LOC2(store, 1, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[1,2] = %f \n", II, JJ, LOC2(store, 1, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[1,3] = %f \n", II, JJ, LOC2(store, 1, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[2,1] = %f \n", II, JJ, LOC2(store, 2, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[2,2] = %f \n", II, JJ, LOC2(store, 2, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[2,3] = %f \n", II, JJ, LOC2(store, 2, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[3,1] = %f \n", II, JJ, LOC2(store, 3, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[3,2] = %f \n", II, JJ, LOC2(store, 3, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PP store[3,3] = %f \n", II, JJ, LOC2(store, 3, 3, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* DS integral, m=0 */ 
  if(I == 2 && J == 0){ 
    DSint_0 ds(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 0, STOREDIM, STOREDIM) += ds.x_4_0;
    LOC2(store, 5, 0, STOREDIM, STOREDIM) += ds.x_5_0;
    LOC2(store, 6, 0, STOREDIM, STOREDIM) += ds.x_6_0;
    LOC2(store, 7, 0, STOREDIM, STOREDIM) += ds.x_7_0;
    LOC2(store, 8, 0, STOREDIM, STOREDIM) += ds.x_8_0;
    LOC2(store, 9, 0, STOREDIM, STOREDIM) += ds.x_9_0;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d DS store[4,0] = %f \n", II, JJ, LOC2(store, 4, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DS store[5,0] = %f \n", II, JJ, LOC2(store, 5, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DS store[6,0] = %f \n", II, JJ, LOC2(store, 6, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DS store[7,0] = %f \n", II, JJ, LOC2(store, 7, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DS store[8,0] = %f \n", II, JJ, LOC2(store, 8, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DS store[9,0] = %f \n", II, JJ, LOC2(store, 9, 0, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* SD integral, m=0 */ 
  if(I == 0 && J == 2){ 
    SDint_0 sd(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 0, 4, STOREDIM, STOREDIM) += sd.x_0_4;
    LOC2(store, 0, 5, STOREDIM, STOREDIM) += sd.x_0_5;
    LOC2(store, 0, 6, STOREDIM, STOREDIM) += sd.x_0_6;
    LOC2(store, 0, 7, STOREDIM, STOREDIM) += sd.x_0_7;
    LOC2(store, 0, 8, STOREDIM, STOREDIM) += sd.x_0_8;
    LOC2(store, 0, 9, STOREDIM, STOREDIM) += sd.x_0_9;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d SD store[0,4] = %f \n", II, JJ, LOC2(store, 0, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SD store[0,5] = %f \n", II, JJ, LOC2(store, 0, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SD store[0,6] = %f \n", II, JJ, LOC2(store, 0, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SD store[0,7] = %f \n", II, JJ, LOC2(store, 0, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SD store[0,8] = %f \n", II, JJ, LOC2(store, 0, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SD store[0,9] = %f \n", II, JJ, LOC2(store, 0, 9, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* DP integral, m=0 */ 
  if(I == 2 && J == 1){ 
    DPint_0 dp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 1, STOREDIM, STOREDIM) += dp.x_4_1;
    LOC2(store, 4, 2, STOREDIM, STOREDIM) += dp.x_4_2;
    LOC2(store, 4, 3, STOREDIM, STOREDIM) += dp.x_4_3;
    LOC2(store, 5, 1, STOREDIM, STOREDIM) += dp.x_5_1;
    LOC2(store, 5, 2, STOREDIM, STOREDIM) += dp.x_5_2;
    LOC2(store, 5, 3, STOREDIM, STOREDIM) += dp.x_5_3;
    LOC2(store, 6, 1, STOREDIM, STOREDIM) += dp.x_6_1;
    LOC2(store, 6, 2, STOREDIM, STOREDIM) += dp.x_6_2;
    LOC2(store, 6, 3, STOREDIM, STOREDIM) += dp.x_6_3;
    LOC2(store, 7, 1, STOREDIM, STOREDIM) += dp.x_7_1;
    LOC2(store, 7, 2, STOREDIM, STOREDIM) += dp.x_7_2;
    LOC2(store, 7, 3, STOREDIM, STOREDIM) += dp.x_7_3;
    LOC2(store, 8, 1, STOREDIM, STOREDIM) += dp.x_8_1;
    LOC2(store, 8, 2, STOREDIM, STOREDIM) += dp.x_8_2;
    LOC2(store, 8, 3, STOREDIM, STOREDIM) += dp.x_8_3;
    LOC2(store, 9, 1, STOREDIM, STOREDIM) += dp.x_9_1;
    LOC2(store, 9, 2, STOREDIM, STOREDIM) += dp.x_9_2;
    LOC2(store, 9, 3, STOREDIM, STOREDIM) += dp.x_9_3;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d DP store[4,1] = %f \n", II, JJ, LOC2(store, 4, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[4,2] = %f \n", II, JJ, LOC2(store, 4, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[4,3] = %f \n", II, JJ, LOC2(store, 4, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[5,1] = %f \n", II, JJ, LOC2(store, 5, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[5,2] = %f \n", II, JJ, LOC2(store, 5, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[5,3] = %f \n", II, JJ, LOC2(store, 5, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[6,1] = %f \n", II, JJ, LOC2(store, 6, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[6,2] = %f \n", II, JJ, LOC2(store, 6, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[6,3] = %f \n", II, JJ, LOC2(store, 6, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[7,1] = %f \n", II, JJ, LOC2(store, 7, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[7,2] = %f \n", II, JJ, LOC2(store, 7, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[7,3] = %f \n", II, JJ, LOC2(store, 7, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[8,1] = %f \n", II, JJ, LOC2(store, 8, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[8,2] = %f \n", II, JJ, LOC2(store, 8, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[8,3] = %f \n", II, JJ, LOC2(store, 8, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[9,1] = %f \n", II, JJ, LOC2(store, 9, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[9,2] = %f \n", II, JJ, LOC2(store, 9, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DP store[9,3] = %f \n", II, JJ, LOC2(store, 9, 3, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* PD integral, m=0 */ 
  if(I == 1 && J == 2){ 
    PDint_0 pd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 1, 4, STOREDIM, STOREDIM) += pd.x_1_4;
    LOC2(store, 2, 4, STOREDIM, STOREDIM) += pd.x_2_4;
    LOC2(store, 3, 4, STOREDIM, STOREDIM) += pd.x_3_4;
    LOC2(store, 1, 5, STOREDIM, STOREDIM) += pd.x_1_5;
    LOC2(store, 2, 5, STOREDIM, STOREDIM) += pd.x_2_5;
    LOC2(store, 3, 5, STOREDIM, STOREDIM) += pd.x_3_5;
    LOC2(store, 1, 6, STOREDIM, STOREDIM) += pd.x_1_6;
    LOC2(store, 2, 6, STOREDIM, STOREDIM) += pd.x_2_6;
    LOC2(store, 3, 6, STOREDIM, STOREDIM) += pd.x_3_6;
    LOC2(store, 1, 7, STOREDIM, STOREDIM) += pd.x_1_7;
    LOC2(store, 2, 7, STOREDIM, STOREDIM) += pd.x_2_7;
    LOC2(store, 3, 7, STOREDIM, STOREDIM) += pd.x_3_7;
    LOC2(store, 1, 8, STOREDIM, STOREDIM) += pd.x_1_8;
    LOC2(store, 2, 8, STOREDIM, STOREDIM) += pd.x_2_8;
    LOC2(store, 3, 8, STOREDIM, STOREDIM) += pd.x_3_8;
    LOC2(store, 1, 9, STOREDIM, STOREDIM) += pd.x_1_9;
    LOC2(store, 2, 9, STOREDIM, STOREDIM) += pd.x_2_9;
    LOC2(store, 3, 9, STOREDIM, STOREDIM) += pd.x_3_9;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d PD store[1,4] = %f \n", II, JJ, LOC2(store, 1, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[2,4] = %f \n", II, JJ, LOC2(store, 2, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[3,4] = %f \n", II, JJ, LOC2(store, 3, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[1,5] = %f \n", II, JJ, LOC2(store, 1, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[2,5] = %f \n", II, JJ, LOC2(store, 2, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[3,5] = %f \n", II, JJ, LOC2(store, 3, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[1,6] = %f \n", II, JJ, LOC2(store, 1, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[2,6] = %f \n", II, JJ, LOC2(store, 2, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[3,6] = %f \n", II, JJ, LOC2(store, 3, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[1,7] = %f \n", II, JJ, LOC2(store, 1, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[2,7] = %f \n", II, JJ, LOC2(store, 2, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[3,7] = %f \n", II, JJ, LOC2(store, 3, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[1,8] = %f \n", II, JJ, LOC2(store, 1, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[2,8] = %f \n", II, JJ, LOC2(store, 2, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[3,8] = %f \n", II, JJ, LOC2(store, 3, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[1,9] = %f \n", II, JJ, LOC2(store, 1, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[2,9] = %f \n", II, JJ, LOC2(store, 2, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PD store[3,9] = %f \n", II, JJ, LOC2(store, 3, 9, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* DD integral, m=0 */ 
  if(I == 2 && J == 2){ 
    DDint_0 dd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 4, STOREDIM, STOREDIM) += dd.x_4_4;
    LOC2(store, 5, 4, STOREDIM, STOREDIM) += dd.x_5_4;
    LOC2(store, 6, 4, STOREDIM, STOREDIM) += dd.x_6_4;
    LOC2(store, 7, 4, STOREDIM, STOREDIM) += dd.x_7_4;
    LOC2(store, 8, 4, STOREDIM, STOREDIM) += dd.x_8_4;
    LOC2(store, 9, 4, STOREDIM, STOREDIM) += dd.x_9_4;
    LOC2(store, 4, 5, STOREDIM, STOREDIM) += dd.x_4_5;
    LOC2(store, 5, 5, STOREDIM, STOREDIM) += dd.x_5_5;
    LOC2(store, 6, 5, STOREDIM, STOREDIM) += dd.x_6_5;
    LOC2(store, 7, 5, STOREDIM, STOREDIM) += dd.x_7_5;
    LOC2(store, 8, 5, STOREDIM, STOREDIM) += dd.x_8_5;
    LOC2(store, 9, 5, STOREDIM, STOREDIM) += dd.x_9_5;
    LOC2(store, 4, 6, STOREDIM, STOREDIM) += dd.x_4_6;
    LOC2(store, 5, 6, STOREDIM, STOREDIM) += dd.x_5_6;
    LOC2(store, 6, 6, STOREDIM, STOREDIM) += dd.x_6_6;
    LOC2(store, 7, 6, STOREDIM, STOREDIM) += dd.x_7_6;
    LOC2(store, 8, 6, STOREDIM, STOREDIM) += dd.x_8_6;
    LOC2(store, 9, 6, STOREDIM, STOREDIM) += dd.x_9_6;
    LOC2(store, 4, 7, STOREDIM, STOREDIM) += dd.x_4_7;
    LOC2(store, 5, 7, STOREDIM, STOREDIM) += dd.x_5_7;
    LOC2(store, 6, 7, STOREDIM, STOREDIM) += dd.x_6_7;
    LOC2(store, 7, 7, STOREDIM, STOREDIM) += dd.x_7_7;
    LOC2(store, 8, 7, STOREDIM, STOREDIM) += dd.x_8_7;
    LOC2(store, 9, 7, STOREDIM, STOREDIM) += dd.x_9_7;
    LOC2(store, 4, 8, STOREDIM, STOREDIM) += dd.x_4_8;
    LOC2(store, 5, 8, STOREDIM, STOREDIM) += dd.x_5_8;
    LOC2(store, 6, 8, STOREDIM, STOREDIM) += dd.x_6_8;
    LOC2(store, 7, 8, STOREDIM, STOREDIM) += dd.x_7_8;
    LOC2(store, 8, 8, STOREDIM, STOREDIM) += dd.x_8_8;
    LOC2(store, 9, 8, STOREDIM, STOREDIM) += dd.x_9_8;
    LOC2(store, 4, 9, STOREDIM, STOREDIM) += dd.x_4_9;
    LOC2(store, 5, 9, STOREDIM, STOREDIM) += dd.x_5_9;
    LOC2(store, 6, 9, STOREDIM, STOREDIM) += dd.x_6_9;
    LOC2(store, 7, 9, STOREDIM, STOREDIM) += dd.x_7_9;
    LOC2(store, 8, 9, STOREDIM, STOREDIM) += dd.x_8_9;
    LOC2(store, 9, 9, STOREDIM, STOREDIM) += dd.x_9_9;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d DD store[4,4] = %f \n", II, JJ, LOC2(store, 4, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[5,4] = %f \n", II, JJ, LOC2(store, 5, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[6,4] = %f \n", II, JJ, LOC2(store, 6, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[7,4] = %f \n", II, JJ, LOC2(store, 7, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[8,4] = %f \n", II, JJ, LOC2(store, 8, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[9,4] = %f \n", II, JJ, LOC2(store, 9, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[4,5] = %f \n", II, JJ, LOC2(store, 4, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[5,5] = %f \n", II, JJ, LOC2(store, 5, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[6,5] = %f \n", II, JJ, LOC2(store, 6, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[7,5] = %f \n", II, JJ, LOC2(store, 7, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[8,5] = %f \n", II, JJ, LOC2(store, 8, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[9,5] = %f \n", II, JJ, LOC2(store, 9, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[4,6] = %f \n", II, JJ, LOC2(store, 4, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[5,6] = %f \n", II, JJ, LOC2(store, 5, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[6,6] = %f \n", II, JJ, LOC2(store, 6, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[7,6] = %f \n", II, JJ, LOC2(store, 7, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[8,6] = %f \n", II, JJ, LOC2(store, 8, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[9,6] = %f \n", II, JJ, LOC2(store, 9, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[4,7] = %f \n", II, JJ, LOC2(store, 4, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[5,7] = %f \n", II, JJ, LOC2(store, 5, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[6,7] = %f \n", II, JJ, LOC2(store, 6, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[7,7] = %f \n", II, JJ, LOC2(store, 7, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[8,7] = %f \n", II, JJ, LOC2(store, 8, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[9,7] = %f \n", II, JJ, LOC2(store, 9, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[4,8] = %f \n", II, JJ, LOC2(store, 4, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[5,8] = %f \n", II, JJ, LOC2(store, 5, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[6,8] = %f \n", II, JJ, LOC2(store, 6, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[7,8] = %f \n", II, JJ, LOC2(store, 7, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[8,8] = %f \n", II, JJ, LOC2(store, 8, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[9,8] = %f \n", II, JJ, LOC2(store, 9, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[4,9] = %f \n", II, JJ, LOC2(store, 4, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[5,9] = %f \n", II, JJ, LOC2(store, 5, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[6,9] = %f \n", II, JJ, LOC2(store, 6, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[7,9] = %f \n", II, JJ, LOC2(store, 7, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[8,9] = %f \n", II, JJ, LOC2(store, 8, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DD store[9,9] = %f \n", II, JJ, LOC2(store, 9, 9, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* FS integral, m=0 */ 
  if(I == 3 && J == 0){ 
    FSint_0 fs(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 10, 0, STOREDIM, STOREDIM) += fs.x_10_0;
    LOC2(store, 11, 0, STOREDIM, STOREDIM) += fs.x_11_0;
    LOC2(store, 12, 0, STOREDIM, STOREDIM) += fs.x_12_0;
    LOC2(store, 13, 0, STOREDIM, STOREDIM) += fs.x_13_0;
    LOC2(store, 14, 0, STOREDIM, STOREDIM) += fs.x_14_0;
    LOC2(store, 15, 0, STOREDIM, STOREDIM) += fs.x_15_0;
    LOC2(store, 16, 0, STOREDIM, STOREDIM) += fs.x_16_0;
    LOC2(store, 17, 0, STOREDIM, STOREDIM) += fs.x_17_0;
    LOC2(store, 18, 0, STOREDIM, STOREDIM) += fs.x_18_0;
    LOC2(store, 19, 0, STOREDIM, STOREDIM) += fs.x_19_0;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d FS store[10,0] = %f \n", II, JJ, LOC2(store, 10, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[11,0] = %f \n", II, JJ, LOC2(store, 11, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[12,0] = %f \n", II, JJ, LOC2(store, 12, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[13,0] = %f \n", II, JJ, LOC2(store, 13, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[14,0] = %f \n", II, JJ, LOC2(store, 14, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[15,0] = %f \n", II, JJ, LOC2(store, 15, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[16,0] = %f \n", II, JJ, LOC2(store, 16, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[17,0] = %f \n", II, JJ, LOC2(store, 17, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[18,0] = %f \n", II, JJ, LOC2(store, 18, 0, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FS store[19,0] = %f \n", II, JJ, LOC2(store, 19, 0, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* SF integral, m=0 */ 
  if(I == 0 && J == 3){ 
    SFint_0 sf(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 0, 10, STOREDIM, STOREDIM) += sf.x_0_10;
    LOC2(store, 0, 11, STOREDIM, STOREDIM) += sf.x_0_11;
    LOC2(store, 0, 12, STOREDIM, STOREDIM) += sf.x_0_12;
    LOC2(store, 0, 13, STOREDIM, STOREDIM) += sf.x_0_13;
    LOC2(store, 0, 14, STOREDIM, STOREDIM) += sf.x_0_14;
    LOC2(store, 0, 15, STOREDIM, STOREDIM) += sf.x_0_15;
    LOC2(store, 0, 16, STOREDIM, STOREDIM) += sf.x_0_16;
    LOC2(store, 0, 17, STOREDIM, STOREDIM) += sf.x_0_17;
    LOC2(store, 0, 18, STOREDIM, STOREDIM) += sf.x_0_18;
    LOC2(store, 0, 19, STOREDIM, STOREDIM) += sf.x_0_19;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d SF store[0,10] = %f \n", II, JJ, LOC2(store, 0, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,11] = %f \n", II, JJ, LOC2(store, 0, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,12] = %f \n", II, JJ, LOC2(store, 0, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,13] = %f \n", II, JJ, LOC2(store, 0, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,14] = %f \n", II, JJ, LOC2(store, 0, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,15] = %f \n", II, JJ, LOC2(store, 0, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,16] = %f \n", II, JJ, LOC2(store, 0, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,17] = %f \n", II, JJ, LOC2(store, 0, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,18] = %f \n", II, JJ, LOC2(store, 0, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d SF store[0,19] = %f \n", II, JJ, LOC2(store, 0, 19, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* FP integral, m=0 */ 
  if(I == 3 && J == 1){ 
    FPint_0 fp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 10, 1, STOREDIM, STOREDIM) += fp.x_10_1;
    LOC2(store, 10, 2, STOREDIM, STOREDIM) += fp.x_10_2;
    LOC2(store, 10, 3, STOREDIM, STOREDIM) += fp.x_10_3;
    LOC2(store, 11, 1, STOREDIM, STOREDIM) += fp.x_11_1;
    LOC2(store, 11, 2, STOREDIM, STOREDIM) += fp.x_11_2;
    LOC2(store, 11, 3, STOREDIM, STOREDIM) += fp.x_11_3;
    LOC2(store, 12, 1, STOREDIM, STOREDIM) += fp.x_12_1;
    LOC2(store, 12, 2, STOREDIM, STOREDIM) += fp.x_12_2;
    LOC2(store, 12, 3, STOREDIM, STOREDIM) += fp.x_12_3;
    LOC2(store, 13, 1, STOREDIM, STOREDIM) += fp.x_13_1;
    LOC2(store, 13, 2, STOREDIM, STOREDIM) += fp.x_13_2;
    LOC2(store, 13, 3, STOREDIM, STOREDIM) += fp.x_13_3;
    LOC2(store, 14, 1, STOREDIM, STOREDIM) += fp.x_14_1;
    LOC2(store, 14, 2, STOREDIM, STOREDIM) += fp.x_14_2;
    LOC2(store, 14, 3, STOREDIM, STOREDIM) += fp.x_14_3;
    LOC2(store, 15, 1, STOREDIM, STOREDIM) += fp.x_15_1;
    LOC2(store, 15, 2, STOREDIM, STOREDIM) += fp.x_15_2;
    LOC2(store, 15, 3, STOREDIM, STOREDIM) += fp.x_15_3;
    LOC2(store, 16, 1, STOREDIM, STOREDIM) += fp.x_16_1;
    LOC2(store, 16, 2, STOREDIM, STOREDIM) += fp.x_16_2;
    LOC2(store, 16, 3, STOREDIM, STOREDIM) += fp.x_16_3;
    LOC2(store, 17, 1, STOREDIM, STOREDIM) += fp.x_17_1;
    LOC2(store, 17, 2, STOREDIM, STOREDIM) += fp.x_17_2;
    LOC2(store, 17, 3, STOREDIM, STOREDIM) += fp.x_17_3;
    LOC2(store, 18, 1, STOREDIM, STOREDIM) += fp.x_18_1;
    LOC2(store, 18, 2, STOREDIM, STOREDIM) += fp.x_18_2;
    LOC2(store, 18, 3, STOREDIM, STOREDIM) += fp.x_18_3;
    LOC2(store, 19, 1, STOREDIM, STOREDIM) += fp.x_19_1;
    LOC2(store, 19, 2, STOREDIM, STOREDIM) += fp.x_19_2;
    LOC2(store, 19, 3, STOREDIM, STOREDIM) += fp.x_19_3;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d FP store[10,1] = %f \n", II, JJ, LOC2(store, 10, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[10,2] = %f \n", II, JJ, LOC2(store, 10, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[10,3] = %f \n", II, JJ, LOC2(store, 10, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[11,1] = %f \n", II, JJ, LOC2(store, 11, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[11,2] = %f \n", II, JJ, LOC2(store, 11, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[11,3] = %f \n", II, JJ, LOC2(store, 11, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[12,1] = %f \n", II, JJ, LOC2(store, 12, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[12,2] = %f \n", II, JJ, LOC2(store, 12, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[12,3] = %f \n", II, JJ, LOC2(store, 12, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[13,1] = %f \n", II, JJ, LOC2(store, 13, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[13,2] = %f \n", II, JJ, LOC2(store, 13, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[13,3] = %f \n", II, JJ, LOC2(store, 13, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[14,1] = %f \n", II, JJ, LOC2(store, 14, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[14,2] = %f \n", II, JJ, LOC2(store, 14, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[14,3] = %f \n", II, JJ, LOC2(store, 14, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[15,1] = %f \n", II, JJ, LOC2(store, 15, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[15,2] = %f \n", II, JJ, LOC2(store, 15, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[15,3] = %f \n", II, JJ, LOC2(store, 15, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[16,1] = %f \n", II, JJ, LOC2(store, 16, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[16,2] = %f \n", II, JJ, LOC2(store, 16, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[16,3] = %f \n", II, JJ, LOC2(store, 16, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[17,1] = %f \n", II, JJ, LOC2(store, 17, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[17,2] = %f \n", II, JJ, LOC2(store, 17, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[17,3] = %f \n", II, JJ, LOC2(store, 17, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[18,1] = %f \n", II, JJ, LOC2(store, 18, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[18,2] = %f \n", II, JJ, LOC2(store, 18, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[18,3] = %f \n", II, JJ, LOC2(store, 18, 3, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[19,1] = %f \n", II, JJ, LOC2(store, 19, 1, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[19,2] = %f \n", II, JJ, LOC2(store, 19, 2, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FP store[19,3] = %f \n", II, JJ, LOC2(store, 19, 3, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* PF integral, m=0 */ 
  if(I == 1 && J == 3){ 
    PFint_0 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 1, 10, STOREDIM, STOREDIM) += pf.x_1_10;
    LOC2(store, 2, 10, STOREDIM, STOREDIM) += pf.x_2_10;
    LOC2(store, 3, 10, STOREDIM, STOREDIM) += pf.x_3_10;
    LOC2(store, 1, 11, STOREDIM, STOREDIM) += pf.x_1_11;
    LOC2(store, 2, 11, STOREDIM, STOREDIM) += pf.x_2_11;
    LOC2(store, 3, 11, STOREDIM, STOREDIM) += pf.x_3_11;
    LOC2(store, 1, 12, STOREDIM, STOREDIM) += pf.x_1_12;
    LOC2(store, 2, 12, STOREDIM, STOREDIM) += pf.x_2_12;
    LOC2(store, 3, 12, STOREDIM, STOREDIM) += pf.x_3_12;
    LOC2(store, 1, 13, STOREDIM, STOREDIM) += pf.x_1_13;
    LOC2(store, 2, 13, STOREDIM, STOREDIM) += pf.x_2_13;
    LOC2(store, 3, 13, STOREDIM, STOREDIM) += pf.x_3_13;
    LOC2(store, 1, 14, STOREDIM, STOREDIM) += pf.x_1_14;
    LOC2(store, 2, 14, STOREDIM, STOREDIM) += pf.x_2_14;
    LOC2(store, 3, 14, STOREDIM, STOREDIM) += pf.x_3_14;
    LOC2(store, 1, 15, STOREDIM, STOREDIM) += pf.x_1_15;
    LOC2(store, 2, 15, STOREDIM, STOREDIM) += pf.x_2_15;
    LOC2(store, 3, 15, STOREDIM, STOREDIM) += pf.x_3_15;
    LOC2(store, 1, 16, STOREDIM, STOREDIM) += pf.x_1_16;
    LOC2(store, 2, 16, STOREDIM, STOREDIM) += pf.x_2_16;
    LOC2(store, 3, 16, STOREDIM, STOREDIM) += pf.x_3_16;
    LOC2(store, 1, 17, STOREDIM, STOREDIM) += pf.x_1_17;
    LOC2(store, 2, 17, STOREDIM, STOREDIM) += pf.x_2_17;
    LOC2(store, 3, 17, STOREDIM, STOREDIM) += pf.x_3_17;
    LOC2(store, 1, 18, STOREDIM, STOREDIM) += pf.x_1_18;
    LOC2(store, 2, 18, STOREDIM, STOREDIM) += pf.x_2_18;
    LOC2(store, 3, 18, STOREDIM, STOREDIM) += pf.x_3_18;
    LOC2(store, 1, 19, STOREDIM, STOREDIM) += pf.x_1_19;
    LOC2(store, 2, 19, STOREDIM, STOREDIM) += pf.x_2_19;
    LOC2(store, 3, 19, STOREDIM, STOREDIM) += pf.x_3_19;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d PF store[1,10] = %f \n", II, JJ, LOC2(store, 1, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,10] = %f \n", II, JJ, LOC2(store, 2, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,10] = %f \n", II, JJ, LOC2(store, 3, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,11] = %f \n", II, JJ, LOC2(store, 1, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,11] = %f \n", II, JJ, LOC2(store, 2, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,11] = %f \n", II, JJ, LOC2(store, 3, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,12] = %f \n", II, JJ, LOC2(store, 1, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,12] = %f \n", II, JJ, LOC2(store, 2, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,12] = %f \n", II, JJ, LOC2(store, 3, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,13] = %f \n", II, JJ, LOC2(store, 1, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,13] = %f \n", II, JJ, LOC2(store, 2, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,13] = %f \n", II, JJ, LOC2(store, 3, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,14] = %f \n", II, JJ, LOC2(store, 1, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,14] = %f \n", II, JJ, LOC2(store, 2, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,14] = %f \n", II, JJ, LOC2(store, 3, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,15] = %f \n", II, JJ, LOC2(store, 1, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,15] = %f \n", II, JJ, LOC2(store, 2, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,15] = %f \n", II, JJ, LOC2(store, 3, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,16] = %f \n", II, JJ, LOC2(store, 1, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,16] = %f \n", II, JJ, LOC2(store, 2, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,16] = %f \n", II, JJ, LOC2(store, 3, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,17] = %f \n", II, JJ, LOC2(store, 1, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,17] = %f \n", II, JJ, LOC2(store, 2, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,17] = %f \n", II, JJ, LOC2(store, 3, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,18] = %f \n", II, JJ, LOC2(store, 1, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,18] = %f \n", II, JJ, LOC2(store, 2, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,18] = %f \n", II, JJ, LOC2(store, 3, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[1,19] = %f \n", II, JJ, LOC2(store, 1, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[2,19] = %f \n", II, JJ, LOC2(store, 2, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d PF store[3,19] = %f \n", II, JJ, LOC2(store, 3, 19, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* FD integral, m=0 */ 
  if(I == 3 && J == 2){ 
    FDint_0 fd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 10, 4, STOREDIM, STOREDIM) += fd.x_10_4;
    LOC2(store, 10, 5, STOREDIM, STOREDIM) += fd.x_10_5;
    LOC2(store, 10, 6, STOREDIM, STOREDIM) += fd.x_10_6;
    LOC2(store, 10, 7, STOREDIM, STOREDIM) += fd.x_10_7;
    LOC2(store, 10, 8, STOREDIM, STOREDIM) += fd.x_10_8;
    LOC2(store, 10, 9, STOREDIM, STOREDIM) += fd.x_10_9;
    LOC2(store, 11, 4, STOREDIM, STOREDIM) += fd.x_11_4;
    LOC2(store, 11, 5, STOREDIM, STOREDIM) += fd.x_11_5;
    LOC2(store, 11, 6, STOREDIM, STOREDIM) += fd.x_11_6;
    LOC2(store, 11, 7, STOREDIM, STOREDIM) += fd.x_11_7;
    LOC2(store, 11, 8, STOREDIM, STOREDIM) += fd.x_11_8;
    LOC2(store, 11, 9, STOREDIM, STOREDIM) += fd.x_11_9;
    LOC2(store, 12, 4, STOREDIM, STOREDIM) += fd.x_12_4;
    LOC2(store, 12, 5, STOREDIM, STOREDIM) += fd.x_12_5;
    LOC2(store, 12, 6, STOREDIM, STOREDIM) += fd.x_12_6;
    LOC2(store, 12, 7, STOREDIM, STOREDIM) += fd.x_12_7;
    LOC2(store, 12, 8, STOREDIM, STOREDIM) += fd.x_12_8;
    LOC2(store, 12, 9, STOREDIM, STOREDIM) += fd.x_12_9;
    LOC2(store, 13, 4, STOREDIM, STOREDIM) += fd.x_13_4;
    LOC2(store, 13, 5, STOREDIM, STOREDIM) += fd.x_13_5;
    LOC2(store, 13, 6, STOREDIM, STOREDIM) += fd.x_13_6;
    LOC2(store, 13, 7, STOREDIM, STOREDIM) += fd.x_13_7;
    LOC2(store, 13, 8, STOREDIM, STOREDIM) += fd.x_13_8;
    LOC2(store, 13, 9, STOREDIM, STOREDIM) += fd.x_13_9;
    LOC2(store, 14, 4, STOREDIM, STOREDIM) += fd.x_14_4;
    LOC2(store, 14, 5, STOREDIM, STOREDIM) += fd.x_14_5;
    LOC2(store, 14, 6, STOREDIM, STOREDIM) += fd.x_14_6;
    LOC2(store, 14, 7, STOREDIM, STOREDIM) += fd.x_14_7;
    LOC2(store, 14, 8, STOREDIM, STOREDIM) += fd.x_14_8;
    LOC2(store, 14, 9, STOREDIM, STOREDIM) += fd.x_14_9;
    LOC2(store, 15, 4, STOREDIM, STOREDIM) += fd.x_15_4;
    LOC2(store, 15, 5, STOREDIM, STOREDIM) += fd.x_15_5;
    LOC2(store, 15, 6, STOREDIM, STOREDIM) += fd.x_15_6;
    LOC2(store, 15, 7, STOREDIM, STOREDIM) += fd.x_15_7;
    LOC2(store, 15, 8, STOREDIM, STOREDIM) += fd.x_15_8;
    LOC2(store, 15, 9, STOREDIM, STOREDIM) += fd.x_15_9;
    LOC2(store, 16, 4, STOREDIM, STOREDIM) += fd.x_16_4;
    LOC2(store, 16, 5, STOREDIM, STOREDIM) += fd.x_16_5;
    LOC2(store, 16, 6, STOREDIM, STOREDIM) += fd.x_16_6;
    LOC2(store, 16, 7, STOREDIM, STOREDIM) += fd.x_16_7;
    LOC2(store, 16, 8, STOREDIM, STOREDIM) += fd.x_16_8;
    LOC2(store, 16, 9, STOREDIM, STOREDIM) += fd.x_16_9;
    LOC2(store, 17, 4, STOREDIM, STOREDIM) += fd.x_17_4;
    LOC2(store, 17, 5, STOREDIM, STOREDIM) += fd.x_17_5;
    LOC2(store, 17, 6, STOREDIM, STOREDIM) += fd.x_17_6;
    LOC2(store, 17, 7, STOREDIM, STOREDIM) += fd.x_17_7;
    LOC2(store, 17, 8, STOREDIM, STOREDIM) += fd.x_17_8;
    LOC2(store, 17, 9, STOREDIM, STOREDIM) += fd.x_17_9;
    LOC2(store, 18, 4, STOREDIM, STOREDIM) += fd.x_18_4;
    LOC2(store, 18, 5, STOREDIM, STOREDIM) += fd.x_18_5;
    LOC2(store, 18, 6, STOREDIM, STOREDIM) += fd.x_18_6;
    LOC2(store, 18, 7, STOREDIM, STOREDIM) += fd.x_18_7;
    LOC2(store, 18, 8, STOREDIM, STOREDIM) += fd.x_18_8;
    LOC2(store, 18, 9, STOREDIM, STOREDIM) += fd.x_18_9;
    LOC2(store, 19, 4, STOREDIM, STOREDIM) += fd.x_19_4;
    LOC2(store, 19, 5, STOREDIM, STOREDIM) += fd.x_19_5;
    LOC2(store, 19, 6, STOREDIM, STOREDIM) += fd.x_19_6;
    LOC2(store, 19, 7, STOREDIM, STOREDIM) += fd.x_19_7;
    LOC2(store, 19, 8, STOREDIM, STOREDIM) += fd.x_19_8;
    LOC2(store, 19, 9, STOREDIM, STOREDIM) += fd.x_19_9;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d FD store[10,4] = %f \n", II, JJ, LOC2(store, 10, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[10,5] = %f \n", II, JJ, LOC2(store, 10, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[10,6] = %f \n", II, JJ, LOC2(store, 10, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[10,7] = %f \n", II, JJ, LOC2(store, 10, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[10,8] = %f \n", II, JJ, LOC2(store, 10, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[10,9] = %f \n", II, JJ, LOC2(store, 10, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[11,4] = %f \n", II, JJ, LOC2(store, 11, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[11,5] = %f \n", II, JJ, LOC2(store, 11, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[11,6] = %f \n", II, JJ, LOC2(store, 11, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[11,7] = %f \n", II, JJ, LOC2(store, 11, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[11,8] = %f \n", II, JJ, LOC2(store, 11, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[11,9] = %f \n", II, JJ, LOC2(store, 11, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[12,4] = %f \n", II, JJ, LOC2(store, 12, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[12,5] = %f \n", II, JJ, LOC2(store, 12, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[12,6] = %f \n", II, JJ, LOC2(store, 12, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[12,7] = %f \n", II, JJ, LOC2(store, 12, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[12,8] = %f \n", II, JJ, LOC2(store, 12, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[12,9] = %f \n", II, JJ, LOC2(store, 12, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[13,4] = %f \n", II, JJ, LOC2(store, 13, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[13,5] = %f \n", II, JJ, LOC2(store, 13, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[13,6] = %f \n", II, JJ, LOC2(store, 13, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[13,7] = %f \n", II, JJ, LOC2(store, 13, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[13,8] = %f \n", II, JJ, LOC2(store, 13, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[13,9] = %f \n", II, JJ, LOC2(store, 13, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[14,4] = %f \n", II, JJ, LOC2(store, 14, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[14,5] = %f \n", II, JJ, LOC2(store, 14, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[14,6] = %f \n", II, JJ, LOC2(store, 14, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[14,7] = %f \n", II, JJ, LOC2(store, 14, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[14,8] = %f \n", II, JJ, LOC2(store, 14, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[14,9] = %f \n", II, JJ, LOC2(store, 14, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[15,4] = %f \n", II, JJ, LOC2(store, 15, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[15,5] = %f \n", II, JJ, LOC2(store, 15, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[15,6] = %f \n", II, JJ, LOC2(store, 15, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[15,7] = %f \n", II, JJ, LOC2(store, 15, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[15,8] = %f \n", II, JJ, LOC2(store, 15, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[15,9] = %f \n", II, JJ, LOC2(store, 15, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[16,4] = %f \n", II, JJ, LOC2(store, 16, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[16,5] = %f \n", II, JJ, LOC2(store, 16, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[16,6] = %f \n", II, JJ, LOC2(store, 16, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[16,7] = %f \n", II, JJ, LOC2(store, 16, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[16,8] = %f \n", II, JJ, LOC2(store, 16, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[16,9] = %f \n", II, JJ, LOC2(store, 16, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[17,4] = %f \n", II, JJ, LOC2(store, 17, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[17,5] = %f \n", II, JJ, LOC2(store, 17, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[17,6] = %f \n", II, JJ, LOC2(store, 17, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[17,7] = %f \n", II, JJ, LOC2(store, 17, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[17,8] = %f \n", II, JJ, LOC2(store, 17, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[17,9] = %f \n", II, JJ, LOC2(store, 17, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[18,4] = %f \n", II, JJ, LOC2(store, 18, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[18,5] = %f \n", II, JJ, LOC2(store, 18, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[18,6] = %f \n", II, JJ, LOC2(store, 18, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[18,7] = %f \n", II, JJ, LOC2(store, 18, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[18,8] = %f \n", II, JJ, LOC2(store, 18, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[18,9] = %f \n", II, JJ, LOC2(store, 18, 9, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[19,4] = %f \n", II, JJ, LOC2(store, 19, 4, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[19,5] = %f \n", II, JJ, LOC2(store, 19, 5, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[19,6] = %f \n", II, JJ, LOC2(store, 19, 6, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[19,7] = %f \n", II, JJ, LOC2(store, 19, 7, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[19,8] = %f \n", II, JJ, LOC2(store, 19, 8, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FD store[19,9] = %f \n", II, JJ, LOC2(store, 19, 9, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* DF integral, m=0 */ 
  if(I == 2 && J == 3){ 
    DFint_0 df(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 10, STOREDIM, STOREDIM) += df.x_4_10;
    LOC2(store, 5, 10, STOREDIM, STOREDIM) += df.x_5_10;
    LOC2(store, 6, 10, STOREDIM, STOREDIM) += df.x_6_10;
    LOC2(store, 7, 10, STOREDIM, STOREDIM) += df.x_7_10;
    LOC2(store, 8, 10, STOREDIM, STOREDIM) += df.x_8_10;
    LOC2(store, 9, 10, STOREDIM, STOREDIM) += df.x_9_10;
    LOC2(store, 4, 11, STOREDIM, STOREDIM) += df.x_4_11;
    LOC2(store, 5, 11, STOREDIM, STOREDIM) += df.x_5_11;
    LOC2(store, 6, 11, STOREDIM, STOREDIM) += df.x_6_11;
    LOC2(store, 7, 11, STOREDIM, STOREDIM) += df.x_7_11;
    LOC2(store, 8, 11, STOREDIM, STOREDIM) += df.x_8_11;
    LOC2(store, 9, 11, STOREDIM, STOREDIM) += df.x_9_11;
    LOC2(store, 4, 12, STOREDIM, STOREDIM) += df.x_4_12;
    LOC2(store, 5, 12, STOREDIM, STOREDIM) += df.x_5_12;
    LOC2(store, 6, 12, STOREDIM, STOREDIM) += df.x_6_12;
    LOC2(store, 7, 12, STOREDIM, STOREDIM) += df.x_7_12;
    LOC2(store, 8, 12, STOREDIM, STOREDIM) += df.x_8_12;
    LOC2(store, 9, 12, STOREDIM, STOREDIM) += df.x_9_12;
    LOC2(store, 4, 13, STOREDIM, STOREDIM) += df.x_4_13;
    LOC2(store, 5, 13, STOREDIM, STOREDIM) += df.x_5_13;
    LOC2(store, 6, 13, STOREDIM, STOREDIM) += df.x_6_13;
    LOC2(store, 7, 13, STOREDIM, STOREDIM) += df.x_7_13;
    LOC2(store, 8, 13, STOREDIM, STOREDIM) += df.x_8_13;
    LOC2(store, 9, 13, STOREDIM, STOREDIM) += df.x_9_13;
    LOC2(store, 4, 14, STOREDIM, STOREDIM) += df.x_4_14;
    LOC2(store, 5, 14, STOREDIM, STOREDIM) += df.x_5_14;
    LOC2(store, 6, 14, STOREDIM, STOREDIM) += df.x_6_14;
    LOC2(store, 7, 14, STOREDIM, STOREDIM) += df.x_7_14;
    LOC2(store, 8, 14, STOREDIM, STOREDIM) += df.x_8_14;
    LOC2(store, 9, 14, STOREDIM, STOREDIM) += df.x_9_14;
    LOC2(store, 4, 15, STOREDIM, STOREDIM) += df.x_4_15;
    LOC2(store, 5, 15, STOREDIM, STOREDIM) += df.x_5_15;
    LOC2(store, 6, 15, STOREDIM, STOREDIM) += df.x_6_15;
    LOC2(store, 7, 15, STOREDIM, STOREDIM) += df.x_7_15;
    LOC2(store, 8, 15, STOREDIM, STOREDIM) += df.x_8_15;
    LOC2(store, 9, 15, STOREDIM, STOREDIM) += df.x_9_15;
    LOC2(store, 4, 16, STOREDIM, STOREDIM) += df.x_4_16;
    LOC2(store, 5, 16, STOREDIM, STOREDIM) += df.x_5_16;
    LOC2(store, 6, 16, STOREDIM, STOREDIM) += df.x_6_16;
    LOC2(store, 7, 16, STOREDIM, STOREDIM) += df.x_7_16;
    LOC2(store, 8, 16, STOREDIM, STOREDIM) += df.x_8_16;
    LOC2(store, 9, 16, STOREDIM, STOREDIM) += df.x_9_16;
    LOC2(store, 4, 17, STOREDIM, STOREDIM) += df.x_4_17;
    LOC2(store, 5, 17, STOREDIM, STOREDIM) += df.x_5_17;
    LOC2(store, 6, 17, STOREDIM, STOREDIM) += df.x_6_17;
    LOC2(store, 7, 17, STOREDIM, STOREDIM) += df.x_7_17;
    LOC2(store, 8, 17, STOREDIM, STOREDIM) += df.x_8_17;
    LOC2(store, 9, 17, STOREDIM, STOREDIM) += df.x_9_17;
    LOC2(store, 4, 18, STOREDIM, STOREDIM) += df.x_4_18;
    LOC2(store, 5, 18, STOREDIM, STOREDIM) += df.x_5_18;
    LOC2(store, 6, 18, STOREDIM, STOREDIM) += df.x_6_18;
    LOC2(store, 7, 18, STOREDIM, STOREDIM) += df.x_7_18;
    LOC2(store, 8, 18, STOREDIM, STOREDIM) += df.x_8_18;
    LOC2(store, 9, 18, STOREDIM, STOREDIM) += df.x_9_18;
    LOC2(store, 4, 19, STOREDIM, STOREDIM) += df.x_4_19;
    LOC2(store, 5, 19, STOREDIM, STOREDIM) += df.x_5_19;
    LOC2(store, 6, 19, STOREDIM, STOREDIM) += df.x_6_19;
    LOC2(store, 7, 19, STOREDIM, STOREDIM) += df.x_7_19;
    LOC2(store, 8, 19, STOREDIM, STOREDIM) += df.x_8_19;
    LOC2(store, 9, 19, STOREDIM, STOREDIM) += df.x_9_19;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d DF store[4,10] = %f \n", II, JJ, LOC2(store, 4, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,10] = %f \n", II, JJ, LOC2(store, 5, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,10] = %f \n", II, JJ, LOC2(store, 6, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,10] = %f \n", II, JJ, LOC2(store, 7, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,10] = %f \n", II, JJ, LOC2(store, 8, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,10] = %f \n", II, JJ, LOC2(store, 9, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,11] = %f \n", II, JJ, LOC2(store, 4, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,11] = %f \n", II, JJ, LOC2(store, 5, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,11] = %f \n", II, JJ, LOC2(store, 6, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,11] = %f \n", II, JJ, LOC2(store, 7, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,11] = %f \n", II, JJ, LOC2(store, 8, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,11] = %f \n", II, JJ, LOC2(store, 9, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,12] = %f \n", II, JJ, LOC2(store, 4, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,12] = %f \n", II, JJ, LOC2(store, 5, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,12] = %f \n", II, JJ, LOC2(store, 6, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,12] = %f \n", II, JJ, LOC2(store, 7, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,12] = %f \n", II, JJ, LOC2(store, 8, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,12] = %f \n", II, JJ, LOC2(store, 9, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,13] = %f \n", II, JJ, LOC2(store, 4, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,13] = %f \n", II, JJ, LOC2(store, 5, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,13] = %f \n", II, JJ, LOC2(store, 6, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,13] = %f \n", II, JJ, LOC2(store, 7, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,13] = %f \n", II, JJ, LOC2(store, 8, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,13] = %f \n", II, JJ, LOC2(store, 9, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,14] = %f \n", II, JJ, LOC2(store, 4, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,14] = %f \n", II, JJ, LOC2(store, 5, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,14] = %f \n", II, JJ, LOC2(store, 6, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,14] = %f \n", II, JJ, LOC2(store, 7, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,14] = %f \n", II, JJ, LOC2(store, 8, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,14] = %f \n", II, JJ, LOC2(store, 9, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,15] = %f \n", II, JJ, LOC2(store, 4, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,15] = %f \n", II, JJ, LOC2(store, 5, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,15] = %f \n", II, JJ, LOC2(store, 6, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,15] = %f \n", II, JJ, LOC2(store, 7, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,15] = %f \n", II, JJ, LOC2(store, 8, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,15] = %f \n", II, JJ, LOC2(store, 9, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,16] = %f \n", II, JJ, LOC2(store, 4, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,16] = %f \n", II, JJ, LOC2(store, 5, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,16] = %f \n", II, JJ, LOC2(store, 6, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,16] = %f \n", II, JJ, LOC2(store, 7, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,16] = %f \n", II, JJ, LOC2(store, 8, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,16] = %f \n", II, JJ, LOC2(store, 9, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,17] = %f \n", II, JJ, LOC2(store, 4, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,17] = %f \n", II, JJ, LOC2(store, 5, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,17] = %f \n", II, JJ, LOC2(store, 6, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,17] = %f \n", II, JJ, LOC2(store, 7, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,17] = %f \n", II, JJ, LOC2(store, 8, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,17] = %f \n", II, JJ, LOC2(store, 9, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,18] = %f \n", II, JJ, LOC2(store, 4, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,18] = %f \n", II, JJ, LOC2(store, 5, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,18] = %f \n", II, JJ, LOC2(store, 6, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,18] = %f \n", II, JJ, LOC2(store, 7, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,18] = %f \n", II, JJ, LOC2(store, 8, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,18] = %f \n", II, JJ, LOC2(store, 9, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[4,19] = %f \n", II, JJ, LOC2(store, 4, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[5,19] = %f \n", II, JJ, LOC2(store, 5, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[6,19] = %f \n", II, JJ, LOC2(store, 6, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[7,19] = %f \n", II, JJ, LOC2(store, 7, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[8,19] = %f \n", II, JJ, LOC2(store, 8, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d DF store[9,19] = %f \n", II, JJ, LOC2(store, 9, 19, STOREDIM, STOREDIM)); 
#endif 

  } 

  /* FF integral, m=0 */ 
  if(I == 3 && J == 3){ 
    FFint_0 ff(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 10, 10, STOREDIM, STOREDIM) += ff.x_10_10;
    LOC2(store, 10, 11, STOREDIM, STOREDIM) += ff.x_10_11;
    LOC2(store, 10, 12, STOREDIM, STOREDIM) += ff.x_10_12;
    LOC2(store, 10, 13, STOREDIM, STOREDIM) += ff.x_10_13;
    LOC2(store, 10, 14, STOREDIM, STOREDIM) += ff.x_10_14;
    LOC2(store, 10, 15, STOREDIM, STOREDIM) += ff.x_10_15;
    LOC2(store, 10, 16, STOREDIM, STOREDIM) += ff.x_10_16;
    LOC2(store, 10, 17, STOREDIM, STOREDIM) += ff.x_10_17;
    LOC2(store, 10, 18, STOREDIM, STOREDIM) += ff.x_10_18;
    LOC2(store, 10, 19, STOREDIM, STOREDIM) += ff.x_10_19;
    LOC2(store, 11, 10, STOREDIM, STOREDIM) += ff.x_11_10;
    LOC2(store, 11, 11, STOREDIM, STOREDIM) += ff.x_11_11;
    LOC2(store, 11, 12, STOREDIM, STOREDIM) += ff.x_11_12;
    LOC2(store, 11, 13, STOREDIM, STOREDIM) += ff.x_11_13;
    LOC2(store, 11, 14, STOREDIM, STOREDIM) += ff.x_11_14;
    LOC2(store, 11, 15, STOREDIM, STOREDIM) += ff.x_11_15;
    LOC2(store, 11, 16, STOREDIM, STOREDIM) += ff.x_11_16;
    LOC2(store, 11, 17, STOREDIM, STOREDIM) += ff.x_11_17;
    LOC2(store, 11, 18, STOREDIM, STOREDIM) += ff.x_11_18;
    LOC2(store, 11, 19, STOREDIM, STOREDIM) += ff.x_11_19;
    LOC2(store, 12, 10, STOREDIM, STOREDIM) += ff.x_12_10;
    LOC2(store, 12, 11, STOREDIM, STOREDIM) += ff.x_12_11;
    LOC2(store, 12, 12, STOREDIM, STOREDIM) += ff.x_12_12;
    LOC2(store, 12, 13, STOREDIM, STOREDIM) += ff.x_12_13;
    LOC2(store, 12, 14, STOREDIM, STOREDIM) += ff.x_12_14;
    LOC2(store, 12, 15, STOREDIM, STOREDIM) += ff.x_12_15;
    LOC2(store, 12, 16, STOREDIM, STOREDIM) += ff.x_12_16;
    LOC2(store, 12, 17, STOREDIM, STOREDIM) += ff.x_12_17;
    LOC2(store, 12, 18, STOREDIM, STOREDIM) += ff.x_12_18;
    LOC2(store, 12, 19, STOREDIM, STOREDIM) += ff.x_12_19;
    LOC2(store, 13, 10, STOREDIM, STOREDIM) += ff.x_13_10;
    LOC2(store, 13, 11, STOREDIM, STOREDIM) += ff.x_13_11;
    LOC2(store, 13, 12, STOREDIM, STOREDIM) += ff.x_13_12;
    LOC2(store, 13, 13, STOREDIM, STOREDIM) += ff.x_13_13;
    LOC2(store, 13, 14, STOREDIM, STOREDIM) += ff.x_13_14;
    LOC2(store, 13, 15, STOREDIM, STOREDIM) += ff.x_13_15;
    LOC2(store, 13, 16, STOREDIM, STOREDIM) += ff.x_13_16;
    LOC2(store, 13, 17, STOREDIM, STOREDIM) += ff.x_13_17;
    LOC2(store, 13, 18, STOREDIM, STOREDIM) += ff.x_13_18;
    LOC2(store, 13, 19, STOREDIM, STOREDIM) += ff.x_13_19;
    LOC2(store, 14, 10, STOREDIM, STOREDIM) += ff.x_14_10;
    LOC2(store, 14, 11, STOREDIM, STOREDIM) += ff.x_14_11;
    LOC2(store, 14, 12, STOREDIM, STOREDIM) += ff.x_14_12;
    LOC2(store, 14, 13, STOREDIM, STOREDIM) += ff.x_14_13;
    LOC2(store, 14, 14, STOREDIM, STOREDIM) += ff.x_14_14;
    LOC2(store, 14, 15, STOREDIM, STOREDIM) += ff.x_14_15;
    LOC2(store, 14, 16, STOREDIM, STOREDIM) += ff.x_14_16;
    LOC2(store, 14, 17, STOREDIM, STOREDIM) += ff.x_14_17;
    LOC2(store, 14, 18, STOREDIM, STOREDIM) += ff.x_14_18;
    LOC2(store, 14, 19, STOREDIM, STOREDIM) += ff.x_14_19;
    LOC2(store, 15, 10, STOREDIM, STOREDIM) += ff.x_15_10;
    LOC2(store, 15, 11, STOREDIM, STOREDIM) += ff.x_15_11;
    LOC2(store, 15, 12, STOREDIM, STOREDIM) += ff.x_15_12;
    LOC2(store, 15, 13, STOREDIM, STOREDIM) += ff.x_15_13;
    LOC2(store, 15, 14, STOREDIM, STOREDIM) += ff.x_15_14;
    LOC2(store, 15, 15, STOREDIM, STOREDIM) += ff.x_15_15;
    LOC2(store, 15, 16, STOREDIM, STOREDIM) += ff.x_15_16;
    LOC2(store, 15, 17, STOREDIM, STOREDIM) += ff.x_15_17;
    LOC2(store, 15, 18, STOREDIM, STOREDIM) += ff.x_15_18;
    LOC2(store, 15, 19, STOREDIM, STOREDIM) += ff.x_15_19;
    LOC2(store, 16, 10, STOREDIM, STOREDIM) += ff.x_16_10;
    LOC2(store, 16, 11, STOREDIM, STOREDIM) += ff.x_16_11;
    LOC2(store, 16, 12, STOREDIM, STOREDIM) += ff.x_16_12;
    LOC2(store, 16, 13, STOREDIM, STOREDIM) += ff.x_16_13;
    LOC2(store, 16, 14, STOREDIM, STOREDIM) += ff.x_16_14;
    LOC2(store, 16, 15, STOREDIM, STOREDIM) += ff.x_16_15;
    LOC2(store, 16, 16, STOREDIM, STOREDIM) += ff.x_16_16;
    LOC2(store, 16, 17, STOREDIM, STOREDIM) += ff.x_16_17;
    LOC2(store, 16, 18, STOREDIM, STOREDIM) += ff.x_16_18;
    LOC2(store, 16, 19, STOREDIM, STOREDIM) += ff.x_16_19;
    LOC2(store, 17, 10, STOREDIM, STOREDIM) += ff.x_17_10;
    LOC2(store, 17, 11, STOREDIM, STOREDIM) += ff.x_17_11;
    LOC2(store, 17, 12, STOREDIM, STOREDIM) += ff.x_17_12;
    LOC2(store, 17, 13, STOREDIM, STOREDIM) += ff.x_17_13;
    LOC2(store, 17, 14, STOREDIM, STOREDIM) += ff.x_17_14;
    LOC2(store, 17, 15, STOREDIM, STOREDIM) += ff.x_17_15;
    LOC2(store, 17, 16, STOREDIM, STOREDIM) += ff.x_17_16;
    LOC2(store, 17, 17, STOREDIM, STOREDIM) += ff.x_17_17;
    LOC2(store, 17, 18, STOREDIM, STOREDIM) += ff.x_17_18;
    LOC2(store, 17, 19, STOREDIM, STOREDIM) += ff.x_17_19;
    LOC2(store, 18, 10, STOREDIM, STOREDIM) += ff.x_18_10;
    LOC2(store, 18, 11, STOREDIM, STOREDIM) += ff.x_18_11;
    LOC2(store, 18, 12, STOREDIM, STOREDIM) += ff.x_18_12;
    LOC2(store, 18, 13, STOREDIM, STOREDIM) += ff.x_18_13;
    LOC2(store, 18, 14, STOREDIM, STOREDIM) += ff.x_18_14;
    LOC2(store, 18, 15, STOREDIM, STOREDIM) += ff.x_18_15;
    LOC2(store, 18, 16, STOREDIM, STOREDIM) += ff.x_18_16;
    LOC2(store, 18, 17, STOREDIM, STOREDIM) += ff.x_18_17;
    LOC2(store, 18, 18, STOREDIM, STOREDIM) += ff.x_18_18;
    LOC2(store, 18, 19, STOREDIM, STOREDIM) += ff.x_18_19;
    LOC2(store, 19, 10, STOREDIM, STOREDIM) += ff.x_19_10;
    LOC2(store, 19, 11, STOREDIM, STOREDIM) += ff.x_19_11;
    LOC2(store, 19, 12, STOREDIM, STOREDIM) += ff.x_19_12;
    LOC2(store, 19, 13, STOREDIM, STOREDIM) += ff.x_19_13;
    LOC2(store, 19, 14, STOREDIM, STOREDIM) += ff.x_19_14;
    LOC2(store, 19, 15, STOREDIM, STOREDIM) += ff.x_19_15;
    LOC2(store, 19, 16, STOREDIM, STOREDIM) += ff.x_19_16;
    LOC2(store, 19, 17, STOREDIM, STOREDIM) += ff.x_19_17;
    LOC2(store, 19, 18, STOREDIM, STOREDIM) += ff.x_19_18;
    LOC2(store, 19, 19, STOREDIM, STOREDIM) += ff.x_19_19;

#ifdef DEBUG_OEI 
    printf("II %d JJ %d FF store[10,10] = %f \n", II, JJ, LOC2(store, 10, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,11] = %f \n", II, JJ, LOC2(store, 10, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,12] = %f \n", II, JJ, LOC2(store, 10, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,13] = %f \n", II, JJ, LOC2(store, 10, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,14] = %f \n", II, JJ, LOC2(store, 10, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,15] = %f \n", II, JJ, LOC2(store, 10, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,16] = %f \n", II, JJ, LOC2(store, 10, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,17] = %f \n", II, JJ, LOC2(store, 10, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,18] = %f \n", II, JJ, LOC2(store, 10, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[10,19] = %f \n", II, JJ, LOC2(store, 10, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,10] = %f \n", II, JJ, LOC2(store, 11, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,11] = %f \n", II, JJ, LOC2(store, 11, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,12] = %f \n", II, JJ, LOC2(store, 11, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,13] = %f \n", II, JJ, LOC2(store, 11, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,14] = %f \n", II, JJ, LOC2(store, 11, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,15] = %f \n", II, JJ, LOC2(store, 11, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,16] = %f \n", II, JJ, LOC2(store, 11, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,17] = %f \n", II, JJ, LOC2(store, 11, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,18] = %f \n", II, JJ, LOC2(store, 11, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[11,19] = %f \n", II, JJ, LOC2(store, 11, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,10] = %f \n", II, JJ, LOC2(store, 12, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,11] = %f \n", II, JJ, LOC2(store, 12, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,12] = %f \n", II, JJ, LOC2(store, 12, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,13] = %f \n", II, JJ, LOC2(store, 12, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,14] = %f \n", II, JJ, LOC2(store, 12, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,15] = %f \n", II, JJ, LOC2(store, 12, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,16] = %f \n", II, JJ, LOC2(store, 12, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,17] = %f \n", II, JJ, LOC2(store, 12, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,18] = %f \n", II, JJ, LOC2(store, 12, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[12,19] = %f \n", II, JJ, LOC2(store, 12, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,10] = %f \n", II, JJ, LOC2(store, 13, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,11] = %f \n", II, JJ, LOC2(store, 13, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,12] = %f \n", II, JJ, LOC2(store, 13, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,13] = %f \n", II, JJ, LOC2(store, 13, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,14] = %f \n", II, JJ, LOC2(store, 13, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,15] = %f \n", II, JJ, LOC2(store, 13, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,16] = %f \n", II, JJ, LOC2(store, 13, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,17] = %f \n", II, JJ, LOC2(store, 13, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,18] = %f \n", II, JJ, LOC2(store, 13, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[13,19] = %f \n", II, JJ, LOC2(store, 13, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,10] = %f \n", II, JJ, LOC2(store, 14, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,11] = %f \n", II, JJ, LOC2(store, 14, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,12] = %f \n", II, JJ, LOC2(store, 14, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,13] = %f \n", II, JJ, LOC2(store, 14, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,14] = %f \n", II, JJ, LOC2(store, 14, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,15] = %f \n", II, JJ, LOC2(store, 14, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,16] = %f \n", II, JJ, LOC2(store, 14, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,17] = %f \n", II, JJ, LOC2(store, 14, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,18] = %f \n", II, JJ, LOC2(store, 14, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[14,19] = %f \n", II, JJ, LOC2(store, 14, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,10] = %f \n", II, JJ, LOC2(store, 15, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,11] = %f \n", II, JJ, LOC2(store, 15, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,12] = %f \n", II, JJ, LOC2(store, 15, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,13] = %f \n", II, JJ, LOC2(store, 15, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,14] = %f \n", II, JJ, LOC2(store, 15, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,15] = %f \n", II, JJ, LOC2(store, 15, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,16] = %f \n", II, JJ, LOC2(store, 15, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,17] = %f \n", II, JJ, LOC2(store, 15, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,18] = %f \n", II, JJ, LOC2(store, 15, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[15,19] = %f \n", II, JJ, LOC2(store, 15, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,10] = %f \n", II, JJ, LOC2(store, 16, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,11] = %f \n", II, JJ, LOC2(store, 16, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,12] = %f \n", II, JJ, LOC2(store, 16, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,13] = %f \n", II, JJ, LOC2(store, 16, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,14] = %f \n", II, JJ, LOC2(store, 16, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,15] = %f \n", II, JJ, LOC2(store, 16, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,16] = %f \n", II, JJ, LOC2(store, 16, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,17] = %f \n", II, JJ, LOC2(store, 16, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,18] = %f \n", II, JJ, LOC2(store, 16, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[16,19] = %f \n", II, JJ, LOC2(store, 16, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,10] = %f \n", II, JJ, LOC2(store, 17, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,11] = %f \n", II, JJ, LOC2(store, 17, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,12] = %f \n", II, JJ, LOC2(store, 17, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,13] = %f \n", II, JJ, LOC2(store, 17, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,14] = %f \n", II, JJ, LOC2(store, 17, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,15] = %f \n", II, JJ, LOC2(store, 17, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,16] = %f \n", II, JJ, LOC2(store, 17, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,17] = %f \n", II, JJ, LOC2(store, 17, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,18] = %f \n", II, JJ, LOC2(store, 17, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[17,19] = %f \n", II, JJ, LOC2(store, 17, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,10] = %f \n", II, JJ, LOC2(store, 18, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,11] = %f \n", II, JJ, LOC2(store, 18, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,12] = %f \n", II, JJ, LOC2(store, 18, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,13] = %f \n", II, JJ, LOC2(store, 18, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,14] = %f \n", II, JJ, LOC2(store, 18, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,15] = %f \n", II, JJ, LOC2(store, 18, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,16] = %f \n", II, JJ, LOC2(store, 18, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,17] = %f \n", II, JJ, LOC2(store, 18, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,18] = %f \n", II, JJ, LOC2(store, 18, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[18,19] = %f \n", II, JJ, LOC2(store, 18, 19, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,10] = %f \n", II, JJ, LOC2(store, 19, 10, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,11] = %f \n", II, JJ, LOC2(store, 19, 11, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,12] = %f \n", II, JJ, LOC2(store, 19, 12, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,13] = %f \n", II, JJ, LOC2(store, 19, 13, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,14] = %f \n", II, JJ, LOC2(store, 19, 14, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,15] = %f \n", II, JJ, LOC2(store, 19, 15, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,16] = %f \n", II, JJ, LOC2(store, 19, 16, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,17] = %f \n", II, JJ, LOC2(store, 19, 17, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,18] = %f \n", II, JJ, LOC2(store, 19, 18, STOREDIM, STOREDIM)); 
    printf("II %d JJ %d FF store[19,19] = %f \n", II, JJ, LOC2(store, 19, 19, STOREDIM, STOREDIM)); 
#endif 

  } 

 } 
