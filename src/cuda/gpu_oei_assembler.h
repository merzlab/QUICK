/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 10/07/2021                !
 !                                                                     !
 ! Copyright (C) 2020-2021 Merz lab                                    !
 ! Copyright (C) 2020-2021 Götz lab                                    !
 !                                                                     !
 ! This Source Code Form is subject to the terms of the Mozilla Public !
 ! License, v. 2.0. If a copy of the MPL was not distributed with this !
 ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
 !_____________________________________________________________________!
*/

__device__ __inline__ void OEint_vertical(int I, int J, QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
        QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta,
        QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  /* SS integral, m=0 */ 
  if(I == 0 && J == 0){ 
    LOC2(store, 0, 0, STOREDIM, STOREDIM) = VY(0, 0, 0);

    printf("II %d JJ %d store SS: %f \n", I, J, LOC2(store, 0, 0, STOREDIM, STOREDIM));

  } 

  /* PS integral, m=0 */ 
  if(I == 1 && J == 0){ 
    PSint_0 ps(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); 
    LOC2(store, 1, 0, STOREDIM, STOREDIM) = ps.x_1_0;
    LOC2(store, 2, 0, STOREDIM, STOREDIM) = ps.x_2_0;
    LOC2(store, 3, 0, STOREDIM, STOREDIM) = ps.x_3_0;

    printf("II %d JJ %d store PS: x_1_0 %f \n", I, J, LOC2(store, 1, 0, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PS: x_2_0 %f \n", I, J, LOC2(store, 2, 0, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PS: x_3_0 %f \n", I, J, LOC2(store, 3, 0, STOREDIM, STOREDIM));

  } 

  /* SP integral, m=0 */ 
  if(I == 0 && J == 1){ 
    SPint_0 sp(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); 
    LOC2(store, 0, 1, STOREDIM, STOREDIM) = sp.x_0_1;
    LOC2(store, 0, 2, STOREDIM, STOREDIM) = sp.x_0_2;
    LOC2(store, 0, 3, STOREDIM, STOREDIM) = sp.x_0_3;

    printf("II %d JJ %d store SP: x_0_1 %f \n", I, J, LOC2(store, 0, 1, STOREDIM, STOREDIM));
    printf("II %d JJ %d store SP: x_0_2 %f \n", I, J, LOC2(store, 0, 2, STOREDIM, STOREDIM));
    printf("II %d JJ %d store SP: x_0_3 %f \n", I, J, LOC2(store, 0, 3, STOREDIM, STOREDIM));

  } 

  /* PP integral, m=0 */ 
  if(I == 1 && J == 1){ 
    PPint_0 pp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 1, 1, STOREDIM, STOREDIM) = pp.x_1_1;
    LOC2(store, 1, 2, STOREDIM, STOREDIM) = pp.x_1_2;
    LOC2(store, 1, 3, STOREDIM, STOREDIM) = pp.x_1_3;
    LOC2(store, 2, 1, STOREDIM, STOREDIM) = pp.x_2_1;
    LOC2(store, 2, 2, STOREDIM, STOREDIM) = pp.x_2_2;
    LOC2(store, 2, 3, STOREDIM, STOREDIM) = pp.x_2_3;
    LOC2(store, 3, 1, STOREDIM, STOREDIM) = pp.x_3_1;
    LOC2(store, 3, 2, STOREDIM, STOREDIM) = pp.x_3_2;
    LOC2(store, 3, 3, STOREDIM, STOREDIM) = pp.x_3_3;

    printf("II %d JJ %d store PP: x_1_1 %f \n", I, J, LOC2(store, 1, 1, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_1_2 %f \n", I, J, LOC2(store, 1, 2, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_1_3 %f \n", I, J, LOC2(store, 1, 3, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_2_1 %f \n", I, J, LOC2(store, 2, 1, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_2_2 %f \n", I, J, LOC2(store, 2, 2, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_2_3 %f \n", I, J, LOC2(store, 2, 3, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_3_1 %f \n", I, J, LOC2(store, 3, 1, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_3_2 %f \n", I, J, LOC2(store, 3, 2, STOREDIM, STOREDIM));
    printf("II %d JJ %d store PP: x_3_3 %f \n", I, J, LOC2(store, 3, 3, STOREDIM, STOREDIM));

  } 

  /* DS integral, m=0 */ 
  if(I == 2 && J == 0){ 
    DSint_0 ds(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 0, STOREDIM, STOREDIM) = ds.x_4_0;
    LOC2(store, 5, 0, STOREDIM, STOREDIM) = ds.x_5_0;
    LOC2(store, 6, 0, STOREDIM, STOREDIM) = ds.x_6_0;
    LOC2(store, 7, 0, STOREDIM, STOREDIM) = ds.x_7_0;
    LOC2(store, 8, 0, STOREDIM, STOREDIM) = ds.x_8_0;
    LOC2(store, 9, 0, STOREDIM, STOREDIM) = ds.x_9_0;
  } 

  /* SD integral, m=0 */ 
  if(I == 0 && J == 2){ 
    SDint_0 sd(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 0, 4, STOREDIM, STOREDIM) = sd.x_0_4;
    LOC2(store, 0, 5, STOREDIM, STOREDIM) = sd.x_0_5;
    LOC2(store, 0, 6, STOREDIM, STOREDIM) = sd.x_0_6;
    LOC2(store, 0, 7, STOREDIM, STOREDIM) = sd.x_0_7;
    LOC2(store, 0, 8, STOREDIM, STOREDIM) = sd.x_0_8;
    LOC2(store, 0, 9, STOREDIM, STOREDIM) = sd.x_0_9;
  } 

  /* DP integral, m=0 */ 
  if(I == 2 && J == 1){ 
    DPint_0 dp(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 1, STOREDIM, STOREDIM) = dp.x_4_1;
    LOC2(store, 4, 2, STOREDIM, STOREDIM) = dp.x_4_2;
    LOC2(store, 4, 3, STOREDIM, STOREDIM) = dp.x_4_3;
    LOC2(store, 5, 1, STOREDIM, STOREDIM) = dp.x_5_1;
    LOC2(store, 5, 2, STOREDIM, STOREDIM) = dp.x_5_2;
    LOC2(store, 5, 3, STOREDIM, STOREDIM) = dp.x_5_3;
    LOC2(store, 6, 1, STOREDIM, STOREDIM) = dp.x_6_1;
    LOC2(store, 6, 2, STOREDIM, STOREDIM) = dp.x_6_2;
    LOC2(store, 6, 3, STOREDIM, STOREDIM) = dp.x_6_3;
    LOC2(store, 7, 1, STOREDIM, STOREDIM) = dp.x_7_1;
    LOC2(store, 7, 2, STOREDIM, STOREDIM) = dp.x_7_2;
    LOC2(store, 7, 3, STOREDIM, STOREDIM) = dp.x_7_3;
    LOC2(store, 8, 1, STOREDIM, STOREDIM) = dp.x_8_1;
    LOC2(store, 8, 2, STOREDIM, STOREDIM) = dp.x_8_2;
    LOC2(store, 8, 3, STOREDIM, STOREDIM) = dp.x_8_3;
    LOC2(store, 9, 1, STOREDIM, STOREDIM) = dp.x_9_1;
    LOC2(store, 9, 2, STOREDIM, STOREDIM) = dp.x_9_2;
    LOC2(store, 9, 3, STOREDIM, STOREDIM) = dp.x_9_3;
  } 

  /* PD integral, m=0 */ 
  if(I == 1 && J == 2){ 
    PDint_0 pd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 1, 4, STOREDIM, STOREDIM) = pd.x_1_4;
    LOC2(store, 2, 4, STOREDIM, STOREDIM) = pd.x_2_4;
    LOC2(store, 3, 4, STOREDIM, STOREDIM) = pd.x_3_4;
    LOC2(store, 1, 5, STOREDIM, STOREDIM) = pd.x_1_5;
    LOC2(store, 2, 5, STOREDIM, STOREDIM) = pd.x_2_5;
    LOC2(store, 3, 5, STOREDIM, STOREDIM) = pd.x_3_5;
    LOC2(store, 1, 6, STOREDIM, STOREDIM) = pd.x_1_6;
    LOC2(store, 2, 6, STOREDIM, STOREDIM) = pd.x_2_6;
    LOC2(store, 3, 6, STOREDIM, STOREDIM) = pd.x_3_6;
    LOC2(store, 1, 7, STOREDIM, STOREDIM) = pd.x_1_7;
    LOC2(store, 2, 7, STOREDIM, STOREDIM) = pd.x_2_7;
    LOC2(store, 3, 7, STOREDIM, STOREDIM) = pd.x_3_7;
    LOC2(store, 1, 8, STOREDIM, STOREDIM) = pd.x_1_8;
    LOC2(store, 2, 8, STOREDIM, STOREDIM) = pd.x_2_8;
    LOC2(store, 3, 8, STOREDIM, STOREDIM) = pd.x_3_8;
    LOC2(store, 1, 9, STOREDIM, STOREDIM) = pd.x_1_9;
    LOC2(store, 2, 9, STOREDIM, STOREDIM) = pd.x_2_9;
    LOC2(store, 3, 9, STOREDIM, STOREDIM) = pd.x_3_9;
  } 

  /* DD integral, m=0 */ 
  if(I == 2 && J == 2){ 
    DDint_0 dd(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); 
    LOC2(store, 4, 4, STOREDIM, STOREDIM) = dd.x_4_4;
    LOC2(store, 5, 4, STOREDIM, STOREDIM) = dd.x_5_4;
    LOC2(store, 6, 4, STOREDIM, STOREDIM) = dd.x_6_4;
    LOC2(store, 7, 4, STOREDIM, STOREDIM) = dd.x_7_4;
    LOC2(store, 8, 4, STOREDIM, STOREDIM) = dd.x_8_4;
    LOC2(store, 9, 4, STOREDIM, STOREDIM) = dd.x_9_4;
    LOC2(store, 4, 5, STOREDIM, STOREDIM) = dd.x_4_5;
    LOC2(store, 5, 5, STOREDIM, STOREDIM) = dd.x_5_5;
    LOC2(store, 6, 5, STOREDIM, STOREDIM) = dd.x_6_5;
    LOC2(store, 7, 5, STOREDIM, STOREDIM) = dd.x_7_5;
    LOC2(store, 8, 5, STOREDIM, STOREDIM) = dd.x_8_5;
    LOC2(store, 9, 5, STOREDIM, STOREDIM) = dd.x_9_5;
    LOC2(store, 4, 6, STOREDIM, STOREDIM) = dd.x_4_6;
    LOC2(store, 5, 6, STOREDIM, STOREDIM) = dd.x_5_6;
    LOC2(store, 6, 6, STOREDIM, STOREDIM) = dd.x_6_6;
    LOC2(store, 7, 6, STOREDIM, STOREDIM) = dd.x_7_6;
    LOC2(store, 8, 6, STOREDIM, STOREDIM) = dd.x_8_6;
    LOC2(store, 9, 6, STOREDIM, STOREDIM) = dd.x_9_6;
    LOC2(store, 4, 7, STOREDIM, STOREDIM) = dd.x_4_7;
    LOC2(store, 5, 7, STOREDIM, STOREDIM) = dd.x_5_7;
    LOC2(store, 6, 7, STOREDIM, STOREDIM) = dd.x_6_7;
    LOC2(store, 7, 7, STOREDIM, STOREDIM) = dd.x_7_7;
    LOC2(store, 8, 7, STOREDIM, STOREDIM) = dd.x_8_7;
    LOC2(store, 9, 7, STOREDIM, STOREDIM) = dd.x_9_7;
    LOC2(store, 4, 8, STOREDIM, STOREDIM) = dd.x_4_8;
    LOC2(store, 5, 8, STOREDIM, STOREDIM) = dd.x_5_8;
    LOC2(store, 6, 8, STOREDIM, STOREDIM) = dd.x_6_8;
    LOC2(store, 7, 8, STOREDIM, STOREDIM) = dd.x_7_8;
    LOC2(store, 8, 8, STOREDIM, STOREDIM) = dd.x_8_8;
    LOC2(store, 9, 8, STOREDIM, STOREDIM) = dd.x_9_8;
    LOC2(store, 4, 9, STOREDIM, STOREDIM) = dd.x_4_9;
    LOC2(store, 5, 9, STOREDIM, STOREDIM) = dd.x_5_9;
    LOC2(store, 6, 9, STOREDIM, STOREDIM) = dd.x_6_9;
    LOC2(store, 7, 9, STOREDIM, STOREDIM) = dd.x_7_9;
    LOC2(store, 8, 9, STOREDIM, STOREDIM) = dd.x_8_9;
    LOC2(store, 9, 9, STOREDIM, STOREDIM) = dd.x_9_9;
  } 

 } 
