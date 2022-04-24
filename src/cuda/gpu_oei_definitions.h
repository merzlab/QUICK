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


/* PS true integral, m=0 */ 
__device__ __inline__ PSint_0::PSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 0) - PCx * VY(0, 0, 1);
  x_2_0 = PAy * VY(0, 0, 0) - PCy * VY(0, 0, 1);
  x_3_0 = PAz * VY(0, 0, 0) - PCz * VY(0, 0, 1);
} 


/* PS auxilary integral, m=1 */ 
__device__ __inline__ PSint_1::PSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 1) - PCx * VY(0, 0, 2);
  x_2_0 = PAy * VY(0, 0, 1) - PCy * VY(0, 0, 2);
  x_3_0 = PAz * VY(0, 0, 1) - PCz * VY(0, 0, 2);
} 


/* PS auxilary integral, m=2 */ 
__device__ __inline__ PSint_2::PSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 2) - PCx * VY(0, 0, 3);
  x_2_0 = PAy * VY(0, 0, 2) - PCy * VY(0, 0, 3);
  x_3_0 = PAz * VY(0, 0, 2) - PCz * VY(0, 0, 3);
} 


/* PS auxilary integral, m=3 */ 
__device__ __inline__ PSint_3::PSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 3) - PCx * VY(0, 0, 4);
  x_2_0 = PAy * VY(0, 0, 3) - PCy * VY(0, 0, 4);
  x_3_0 = PAz * VY(0, 0, 3) - PCz * VY(0, 0, 4);
} 


/* PS auxilary integral, m=4 */ 
__device__ __inline__ PSint_4::PSint_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 4) - PCx * VY(0, 0, 5);
  x_2_0 = PAy * VY(0, 0, 4) - PCy * VY(0, 0, 5);
  x_3_0 = PAz * VY(0, 0, 4) - PCz * VY(0, 0, 5);
} 


/* PS auxilary integral, m=5 */ 
__device__ __inline__ PSint_5::PSint_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 5) - PCx * VY(0, 0, 6);
  x_2_0 = PAy * VY(0, 0, 5) - PCy * VY(0, 0, 6);
  x_3_0 = PAz * VY(0, 0, 5) - PCz * VY(0, 0, 6);
} 


/* SP true integral, m=0 */ 
__device__ __inline__ SPint_0::SPint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 0) - PCx * VY(0, 0, 1);
  x_0_2 = PBy * VY(0, 0, 0) - PCy * VY(0, 0, 1);
  x_0_3 = PBz * VY(0, 0, 0) - PCz * VY(0, 0, 1);
} 


/* SP auxilary integral, m=1 */ 
__device__ __inline__ SPint_1::SPint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 1) - PCx * VY(0, 0, 2);
  x_0_2 = PBy * VY(0, 0, 1) - PCy * VY(0, 0, 2);
  x_0_3 = PBz * VY(0, 0, 1) - PCz * VY(0, 0, 2);
} 


/* SP auxilary integral, m=2 */ 
__device__ __inline__ SPint_2::SPint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 2) - PCx * VY(0, 0, 3);
  x_0_2 = PBy * VY(0, 0, 2) - PCy * VY(0, 0, 3);
  x_0_3 = PBz * VY(0, 0, 2) - PCz * VY(0, 0, 3);
} 


/* SP auxilary integral, m=3 */ 
__device__ __inline__ SPint_3::SPint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 3) - PCx * VY(0, 0, 4);
  x_0_2 = PBy * VY(0, 0, 3) - PCy * VY(0, 0, 4);
  x_0_3 = PBz * VY(0, 0, 3) - PCz * VY(0, 0, 4);
} 


/* SP auxilary integral, m=4 */ 
__device__ __inline__ SPint_4::SPint_4(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 4) - PCx * VY(0, 0, 5);
  x_0_2 = PBy * VY(0, 0, 4) - PCy * VY(0, 0, 5);
  x_0_3 = PBz * VY(0, 0, 4) - PCz * VY(0, 0, 5);
} 


/* SP auxilary integral, m=5 */ 
__device__ __inline__ SPint_5::SPint_5(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 5) - PCx * VY(0, 0, 6);
  x_0_2 = PBy * VY(0, 0, 5) - PCy * VY(0, 0, 6);
  x_0_3 = PBz * VY(0, 0, 5) - PCz * VY(0, 0, 6);
} 


/* PP true integral, m=0 */ 
__device__ __inline__ PPint_0::PPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 

  x_1_1 = PBx * ps_0.x_1_0 - PCx * ps_1.x_1_0; 
  x_1_1 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_1_2 = PBy * ps_0.x_1_0 - PCy * ps_1.x_1_0; 
  x_1_3 = PBz * ps_0.x_1_0 - PCz * ps_1.x_1_0; 
  x_2_1 = PBx * ps_0.x_2_0 - PCx * ps_1.x_2_0; 
  x_2_2 = PBy * ps_0.x_2_0 - PCy * ps_1.x_2_0; 
  x_2_2 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_2_3 = PBz * ps_0.x_2_0 - PCz * ps_1.x_2_0; 
  x_3_1 = PBx * ps_0.x_3_0 - PCx * ps_1.x_3_0; 
  x_3_2 = PBy * ps_0.x_3_0 - PCy * ps_1.x_3_0; 
  x_3_3 = PBz * ps_0.x_3_0 - PCz * ps_1.x_3_0; 
  x_3_3 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 

 } 

/* PP auxilary integral, m=1 */ 
__device__ __inline__ PPint_1::PPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 

  x_1_1 = PBx * ps_1.x_1_0 - PCx * ps_2.x_1_0; 
  x_1_1 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_1_2 = PBy * ps_1.x_1_0 - PCy * ps_2.x_1_0; 
  x_1_3 = PBz * ps_1.x_1_0 - PCz * ps_2.x_1_0; 
  x_2_1 = PBx * ps_1.x_2_0 - PCx * ps_2.x_2_0; 
  x_2_2 = PBy * ps_1.x_2_0 - PCy * ps_2.x_2_0; 
  x_2_2 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_2_3 = PBz * ps_1.x_2_0 - PCz * ps_2.x_2_0; 
  x_3_1 = PBx * ps_1.x_3_0 - PCx * ps_2.x_3_0; 
  x_3_2 = PBy * ps_1.x_3_0 - PCy * ps_2.x_3_0; 
  x_3_3 = PBz * ps_1.x_3_0 - PCz * ps_2.x_3_0; 
  x_3_3 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 

 } 

/* PP auxilary integral, m=2 */ 
__device__ __inline__ PPint_2::PPint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 

  x_1_1 = PBx * ps_2.x_1_0 - PCx * ps_3.x_1_0; 
  x_1_1 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_1_2 = PBy * ps_2.x_1_0 - PCy * ps_3.x_1_0; 
  x_1_3 = PBz * ps_2.x_1_0 - PCz * ps_3.x_1_0; 
  x_2_1 = PBx * ps_2.x_2_0 - PCx * ps_3.x_2_0; 
  x_2_2 = PBy * ps_2.x_2_0 - PCy * ps_3.x_2_0; 
  x_2_2 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_2_3 = PBz * ps_2.x_2_0 - PCz * ps_3.x_2_0; 
  x_3_1 = PBx * ps_2.x_3_0 - PCx * ps_3.x_3_0; 
  x_3_2 = PBy * ps_2.x_3_0 - PCy * ps_3.x_3_0; 
  x_3_3 = PBz * ps_2.x_3_0 - PCz * ps_3.x_3_0; 
  x_3_3 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 

 } 

/* DS true integral, m=0 */ 
__device__ __inline__ DSint_0::DSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 

  x_4_0 = PAx * ps_0.x_2_0 - PCx * ps_1.x_2_0; 
  x_5_0 = PAy * ps_0.x_3_0 - PCy * ps_1.x_3_0; 
  x_6_0 = PAx * ps_0.x_3_0 - PCx * ps_1.x_3_0; 
  x_7_0 = PAx * ps_0.x_1_0 - PCx * ps_1.x_1_0; 
  x_7_0 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_8_0 = PAy * ps_0.x_2_0 - PCy * ps_1.x_2_0; 
  x_8_0 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_9_0 = PAz * ps_0.x_3_0 - PCz * ps_1.x_3_0; 
  x_9_0 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 

 } 

/* DS auxilary integral, m=1 */ 
__device__ __inline__ DSint_1::DSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 

  x_4_0 = PAx * ps_1.x_2_0 - PCx * ps_2.x_2_0; 
  x_5_0 = PAy * ps_1.x_3_0 - PCy * ps_2.x_3_0; 
  x_6_0 = PAx * ps_1.x_3_0 - PCx * ps_2.x_3_0; 
  x_7_0 = PAx * ps_1.x_1_0 - PCx * ps_2.x_1_0; 
  x_7_0 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_8_0 = PAy * ps_1.x_2_0 - PCy * ps_2.x_2_0; 
  x_8_0 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_9_0 = PAz * ps_1.x_3_0 - PCz * ps_2.x_3_0; 
  x_9_0 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 

 } 

/* DS auxilary integral, m=2 */ 
__device__ __inline__ DSint_2::DSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 

  x_4_0 = PAx * ps_2.x_2_0 - PCx * ps_3.x_2_0; 
  x_5_0 = PAy * ps_2.x_3_0 - PCy * ps_3.x_3_0; 
  x_6_0 = PAx * ps_2.x_3_0 - PCx * ps_3.x_3_0; 
  x_7_0 = PAx * ps_2.x_1_0 - PCx * ps_3.x_1_0; 
  x_7_0 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_8_0 = PAy * ps_2.x_2_0 - PCy * ps_3.x_2_0; 
  x_8_0 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_9_0 = PAz * ps_2.x_3_0 - PCz * ps_3.x_3_0; 
  x_9_0 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 

 } 

/* DS auxilary integral, m=3 */ 
__device__ __inline__ DSint_3::DSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  PSint_4 ps_4(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=4 

  x_4_0 = PAx * ps_3.x_2_0 - PCx * ps_4.x_2_0; 
  x_5_0 = PAy * ps_3.x_3_0 - PCy * ps_4.x_3_0; 
  x_6_0 = PAx * ps_3.x_3_0 - PCx * ps_4.x_3_0; 
  x_7_0 = PAx * ps_3.x_1_0 - PCx * ps_4.x_1_0; 
  x_7_0 += TwoZetaInv * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_8_0 = PAy * ps_3.x_2_0 - PCy * ps_4.x_2_0; 
  x_8_0 += TwoZetaInv * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_9_0 = PAz * ps_3.x_3_0 - PCz * ps_4.x_3_0; 
  x_9_0 += TwoZetaInv * (VY(0, 0, 3) - VY(0, 0, 4)); 

 } 

/* DS auxilary integral, m=4 */ 
__device__ __inline__ DSint_4::DSint_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_4 ps_4(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=4 
  PSint_5 ps_5(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=5 

  x_4_0 = PAx * ps_4.x_2_0 - PCx * ps_5.x_2_0; 
  x_5_0 = PAy * ps_4.x_3_0 - PCy * ps_5.x_3_0; 
  x_6_0 = PAx * ps_4.x_3_0 - PCx * ps_5.x_3_0; 
  x_7_0 = PAx * ps_4.x_1_0 - PCx * ps_5.x_1_0; 
  x_7_0 += TwoZetaInv * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_8_0 = PAy * ps_4.x_2_0 - PCy * ps_5.x_2_0; 
  x_8_0 += TwoZetaInv * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_9_0 = PAz * ps_4.x_3_0 - PCz * ps_5.x_3_0; 
  x_9_0 += TwoZetaInv * (VY(0, 0, 4) - VY(0, 0, 5)); 

 } 

/* SD true integral, m=0 */ 
__device__ __inline__ SDint_0::SDint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=1 

  x_0_4 = PBx * sp_0.x_0_2 - PCx * sp_1.x_0_2; 
  x_0_5 = PBy * sp_0.x_0_3 - PCy * sp_1.x_0_3; 
  x_0_6 = PBx * sp_0.x_0_3 - PCx * sp_1.x_0_3; 
  x_0_7 = PBx * sp_0.x_0_1 - PCx * sp_1.x_0_1; 
  x_0_7 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_0_8 = PBy * sp_0.x_0_2 - PCy * sp_1.x_0_2; 
  x_0_8 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_0_9 = PBz * sp_0.x_0_3 - PCz * sp_1.x_0_3; 
  x_0_9 += TwoZetaInv * (VY(0, 0, 0) - VY(0, 0, 1)); 

 } 

/* SD auxilary integral, m=1 */ 
__device__ __inline__ SDint_1::SDint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=1 
  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=2 

  x_0_4 = PBx * sp_1.x_0_2 - PCx * sp_2.x_0_2; 
  x_0_5 = PBy * sp_1.x_0_3 - PCy * sp_2.x_0_3; 
  x_0_6 = PBx * sp_1.x_0_3 - PCx * sp_2.x_0_3; 
  x_0_7 = PBx * sp_1.x_0_1 - PCx * sp_2.x_0_1; 
  x_0_7 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_0_8 = PBy * sp_1.x_0_2 - PCy * sp_2.x_0_2; 
  x_0_8 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_0_9 = PBz * sp_1.x_0_3 - PCz * sp_2.x_0_3; 
  x_0_9 += TwoZetaInv * (VY(0, 0, 1) - VY(0, 0, 2)); 

 } 

/* SD auxilary integral, m=2 */ 
__device__ __inline__ SDint_2::SDint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=2 
  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=3 

  x_0_4 = PBx * sp_2.x_0_2 - PCx * sp_3.x_0_2; 
  x_0_5 = PBy * sp_2.x_0_3 - PCy * sp_3.x_0_3; 
  x_0_6 = PBx * sp_2.x_0_3 - PCx * sp_3.x_0_3; 
  x_0_7 = PBx * sp_2.x_0_1 - PCx * sp_3.x_0_1; 
  x_0_7 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_0_8 = PBy * sp_2.x_0_2 - PCy * sp_3.x_0_2; 
  x_0_8 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_0_9 = PBz * sp_2.x_0_3 - PCz * sp_3.x_0_3; 
  x_0_9 += TwoZetaInv * (VY(0, 0, 2) - VY(0, 0, 3)); 

 } 

/* SD auxilary integral, m=3 */ 
__device__ __inline__ SDint_3::SDint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=3 
  SPint_4 sp_4(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=4 

  x_0_4 = PBx * sp_3.x_0_2 - PCx * sp_4.x_0_2; 
  x_0_5 = PBy * sp_3.x_0_3 - PCy * sp_4.x_0_3; 
  x_0_6 = PBx * sp_3.x_0_3 - PCx * sp_4.x_0_3; 
  x_0_7 = PBx * sp_3.x_0_1 - PCx * sp_4.x_0_1; 
  x_0_7 += TwoZetaInv * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_0_8 = PBy * sp_3.x_0_2 - PCy * sp_4.x_0_2; 
  x_0_8 += TwoZetaInv * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_0_9 = PBz * sp_3.x_0_3 - PCz * sp_4.x_0_3; 
  x_0_9 += TwoZetaInv * (VY(0, 0, 3) - VY(0, 0, 4)); 

 } 

/* SD auxilary integral, m=4 */ 
__device__ __inline__ SDint_4::SDint_4(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_4 sp_4(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=4 
  SPint_5 sp_5(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=5 

  x_0_4 = PBx * sp_4.x_0_2 - PCx * sp_5.x_0_2; 
  x_0_5 = PBy * sp_4.x_0_3 - PCy * sp_5.x_0_3; 
  x_0_6 = PBx * sp_4.x_0_3 - PCx * sp_5.x_0_3; 
  x_0_7 = PBx * sp_4.x_0_1 - PCx * sp_5.x_0_1; 
  x_0_7 += TwoZetaInv * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_0_8 = PBy * sp_4.x_0_2 - PCy * sp_5.x_0_2; 
  x_0_8 += TwoZetaInv * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_0_9 = PBz * sp_4.x_0_3 - PCz * sp_5.x_0_3; 
  x_0_9 += TwoZetaInv * (VY(0, 0, 4) - VY(0, 0, 5)); 

 } 

/* DP true integral, m=0 */ 
__device__ __inline__ DPint_0::DPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_4_1 = PBx * ds_0.x_4_0 - PCx * ds_1.x_4_0; 
  x_4_1 += TwoZetaInv * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_4_2 = PBy * ds_0.x_4_0 - PCy * ds_1.x_4_0; 
  x_4_2 += TwoZetaInv * 1.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_4_3 = PBz * ds_0.x_4_0 - PCz * ds_1.x_4_0; 
  x_5_1 = PBx * ds_0.x_5_0 - PCx * ds_1.x_5_0; 
  x_5_2 = PBy * ds_0.x_5_0 - PCy * ds_1.x_5_0; 
  x_5_2 += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_5_3 = PBz * ds_0.x_5_0 - PCz * ds_1.x_5_0; 
  x_5_3 += TwoZetaInv * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_6_1 = PBx * ds_0.x_6_0 - PCx * ds_1.x_6_0; 
  x_6_1 += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_6_2 = PBy * ds_0.x_6_0 - PCy * ds_1.x_6_0; 
  x_6_3 = PBz * ds_0.x_6_0 - PCz * ds_1.x_6_0; 
  x_6_3 += TwoZetaInv * 1.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_7_1 = PBx * ds_0.x_7_0 - PCx * ds_1.x_7_0; 
  x_7_1 += TwoZetaInv * 2.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_7_2 = PBy * ds_0.x_7_0 - PCy * ds_1.x_7_0; 
  x_7_3 = PBz * ds_0.x_7_0 - PCz * ds_1.x_7_0; 
  x_8_1 = PBx * ds_0.x_8_0 - PCx * ds_1.x_8_0; 
  x_8_2 = PBy * ds_0.x_8_0 - PCy * ds_1.x_8_0; 
  x_8_2 += TwoZetaInv * 2.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_8_3 = PBz * ds_0.x_8_0 - PCz * ds_1.x_8_0; 
  x_9_1 = PBx * ds_0.x_9_0 - PCx * ds_1.x_9_0; 
  x_9_2 = PBy * ds_0.x_9_0 - PCy * ds_1.x_9_0; 
  x_9_3 = PBz * ds_0.x_9_0 - PCz * ds_1.x_9_0; 
  x_9_3 += TwoZetaInv * 2.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 

 } 

/* DP integral partial class - Part 1, m=0 */ 
__device__ __inline__ DPint_0_1::DPint_0_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_4_1 = PBx * ds_0.x_4_0 - PCx * ds_1.x_4_0; 
  x_4_1 += TwoZetaInv * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_4_2 = PBy * ds_0.x_4_0 - PCy * ds_1.x_4_0; 
  x_4_2 += TwoZetaInv * 1.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_4_3 = PBz * ds_0.x_4_0 - PCz * ds_1.x_4_0; 

 } 

/* DP integral partial class - Part 2, m=0 */ 
__device__ __inline__ DPint_0_2::DPint_0_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_5_1 = PBx * ds_0.x_5_0 - PCx * ds_1.x_5_0; 
  x_5_2 = PBy * ds_0.x_5_0 - PCy * ds_1.x_5_0; 
  x_5_2 += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_5_3 = PBz * ds_0.x_5_0 - PCz * ds_1.x_5_0; 
  x_5_3 += TwoZetaInv * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 

 } 

/* DP integral partial class - Part 3, m=0 */ 
__device__ __inline__ DPint_0_3::DPint_0_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_6_1 = PBx * ds_0.x_6_0 - PCx * ds_1.x_6_0; 
  x_6_1 += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_6_2 = PBy * ds_0.x_6_0 - PCy * ds_1.x_6_0; 
  x_6_3 = PBz * ds_0.x_6_0 - PCz * ds_1.x_6_0; 
  x_6_3 += TwoZetaInv * 1.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 

 } 

/* DP integral partial class - Part 4, m=0 */ 
__device__ __inline__ DPint_0_4::DPint_0_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_7_1 = PBx * ds_0.x_7_0 - PCx * ds_1.x_7_0; 
  x_7_1 += TwoZetaInv * 2.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_7_2 = PBy * ds_0.x_7_0 - PCy * ds_1.x_7_0; 
  x_7_3 = PBz * ds_0.x_7_0 - PCz * ds_1.x_7_0; 

 } 

/* DP integral partial class - Part 5, m=0 */ 
__device__ __inline__ DPint_0_5::DPint_0_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_8_1 = PBx * ds_0.x_8_0 - PCx * ds_1.x_8_0; 
  x_8_2 = PBy * ds_0.x_8_0 - PCy * ds_1.x_8_0; 
  x_8_2 += TwoZetaInv * 2.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_8_3 = PBz * ds_0.x_8_0 - PCz * ds_1.x_8_0; 

 } 

/* DP integral partial class - Part 6, m=0 */ 
__device__ __inline__ DPint_0_6::DPint_0_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

  x_9_1 = PBx * ds_0.x_9_0 - PCx * ds_1.x_9_0; 
  x_9_2 = PBy * ds_0.x_9_0 - PCy * ds_1.x_9_0; 
  x_9_3 = PBz * ds_0.x_9_0 - PCz * ds_1.x_9_0; 
  x_9_3 += TwoZetaInv * 2.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 

 } 

/* DP auxilary integral, m=1 */ 
__device__ __inline__ DPint_1::DPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_4_1 = PBx * ds_1.x_4_0 - PCx * ds_2.x_4_0; 
  x_4_1 += TwoZetaInv * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_4_2 = PBy * ds_1.x_4_0 - PCy * ds_2.x_4_0; 
  x_4_2 += TwoZetaInv * 1.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_4_3 = PBz * ds_1.x_4_0 - PCz * ds_2.x_4_0; 
  x_5_1 = PBx * ds_1.x_5_0 - PCx * ds_2.x_5_0; 
  x_5_2 = PBy * ds_1.x_5_0 - PCy * ds_2.x_5_0; 
  x_5_2 += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_5_3 = PBz * ds_1.x_5_0 - PCz * ds_2.x_5_0; 
  x_5_3 += TwoZetaInv * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_6_1 = PBx * ds_1.x_6_0 - PCx * ds_2.x_6_0; 
  x_6_1 += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_6_2 = PBy * ds_1.x_6_0 - PCy * ds_2.x_6_0; 
  x_6_3 = PBz * ds_1.x_6_0 - PCz * ds_2.x_6_0; 
  x_6_3 += TwoZetaInv * 1.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_7_1 = PBx * ds_1.x_7_0 - PCx * ds_2.x_7_0; 
  x_7_1 += TwoZetaInv * 2.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_7_2 = PBy * ds_1.x_7_0 - PCy * ds_2.x_7_0; 
  x_7_3 = PBz * ds_1.x_7_0 - PCz * ds_2.x_7_0; 
  x_8_1 = PBx * ds_1.x_8_0 - PCx * ds_2.x_8_0; 
  x_8_2 = PBy * ds_1.x_8_0 - PCy * ds_2.x_8_0; 
  x_8_2 += TwoZetaInv * 2.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_8_3 = PBz * ds_1.x_8_0 - PCz * ds_2.x_8_0; 
  x_9_1 = PBx * ds_1.x_9_0 - PCx * ds_2.x_9_0; 
  x_9_2 = PBy * ds_1.x_9_0 - PCy * ds_2.x_9_0; 
  x_9_3 = PBz * ds_1.x_9_0 - PCz * ds_2.x_9_0; 
  x_9_3 += TwoZetaInv * 2.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 

 } 

/* DP integral partial class - Part 1, m=1 */ 
__device__ __inline__ DPint_1_1::DPint_1_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_4_1 = PBx * ds_1.x_4_0 - PCx * ds_2.x_4_0; 
  x_4_1 += TwoZetaInv * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_4_2 = PBy * ds_1.x_4_0 - PCy * ds_2.x_4_0; 
  x_4_2 += TwoZetaInv * 1.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_4_3 = PBz * ds_1.x_4_0 - PCz * ds_2.x_4_0; 

 } 

/* DP integral partial class - Part 2, m=1 */ 
__device__ __inline__ DPint_1_2::DPint_1_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_5_1 = PBx * ds_1.x_5_0 - PCx * ds_2.x_5_0; 
  x_5_2 = PBy * ds_1.x_5_0 - PCy * ds_2.x_5_0; 
  x_5_2 += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_5_3 = PBz * ds_1.x_5_0 - PCz * ds_2.x_5_0; 
  x_5_3 += TwoZetaInv * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 

 } 

/* DP integral partial class - Part 3, m=1 */ 
__device__ __inline__ DPint_1_3::DPint_1_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_6_1 = PBx * ds_1.x_6_0 - PCx * ds_2.x_6_0; 
  x_6_1 += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_6_2 = PBy * ds_1.x_6_0 - PCy * ds_2.x_6_0; 
  x_6_3 = PBz * ds_1.x_6_0 - PCz * ds_2.x_6_0; 
  x_6_3 += TwoZetaInv * 1.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 

 } 

/* DP integral partial class - Part 4, m=1 */ 
__device__ __inline__ DPint_1_4::DPint_1_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_7_1 = PBx * ds_1.x_7_0 - PCx * ds_2.x_7_0; 
  x_7_1 += TwoZetaInv * 2.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_7_2 = PBy * ds_1.x_7_0 - PCy * ds_2.x_7_0; 
  x_7_3 = PBz * ds_1.x_7_0 - PCz * ds_2.x_7_0; 

 } 

/* DP integral partial class - Part 5, m=1 */ 
__device__ __inline__ DPint_1_5::DPint_1_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_8_1 = PBx * ds_1.x_8_0 - PCx * ds_2.x_8_0; 
  x_8_2 = PBy * ds_1.x_8_0 - PCy * ds_2.x_8_0; 
  x_8_2 += TwoZetaInv * 2.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_8_3 = PBz * ds_1.x_8_0 - PCz * ds_2.x_8_0; 

 } 

/* DP integral partial class - Part 6, m=1 */ 
__device__ __inline__ DPint_1_6::DPint_1_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

  x_9_1 = PBx * ds_1.x_9_0 - PCx * ds_2.x_9_0; 
  x_9_2 = PBy * ds_1.x_9_0 - PCy * ds_2.x_9_0; 
  x_9_3 = PBz * ds_1.x_9_0 - PCz * ds_2.x_9_0; 
  x_9_3 += TwoZetaInv * 2.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 

 } 

/* DP auxilary integral, m=2 */ 
__device__ __inline__ DPint_2::DPint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_4_1 = PBx * ds_2.x_4_0 - PCx * ds_3.x_4_0; 
  x_4_1 += TwoZetaInv * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_4_2 = PBy * ds_2.x_4_0 - PCy * ds_3.x_4_0; 
  x_4_2 += TwoZetaInv * 1.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_4_3 = PBz * ds_2.x_4_0 - PCz * ds_3.x_4_0; 
  x_5_1 = PBx * ds_2.x_5_0 - PCx * ds_3.x_5_0; 
  x_5_2 = PBy * ds_2.x_5_0 - PCy * ds_3.x_5_0; 
  x_5_2 += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_5_3 = PBz * ds_2.x_5_0 - PCz * ds_3.x_5_0; 
  x_5_3 += TwoZetaInv * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_6_1 = PBx * ds_2.x_6_0 - PCx * ds_3.x_6_0; 
  x_6_1 += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_6_2 = PBy * ds_2.x_6_0 - PCy * ds_3.x_6_0; 
  x_6_3 = PBz * ds_2.x_6_0 - PCz * ds_3.x_6_0; 
  x_6_3 += TwoZetaInv * 1.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_7_1 = PBx * ds_2.x_7_0 - PCx * ds_3.x_7_0; 
  x_7_1 += TwoZetaInv * 2.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_7_2 = PBy * ds_2.x_7_0 - PCy * ds_3.x_7_0; 
  x_7_3 = PBz * ds_2.x_7_0 - PCz * ds_3.x_7_0; 
  x_8_1 = PBx * ds_2.x_8_0 - PCx * ds_3.x_8_0; 
  x_8_2 = PBy * ds_2.x_8_0 - PCy * ds_3.x_8_0; 
  x_8_2 += TwoZetaInv * 2.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_8_3 = PBz * ds_2.x_8_0 - PCz * ds_3.x_8_0; 
  x_9_1 = PBx * ds_2.x_9_0 - PCx * ds_3.x_9_0; 
  x_9_2 = PBy * ds_2.x_9_0 - PCy * ds_3.x_9_0; 
  x_9_3 = PBz * ds_2.x_9_0 - PCz * ds_3.x_9_0; 
  x_9_3 += TwoZetaInv * 2.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 

 } 

/* DP integral partial class - Part 1, m=2 */ 
__device__ __inline__ DPint_2_1::DPint_2_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_4_1 = PBx * ds_2.x_4_0 - PCx * ds_3.x_4_0; 
  x_4_1 += TwoZetaInv * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_4_2 = PBy * ds_2.x_4_0 - PCy * ds_3.x_4_0; 
  x_4_2 += TwoZetaInv * 1.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_4_3 = PBz * ds_2.x_4_0 - PCz * ds_3.x_4_0; 

 } 

/* DP integral partial class - Part 2, m=2 */ 
__device__ __inline__ DPint_2_2::DPint_2_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_5_1 = PBx * ds_2.x_5_0 - PCx * ds_3.x_5_0; 
  x_5_2 = PBy * ds_2.x_5_0 - PCy * ds_3.x_5_0; 
  x_5_2 += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_5_3 = PBz * ds_2.x_5_0 - PCz * ds_3.x_5_0; 
  x_5_3 += TwoZetaInv * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 

 } 

/* DP integral partial class - Part 3, m=2 */ 
__device__ __inline__ DPint_2_3::DPint_2_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_6_1 = PBx * ds_2.x_6_0 - PCx * ds_3.x_6_0; 
  x_6_1 += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_6_2 = PBy * ds_2.x_6_0 - PCy * ds_3.x_6_0; 
  x_6_3 = PBz * ds_2.x_6_0 - PCz * ds_3.x_6_0; 
  x_6_3 += TwoZetaInv * 1.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 

 } 

/* DP integral partial class - Part 4, m=2 */ 
__device__ __inline__ DPint_2_4::DPint_2_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_7_1 = PBx * ds_2.x_7_0 - PCx * ds_3.x_7_0; 
  x_7_1 += TwoZetaInv * 2.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_7_2 = PBy * ds_2.x_7_0 - PCy * ds_3.x_7_0; 
  x_7_3 = PBz * ds_2.x_7_0 - PCz * ds_3.x_7_0; 

 } 

/* DP integral partial class - Part 5, m=2 */ 
__device__ __inline__ DPint_2_5::DPint_2_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_8_1 = PBx * ds_2.x_8_0 - PCx * ds_3.x_8_0; 
  x_8_2 = PBy * ds_2.x_8_0 - PCy * ds_3.x_8_0; 
  x_8_2 += TwoZetaInv * 2.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_8_3 = PBz * ds_2.x_8_0 - PCz * ds_3.x_8_0; 

 } 

/* DP integral partial class - Part 6, m=2 */ 
__device__ __inline__ DPint_2_6::DPint_2_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

  x_9_1 = PBx * ds_2.x_9_0 - PCx * ds_3.x_9_0; 
  x_9_2 = PBy * ds_2.x_9_0 - PCy * ds_3.x_9_0; 
  x_9_3 = PBz * ds_2.x_9_0 - PCz * ds_3.x_9_0; 
  x_9_3 += TwoZetaInv * 2.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 

 } 

/* PD true integral, m=0 */ 
__device__ __inline__ PDint_0::PDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=1 
  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 

  x_1_4 = PAx * sd_0.x_0_4 - PCx * sd_1.x_0_4; 
  x_1_4 += TwoZetaInv * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_2_4 = PAy * sd_0.x_0_4 - PCy * sd_1.x_0_4; 
  x_2_4 += TwoZetaInv * 1.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_3_4 = PAz * sd_0.x_0_4 - PCz * sd_1.x_0_4; 
  x_1_5 = PAx * sd_0.x_0_5 - PCx * sd_1.x_0_5; 
  x_2_5 = PAy * sd_0.x_0_5 - PCy * sd_1.x_0_5; 
  x_2_5 += TwoZetaInv * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_3_5 = PAz * sd_0.x_0_5 - PCz * sd_1.x_0_5; 
  x_3_5 += TwoZetaInv * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_1_6 = PAx * sd_0.x_0_6 - PCx * sd_1.x_0_6; 
  x_1_6 += TwoZetaInv * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_2_6 = PAy * sd_0.x_0_6 - PCy * sd_1.x_0_6; 
  x_3_6 = PAz * sd_0.x_0_6 - PCz * sd_1.x_0_6; 
  x_3_6 += TwoZetaInv * 1.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_1_7 = PAx * sd_0.x_0_7 - PCx * sd_1.x_0_7; 
  x_1_7 += TwoZetaInv * 2.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_2_7 = PAy * sd_0.x_0_7 - PCy * sd_1.x_0_7; 
  x_3_7 = PAz * sd_0.x_0_7 - PCz * sd_1.x_0_7; 
  x_1_8 = PAx * sd_0.x_0_8 - PCx * sd_1.x_0_8; 
  x_2_8 = PAy * sd_0.x_0_8 - PCy * sd_1.x_0_8; 
  x_2_8 += TwoZetaInv * 2.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_3_8 = PAz * sd_0.x_0_8 - PCz * sd_1.x_0_8; 
  x_1_9 = PAx * sd_0.x_0_9 - PCx * sd_1.x_0_9; 
  x_2_9 = PAy * sd_0.x_0_9 - PCy * sd_1.x_0_9; 
  x_3_9 = PAz * sd_0.x_0_9 - PCz * sd_1.x_0_9; 
  x_3_9 += TwoZetaInv * 2.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 

 } 

/* PD auxilary integral, m=1 */ 
__device__ __inline__ PDint_1::PDint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=1 
  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=2 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 

  x_1_4 = PAx * sd_1.x_0_4 - PCx * sd_2.x_0_4; 
  x_1_4 += TwoZetaInv * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_2_4 = PAy * sd_1.x_0_4 - PCy * sd_2.x_0_4; 
  x_2_4 += TwoZetaInv * 1.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_3_4 = PAz * sd_1.x_0_4 - PCz * sd_2.x_0_4; 
  x_1_5 = PAx * sd_1.x_0_5 - PCx * sd_2.x_0_5; 
  x_2_5 = PAy * sd_1.x_0_5 - PCy * sd_2.x_0_5; 
  x_2_5 += TwoZetaInv * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_3_5 = PAz * sd_1.x_0_5 - PCz * sd_2.x_0_5; 
  x_3_5 += TwoZetaInv * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_1_6 = PAx * sd_1.x_0_6 - PCx * sd_2.x_0_6; 
  x_1_6 += TwoZetaInv * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_2_6 = PAy * sd_1.x_0_6 - PCy * sd_2.x_0_6; 
  x_3_6 = PAz * sd_1.x_0_6 - PCz * sd_2.x_0_6; 
  x_3_6 += TwoZetaInv * 1.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_1_7 = PAx * sd_1.x_0_7 - PCx * sd_2.x_0_7; 
  x_1_7 += TwoZetaInv * 2.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_2_7 = PAy * sd_1.x_0_7 - PCy * sd_2.x_0_7; 
  x_3_7 = PAz * sd_1.x_0_7 - PCz * sd_2.x_0_7; 
  x_1_8 = PAx * sd_1.x_0_8 - PCx * sd_2.x_0_8; 
  x_2_8 = PAy * sd_1.x_0_8 - PCy * sd_2.x_0_8; 
  x_2_8 += TwoZetaInv * 2.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_3_8 = PAz * sd_1.x_0_8 - PCz * sd_2.x_0_8; 
  x_1_9 = PAx * sd_1.x_0_9 - PCx * sd_2.x_0_9; 
  x_2_9 = PAy * sd_1.x_0_9 - PCy * sd_2.x_0_9; 
  x_3_9 = PAz * sd_1.x_0_9 - PCz * sd_2.x_0_9; 
  x_3_9 += TwoZetaInv * 2.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 

 } 

/* PD auxilary integral, m=2 */ 
__device__ __inline__ PDint_2::PDint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=2 
  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=3 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 

  x_1_4 = PAx * sd_2.x_0_4 - PCx * sd_3.x_0_4; 
  x_1_4 += TwoZetaInv * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_2_4 = PAy * sd_2.x_0_4 - PCy * sd_3.x_0_4; 
  x_2_4 += TwoZetaInv * 1.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_3_4 = PAz * sd_2.x_0_4 - PCz * sd_3.x_0_4; 
  x_1_5 = PAx * sd_2.x_0_5 - PCx * sd_3.x_0_5; 
  x_2_5 = PAy * sd_2.x_0_5 - PCy * sd_3.x_0_5; 
  x_2_5 += TwoZetaInv * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_3_5 = PAz * sd_2.x_0_5 - PCz * sd_3.x_0_5; 
  x_3_5 += TwoZetaInv * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_1_6 = PAx * sd_2.x_0_6 - PCx * sd_3.x_0_6; 
  x_1_6 += TwoZetaInv * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_2_6 = PAy * sd_2.x_0_6 - PCy * sd_3.x_0_6; 
  x_3_6 = PAz * sd_2.x_0_6 - PCz * sd_3.x_0_6; 
  x_3_6 += TwoZetaInv * 1.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_1_7 = PAx * sd_2.x_0_7 - PCx * sd_3.x_0_7; 
  x_1_7 += TwoZetaInv * 2.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_2_7 = PAy * sd_2.x_0_7 - PCy * sd_3.x_0_7; 
  x_3_7 = PAz * sd_2.x_0_7 - PCz * sd_3.x_0_7; 
  x_1_8 = PAx * sd_2.x_0_8 - PCx * sd_3.x_0_8; 
  x_2_8 = PAy * sd_2.x_0_8 - PCy * sd_3.x_0_8; 
  x_2_8 += TwoZetaInv * 2.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_3_8 = PAz * sd_2.x_0_8 - PCz * sd_3.x_0_8; 
  x_1_9 = PAx * sd_2.x_0_9 - PCx * sd_3.x_0_9; 
  x_2_9 = PAy * sd_2.x_0_9 - PCy * sd_3.x_0_9; 
  x_3_9 = PAz * sd_2.x_0_9 - PCz * sd_3.x_0_9; 
  x_3_9 += TwoZetaInv * 2.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 

 } 

/* DD true integral, m=0 */ 
__device__ __inline__ DDint_0::DDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PPint_0 pp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|p] for m=0 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
#ifndef USE_PARTIAL_DP 
  DPint_0 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
#endif 
  PPint_1 pp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|p] for m=1 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
#ifndef USE_PARTIAL_DP 
  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
#endif 

#ifdef REG_DD 
  x_4_4 = PBx * dp_0.x_4_2 - PCx * dp_1.x_4_2; 
  x_4_4 += TwoZetaInv * 1.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  x_4_5 = PBy * dp_0.x_4_3 - PCy * dp_1.x_4_3; 
  x_4_5 += TwoZetaInv * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_4_6 = PBx * dp_0.x_4_3 - PCx * dp_1.x_4_3; 
  x_4_6 += TwoZetaInv * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_4_7 = PBx * dp_0.x_4_1 - PCx * dp_1.x_4_1; 
  x_4_7 += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_4_7 += TwoZetaInv * 1.000000 * (pp_0.x_2_1 - pp_1.x_2_1); 
  x_4_8 = PBy * dp_0.x_4_2 - PCy * dp_1.x_4_2; 
  x_4_8 += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_4_8 += TwoZetaInv * 1.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  x_4_9 = PBz * dp_0.x_4_3 - PCz * dp_1.x_4_3; 
  x_4_9 += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_5_4 = PBx * dp_0.x_5_2 - PCx * dp_1.x_5_2; 
  x_5_5 = PBy * dp_0.x_5_3 - PCy * dp_1.x_5_3; 
  x_5_5 += TwoZetaInv * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  x_5_6 = PBx * dp_0.x_5_3 - PCx * dp_1.x_5_3; 
  x_5_7 = PBx * dp_0.x_5_1 - PCx * dp_1.x_5_1; 
  x_5_7 += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_5_8 = PBy * dp_0.x_5_2 - PCy * dp_1.x_5_2; 
  x_5_8 += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_5_8 += TwoZetaInv * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  x_5_9 = PBz * dp_0.x_5_3 - PCz * dp_1.x_5_3; 
  x_5_9 += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_5_9 += TwoZetaInv * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_6_4 = PBx * dp_0.x_6_2 - PCx * dp_1.x_6_2; 
  x_6_4 += TwoZetaInv * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  x_6_5 = PBy * dp_0.x_6_3 - PCy * dp_1.x_6_3; 
  x_6_6 = PBx * dp_0.x_6_3 - PCx * dp_1.x_6_3; 
  x_6_6 += TwoZetaInv * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  x_6_7 = PBx * dp_0.x_6_1 - PCx * dp_1.x_6_1; 
  x_6_7 += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_6_7 += TwoZetaInv * 1.000000 * (pp_0.x_3_1 - pp_1.x_3_1); 
  x_6_8 = PBy * dp_0.x_6_2 - PCy * dp_1.x_6_2; 
  x_6_8 += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_6_9 = PBz * dp_0.x_6_3 - PCz * dp_1.x_6_3; 
  x_6_9 += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_6_9 += TwoZetaInv * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_7_4 = PBx * dp_0.x_7_2 - PCx * dp_1.x_7_2; 
  x_7_4 += TwoZetaInv * 2.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  x_7_5 = PBy * dp_0.x_7_3 - PCy * dp_1.x_7_3; 
  x_7_6 = PBx * dp_0.x_7_3 - PCx * dp_1.x_7_3; 
  x_7_6 += TwoZetaInv * 2.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_7_7 = PBx * dp_0.x_7_1 - PCx * dp_1.x_7_1; 
  x_7_7 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_7_7 += TwoZetaInv * 2.000000 * (pp_0.x_1_1 - pp_1.x_1_1); 
  x_7_8 = PBy * dp_0.x_7_2 - PCy * dp_1.x_7_2; 
  x_7_8 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_7_9 = PBz * dp_0.x_7_3 - PCz * dp_1.x_7_3; 
  x_7_9 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_8_4 = PBx * dp_0.x_8_2 - PCx * dp_1.x_8_2; 
  x_8_5 = PBy * dp_0.x_8_3 - PCy * dp_1.x_8_3; 
  x_8_5 += TwoZetaInv * 2.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_8_6 = PBx * dp_0.x_8_3 - PCx * dp_1.x_8_3; 
  x_8_7 = PBx * dp_0.x_8_1 - PCx * dp_1.x_8_1; 
  x_8_7 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_8_8 = PBy * dp_0.x_8_2 - PCy * dp_1.x_8_2; 
  x_8_8 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_8_8 += TwoZetaInv * 2.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  x_8_9 = PBz * dp_0.x_8_3 - PCz * dp_1.x_8_3; 
  x_8_9 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_9_4 = PBx * dp_0.x_9_2 - PCx * dp_1.x_9_2; 
  x_9_5 = PBy * dp_0.x_9_3 - PCy * dp_1.x_9_3; 
  x_9_6 = PBx * dp_0.x_9_3 - PCx * dp_1.x_9_3; 
  x_9_7 = PBx * dp_0.x_9_1 - PCx * dp_1.x_9_1; 
  x_9_7 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_8 = PBy * dp_0.x_9_2 - PCy * dp_1.x_9_2; 
  x_9_8 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_9 = PBz * dp_0.x_9_3 - PCz * dp_1.x_9_3; 
  x_9_9 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_9 += TwoZetaInv * 2.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
#else 
#ifdef USE_PARTIAL_DP 
  { 
    DPint_0_1 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
    DPint_1_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 

    QUICKDouble val; 

    val = PBx * dp_0.x_4_2 - PCx * dp_1.x_4_2; 
    val += TwoZetaInv * 1.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
    LOCSTOREFULL(store, 4, 4, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_4_3 - PCy * dp_1.x_4_3; 
    val += TwoZetaInv * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
    LOCSTOREFULL(store, 4, 5, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_4_3 - PCx * dp_1.x_4_3; 
    val += TwoZetaInv * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
    LOCSTOREFULL(store, 4, 6, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_4_1 - PCx * dp_1.x_4_1; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
    val += TwoZetaInv * 1.000000 * (pp_0.x_2_1 - pp_1.x_2_1); 
    LOCSTOREFULL(store, 4, 7, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_4_2 - PCy * dp_1.x_4_2; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
    val += TwoZetaInv * 1.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
    LOCSTOREFULL(store, 4, 8, STOREDIM, STOREDIM, 0) = val; 
    val = PBz * dp_0.x_4_3 - PCz * dp_1.x_4_3; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
    LOCSTOREFULL(store, 4, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    DPint_0_2 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
    DPint_1_2 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 

    QUICKDouble val; 

    val = PBx * dp_0.x_5_2 - PCx * dp_1.x_5_2; 
    LOCSTOREFULL(store, 5, 4, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_5_3 - PCy * dp_1.x_5_3; 
    val += TwoZetaInv * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
    LOCSTOREFULL(store, 5, 5, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_5_3 - PCx * dp_1.x_5_3; 
    LOCSTOREFULL(store, 5, 6, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_5_1 - PCx * dp_1.x_5_1; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
    LOCSTOREFULL(store, 5, 7, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_5_2 - PCy * dp_1.x_5_2; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
    val += TwoZetaInv * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
    LOCSTOREFULL(store, 5, 8, STOREDIM, STOREDIM, 0) = val; 
    val = PBz * dp_0.x_5_3 - PCz * dp_1.x_5_3; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
    val += TwoZetaInv * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
    LOCSTOREFULL(store, 5, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    DPint_0_3 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
    DPint_1_3 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 

    QUICKDouble val; 

    val = PBx * dp_0.x_6_2 - PCx * dp_1.x_6_2; 
    val += TwoZetaInv * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
    LOCSTOREFULL(store, 6, 4, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_6_3 - PCy * dp_1.x_6_3; 
    LOCSTOREFULL(store, 6, 5, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_6_3 - PCx * dp_1.x_6_3; 
    val += TwoZetaInv * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
    LOCSTOREFULL(store, 6, 6, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_6_1 - PCx * dp_1.x_6_1; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
    val += TwoZetaInv * 1.000000 * (pp_0.x_3_1 - pp_1.x_3_1); 
    LOCSTOREFULL(store, 6, 7, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_6_2 - PCy * dp_1.x_6_2; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
    LOCSTOREFULL(store, 6, 8, STOREDIM, STOREDIM, 0) = val; 
    val = PBz * dp_0.x_6_3 - PCz * dp_1.x_6_3; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
    val += TwoZetaInv * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
    LOCSTOREFULL(store, 6, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    DPint_0_4 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
    DPint_1_4 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 

    QUICKDouble val; 

    val = PBx * dp_0.x_7_2 - PCx * dp_1.x_7_2; 
    val += TwoZetaInv * 2.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
    LOCSTOREFULL(store, 7, 4, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_7_3 - PCy * dp_1.x_7_3; 
    LOCSTOREFULL(store, 7, 5, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_7_3 - PCx * dp_1.x_7_3; 
    val += TwoZetaInv * 2.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
    LOCSTOREFULL(store, 7, 6, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_7_1 - PCx * dp_1.x_7_1; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
    val += TwoZetaInv * 2.000000 * (pp_0.x_1_1 - pp_1.x_1_1); 
    LOCSTOREFULL(store, 7, 7, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_7_2 - PCy * dp_1.x_7_2; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
    LOCSTOREFULL(store, 7, 8, STOREDIM, STOREDIM, 0) = val; 
    val = PBz * dp_0.x_7_3 - PCz * dp_1.x_7_3; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
    LOCSTOREFULL(store, 7, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    DPint_0_5 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
    DPint_1_5 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 

    QUICKDouble val; 

    val = PBx * dp_0.x_8_2 - PCx * dp_1.x_8_2; 
    LOCSTOREFULL(store, 8, 4, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_8_3 - PCy * dp_1.x_8_3; 
    val += TwoZetaInv * 2.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
    LOCSTOREFULL(store, 8, 5, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_8_3 - PCx * dp_1.x_8_3; 
    LOCSTOREFULL(store, 8, 6, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_8_1 - PCx * dp_1.x_8_1; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
    LOCSTOREFULL(store, 8, 7, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_8_2 - PCy * dp_1.x_8_2; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
    val += TwoZetaInv * 2.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
    LOCSTOREFULL(store, 8, 8, STOREDIM, STOREDIM, 0) = val; 
    val = PBz * dp_0.x_8_3 - PCz * dp_1.x_8_3; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
    LOCSTOREFULL(store, 8, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    DPint_0_6 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
    DPint_1_6 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 

    QUICKDouble val; 

    val = PBx * dp_0.x_9_2 - PCx * dp_1.x_9_2; 
    LOCSTOREFULL(store, 9, 4, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_9_3 - PCy * dp_1.x_9_3; 
    LOCSTOREFULL(store, 9, 5, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_9_3 - PCx * dp_1.x_9_3; 
    LOCSTOREFULL(store, 9, 6, STOREDIM, STOREDIM, 0) = val; 
    val = PBx * dp_0.x_9_1 - PCx * dp_1.x_9_1; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
    LOCSTOREFULL(store, 9, 7, STOREDIM, STOREDIM, 0) = val; 
    val = PBy * dp_0.x_9_2 - PCy * dp_1.x_9_2; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
    LOCSTOREFULL(store, 9, 8, STOREDIM, STOREDIM, 0) = val; 
    val = PBz * dp_0.x_9_3 - PCz * dp_1.x_9_3; 
    val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
    val += TwoZetaInv * 2.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
    LOCSTOREFULL(store, 9, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

#else 

  QUICKDouble val; 
  val = PBx * dp_0.x_4_2 - PCx * dp_1.x_4_2; 
  val += TwoZetaInv * 1.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  LOCSTOREFULL(store, 4, 4, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_4_3 - PCy * dp_1.x_4_3; 
  val += TwoZetaInv * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  LOCSTOREFULL(store, 4, 5, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_4_3 - PCx * dp_1.x_4_3; 
  val += TwoZetaInv * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  LOCSTOREFULL(store, 4, 6, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_4_1 - PCx * dp_1.x_4_1; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  val += TwoZetaInv * 1.000000 * (pp_0.x_2_1 - pp_1.x_2_1); 
  LOCSTOREFULL(store, 4, 7, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_4_2 - PCy * dp_1.x_4_2; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  val += TwoZetaInv * 1.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  LOCSTOREFULL(store, 4, 8, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * dp_0.x_4_3 - PCz * dp_1.x_4_3; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  LOCSTOREFULL(store, 4, 9, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_5_2 - PCx * dp_1.x_5_2; 
  LOCSTOREFULL(store, 5, 4, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_5_3 - PCy * dp_1.x_5_3; 
  val += TwoZetaInv * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  LOCSTOREFULL(store, 5, 5, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_5_3 - PCx * dp_1.x_5_3; 
  LOCSTOREFULL(store, 5, 6, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_5_1 - PCx * dp_1.x_5_1; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  LOCSTOREFULL(store, 5, 7, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_5_2 - PCy * dp_1.x_5_2; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  val += TwoZetaInv * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  LOCSTOREFULL(store, 5, 8, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * dp_0.x_5_3 - PCz * dp_1.x_5_3; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  val += TwoZetaInv * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  LOCSTOREFULL(store, 5, 9, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_6_2 - PCx * dp_1.x_6_2; 
  val += TwoZetaInv * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  LOCSTOREFULL(store, 6, 4, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_6_3 - PCy * dp_1.x_6_3; 
  LOCSTOREFULL(store, 6, 5, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_6_3 - PCx * dp_1.x_6_3; 
  val += TwoZetaInv * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  LOCSTOREFULL(store, 6, 6, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_6_1 - PCx * dp_1.x_6_1; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  val += TwoZetaInv * 1.000000 * (pp_0.x_3_1 - pp_1.x_3_1); 
  LOCSTOREFULL(store, 6, 7, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_6_2 - PCy * dp_1.x_6_2; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  LOCSTOREFULL(store, 6, 8, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * dp_0.x_6_3 - PCz * dp_1.x_6_3; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  val += TwoZetaInv * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  LOCSTOREFULL(store, 6, 9, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_7_2 - PCx * dp_1.x_7_2; 
  val += TwoZetaInv * 2.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  LOCSTOREFULL(store, 7, 4, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_7_3 - PCy * dp_1.x_7_3; 
  LOCSTOREFULL(store, 7, 5, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_7_3 - PCx * dp_1.x_7_3; 
  val += TwoZetaInv * 2.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  LOCSTOREFULL(store, 7, 6, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_7_1 - PCx * dp_1.x_7_1; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  val += TwoZetaInv * 2.000000 * (pp_0.x_1_1 - pp_1.x_1_1); 
  LOCSTOREFULL(store, 7, 7, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_7_2 - PCy * dp_1.x_7_2; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  LOCSTOREFULL(store, 7, 8, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * dp_0.x_7_3 - PCz * dp_1.x_7_3; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  LOCSTOREFULL(store, 7, 9, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_8_2 - PCx * dp_1.x_8_2; 
  LOCSTOREFULL(store, 8, 4, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_8_3 - PCy * dp_1.x_8_3; 
  val += TwoZetaInv * 2.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  LOCSTOREFULL(store, 8, 5, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_8_3 - PCx * dp_1.x_8_3; 
  LOCSTOREFULL(store, 8, 6, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_8_1 - PCx * dp_1.x_8_1; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  LOCSTOREFULL(store, 8, 7, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_8_2 - PCy * dp_1.x_8_2; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  val += TwoZetaInv * 2.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  LOCSTOREFULL(store, 8, 8, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * dp_0.x_8_3 - PCz * dp_1.x_8_3; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  LOCSTOREFULL(store, 8, 9, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_9_2 - PCx * dp_1.x_9_2; 
  LOCSTOREFULL(store, 9, 4, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_9_3 - PCy * dp_1.x_9_3; 
  LOCSTOREFULL(store, 9, 5, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_9_3 - PCx * dp_1.x_9_3; 
  LOCSTOREFULL(store, 9, 6, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * dp_0.x_9_1 - PCx * dp_1.x_9_1; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  LOCSTOREFULL(store, 9, 7, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * dp_0.x_9_2 - PCy * dp_1.x_9_2; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  LOCSTOREFULL(store, 9, 8, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * dp_0.x_9_3 - PCz * dp_1.x_9_3; 
  val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  val += TwoZetaInv * 2.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  LOCSTOREFULL(store, 9, 9, STOREDIM, STOREDIM, 0) = val; 
#endif 
#endif 

 } 

/* DD auxilary integral, m=1 */ 
__device__ __inline__ DDint_1::DDint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PPint_1 pp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|p] for m=1 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
#ifndef USE_PARTIAL_DP 
  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
#endif 
  PPint_2 pp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|p] for m=2 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
#ifndef USE_PARTIAL_DP 
  DPint_2 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 
#endif 

#ifdef REG_DD 
  x_4_4 = PBx * dp_1.x_4_2 - PCx * dp_2.x_4_2; 
  x_4_4 += TwoZetaInv * 1.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
  x_4_5 = PBy * dp_1.x_4_3 - PCy * dp_2.x_4_3; 
  x_4_5 += TwoZetaInv * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  x_4_6 = PBx * dp_1.x_4_3 - PCx * dp_2.x_4_3; 
  x_4_6 += TwoZetaInv * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  x_4_7 = PBx * dp_1.x_4_1 - PCx * dp_2.x_4_1; 
  x_4_7 += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_4_7 += TwoZetaInv * 1.000000 * (pp_1.x_2_1 - pp_2.x_2_1); 
  x_4_8 = PBy * dp_1.x_4_2 - PCy * dp_2.x_4_2; 
  x_4_8 += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_4_8 += TwoZetaInv * 1.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
  x_4_9 = PBz * dp_1.x_4_3 - PCz * dp_2.x_4_3; 
  x_4_9 += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_5_4 = PBx * dp_1.x_5_2 - PCx * dp_2.x_5_2; 
  x_5_5 = PBy * dp_1.x_5_3 - PCy * dp_2.x_5_3; 
  x_5_5 += TwoZetaInv * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  x_5_6 = PBx * dp_1.x_5_3 - PCx * dp_2.x_5_3; 
  x_5_7 = PBx * dp_1.x_5_1 - PCx * dp_2.x_5_1; 
  x_5_7 += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_5_8 = PBy * dp_1.x_5_2 - PCy * dp_2.x_5_2; 
  x_5_8 += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_5_8 += TwoZetaInv * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
  x_5_9 = PBz * dp_1.x_5_3 - PCz * dp_2.x_5_3; 
  x_5_9 += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_5_9 += TwoZetaInv * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  x_6_4 = PBx * dp_1.x_6_2 - PCx * dp_2.x_6_2; 
  x_6_4 += TwoZetaInv * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
  x_6_5 = PBy * dp_1.x_6_3 - PCy * dp_2.x_6_3; 
  x_6_6 = PBx * dp_1.x_6_3 - PCx * dp_2.x_6_3; 
  x_6_6 += TwoZetaInv * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  x_6_7 = PBx * dp_1.x_6_1 - PCx * dp_2.x_6_1; 
  x_6_7 += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_6_7 += TwoZetaInv * 1.000000 * (pp_1.x_3_1 - pp_2.x_3_1); 
  x_6_8 = PBy * dp_1.x_6_2 - PCy * dp_2.x_6_2; 
  x_6_8 += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_6_9 = PBz * dp_1.x_6_3 - PCz * dp_2.x_6_3; 
  x_6_9 += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_6_9 += TwoZetaInv * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  x_7_4 = PBx * dp_1.x_7_2 - PCx * dp_2.x_7_2; 
  x_7_4 += TwoZetaInv * 2.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
  x_7_5 = PBy * dp_1.x_7_3 - PCy * dp_2.x_7_3; 
  x_7_6 = PBx * dp_1.x_7_3 - PCx * dp_2.x_7_3; 
  x_7_6 += TwoZetaInv * 2.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  x_7_7 = PBx * dp_1.x_7_1 - PCx * dp_2.x_7_1; 
  x_7_7 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_7_7 += TwoZetaInv * 2.000000 * (pp_1.x_1_1 - pp_2.x_1_1); 
  x_7_8 = PBy * dp_1.x_7_2 - PCy * dp_2.x_7_2; 
  x_7_8 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_7_9 = PBz * dp_1.x_7_3 - PCz * dp_2.x_7_3; 
  x_7_9 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_8_4 = PBx * dp_1.x_8_2 - PCx * dp_2.x_8_2; 
  x_8_5 = PBy * dp_1.x_8_3 - PCy * dp_2.x_8_3; 
  x_8_5 += TwoZetaInv * 2.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  x_8_6 = PBx * dp_1.x_8_3 - PCx * dp_2.x_8_3; 
  x_8_7 = PBx * dp_1.x_8_1 - PCx * dp_2.x_8_1; 
  x_8_7 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_8_8 = PBy * dp_1.x_8_2 - PCy * dp_2.x_8_2; 
  x_8_8 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_8_8 += TwoZetaInv * 2.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
  x_8_9 = PBz * dp_1.x_8_3 - PCz * dp_2.x_8_3; 
  x_8_9 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_9_4 = PBx * dp_1.x_9_2 - PCx * dp_2.x_9_2; 
  x_9_5 = PBy * dp_1.x_9_3 - PCy * dp_2.x_9_3; 
  x_9_6 = PBx * dp_1.x_9_3 - PCx * dp_2.x_9_3; 
  x_9_7 = PBx * dp_1.x_9_1 - PCx * dp_2.x_9_1; 
  x_9_7 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_9_8 = PBy * dp_1.x_9_2 - PCy * dp_2.x_9_2; 
  x_9_8 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_9_9 = PBz * dp_1.x_9_3 - PCz * dp_2.x_9_3; 
  x_9_9 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_9_9 += TwoZetaInv * 2.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
#else 
#ifdef USE_PARTIAL_DP 
  { 
    DPint_1_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
    DPint_2_1 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 

    QUICKDouble val; 

    val = PBx * dp_1.x_4_2 - PCx * dp_2.x_4_2; 
    val += TwoZetaInv * 1.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
    LOCSTOREFULL(store, 4, 4, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_4_3 - PCy * dp_2.x_4_3; 
    val += TwoZetaInv * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
    LOCSTOREFULL(store, 4, 5, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_4_3 - PCx * dp_2.x_4_3; 
    val += TwoZetaInv * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
    LOCSTOREFULL(store, 4, 6, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_4_1 - PCx * dp_2.x_4_1; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
    val += TwoZetaInv * 1.000000 * (pp_1.x_2_1 - pp_2.x_2_1); 
    LOCSTOREFULL(store, 4, 7, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_4_2 - PCy * dp_2.x_4_2; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
    val += TwoZetaInv * 1.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
    LOCSTOREFULL(store, 4, 8, STOREDIM, STOREDIM, 1) = val; 
    val = PBz * dp_1.x_4_3 - PCz * dp_2.x_4_3; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
    LOCSTOREFULL(store, 4, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    DPint_1_2 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
    DPint_2_2 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 

    QUICKDouble val; 

    val = PBx * dp_1.x_5_2 - PCx * dp_2.x_5_2; 
    LOCSTOREFULL(store, 5, 4, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_5_3 - PCy * dp_2.x_5_3; 
    val += TwoZetaInv * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
    LOCSTOREFULL(store, 5, 5, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_5_3 - PCx * dp_2.x_5_3; 
    LOCSTOREFULL(store, 5, 6, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_5_1 - PCx * dp_2.x_5_1; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
    LOCSTOREFULL(store, 5, 7, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_5_2 - PCy * dp_2.x_5_2; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
    val += TwoZetaInv * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
    LOCSTOREFULL(store, 5, 8, STOREDIM, STOREDIM, 1) = val; 
    val = PBz * dp_1.x_5_3 - PCz * dp_2.x_5_3; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
    val += TwoZetaInv * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
    LOCSTOREFULL(store, 5, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    DPint_1_3 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
    DPint_2_3 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 

    QUICKDouble val; 

    val = PBx * dp_1.x_6_2 - PCx * dp_2.x_6_2; 
    val += TwoZetaInv * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
    LOCSTOREFULL(store, 6, 4, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_6_3 - PCy * dp_2.x_6_3; 
    LOCSTOREFULL(store, 6, 5, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_6_3 - PCx * dp_2.x_6_3; 
    val += TwoZetaInv * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
    LOCSTOREFULL(store, 6, 6, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_6_1 - PCx * dp_2.x_6_1; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
    val += TwoZetaInv * 1.000000 * (pp_1.x_3_1 - pp_2.x_3_1); 
    LOCSTOREFULL(store, 6, 7, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_6_2 - PCy * dp_2.x_6_2; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
    LOCSTOREFULL(store, 6, 8, STOREDIM, STOREDIM, 1) = val; 
    val = PBz * dp_1.x_6_3 - PCz * dp_2.x_6_3; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
    val += TwoZetaInv * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
    LOCSTOREFULL(store, 6, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    DPint_1_4 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
    DPint_2_4 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 

    QUICKDouble val; 

    val = PBx * dp_1.x_7_2 - PCx * dp_2.x_7_2; 
    val += TwoZetaInv * 2.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
    LOCSTOREFULL(store, 7, 4, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_7_3 - PCy * dp_2.x_7_3; 
    LOCSTOREFULL(store, 7, 5, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_7_3 - PCx * dp_2.x_7_3; 
    val += TwoZetaInv * 2.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
    LOCSTOREFULL(store, 7, 6, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_7_1 - PCx * dp_2.x_7_1; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
    val += TwoZetaInv * 2.000000 * (pp_1.x_1_1 - pp_2.x_1_1); 
    LOCSTOREFULL(store, 7, 7, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_7_2 - PCy * dp_2.x_7_2; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
    LOCSTOREFULL(store, 7, 8, STOREDIM, STOREDIM, 1) = val; 
    val = PBz * dp_1.x_7_3 - PCz * dp_2.x_7_3; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
    LOCSTOREFULL(store, 7, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    DPint_1_5 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
    DPint_2_5 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 

    QUICKDouble val; 

    val = PBx * dp_1.x_8_2 - PCx * dp_2.x_8_2; 
    LOCSTOREFULL(store, 8, 4, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_8_3 - PCy * dp_2.x_8_3; 
    val += TwoZetaInv * 2.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
    LOCSTOREFULL(store, 8, 5, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_8_3 - PCx * dp_2.x_8_3; 
    LOCSTOREFULL(store, 8, 6, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_8_1 - PCx * dp_2.x_8_1; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
    LOCSTOREFULL(store, 8, 7, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_8_2 - PCy * dp_2.x_8_2; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
    val += TwoZetaInv * 2.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
    LOCSTOREFULL(store, 8, 8, STOREDIM, STOREDIM, 1) = val; 
    val = PBz * dp_1.x_8_3 - PCz * dp_2.x_8_3; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
    LOCSTOREFULL(store, 8, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    DPint_1_6 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
    DPint_2_6 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 

    QUICKDouble val; 

    val = PBx * dp_1.x_9_2 - PCx * dp_2.x_9_2; 
    LOCSTOREFULL(store, 9, 4, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_9_3 - PCy * dp_2.x_9_3; 
    LOCSTOREFULL(store, 9, 5, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_9_3 - PCx * dp_2.x_9_3; 
    LOCSTOREFULL(store, 9, 6, STOREDIM, STOREDIM, 1) = val; 
    val = PBx * dp_1.x_9_1 - PCx * dp_2.x_9_1; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
    LOCSTOREFULL(store, 9, 7, STOREDIM, STOREDIM, 1) = val; 
    val = PBy * dp_1.x_9_2 - PCy * dp_2.x_9_2; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
    LOCSTOREFULL(store, 9, 8, STOREDIM, STOREDIM, 1) = val; 
    val = PBz * dp_1.x_9_3 - PCz * dp_2.x_9_3; 
    val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
    val += TwoZetaInv * 2.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
    LOCSTOREFULL(store, 9, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

#else 

  QUICKDouble val; 
  val = PBx * dp_1.x_4_2 - PCx * dp_2.x_4_2; 
  val += TwoZetaInv * 1.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
  LOCSTOREFULL(store, 4, 4, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_4_3 - PCy * dp_2.x_4_3; 
  val += TwoZetaInv * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  LOCSTOREFULL(store, 4, 5, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_4_3 - PCx * dp_2.x_4_3; 
  val += TwoZetaInv * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  LOCSTOREFULL(store, 4, 6, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_4_1 - PCx * dp_2.x_4_1; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  val += TwoZetaInv * 1.000000 * (pp_1.x_2_1 - pp_2.x_2_1); 
  LOCSTOREFULL(store, 4, 7, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_4_2 - PCy * dp_2.x_4_2; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  val += TwoZetaInv * 1.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
  LOCSTOREFULL(store, 4, 8, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * dp_1.x_4_3 - PCz * dp_2.x_4_3; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  LOCSTOREFULL(store, 4, 9, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_5_2 - PCx * dp_2.x_5_2; 
  LOCSTOREFULL(store, 5, 4, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_5_3 - PCy * dp_2.x_5_3; 
  val += TwoZetaInv * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  LOCSTOREFULL(store, 5, 5, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_5_3 - PCx * dp_2.x_5_3; 
  LOCSTOREFULL(store, 5, 6, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_5_1 - PCx * dp_2.x_5_1; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  LOCSTOREFULL(store, 5, 7, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_5_2 - PCy * dp_2.x_5_2; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  val += TwoZetaInv * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
  LOCSTOREFULL(store, 5, 8, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * dp_1.x_5_3 - PCz * dp_2.x_5_3; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  val += TwoZetaInv * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  LOCSTOREFULL(store, 5, 9, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_6_2 - PCx * dp_2.x_6_2; 
  val += TwoZetaInv * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
  LOCSTOREFULL(store, 6, 4, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_6_3 - PCy * dp_2.x_6_3; 
  LOCSTOREFULL(store, 6, 5, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_6_3 - PCx * dp_2.x_6_3; 
  val += TwoZetaInv * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  LOCSTOREFULL(store, 6, 6, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_6_1 - PCx * dp_2.x_6_1; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  val += TwoZetaInv * 1.000000 * (pp_1.x_3_1 - pp_2.x_3_1); 
  LOCSTOREFULL(store, 6, 7, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_6_2 - PCy * dp_2.x_6_2; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  LOCSTOREFULL(store, 6, 8, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * dp_1.x_6_3 - PCz * dp_2.x_6_3; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  val += TwoZetaInv * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  LOCSTOREFULL(store, 6, 9, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_7_2 - PCx * dp_2.x_7_2; 
  val += TwoZetaInv * 2.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
  LOCSTOREFULL(store, 7, 4, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_7_3 - PCy * dp_2.x_7_3; 
  LOCSTOREFULL(store, 7, 5, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_7_3 - PCx * dp_2.x_7_3; 
  val += TwoZetaInv * 2.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  LOCSTOREFULL(store, 7, 6, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_7_1 - PCx * dp_2.x_7_1; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  val += TwoZetaInv * 2.000000 * (pp_1.x_1_1 - pp_2.x_1_1); 
  LOCSTOREFULL(store, 7, 7, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_7_2 - PCy * dp_2.x_7_2; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  LOCSTOREFULL(store, 7, 8, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * dp_1.x_7_3 - PCz * dp_2.x_7_3; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  LOCSTOREFULL(store, 7, 9, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_8_2 - PCx * dp_2.x_8_2; 
  LOCSTOREFULL(store, 8, 4, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_8_3 - PCy * dp_2.x_8_3; 
  val += TwoZetaInv * 2.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  LOCSTOREFULL(store, 8, 5, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_8_3 - PCx * dp_2.x_8_3; 
  LOCSTOREFULL(store, 8, 6, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_8_1 - PCx * dp_2.x_8_1; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  LOCSTOREFULL(store, 8, 7, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_8_2 - PCy * dp_2.x_8_2; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  val += TwoZetaInv * 2.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
  LOCSTOREFULL(store, 8, 8, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * dp_1.x_8_3 - PCz * dp_2.x_8_3; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  LOCSTOREFULL(store, 8, 9, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_9_2 - PCx * dp_2.x_9_2; 
  LOCSTOREFULL(store, 9, 4, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_9_3 - PCy * dp_2.x_9_3; 
  LOCSTOREFULL(store, 9, 5, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_9_3 - PCx * dp_2.x_9_3; 
  LOCSTOREFULL(store, 9, 6, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * dp_1.x_9_1 - PCx * dp_2.x_9_1; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  LOCSTOREFULL(store, 9, 7, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * dp_1.x_9_2 - PCy * dp_2.x_9_2; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  LOCSTOREFULL(store, 9, 8, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * dp_1.x_9_3 - PCz * dp_2.x_9_3; 
  val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  val += TwoZetaInv * 2.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  LOCSTOREFULL(store, 9, 9, STOREDIM, STOREDIM, 1) = val; 
#endif 
#endif 

 } 

/* FS true integral, m=0 */ 
__device__ __inline__ FSint_0::FSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 

#ifdef REG_FS 
  x_10_0 = PAx * ds_0.x_5_0 - PCx * ds_1.x_5_0; 
  x_11_0 = PAx * ds_0.x_4_0 - PCx * ds_1.x_4_0; 
  x_11_0 += TwoZetaInv * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_12_0 = PAx * ds_0.x_8_0 - PCx * ds_1.x_8_0; 
  x_13_0 = PAx * ds_0.x_6_0 - PCx * ds_1.x_6_0; 
  x_13_0 += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_14_0 = PAx * ds_0.x_9_0 - PCx * ds_1.x_9_0; 
  x_15_0 = PAy * ds_0.x_5_0 - PCy * ds_1.x_5_0; 
  x_15_0 += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_16_0 = PAy * ds_0.x_9_0 - PCy * ds_1.x_9_0; 
  x_17_0 = PAx * ds_0.x_7_0 - PCx * ds_1.x_7_0; 
  x_17_0 += TwoZetaInv * 2.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_18_0 = PAy * ds_0.x_8_0 - PCy * ds_1.x_8_0; 
  x_18_0 += TwoZetaInv * 2.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_19_0 = PAz * ds_0.x_9_0 - PCz * ds_1.x_9_0; 
  x_19_0 += TwoZetaInv * 2.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
#else 
  QUICKDouble val; 
  val = PAx * ds_0.x_5_0 - PCx * ds_1.x_5_0; 
  LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAx * ds_0.x_4_0 - PCx * ds_1.x_4_0; 
  val += TwoZetaInv * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAx * ds_0.x_8_0 - PCx * ds_1.x_8_0; 
  LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAx * ds_0.x_6_0 - PCx * ds_1.x_6_0; 
  val += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAx * ds_0.x_9_0 - PCx * ds_1.x_9_0; 
  LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAy * ds_0.x_5_0 - PCy * ds_1.x_5_0; 
  val += TwoZetaInv * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAy * ds_0.x_9_0 - PCy * ds_1.x_9_0; 
  LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAx * ds_0.x_7_0 - PCx * ds_1.x_7_0; 
  val += TwoZetaInv * 2.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAy * ds_0.x_8_0 - PCy * ds_1.x_8_0; 
  val += TwoZetaInv * 2.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) = val; 
  val = PAz * ds_0.x_9_0 - PCz * ds_1.x_9_0; 
  val += TwoZetaInv * 2.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) = val; 
#endif 

 } 

/* FS auxilary integral, m=1 */ 
__device__ __inline__ FSint_1::FSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 

#ifdef REG_FS 
  x_10_0 = PAx * ds_1.x_5_0 - PCx * ds_2.x_5_0; 
  x_11_0 = PAx * ds_1.x_4_0 - PCx * ds_2.x_4_0; 
  x_11_0 += TwoZetaInv * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_12_0 = PAx * ds_1.x_8_0 - PCx * ds_2.x_8_0; 
  x_13_0 = PAx * ds_1.x_6_0 - PCx * ds_2.x_6_0; 
  x_13_0 += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_14_0 = PAx * ds_1.x_9_0 - PCx * ds_2.x_9_0; 
  x_15_0 = PAy * ds_1.x_5_0 - PCy * ds_2.x_5_0; 
  x_15_0 += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_16_0 = PAy * ds_1.x_9_0 - PCy * ds_2.x_9_0; 
  x_17_0 = PAx * ds_1.x_7_0 - PCx * ds_2.x_7_0; 
  x_17_0 += TwoZetaInv * 2.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_18_0 = PAy * ds_1.x_8_0 - PCy * ds_2.x_8_0; 
  x_18_0 += TwoZetaInv * 2.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_19_0 = PAz * ds_1.x_9_0 - PCz * ds_2.x_9_0; 
  x_19_0 += TwoZetaInv * 2.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
#else 
  QUICKDouble val; 
  val = PAx * ds_1.x_5_0 - PCx * ds_2.x_5_0; 
  LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAx * ds_1.x_4_0 - PCx * ds_2.x_4_0; 
  val += TwoZetaInv * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAx * ds_1.x_8_0 - PCx * ds_2.x_8_0; 
  LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAx * ds_1.x_6_0 - PCx * ds_2.x_6_0; 
  val += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAx * ds_1.x_9_0 - PCx * ds_2.x_9_0; 
  LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAy * ds_1.x_5_0 - PCy * ds_2.x_5_0; 
  val += TwoZetaInv * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAy * ds_1.x_9_0 - PCy * ds_2.x_9_0; 
  LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAx * ds_1.x_7_0 - PCx * ds_2.x_7_0; 
  val += TwoZetaInv * 2.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAy * ds_1.x_8_0 - PCy * ds_2.x_8_0; 
  val += TwoZetaInv * 2.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) = val; 
  val = PAz * ds_1.x_9_0 - PCz * ds_2.x_9_0; 
  val += TwoZetaInv * 2.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) = val; 
#endif 

 } 

/* FS auxilary integral, m=2 */ 
__device__ __inline__ FSint_2::FSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 

#ifdef REG_FS 
  x_10_0 = PAx * ds_2.x_5_0 - PCx * ds_3.x_5_0; 
  x_11_0 = PAx * ds_2.x_4_0 - PCx * ds_3.x_4_0; 
  x_11_0 += TwoZetaInv * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_12_0 = PAx * ds_2.x_8_0 - PCx * ds_3.x_8_0; 
  x_13_0 = PAx * ds_2.x_6_0 - PCx * ds_3.x_6_0; 
  x_13_0 += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_14_0 = PAx * ds_2.x_9_0 - PCx * ds_3.x_9_0; 
  x_15_0 = PAy * ds_2.x_5_0 - PCy * ds_3.x_5_0; 
  x_15_0 += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_16_0 = PAy * ds_2.x_9_0 - PCy * ds_3.x_9_0; 
  x_17_0 = PAx * ds_2.x_7_0 - PCx * ds_3.x_7_0; 
  x_17_0 += TwoZetaInv * 2.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_18_0 = PAy * ds_2.x_8_0 - PCy * ds_3.x_8_0; 
  x_18_0 += TwoZetaInv * 2.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_19_0 = PAz * ds_2.x_9_0 - PCz * ds_3.x_9_0; 
  x_19_0 += TwoZetaInv * 2.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
#else 
  QUICKDouble val; 
  val = PAx * ds_2.x_5_0 - PCx * ds_3.x_5_0; 
  LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAx * ds_2.x_4_0 - PCx * ds_3.x_4_0; 
  val += TwoZetaInv * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAx * ds_2.x_8_0 - PCx * ds_3.x_8_0; 
  LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAx * ds_2.x_6_0 - PCx * ds_3.x_6_0; 
  val += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAx * ds_2.x_9_0 - PCx * ds_3.x_9_0; 
  LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAy * ds_2.x_5_0 - PCy * ds_3.x_5_0; 
  val += TwoZetaInv * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAy * ds_2.x_9_0 - PCy * ds_3.x_9_0; 
  LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAx * ds_2.x_7_0 - PCx * ds_3.x_7_0; 
  val += TwoZetaInv * 2.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAy * ds_2.x_8_0 - PCy * ds_3.x_8_0; 
  val += TwoZetaInv * 2.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) = val; 
  val = PAz * ds_2.x_9_0 - PCz * ds_3.x_9_0; 
  val += TwoZetaInv * 2.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) = val; 
#endif 

 } 

/* FS auxilary integral, m=3 */ 
__device__ __inline__ FSint_3::FSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=3 
  PSint_4 ps_4(PAx, PAy, PAz, PCx, PCy, PCz, store, YVerticalTemp); // construct [p|s] for m=4 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  DSint_4 ds_4(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=4 

#ifdef REG_FS 
  x_10_0 = PAx * ds_3.x_5_0 - PCx * ds_4.x_5_0; 
  x_11_0 = PAx * ds_3.x_4_0 - PCx * ds_4.x_4_0; 
  x_11_0 += TwoZetaInv * 1.000000 * (ps_3.x_2_0 - ps_4.x_2_0); 
  x_12_0 = PAx * ds_3.x_8_0 - PCx * ds_4.x_8_0; 
  x_13_0 = PAx * ds_3.x_6_0 - PCx * ds_4.x_6_0; 
  x_13_0 += TwoZetaInv * 1.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  x_14_0 = PAx * ds_3.x_9_0 - PCx * ds_4.x_9_0; 
  x_15_0 = PAy * ds_3.x_5_0 - PCy * ds_4.x_5_0; 
  x_15_0 += TwoZetaInv * 1.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  x_16_0 = PAy * ds_3.x_9_0 - PCy * ds_4.x_9_0; 
  x_17_0 = PAx * ds_3.x_7_0 - PCx * ds_4.x_7_0; 
  x_17_0 += TwoZetaInv * 2.000000 * (ps_3.x_1_0 - ps_4.x_1_0); 
  x_18_0 = PAy * ds_3.x_8_0 - PCy * ds_4.x_8_0; 
  x_18_0 += TwoZetaInv * 2.000000 * (ps_3.x_2_0 - ps_4.x_2_0); 
  x_19_0 = PAz * ds_3.x_9_0 - PCz * ds_4.x_9_0; 
  x_19_0 += TwoZetaInv * 2.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
#else 
  QUICKDouble val; 
  val = PAx * ds_3.x_5_0 - PCx * ds_4.x_5_0; 
  LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAx * ds_3.x_4_0 - PCx * ds_4.x_4_0; 
  val += TwoZetaInv * 1.000000 * (ps_3.x_2_0 - ps_4.x_2_0); 
  LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAx * ds_3.x_8_0 - PCx * ds_4.x_8_0; 
  LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAx * ds_3.x_6_0 - PCx * ds_4.x_6_0; 
  val += TwoZetaInv * 1.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAx * ds_3.x_9_0 - PCx * ds_4.x_9_0; 
  LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAy * ds_3.x_5_0 - PCy * ds_4.x_5_0; 
  val += TwoZetaInv * 1.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAy * ds_3.x_9_0 - PCy * ds_4.x_9_0; 
  LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAx * ds_3.x_7_0 - PCx * ds_4.x_7_0; 
  val += TwoZetaInv * 2.000000 * (ps_3.x_1_0 - ps_4.x_1_0); 
  LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAy * ds_3.x_8_0 - PCy * ds_4.x_8_0; 
  val += TwoZetaInv * 2.000000 * (ps_3.x_2_0 - ps_4.x_2_0); 
  LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3) = val; 
  val = PAz * ds_3.x_9_0 - PCz * ds_4.x_9_0; 
  val += TwoZetaInv * 2.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3) = val; 
#endif 

 } 

/* SF true integral, m=0 */ 
__device__ __inline__ SFint_0::SFint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=1 
  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 

#ifdef REG_SF 
  x_0_10 = PBx * sd_0.x_0_5 - PCx * sd_1.x_0_5; 
  x_0_11 = PBx * sd_0.x_0_4 - PCx * sd_1.x_0_4; 
  x_0_11 += TwoZetaInv * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_0_12 = PBx * sd_0.x_0_8 - PCx * sd_1.x_0_8; 
  x_0_13 = PBx * sd_0.x_0_6 - PCx * sd_1.x_0_6; 
  x_0_13 += TwoZetaInv * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_0_14 = PBx * sd_0.x_0_9 - PCx * sd_1.x_0_9; 
  x_0_15 = PBy * sd_0.x_0_5 - PCy * sd_1.x_0_5; 
  x_0_15 += TwoZetaInv * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_0_16 = PBy * sd_0.x_0_9 - PCy * sd_1.x_0_9; 
  x_0_17 = PBx * sd_0.x_0_7 - PCx * sd_1.x_0_7; 
  x_0_17 += TwoZetaInv * 2.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_0_18 = PBy * sd_0.x_0_8 - PCy * sd_1.x_0_8; 
  x_0_18 += TwoZetaInv * 2.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_0_19 = PBz * sd_0.x_0_9 - PCz * sd_1.x_0_9; 
  x_0_19 += TwoZetaInv * 2.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
#else 
  QUICKDouble val; 
  val = PBx * sd_0.x_0_5 - PCx * sd_1.x_0_5; 
  LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * sd_0.x_0_4 - PCx * sd_1.x_0_4; 
  val += TwoZetaInv * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * sd_0.x_0_8 - PCx * sd_1.x_0_8; 
  LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * sd_0.x_0_6 - PCx * sd_1.x_0_6; 
  val += TwoZetaInv * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * sd_0.x_0_9 - PCx * sd_1.x_0_9; 
  LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * sd_0.x_0_5 - PCy * sd_1.x_0_5; 
  val += TwoZetaInv * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * sd_0.x_0_9 - PCy * sd_1.x_0_9; 
  LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) = val; 
  val = PBx * sd_0.x_0_7 - PCx * sd_1.x_0_7; 
  val += TwoZetaInv * 2.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) = val; 
  val = PBy * sd_0.x_0_8 - PCy * sd_1.x_0_8; 
  val += TwoZetaInv * 2.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) = val; 
  val = PBz * sd_0.x_0_9 - PCz * sd_1.x_0_9; 
  val += TwoZetaInv * 2.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) = val; 
#endif 

 } 

/* SF auxilary integral, m=1 */ 
__device__ __inline__ SFint_1::SFint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=1 
  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=2 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 

#ifdef REG_SF 
  x_0_10 = PBx * sd_1.x_0_5 - PCx * sd_2.x_0_5; 
  x_0_11 = PBx * sd_1.x_0_4 - PCx * sd_2.x_0_4; 
  x_0_11 += TwoZetaInv * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_0_12 = PBx * sd_1.x_0_8 - PCx * sd_2.x_0_8; 
  x_0_13 = PBx * sd_1.x_0_6 - PCx * sd_2.x_0_6; 
  x_0_13 += TwoZetaInv * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_0_14 = PBx * sd_1.x_0_9 - PCx * sd_2.x_0_9; 
  x_0_15 = PBy * sd_1.x_0_5 - PCy * sd_2.x_0_5; 
  x_0_15 += TwoZetaInv * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_0_16 = PBy * sd_1.x_0_9 - PCy * sd_2.x_0_9; 
  x_0_17 = PBx * sd_1.x_0_7 - PCx * sd_2.x_0_7; 
  x_0_17 += TwoZetaInv * 2.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_0_18 = PBy * sd_1.x_0_8 - PCy * sd_2.x_0_8; 
  x_0_18 += TwoZetaInv * 2.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_0_19 = PBz * sd_1.x_0_9 - PCz * sd_2.x_0_9; 
  x_0_19 += TwoZetaInv * 2.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
#else 
  QUICKDouble val; 
  val = PBx * sd_1.x_0_5 - PCx * sd_2.x_0_5; 
  LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * sd_1.x_0_4 - PCx * sd_2.x_0_4; 
  val += TwoZetaInv * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * sd_1.x_0_8 - PCx * sd_2.x_0_8; 
  LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * sd_1.x_0_6 - PCx * sd_2.x_0_6; 
  val += TwoZetaInv * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * sd_1.x_0_9 - PCx * sd_2.x_0_9; 
  LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * sd_1.x_0_5 - PCy * sd_2.x_0_5; 
  val += TwoZetaInv * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * sd_1.x_0_9 - PCy * sd_2.x_0_9; 
  LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) = val; 
  val = PBx * sd_1.x_0_7 - PCx * sd_2.x_0_7; 
  val += TwoZetaInv * 2.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) = val; 
  val = PBy * sd_1.x_0_8 - PCy * sd_2.x_0_8; 
  val += TwoZetaInv * 2.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) = val; 
  val = PBz * sd_1.x_0_9 - PCz * sd_2.x_0_9; 
  val += TwoZetaInv * 2.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) = val; 
#endif 

 } 

/* SF auxilary integral, m=2 */ 
__device__ __inline__ SFint_2::SFint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=2 
  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=3 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 

#ifdef REG_SF 
  x_0_10 = PBx * sd_2.x_0_5 - PCx * sd_3.x_0_5; 
  x_0_11 = PBx * sd_2.x_0_4 - PCx * sd_3.x_0_4; 
  x_0_11 += TwoZetaInv * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_0_12 = PBx * sd_2.x_0_8 - PCx * sd_3.x_0_8; 
  x_0_13 = PBx * sd_2.x_0_6 - PCx * sd_3.x_0_6; 
  x_0_13 += TwoZetaInv * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_0_14 = PBx * sd_2.x_0_9 - PCx * sd_3.x_0_9; 
  x_0_15 = PBy * sd_2.x_0_5 - PCy * sd_3.x_0_5; 
  x_0_15 += TwoZetaInv * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_0_16 = PBy * sd_2.x_0_9 - PCy * sd_3.x_0_9; 
  x_0_17 = PBx * sd_2.x_0_7 - PCx * sd_3.x_0_7; 
  x_0_17 += TwoZetaInv * 2.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_0_18 = PBy * sd_2.x_0_8 - PCy * sd_3.x_0_8; 
  x_0_18 += TwoZetaInv * 2.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_0_19 = PBz * sd_2.x_0_9 - PCz * sd_3.x_0_9; 
  x_0_19 += TwoZetaInv * 2.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
#else 
  QUICKDouble val; 
  val = PBx * sd_2.x_0_5 - PCx * sd_3.x_0_5; 
  LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2) = val; 
  val = PBx * sd_2.x_0_4 - PCx * sd_3.x_0_4; 
  val += TwoZetaInv * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2) = val; 
  val = PBx * sd_2.x_0_8 - PCx * sd_3.x_0_8; 
  LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2) = val; 
  val = PBx * sd_2.x_0_6 - PCx * sd_3.x_0_6; 
  val += TwoZetaInv * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2) = val; 
  val = PBx * sd_2.x_0_9 - PCx * sd_3.x_0_9; 
  LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2) = val; 
  val = PBy * sd_2.x_0_5 - PCy * sd_3.x_0_5; 
  val += TwoZetaInv * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2) = val; 
  val = PBy * sd_2.x_0_9 - PCy * sd_3.x_0_9; 
  LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2) = val; 
  val = PBx * sd_2.x_0_7 - PCx * sd_3.x_0_7; 
  val += TwoZetaInv * 2.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2) = val; 
  val = PBy * sd_2.x_0_8 - PCy * sd_3.x_0_8; 
  val += TwoZetaInv * 2.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2) = val; 
  val = PBz * sd_2.x_0_9 - PCz * sd_3.x_0_9; 
  val += TwoZetaInv * 2.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2) = val; 
#endif 

 } 

/* SF auxilary integral, m=3 */ 
__device__ __inline__ SFint_3::SFint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=3 
  SPint_4 sp_4(PBx, PBy, PBz, PCx, PCy, PCz, store, YVerticalTemp); // construct [s|p] for m=4 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SDint_4 sd_4(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=4 

#ifdef REG_SF 
  x_0_10 = PBx * sd_3.x_0_5 - PCx * sd_4.x_0_5; 
  x_0_11 = PBx * sd_3.x_0_4 - PCx * sd_4.x_0_4; 
  x_0_11 += TwoZetaInv * 1.000000 * (sp_3.x_0_2 - sp_4.x_0_2); 
  x_0_12 = PBx * sd_3.x_0_8 - PCx * sd_4.x_0_8; 
  x_0_13 = PBx * sd_3.x_0_6 - PCx * sd_4.x_0_6; 
  x_0_13 += TwoZetaInv * 1.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  x_0_14 = PBx * sd_3.x_0_9 - PCx * sd_4.x_0_9; 
  x_0_15 = PBy * sd_3.x_0_5 - PCy * sd_4.x_0_5; 
  x_0_15 += TwoZetaInv * 1.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  x_0_16 = PBy * sd_3.x_0_9 - PCy * sd_4.x_0_9; 
  x_0_17 = PBx * sd_3.x_0_7 - PCx * sd_4.x_0_7; 
  x_0_17 += TwoZetaInv * 2.000000 * (sp_3.x_0_1 - sp_4.x_0_1); 
  x_0_18 = PBy * sd_3.x_0_8 - PCy * sd_4.x_0_8; 
  x_0_18 += TwoZetaInv * 2.000000 * (sp_3.x_0_2 - sp_4.x_0_2); 
  x_0_19 = PBz * sd_3.x_0_9 - PCz * sd_4.x_0_9; 
  x_0_19 += TwoZetaInv * 2.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
#else 
  QUICKDouble val; 
  val = PBx * sd_3.x_0_5 - PCx * sd_4.x_0_5; 
  LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3) = val; 
  val = PBx * sd_3.x_0_4 - PCx * sd_4.x_0_4; 
  val += TwoZetaInv * 1.000000 * (sp_3.x_0_2 - sp_4.x_0_2); 
  LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3) = val; 
  val = PBx * sd_3.x_0_8 - PCx * sd_4.x_0_8; 
  LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3) = val; 
  val = PBx * sd_3.x_0_6 - PCx * sd_4.x_0_6; 
  val += TwoZetaInv * 1.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3) = val; 
  val = PBx * sd_3.x_0_9 - PCx * sd_4.x_0_9; 
  LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3) = val; 
  val = PBy * sd_3.x_0_5 - PCy * sd_4.x_0_5; 
  val += TwoZetaInv * 1.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3) = val; 
  val = PBy * sd_3.x_0_9 - PCy * sd_4.x_0_9; 
  LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3) = val; 
  val = PBx * sd_3.x_0_7 - PCx * sd_4.x_0_7; 
  val += TwoZetaInv * 2.000000 * (sp_3.x_0_1 - sp_4.x_0_1); 
  LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3) = val; 
  val = PBy * sd_3.x_0_8 - PCy * sd_4.x_0_8; 
  val += TwoZetaInv * 2.000000 * (sp_3.x_0_2 - sp_4.x_0_2); 
  LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3) = val; 
  val = PBz * sd_3.x_0_9 - PCz * sd_4.x_0_9; 
  val += TwoZetaInv * 2.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3) = val; 
#endif 

 } 

/* FP true integral, m=0 */ 
__device__ __inline__ FPint_0::FPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FP 
#ifdef REG_FS 
  x_10_1 = PBx * fs_0.x_10_0 - PCx * fs_1.x_10_0; 
#else 
  x_10_1 = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_10_1 += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
#ifdef REG_FS 
  x_10_2 = PBy * fs_0.x_10_0 - PCy * fs_1.x_10_0; 
#else 
  x_10_2 = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_10_2 += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
#ifdef REG_FS 
  x_10_3 = PBz * fs_0.x_10_0 - PCz * fs_1.x_10_0; 
#else 
  x_10_3 = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_10_3 += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
#ifdef REG_FS 
  x_11_1 = PBx * fs_0.x_11_0 - PCx * fs_1.x_11_0; 
#else 
  x_11_1 = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_11_1 += TwoZetaInv * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
#ifdef REG_FS 
  x_11_2 = PBy * fs_0.x_11_0 - PCy * fs_1.x_11_0; 
#else 
  x_11_2 = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_11_2 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
#ifdef REG_FS 
  x_11_3 = PBz * fs_0.x_11_0 - PCz * fs_1.x_11_0; 
#else 
  x_11_3 = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_12_1 = PBx * fs_0.x_12_0 - PCx * fs_1.x_12_0; 
#else 
  x_12_1 = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_12_1 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
#ifdef REG_FS 
  x_12_2 = PBy * fs_0.x_12_0 - PCy * fs_1.x_12_0; 
#else 
  x_12_2 = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_12_2 += TwoZetaInv * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
#ifdef REG_FS 
  x_12_3 = PBz * fs_0.x_12_0 - PCz * fs_1.x_12_0; 
#else 
  x_12_3 = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_13_1 = PBx * fs_0.x_13_0 - PCx * fs_1.x_13_0; 
#else 
  x_13_1 = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_13_1 += TwoZetaInv * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
#ifdef REG_FS 
  x_13_2 = PBy * fs_0.x_13_0 - PCy * fs_1.x_13_0; 
#else 
  x_13_2 = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_13_3 = PBz * fs_0.x_13_0 - PCz * fs_1.x_13_0; 
#else 
  x_13_3 = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_13_3 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
#ifdef REG_FS 
  x_14_1 = PBx * fs_0.x_14_0 - PCx * fs_1.x_14_0; 
#else 
  x_14_1 = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_14_1 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
#ifdef REG_FS 
  x_14_2 = PBy * fs_0.x_14_0 - PCy * fs_1.x_14_0; 
#else 
  x_14_2 = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_14_3 = PBz * fs_0.x_14_0 - PCz * fs_1.x_14_0; 
#else 
  x_14_3 = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_14_3 += TwoZetaInv * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
#ifdef REG_FS 
  x_15_1 = PBx * fs_0.x_15_0 - PCx * fs_1.x_15_0; 
#else 
  x_15_1 = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_15_2 = PBy * fs_0.x_15_0 - PCy * fs_1.x_15_0; 
#else 
  x_15_2 = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_15_2 += TwoZetaInv * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
#ifdef REG_FS 
  x_15_3 = PBz * fs_0.x_15_0 - PCz * fs_1.x_15_0; 
#else 
  x_15_3 = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_15_3 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
#ifdef REG_FS 
  x_16_1 = PBx * fs_0.x_16_0 - PCx * fs_1.x_16_0; 
#else 
  x_16_1 = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_16_2 = PBy * fs_0.x_16_0 - PCy * fs_1.x_16_0; 
#else 
  x_16_2 = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_16_2 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
#ifdef REG_FS 
  x_16_3 = PBz * fs_0.x_16_0 - PCz * fs_1.x_16_0; 
#else 
  x_16_3 = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_16_3 += TwoZetaInv * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
#ifdef REG_FS 
  x_17_1 = PBx * fs_0.x_17_0 - PCx * fs_1.x_17_0; 
#else 
  x_17_1 = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_17_1 += TwoZetaInv * 3.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
#ifdef REG_FS 
  x_17_2 = PBy * fs_0.x_17_0 - PCy * fs_1.x_17_0; 
#else 
  x_17_2 = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_17_3 = PBz * fs_0.x_17_0 - PCz * fs_1.x_17_0; 
#else 
  x_17_3 = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_18_1 = PBx * fs_0.x_18_0 - PCx * fs_1.x_18_0; 
#else 
  x_18_1 = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_18_2 = PBy * fs_0.x_18_0 - PCy * fs_1.x_18_0; 
#else 
  x_18_2 = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_18_2 += TwoZetaInv * 3.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
#ifdef REG_FS 
  x_18_3 = PBz * fs_0.x_18_0 - PCz * fs_1.x_18_0; 
#else 
  x_18_3 = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_19_1 = PBx * fs_0.x_19_0 - PCx * fs_1.x_19_0; 
#else 
  x_19_1 = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_19_2 = PBy * fs_0.x_19_0 - PCy * fs_1.x_19_0; 
#else 
  x_19_2 = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_19_3 = PBz * fs_0.x_19_0 - PCz * fs_1.x_19_0; 
#else 
  x_19_3 = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_19_3 += TwoZetaInv * 3.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
#else 
  QUICKDouble val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_10_0 - PCx * fs_1.x_10_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_10_0 - PCy * fs_1.x_10_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_10_0 - PCz * fs_1.x_10_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_11_0 - PCx * fs_1.x_11_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_11_0 - PCy * fs_1.x_11_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_11_0 - PCz * fs_1.x_11_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_12_0 - PCx * fs_1.x_12_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_12_0 - PCy * fs_1.x_12_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_12_0 - PCz * fs_1.x_12_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_13_0 - PCx * fs_1.x_13_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_13_0 - PCy * fs_1.x_13_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_13_0 - PCz * fs_1.x_13_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_14_0 - PCx * fs_1.x_14_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_14_0 - PCy * fs_1.x_14_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_14_0 - PCz * fs_1.x_14_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_15_0 - PCx * fs_1.x_15_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_15_0 - PCy * fs_1.x_15_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_15_0 - PCz * fs_1.x_15_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_16_0 - PCx * fs_1.x_16_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_16_0 - PCy * fs_1.x_16_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_16_0 - PCz * fs_1.x_16_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_17_0 - PCx * fs_1.x_17_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_17_0 - PCy * fs_1.x_17_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_17_0 - PCz * fs_1.x_17_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_18_0 - PCx * fs_1.x_18_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_18_0 - PCy * fs_1.x_18_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_18_0 - PCz * fs_1.x_18_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBx * fs_0.x_19_0 - PCx * fs_1.x_19_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBy * fs_0.x_19_0 - PCy * fs_1.x_19_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FS 
  val = PBz * fs_0.x_19_0 - PCz * fs_1.x_19_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) = val; 
#endif 

 } 

/* FP integral partial class - Part 1, m=0 */ 
__device__ __inline__ FPint_0_1::FPint_0_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_10_1 = PBx * fs_0.x_10_0 - PCx * fs_1.x_10_0; 
#else 
  x_10_1 = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_10_1 += TwoZetaInv * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
#ifdef REG_FS 
  x_10_2 = PBy * fs_0.x_10_0 - PCy * fs_1.x_10_0; 
#else 
  x_10_2 = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_10_2 += TwoZetaInv * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
#ifdef REG_FS 
  x_10_3 = PBz * fs_0.x_10_0 - PCz * fs_1.x_10_0; 
#else 
  x_10_3 = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_10_3 += TwoZetaInv * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 

 } 

/* FP integral partial class - Part 2, m=0 */ 
__device__ __inline__ FPint_0_2::FPint_0_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_11_1 = PBx * fs_0.x_11_0 - PCx * fs_1.x_11_0; 
#else 
  x_11_1 = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_11_1 += TwoZetaInv * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
#ifdef REG_FS 
  x_11_2 = PBy * fs_0.x_11_0 - PCy * fs_1.x_11_0; 
#else 
  x_11_2 = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_11_2 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
#ifdef REG_FS 
  x_11_3 = PBz * fs_0.x_11_0 - PCz * fs_1.x_11_0; 
#else 
  x_11_3 = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* FP integral partial class - Part 3, m=0 */ 
__device__ __inline__ FPint_0_3::FPint_0_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_12_1 = PBx * fs_0.x_12_0 - PCx * fs_1.x_12_0; 
#else 
  x_12_1 = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_12_1 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
#ifdef REG_FS 
  x_12_2 = PBy * fs_0.x_12_0 - PCy * fs_1.x_12_0; 
#else 
  x_12_2 = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_12_2 += TwoZetaInv * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
#ifdef REG_FS 
  x_12_3 = PBz * fs_0.x_12_0 - PCz * fs_1.x_12_0; 
#else 
  x_12_3 = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* FP integral partial class - Part 4, m=0 */ 
__device__ __inline__ FPint_0_4::FPint_0_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_13_1 = PBx * fs_0.x_13_0 - PCx * fs_1.x_13_0; 
#else 
  x_13_1 = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_13_1 += TwoZetaInv * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
#ifdef REG_FS 
  x_13_2 = PBy * fs_0.x_13_0 - PCy * fs_1.x_13_0; 
#else 
  x_13_2 = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_13_3 = PBz * fs_0.x_13_0 - PCz * fs_1.x_13_0; 
#else 
  x_13_3 = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_13_3 += TwoZetaInv * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 

 } 

/* FP integral partial class - Part 5, m=0 */ 
__device__ __inline__ FPint_0_5::FPint_0_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_14_1 = PBx * fs_0.x_14_0 - PCx * fs_1.x_14_0; 
#else 
  x_14_1 = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_14_1 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
#ifdef REG_FS 
  x_14_2 = PBy * fs_0.x_14_0 - PCy * fs_1.x_14_0; 
#else 
  x_14_2 = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_14_3 = PBz * fs_0.x_14_0 - PCz * fs_1.x_14_0; 
#else 
  x_14_3 = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_14_3 += TwoZetaInv * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 

 } 

/* FP integral partial class - Part 6, m=0 */ 
__device__ __inline__ FPint_0_6::FPint_0_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_15_1 = PBx * fs_0.x_15_0 - PCx * fs_1.x_15_0; 
#else 
  x_15_1 = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_15_2 = PBy * fs_0.x_15_0 - PCy * fs_1.x_15_0; 
#else 
  x_15_2 = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_15_2 += TwoZetaInv * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
#ifdef REG_FS 
  x_15_3 = PBz * fs_0.x_15_0 - PCz * fs_1.x_15_0; 
#else 
  x_15_3 = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_15_3 += TwoZetaInv * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 

 } 

/* FP integral partial class - Part 7, m=0 */ 
__device__ __inline__ FPint_0_7::FPint_0_7(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_16_1 = PBx * fs_0.x_16_0 - PCx * fs_1.x_16_0; 
#else 
  x_16_1 = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_16_2 = PBy * fs_0.x_16_0 - PCy * fs_1.x_16_0; 
#else 
  x_16_2 = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_16_2 += TwoZetaInv * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
#ifdef REG_FS 
  x_16_3 = PBz * fs_0.x_16_0 - PCz * fs_1.x_16_0; 
#else 
  x_16_3 = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_16_3 += TwoZetaInv * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 

 } 

/* FP integral partial class - Part 8, m=0 */ 
__device__ __inline__ FPint_0_8::FPint_0_8(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_17_1 = PBx * fs_0.x_17_0 - PCx * fs_1.x_17_0; 
#else 
  x_17_1 = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_17_1 += TwoZetaInv * 3.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
#ifdef REG_FS 
  x_17_2 = PBy * fs_0.x_17_0 - PCy * fs_1.x_17_0; 
#else 
  x_17_2 = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_17_3 = PBz * fs_0.x_17_0 - PCz * fs_1.x_17_0; 
#else 
  x_17_3 = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* FP integral partial class - Part 9, m=0 */ 
__device__ __inline__ FPint_0_9::FPint_0_9(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_18_1 = PBx * fs_0.x_18_0 - PCx * fs_1.x_18_0; 
#else 
  x_18_1 = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_18_2 = PBy * fs_0.x_18_0 - PCy * fs_1.x_18_0; 
#else 
  x_18_2 = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_18_2 += TwoZetaInv * 3.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
#ifdef REG_FS 
  x_18_3 = PBz * fs_0.x_18_0 - PCz * fs_1.x_18_0; 
#else 
  x_18_3 = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* FP integral partial class - Part 10, m=0 */ 
__device__ __inline__ FPint_0_10::FPint_0_10(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 

#ifdef REG_FS 
  x_19_1 = PBx * fs_0.x_19_0 - PCx * fs_1.x_19_0; 
#else 
  x_19_1 = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_19_2 = PBy * fs_0.x_19_0 - PCy * fs_1.x_19_0; 
#else 
  x_19_2 = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  x_19_3 = PBz * fs_0.x_19_0 - PCz * fs_1.x_19_0; 
#else 
  x_19_3 = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1); 
#endif 
  x_19_3 += TwoZetaInv * 3.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 

 } 

/* FP auxilary integral, m=1 */ 
__device__ __inline__ FPint_1::FPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FP 
#ifdef REG_FS 
  x_10_1 = PBx * fs_1.x_10_0 - PCx * fs_2.x_10_0; 
#else 
  x_10_1 = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_10_1 += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
#ifdef REG_FS 
  x_10_2 = PBy * fs_1.x_10_0 - PCy * fs_2.x_10_0; 
#else 
  x_10_2 = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_10_2 += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
#ifdef REG_FS 
  x_10_3 = PBz * fs_1.x_10_0 - PCz * fs_2.x_10_0; 
#else 
  x_10_3 = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_10_3 += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
#ifdef REG_FS 
  x_11_1 = PBx * fs_1.x_11_0 - PCx * fs_2.x_11_0; 
#else 
  x_11_1 = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_11_1 += TwoZetaInv * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
#ifdef REG_FS 
  x_11_2 = PBy * fs_1.x_11_0 - PCy * fs_2.x_11_0; 
#else 
  x_11_2 = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_11_2 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
#ifdef REG_FS 
  x_11_3 = PBz * fs_1.x_11_0 - PCz * fs_2.x_11_0; 
#else 
  x_11_3 = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_12_1 = PBx * fs_1.x_12_0 - PCx * fs_2.x_12_0; 
#else 
  x_12_1 = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_12_1 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
#ifdef REG_FS 
  x_12_2 = PBy * fs_1.x_12_0 - PCy * fs_2.x_12_0; 
#else 
  x_12_2 = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_12_2 += TwoZetaInv * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
#ifdef REG_FS 
  x_12_3 = PBz * fs_1.x_12_0 - PCz * fs_2.x_12_0; 
#else 
  x_12_3 = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_13_1 = PBx * fs_1.x_13_0 - PCx * fs_2.x_13_0; 
#else 
  x_13_1 = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_13_1 += TwoZetaInv * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
#ifdef REG_FS 
  x_13_2 = PBy * fs_1.x_13_0 - PCy * fs_2.x_13_0; 
#else 
  x_13_2 = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_13_3 = PBz * fs_1.x_13_0 - PCz * fs_2.x_13_0; 
#else 
  x_13_3 = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_13_3 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
#ifdef REG_FS 
  x_14_1 = PBx * fs_1.x_14_0 - PCx * fs_2.x_14_0; 
#else 
  x_14_1 = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_14_1 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
#ifdef REG_FS 
  x_14_2 = PBy * fs_1.x_14_0 - PCy * fs_2.x_14_0; 
#else 
  x_14_2 = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_14_3 = PBz * fs_1.x_14_0 - PCz * fs_2.x_14_0; 
#else 
  x_14_3 = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_14_3 += TwoZetaInv * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
#ifdef REG_FS 
  x_15_1 = PBx * fs_1.x_15_0 - PCx * fs_2.x_15_0; 
#else 
  x_15_1 = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_15_2 = PBy * fs_1.x_15_0 - PCy * fs_2.x_15_0; 
#else 
  x_15_2 = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_15_2 += TwoZetaInv * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
#ifdef REG_FS 
  x_15_3 = PBz * fs_1.x_15_0 - PCz * fs_2.x_15_0; 
#else 
  x_15_3 = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_15_3 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
#ifdef REG_FS 
  x_16_1 = PBx * fs_1.x_16_0 - PCx * fs_2.x_16_0; 
#else 
  x_16_1 = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_16_2 = PBy * fs_1.x_16_0 - PCy * fs_2.x_16_0; 
#else 
  x_16_2 = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_16_2 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
#ifdef REG_FS 
  x_16_3 = PBz * fs_1.x_16_0 - PCz * fs_2.x_16_0; 
#else 
  x_16_3 = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_16_3 += TwoZetaInv * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
#ifdef REG_FS 
  x_17_1 = PBx * fs_1.x_17_0 - PCx * fs_2.x_17_0; 
#else 
  x_17_1 = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_17_1 += TwoZetaInv * 3.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
#ifdef REG_FS 
  x_17_2 = PBy * fs_1.x_17_0 - PCy * fs_2.x_17_0; 
#else 
  x_17_2 = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_17_3 = PBz * fs_1.x_17_0 - PCz * fs_2.x_17_0; 
#else 
  x_17_3 = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_18_1 = PBx * fs_1.x_18_0 - PCx * fs_2.x_18_0; 
#else 
  x_18_1 = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_18_2 = PBy * fs_1.x_18_0 - PCy * fs_2.x_18_0; 
#else 
  x_18_2 = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_18_2 += TwoZetaInv * 3.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
#ifdef REG_FS 
  x_18_3 = PBz * fs_1.x_18_0 - PCz * fs_2.x_18_0; 
#else 
  x_18_3 = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_19_1 = PBx * fs_1.x_19_0 - PCx * fs_2.x_19_0; 
#else 
  x_19_1 = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_19_2 = PBy * fs_1.x_19_0 - PCy * fs_2.x_19_0; 
#else 
  x_19_2 = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_19_3 = PBz * fs_1.x_19_0 - PCz * fs_2.x_19_0; 
#else 
  x_19_3 = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_19_3 += TwoZetaInv * 3.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
#else 
  QUICKDouble val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_10_0 - PCx * fs_2.x_10_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_10_0 - PCy * fs_2.x_10_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_10_0 - PCz * fs_2.x_10_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_11_0 - PCx * fs_2.x_11_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_11_0 - PCy * fs_2.x_11_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_11_0 - PCz * fs_2.x_11_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_12_0 - PCx * fs_2.x_12_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_12_0 - PCy * fs_2.x_12_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_12_0 - PCz * fs_2.x_12_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_13_0 - PCx * fs_2.x_13_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_13_0 - PCy * fs_2.x_13_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_13_0 - PCz * fs_2.x_13_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_14_0 - PCx * fs_2.x_14_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_14_0 - PCy * fs_2.x_14_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_14_0 - PCz * fs_2.x_14_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_15_0 - PCx * fs_2.x_15_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_15_0 - PCy * fs_2.x_15_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_15_0 - PCz * fs_2.x_15_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_16_0 - PCx * fs_2.x_16_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_16_0 - PCy * fs_2.x_16_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_16_0 - PCz * fs_2.x_16_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_17_0 - PCx * fs_2.x_17_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_17_0 - PCy * fs_2.x_17_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_17_0 - PCz * fs_2.x_17_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_18_0 - PCx * fs_2.x_18_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_18_0 - PCy * fs_2.x_18_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_18_0 - PCz * fs_2.x_18_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBx * fs_1.x_19_0 - PCx * fs_2.x_19_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBy * fs_1.x_19_0 - PCy * fs_2.x_19_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FS 
  val = PBz * fs_1.x_19_0 - PCz * fs_2.x_19_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) = val; 
#endif 

 } 

/* FP integral partial class - Part 1, m=1 */ 
__device__ __inline__ FPint_1_1::FPint_1_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_10_1 = PBx * fs_1.x_10_0 - PCx * fs_2.x_10_0; 
#else 
  x_10_1 = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_10_1 += TwoZetaInv * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
#ifdef REG_FS 
  x_10_2 = PBy * fs_1.x_10_0 - PCy * fs_2.x_10_0; 
#else 
  x_10_2 = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_10_2 += TwoZetaInv * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
#ifdef REG_FS 
  x_10_3 = PBz * fs_1.x_10_0 - PCz * fs_2.x_10_0; 
#else 
  x_10_3 = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_10_3 += TwoZetaInv * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 

 } 

/* FP integral partial class - Part 2, m=1 */ 
__device__ __inline__ FPint_1_2::FPint_1_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_11_1 = PBx * fs_1.x_11_0 - PCx * fs_2.x_11_0; 
#else 
  x_11_1 = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_11_1 += TwoZetaInv * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
#ifdef REG_FS 
  x_11_2 = PBy * fs_1.x_11_0 - PCy * fs_2.x_11_0; 
#else 
  x_11_2 = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_11_2 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
#ifdef REG_FS 
  x_11_3 = PBz * fs_1.x_11_0 - PCz * fs_2.x_11_0; 
#else 
  x_11_3 = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* FP integral partial class - Part 3, m=1 */ 
__device__ __inline__ FPint_1_3::FPint_1_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_12_1 = PBx * fs_1.x_12_0 - PCx * fs_2.x_12_0; 
#else 
  x_12_1 = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_12_1 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
#ifdef REG_FS 
  x_12_2 = PBy * fs_1.x_12_0 - PCy * fs_2.x_12_0; 
#else 
  x_12_2 = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_12_2 += TwoZetaInv * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
#ifdef REG_FS 
  x_12_3 = PBz * fs_1.x_12_0 - PCz * fs_2.x_12_0; 
#else 
  x_12_3 = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* FP integral partial class - Part 4, m=1 */ 
__device__ __inline__ FPint_1_4::FPint_1_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_13_1 = PBx * fs_1.x_13_0 - PCx * fs_2.x_13_0; 
#else 
  x_13_1 = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_13_1 += TwoZetaInv * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
#ifdef REG_FS 
  x_13_2 = PBy * fs_1.x_13_0 - PCy * fs_2.x_13_0; 
#else 
  x_13_2 = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_13_3 = PBz * fs_1.x_13_0 - PCz * fs_2.x_13_0; 
#else 
  x_13_3 = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_13_3 += TwoZetaInv * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 

 } 

/* FP integral partial class - Part 5, m=1 */ 
__device__ __inline__ FPint_1_5::FPint_1_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_14_1 = PBx * fs_1.x_14_0 - PCx * fs_2.x_14_0; 
#else 
  x_14_1 = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_14_1 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
#ifdef REG_FS 
  x_14_2 = PBy * fs_1.x_14_0 - PCy * fs_2.x_14_0; 
#else 
  x_14_2 = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_14_3 = PBz * fs_1.x_14_0 - PCz * fs_2.x_14_0; 
#else 
  x_14_3 = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_14_3 += TwoZetaInv * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 

 } 

/* FP integral partial class - Part 6, m=1 */ 
__device__ __inline__ FPint_1_6::FPint_1_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_15_1 = PBx * fs_1.x_15_0 - PCx * fs_2.x_15_0; 
#else 
  x_15_1 = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_15_2 = PBy * fs_1.x_15_0 - PCy * fs_2.x_15_0; 
#else 
  x_15_2 = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_15_2 += TwoZetaInv * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
#ifdef REG_FS 
  x_15_3 = PBz * fs_1.x_15_0 - PCz * fs_2.x_15_0; 
#else 
  x_15_3 = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_15_3 += TwoZetaInv * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 

 } 

/* FP integral partial class - Part 7, m=1 */ 
__device__ __inline__ FPint_1_7::FPint_1_7(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_16_1 = PBx * fs_1.x_16_0 - PCx * fs_2.x_16_0; 
#else 
  x_16_1 = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_16_2 = PBy * fs_1.x_16_0 - PCy * fs_2.x_16_0; 
#else 
  x_16_2 = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_16_2 += TwoZetaInv * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
#ifdef REG_FS 
  x_16_3 = PBz * fs_1.x_16_0 - PCz * fs_2.x_16_0; 
#else 
  x_16_3 = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_16_3 += TwoZetaInv * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 

 } 

/* FP integral partial class - Part 8, m=1 */ 
__device__ __inline__ FPint_1_8::FPint_1_8(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_17_1 = PBx * fs_1.x_17_0 - PCx * fs_2.x_17_0; 
#else 
  x_17_1 = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_17_1 += TwoZetaInv * 3.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
#ifdef REG_FS 
  x_17_2 = PBy * fs_1.x_17_0 - PCy * fs_2.x_17_0; 
#else 
  x_17_2 = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_17_3 = PBz * fs_1.x_17_0 - PCz * fs_2.x_17_0; 
#else 
  x_17_3 = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* FP integral partial class - Part 9, m=1 */ 
__device__ __inline__ FPint_1_9::FPint_1_9(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_18_1 = PBx * fs_1.x_18_0 - PCx * fs_2.x_18_0; 
#else 
  x_18_1 = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_18_2 = PBy * fs_1.x_18_0 - PCy * fs_2.x_18_0; 
#else 
  x_18_2 = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_18_2 += TwoZetaInv * 3.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
#ifdef REG_FS 
  x_18_3 = PBz * fs_1.x_18_0 - PCz * fs_2.x_18_0; 
#else 
  x_18_3 = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* FP integral partial class - Part 10, m=1 */ 
__device__ __inline__ FPint_1_10::FPint_1_10(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 

#ifdef REG_FS 
  x_19_1 = PBx * fs_1.x_19_0 - PCx * fs_2.x_19_0; 
#else 
  x_19_1 = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_19_2 = PBy * fs_1.x_19_0 - PCy * fs_2.x_19_0; 
#else 
  x_19_2 = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  x_19_3 = PBz * fs_1.x_19_0 - PCz * fs_2.x_19_0; 
#else 
  x_19_3 = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2); 
#endif 
  x_19_3 += TwoZetaInv * 3.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 

 } 

/* FP auxilary integral, m=2 */ 
__device__ __inline__ FPint_2::FPint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FP 
#ifdef REG_FS 
  x_10_1 = PBx * fs_2.x_10_0 - PCx * fs_3.x_10_0; 
#else 
  x_10_1 = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_10_1 += TwoZetaInv * 1.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
#ifdef REG_FS 
  x_10_2 = PBy * fs_2.x_10_0 - PCy * fs_3.x_10_0; 
#else 
  x_10_2 = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_10_2 += TwoZetaInv * 1.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
#ifdef REG_FS 
  x_10_3 = PBz * fs_2.x_10_0 - PCz * fs_3.x_10_0; 
#else 
  x_10_3 = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_10_3 += TwoZetaInv * 1.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
#ifdef REG_FS 
  x_11_1 = PBx * fs_2.x_11_0 - PCx * fs_3.x_11_0; 
#else 
  x_11_1 = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_11_1 += TwoZetaInv * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
#ifdef REG_FS 
  x_11_2 = PBy * fs_2.x_11_0 - PCy * fs_3.x_11_0; 
#else 
  x_11_2 = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_11_2 += TwoZetaInv * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
#ifdef REG_FS 
  x_11_3 = PBz * fs_2.x_11_0 - PCz * fs_3.x_11_0; 
#else 
  x_11_3 = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_12_1 = PBx * fs_2.x_12_0 - PCx * fs_3.x_12_0; 
#else 
  x_12_1 = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_12_1 += TwoZetaInv * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
#ifdef REG_FS 
  x_12_2 = PBy * fs_2.x_12_0 - PCy * fs_3.x_12_0; 
#else 
  x_12_2 = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_12_2 += TwoZetaInv * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
#ifdef REG_FS 
  x_12_3 = PBz * fs_2.x_12_0 - PCz * fs_3.x_12_0; 
#else 
  x_12_3 = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_13_1 = PBx * fs_2.x_13_0 - PCx * fs_3.x_13_0; 
#else 
  x_13_1 = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_13_1 += TwoZetaInv * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
#ifdef REG_FS 
  x_13_2 = PBy * fs_2.x_13_0 - PCy * fs_3.x_13_0; 
#else 
  x_13_2 = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_13_3 = PBz * fs_2.x_13_0 - PCz * fs_3.x_13_0; 
#else 
  x_13_3 = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_13_3 += TwoZetaInv * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
#ifdef REG_FS 
  x_14_1 = PBx * fs_2.x_14_0 - PCx * fs_3.x_14_0; 
#else 
  x_14_1 = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_14_1 += TwoZetaInv * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
#ifdef REG_FS 
  x_14_2 = PBy * fs_2.x_14_0 - PCy * fs_3.x_14_0; 
#else 
  x_14_2 = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_14_3 = PBz * fs_2.x_14_0 - PCz * fs_3.x_14_0; 
#else 
  x_14_3 = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_14_3 += TwoZetaInv * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
#ifdef REG_FS 
  x_15_1 = PBx * fs_2.x_15_0 - PCx * fs_3.x_15_0; 
#else 
  x_15_1 = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_15_2 = PBy * fs_2.x_15_0 - PCy * fs_3.x_15_0; 
#else 
  x_15_2 = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_15_2 += TwoZetaInv * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
#ifdef REG_FS 
  x_15_3 = PBz * fs_2.x_15_0 - PCz * fs_3.x_15_0; 
#else 
  x_15_3 = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_15_3 += TwoZetaInv * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
#ifdef REG_FS 
  x_16_1 = PBx * fs_2.x_16_0 - PCx * fs_3.x_16_0; 
#else 
  x_16_1 = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_16_2 = PBy * fs_2.x_16_0 - PCy * fs_3.x_16_0; 
#else 
  x_16_2 = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_16_2 += TwoZetaInv * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
#ifdef REG_FS 
  x_16_3 = PBz * fs_2.x_16_0 - PCz * fs_3.x_16_0; 
#else 
  x_16_3 = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_16_3 += TwoZetaInv * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
#ifdef REG_FS 
  x_17_1 = PBx * fs_2.x_17_0 - PCx * fs_3.x_17_0; 
#else 
  x_17_1 = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_17_1 += TwoZetaInv * 3.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
#ifdef REG_FS 
  x_17_2 = PBy * fs_2.x_17_0 - PCy * fs_3.x_17_0; 
#else 
  x_17_2 = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_17_3 = PBz * fs_2.x_17_0 - PCz * fs_3.x_17_0; 
#else 
  x_17_3 = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_18_1 = PBx * fs_2.x_18_0 - PCx * fs_3.x_18_0; 
#else 
  x_18_1 = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_18_2 = PBy * fs_2.x_18_0 - PCy * fs_3.x_18_0; 
#else 
  x_18_2 = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_18_2 += TwoZetaInv * 3.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
#ifdef REG_FS 
  x_18_3 = PBz * fs_2.x_18_0 - PCz * fs_3.x_18_0; 
#else 
  x_18_3 = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_19_1 = PBx * fs_2.x_19_0 - PCx * fs_3.x_19_0; 
#else 
  x_19_1 = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_19_2 = PBy * fs_2.x_19_0 - PCy * fs_3.x_19_0; 
#else 
  x_19_2 = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_19_3 = PBz * fs_2.x_19_0 - PCz * fs_3.x_19_0; 
#else 
  x_19_3 = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_19_3 += TwoZetaInv * 3.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
#else 
  QUICKDouble val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_10_0 - PCx * fs_3.x_10_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
  LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_10_0 - PCy * fs_3.x_10_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
  LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_10_0 - PCz * fs_3.x_10_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
  LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_11_0 - PCx * fs_3.x_11_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
  LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_11_0 - PCy * fs_3.x_11_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
  LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_11_0 - PCz * fs_3.x_11_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_12_0 - PCx * fs_3.x_12_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
  LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_12_0 - PCy * fs_3.x_12_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
  LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_12_0 - PCz * fs_3.x_12_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_13_0 - PCx * fs_3.x_13_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
  LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_13_0 - PCy * fs_3.x_13_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_13_0 - PCz * fs_3.x_13_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
  LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_14_0 - PCx * fs_3.x_14_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
  LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_14_0 - PCy * fs_3.x_14_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_14_0 - PCz * fs_3.x_14_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
  LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_15_0 - PCx * fs_3.x_15_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_15_0 - PCy * fs_3.x_15_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
  LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_15_0 - PCz * fs_3.x_15_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
  LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_16_0 - PCx * fs_3.x_16_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_16_0 - PCy * fs_3.x_16_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
  LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_16_0 - PCz * fs_3.x_16_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
  LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_17_0 - PCx * fs_3.x_17_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
  LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_17_0 - PCy * fs_3.x_17_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_17_0 - PCz * fs_3.x_17_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_18_0 - PCx * fs_3.x_18_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_18_0 - PCy * fs_3.x_18_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
  LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_18_0 - PCz * fs_3.x_18_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBx * fs_2.x_19_0 - PCx * fs_3.x_19_0; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBy * fs_2.x_19_0 - PCy * fs_3.x_19_0; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_FS 
  val = PBz * fs_2.x_19_0 - PCz * fs_3.x_19_0; 
#else 
  val = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 3.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
  LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2) = val; 
#endif 

 } 

/* FP integral partial class - Part 1, m=2 */ 
__device__ __inline__ FPint_2_1::FPint_2_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_10_1 = PBx * fs_2.x_10_0 - PCx * fs_3.x_10_0; 
#else 
  x_10_1 = PBx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_10_1 += TwoZetaInv * 1.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
#ifdef REG_FS 
  x_10_2 = PBy * fs_2.x_10_0 - PCy * fs_3.x_10_0; 
#else 
  x_10_2 = PBy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_10_2 += TwoZetaInv * 1.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
#ifdef REG_FS 
  x_10_3 = PBz * fs_2.x_10_0 - PCz * fs_3.x_10_0; 
#else 
  x_10_3 = PBz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_10_3 += TwoZetaInv * 1.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 

 } 

/* FP integral partial class - Part 2, m=2 */ 
__device__ __inline__ FPint_2_2::FPint_2_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_11_1 = PBx * fs_2.x_11_0 - PCx * fs_3.x_11_0; 
#else 
  x_11_1 = PBx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_11_1 += TwoZetaInv * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
#ifdef REG_FS 
  x_11_2 = PBy * fs_2.x_11_0 - PCy * fs_3.x_11_0; 
#else 
  x_11_2 = PBy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_11_2 += TwoZetaInv * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
#ifdef REG_FS 
  x_11_3 = PBz * fs_2.x_11_0 - PCz * fs_3.x_11_0; 
#else 
  x_11_3 = PBz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* FP integral partial class - Part 3, m=2 */ 
__device__ __inline__ FPint_2_3::FPint_2_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_12_1 = PBx * fs_2.x_12_0 - PCx * fs_3.x_12_0; 
#else 
  x_12_1 = PBx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_12_1 += TwoZetaInv * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
#ifdef REG_FS 
  x_12_2 = PBy * fs_2.x_12_0 - PCy * fs_3.x_12_0; 
#else 
  x_12_2 = PBy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_12_2 += TwoZetaInv * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
#ifdef REG_FS 
  x_12_3 = PBz * fs_2.x_12_0 - PCz * fs_3.x_12_0; 
#else 
  x_12_3 = PBz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* FP integral partial class - Part 4, m=2 */ 
__device__ __inline__ FPint_2_4::FPint_2_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_13_1 = PBx * fs_2.x_13_0 - PCx * fs_3.x_13_0; 
#else 
  x_13_1 = PBx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_13_1 += TwoZetaInv * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
#ifdef REG_FS 
  x_13_2 = PBy * fs_2.x_13_0 - PCy * fs_3.x_13_0; 
#else 
  x_13_2 = PBy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_13_3 = PBz * fs_2.x_13_0 - PCz * fs_3.x_13_0; 
#else 
  x_13_3 = PBz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_13_3 += TwoZetaInv * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 

 } 

/* FP integral partial class - Part 5, m=2 */ 
__device__ __inline__ FPint_2_5::FPint_2_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_14_1 = PBx * fs_2.x_14_0 - PCx * fs_3.x_14_0; 
#else 
  x_14_1 = PBx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_14_1 += TwoZetaInv * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
#ifdef REG_FS 
  x_14_2 = PBy * fs_2.x_14_0 - PCy * fs_3.x_14_0; 
#else 
  x_14_2 = PBy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_14_3 = PBz * fs_2.x_14_0 - PCz * fs_3.x_14_0; 
#else 
  x_14_3 = PBz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_14_3 += TwoZetaInv * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 

 } 

/* FP integral partial class - Part 6, m=2 */ 
__device__ __inline__ FPint_2_6::FPint_2_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_15_1 = PBx * fs_2.x_15_0 - PCx * fs_3.x_15_0; 
#else 
  x_15_1 = PBx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_15_2 = PBy * fs_2.x_15_0 - PCy * fs_3.x_15_0; 
#else 
  x_15_2 = PBy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_15_2 += TwoZetaInv * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
#ifdef REG_FS 
  x_15_3 = PBz * fs_2.x_15_0 - PCz * fs_3.x_15_0; 
#else 
  x_15_3 = PBz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_15_3 += TwoZetaInv * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 

 } 

/* FP integral partial class - Part 7, m=2 */ 
__device__ __inline__ FPint_2_7::FPint_2_7(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_16_1 = PBx * fs_2.x_16_0 - PCx * fs_3.x_16_0; 
#else 
  x_16_1 = PBx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_16_2 = PBy * fs_2.x_16_0 - PCy * fs_3.x_16_0; 
#else 
  x_16_2 = PBy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_16_2 += TwoZetaInv * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
#ifdef REG_FS 
  x_16_3 = PBz * fs_2.x_16_0 - PCz * fs_3.x_16_0; 
#else 
  x_16_3 = PBz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_16_3 += TwoZetaInv * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 

 } 

/* FP integral partial class - Part 8, m=2 */ 
__device__ __inline__ FPint_2_8::FPint_2_8(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_17_1 = PBx * fs_2.x_17_0 - PCx * fs_3.x_17_0; 
#else 
  x_17_1 = PBx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_17_1 += TwoZetaInv * 3.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
#ifdef REG_FS 
  x_17_2 = PBy * fs_2.x_17_0 - PCy * fs_3.x_17_0; 
#else 
  x_17_2 = PBy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_17_3 = PBz * fs_2.x_17_0 - PCz * fs_3.x_17_0; 
#else 
  x_17_3 = PBz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* FP integral partial class - Part 9, m=2 */ 
__device__ __inline__ FPint_2_9::FPint_2_9(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_18_1 = PBx * fs_2.x_18_0 - PCx * fs_3.x_18_0; 
#else 
  x_18_1 = PBx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_18_2 = PBy * fs_2.x_18_0 - PCy * fs_3.x_18_0; 
#else 
  x_18_2 = PBy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_18_2 += TwoZetaInv * 3.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
#ifdef REG_FS 
  x_18_3 = PBz * fs_2.x_18_0 - PCz * fs_3.x_18_0; 
#else 
  x_18_3 = PBz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* FP integral partial class - Part 10, m=2 */ 
__device__ __inline__ FPint_2_10::FPint_2_10(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                    QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                    QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=3 

#ifdef REG_FS 
  x_19_1 = PBx * fs_2.x_19_0 - PCx * fs_3.x_19_0; 
#else 
  x_19_1 = PBx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_19_2 = PBy * fs_2.x_19_0 - PCy * fs_3.x_19_0; 
#else 
  x_19_2 = PBy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_FS 
  x_19_3 = PBz * fs_2.x_19_0 - PCz * fs_3.x_19_0; 
#else 
  x_19_3 = PBz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 3); 
#endif 
  x_19_3 += TwoZetaInv * 3.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 

 } 

/* PF true integral, m=0 */ 
__device__ __inline__ PFint_0::PFint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_PF 
#ifdef REG_SF 
  x_1_10 = PAx * sf_0.x_0_10 - PCx * sf_1.x_0_10; 
#else 
  x_1_10 = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_10 += TwoZetaInv * 1.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
#ifdef REG_SF 
  x_2_10 = PAy * sf_0.x_0_10 - PCy * sf_1.x_0_10; 
#else 
  x_2_10 = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_10 += TwoZetaInv * 1.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
#ifdef REG_SF 
  x_3_10 = PAz * sf_0.x_0_10 - PCz * sf_1.x_0_10; 
#else 
  x_3_10 = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_10 += TwoZetaInv * 1.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
#ifdef REG_SF 
  x_1_11 = PAx * sf_0.x_0_11 - PCx * sf_1.x_0_11; 
#else 
  x_1_11 = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_11 += TwoZetaInv * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
#ifdef REG_SF 
  x_2_11 = PAy * sf_0.x_0_11 - PCy * sf_1.x_0_11; 
#else 
  x_2_11 = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_11 += TwoZetaInv * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
#ifdef REG_SF 
  x_3_11 = PAz * sf_0.x_0_11 - PCz * sf_1.x_0_11; 
#else 
  x_3_11 = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_1_12 = PAx * sf_0.x_0_12 - PCx * sf_1.x_0_12; 
#else 
  x_1_12 = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_12 += TwoZetaInv * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
#ifdef REG_SF 
  x_2_12 = PAy * sf_0.x_0_12 - PCy * sf_1.x_0_12; 
#else 
  x_2_12 = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_12 += TwoZetaInv * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
#ifdef REG_SF 
  x_3_12 = PAz * sf_0.x_0_12 - PCz * sf_1.x_0_12; 
#else 
  x_3_12 = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_1_13 = PAx * sf_0.x_0_13 - PCx * sf_1.x_0_13; 
#else 
  x_1_13 = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_13 += TwoZetaInv * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
#ifdef REG_SF 
  x_2_13 = PAy * sf_0.x_0_13 - PCy * sf_1.x_0_13; 
#else 
  x_2_13 = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_13 = PAz * sf_0.x_0_13 - PCz * sf_1.x_0_13; 
#else 
  x_3_13 = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_13 += TwoZetaInv * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
#ifdef REG_SF 
  x_1_14 = PAx * sf_0.x_0_14 - PCx * sf_1.x_0_14; 
#else 
  x_1_14 = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_14 += TwoZetaInv * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
#ifdef REG_SF 
  x_2_14 = PAy * sf_0.x_0_14 - PCy * sf_1.x_0_14; 
#else 
  x_2_14 = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_14 = PAz * sf_0.x_0_14 - PCz * sf_1.x_0_14; 
#else 
  x_3_14 = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_14 += TwoZetaInv * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
#ifdef REG_SF 
  x_1_15 = PAx * sf_0.x_0_15 - PCx * sf_1.x_0_15; 
#else 
  x_1_15 = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_15 = PAy * sf_0.x_0_15 - PCy * sf_1.x_0_15; 
#else 
  x_2_15 = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_15 += TwoZetaInv * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
#ifdef REG_SF 
  x_3_15 = PAz * sf_0.x_0_15 - PCz * sf_1.x_0_15; 
#else 
  x_3_15 = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_15 += TwoZetaInv * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
#ifdef REG_SF 
  x_1_16 = PAx * sf_0.x_0_16 - PCx * sf_1.x_0_16; 
#else 
  x_1_16 = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_16 = PAy * sf_0.x_0_16 - PCy * sf_1.x_0_16; 
#else 
  x_2_16 = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_16 += TwoZetaInv * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
#ifdef REG_SF 
  x_3_16 = PAz * sf_0.x_0_16 - PCz * sf_1.x_0_16; 
#else 
  x_3_16 = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_16 += TwoZetaInv * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
#ifdef REG_SF 
  x_1_17 = PAx * sf_0.x_0_17 - PCx * sf_1.x_0_17; 
#else 
  x_1_17 = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_17 += TwoZetaInv * 3.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
#ifdef REG_SF 
  x_2_17 = PAy * sf_0.x_0_17 - PCy * sf_1.x_0_17; 
#else 
  x_2_17 = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_17 = PAz * sf_0.x_0_17 - PCz * sf_1.x_0_17; 
#else 
  x_3_17 = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_1_18 = PAx * sf_0.x_0_18 - PCx * sf_1.x_0_18; 
#else 
  x_1_18 = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_18 = PAy * sf_0.x_0_18 - PCy * sf_1.x_0_18; 
#else 
  x_2_18 = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_18 += TwoZetaInv * 3.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
#ifdef REG_SF 
  x_3_18 = PAz * sf_0.x_0_18 - PCz * sf_1.x_0_18; 
#else 
  x_3_18 = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_1_19 = PAx * sf_0.x_0_19 - PCx * sf_1.x_0_19; 
#else 
  x_1_19 = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_19 = PAy * sf_0.x_0_19 - PCy * sf_1.x_0_19; 
#else 
  x_2_19 = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_19 = PAz * sf_0.x_0_19 - PCz * sf_1.x_0_19; 
#else 
  x_3_19 = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_19 += TwoZetaInv * 3.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
#else 
  QUICKDouble val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_10 - PCx * sf_1.x_0_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
  LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_10 - PCy * sf_1.x_0_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
  LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_10 - PCz * sf_1.x_0_10; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
  LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_11 - PCx * sf_1.x_0_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
  LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_11 - PCy * sf_1.x_0_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
  LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_11 - PCz * sf_1.x_0_11; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_12 - PCx * sf_1.x_0_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
  LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_12 - PCy * sf_1.x_0_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
  LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_12 - PCz * sf_1.x_0_12; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_13 - PCx * sf_1.x_0_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
  LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_13 - PCy * sf_1.x_0_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_13 - PCz * sf_1.x_0_13; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
  LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_14 - PCx * sf_1.x_0_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
  LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_14 - PCy * sf_1.x_0_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_14 - PCz * sf_1.x_0_14; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
  LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_15 - PCx * sf_1.x_0_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_15 - PCy * sf_1.x_0_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
  LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_15 - PCz * sf_1.x_0_15; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
  LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_16 - PCx * sf_1.x_0_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_16 - PCy * sf_1.x_0_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
  LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_16 - PCz * sf_1.x_0_16; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
  LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_17 - PCx * sf_1.x_0_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
  LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_17 - PCy * sf_1.x_0_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_17 - PCz * sf_1.x_0_17; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_18 - PCx * sf_1.x_0_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_18 - PCy * sf_1.x_0_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
  LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_18 - PCz * sf_1.x_0_18; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAx * sf_0.x_0_19 - PCx * sf_1.x_0_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAy * sf_0.x_0_19 - PCy * sf_1.x_0_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_SF 
  val = PAz * sf_0.x_0_19 - PCz * sf_1.x_0_19; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
  LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) = val; 
#endif 

 } 

/* PF integral partial class - Part 1, m=0 */ 
__device__ __inline__ PFint_0_1::PFint_0_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_10 = PAx * sf_0.x_0_10 - PCx * sf_1.x_0_10; 
#else 
  x_1_10 = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_10 += TwoZetaInv * 1.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
#ifdef REG_SF 
  x_2_10 = PAy * sf_0.x_0_10 - PCy * sf_1.x_0_10; 
#else 
  x_2_10 = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_10 += TwoZetaInv * 1.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
#ifdef REG_SF 
  x_3_10 = PAz * sf_0.x_0_10 - PCz * sf_1.x_0_10; 
#else 
  x_3_10 = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_10 += TwoZetaInv * 1.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 

 } 

/* PF integral partial class - Part 2, m=0 */ 
__device__ __inline__ PFint_0_2::PFint_0_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_11 = PAx * sf_0.x_0_11 - PCx * sf_1.x_0_11; 
#else 
  x_1_11 = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_11 += TwoZetaInv * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
#ifdef REG_SF 
  x_2_11 = PAy * sf_0.x_0_11 - PCy * sf_1.x_0_11; 
#else 
  x_2_11 = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_11 += TwoZetaInv * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
#ifdef REG_SF 
  x_3_11 = PAz * sf_0.x_0_11 - PCz * sf_1.x_0_11; 
#else 
  x_3_11 = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* PF integral partial class - Part 3, m=0 */ 
__device__ __inline__ PFint_0_3::PFint_0_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_12 = PAx * sf_0.x_0_12 - PCx * sf_1.x_0_12; 
#else 
  x_1_12 = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_12 += TwoZetaInv * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
#ifdef REG_SF 
  x_2_12 = PAy * sf_0.x_0_12 - PCy * sf_1.x_0_12; 
#else 
  x_2_12 = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_12 += TwoZetaInv * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
#ifdef REG_SF 
  x_3_12 = PAz * sf_0.x_0_12 - PCz * sf_1.x_0_12; 
#else 
  x_3_12 = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* PF integral partial class - Part 4, m=0 */ 
__device__ __inline__ PFint_0_4::PFint_0_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_13 = PAx * sf_0.x_0_13 - PCx * sf_1.x_0_13; 
#else 
  x_1_13 = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_13 += TwoZetaInv * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
#ifdef REG_SF 
  x_2_13 = PAy * sf_0.x_0_13 - PCy * sf_1.x_0_13; 
#else 
  x_2_13 = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_13 = PAz * sf_0.x_0_13 - PCz * sf_1.x_0_13; 
#else 
  x_3_13 = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_13 += TwoZetaInv * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 

 } 

/* PF integral partial class - Part 5, m=0 */ 
__device__ __inline__ PFint_0_5::PFint_0_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_14 = PAx * sf_0.x_0_14 - PCx * sf_1.x_0_14; 
#else 
  x_1_14 = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_14 += TwoZetaInv * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
#ifdef REG_SF 
  x_2_14 = PAy * sf_0.x_0_14 - PCy * sf_1.x_0_14; 
#else 
  x_2_14 = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_14 = PAz * sf_0.x_0_14 - PCz * sf_1.x_0_14; 
#else 
  x_3_14 = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_14 += TwoZetaInv * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 

 } 

/* PF integral partial class - Part 6, m=0 */ 
__device__ __inline__ PFint_0_6::PFint_0_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_15 = PAx * sf_0.x_0_15 - PCx * sf_1.x_0_15; 
#else 
  x_1_15 = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_15 = PAy * sf_0.x_0_15 - PCy * sf_1.x_0_15; 
#else 
  x_2_15 = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_15 += TwoZetaInv * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
#ifdef REG_SF 
  x_3_15 = PAz * sf_0.x_0_15 - PCz * sf_1.x_0_15; 
#else 
  x_3_15 = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_15 += TwoZetaInv * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 

 } 

/* PF integral partial class - Part 7, m=0 */ 
__device__ __inline__ PFint_0_7::PFint_0_7(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_16 = PAx * sf_0.x_0_16 - PCx * sf_1.x_0_16; 
#else 
  x_1_16 = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_16 = PAy * sf_0.x_0_16 - PCy * sf_1.x_0_16; 
#else 
  x_2_16 = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_16 += TwoZetaInv * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
#ifdef REG_SF 
  x_3_16 = PAz * sf_0.x_0_16 - PCz * sf_1.x_0_16; 
#else 
  x_3_16 = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_16 += TwoZetaInv * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 

 } 

/* PF integral partial class - Part 8, m=0 */ 
__device__ __inline__ PFint_0_8::PFint_0_8(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_17 = PAx * sf_0.x_0_17 - PCx * sf_1.x_0_17; 
#else 
  x_1_17 = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
  x_1_17 += TwoZetaInv * 3.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
#ifdef REG_SF 
  x_2_17 = PAy * sf_0.x_0_17 - PCy * sf_1.x_0_17; 
#else 
  x_2_17 = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_17 = PAz * sf_0.x_0_17 - PCz * sf_1.x_0_17; 
#else 
  x_3_17 = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* PF integral partial class - Part 9, m=0 */ 
__device__ __inline__ PFint_0_9::PFint_0_9(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_18 = PAx * sf_0.x_0_18 - PCx * sf_1.x_0_18; 
#else 
  x_1_18 = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_18 = PAy * sf_0.x_0_18 - PCy * sf_1.x_0_18; 
#else 
  x_2_18 = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 
  x_2_18 += TwoZetaInv * 3.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
#ifdef REG_SF 
  x_3_18 = PAz * sf_0.x_0_18 - PCz * sf_1.x_0_18; 
#else 
  x_3_18 = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1); 
#endif 

 } 

/* PF integral partial class - Part 10, m=0 */ 
__device__ __inline__ PFint_0_10::PFint_0_10(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 

#ifdef REG_SF 
  x_1_19 = PAx * sf_0.x_0_19 - PCx * sf_1.x_0_19; 
#else 
  x_1_19 = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0)- PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_2_19 = PAy * sf_0.x_0_19 - PCy * sf_1.x_0_19; 
#else 
  x_2_19 = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0)- PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  x_3_19 = PAz * sf_0.x_0_19 - PCz * sf_1.x_0_19; 
#else 
  x_3_19 = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0)- PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1); 
#endif 
  x_3_19 += TwoZetaInv * 3.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 

 } 

/* PF auxilary integral, m=1 */ 
__device__ __inline__ PFint_1::PFint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_PF 
#ifdef REG_SF 
  x_1_10 = PAx * sf_1.x_0_10 - PCx * sf_2.x_0_10; 
#else 
  x_1_10 = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_10 += TwoZetaInv * 1.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
#ifdef REG_SF 
  x_2_10 = PAy * sf_1.x_0_10 - PCy * sf_2.x_0_10; 
#else 
  x_2_10 = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_10 += TwoZetaInv * 1.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
#ifdef REG_SF 
  x_3_10 = PAz * sf_1.x_0_10 - PCz * sf_2.x_0_10; 
#else 
  x_3_10 = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_10 += TwoZetaInv * 1.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
#ifdef REG_SF 
  x_1_11 = PAx * sf_1.x_0_11 - PCx * sf_2.x_0_11; 
#else 
  x_1_11 = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_11 += TwoZetaInv * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
#ifdef REG_SF 
  x_2_11 = PAy * sf_1.x_0_11 - PCy * sf_2.x_0_11; 
#else 
  x_2_11 = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_11 += TwoZetaInv * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
#ifdef REG_SF 
  x_3_11 = PAz * sf_1.x_0_11 - PCz * sf_2.x_0_11; 
#else 
  x_3_11 = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_1_12 = PAx * sf_1.x_0_12 - PCx * sf_2.x_0_12; 
#else 
  x_1_12 = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_12 += TwoZetaInv * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
#ifdef REG_SF 
  x_2_12 = PAy * sf_1.x_0_12 - PCy * sf_2.x_0_12; 
#else 
  x_2_12 = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_12 += TwoZetaInv * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
#ifdef REG_SF 
  x_3_12 = PAz * sf_1.x_0_12 - PCz * sf_2.x_0_12; 
#else 
  x_3_12 = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_1_13 = PAx * sf_1.x_0_13 - PCx * sf_2.x_0_13; 
#else 
  x_1_13 = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_13 += TwoZetaInv * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
#ifdef REG_SF 
  x_2_13 = PAy * sf_1.x_0_13 - PCy * sf_2.x_0_13; 
#else 
  x_2_13 = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_13 = PAz * sf_1.x_0_13 - PCz * sf_2.x_0_13; 
#else 
  x_3_13 = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_13 += TwoZetaInv * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
#ifdef REG_SF 
  x_1_14 = PAx * sf_1.x_0_14 - PCx * sf_2.x_0_14; 
#else 
  x_1_14 = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_14 += TwoZetaInv * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
#ifdef REG_SF 
  x_2_14 = PAy * sf_1.x_0_14 - PCy * sf_2.x_0_14; 
#else 
  x_2_14 = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_14 = PAz * sf_1.x_0_14 - PCz * sf_2.x_0_14; 
#else 
  x_3_14 = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_14 += TwoZetaInv * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
#ifdef REG_SF 
  x_1_15 = PAx * sf_1.x_0_15 - PCx * sf_2.x_0_15; 
#else 
  x_1_15 = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_15 = PAy * sf_1.x_0_15 - PCy * sf_2.x_0_15; 
#else 
  x_2_15 = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_15 += TwoZetaInv * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
#ifdef REG_SF 
  x_3_15 = PAz * sf_1.x_0_15 - PCz * sf_2.x_0_15; 
#else 
  x_3_15 = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_15 += TwoZetaInv * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
#ifdef REG_SF 
  x_1_16 = PAx * sf_1.x_0_16 - PCx * sf_2.x_0_16; 
#else 
  x_1_16 = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_16 = PAy * sf_1.x_0_16 - PCy * sf_2.x_0_16; 
#else 
  x_2_16 = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_16 += TwoZetaInv * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
#ifdef REG_SF 
  x_3_16 = PAz * sf_1.x_0_16 - PCz * sf_2.x_0_16; 
#else 
  x_3_16 = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_16 += TwoZetaInv * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
#ifdef REG_SF 
  x_1_17 = PAx * sf_1.x_0_17 - PCx * sf_2.x_0_17; 
#else 
  x_1_17 = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_17 += TwoZetaInv * 3.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
#ifdef REG_SF 
  x_2_17 = PAy * sf_1.x_0_17 - PCy * sf_2.x_0_17; 
#else 
  x_2_17 = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_17 = PAz * sf_1.x_0_17 - PCz * sf_2.x_0_17; 
#else 
  x_3_17 = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_1_18 = PAx * sf_1.x_0_18 - PCx * sf_2.x_0_18; 
#else 
  x_1_18 = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_18 = PAy * sf_1.x_0_18 - PCy * sf_2.x_0_18; 
#else 
  x_2_18 = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_18 += TwoZetaInv * 3.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
#ifdef REG_SF 
  x_3_18 = PAz * sf_1.x_0_18 - PCz * sf_2.x_0_18; 
#else 
  x_3_18 = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_1_19 = PAx * sf_1.x_0_19 - PCx * sf_2.x_0_19; 
#else 
  x_1_19 = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_19 = PAy * sf_1.x_0_19 - PCy * sf_2.x_0_19; 
#else 
  x_2_19 = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_19 = PAz * sf_1.x_0_19 - PCz * sf_2.x_0_19; 
#else 
  x_3_19 = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_19 += TwoZetaInv * 3.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
#else 
  QUICKDouble val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_10 - PCx * sf_2.x_0_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
  LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_10 - PCy * sf_2.x_0_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
  LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_10 - PCz * sf_2.x_0_10; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
  LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_11 - PCx * sf_2.x_0_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
  LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_11 - PCy * sf_2.x_0_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
  LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_11 - PCz * sf_2.x_0_11; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_12 - PCx * sf_2.x_0_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
  LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_12 - PCy * sf_2.x_0_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
  LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_12 - PCz * sf_2.x_0_12; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_13 - PCx * sf_2.x_0_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
  LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_13 - PCy * sf_2.x_0_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_13 - PCz * sf_2.x_0_13; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
  LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_14 - PCx * sf_2.x_0_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
  LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_14 - PCy * sf_2.x_0_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_14 - PCz * sf_2.x_0_14; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
  LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_15 - PCx * sf_2.x_0_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_15 - PCy * sf_2.x_0_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
  LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_15 - PCz * sf_2.x_0_15; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
  LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_16 - PCx * sf_2.x_0_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_16 - PCy * sf_2.x_0_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
  LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_16 - PCz * sf_2.x_0_16; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
  LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_17 - PCx * sf_2.x_0_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
  LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_17 - PCy * sf_2.x_0_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_17 - PCz * sf_2.x_0_17; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_18 - PCx * sf_2.x_0_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_18 - PCy * sf_2.x_0_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
  LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_18 - PCz * sf_2.x_0_18; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAx * sf_1.x_0_19 - PCx * sf_2.x_0_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAy * sf_1.x_0_19 - PCy * sf_2.x_0_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_SF 
  val = PAz * sf_1.x_0_19 - PCz * sf_2.x_0_19; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
  LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) = val; 
#endif 

 } 

/* PF integral partial class - Part 1, m=1 */ 
__device__ __inline__ PFint_1_1::PFint_1_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_10 = PAx * sf_1.x_0_10 - PCx * sf_2.x_0_10; 
#else 
  x_1_10 = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_10 += TwoZetaInv * 1.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
#ifdef REG_SF 
  x_2_10 = PAy * sf_1.x_0_10 - PCy * sf_2.x_0_10; 
#else 
  x_2_10 = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_10 += TwoZetaInv * 1.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
#ifdef REG_SF 
  x_3_10 = PAz * sf_1.x_0_10 - PCz * sf_2.x_0_10; 
#else 
  x_3_10 = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_10 += TwoZetaInv * 1.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 

 } 

/* PF integral partial class - Part 2, m=1 */ 
__device__ __inline__ PFint_1_2::PFint_1_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_11 = PAx * sf_1.x_0_11 - PCx * sf_2.x_0_11; 
#else 
  x_1_11 = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_11 += TwoZetaInv * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
#ifdef REG_SF 
  x_2_11 = PAy * sf_1.x_0_11 - PCy * sf_2.x_0_11; 
#else 
  x_2_11 = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_11 += TwoZetaInv * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
#ifdef REG_SF 
  x_3_11 = PAz * sf_1.x_0_11 - PCz * sf_2.x_0_11; 
#else 
  x_3_11 = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* PF integral partial class - Part 3, m=1 */ 
__device__ __inline__ PFint_1_3::PFint_1_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_12 = PAx * sf_1.x_0_12 - PCx * sf_2.x_0_12; 
#else 
  x_1_12 = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_12 += TwoZetaInv * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
#ifdef REG_SF 
  x_2_12 = PAy * sf_1.x_0_12 - PCy * sf_2.x_0_12; 
#else 
  x_2_12 = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_12 += TwoZetaInv * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
#ifdef REG_SF 
  x_3_12 = PAz * sf_1.x_0_12 - PCz * sf_2.x_0_12; 
#else 
  x_3_12 = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* PF integral partial class - Part 4, m=1 */ 
__device__ __inline__ PFint_1_4::PFint_1_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_13 = PAx * sf_1.x_0_13 - PCx * sf_2.x_0_13; 
#else 
  x_1_13 = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_13 += TwoZetaInv * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
#ifdef REG_SF 
  x_2_13 = PAy * sf_1.x_0_13 - PCy * sf_2.x_0_13; 
#else 
  x_2_13 = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_13 = PAz * sf_1.x_0_13 - PCz * sf_2.x_0_13; 
#else 
  x_3_13 = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_13 += TwoZetaInv * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 

 } 

/* PF integral partial class - Part 5, m=1 */ 
__device__ __inline__ PFint_1_5::PFint_1_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_14 = PAx * sf_1.x_0_14 - PCx * sf_2.x_0_14; 
#else 
  x_1_14 = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_14 += TwoZetaInv * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
#ifdef REG_SF 
  x_2_14 = PAy * sf_1.x_0_14 - PCy * sf_2.x_0_14; 
#else 
  x_2_14 = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_14 = PAz * sf_1.x_0_14 - PCz * sf_2.x_0_14; 
#else 
  x_3_14 = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_14 += TwoZetaInv * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 

 } 

/* PF integral partial class - Part 6, m=1 */ 
__device__ __inline__ PFint_1_6::PFint_1_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_15 = PAx * sf_1.x_0_15 - PCx * sf_2.x_0_15; 
#else 
  x_1_15 = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_15 = PAy * sf_1.x_0_15 - PCy * sf_2.x_0_15; 
#else 
  x_2_15 = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_15 += TwoZetaInv * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
#ifdef REG_SF 
  x_3_15 = PAz * sf_1.x_0_15 - PCz * sf_2.x_0_15; 
#else 
  x_3_15 = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_15 += TwoZetaInv * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 

 } 

/* PF integral partial class - Part 7, m=1 */ 
__device__ __inline__ PFint_1_7::PFint_1_7(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_16 = PAx * sf_1.x_0_16 - PCx * sf_2.x_0_16; 
#else 
  x_1_16 = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_16 = PAy * sf_1.x_0_16 - PCy * sf_2.x_0_16; 
#else 
  x_2_16 = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_16 += TwoZetaInv * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
#ifdef REG_SF 
  x_3_16 = PAz * sf_1.x_0_16 - PCz * sf_2.x_0_16; 
#else 
  x_3_16 = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_16 += TwoZetaInv * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 

 } 

/* PF integral partial class - Part 8, m=1 */ 
__device__ __inline__ PFint_1_8::PFint_1_8(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_17 = PAx * sf_1.x_0_17 - PCx * sf_2.x_0_17; 
#else 
  x_1_17 = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
  x_1_17 += TwoZetaInv * 3.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
#ifdef REG_SF 
  x_2_17 = PAy * sf_1.x_0_17 - PCy * sf_2.x_0_17; 
#else 
  x_2_17 = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_17 = PAz * sf_1.x_0_17 - PCz * sf_2.x_0_17; 
#else 
  x_3_17 = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* PF integral partial class - Part 9, m=1 */ 
__device__ __inline__ PFint_1_9::PFint_1_9(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_18 = PAx * sf_1.x_0_18 - PCx * sf_2.x_0_18; 
#else 
  x_1_18 = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_18 = PAy * sf_1.x_0_18 - PCy * sf_2.x_0_18; 
#else 
  x_2_18 = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 
  x_2_18 += TwoZetaInv * 3.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
#ifdef REG_SF 
  x_3_18 = PAz * sf_1.x_0_18 - PCz * sf_2.x_0_18; 
#else 
  x_3_18 = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2); 
#endif 

 } 

/* PF integral partial class - Part 10, m=1 */ 
__device__ __inline__ PFint_1_10::PFint_1_10(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 

#ifdef REG_SF 
  x_1_19 = PAx * sf_1.x_0_19 - PCx * sf_2.x_0_19; 
#else 
  x_1_19 = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)- PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_2_19 = PAy * sf_1.x_0_19 - PCy * sf_2.x_0_19; 
#else 
  x_2_19 = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)- PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  x_3_19 = PAz * sf_1.x_0_19 - PCz * sf_2.x_0_19; 
#else 
  x_3_19 = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)- PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2); 
#endif 
  x_3_19 += TwoZetaInv * 3.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 

 } 

/* PF auxilary integral, m=2 */ 
__device__ __inline__ PFint_2::PFint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_PF 
#ifdef REG_SF 
  x_1_10 = PAx * sf_2.x_0_10 - PCx * sf_3.x_0_10; 
#else 
  x_1_10 = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_10 += TwoZetaInv * 1.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
#ifdef REG_SF 
  x_2_10 = PAy * sf_2.x_0_10 - PCy * sf_3.x_0_10; 
#else 
  x_2_10 = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_10 += TwoZetaInv * 1.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
#ifdef REG_SF 
  x_3_10 = PAz * sf_2.x_0_10 - PCz * sf_3.x_0_10; 
#else 
  x_3_10 = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_10 += TwoZetaInv * 1.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
#ifdef REG_SF 
  x_1_11 = PAx * sf_2.x_0_11 - PCx * sf_3.x_0_11; 
#else 
  x_1_11 = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_11 += TwoZetaInv * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
#ifdef REG_SF 
  x_2_11 = PAy * sf_2.x_0_11 - PCy * sf_3.x_0_11; 
#else 
  x_2_11 = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_11 += TwoZetaInv * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
#ifdef REG_SF 
  x_3_11 = PAz * sf_2.x_0_11 - PCz * sf_3.x_0_11; 
#else 
  x_3_11 = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_1_12 = PAx * sf_2.x_0_12 - PCx * sf_3.x_0_12; 
#else 
  x_1_12 = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_12 += TwoZetaInv * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
#ifdef REG_SF 
  x_2_12 = PAy * sf_2.x_0_12 - PCy * sf_3.x_0_12; 
#else 
  x_2_12 = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_12 += TwoZetaInv * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
#ifdef REG_SF 
  x_3_12 = PAz * sf_2.x_0_12 - PCz * sf_3.x_0_12; 
#else 
  x_3_12 = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_1_13 = PAx * sf_2.x_0_13 - PCx * sf_3.x_0_13; 
#else 
  x_1_13 = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_13 += TwoZetaInv * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
#ifdef REG_SF 
  x_2_13 = PAy * sf_2.x_0_13 - PCy * sf_3.x_0_13; 
#else 
  x_2_13 = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_13 = PAz * sf_2.x_0_13 - PCz * sf_3.x_0_13; 
#else 
  x_3_13 = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_13 += TwoZetaInv * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
#ifdef REG_SF 
  x_1_14 = PAx * sf_2.x_0_14 - PCx * sf_3.x_0_14; 
#else 
  x_1_14 = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_14 += TwoZetaInv * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
#ifdef REG_SF 
  x_2_14 = PAy * sf_2.x_0_14 - PCy * sf_3.x_0_14; 
#else 
  x_2_14 = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_14 = PAz * sf_2.x_0_14 - PCz * sf_3.x_0_14; 
#else 
  x_3_14 = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_14 += TwoZetaInv * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
#ifdef REG_SF 
  x_1_15 = PAx * sf_2.x_0_15 - PCx * sf_3.x_0_15; 
#else 
  x_1_15 = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_15 = PAy * sf_2.x_0_15 - PCy * sf_3.x_0_15; 
#else 
  x_2_15 = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_15 += TwoZetaInv * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
#ifdef REG_SF 
  x_3_15 = PAz * sf_2.x_0_15 - PCz * sf_3.x_0_15; 
#else 
  x_3_15 = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_15 += TwoZetaInv * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
#ifdef REG_SF 
  x_1_16 = PAx * sf_2.x_0_16 - PCx * sf_3.x_0_16; 
#else 
  x_1_16 = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_16 = PAy * sf_2.x_0_16 - PCy * sf_3.x_0_16; 
#else 
  x_2_16 = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_16 += TwoZetaInv * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
#ifdef REG_SF 
  x_3_16 = PAz * sf_2.x_0_16 - PCz * sf_3.x_0_16; 
#else 
  x_3_16 = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_16 += TwoZetaInv * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
#ifdef REG_SF 
  x_1_17 = PAx * sf_2.x_0_17 - PCx * sf_3.x_0_17; 
#else 
  x_1_17 = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_17 += TwoZetaInv * 3.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
#ifdef REG_SF 
  x_2_17 = PAy * sf_2.x_0_17 - PCy * sf_3.x_0_17; 
#else 
  x_2_17 = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_17 = PAz * sf_2.x_0_17 - PCz * sf_3.x_0_17; 
#else 
  x_3_17 = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_1_18 = PAx * sf_2.x_0_18 - PCx * sf_3.x_0_18; 
#else 
  x_1_18 = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_18 = PAy * sf_2.x_0_18 - PCy * sf_3.x_0_18; 
#else 
  x_2_18 = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_18 += TwoZetaInv * 3.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
#ifdef REG_SF 
  x_3_18 = PAz * sf_2.x_0_18 - PCz * sf_3.x_0_18; 
#else 
  x_3_18 = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_1_19 = PAx * sf_2.x_0_19 - PCx * sf_3.x_0_19; 
#else 
  x_1_19 = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_19 = PAy * sf_2.x_0_19 - PCy * sf_3.x_0_19; 
#else 
  x_2_19 = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_19 = PAz * sf_2.x_0_19 - PCz * sf_3.x_0_19; 
#else 
  x_3_19 = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_19 += TwoZetaInv * 3.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
#else 
  QUICKDouble val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_10 - PCx * sf_3.x_0_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
  LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_10 - PCy * sf_3.x_0_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
  LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_10 - PCz * sf_3.x_0_10; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
  LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_11 - PCx * sf_3.x_0_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
  LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_11 - PCy * sf_3.x_0_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
  LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_11 - PCz * sf_3.x_0_11; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_12 - PCx * sf_3.x_0_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
  LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_12 - PCy * sf_3.x_0_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
  LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_12 - PCz * sf_3.x_0_12; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_13 - PCx * sf_3.x_0_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
  LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_13 - PCy * sf_3.x_0_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_13 - PCz * sf_3.x_0_13; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
  LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_14 - PCx * sf_3.x_0_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
  LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_14 - PCy * sf_3.x_0_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_14 - PCz * sf_3.x_0_14; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
  LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_15 - PCx * sf_3.x_0_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_15 - PCy * sf_3.x_0_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
  LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_15 - PCz * sf_3.x_0_15; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
  LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_16 - PCx * sf_3.x_0_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_16 - PCy * sf_3.x_0_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
  LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_16 - PCz * sf_3.x_0_16; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
  LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_17 - PCx * sf_3.x_0_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
  LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_17 - PCy * sf_3.x_0_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_17 - PCz * sf_3.x_0_17; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_18 - PCx * sf_3.x_0_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_18 - PCy * sf_3.x_0_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
  LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_18 - PCz * sf_3.x_0_18; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAx * sf_2.x_0_19 - PCx * sf_3.x_0_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2) - PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAy * sf_2.x_0_19 - PCy * sf_3.x_0_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2) - PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
  LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 2) = val; 
#ifdef REG_SF 
  val = PAz * sf_2.x_0_19 - PCz * sf_3.x_0_19; 
#else 
  val = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2) - PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
  val += TwoZetaInv * 3.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
  LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2) = val; 
#endif 

 } 

/* PF integral partial class - Part 1, m=2 */ 
__device__ __inline__ PFint_2_1::PFint_2_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_10 = PAx * sf_2.x_0_10 - PCx * sf_3.x_0_10; 
#else 
  x_1_10 = PAx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_10 += TwoZetaInv * 1.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
#ifdef REG_SF 
  x_2_10 = PAy * sf_2.x_0_10 - PCy * sf_3.x_0_10; 
#else 
  x_2_10 = PAy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_10 += TwoZetaInv * 1.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
#ifdef REG_SF 
  x_3_10 = PAz * sf_2.x_0_10 - PCz * sf_3.x_0_10; 
#else 
  x_3_10 = PAz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_10 += TwoZetaInv * 1.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 

 } 

/* PF integral partial class - Part 2, m=2 */ 
__device__ __inline__ PFint_2_2::PFint_2_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_11 = PAx * sf_2.x_0_11 - PCx * sf_3.x_0_11; 
#else 
  x_1_11 = PAx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_11 += TwoZetaInv * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
#ifdef REG_SF 
  x_2_11 = PAy * sf_2.x_0_11 - PCy * sf_3.x_0_11; 
#else 
  x_2_11 = PAy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_11 += TwoZetaInv * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
#ifdef REG_SF 
  x_3_11 = PAz * sf_2.x_0_11 - PCz * sf_3.x_0_11; 
#else 
  x_3_11 = PAz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* PF integral partial class - Part 3, m=2 */ 
__device__ __inline__ PFint_2_3::PFint_2_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_12 = PAx * sf_2.x_0_12 - PCx * sf_3.x_0_12; 
#else 
  x_1_12 = PAx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_12 += TwoZetaInv * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
#ifdef REG_SF 
  x_2_12 = PAy * sf_2.x_0_12 - PCy * sf_3.x_0_12; 
#else 
  x_2_12 = PAy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_12 += TwoZetaInv * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
#ifdef REG_SF 
  x_3_12 = PAz * sf_2.x_0_12 - PCz * sf_3.x_0_12; 
#else 
  x_3_12 = PAz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* PF integral partial class - Part 4, m=2 */ 
__device__ __inline__ PFint_2_4::PFint_2_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_13 = PAx * sf_2.x_0_13 - PCx * sf_3.x_0_13; 
#else 
  x_1_13 = PAx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_13 += TwoZetaInv * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
#ifdef REG_SF 
  x_2_13 = PAy * sf_2.x_0_13 - PCy * sf_3.x_0_13; 
#else 
  x_2_13 = PAy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_13 = PAz * sf_2.x_0_13 - PCz * sf_3.x_0_13; 
#else 
  x_3_13 = PAz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_13 += TwoZetaInv * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 

 } 

/* PF integral partial class - Part 5, m=2 */ 
__device__ __inline__ PFint_2_5::PFint_2_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_14 = PAx * sf_2.x_0_14 - PCx * sf_3.x_0_14; 
#else 
  x_1_14 = PAx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_14 += TwoZetaInv * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
#ifdef REG_SF 
  x_2_14 = PAy * sf_2.x_0_14 - PCy * sf_3.x_0_14; 
#else 
  x_2_14 = PAy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_14 = PAz * sf_2.x_0_14 - PCz * sf_3.x_0_14; 
#else 
  x_3_14 = PAz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_14 += TwoZetaInv * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 

 } 

/* PF integral partial class - Part 6, m=2 */ 
__device__ __inline__ PFint_2_6::PFint_2_6(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_15 = PAx * sf_2.x_0_15 - PCx * sf_3.x_0_15; 
#else 
  x_1_15 = PAx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_15 = PAy * sf_2.x_0_15 - PCy * sf_3.x_0_15; 
#else 
  x_2_15 = PAy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_15 += TwoZetaInv * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
#ifdef REG_SF 
  x_3_15 = PAz * sf_2.x_0_15 - PCz * sf_3.x_0_15; 
#else 
  x_3_15 = PAz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_15 += TwoZetaInv * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 

 } 

/* PF integral partial class - Part 7, m=2 */ 
__device__ __inline__ PFint_2_7::PFint_2_7(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_16 = PAx * sf_2.x_0_16 - PCx * sf_3.x_0_16; 
#else 
  x_1_16 = PAx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_16 = PAy * sf_2.x_0_16 - PCy * sf_3.x_0_16; 
#else 
  x_2_16 = PAy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_16 += TwoZetaInv * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
#ifdef REG_SF 
  x_3_16 = PAz * sf_2.x_0_16 - PCz * sf_3.x_0_16; 
#else 
  x_3_16 = PAz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_16 += TwoZetaInv * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 

 } 

/* PF integral partial class - Part 8, m=2 */ 
__device__ __inline__ PFint_2_8::PFint_2_8(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_17 = PAx * sf_2.x_0_17 - PCx * sf_3.x_0_17; 
#else 
  x_1_17 = PAx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
  x_1_17 += TwoZetaInv * 3.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
#ifdef REG_SF 
  x_2_17 = PAy * sf_2.x_0_17 - PCy * sf_3.x_0_17; 
#else 
  x_2_17 = PAy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_17 = PAz * sf_2.x_0_17 - PCz * sf_3.x_0_17; 
#else 
  x_3_17 = PAz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* PF integral partial class - Part 9, m=2 */ 
__device__ __inline__ PFint_2_9::PFint_2_9(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_18 = PAx * sf_2.x_0_18 - PCx * sf_3.x_0_18; 
#else 
  x_1_18 = PAx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_18 = PAy * sf_2.x_0_18 - PCy * sf_3.x_0_18; 
#else 
  x_2_18 = PAy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 
  x_2_18 += TwoZetaInv * 3.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
#ifdef REG_SF 
  x_3_18 = PAz * sf_2.x_0_18 - PCz * sf_3.x_0_18; 
#else 
  x_3_18 = PAz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 3); 
#endif 

 } 

/* PF integral partial class - Part 10, m=2 */ 
__device__ __inline__ PFint_2_10::PFint_2_10(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=3 

#ifdef REG_SF 
  x_1_19 = PAx * sf_2.x_0_19 - PCx * sf_3.x_0_19; 
#else 
  x_1_19 = PAx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)- PCx * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_2_19 = PAy * sf_2.x_0_19 - PCy * sf_3.x_0_19; 
#else 
  x_2_19 = PAy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)- PCy * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
#ifdef REG_SF 
  x_3_19 = PAz * sf_2.x_0_19 - PCz * sf_3.x_0_19; 
#else 
  x_3_19 = PAz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)- PCz * LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 3); 
#endif 
  x_3_19 += TwoZetaInv * 3.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 

 } 

/* FD true integral, m=0 */ 
__device__ __inline__ FDint_0::FDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DPint_0 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=0 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=0 
#ifndef USE_PARTIAL_FP 
  FPint_0 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
#endif 
  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
#ifndef USE_PARTIAL_FP 
  FPint_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
#endif 

#ifdef REG_FD 
  x_10_4 = PBx * fp_0.x_10_2 - PCx * fp_1.x_10_2; 
  x_10_4 += TwoZetaInv * 1.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
  x_10_5 = PBy * fp_0.x_10_3 - PCy * fp_1.x_10_3; 
  x_10_5 += TwoZetaInv * 1.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  x_10_6 = PBx * fp_0.x_10_3 - PCx * fp_1.x_10_3; 
  x_10_6 += TwoZetaInv * 1.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  x_10_7 = PBx * fp_0.x_10_1 - PCx * fp_1.x_10_1; 
  x_10_7 += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
  x_10_7 += TwoZetaInv * 1.000000 * (dp_0.x_5_1 - dp_1.x_5_1); 
  x_10_8 = PBy * fp_0.x_10_2 - PCy * fp_1.x_10_2; 
  x_10_8 += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
  x_10_8 += TwoZetaInv * 1.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
  x_10_9 = PBz * fp_0.x_10_3 - PCz * fp_1.x_10_3; 
  x_10_9 += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
  x_10_9 += TwoZetaInv * 1.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  x_11_4 = PBx * fp_0.x_11_2 - PCx * fp_1.x_11_2; 
  x_11_4 += TwoZetaInv * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
  x_11_5 = PBy * fp_0.x_11_3 - PCy * fp_1.x_11_3; 
  x_11_5 += TwoZetaInv * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  x_11_6 = PBx * fp_0.x_11_3 - PCx * fp_1.x_11_3; 
  x_11_6 += TwoZetaInv * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  x_11_7 = PBx * fp_0.x_11_1 - PCx * fp_1.x_11_1; 
  x_11_7 += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
  x_11_7 += TwoZetaInv * 2.000000 * (dp_0.x_4_1 - dp_1.x_4_1); 
  x_11_8 = PBy * fp_0.x_11_2 - PCy * fp_1.x_11_2; 
  x_11_8 += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
  x_11_8 += TwoZetaInv * 1.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
  x_11_9 = PBz * fp_0.x_11_3 - PCz * fp_1.x_11_3; 
  x_11_9 += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
  x_12_4 = PBx * fp_0.x_12_2 - PCx * fp_1.x_12_2; 
  x_12_4 += TwoZetaInv * 1.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
  x_12_5 = PBy * fp_0.x_12_3 - PCy * fp_1.x_12_3; 
  x_12_5 += TwoZetaInv * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  x_12_6 = PBx * fp_0.x_12_3 - PCx * fp_1.x_12_3; 
  x_12_6 += TwoZetaInv * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  x_12_7 = PBx * fp_0.x_12_1 - PCx * fp_1.x_12_1; 
  x_12_7 += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
  x_12_7 += TwoZetaInv * 1.000000 * (dp_0.x_8_1 - dp_1.x_8_1); 
  x_12_8 = PBy * fp_0.x_12_2 - PCy * fp_1.x_12_2; 
  x_12_8 += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
  x_12_8 += TwoZetaInv * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
  x_12_9 = PBz * fp_0.x_12_3 - PCz * fp_1.x_12_3; 
  x_12_9 += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
  x_13_4 = PBx * fp_0.x_13_2 - PCx * fp_1.x_13_2; 
  x_13_4 += TwoZetaInv * 2.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
  x_13_5 = PBy * fp_0.x_13_3 - PCy * fp_1.x_13_3; 
  x_13_6 = PBx * fp_0.x_13_3 - PCx * fp_1.x_13_3; 
  x_13_6 += TwoZetaInv * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  x_13_7 = PBx * fp_0.x_13_1 - PCx * fp_1.x_13_1; 
  x_13_7 += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
  x_13_7 += TwoZetaInv * 2.000000 * (dp_0.x_6_1 - dp_1.x_6_1); 
  x_13_8 = PBy * fp_0.x_13_2 - PCy * fp_1.x_13_2; 
  x_13_8 += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
  x_13_9 = PBz * fp_0.x_13_3 - PCz * fp_1.x_13_3; 
  x_13_9 += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
  x_13_9 += TwoZetaInv * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  x_14_4 = PBx * fp_0.x_14_2 - PCx * fp_1.x_14_2; 
  x_14_4 += TwoZetaInv * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
  x_14_5 = PBy * fp_0.x_14_3 - PCy * fp_1.x_14_3; 
  x_14_6 = PBx * fp_0.x_14_3 - PCx * fp_1.x_14_3; 
  x_14_6 += TwoZetaInv * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  x_14_7 = PBx * fp_0.x_14_1 - PCx * fp_1.x_14_1; 
  x_14_7 += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
  x_14_7 += TwoZetaInv * 1.000000 * (dp_0.x_9_1 - dp_1.x_9_1); 
  x_14_8 = PBy * fp_0.x_14_2 - PCy * fp_1.x_14_2; 
  x_14_8 += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
  x_14_9 = PBz * fp_0.x_14_3 - PCz * fp_1.x_14_3; 
  x_14_9 += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
  x_14_9 += TwoZetaInv * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  x_15_4 = PBx * fp_0.x_15_2 - PCx * fp_1.x_15_2; 
  x_15_5 = PBy * fp_0.x_15_3 - PCy * fp_1.x_15_3; 
  x_15_5 += TwoZetaInv * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  x_15_6 = PBx * fp_0.x_15_3 - PCx * fp_1.x_15_3; 
  x_15_7 = PBx * fp_0.x_15_1 - PCx * fp_1.x_15_1; 
  x_15_7 += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
  x_15_8 = PBy * fp_0.x_15_2 - PCy * fp_1.x_15_2; 
  x_15_8 += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
  x_15_8 += TwoZetaInv * 2.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
  x_15_9 = PBz * fp_0.x_15_3 - PCz * fp_1.x_15_3; 
  x_15_9 += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
  x_15_9 += TwoZetaInv * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  x_16_4 = PBx * fp_0.x_16_2 - PCx * fp_1.x_16_2; 
  x_16_5 = PBy * fp_0.x_16_3 - PCy * fp_1.x_16_3; 
  x_16_5 += TwoZetaInv * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  x_16_6 = PBx * fp_0.x_16_3 - PCx * fp_1.x_16_3; 
  x_16_7 = PBx * fp_0.x_16_1 - PCx * fp_1.x_16_1; 
  x_16_7 += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
  x_16_8 = PBy * fp_0.x_16_2 - PCy * fp_1.x_16_2; 
  x_16_8 += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
  x_16_8 += TwoZetaInv * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
  x_16_9 = PBz * fp_0.x_16_3 - PCz * fp_1.x_16_3; 
  x_16_9 += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
  x_16_9 += TwoZetaInv * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  x_17_4 = PBx * fp_0.x_17_2 - PCx * fp_1.x_17_2; 
  x_17_4 += TwoZetaInv * 3.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
  x_17_5 = PBy * fp_0.x_17_3 - PCy * fp_1.x_17_3; 
  x_17_6 = PBx * fp_0.x_17_3 - PCx * fp_1.x_17_3; 
  x_17_6 += TwoZetaInv * 3.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  x_17_7 = PBx * fp_0.x_17_1 - PCx * fp_1.x_17_1; 
  x_17_7 += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
  x_17_7 += TwoZetaInv * 3.000000 * (dp_0.x_7_1 - dp_1.x_7_1); 
  x_17_8 = PBy * fp_0.x_17_2 - PCy * fp_1.x_17_2; 
  x_17_8 += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
  x_17_9 = PBz * fp_0.x_17_3 - PCz * fp_1.x_17_3; 
  x_17_9 += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
  x_18_4 = PBx * fp_0.x_18_2 - PCx * fp_1.x_18_2; 
  x_18_5 = PBy * fp_0.x_18_3 - PCy * fp_1.x_18_3; 
  x_18_5 += TwoZetaInv * 3.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  x_18_6 = PBx * fp_0.x_18_3 - PCx * fp_1.x_18_3; 
  x_18_7 = PBx * fp_0.x_18_1 - PCx * fp_1.x_18_1; 
  x_18_7 += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
  x_18_8 = PBy * fp_0.x_18_2 - PCy * fp_1.x_18_2; 
  x_18_8 += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
  x_18_8 += TwoZetaInv * 3.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
  x_18_9 = PBz * fp_0.x_18_3 - PCz * fp_1.x_18_3; 
  x_18_9 += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
  x_19_4 = PBx * fp_0.x_19_2 - PCx * fp_1.x_19_2; 
  x_19_5 = PBy * fp_0.x_19_3 - PCy * fp_1.x_19_3; 
  x_19_6 = PBx * fp_0.x_19_3 - PCx * fp_1.x_19_3; 
  x_19_7 = PBx * fp_0.x_19_1 - PCx * fp_1.x_19_1; 
  x_19_7 += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
  x_19_8 = PBy * fp_0.x_19_2 - PCy * fp_1.x_19_2; 
  x_19_8 += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
  x_19_9 = PBz * fp_0.x_19_3 - PCz * fp_1.x_19_3; 
  x_19_9 += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
  x_19_9 += TwoZetaInv * 3.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
#else 
#ifdef USE_PARTIAL_FP 
  { 
    FPint_0_1 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_10_2 - PCx * fp_1.x_10_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
    LOCSTOREFULL(store, 10, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_10_3 - PCy * fp_1.x_10_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
    LOCSTOREFULL(store, 10, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_10_3 - PCx * fp_1.x_10_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
    LOCSTOREFULL(store, 10, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_10_1 - PCx * fp_1.x_10_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_5_1 - dp_1.x_5_1); 
    LOCSTOREFULL(store, 10, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_10_2 - PCy * fp_1.x_10_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
    LOCSTOREFULL(store, 10, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_10_3 - PCz * fp_1.x_10_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
    LOCSTOREFULL(store, 10, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_2 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_2 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_11_2 - PCx * fp_1.x_11_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
    LOCSTOREFULL(store, 11, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_11_3 - PCy * fp_1.x_11_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
    LOCSTOREFULL(store, 11, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_11_3 - PCx * fp_1.x_11_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
    LOCSTOREFULL(store, 11, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_11_1 - PCx * fp_1.x_11_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_4_1 - dp_1.x_4_1); 
    LOCSTOREFULL(store, 11, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_11_2 - PCy * fp_1.x_11_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
    LOCSTOREFULL(store, 11, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_11_3 - PCz * fp_1.x_11_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 11, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_3 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_3 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_12_2 - PCx * fp_1.x_12_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
    LOCSTOREFULL(store, 12, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_12_3 - PCy * fp_1.x_12_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
    LOCSTOREFULL(store, 12, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_12_3 - PCx * fp_1.x_12_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
    LOCSTOREFULL(store, 12, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_12_1 - PCx * fp_1.x_12_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_8_1 - dp_1.x_8_1); 
    LOCSTOREFULL(store, 12, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_12_2 - PCy * fp_1.x_12_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
    LOCSTOREFULL(store, 12, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_12_3 - PCz * fp_1.x_12_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 12, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_4 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_4 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_13_2 - PCx * fp_1.x_13_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
    LOCSTOREFULL(store, 13, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_13_3 - PCy * fp_1.x_13_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 13, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_13_3 - PCx * fp_1.x_13_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
    LOCSTOREFULL(store, 13, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_13_1 - PCx * fp_1.x_13_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_6_1 - dp_1.x_6_1); 
    LOCSTOREFULL(store, 13, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_13_2 - PCy * fp_1.x_13_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 13, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_13_3 - PCz * fp_1.x_13_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
    LOCSTOREFULL(store, 13, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_5 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_5 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_14_2 - PCx * fp_1.x_14_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
    LOCSTOREFULL(store, 14, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_14_3 - PCy * fp_1.x_14_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 14, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_14_3 - PCx * fp_1.x_14_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
    LOCSTOREFULL(store, 14, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_14_1 - PCx * fp_1.x_14_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_9_1 - dp_1.x_9_1); 
    LOCSTOREFULL(store, 14, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_14_2 - PCy * fp_1.x_14_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 14, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_14_3 - PCz * fp_1.x_14_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
    LOCSTOREFULL(store, 14, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_6 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_6 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_15_2 - PCx * fp_1.x_15_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 15, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_15_3 - PCy * fp_1.x_15_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
    LOCSTOREFULL(store, 15, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_15_3 - PCx * fp_1.x_15_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 15, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_15_1 - PCx * fp_1.x_15_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 15, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_15_2 - PCy * fp_1.x_15_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
    LOCSTOREFULL(store, 15, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_15_3 - PCz * fp_1.x_15_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
    LOCSTOREFULL(store, 15, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_7 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_7 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_16_2 - PCx * fp_1.x_16_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 16, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_16_3 - PCy * fp_1.x_16_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
    LOCSTOREFULL(store, 16, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_16_3 - PCx * fp_1.x_16_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 16, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_16_1 - PCx * fp_1.x_16_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 16, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_16_2 - PCy * fp_1.x_16_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
    LOCSTOREFULL(store, 16, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_16_3 - PCz * fp_1.x_16_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
    LOCSTOREFULL(store, 16, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_8 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_8 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_17_2 - PCx * fp_1.x_17_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
    LOCSTOREFULL(store, 17, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_17_3 - PCy * fp_1.x_17_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 17, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_17_3 - PCx * fp_1.x_17_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
    LOCSTOREFULL(store, 17, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_17_1 - PCx * fp_1.x_17_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_0.x_7_1 - dp_1.x_7_1); 
    LOCSTOREFULL(store, 17, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_17_2 - PCy * fp_1.x_17_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 17, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_17_3 - PCz * fp_1.x_17_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 17, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_9 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_9 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_18_2 - PCx * fp_1.x_18_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 18, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_18_3 - PCy * fp_1.x_18_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
    LOCSTOREFULL(store, 18, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_18_3 - PCx * fp_1.x_18_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 18, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_18_1 - PCx * fp_1.x_18_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 18, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_18_2 - PCy * fp_1.x_18_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
    LOCSTOREFULL(store, 18, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_18_3 - PCz * fp_1.x_18_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 18, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    FPint_0_10 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
    FPint_1_10 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_0.x_19_2 - PCx * fp_1.x_19_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 19, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_19_3 - PCy * fp_1.x_19_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 19, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_19_3 - PCx * fp_1.x_19_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 19, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBx * fp_0.x_19_1 - PCx * fp_1.x_19_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 19, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBy * fp_0.x_19_2 - PCy * fp_1.x_19_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 19, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
    val = PBz * fp_0.x_19_3 - PCz * fp_1.x_19_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
    LOCSTOREFULL(store, 19, 9, STOREDIM, STOREDIM, 0) = val; 
  } 

#else 
  QUICKDouble val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_10_2 - PCx * fp_1.x_10_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
  LOCSTOREFULL(store, 10, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_10_3 - PCy * fp_1.x_10_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  LOCSTOREFULL(store, 10, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_10_3 - PCx * fp_1.x_10_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  LOCSTOREFULL(store, 10, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_10_1 - PCx * fp_1.x_10_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_5_1 - dp_1.x_5_1); 
  LOCSTOREFULL(store, 10, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_10_2 - PCy * fp_1.x_10_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
  LOCSTOREFULL(store, 10, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_10_3 - PCz * fp_1.x_10_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_10_0 - fs_1.x_10_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  LOCSTOREFULL(store, 10, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_11_2 - PCx * fp_1.x_11_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
  LOCSTOREFULL(store, 11, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_11_3 - PCy * fp_1.x_11_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  LOCSTOREFULL(store, 11, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_11_3 - PCx * fp_1.x_11_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  LOCSTOREFULL(store, 11, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_11_1 - PCx * fp_1.x_11_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_4_1 - dp_1.x_4_1); 
  LOCSTOREFULL(store, 11, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_11_2 - PCy * fp_1.x_11_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
  LOCSTOREFULL(store, 11, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_11_3 - PCz * fp_1.x_11_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_11_0 - fs_1.x_11_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 11, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_12_2 - PCx * fp_1.x_12_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
  LOCSTOREFULL(store, 12, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_12_3 - PCy * fp_1.x_12_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  LOCSTOREFULL(store, 12, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_12_3 - PCx * fp_1.x_12_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  LOCSTOREFULL(store, 12, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_12_1 - PCx * fp_1.x_12_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_8_1 - dp_1.x_8_1); 
  LOCSTOREFULL(store, 12, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_12_2 - PCy * fp_1.x_12_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
  LOCSTOREFULL(store, 12, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_12_3 - PCz * fp_1.x_12_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_12_0 - fs_1.x_12_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 12, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_13_2 - PCx * fp_1.x_13_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
  LOCSTOREFULL(store, 13, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_13_3 - PCy * fp_1.x_13_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 13, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_13_3 - PCx * fp_1.x_13_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  LOCSTOREFULL(store, 13, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_13_1 - PCx * fp_1.x_13_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_6_1 - dp_1.x_6_1); 
  LOCSTOREFULL(store, 13, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_13_2 - PCy * fp_1.x_13_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 13, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_13_3 - PCz * fp_1.x_13_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_13_0 - fs_1.x_13_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  LOCSTOREFULL(store, 13, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_14_2 - PCx * fp_1.x_14_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
  LOCSTOREFULL(store, 14, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_14_3 - PCy * fp_1.x_14_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 14, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_14_3 - PCx * fp_1.x_14_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  LOCSTOREFULL(store, 14, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_14_1 - PCx * fp_1.x_14_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_9_1 - dp_1.x_9_1); 
  LOCSTOREFULL(store, 14, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_14_2 - PCy * fp_1.x_14_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 14, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_14_3 - PCz * fp_1.x_14_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_14_0 - fs_1.x_14_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  LOCSTOREFULL(store, 14, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_15_2 - PCx * fp_1.x_15_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 15, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_15_3 - PCy * fp_1.x_15_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  LOCSTOREFULL(store, 15, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_15_3 - PCx * fp_1.x_15_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 15, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_15_1 - PCx * fp_1.x_15_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 15, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_15_2 - PCy * fp_1.x_15_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
  LOCSTOREFULL(store, 15, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_15_3 - PCz * fp_1.x_15_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_15_0 - fs_1.x_15_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  LOCSTOREFULL(store, 15, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_16_2 - PCx * fp_1.x_16_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 16, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_16_3 - PCy * fp_1.x_16_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  LOCSTOREFULL(store, 16, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_16_3 - PCx * fp_1.x_16_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 16, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_16_1 - PCx * fp_1.x_16_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 16, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_16_2 - PCy * fp_1.x_16_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
  LOCSTOREFULL(store, 16, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_16_3 - PCz * fp_1.x_16_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_16_0 - fs_1.x_16_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  LOCSTOREFULL(store, 16, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_17_2 - PCx * fp_1.x_17_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
  LOCSTOREFULL(store, 17, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_17_3 - PCy * fp_1.x_17_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 17, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_17_3 - PCx * fp_1.x_17_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  LOCSTOREFULL(store, 17, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_17_1 - PCx * fp_1.x_17_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_0.x_7_1 - dp_1.x_7_1); 
  LOCSTOREFULL(store, 17, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_17_2 - PCy * fp_1.x_17_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 17, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_17_3 - PCz * fp_1.x_17_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_17_0 - fs_1.x_17_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 17, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_18_2 - PCx * fp_1.x_18_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 18, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_18_3 - PCy * fp_1.x_18_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  LOCSTOREFULL(store, 18, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_18_3 - PCx * fp_1.x_18_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 18, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_18_1 - PCx * fp_1.x_18_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 18, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_18_2 - PCy * fp_1.x_18_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
  LOCSTOREFULL(store, 18, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_18_3 - PCz * fp_1.x_18_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_18_0 - fs_1.x_18_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 18, 9, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_19_2 - PCx * fp_1.x_19_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 19, 4, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_19_3 - PCy * fp_1.x_19_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 19, 5, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_19_3 - PCx * fp_1.x_19_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 19, 6, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBx * fp_0.x_19_1 - PCx * fp_1.x_19_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 19, 7, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBy * fp_0.x_19_2 - PCy * fp_1.x_19_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 19, 8, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_FP 
  val = PBz * fp_0.x_19_3 - PCz * fp_1.x_19_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_0.x_19_0 - fs_1.x_19_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  LOCSTOREFULL(store, 19, 9, STOREDIM, STOREDIM, 0) = val; 
#endif 
#endif 

 } 

/* FD auxilary integral, m=1 */ 
__device__ __inline__ FDint_1::FDint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=1 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=1 
#ifndef USE_PARTIAL_FP 
  FPint_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
#endif 
  DPint_2 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|p] for m=2 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|s] for m=2 
#ifndef USE_PARTIAL_FP 
  FPint_2 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 
#endif 

#ifdef REG_FD 
  x_10_4 = PBx * fp_1.x_10_2 - PCx * fp_2.x_10_2; 
  x_10_4 += TwoZetaInv * 1.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
  x_10_5 = PBy * fp_1.x_10_3 - PCy * fp_2.x_10_3; 
  x_10_5 += TwoZetaInv * 1.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  x_10_6 = PBx * fp_1.x_10_3 - PCx * fp_2.x_10_3; 
  x_10_6 += TwoZetaInv * 1.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  x_10_7 = PBx * fp_1.x_10_1 - PCx * fp_2.x_10_1; 
  x_10_7 += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
  x_10_7 += TwoZetaInv * 1.000000 * (dp_1.x_5_1 - dp_2.x_5_1); 
  x_10_8 = PBy * fp_1.x_10_2 - PCy * fp_2.x_10_2; 
  x_10_8 += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
  x_10_8 += TwoZetaInv * 1.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
  x_10_9 = PBz * fp_1.x_10_3 - PCz * fp_2.x_10_3; 
  x_10_9 += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
  x_10_9 += TwoZetaInv * 1.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  x_11_4 = PBx * fp_1.x_11_2 - PCx * fp_2.x_11_2; 
  x_11_4 += TwoZetaInv * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
  x_11_5 = PBy * fp_1.x_11_3 - PCy * fp_2.x_11_3; 
  x_11_5 += TwoZetaInv * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  x_11_6 = PBx * fp_1.x_11_3 - PCx * fp_2.x_11_3; 
  x_11_6 += TwoZetaInv * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  x_11_7 = PBx * fp_1.x_11_1 - PCx * fp_2.x_11_1; 
  x_11_7 += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
  x_11_7 += TwoZetaInv * 2.000000 * (dp_1.x_4_1 - dp_2.x_4_1); 
  x_11_8 = PBy * fp_1.x_11_2 - PCy * fp_2.x_11_2; 
  x_11_8 += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
  x_11_8 += TwoZetaInv * 1.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
  x_11_9 = PBz * fp_1.x_11_3 - PCz * fp_2.x_11_3; 
  x_11_9 += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
  x_12_4 = PBx * fp_1.x_12_2 - PCx * fp_2.x_12_2; 
  x_12_4 += TwoZetaInv * 1.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
  x_12_5 = PBy * fp_1.x_12_3 - PCy * fp_2.x_12_3; 
  x_12_5 += TwoZetaInv * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  x_12_6 = PBx * fp_1.x_12_3 - PCx * fp_2.x_12_3; 
  x_12_6 += TwoZetaInv * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  x_12_7 = PBx * fp_1.x_12_1 - PCx * fp_2.x_12_1; 
  x_12_7 += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
  x_12_7 += TwoZetaInv * 1.000000 * (dp_1.x_8_1 - dp_2.x_8_1); 
  x_12_8 = PBy * fp_1.x_12_2 - PCy * fp_2.x_12_2; 
  x_12_8 += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
  x_12_8 += TwoZetaInv * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
  x_12_9 = PBz * fp_1.x_12_3 - PCz * fp_2.x_12_3; 
  x_12_9 += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
  x_13_4 = PBx * fp_1.x_13_2 - PCx * fp_2.x_13_2; 
  x_13_4 += TwoZetaInv * 2.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
  x_13_5 = PBy * fp_1.x_13_3 - PCy * fp_2.x_13_3; 
  x_13_6 = PBx * fp_1.x_13_3 - PCx * fp_2.x_13_3; 
  x_13_6 += TwoZetaInv * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  x_13_7 = PBx * fp_1.x_13_1 - PCx * fp_2.x_13_1; 
  x_13_7 += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
  x_13_7 += TwoZetaInv * 2.000000 * (dp_1.x_6_1 - dp_2.x_6_1); 
  x_13_8 = PBy * fp_1.x_13_2 - PCy * fp_2.x_13_2; 
  x_13_8 += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
  x_13_9 = PBz * fp_1.x_13_3 - PCz * fp_2.x_13_3; 
  x_13_9 += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
  x_13_9 += TwoZetaInv * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  x_14_4 = PBx * fp_1.x_14_2 - PCx * fp_2.x_14_2; 
  x_14_4 += TwoZetaInv * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
  x_14_5 = PBy * fp_1.x_14_3 - PCy * fp_2.x_14_3; 
  x_14_6 = PBx * fp_1.x_14_3 - PCx * fp_2.x_14_3; 
  x_14_6 += TwoZetaInv * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  x_14_7 = PBx * fp_1.x_14_1 - PCx * fp_2.x_14_1; 
  x_14_7 += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
  x_14_7 += TwoZetaInv * 1.000000 * (dp_1.x_9_1 - dp_2.x_9_1); 
  x_14_8 = PBy * fp_1.x_14_2 - PCy * fp_2.x_14_2; 
  x_14_8 += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
  x_14_9 = PBz * fp_1.x_14_3 - PCz * fp_2.x_14_3; 
  x_14_9 += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
  x_14_9 += TwoZetaInv * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  x_15_4 = PBx * fp_1.x_15_2 - PCx * fp_2.x_15_2; 
  x_15_5 = PBy * fp_1.x_15_3 - PCy * fp_2.x_15_3; 
  x_15_5 += TwoZetaInv * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  x_15_6 = PBx * fp_1.x_15_3 - PCx * fp_2.x_15_3; 
  x_15_7 = PBx * fp_1.x_15_1 - PCx * fp_2.x_15_1; 
  x_15_7 += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
  x_15_8 = PBy * fp_1.x_15_2 - PCy * fp_2.x_15_2; 
  x_15_8 += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
  x_15_8 += TwoZetaInv * 2.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
  x_15_9 = PBz * fp_1.x_15_3 - PCz * fp_2.x_15_3; 
  x_15_9 += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
  x_15_9 += TwoZetaInv * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  x_16_4 = PBx * fp_1.x_16_2 - PCx * fp_2.x_16_2; 
  x_16_5 = PBy * fp_1.x_16_3 - PCy * fp_2.x_16_3; 
  x_16_5 += TwoZetaInv * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  x_16_6 = PBx * fp_1.x_16_3 - PCx * fp_2.x_16_3; 
  x_16_7 = PBx * fp_1.x_16_1 - PCx * fp_2.x_16_1; 
  x_16_7 += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
  x_16_8 = PBy * fp_1.x_16_2 - PCy * fp_2.x_16_2; 
  x_16_8 += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
  x_16_8 += TwoZetaInv * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
  x_16_9 = PBz * fp_1.x_16_3 - PCz * fp_2.x_16_3; 
  x_16_9 += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
  x_16_9 += TwoZetaInv * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  x_17_4 = PBx * fp_1.x_17_2 - PCx * fp_2.x_17_2; 
  x_17_4 += TwoZetaInv * 3.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
  x_17_5 = PBy * fp_1.x_17_3 - PCy * fp_2.x_17_3; 
  x_17_6 = PBx * fp_1.x_17_3 - PCx * fp_2.x_17_3; 
  x_17_6 += TwoZetaInv * 3.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  x_17_7 = PBx * fp_1.x_17_1 - PCx * fp_2.x_17_1; 
  x_17_7 += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
  x_17_7 += TwoZetaInv * 3.000000 * (dp_1.x_7_1 - dp_2.x_7_1); 
  x_17_8 = PBy * fp_1.x_17_2 - PCy * fp_2.x_17_2; 
  x_17_8 += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
  x_17_9 = PBz * fp_1.x_17_3 - PCz * fp_2.x_17_3; 
  x_17_9 += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
  x_18_4 = PBx * fp_1.x_18_2 - PCx * fp_2.x_18_2; 
  x_18_5 = PBy * fp_1.x_18_3 - PCy * fp_2.x_18_3; 
  x_18_5 += TwoZetaInv * 3.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  x_18_6 = PBx * fp_1.x_18_3 - PCx * fp_2.x_18_3; 
  x_18_7 = PBx * fp_1.x_18_1 - PCx * fp_2.x_18_1; 
  x_18_7 += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
  x_18_8 = PBy * fp_1.x_18_2 - PCy * fp_2.x_18_2; 
  x_18_8 += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
  x_18_8 += TwoZetaInv * 3.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
  x_18_9 = PBz * fp_1.x_18_3 - PCz * fp_2.x_18_3; 
  x_18_9 += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
  x_19_4 = PBx * fp_1.x_19_2 - PCx * fp_2.x_19_2; 
  x_19_5 = PBy * fp_1.x_19_3 - PCy * fp_2.x_19_3; 
  x_19_6 = PBx * fp_1.x_19_3 - PCx * fp_2.x_19_3; 
  x_19_7 = PBx * fp_1.x_19_1 - PCx * fp_2.x_19_1; 
  x_19_7 += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
  x_19_8 = PBy * fp_1.x_19_2 - PCy * fp_2.x_19_2; 
  x_19_8 += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
  x_19_9 = PBz * fp_1.x_19_3 - PCz * fp_2.x_19_3; 
  x_19_9 += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
  x_19_9 += TwoZetaInv * 3.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
#else 
#ifdef USE_PARTIAL_FP 
  { 
    FPint_1_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_1 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_10_2 - PCx * fp_2.x_10_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
    LOCSTOREFULL(store, 10, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_10_3 - PCy * fp_2.x_10_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
    LOCSTOREFULL(store, 10, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_10_3 - PCx * fp_2.x_10_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
    LOCSTOREFULL(store, 10, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_10_1 - PCx * fp_2.x_10_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_5_1 - dp_2.x_5_1); 
    LOCSTOREFULL(store, 10, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_10_2 - PCy * fp_2.x_10_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
    LOCSTOREFULL(store, 10, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_10_3 - PCz * fp_2.x_10_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
    LOCSTOREFULL(store, 10, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_2 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_2 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_11_2 - PCx * fp_2.x_11_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
    LOCSTOREFULL(store, 11, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_11_3 - PCy * fp_2.x_11_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
    LOCSTOREFULL(store, 11, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_11_3 - PCx * fp_2.x_11_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
    LOCSTOREFULL(store, 11, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_11_1 - PCx * fp_2.x_11_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_4_1 - dp_2.x_4_1); 
    LOCSTOREFULL(store, 11, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_11_2 - PCy * fp_2.x_11_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
    LOCSTOREFULL(store, 11, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_11_3 - PCz * fp_2.x_11_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 11, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_3 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_3 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_12_2 - PCx * fp_2.x_12_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
    LOCSTOREFULL(store, 12, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_12_3 - PCy * fp_2.x_12_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
    LOCSTOREFULL(store, 12, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_12_3 - PCx * fp_2.x_12_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
    LOCSTOREFULL(store, 12, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_12_1 - PCx * fp_2.x_12_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_8_1 - dp_2.x_8_1); 
    LOCSTOREFULL(store, 12, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_12_2 - PCy * fp_2.x_12_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
    LOCSTOREFULL(store, 12, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_12_3 - PCz * fp_2.x_12_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 12, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_4 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_4 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_13_2 - PCx * fp_2.x_13_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
    LOCSTOREFULL(store, 13, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_13_3 - PCy * fp_2.x_13_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 13, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_13_3 - PCx * fp_2.x_13_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
    LOCSTOREFULL(store, 13, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_13_1 - PCx * fp_2.x_13_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_6_1 - dp_2.x_6_1); 
    LOCSTOREFULL(store, 13, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_13_2 - PCy * fp_2.x_13_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 13, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_13_3 - PCz * fp_2.x_13_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
    LOCSTOREFULL(store, 13, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_5 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_5 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_14_2 - PCx * fp_2.x_14_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
    LOCSTOREFULL(store, 14, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_14_3 - PCy * fp_2.x_14_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 14, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_14_3 - PCx * fp_2.x_14_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
    LOCSTOREFULL(store, 14, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_14_1 - PCx * fp_2.x_14_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_9_1 - dp_2.x_9_1); 
    LOCSTOREFULL(store, 14, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_14_2 - PCy * fp_2.x_14_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 14, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_14_3 - PCz * fp_2.x_14_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
    LOCSTOREFULL(store, 14, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_6 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_6 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_15_2 - PCx * fp_2.x_15_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 15, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_15_3 - PCy * fp_2.x_15_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
    LOCSTOREFULL(store, 15, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_15_3 - PCx * fp_2.x_15_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 15, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_15_1 - PCx * fp_2.x_15_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 15, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_15_2 - PCy * fp_2.x_15_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
    LOCSTOREFULL(store, 15, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_15_3 - PCz * fp_2.x_15_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
    LOCSTOREFULL(store, 15, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_7 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_7 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_16_2 - PCx * fp_2.x_16_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 16, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_16_3 - PCy * fp_2.x_16_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
    LOCSTOREFULL(store, 16, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_16_3 - PCx * fp_2.x_16_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 16, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_16_1 - PCx * fp_2.x_16_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 16, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_16_2 - PCy * fp_2.x_16_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
    LOCSTOREFULL(store, 16, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_16_3 - PCz * fp_2.x_16_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
    LOCSTOREFULL(store, 16, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_8 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_8 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_17_2 - PCx * fp_2.x_17_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
    LOCSTOREFULL(store, 17, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_17_3 - PCy * fp_2.x_17_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 17, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_17_3 - PCx * fp_2.x_17_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
    LOCSTOREFULL(store, 17, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_17_1 - PCx * fp_2.x_17_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_1.x_7_1 - dp_2.x_7_1); 
    LOCSTOREFULL(store, 17, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_17_2 - PCy * fp_2.x_17_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 17, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_17_3 - PCz * fp_2.x_17_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 17, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_9 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_9 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_18_2 - PCx * fp_2.x_18_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 18, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_18_3 - PCy * fp_2.x_18_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
    LOCSTOREFULL(store, 18, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_18_3 - PCx * fp_2.x_18_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 18, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_18_1 - PCx * fp_2.x_18_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 18, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_18_2 - PCy * fp_2.x_18_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
    LOCSTOREFULL(store, 18, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_18_3 - PCz * fp_2.x_18_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 18, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    FPint_1_10 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
    FPint_2_10 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=2 

    QUICKDouble val; 

#ifdef REG_FP 
    val = PBx * fp_1.x_19_2 - PCx * fp_2.x_19_2; 
#else 
    val = PBx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 19, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_19_3 - PCy * fp_2.x_19_3; 
#else 
    val = PBy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 19, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_19_3 - PCx * fp_2.x_19_3; 
#else 
    val = PBx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 19, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBx * fp_1.x_19_1 - PCx * fp_2.x_19_1; 
#else 
    val = PBx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 19, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBy * fp_1.x_19_2 - PCy * fp_2.x_19_2; 
#else 
    val = PBy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 19, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
    val = PBz * fp_1.x_19_3 - PCz * fp_2.x_19_3; 
#else 
    val = PBz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
    val += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 3.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
    LOCSTOREFULL(store, 19, 9, STOREDIM, STOREDIM, 1) = val; 
  } 

#else 
  QUICKDouble val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_10_2 - PCx * fp_2.x_10_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
  LOCSTOREFULL(store, 10, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_10_3 - PCy * fp_2.x_10_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  LOCSTOREFULL(store, 10, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_10_3 - PCx * fp_2.x_10_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  LOCSTOREFULL(store, 10, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_10_1 - PCx * fp_2.x_10_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 10, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_5_1 - dp_2.x_5_1); 
  LOCSTOREFULL(store, 10, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_10_2 - PCy * fp_2.x_10_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 10, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
  LOCSTOREFULL(store, 10, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_10_3 - PCz * fp_2.x_10_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 10, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_10_0 - fs_2.x_10_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 10, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  LOCSTOREFULL(store, 10, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_11_2 - PCx * fp_2.x_11_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
  LOCSTOREFULL(store, 11, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_11_3 - PCy * fp_2.x_11_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  LOCSTOREFULL(store, 11, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_11_3 - PCx * fp_2.x_11_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  LOCSTOREFULL(store, 11, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_11_1 - PCx * fp_2.x_11_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 11, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_4_1 - dp_2.x_4_1); 
  LOCSTOREFULL(store, 11, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_11_2 - PCy * fp_2.x_11_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 11, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
  LOCSTOREFULL(store, 11, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_11_3 - PCz * fp_2.x_11_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 11, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_11_0 - fs_2.x_11_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 11, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 11, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_12_2 - PCx * fp_2.x_12_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
  LOCSTOREFULL(store, 12, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_12_3 - PCy * fp_2.x_12_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  LOCSTOREFULL(store, 12, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_12_3 - PCx * fp_2.x_12_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  LOCSTOREFULL(store, 12, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_12_1 - PCx * fp_2.x_12_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 12, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_8_1 - dp_2.x_8_1); 
  LOCSTOREFULL(store, 12, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_12_2 - PCy * fp_2.x_12_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 12, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
  LOCSTOREFULL(store, 12, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_12_3 - PCz * fp_2.x_12_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 12, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_12_0 - fs_2.x_12_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 12, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 12, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_13_2 - PCx * fp_2.x_13_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
  LOCSTOREFULL(store, 13, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_13_3 - PCy * fp_2.x_13_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 13, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_13_3 - PCx * fp_2.x_13_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  LOCSTOREFULL(store, 13, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_13_1 - PCx * fp_2.x_13_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 13, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_6_1 - dp_2.x_6_1); 
  LOCSTOREFULL(store, 13, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_13_2 - PCy * fp_2.x_13_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 13, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 13, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_13_3 - PCz * fp_2.x_13_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 13, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_13_0 - fs_2.x_13_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 13, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  LOCSTOREFULL(store, 13, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_14_2 - PCx * fp_2.x_14_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
  LOCSTOREFULL(store, 14, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_14_3 - PCy * fp_2.x_14_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 14, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_14_3 - PCx * fp_2.x_14_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  LOCSTOREFULL(store, 14, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_14_1 - PCx * fp_2.x_14_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 14, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_9_1 - dp_2.x_9_1); 
  LOCSTOREFULL(store, 14, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_14_2 - PCy * fp_2.x_14_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 14, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 14, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_14_3 - PCz * fp_2.x_14_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 14, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_14_0 - fs_2.x_14_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 14, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  LOCSTOREFULL(store, 14, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_15_2 - PCx * fp_2.x_15_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 15, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_15_3 - PCy * fp_2.x_15_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  LOCSTOREFULL(store, 15, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_15_3 - PCx * fp_2.x_15_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 15, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_15_1 - PCx * fp_2.x_15_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 15, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 15, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_15_2 - PCy * fp_2.x_15_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 15, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
  LOCSTOREFULL(store, 15, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_15_3 - PCz * fp_2.x_15_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 15, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_15_0 - fs_2.x_15_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 15, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  LOCSTOREFULL(store, 15, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_16_2 - PCx * fp_2.x_16_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 16, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_16_3 - PCy * fp_2.x_16_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  LOCSTOREFULL(store, 16, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_16_3 - PCx * fp_2.x_16_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 16, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_16_1 - PCx * fp_2.x_16_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 16, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 16, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_16_2 - PCy * fp_2.x_16_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 16, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
  LOCSTOREFULL(store, 16, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_16_3 - PCz * fp_2.x_16_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 16, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_16_0 - fs_2.x_16_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 16, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  LOCSTOREFULL(store, 16, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_17_2 - PCx * fp_2.x_17_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
  LOCSTOREFULL(store, 17, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_17_3 - PCy * fp_2.x_17_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 17, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_17_3 - PCx * fp_2.x_17_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  LOCSTOREFULL(store, 17, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_17_1 - PCx * fp_2.x_17_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 17, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_1.x_7_1 - dp_2.x_7_1); 
  LOCSTOREFULL(store, 17, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_17_2 - PCy * fp_2.x_17_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 17, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 17, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_17_3 - PCz * fp_2.x_17_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 17, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_17_0 - fs_2.x_17_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 17, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 17, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_18_2 - PCx * fp_2.x_18_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 18, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_18_3 - PCy * fp_2.x_18_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  LOCSTOREFULL(store, 18, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_18_3 - PCx * fp_2.x_18_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 18, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_18_1 - PCx * fp_2.x_18_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 18, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 18, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_18_2 - PCy * fp_2.x_18_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 18, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
  LOCSTOREFULL(store, 18, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_18_3 - PCz * fp_2.x_18_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 18, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_18_0 - fs_2.x_18_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 18, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 18, 9, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_19_2 - PCx * fp_2.x_19_2; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 19, 4, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_19_3 - PCy * fp_2.x_19_3; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 19, 5, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_19_3 - PCx * fp_2.x_19_3; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 19, 6, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBx * fp_1.x_19_1 - PCx * fp_2.x_19_1; 
#else 
  val = PBx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 19, 1, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 19, 7, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBy * fp_1.x_19_2 - PCy * fp_2.x_19_2; 
#else 
  val = PBy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 19, 2, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 19, 8, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_FP 
  val = PBz * fp_1.x_19_3 - PCz * fp_2.x_19_3; 
#else 
  val = PBz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 19, 3, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_FS 
  val += TwoZetaInv * (fs_1.x_19_0 - fs_2.x_19_0); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 19, 0, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 3.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  LOCSTOREFULL(store, 19, 9, STOREDIM, STOREDIM, 1) = val; 
#endif 
#endif 

 } 

/* DF true integral, m=0 */ 
__device__ __inline__ DFint_0::DFint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PDint_0 pd_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|d] for m=0 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=0 
#ifndef USE_PARTIAL_PF 
  PFint_0 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
#endif 
  PDint_1 pd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|d] for m=1 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
#ifndef USE_PARTIAL_PF 
  PFint_1 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
#endif 

#ifdef REG_DF 
  x_4_10 = PAx * pf_0.x_2_10 - PCx * pf_1.x_2_10; 
  x_4_10 += TwoZetaInv * 1.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
  x_5_10 = PAy * pf_0.x_3_10 - PCy * pf_1.x_3_10; 
  x_5_10 += TwoZetaInv * 1.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  x_6_10 = PAx * pf_0.x_3_10 - PCx * pf_1.x_3_10; 
  x_6_10 += TwoZetaInv * 1.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  x_7_10 = PAx * pf_0.x_1_10 - PCx * pf_1.x_1_10; 
  x_7_10 += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
  x_7_10 += TwoZetaInv * 1.000000 * (pd_0.x_1_5 - pd_1.x_1_5); 
  x_8_10 = PAy * pf_0.x_2_10 - PCy * pf_1.x_2_10; 
  x_8_10 += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
  x_8_10 += TwoZetaInv * 1.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
  x_9_10 = PAz * pf_0.x_3_10 - PCz * pf_1.x_3_10; 
  x_9_10 += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
  x_9_10 += TwoZetaInv * 1.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  x_4_11 = PAx * pf_0.x_2_11 - PCx * pf_1.x_2_11; 
  x_4_11 += TwoZetaInv * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
  x_5_11 = PAy * pf_0.x_3_11 - PCy * pf_1.x_3_11; 
  x_5_11 += TwoZetaInv * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  x_6_11 = PAx * pf_0.x_3_11 - PCx * pf_1.x_3_11; 
  x_6_11 += TwoZetaInv * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  x_7_11 = PAx * pf_0.x_1_11 - PCx * pf_1.x_1_11; 
  x_7_11 += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
  x_7_11 += TwoZetaInv * 2.000000 * (pd_0.x_1_4 - pd_1.x_1_4); 
  x_8_11 = PAy * pf_0.x_2_11 - PCy * pf_1.x_2_11; 
  x_8_11 += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
  x_8_11 += TwoZetaInv * 1.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
  x_9_11 = PAz * pf_0.x_3_11 - PCz * pf_1.x_3_11; 
  x_9_11 += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
  x_4_12 = PAx * pf_0.x_2_12 - PCx * pf_1.x_2_12; 
  x_4_12 += TwoZetaInv * 1.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
  x_5_12 = PAy * pf_0.x_3_12 - PCy * pf_1.x_3_12; 
  x_5_12 += TwoZetaInv * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  x_6_12 = PAx * pf_0.x_3_12 - PCx * pf_1.x_3_12; 
  x_6_12 += TwoZetaInv * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  x_7_12 = PAx * pf_0.x_1_12 - PCx * pf_1.x_1_12; 
  x_7_12 += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
  x_7_12 += TwoZetaInv * 1.000000 * (pd_0.x_1_8 - pd_1.x_1_8); 
  x_8_12 = PAy * pf_0.x_2_12 - PCy * pf_1.x_2_12; 
  x_8_12 += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
  x_8_12 += TwoZetaInv * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
  x_9_12 = PAz * pf_0.x_3_12 - PCz * pf_1.x_3_12; 
  x_9_12 += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
  x_4_13 = PAx * pf_0.x_2_13 - PCx * pf_1.x_2_13; 
  x_4_13 += TwoZetaInv * 2.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
  x_5_13 = PAy * pf_0.x_3_13 - PCy * pf_1.x_3_13; 
  x_6_13 = PAx * pf_0.x_3_13 - PCx * pf_1.x_3_13; 
  x_6_13 += TwoZetaInv * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  x_7_13 = PAx * pf_0.x_1_13 - PCx * pf_1.x_1_13; 
  x_7_13 += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
  x_7_13 += TwoZetaInv * 2.000000 * (pd_0.x_1_6 - pd_1.x_1_6); 
  x_8_13 = PAy * pf_0.x_2_13 - PCy * pf_1.x_2_13; 
  x_8_13 += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
  x_9_13 = PAz * pf_0.x_3_13 - PCz * pf_1.x_3_13; 
  x_9_13 += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
  x_9_13 += TwoZetaInv * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  x_4_14 = PAx * pf_0.x_2_14 - PCx * pf_1.x_2_14; 
  x_4_14 += TwoZetaInv * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
  x_5_14 = PAy * pf_0.x_3_14 - PCy * pf_1.x_3_14; 
  x_6_14 = PAx * pf_0.x_3_14 - PCx * pf_1.x_3_14; 
  x_6_14 += TwoZetaInv * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  x_7_14 = PAx * pf_0.x_1_14 - PCx * pf_1.x_1_14; 
  x_7_14 += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
  x_7_14 += TwoZetaInv * 1.000000 * (pd_0.x_1_9 - pd_1.x_1_9); 
  x_8_14 = PAy * pf_0.x_2_14 - PCy * pf_1.x_2_14; 
  x_8_14 += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
  x_9_14 = PAz * pf_0.x_3_14 - PCz * pf_1.x_3_14; 
  x_9_14 += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
  x_9_14 += TwoZetaInv * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  x_4_15 = PAx * pf_0.x_2_15 - PCx * pf_1.x_2_15; 
  x_5_15 = PAy * pf_0.x_3_15 - PCy * pf_1.x_3_15; 
  x_5_15 += TwoZetaInv * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  x_6_15 = PAx * pf_0.x_3_15 - PCx * pf_1.x_3_15; 
  x_7_15 = PAx * pf_0.x_1_15 - PCx * pf_1.x_1_15; 
  x_7_15 += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
  x_8_15 = PAy * pf_0.x_2_15 - PCy * pf_1.x_2_15; 
  x_8_15 += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
  x_8_15 += TwoZetaInv * 2.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
  x_9_15 = PAz * pf_0.x_3_15 - PCz * pf_1.x_3_15; 
  x_9_15 += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
  x_9_15 += TwoZetaInv * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  x_4_16 = PAx * pf_0.x_2_16 - PCx * pf_1.x_2_16; 
  x_5_16 = PAy * pf_0.x_3_16 - PCy * pf_1.x_3_16; 
  x_5_16 += TwoZetaInv * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  x_6_16 = PAx * pf_0.x_3_16 - PCx * pf_1.x_3_16; 
  x_7_16 = PAx * pf_0.x_1_16 - PCx * pf_1.x_1_16; 
  x_7_16 += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
  x_8_16 = PAy * pf_0.x_2_16 - PCy * pf_1.x_2_16; 
  x_8_16 += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
  x_8_16 += TwoZetaInv * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
  x_9_16 = PAz * pf_0.x_3_16 - PCz * pf_1.x_3_16; 
  x_9_16 += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
  x_9_16 += TwoZetaInv * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  x_4_17 = PAx * pf_0.x_2_17 - PCx * pf_1.x_2_17; 
  x_4_17 += TwoZetaInv * 3.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
  x_5_17 = PAy * pf_0.x_3_17 - PCy * pf_1.x_3_17; 
  x_6_17 = PAx * pf_0.x_3_17 - PCx * pf_1.x_3_17; 
  x_6_17 += TwoZetaInv * 3.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  x_7_17 = PAx * pf_0.x_1_17 - PCx * pf_1.x_1_17; 
  x_7_17 += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
  x_7_17 += TwoZetaInv * 3.000000 * (pd_0.x_1_7 - pd_1.x_1_7); 
  x_8_17 = PAy * pf_0.x_2_17 - PCy * pf_1.x_2_17; 
  x_8_17 += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
  x_9_17 = PAz * pf_0.x_3_17 - PCz * pf_1.x_3_17; 
  x_9_17 += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
  x_4_18 = PAx * pf_0.x_2_18 - PCx * pf_1.x_2_18; 
  x_5_18 = PAy * pf_0.x_3_18 - PCy * pf_1.x_3_18; 
  x_5_18 += TwoZetaInv * 3.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  x_6_18 = PAx * pf_0.x_3_18 - PCx * pf_1.x_3_18; 
  x_7_18 = PAx * pf_0.x_1_18 - PCx * pf_1.x_1_18; 
  x_7_18 += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
  x_8_18 = PAy * pf_0.x_2_18 - PCy * pf_1.x_2_18; 
  x_8_18 += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
  x_8_18 += TwoZetaInv * 3.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
  x_9_18 = PAz * pf_0.x_3_18 - PCz * pf_1.x_3_18; 
  x_9_18 += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
  x_4_19 = PAx * pf_0.x_2_19 - PCx * pf_1.x_2_19; 
  x_5_19 = PAy * pf_0.x_3_19 - PCy * pf_1.x_3_19; 
  x_6_19 = PAx * pf_0.x_3_19 - PCx * pf_1.x_3_19; 
  x_7_19 = PAx * pf_0.x_1_19 - PCx * pf_1.x_1_19; 
  x_7_19 += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
  x_8_19 = PAy * pf_0.x_2_19 - PCy * pf_1.x_2_19; 
  x_8_19 += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
  x_9_19 = PAz * pf_0.x_3_19 - PCz * pf_1.x_3_19; 
  x_9_19 += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
  x_9_19 += TwoZetaInv * 3.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
#else 
#ifdef USE_PARTIAL_PF 
  { 
    PFint_0_1 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_1 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_10 - PCx * pf_1.x_2_10; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
    LOCSTOREFULL(store, 4, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_10 - PCy * pf_1.x_3_10; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
    LOCSTOREFULL(store, 5, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_10 - PCx * pf_1.x_3_10; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
    LOCSTOREFULL(store, 6, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_10 - PCx * pf_1.x_1_10; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_1_5 - pd_1.x_1_5); 
    LOCSTOREFULL(store, 7, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_10 - PCy * pf_1.x_2_10; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
    LOCSTOREFULL(store, 8, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_10 - PCz * pf_1.x_3_10; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
    LOCSTOREFULL(store, 9, 10, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_2 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_2 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_11 - PCx * pf_1.x_2_11; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
    LOCSTOREFULL(store, 4, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_11 - PCy * pf_1.x_3_11; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
    LOCSTOREFULL(store, 5, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_11 - PCx * pf_1.x_3_11; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
    LOCSTOREFULL(store, 6, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_11 - PCx * pf_1.x_1_11; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_1_4 - pd_1.x_1_4); 
    LOCSTOREFULL(store, 7, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_11 - PCy * pf_1.x_2_11; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
    LOCSTOREFULL(store, 8, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_11 - PCz * pf_1.x_3_11; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 9, 11, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_3 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_3 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_12 - PCx * pf_1.x_2_12; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
    LOCSTOREFULL(store, 4, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_12 - PCy * pf_1.x_3_12; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
    LOCSTOREFULL(store, 5, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_12 - PCx * pf_1.x_3_12; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
    LOCSTOREFULL(store, 6, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_12 - PCx * pf_1.x_1_12; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_1_8 - pd_1.x_1_8); 
    LOCSTOREFULL(store, 7, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_12 - PCy * pf_1.x_2_12; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
    LOCSTOREFULL(store, 8, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_12 - PCz * pf_1.x_3_12; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 9, 12, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_4 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_4 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_13 - PCx * pf_1.x_2_13; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
    LOCSTOREFULL(store, 4, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_13 - PCy * pf_1.x_3_13; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 5, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_13 - PCx * pf_1.x_3_13; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
    LOCSTOREFULL(store, 6, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_13 - PCx * pf_1.x_1_13; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_1_6 - pd_1.x_1_6); 
    LOCSTOREFULL(store, 7, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_13 - PCy * pf_1.x_2_13; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 8, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_13 - PCz * pf_1.x_3_13; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
    LOCSTOREFULL(store, 9, 13, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_5 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_5 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_14 - PCx * pf_1.x_2_14; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
    LOCSTOREFULL(store, 4, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_14 - PCy * pf_1.x_3_14; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 5, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_14 - PCx * pf_1.x_3_14; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
    LOCSTOREFULL(store, 6, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_14 - PCx * pf_1.x_1_14; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_1_9 - pd_1.x_1_9); 
    LOCSTOREFULL(store, 7, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_14 - PCy * pf_1.x_2_14; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 8, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_14 - PCz * pf_1.x_3_14; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
    LOCSTOREFULL(store, 9, 14, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_6 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_6 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_15 - PCx * pf_1.x_2_15; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 4, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_15 - PCy * pf_1.x_3_15; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
    LOCSTOREFULL(store, 5, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_15 - PCx * pf_1.x_3_15; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 6, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_15 - PCx * pf_1.x_1_15; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 7, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_15 - PCy * pf_1.x_2_15; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
    LOCSTOREFULL(store, 8, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_15 - PCz * pf_1.x_3_15; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
    LOCSTOREFULL(store, 9, 15, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_7 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_7 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_16 - PCx * pf_1.x_2_16; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 4, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_16 - PCy * pf_1.x_3_16; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
    LOCSTOREFULL(store, 5, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_16 - PCx * pf_1.x_3_16; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 6, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_16 - PCx * pf_1.x_1_16; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 7, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_16 - PCy * pf_1.x_2_16; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
    LOCSTOREFULL(store, 8, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_16 - PCz * pf_1.x_3_16; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
    LOCSTOREFULL(store, 9, 16, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_8 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_8 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_17 - PCx * pf_1.x_2_17; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
    LOCSTOREFULL(store, 4, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_17 - PCy * pf_1.x_3_17; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 5, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_17 - PCx * pf_1.x_3_17; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
    LOCSTOREFULL(store, 6, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_17 - PCx * pf_1.x_1_17; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_0.x_1_7 - pd_1.x_1_7); 
    LOCSTOREFULL(store, 7, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_17 - PCy * pf_1.x_2_17; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 8, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_17 - PCz * pf_1.x_3_17; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 9, 17, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_9 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_9 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_18 - PCx * pf_1.x_2_18; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 4, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_18 - PCy * pf_1.x_3_18; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
    LOCSTOREFULL(store, 5, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_18 - PCx * pf_1.x_3_18; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 6, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_18 - PCx * pf_1.x_1_18; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 7, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_18 - PCy * pf_1.x_2_18; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
    LOCSTOREFULL(store, 8, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_18 - PCz * pf_1.x_3_18; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 9, 18, STOREDIM, STOREDIM, 0) = val; 
  } 

  { 
    PFint_0_10 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=0 
    PFint_1_10 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_0.x_2_19 - PCx * pf_1.x_2_19; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 4, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_3_19 - PCy * pf_1.x_3_19; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 5, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_3_19 - PCx * pf_1.x_3_19; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1); 
#endif 
    LOCSTOREFULL(store, 6, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAx * pf_0.x_1_19 - PCx * pf_1.x_1_19; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 7, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAy * pf_0.x_2_19 - PCy * pf_1.x_2_19; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)); 
#endif 
    LOCSTOREFULL(store, 8, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
    val = PAz * pf_0.x_3_19 - PCz * pf_1.x_3_19; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
    LOCSTOREFULL(store, 9, 19, STOREDIM, STOREDIM, 0) = val; 
  } 

#else 
  QUICKDouble val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_10 - PCx * pf_1.x_2_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
  LOCSTOREFULL(store, 4, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_10 - PCy * pf_1.x_3_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  LOCSTOREFULL(store, 5, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_10 - PCx * pf_1.x_3_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  LOCSTOREFULL(store, 6, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_10 - PCx * pf_1.x_1_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_1_5 - pd_1.x_1_5); 
  LOCSTOREFULL(store, 7, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_10 - PCy * pf_1.x_2_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
  LOCSTOREFULL(store, 8, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_10 - PCz * pf_1.x_3_10; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_10 - sf_1.x_0_10); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  LOCSTOREFULL(store, 9, 10, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_11 - PCx * pf_1.x_2_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
  LOCSTOREFULL(store, 4, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_11 - PCy * pf_1.x_3_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  LOCSTOREFULL(store, 5, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_11 - PCx * pf_1.x_3_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  LOCSTOREFULL(store, 6, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_11 - PCx * pf_1.x_1_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_1_4 - pd_1.x_1_4); 
  LOCSTOREFULL(store, 7, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_11 - PCy * pf_1.x_2_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
  LOCSTOREFULL(store, 8, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_11 - PCz * pf_1.x_3_11; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_11 - sf_1.x_0_11); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 9, 11, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_12 - PCx * pf_1.x_2_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
  LOCSTOREFULL(store, 4, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_12 - PCy * pf_1.x_3_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  LOCSTOREFULL(store, 5, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_12 - PCx * pf_1.x_3_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  LOCSTOREFULL(store, 6, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_12 - PCx * pf_1.x_1_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_1_8 - pd_1.x_1_8); 
  LOCSTOREFULL(store, 7, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_12 - PCy * pf_1.x_2_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
  LOCSTOREFULL(store, 8, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_12 - PCz * pf_1.x_3_12; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_12 - sf_1.x_0_12); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 9, 12, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_13 - PCx * pf_1.x_2_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
  LOCSTOREFULL(store, 4, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_13 - PCy * pf_1.x_3_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 5, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_13 - PCx * pf_1.x_3_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  LOCSTOREFULL(store, 6, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_13 - PCx * pf_1.x_1_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_1_6 - pd_1.x_1_6); 
  LOCSTOREFULL(store, 7, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_13 - PCy * pf_1.x_2_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 8, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_13 - PCz * pf_1.x_3_13; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_13 - sf_1.x_0_13); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  LOCSTOREFULL(store, 9, 13, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_14 - PCx * pf_1.x_2_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
  LOCSTOREFULL(store, 4, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_14 - PCy * pf_1.x_3_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 5, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_14 - PCx * pf_1.x_3_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  LOCSTOREFULL(store, 6, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_14 - PCx * pf_1.x_1_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_1_9 - pd_1.x_1_9); 
  LOCSTOREFULL(store, 7, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_14 - PCy * pf_1.x_2_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 8, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_14 - PCz * pf_1.x_3_14; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_14 - sf_1.x_0_14); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  LOCSTOREFULL(store, 9, 14, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_15 - PCx * pf_1.x_2_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 4, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_15 - PCy * pf_1.x_3_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  LOCSTOREFULL(store, 5, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_15 - PCx * pf_1.x_3_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 6, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_15 - PCx * pf_1.x_1_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 7, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_15 - PCy * pf_1.x_2_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
  LOCSTOREFULL(store, 8, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_15 - PCz * pf_1.x_3_15; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_15 - sf_1.x_0_15); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  LOCSTOREFULL(store, 9, 15, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_16 - PCx * pf_1.x_2_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 4, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_16 - PCy * pf_1.x_3_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  LOCSTOREFULL(store, 5, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_16 - PCx * pf_1.x_3_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 6, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_16 - PCx * pf_1.x_1_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 7, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_16 - PCy * pf_1.x_2_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
  LOCSTOREFULL(store, 8, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_16 - PCz * pf_1.x_3_16; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_16 - sf_1.x_0_16); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  LOCSTOREFULL(store, 9, 16, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_17 - PCx * pf_1.x_2_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
  LOCSTOREFULL(store, 4, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_17 - PCy * pf_1.x_3_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 5, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_17 - PCx * pf_1.x_3_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  LOCSTOREFULL(store, 6, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_17 - PCx * pf_1.x_1_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_0.x_1_7 - pd_1.x_1_7); 
  LOCSTOREFULL(store, 7, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_17 - PCy * pf_1.x_2_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 8, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_17 - PCz * pf_1.x_3_17; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_17 - sf_1.x_0_17); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 9, 17, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_18 - PCx * pf_1.x_2_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 4, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_18 - PCy * pf_1.x_3_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  LOCSTOREFULL(store, 5, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_18 - PCx * pf_1.x_3_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 6, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_18 - PCx * pf_1.x_1_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 7, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_18 - PCy * pf_1.x_2_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
  LOCSTOREFULL(store, 8, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_18 - PCz * pf_1.x_3_18; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_18 - sf_1.x_0_18); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 9, 18, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_2_19 - PCx * pf_1.x_2_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 4, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_3_19 - PCy * pf_1.x_3_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 5, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_3_19 - PCx * pf_1.x_3_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1); 
#endif 
  LOCSTOREFULL(store, 6, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAx * pf_0.x_1_19 - PCx * pf_1.x_1_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 0) - PCx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 7, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAy * pf_0.x_2_19 - PCy * pf_1.x_2_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 0) - PCy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)); 
#endif 
  LOCSTOREFULL(store, 8, 19, STOREDIM, STOREDIM, 0) = val; 
#ifdef REG_PF 
  val = PAz * pf_0.x_3_19 - PCz * pf_1.x_3_19; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 0) - PCz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_0.x_0_19 - sf_1.x_0_19); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 0) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1)); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  LOCSTOREFULL(store, 9, 19, STOREDIM, STOREDIM, 0) = val; 
#endif 
#endif 

 } 

/* DF auxilary integral, m=1 */ 
__device__ __inline__ DFint_1::DFint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  PDint_1 pd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|d] for m=1 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=1 
#ifndef USE_PARTIAL_PF 
  PFint_1 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
#endif 
  PDint_2 pd_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|d] for m=2 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=2 
#ifndef USE_PARTIAL_PF 
  PFint_2 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 
#endif 

#ifdef REG_DF 
  x_4_10 = PAx * pf_1.x_2_10 - PCx * pf_2.x_2_10; 
  x_4_10 += TwoZetaInv * 1.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
  x_5_10 = PAy * pf_1.x_3_10 - PCy * pf_2.x_3_10; 
  x_5_10 += TwoZetaInv * 1.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  x_6_10 = PAx * pf_1.x_3_10 - PCx * pf_2.x_3_10; 
  x_6_10 += TwoZetaInv * 1.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  x_7_10 = PAx * pf_1.x_1_10 - PCx * pf_2.x_1_10; 
  x_7_10 += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
  x_7_10 += TwoZetaInv * 1.000000 * (pd_1.x_1_5 - pd_2.x_1_5); 
  x_8_10 = PAy * pf_1.x_2_10 - PCy * pf_2.x_2_10; 
  x_8_10 += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
  x_8_10 += TwoZetaInv * 1.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
  x_9_10 = PAz * pf_1.x_3_10 - PCz * pf_2.x_3_10; 
  x_9_10 += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
  x_9_10 += TwoZetaInv * 1.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  x_4_11 = PAx * pf_1.x_2_11 - PCx * pf_2.x_2_11; 
  x_4_11 += TwoZetaInv * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
  x_5_11 = PAy * pf_1.x_3_11 - PCy * pf_2.x_3_11; 
  x_5_11 += TwoZetaInv * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  x_6_11 = PAx * pf_1.x_3_11 - PCx * pf_2.x_3_11; 
  x_6_11 += TwoZetaInv * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  x_7_11 = PAx * pf_1.x_1_11 - PCx * pf_2.x_1_11; 
  x_7_11 += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
  x_7_11 += TwoZetaInv * 2.000000 * (pd_1.x_1_4 - pd_2.x_1_4); 
  x_8_11 = PAy * pf_1.x_2_11 - PCy * pf_2.x_2_11; 
  x_8_11 += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
  x_8_11 += TwoZetaInv * 1.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
  x_9_11 = PAz * pf_1.x_3_11 - PCz * pf_2.x_3_11; 
  x_9_11 += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
  x_4_12 = PAx * pf_1.x_2_12 - PCx * pf_2.x_2_12; 
  x_4_12 += TwoZetaInv * 1.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
  x_5_12 = PAy * pf_1.x_3_12 - PCy * pf_2.x_3_12; 
  x_5_12 += TwoZetaInv * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  x_6_12 = PAx * pf_1.x_3_12 - PCx * pf_2.x_3_12; 
  x_6_12 += TwoZetaInv * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  x_7_12 = PAx * pf_1.x_1_12 - PCx * pf_2.x_1_12; 
  x_7_12 += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
  x_7_12 += TwoZetaInv * 1.000000 * (pd_1.x_1_8 - pd_2.x_1_8); 
  x_8_12 = PAy * pf_1.x_2_12 - PCy * pf_2.x_2_12; 
  x_8_12 += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
  x_8_12 += TwoZetaInv * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
  x_9_12 = PAz * pf_1.x_3_12 - PCz * pf_2.x_3_12; 
  x_9_12 += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
  x_4_13 = PAx * pf_1.x_2_13 - PCx * pf_2.x_2_13; 
  x_4_13 += TwoZetaInv * 2.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
  x_5_13 = PAy * pf_1.x_3_13 - PCy * pf_2.x_3_13; 
  x_6_13 = PAx * pf_1.x_3_13 - PCx * pf_2.x_3_13; 
  x_6_13 += TwoZetaInv * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  x_7_13 = PAx * pf_1.x_1_13 - PCx * pf_2.x_1_13; 
  x_7_13 += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
  x_7_13 += TwoZetaInv * 2.000000 * (pd_1.x_1_6 - pd_2.x_1_6); 
  x_8_13 = PAy * pf_1.x_2_13 - PCy * pf_2.x_2_13; 
  x_8_13 += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
  x_9_13 = PAz * pf_1.x_3_13 - PCz * pf_2.x_3_13; 
  x_9_13 += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
  x_9_13 += TwoZetaInv * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  x_4_14 = PAx * pf_1.x_2_14 - PCx * pf_2.x_2_14; 
  x_4_14 += TwoZetaInv * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
  x_5_14 = PAy * pf_1.x_3_14 - PCy * pf_2.x_3_14; 
  x_6_14 = PAx * pf_1.x_3_14 - PCx * pf_2.x_3_14; 
  x_6_14 += TwoZetaInv * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  x_7_14 = PAx * pf_1.x_1_14 - PCx * pf_2.x_1_14; 
  x_7_14 += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
  x_7_14 += TwoZetaInv * 1.000000 * (pd_1.x_1_9 - pd_2.x_1_9); 
  x_8_14 = PAy * pf_1.x_2_14 - PCy * pf_2.x_2_14; 
  x_8_14 += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
  x_9_14 = PAz * pf_1.x_3_14 - PCz * pf_2.x_3_14; 
  x_9_14 += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
  x_9_14 += TwoZetaInv * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  x_4_15 = PAx * pf_1.x_2_15 - PCx * pf_2.x_2_15; 
  x_5_15 = PAy * pf_1.x_3_15 - PCy * pf_2.x_3_15; 
  x_5_15 += TwoZetaInv * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  x_6_15 = PAx * pf_1.x_3_15 - PCx * pf_2.x_3_15; 
  x_7_15 = PAx * pf_1.x_1_15 - PCx * pf_2.x_1_15; 
  x_7_15 += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
  x_8_15 = PAy * pf_1.x_2_15 - PCy * pf_2.x_2_15; 
  x_8_15 += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
  x_8_15 += TwoZetaInv * 2.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
  x_9_15 = PAz * pf_1.x_3_15 - PCz * pf_2.x_3_15; 
  x_9_15 += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
  x_9_15 += TwoZetaInv * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  x_4_16 = PAx * pf_1.x_2_16 - PCx * pf_2.x_2_16; 
  x_5_16 = PAy * pf_1.x_3_16 - PCy * pf_2.x_3_16; 
  x_5_16 += TwoZetaInv * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  x_6_16 = PAx * pf_1.x_3_16 - PCx * pf_2.x_3_16; 
  x_7_16 = PAx * pf_1.x_1_16 - PCx * pf_2.x_1_16; 
  x_7_16 += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
  x_8_16 = PAy * pf_1.x_2_16 - PCy * pf_2.x_2_16; 
  x_8_16 += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
  x_8_16 += TwoZetaInv * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
  x_9_16 = PAz * pf_1.x_3_16 - PCz * pf_2.x_3_16; 
  x_9_16 += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
  x_9_16 += TwoZetaInv * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  x_4_17 = PAx * pf_1.x_2_17 - PCx * pf_2.x_2_17; 
  x_4_17 += TwoZetaInv * 3.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
  x_5_17 = PAy * pf_1.x_3_17 - PCy * pf_2.x_3_17; 
  x_6_17 = PAx * pf_1.x_3_17 - PCx * pf_2.x_3_17; 
  x_6_17 += TwoZetaInv * 3.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  x_7_17 = PAx * pf_1.x_1_17 - PCx * pf_2.x_1_17; 
  x_7_17 += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
  x_7_17 += TwoZetaInv * 3.000000 * (pd_1.x_1_7 - pd_2.x_1_7); 
  x_8_17 = PAy * pf_1.x_2_17 - PCy * pf_2.x_2_17; 
  x_8_17 += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
  x_9_17 = PAz * pf_1.x_3_17 - PCz * pf_2.x_3_17; 
  x_9_17 += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
  x_4_18 = PAx * pf_1.x_2_18 - PCx * pf_2.x_2_18; 
  x_5_18 = PAy * pf_1.x_3_18 - PCy * pf_2.x_3_18; 
  x_5_18 += TwoZetaInv * 3.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  x_6_18 = PAx * pf_1.x_3_18 - PCx * pf_2.x_3_18; 
  x_7_18 = PAx * pf_1.x_1_18 - PCx * pf_2.x_1_18; 
  x_7_18 += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
  x_8_18 = PAy * pf_1.x_2_18 - PCy * pf_2.x_2_18; 
  x_8_18 += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
  x_8_18 += TwoZetaInv * 3.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
  x_9_18 = PAz * pf_1.x_3_18 - PCz * pf_2.x_3_18; 
  x_9_18 += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
  x_4_19 = PAx * pf_1.x_2_19 - PCx * pf_2.x_2_19; 
  x_5_19 = PAy * pf_1.x_3_19 - PCy * pf_2.x_3_19; 
  x_6_19 = PAx * pf_1.x_3_19 - PCx * pf_2.x_3_19; 
  x_7_19 = PAx * pf_1.x_1_19 - PCx * pf_2.x_1_19; 
  x_7_19 += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
  x_8_19 = PAy * pf_1.x_2_19 - PCy * pf_2.x_2_19; 
  x_8_19 += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
  x_9_19 = PAz * pf_1.x_3_19 - PCz * pf_2.x_3_19; 
  x_9_19 += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
  x_9_19 += TwoZetaInv * 3.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
#else 
#ifdef USE_PARTIAL_PF 
  { 
    PFint_1_1 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_1 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_10 - PCx * pf_2.x_2_10; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
    LOCSTOREFULL(store, 4, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_10 - PCy * pf_2.x_3_10; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
    LOCSTOREFULL(store, 5, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_10 - PCx * pf_2.x_3_10; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
    LOCSTOREFULL(store, 6, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_10 - PCx * pf_2.x_1_10; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_1_5 - pd_2.x_1_5); 
    LOCSTOREFULL(store, 7, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_10 - PCy * pf_2.x_2_10; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
    LOCSTOREFULL(store, 8, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_10 - PCz * pf_2.x_3_10; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
    LOCSTOREFULL(store, 9, 10, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_2 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_2 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_11 - PCx * pf_2.x_2_11; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
    LOCSTOREFULL(store, 4, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_11 - PCy * pf_2.x_3_11; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
    LOCSTOREFULL(store, 5, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_11 - PCx * pf_2.x_3_11; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
    LOCSTOREFULL(store, 6, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_11 - PCx * pf_2.x_1_11; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_1_4 - pd_2.x_1_4); 
    LOCSTOREFULL(store, 7, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_11 - PCy * pf_2.x_2_11; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
    LOCSTOREFULL(store, 8, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_11 - PCz * pf_2.x_3_11; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 9, 11, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_3 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_3 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_12 - PCx * pf_2.x_2_12; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
    LOCSTOREFULL(store, 4, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_12 - PCy * pf_2.x_3_12; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
    LOCSTOREFULL(store, 5, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_12 - PCx * pf_2.x_3_12; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
    LOCSTOREFULL(store, 6, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_12 - PCx * pf_2.x_1_12; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_1_8 - pd_2.x_1_8); 
    LOCSTOREFULL(store, 7, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_12 - PCy * pf_2.x_2_12; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
    LOCSTOREFULL(store, 8, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_12 - PCz * pf_2.x_3_12; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 9, 12, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_4 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_4 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_13 - PCx * pf_2.x_2_13; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
    LOCSTOREFULL(store, 4, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_13 - PCy * pf_2.x_3_13; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 5, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_13 - PCx * pf_2.x_3_13; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
    LOCSTOREFULL(store, 6, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_13 - PCx * pf_2.x_1_13; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_1_6 - pd_2.x_1_6); 
    LOCSTOREFULL(store, 7, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_13 - PCy * pf_2.x_2_13; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 8, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_13 - PCz * pf_2.x_3_13; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
    LOCSTOREFULL(store, 9, 13, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_5 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_5 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_14 - PCx * pf_2.x_2_14; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
    LOCSTOREFULL(store, 4, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_14 - PCy * pf_2.x_3_14; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 5, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_14 - PCx * pf_2.x_3_14; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
    LOCSTOREFULL(store, 6, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_14 - PCx * pf_2.x_1_14; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_1_9 - pd_2.x_1_9); 
    LOCSTOREFULL(store, 7, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_14 - PCy * pf_2.x_2_14; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 8, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_14 - PCz * pf_2.x_3_14; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
    LOCSTOREFULL(store, 9, 14, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_6 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_6 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_15 - PCx * pf_2.x_2_15; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 4, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_15 - PCy * pf_2.x_3_15; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
    LOCSTOREFULL(store, 5, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_15 - PCx * pf_2.x_3_15; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 6, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_15 - PCx * pf_2.x_1_15; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 7, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_15 - PCy * pf_2.x_2_15; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
    LOCSTOREFULL(store, 8, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_15 - PCz * pf_2.x_3_15; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
    LOCSTOREFULL(store, 9, 15, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_7 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_7 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_16 - PCx * pf_2.x_2_16; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 4, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_16 - PCy * pf_2.x_3_16; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
    LOCSTOREFULL(store, 5, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_16 - PCx * pf_2.x_3_16; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 6, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_16 - PCx * pf_2.x_1_16; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 7, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_16 - PCy * pf_2.x_2_16; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
    LOCSTOREFULL(store, 8, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_16 - PCz * pf_2.x_3_16; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
    LOCSTOREFULL(store, 9, 16, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_8 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_8 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_17 - PCx * pf_2.x_2_17; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
    LOCSTOREFULL(store, 4, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_17 - PCy * pf_2.x_3_17; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 5, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_17 - PCx * pf_2.x_3_17; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
    LOCSTOREFULL(store, 6, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_17 - PCx * pf_2.x_1_17; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_1.x_1_7 - pd_2.x_1_7); 
    LOCSTOREFULL(store, 7, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_17 - PCy * pf_2.x_2_17; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 8, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_17 - PCz * pf_2.x_3_17; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 9, 17, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_9 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_9 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_18 - PCx * pf_2.x_2_18; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 4, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_18 - PCy * pf_2.x_3_18; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
    LOCSTOREFULL(store, 5, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_18 - PCx * pf_2.x_3_18; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 6, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_18 - PCx * pf_2.x_1_18; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 7, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_18 - PCy * pf_2.x_2_18; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
    LOCSTOREFULL(store, 8, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_18 - PCz * pf_2.x_3_18; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 9, 18, STOREDIM, STOREDIM, 1) = val; 
  } 

  { 
    PFint_1_10 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=1 
    PFint_2_10 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [p|f] for m=2 

    QUICKDouble val; 

#ifdef REG_PF 
    val = PAx * pf_1.x_2_19 - PCx * pf_2.x_2_19; 
#else 
    val = PAx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 4, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_3_19 - PCy * pf_2.x_3_19; 
#else 
    val = PAy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 5, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_3_19 - PCx * pf_2.x_3_19; 
#else 
    val = PAx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2); 
#endif 
    LOCSTOREFULL(store, 6, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAx * pf_1.x_1_19 - PCx * pf_2.x_1_19; 
#else 
    val = PAx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 7, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAy * pf_1.x_2_19 - PCy * pf_2.x_2_19; 
#else 
    val = PAy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)); 
#endif 
    LOCSTOREFULL(store, 8, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
    val = PAz * pf_1.x_3_19 - PCz * pf_2.x_3_19; 
#else 
    val = PAz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
    val += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
#else 
    val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)); 
#endif 
    val += TwoZetaInv * 3.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
    LOCSTOREFULL(store, 9, 19, STOREDIM, STOREDIM, 1) = val; 
  } 

#else 
  QUICKDouble val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_10 - PCx * pf_2.x_2_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
  LOCSTOREFULL(store, 4, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_10 - PCy * pf_2.x_3_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  LOCSTOREFULL(store, 5, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_10 - PCx * pf_2.x_3_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  LOCSTOREFULL(store, 6, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_10 - PCx * pf_2.x_1_10; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 10, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_1_5 - pd_2.x_1_5); 
  LOCSTOREFULL(store, 7, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_10 - PCy * pf_2.x_2_10; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 10, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
  LOCSTOREFULL(store, 8, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_10 - PCz * pf_2.x_3_10; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 10, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_10 - sf_2.x_0_10); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 10, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  LOCSTOREFULL(store, 9, 10, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_11 - PCx * pf_2.x_2_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
  LOCSTOREFULL(store, 4, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_11 - PCy * pf_2.x_3_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  LOCSTOREFULL(store, 5, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_11 - PCx * pf_2.x_3_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  LOCSTOREFULL(store, 6, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_11 - PCx * pf_2.x_1_11; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_1_4 - pd_2.x_1_4); 
  LOCSTOREFULL(store, 7, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_11 - PCy * pf_2.x_2_11; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
  LOCSTOREFULL(store, 8, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_11 - PCz * pf_2.x_3_11; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 11, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_11 - sf_2.x_0_11); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 11, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 9, 11, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_12 - PCx * pf_2.x_2_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
  LOCSTOREFULL(store, 4, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_12 - PCy * pf_2.x_3_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  LOCSTOREFULL(store, 5, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_12 - PCx * pf_2.x_3_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  LOCSTOREFULL(store, 6, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_12 - PCx * pf_2.x_1_12; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_1_8 - pd_2.x_1_8); 
  LOCSTOREFULL(store, 7, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_12 - PCy * pf_2.x_2_12; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
  LOCSTOREFULL(store, 8, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_12 - PCz * pf_2.x_3_12; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 12, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_12 - sf_2.x_0_12); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 12, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 9, 12, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_13 - PCx * pf_2.x_2_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
  LOCSTOREFULL(store, 4, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_13 - PCy * pf_2.x_3_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 5, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_13 - PCx * pf_2.x_3_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  LOCSTOREFULL(store, 6, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_13 - PCx * pf_2.x_1_13; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_1_6 - pd_2.x_1_6); 
  LOCSTOREFULL(store, 7, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_13 - PCy * pf_2.x_2_13; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 8, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_13 - PCz * pf_2.x_3_13; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 13, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_13 - sf_2.x_0_13); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 13, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  LOCSTOREFULL(store, 9, 13, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_14 - PCx * pf_2.x_2_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
  LOCSTOREFULL(store, 4, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_14 - PCy * pf_2.x_3_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 5, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_14 - PCx * pf_2.x_3_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  LOCSTOREFULL(store, 6, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_14 - PCx * pf_2.x_1_14; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_1_9 - pd_2.x_1_9); 
  LOCSTOREFULL(store, 7, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_14 - PCy * pf_2.x_2_14; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 8, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_14 - PCz * pf_2.x_3_14; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 14, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_14 - sf_2.x_0_14); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 14, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  LOCSTOREFULL(store, 9, 14, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_15 - PCx * pf_2.x_2_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 4, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_15 - PCy * pf_2.x_3_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  LOCSTOREFULL(store, 5, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_15 - PCx * pf_2.x_3_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 6, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_15 - PCx * pf_2.x_1_15; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 7, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_15 - PCy * pf_2.x_2_15; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
  LOCSTOREFULL(store, 8, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_15 - PCz * pf_2.x_3_15; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 15, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_15 - sf_2.x_0_15); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 15, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  LOCSTOREFULL(store, 9, 15, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_16 - PCx * pf_2.x_2_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 4, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_16 - PCy * pf_2.x_3_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  LOCSTOREFULL(store, 5, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_16 - PCx * pf_2.x_3_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 6, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_16 - PCx * pf_2.x_1_16; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 7, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_16 - PCy * pf_2.x_2_16; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
  LOCSTOREFULL(store, 8, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_16 - PCz * pf_2.x_3_16; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 16, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_16 - sf_2.x_0_16); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 16, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  LOCSTOREFULL(store, 9, 16, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_17 - PCx * pf_2.x_2_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
  LOCSTOREFULL(store, 4, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_17 - PCy * pf_2.x_3_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 5, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_17 - PCx * pf_2.x_3_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  LOCSTOREFULL(store, 6, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_17 - PCx * pf_2.x_1_17; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_1.x_1_7 - pd_2.x_1_7); 
  LOCSTOREFULL(store, 7, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_17 - PCy * pf_2.x_2_17; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 8, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_17 - PCz * pf_2.x_3_17; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 17, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_17 - sf_2.x_0_17); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 17, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 9, 17, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_18 - PCx * pf_2.x_2_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 4, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_18 - PCy * pf_2.x_3_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  LOCSTOREFULL(store, 5, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_18 - PCx * pf_2.x_3_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 6, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_18 - PCx * pf_2.x_1_18; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 7, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_18 - PCy * pf_2.x_2_18; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
  LOCSTOREFULL(store, 8, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_18 - PCz * pf_2.x_3_18; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 18, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_18 - sf_2.x_0_18); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 18, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 9, 18, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_2_19 - PCx * pf_2.x_2_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 4, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_3_19 - PCy * pf_2.x_3_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 5, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_3_19 - PCx * pf_2.x_3_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2); 
#endif 
  LOCSTOREFULL(store, 6, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAx * pf_1.x_1_19 - PCx * pf_2.x_1_19; 
#else 
  val = PAx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 1) - PCx * LOCSTOREFULL(store, 1, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 7, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAy * pf_1.x_2_19 - PCy * pf_2.x_2_19; 
#else 
  val = PAy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 1) - PCy * LOCSTOREFULL(store, 2, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)); 
#endif 
  LOCSTOREFULL(store, 8, 19, STOREDIM, STOREDIM, 1) = val; 
#ifdef REG_PF 
  val = PAz * pf_1.x_3_19 - PCz * pf_2.x_3_19; 
#else 
  val = PAz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 1) - PCz * LOCSTOREFULL(store, 3, 19, STOREDIM, STOREDIM, 2); 
#endif 
#ifdef REG_SF 
  val += TwoZetaInv * (sf_1.x_0_19 - sf_2.x_0_19); 
#else 
  val += TwoZetaInv * (LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 1) - LOCSTOREFULL(store, 0, 19, STOREDIM, STOREDIM, 2)); 
#endif 
  val += TwoZetaInv * 3.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  LOCSTOREFULL(store, 9, 19, STOREDIM, STOREDIM, 1) = val; 
#endif 
#endif 

 } 

/* FF true integral, m=0 */ 
__device__ __inline__ FFint_0::FFint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

  DDint_0 dd_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|d] for m=0 
  FPint_0 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=0 
  FDint_0 fd_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|d] for m=0 
  DDint_1 dd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [d|d] for m=1 
  FPint_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|p] for m=1 
  FDint_1 fd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [f|d] for m=1 

#ifdef REG_FF 
  x_10_10 = PBx * fd_0.x_10_5 - PCx * fd_1.x_10_5; 
  x_10_10 += TwoZetaInv * 1.000000 * (dd_0.x_5_5 - dd_1.x_5_5); 
  x_10_11 = PBx * fd_0.x_10_4 - PCx * fd_1.x_10_4; 
  x_10_11 += TwoZetaInv * 1.000000 * (fp_0.x_10_2 - fp_1.x_10_2); 
  x_10_11 += TwoZetaInv * 1.000000 * (dd_0.x_5_4 - dd_1.x_5_4); 
  x_10_12 = PBx * fd_0.x_10_8 - PCx * fd_1.x_10_8; 
  x_10_12 += TwoZetaInv * 1.000000 * (dd_0.x_5_8 - dd_1.x_5_8); 
  x_10_13 = PBx * fd_0.x_10_6 - PCx * fd_1.x_10_6; 
  x_10_13 += TwoZetaInv * 1.000000 * (fp_0.x_10_3 - fp_1.x_10_3); 
  x_10_13 += TwoZetaInv * 1.000000 * (dd_0.x_5_6 - dd_1.x_5_6); 
  x_10_14 = PBx * fd_0.x_10_9 - PCx * fd_1.x_10_9; 
  x_10_14 += TwoZetaInv * 1.000000 * (dd_0.x_5_9 - dd_1.x_5_9); 
  x_10_15 = PBy * fd_0.x_10_5 - PCy * fd_1.x_10_5; 
  x_10_15 += TwoZetaInv * 1.000000 * (fp_0.x_10_3 - fp_1.x_10_3); 
  x_10_15 += TwoZetaInv * 1.000000 * (dd_0.x_6_5 - dd_1.x_6_5); 
  x_10_16 = PBy * fd_0.x_10_9 - PCy * fd_1.x_10_9; 
  x_10_16 += TwoZetaInv * 1.000000 * (dd_0.x_6_9 - dd_1.x_6_9); 
  x_10_17 = PBx * fd_0.x_10_7 - PCx * fd_1.x_10_7; 
  x_10_17 += TwoZetaInv * 2.000000 * (fp_0.x_10_1 - fp_1.x_10_1); 
  x_10_17 += TwoZetaInv * 1.000000 * (dd_0.x_5_7 - dd_1.x_5_7); 
  x_10_18 = PBy * fd_0.x_10_8 - PCy * fd_1.x_10_8; 
  x_10_18 += TwoZetaInv * 2.000000 * (fp_0.x_10_2 - fp_1.x_10_2); 
  x_10_18 += TwoZetaInv * 1.000000 * (dd_0.x_6_8 - dd_1.x_6_8); 
  x_10_19 = PBz * fd_0.x_10_9 - PCz * fd_1.x_10_9; 
  x_10_19 += TwoZetaInv * 2.000000 * (fp_0.x_10_3 - fp_1.x_10_3); 
  x_10_19 += TwoZetaInv * 1.000000 * (dd_0.x_4_9 - dd_1.x_4_9); 
  x_11_10 = PBx * fd_0.x_11_5 - PCx * fd_1.x_11_5; 
  x_11_10 += TwoZetaInv * 2.000000 * (dd_0.x_4_5 - dd_1.x_4_5); 
  x_11_11 = PBx * fd_0.x_11_4 - PCx * fd_1.x_11_4; 
  x_11_11 += TwoZetaInv * 1.000000 * (fp_0.x_11_2 - fp_1.x_11_2); 
  x_11_11 += TwoZetaInv * 2.000000 * (dd_0.x_4_4 - dd_1.x_4_4); 
  x_11_12 = PBx * fd_0.x_11_8 - PCx * fd_1.x_11_8; 
  x_11_12 += TwoZetaInv * 2.000000 * (dd_0.x_4_8 - dd_1.x_4_8); 
  x_11_13 = PBx * fd_0.x_11_6 - PCx * fd_1.x_11_6; 
  x_11_13 += TwoZetaInv * 1.000000 * (fp_0.x_11_3 - fp_1.x_11_3); 
  x_11_13 += TwoZetaInv * 2.000000 * (dd_0.x_4_6 - dd_1.x_4_6); 
  x_11_14 = PBx * fd_0.x_11_9 - PCx * fd_1.x_11_9; 
  x_11_14 += TwoZetaInv * 2.000000 * (dd_0.x_4_9 - dd_1.x_4_9); 
  x_11_15 = PBy * fd_0.x_11_5 - PCy * fd_1.x_11_5; 
  x_11_15 += TwoZetaInv * 1.000000 * (fp_0.x_11_3 - fp_1.x_11_3); 
  x_11_15 += TwoZetaInv * 1.000000 * (dd_0.x_7_5 - dd_1.x_7_5); 
  x_11_16 = PBy * fd_0.x_11_9 - PCy * fd_1.x_11_9; 
  x_11_16 += TwoZetaInv * 1.000000 * (dd_0.x_7_9 - dd_1.x_7_9); 
  x_11_17 = PBx * fd_0.x_11_7 - PCx * fd_1.x_11_7; 
  x_11_17 += TwoZetaInv * 2.000000 * (fp_0.x_11_1 - fp_1.x_11_1); 
  x_11_17 += TwoZetaInv * 2.000000 * (dd_0.x_4_7 - dd_1.x_4_7); 
  x_11_18 = PBy * fd_0.x_11_8 - PCy * fd_1.x_11_8; 
  x_11_18 += TwoZetaInv * 2.000000 * (fp_0.x_11_2 - fp_1.x_11_2); 
  x_11_18 += TwoZetaInv * 1.000000 * (dd_0.x_7_8 - dd_1.x_7_8); 
  x_11_19 = PBz * fd_0.x_11_9 - PCz * fd_1.x_11_9; 
  x_11_19 += TwoZetaInv * 2.000000 * (fp_0.x_11_3 - fp_1.x_11_3); 
  x_12_10 = PBx * fd_0.x_12_5 - PCx * fd_1.x_12_5; 
  x_12_10 += TwoZetaInv * 1.000000 * (dd_0.x_8_5 - dd_1.x_8_5); 
  x_12_11 = PBx * fd_0.x_12_4 - PCx * fd_1.x_12_4; 
  x_12_11 += TwoZetaInv * 1.000000 * (fp_0.x_12_2 - fp_1.x_12_2); 
  x_12_11 += TwoZetaInv * 1.000000 * (dd_0.x_8_4 - dd_1.x_8_4); 
  x_12_12 = PBx * fd_0.x_12_8 - PCx * fd_1.x_12_8; 
  x_12_12 += TwoZetaInv * 1.000000 * (dd_0.x_8_8 - dd_1.x_8_8); 
  x_12_13 = PBx * fd_0.x_12_6 - PCx * fd_1.x_12_6; 
  x_12_13 += TwoZetaInv * 1.000000 * (fp_0.x_12_3 - fp_1.x_12_3); 
  x_12_13 += TwoZetaInv * 1.000000 * (dd_0.x_8_6 - dd_1.x_8_6); 
  x_12_14 = PBx * fd_0.x_12_9 - PCx * fd_1.x_12_9; 
  x_12_14 += TwoZetaInv * 1.000000 * (dd_0.x_8_9 - dd_1.x_8_9); 
  x_12_15 = PBy * fd_0.x_12_5 - PCy * fd_1.x_12_5; 
  x_12_15 += TwoZetaInv * 1.000000 * (fp_0.x_12_3 - fp_1.x_12_3); 
  x_12_15 += TwoZetaInv * 2.000000 * (dd_0.x_4_5 - dd_1.x_4_5); 
  x_12_16 = PBy * fd_0.x_12_9 - PCy * fd_1.x_12_9; 
  x_12_16 += TwoZetaInv * 2.000000 * (dd_0.x_4_9 - dd_1.x_4_9); 
  x_12_17 = PBx * fd_0.x_12_7 - PCx * fd_1.x_12_7; 
  x_12_17 += TwoZetaInv * 2.000000 * (fp_0.x_12_1 - fp_1.x_12_1); 
  x_12_17 += TwoZetaInv * 1.000000 * (dd_0.x_8_7 - dd_1.x_8_7); 
  x_12_18 = PBy * fd_0.x_12_8 - PCy * fd_1.x_12_8; 
  x_12_18 += TwoZetaInv * 2.000000 * (fp_0.x_12_2 - fp_1.x_12_2); 
  x_12_18 += TwoZetaInv * 2.000000 * (dd_0.x_4_8 - dd_1.x_4_8); 
  x_12_19 = PBz * fd_0.x_12_9 - PCz * fd_1.x_12_9; 
  x_12_19 += TwoZetaInv * 2.000000 * (fp_0.x_12_3 - fp_1.x_12_3); 
  x_13_10 = PBx * fd_0.x_13_5 - PCx * fd_1.x_13_5; 
  x_13_10 += TwoZetaInv * 2.000000 * (dd_0.x_6_5 - dd_1.x_6_5); 
  x_13_11 = PBx * fd_0.x_13_4 - PCx * fd_1.x_13_4; 
  x_13_11 += TwoZetaInv * 1.000000 * (fp_0.x_13_2 - fp_1.x_13_2); 
  x_13_11 += TwoZetaInv * 2.000000 * (dd_0.x_6_4 - dd_1.x_6_4); 
  x_13_12 = PBx * fd_0.x_13_8 - PCx * fd_1.x_13_8; 
  x_13_12 += TwoZetaInv * 2.000000 * (dd_0.x_6_8 - dd_1.x_6_8); 
  x_13_13 = PBx * fd_0.x_13_6 - PCx * fd_1.x_13_6; 
  x_13_13 += TwoZetaInv * 1.000000 * (fp_0.x_13_3 - fp_1.x_13_3); 
  x_13_13 += TwoZetaInv * 2.000000 * (dd_0.x_6_6 - dd_1.x_6_6); 
  x_13_14 = PBx * fd_0.x_13_9 - PCx * fd_1.x_13_9; 
  x_13_14 += TwoZetaInv * 2.000000 * (dd_0.x_6_9 - dd_1.x_6_9); 
  x_13_15 = PBy * fd_0.x_13_5 - PCy * fd_1.x_13_5; 
  x_13_15 += TwoZetaInv * 1.000000 * (fp_0.x_13_3 - fp_1.x_13_3); 
  x_13_16 = PBy * fd_0.x_13_9 - PCy * fd_1.x_13_9; 
  x_13_17 = PBx * fd_0.x_13_7 - PCx * fd_1.x_13_7; 
  x_13_17 += TwoZetaInv * 2.000000 * (fp_0.x_13_1 - fp_1.x_13_1); 
  x_13_17 += TwoZetaInv * 2.000000 * (dd_0.x_6_7 - dd_1.x_6_7); 
  x_13_18 = PBy * fd_0.x_13_8 - PCy * fd_1.x_13_8; 
  x_13_18 += TwoZetaInv * 2.000000 * (fp_0.x_13_2 - fp_1.x_13_2); 
  x_13_19 = PBz * fd_0.x_13_9 - PCz * fd_1.x_13_9; 
  x_13_19 += TwoZetaInv * 2.000000 * (fp_0.x_13_3 - fp_1.x_13_3); 
  x_13_19 += TwoZetaInv * 1.000000 * (dd_0.x_7_9 - dd_1.x_7_9); 
  x_14_10 = PBx * fd_0.x_14_5 - PCx * fd_1.x_14_5; 
  x_14_10 += TwoZetaInv * 1.000000 * (dd_0.x_9_5 - dd_1.x_9_5); 
  x_14_11 = PBx * fd_0.x_14_4 - PCx * fd_1.x_14_4; 
  x_14_11 += TwoZetaInv * 1.000000 * (fp_0.x_14_2 - fp_1.x_14_2); 
  x_14_11 += TwoZetaInv * 1.000000 * (dd_0.x_9_4 - dd_1.x_9_4); 
  x_14_12 = PBx * fd_0.x_14_8 - PCx * fd_1.x_14_8; 
  x_14_12 += TwoZetaInv * 1.000000 * (dd_0.x_9_8 - dd_1.x_9_8); 
  x_14_13 = PBx * fd_0.x_14_6 - PCx * fd_1.x_14_6; 
  x_14_13 += TwoZetaInv * 1.000000 * (fp_0.x_14_3 - fp_1.x_14_3); 
  x_14_13 += TwoZetaInv * 1.000000 * (dd_0.x_9_6 - dd_1.x_9_6); 
  x_14_14 = PBx * fd_0.x_14_9 - PCx * fd_1.x_14_9; 
  x_14_14 += TwoZetaInv * 1.000000 * (dd_0.x_9_9 - dd_1.x_9_9); 
  x_14_15 = PBy * fd_0.x_14_5 - PCy * fd_1.x_14_5; 
  x_14_15 += TwoZetaInv * 1.000000 * (fp_0.x_14_3 - fp_1.x_14_3); 
  x_14_16 = PBy * fd_0.x_14_9 - PCy * fd_1.x_14_9; 
  x_14_17 = PBx * fd_0.x_14_7 - PCx * fd_1.x_14_7; 
  x_14_17 += TwoZetaInv * 2.000000 * (fp_0.x_14_1 - fp_1.x_14_1); 
  x_14_17 += TwoZetaInv * 1.000000 * (dd_0.x_9_7 - dd_1.x_9_7); 
  x_14_18 = PBy * fd_0.x_14_8 - PCy * fd_1.x_14_8; 
  x_14_18 += TwoZetaInv * 2.000000 * (fp_0.x_14_2 - fp_1.x_14_2); 
  x_14_19 = PBz * fd_0.x_14_9 - PCz * fd_1.x_14_9; 
  x_14_19 += TwoZetaInv * 2.000000 * (fp_0.x_14_3 - fp_1.x_14_3); 
  x_14_19 += TwoZetaInv * 2.000000 * (dd_0.x_6_9 - dd_1.x_6_9); 
  x_15_10 = PBx * fd_0.x_15_5 - PCx * fd_1.x_15_5; 
  x_15_11 = PBx * fd_0.x_15_4 - PCx * fd_1.x_15_4; 
  x_15_11 += TwoZetaInv * 1.000000 * (fp_0.x_15_2 - fp_1.x_15_2); 
  x_15_12 = PBx * fd_0.x_15_8 - PCx * fd_1.x_15_8; 
  x_15_13 = PBx * fd_0.x_15_6 - PCx * fd_1.x_15_6; 
  x_15_13 += TwoZetaInv * 1.000000 * (fp_0.x_15_3 - fp_1.x_15_3); 
  x_15_14 = PBx * fd_0.x_15_9 - PCx * fd_1.x_15_9; 
  x_15_15 = PBy * fd_0.x_15_5 - PCy * fd_1.x_15_5; 
  x_15_15 += TwoZetaInv * 1.000000 * (fp_0.x_15_3 - fp_1.x_15_3); 
  x_15_15 += TwoZetaInv * 2.000000 * (dd_0.x_5_5 - dd_1.x_5_5); 
  x_15_16 = PBy * fd_0.x_15_9 - PCy * fd_1.x_15_9; 
  x_15_16 += TwoZetaInv * 2.000000 * (dd_0.x_5_9 - dd_1.x_5_9); 
  x_15_17 = PBx * fd_0.x_15_7 - PCx * fd_1.x_15_7; 
  x_15_17 += TwoZetaInv * 2.000000 * (fp_0.x_15_1 - fp_1.x_15_1); 
  x_15_18 = PBy * fd_0.x_15_8 - PCy * fd_1.x_15_8; 
  x_15_18 += TwoZetaInv * 2.000000 * (fp_0.x_15_2 - fp_1.x_15_2); 
  x_15_18 += TwoZetaInv * 2.000000 * (dd_0.x_5_8 - dd_1.x_5_8); 
  x_15_19 = PBz * fd_0.x_15_9 - PCz * fd_1.x_15_9; 
  x_15_19 += TwoZetaInv * 2.000000 * (fp_0.x_15_3 - fp_1.x_15_3); 
  x_15_19 += TwoZetaInv * 1.000000 * (dd_0.x_8_9 - dd_1.x_8_9); 
  x_16_10 = PBx * fd_0.x_16_5 - PCx * fd_1.x_16_5; 
  x_16_11 = PBx * fd_0.x_16_4 - PCx * fd_1.x_16_4; 
  x_16_11 += TwoZetaInv * 1.000000 * (fp_0.x_16_2 - fp_1.x_16_2); 
  x_16_12 = PBx * fd_0.x_16_8 - PCx * fd_1.x_16_8; 
  x_16_13 = PBx * fd_0.x_16_6 - PCx * fd_1.x_16_6; 
  x_16_13 += TwoZetaInv * 1.000000 * (fp_0.x_16_3 - fp_1.x_16_3); 
  x_16_14 = PBx * fd_0.x_16_9 - PCx * fd_1.x_16_9; 
  x_16_15 = PBy * fd_0.x_16_5 - PCy * fd_1.x_16_5; 
  x_16_15 += TwoZetaInv * 1.000000 * (fp_0.x_16_3 - fp_1.x_16_3); 
  x_16_15 += TwoZetaInv * 1.000000 * (dd_0.x_9_5 - dd_1.x_9_5); 
  x_16_16 = PBy * fd_0.x_16_9 - PCy * fd_1.x_16_9; 
  x_16_16 += TwoZetaInv * 1.000000 * (dd_0.x_9_9 - dd_1.x_9_9); 
  x_16_17 = PBx * fd_0.x_16_7 - PCx * fd_1.x_16_7; 
  x_16_17 += TwoZetaInv * 2.000000 * (fp_0.x_16_1 - fp_1.x_16_1); 
  x_16_18 = PBy * fd_0.x_16_8 - PCy * fd_1.x_16_8; 
  x_16_18 += TwoZetaInv * 2.000000 * (fp_0.x_16_2 - fp_1.x_16_2); 
  x_16_18 += TwoZetaInv * 1.000000 * (dd_0.x_9_8 - dd_1.x_9_8); 
  x_16_19 = PBz * fd_0.x_16_9 - PCz * fd_1.x_16_9; 
  x_16_19 += TwoZetaInv * 2.000000 * (fp_0.x_16_3 - fp_1.x_16_3); 
  x_16_19 += TwoZetaInv * 2.000000 * (dd_0.x_5_9 - dd_1.x_5_9); 
  x_17_10 = PBx * fd_0.x_17_5 - PCx * fd_1.x_17_5; 
  x_17_10 += TwoZetaInv * 3.000000 * (dd_0.x_7_5 - dd_1.x_7_5); 
  x_17_11 = PBx * fd_0.x_17_4 - PCx * fd_1.x_17_4; 
  x_17_11 += TwoZetaInv * 1.000000 * (fp_0.x_17_2 - fp_1.x_17_2); 
  x_17_11 += TwoZetaInv * 3.000000 * (dd_0.x_7_4 - dd_1.x_7_4); 
  x_17_12 = PBx * fd_0.x_17_8 - PCx * fd_1.x_17_8; 
  x_17_12 += TwoZetaInv * 3.000000 * (dd_0.x_7_8 - dd_1.x_7_8); 
  x_17_13 = PBx * fd_0.x_17_6 - PCx * fd_1.x_17_6; 
  x_17_13 += TwoZetaInv * 1.000000 * (fp_0.x_17_3 - fp_1.x_17_3); 
  x_17_13 += TwoZetaInv * 3.000000 * (dd_0.x_7_6 - dd_1.x_7_6); 
  x_17_14 = PBx * fd_0.x_17_9 - PCx * fd_1.x_17_9; 
  x_17_14 += TwoZetaInv * 3.000000 * (dd_0.x_7_9 - dd_1.x_7_9); 
  x_17_15 = PBy * fd_0.x_17_5 - PCy * fd_1.x_17_5; 
  x_17_15 += TwoZetaInv * 1.000000 * (fp_0.x_17_3 - fp_1.x_17_3); 
  x_17_16 = PBy * fd_0.x_17_9 - PCy * fd_1.x_17_9; 
  x_17_17 = PBx * fd_0.x_17_7 - PCx * fd_1.x_17_7; 
  x_17_17 += TwoZetaInv * 2.000000 * (fp_0.x_17_1 - fp_1.x_17_1); 
  x_17_17 += TwoZetaInv * 3.000000 * (dd_0.x_7_7 - dd_1.x_7_7); 
  x_17_18 = PBy * fd_0.x_17_8 - PCy * fd_1.x_17_8; 
  x_17_18 += TwoZetaInv * 2.000000 * (fp_0.x_17_2 - fp_1.x_17_2); 
  x_17_19 = PBz * fd_0.x_17_9 - PCz * fd_1.x_17_9; 
  x_17_19 += TwoZetaInv * 2.000000 * (fp_0.x_17_3 - fp_1.x_17_3); 
  x_18_10 = PBx * fd_0.x_18_5 - PCx * fd_1.x_18_5; 
  x_18_11 = PBx * fd_0.x_18_4 - PCx * fd_1.x_18_4; 
  x_18_11 += TwoZetaInv * 1.000000 * (fp_0.x_18_2 - fp_1.x_18_2); 
  x_18_12 = PBx * fd_0.x_18_8 - PCx * fd_1.x_18_8; 
  x_18_13 = PBx * fd_0.x_18_6 - PCx * fd_1.x_18_6; 
  x_18_13 += TwoZetaInv * 1.000000 * (fp_0.x_18_3 - fp_1.x_18_3); 
  x_18_14 = PBx * fd_0.x_18_9 - PCx * fd_1.x_18_9; 
  x_18_15 = PBy * fd_0.x_18_5 - PCy * fd_1.x_18_5; 
  x_18_15 += TwoZetaInv * 1.000000 * (fp_0.x_18_3 - fp_1.x_18_3); 
  x_18_15 += TwoZetaInv * 3.000000 * (dd_0.x_8_5 - dd_1.x_8_5); 
  x_18_16 = PBy * fd_0.x_18_9 - PCy * fd_1.x_18_9; 
  x_18_16 += TwoZetaInv * 3.000000 * (dd_0.x_8_9 - dd_1.x_8_9); 
  x_18_17 = PBx * fd_0.x_18_7 - PCx * fd_1.x_18_7; 
  x_18_17 += TwoZetaInv * 2.000000 * (fp_0.x_18_1 - fp_1.x_18_1); 
  x_18_18 = PBy * fd_0.x_18_8 - PCy * fd_1.x_18_8; 
  x_18_18 += TwoZetaInv * 2.000000 * (fp_0.x_18_2 - fp_1.x_18_2); 
  x_18_18 += TwoZetaInv * 3.000000 * (dd_0.x_8_8 - dd_1.x_8_8); 
  x_18_19 = PBz * fd_0.x_18_9 - PCz * fd_1.x_18_9; 
  x_18_19 += TwoZetaInv * 2.000000 * (fp_0.x_18_3 - fp_1.x_18_3); 
  x_19_10 = PBx * fd_0.x_19_5 - PCx * fd_1.x_19_5; 
  x_19_11 = PBx * fd_0.x_19_4 - PCx * fd_1.x_19_4; 
  x_19_11 += TwoZetaInv * 1.000000 * (fp_0.x_19_2 - fp_1.x_19_2); 
  x_19_12 = PBx * fd_0.x_19_8 - PCx * fd_1.x_19_8; 
  x_19_13 = PBx * fd_0.x_19_6 - PCx * fd_1.x_19_6; 
  x_19_13 += TwoZetaInv * 1.000000 * (fp_0.x_19_3 - fp_1.x_19_3); 
  x_19_14 = PBx * fd_0.x_19_9 - PCx * fd_1.x_19_9; 
  x_19_15 = PBy * fd_0.x_19_5 - PCy * fd_1.x_19_5; 
  x_19_15 += TwoZetaInv * 1.000000 * (fp_0.x_19_3 - fp_1.x_19_3); 
  x_19_16 = PBy * fd_0.x_19_9 - PCy * fd_1.x_19_9; 
  x_19_17 = PBx * fd_0.x_19_7 - PCx * fd_1.x_19_7; 
  x_19_17 += TwoZetaInv * 2.000000 * (fp_0.x_19_1 - fp_1.x_19_1); 
  x_19_18 = PBy * fd_0.x_19_8 - PCy * fd_1.x_19_8; 
  x_19_18 += TwoZetaInv * 2.000000 * (fp_0.x_19_2 - fp_1.x_19_2); 
  x_19_19 = PBz * fd_0.x_19_9 - PCz * fd_1.x_19_9; 
  x_19_19 += TwoZetaInv * 2.000000 * (fp_0.x_19_3 - fp_1.x_19_3); 
  x_19_19 += TwoZetaInv * 3.000000 * (dd_0.x_9_9 - dd_1.x_9_9); 
#endif 

 } 
