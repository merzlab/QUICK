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


/* PS true integral, m=0 */ 
__device__ __inline__ PSint_0::PSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 0) - PCx * VY(0, 0, 1);
  x_2_0 = PAy * VY(0, 0, 0) - PCy * VY(0, 0, 1);
  x_3_0 = PAz * VY(0, 0, 0) - PCz * VY(0, 0, 1);
} 


/* PS auxilary integral, m=1 */ 
__device__ __inline__ PSint_1::PSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 1) - PCx * VY(0, 0, 2);
  x_2_0 = PAy * VY(0, 0, 1) - PCy * VY(0, 0, 2);
  x_3_0 = PAz * VY(0, 0, 1) - PCz * VY(0, 0, 2);
} 


/* PS auxilary integral, m=2 */ 
__device__ __inline__ PSint_2::PSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 2) - PCx * VY(0, 0, 3);
  x_2_0 = PAy * VY(0, 0, 2) - PCy * VY(0, 0, 3);
  x_3_0 = PAz * VY(0, 0, 2) - PCz * VY(0, 0, 3);
} 


/* PS auxilary integral, m=3 */ 
__device__ __inline__ PSint_3::PSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 3) - PCx * VY(0, 0, 4);
  x_2_0 = PAy * VY(0, 0, 3) - PCy * VY(0, 0, 4);
  x_3_0 = PAz * VY(0, 0, 3) - PCz * VY(0, 0, 4);
} 


/* PS auxilary integral, m=4 */ 
__device__ __inline__ PSint_4::PSint_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 4) - PCx * VY(0, 0, 5);
  x_2_0 = PAy * VY(0, 0, 4) - PCy * VY(0, 0, 5);
  x_3_0 = PAz * VY(0, 0, 4) - PCz * VY(0, 0, 5);
} 


/* PS auxilary integral, m=5 */ 
__device__ __inline__ PSint_5::PSint_5(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 5) - PCx * VY(0, 0, 6);
  x_2_0 = PAy * VY(0, 0, 5) - PCy * VY(0, 0, 6);
  x_3_0 = PAz * VY(0, 0, 5) - PCz * VY(0, 0, 6);
} 


/* SP true integral, m=0 */ 
__device__ __inline__ SPint_0::SPint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 0) - PCx * VY(0, 0, 1);
  x_0_2 = PBy * VY(0, 0, 0) - PCy * VY(0, 0, 1);
  x_0_3 = PBz * VY(0, 0, 0) - PCz * VY(0, 0, 1);
} 


/* SP auxilary integral, m=1 */ 
__device__ __inline__ SPint_1::SPint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 1) - PCx * VY(0, 0, 2);
  x_0_2 = PBy * VY(0, 0, 1) - PCy * VY(0, 0, 2);
  x_0_3 = PBz * VY(0, 0, 1) - PCz * VY(0, 0, 2);
} 


/* SP auxilary integral, m=2 */ 
__device__ __inline__ SPint_2::SPint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 2) - PCx * VY(0, 0, 3);
  x_0_2 = PBy * VY(0, 0, 2) - PCy * VY(0, 0, 3);
  x_0_3 = PBz * VY(0, 0, 2) - PCz * VY(0, 0, 3);
} 


/* SP auxilary integral, m=3 */ 
__device__ __inline__ SPint_3::SPint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 3) - PCx * VY(0, 0, 4);
  x_0_2 = PBy * VY(0, 0, 3) - PCy * VY(0, 0, 4);
  x_0_3 = PBz * VY(0, 0, 3) - PCz * VY(0, 0, 4);
} 


/* SP auxilary integral, m=4 */ 
__device__ __inline__ SPint_4::SPint_4(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 4) - PCx * VY(0, 0, 5);
  x_0_2 = PBy * VY(0, 0, 4) - PCy * VY(0, 0, 5);
  x_0_3 = PBz * VY(0, 0, 4) - PCz * VY(0, 0, 5);
} 


/* SP auxilary integral, m=5 */ 
__device__ __inline__ SPint_5::SPint_5(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 5) - PCx * VY(0, 0, 6);
  x_0_2 = PBy * VY(0, 0, 5) - PCy * VY(0, 0, 6);
  x_0_3 = PBz * VY(0, 0, 5) - PCz * VY(0, 0, 6);
} 


/* PP true integral, m=0 */ 
__device__ __inline__ PPint_0::PPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 

  x_1_1 = PBx * ps_0.x_1_0 - PCx * ps_1.x_1_0; 
  x_1_1 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_1_2 = PBy * ps_0.x_1_0 - PCy * ps_1.x_1_0; 
  x_1_3 = PBz * ps_0.x_1_0 - PCz * ps_1.x_1_0; 
  x_2_1 = PBx * ps_0.x_2_0 - PCx * ps_1.x_2_0; 
  x_2_2 = PBy * ps_0.x_2_0 - PCy * ps_1.x_2_0; 
  x_2_2 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_2_3 = PBz * ps_0.x_2_0 - PCz * ps_1.x_2_0; 
  x_3_1 = PBx * ps_0.x_3_0 - PCx * ps_1.x_3_0; 
  x_3_2 = PBy * ps_0.x_3_0 - PCy * ps_1.x_3_0; 
  x_3_3 = PBz * ps_0.x_3_0 - PCz * ps_1.x_3_0; 
  x_3_3 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 

 } 

/* PP auxilary integral, m=1 */ 
__device__ __inline__ PPint_1::PPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 

  x_1_1 = PBx * ps_1.x_1_0 - PCx * ps_2.x_1_0; 
  x_1_1 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_1_2 = PBy * ps_1.x_1_0 - PCy * ps_2.x_1_0; 
  x_1_3 = PBz * ps_1.x_1_0 - PCz * ps_2.x_1_0; 
  x_2_1 = PBx * ps_1.x_2_0 - PCx * ps_2.x_2_0; 
  x_2_2 = PBy * ps_1.x_2_0 - PCy * ps_2.x_2_0; 
  x_2_2 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_2_3 = PBz * ps_1.x_2_0 - PCz * ps_2.x_2_0; 
  x_3_1 = PBx * ps_1.x_3_0 - PCx * ps_2.x_3_0; 
  x_3_2 = PBy * ps_1.x_3_0 - PCy * ps_2.x_3_0; 
  x_3_3 = PBz * ps_1.x_3_0 - PCz * ps_2.x_3_0; 
  x_3_3 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 

 } 

/* PP auxilary integral, m=2 */ 
__device__ __inline__ PPint_2::PPint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=3 

  x_1_1 = PBx * ps_2.x_1_0 - PCx * ps_3.x_1_0; 
  x_1_1 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_1_2 = PBy * ps_2.x_1_0 - PCy * ps_3.x_1_0; 
  x_1_3 = PBz * ps_2.x_1_0 - PCz * ps_3.x_1_0; 
  x_2_1 = PBx * ps_2.x_2_0 - PCx * ps_3.x_2_0; 
  x_2_2 = PBy * ps_2.x_2_0 - PCy * ps_3.x_2_0; 
  x_2_2 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_2_3 = PBz * ps_2.x_2_0 - PCz * ps_3.x_2_0; 
  x_3_1 = PBx * ps_2.x_3_0 - PCx * ps_3.x_3_0; 
  x_3_2 = PBy * ps_2.x_3_0 - PCy * ps_3.x_3_0; 
  x_3_3 = PBz * ps_2.x_3_0 - PCz * ps_3.x_3_0; 
  x_3_3 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 

 } 

/* DS true integral, m=0 */ 
__device__ __inline__ DSint_0::DSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 

  x_4_0 = PAx * ps_0.x_2_0 - PCx * ps_1.x_2_0; 
  x_5_0 = PAy * ps_0.x_3_0 - PCy * ps_1.x_3_0; 
  x_6_0 = PAx * ps_0.x_3_0 - PCx * ps_1.x_3_0; 
  x_7_0 = PAx * ps_0.x_1_0 - PCx * ps_1.x_1_0; 
  x_7_0 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_8_0 = PAy * ps_0.x_2_0 - PCy * ps_1.x_2_0; 
  x_8_0 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_9_0 = PAz * ps_0.x_3_0 - PCz * ps_1.x_3_0; 
  x_9_0 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 

 } 

/* DS auxilary integral, m=1 */ 
__device__ __inline__ DSint_1::DSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 

  x_4_0 = PAx * ps_1.x_2_0 - PCx * ps_2.x_2_0; 
  x_5_0 = PAy * ps_1.x_3_0 - PCy * ps_2.x_3_0; 
  x_6_0 = PAx * ps_1.x_3_0 - PCx * ps_2.x_3_0; 
  x_7_0 = PAx * ps_1.x_1_0 - PCx * ps_2.x_1_0; 
  x_7_0 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_8_0 = PAy * ps_1.x_2_0 - PCy * ps_2.x_2_0; 
  x_8_0 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_9_0 = PAz * ps_1.x_3_0 - PCz * ps_2.x_3_0; 
  x_9_0 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 

 } 

/* DS auxilary integral, m=2 */ 
__device__ __inline__ DSint_2::DSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=3 

  x_4_0 = PAx * ps_2.x_2_0 - PCx * ps_3.x_2_0; 
  x_5_0 = PAy * ps_2.x_3_0 - PCy * ps_3.x_3_0; 
  x_6_0 = PAx * ps_2.x_3_0 - PCx * ps_3.x_3_0; 
  x_7_0 = PAx * ps_2.x_1_0 - PCx * ps_3.x_1_0; 
  x_7_0 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_8_0 = PAy * ps_2.x_2_0 - PCy * ps_3.x_2_0; 
  x_8_0 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_9_0 = PAz * ps_2.x_3_0 - PCz * ps_3.x_3_0; 
  x_9_0 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 

 } 

/* DS auxilary integral, m=3 */ 
__device__ __inline__ DSint_3::DSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=3 
  PSint_4 ps_4(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=4 

  x_4_0 = PAx * ps_3.x_2_0 - PCx * ps_4.x_2_0; 
  x_5_0 = PAy * ps_3.x_3_0 - PCy * ps_4.x_3_0; 
  x_6_0 = PAx * ps_3.x_3_0 - PCx * ps_4.x_3_0; 
  x_7_0 = PAx * ps_3.x_1_0 - PCx * ps_4.x_1_0; 
  x_7_0 += 0.5/Zeta * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_8_0 = PAy * ps_3.x_2_0 - PCy * ps_4.x_2_0; 
  x_8_0 += 0.5/Zeta * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_9_0 = PAz * ps_3.x_3_0 - PCz * ps_4.x_3_0; 
  x_9_0 += 0.5/Zeta * (VY(0, 0, 3) - VY(0, 0, 4)); 

 } 

/* DS auxilary integral, m=4 */ 
__device__ __inline__ DSint_4::DSint_4(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_4 ps_4(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=4 
  PSint_5 ps_5(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=5 

  x_4_0 = PAx * ps_4.x_2_0 - PCx * ps_5.x_2_0; 
  x_5_0 = PAy * ps_4.x_3_0 - PCy * ps_5.x_3_0; 
  x_6_0 = PAx * ps_4.x_3_0 - PCx * ps_5.x_3_0; 
  x_7_0 = PAx * ps_4.x_1_0 - PCx * ps_5.x_1_0; 
  x_7_0 += 0.5/Zeta * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_8_0 = PAy * ps_4.x_2_0 - PCy * ps_5.x_2_0; 
  x_8_0 += 0.5/Zeta * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_9_0 = PAz * ps_4.x_3_0 - PCz * ps_5.x_3_0; 
  x_9_0 += 0.5/Zeta * (VY(0, 0, 4) - VY(0, 0, 5)); 

 } 

/* SD true integral, m=0 */ 
__device__ __inline__ SDint_0::SDint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 

  x_0_4 = PBx * sp_0.x_0_2 - PCx * sp_1.x_0_2; 
  x_0_5 = PBy * sp_0.x_0_3 - PCy * sp_1.x_0_3; 
  x_0_6 = PBx * sp_0.x_0_3 - PCx * sp_1.x_0_3; 
  x_0_7 = PBx * sp_0.x_0_1 - PCx * sp_1.x_0_1; 
  x_0_7 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_0_8 = PBy * sp_0.x_0_2 - PCy * sp_1.x_0_2; 
  x_0_8 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 
  x_0_9 = PBz * sp_0.x_0_3 - PCz * sp_1.x_0_3; 
  x_0_9 += 0.5/Zeta * (VY(0, 0, 0) - VY(0, 0, 1)); 

 } 

/* SD auxilary integral, m=1 */ 
__device__ __inline__ SDint_1::SDint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 
  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=2 

  x_0_4 = PBx * sp_1.x_0_2 - PCx * sp_2.x_0_2; 
  x_0_5 = PBy * sp_1.x_0_3 - PCy * sp_2.x_0_3; 
  x_0_6 = PBx * sp_1.x_0_3 - PCx * sp_2.x_0_3; 
  x_0_7 = PBx * sp_1.x_0_1 - PCx * sp_2.x_0_1; 
  x_0_7 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_0_8 = PBy * sp_1.x_0_2 - PCy * sp_2.x_0_2; 
  x_0_8 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 
  x_0_9 = PBz * sp_1.x_0_3 - PCz * sp_2.x_0_3; 
  x_0_9 += 0.5/Zeta * (VY(0, 0, 1) - VY(0, 0, 2)); 

 } 

/* SD auxilary integral, m=2 */ 
__device__ __inline__ SDint_2::SDint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=2 
  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=3 

  x_0_4 = PBx * sp_2.x_0_2 - PCx * sp_3.x_0_2; 
  x_0_5 = PBy * sp_2.x_0_3 - PCy * sp_3.x_0_3; 
  x_0_6 = PBx * sp_2.x_0_3 - PCx * sp_3.x_0_3; 
  x_0_7 = PBx * sp_2.x_0_1 - PCx * sp_3.x_0_1; 
  x_0_7 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_0_8 = PBy * sp_2.x_0_2 - PCy * sp_3.x_0_2; 
  x_0_8 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 
  x_0_9 = PBz * sp_2.x_0_3 - PCz * sp_3.x_0_3; 
  x_0_9 += 0.5/Zeta * (VY(0, 0, 2) - VY(0, 0, 3)); 

 } 

/* SD auxilary integral, m=3 */ 
__device__ __inline__ SDint_3::SDint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=3 
  SPint_4 sp_4(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=4 

  x_0_4 = PBx * sp_3.x_0_2 - PCx * sp_4.x_0_2; 
  x_0_5 = PBy * sp_3.x_0_3 - PCy * sp_4.x_0_3; 
  x_0_6 = PBx * sp_3.x_0_3 - PCx * sp_4.x_0_3; 
  x_0_7 = PBx * sp_3.x_0_1 - PCx * sp_4.x_0_1; 
  x_0_7 += 0.5/Zeta * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_0_8 = PBy * sp_3.x_0_2 - PCy * sp_4.x_0_2; 
  x_0_8 += 0.5/Zeta * (VY(0, 0, 3) - VY(0, 0, 4)); 
  x_0_9 = PBz * sp_3.x_0_3 - PCz * sp_4.x_0_3; 
  x_0_9 += 0.5/Zeta * (VY(0, 0, 3) - VY(0, 0, 4)); 

 } 

/* SD auxilary integral, m=4 */ 
__device__ __inline__ SDint_4::SDint_4(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_4 sp_4(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=4 
  SPint_5 sp_5(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=5 

  x_0_4 = PBx * sp_4.x_0_2 - PCx * sp_5.x_0_2; 
  x_0_5 = PBy * sp_4.x_0_3 - PCy * sp_5.x_0_3; 
  x_0_6 = PBx * sp_4.x_0_3 - PCx * sp_5.x_0_3; 
  x_0_7 = PBx * sp_4.x_0_1 - PCx * sp_5.x_0_1; 
  x_0_7 += 0.5/Zeta * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_0_8 = PBy * sp_4.x_0_2 - PCy * sp_5.x_0_2; 
  x_0_8 += 0.5/Zeta * (VY(0, 0, 4) - VY(0, 0, 5)); 
  x_0_9 = PBz * sp_4.x_0_3 - PCz * sp_5.x_0_3; 
  x_0_9 += 0.5/Zeta * (VY(0, 0, 4) - VY(0, 0, 5)); 

 } 

/* DP true integral, m=0 */ 
__device__ __inline__ DPint_0::DPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 

  x_4_1 = PBx * ds_0.x_4_0 - PCx * ds_1.x_4_0; 
  x_4_1 += 0.5/Zeta * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_4_2 = PBy * ds_0.x_4_0 - PCy * ds_1.x_4_0; 
  x_4_2 += 0.5/Zeta * 1.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_4_3 = PBz * ds_0.x_4_0 - PCz * ds_1.x_4_0; 
  x_5_1 = PBx * ds_0.x_5_0 - PCx * ds_1.x_5_0; 
  x_5_2 = PBy * ds_0.x_5_0 - PCy * ds_1.x_5_0; 
  x_5_2 += 0.5/Zeta * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_5_3 = PBz * ds_0.x_5_0 - PCz * ds_1.x_5_0; 
  x_5_3 += 0.5/Zeta * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_6_1 = PBx * ds_0.x_6_0 - PCx * ds_1.x_6_0; 
  x_6_1 += 0.5/Zeta * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_6_2 = PBy * ds_0.x_6_0 - PCy * ds_1.x_6_0; 
  x_6_3 = PBz * ds_0.x_6_0 - PCz * ds_1.x_6_0; 
  x_6_3 += 0.5/Zeta * 1.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_7_1 = PBx * ds_0.x_7_0 - PCx * ds_1.x_7_0; 
  x_7_1 += 0.5/Zeta * 2.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_7_2 = PBy * ds_0.x_7_0 - PCy * ds_1.x_7_0; 
  x_7_3 = PBz * ds_0.x_7_0 - PCz * ds_1.x_7_0; 
  x_8_1 = PBx * ds_0.x_8_0 - PCx * ds_1.x_8_0; 
  x_8_2 = PBy * ds_0.x_8_0 - PCy * ds_1.x_8_0; 
  x_8_2 += 0.5/Zeta * 2.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_8_3 = PBz * ds_0.x_8_0 - PCz * ds_1.x_8_0; 
  x_9_1 = PBx * ds_0.x_9_0 - PCx * ds_1.x_9_0; 
  x_9_2 = PBy * ds_0.x_9_0 - PCy * ds_1.x_9_0; 
  x_9_3 = PBz * ds_0.x_9_0 - PCz * ds_1.x_9_0; 
  x_9_3 += 0.5/Zeta * 2.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 

 } 

/* DP auxilary integral, m=1 */ 
__device__ __inline__ DPint_1::DPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 

  x_4_1 = PBx * ds_1.x_4_0 - PCx * ds_2.x_4_0; 
  x_4_1 += 0.5/Zeta * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_4_2 = PBy * ds_1.x_4_0 - PCy * ds_2.x_4_0; 
  x_4_2 += 0.5/Zeta * 1.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_4_3 = PBz * ds_1.x_4_0 - PCz * ds_2.x_4_0; 
  x_5_1 = PBx * ds_1.x_5_0 - PCx * ds_2.x_5_0; 
  x_5_2 = PBy * ds_1.x_5_0 - PCy * ds_2.x_5_0; 
  x_5_2 += 0.5/Zeta * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_5_3 = PBz * ds_1.x_5_0 - PCz * ds_2.x_5_0; 
  x_5_3 += 0.5/Zeta * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_6_1 = PBx * ds_1.x_6_0 - PCx * ds_2.x_6_0; 
  x_6_1 += 0.5/Zeta * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_6_2 = PBy * ds_1.x_6_0 - PCy * ds_2.x_6_0; 
  x_6_3 = PBz * ds_1.x_6_0 - PCz * ds_2.x_6_0; 
  x_6_3 += 0.5/Zeta * 1.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_7_1 = PBx * ds_1.x_7_0 - PCx * ds_2.x_7_0; 
  x_7_1 += 0.5/Zeta * 2.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_7_2 = PBy * ds_1.x_7_0 - PCy * ds_2.x_7_0; 
  x_7_3 = PBz * ds_1.x_7_0 - PCz * ds_2.x_7_0; 
  x_8_1 = PBx * ds_1.x_8_0 - PCx * ds_2.x_8_0; 
  x_8_2 = PBy * ds_1.x_8_0 - PCy * ds_2.x_8_0; 
  x_8_2 += 0.5/Zeta * 2.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_8_3 = PBz * ds_1.x_8_0 - PCz * ds_2.x_8_0; 
  x_9_1 = PBx * ds_1.x_9_0 - PCx * ds_2.x_9_0; 
  x_9_2 = PBy * ds_1.x_9_0 - PCy * ds_2.x_9_0; 
  x_9_3 = PBz * ds_1.x_9_0 - PCz * ds_2.x_9_0; 
  x_9_3 += 0.5/Zeta * 2.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 

 } 

/* DP auxilary integral, m=2 */ 
__device__ __inline__ DPint_2::DPint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=3 

  x_4_1 = PBx * ds_2.x_4_0 - PCx * ds_3.x_4_0; 
  x_4_1 += 0.5/Zeta * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_4_2 = PBy * ds_2.x_4_0 - PCy * ds_3.x_4_0; 
  x_4_2 += 0.5/Zeta * 1.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_4_3 = PBz * ds_2.x_4_0 - PCz * ds_3.x_4_0; 
  x_5_1 = PBx * ds_2.x_5_0 - PCx * ds_3.x_5_0; 
  x_5_2 = PBy * ds_2.x_5_0 - PCy * ds_3.x_5_0; 
  x_5_2 += 0.5/Zeta * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_5_3 = PBz * ds_2.x_5_0 - PCz * ds_3.x_5_0; 
  x_5_3 += 0.5/Zeta * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_6_1 = PBx * ds_2.x_6_0 - PCx * ds_3.x_6_0; 
  x_6_1 += 0.5/Zeta * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_6_2 = PBy * ds_2.x_6_0 - PCy * ds_3.x_6_0; 
  x_6_3 = PBz * ds_2.x_6_0 - PCz * ds_3.x_6_0; 
  x_6_3 += 0.5/Zeta * 1.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_7_1 = PBx * ds_2.x_7_0 - PCx * ds_3.x_7_0; 
  x_7_1 += 0.5/Zeta * 2.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_7_2 = PBy * ds_2.x_7_0 - PCy * ds_3.x_7_0; 
  x_7_3 = PBz * ds_2.x_7_0 - PCz * ds_3.x_7_0; 
  x_8_1 = PBx * ds_2.x_8_0 - PCx * ds_3.x_8_0; 
  x_8_2 = PBy * ds_2.x_8_0 - PCy * ds_3.x_8_0; 
  x_8_2 += 0.5/Zeta * 2.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_8_3 = PBz * ds_2.x_8_0 - PCz * ds_3.x_8_0; 
  x_9_1 = PBx * ds_2.x_9_0 - PCx * ds_3.x_9_0; 
  x_9_2 = PBy * ds_2.x_9_0 - PCy * ds_3.x_9_0; 
  x_9_3 = PBz * ds_2.x_9_0 - PCz * ds_3.x_9_0; 
  x_9_3 += 0.5/Zeta * 2.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 

 } 

/* PD true integral, m=0 */ 
__device__ __inline__ PDint_0::PDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 
  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 

  x_1_4 = PAx * sd_0.x_0_4 - PCx * sd_1.x_0_4; 
  x_1_4 += 0.5/Zeta * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_2_4 = PAy * sd_0.x_0_4 - PCy * sd_1.x_0_4; 
  x_2_4 += 0.5/Zeta * 1.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_3_4 = PAz * sd_0.x_0_4 - PCz * sd_1.x_0_4; 
  x_1_5 = PAx * sd_0.x_0_5 - PCx * sd_1.x_0_5; 
  x_2_5 = PAy * sd_0.x_0_5 - PCy * sd_1.x_0_5; 
  x_2_5 += 0.5/Zeta * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_3_5 = PAz * sd_0.x_0_5 - PCz * sd_1.x_0_5; 
  x_3_5 += 0.5/Zeta * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_1_6 = PAx * sd_0.x_0_6 - PCx * sd_1.x_0_6; 
  x_1_6 += 0.5/Zeta * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_2_6 = PAy * sd_0.x_0_6 - PCy * sd_1.x_0_6; 
  x_3_6 = PAz * sd_0.x_0_6 - PCz * sd_1.x_0_6; 
  x_3_6 += 0.5/Zeta * 1.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_1_7 = PAx * sd_0.x_0_7 - PCx * sd_1.x_0_7; 
  x_1_7 += 0.5/Zeta * 2.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_2_7 = PAy * sd_0.x_0_7 - PCy * sd_1.x_0_7; 
  x_3_7 = PAz * sd_0.x_0_7 - PCz * sd_1.x_0_7; 
  x_1_8 = PAx * sd_0.x_0_8 - PCx * sd_1.x_0_8; 
  x_2_8 = PAy * sd_0.x_0_8 - PCy * sd_1.x_0_8; 
  x_2_8 += 0.5/Zeta * 2.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_3_8 = PAz * sd_0.x_0_8 - PCz * sd_1.x_0_8; 
  x_1_9 = PAx * sd_0.x_0_9 - PCx * sd_1.x_0_9; 
  x_2_9 = PAy * sd_0.x_0_9 - PCy * sd_1.x_0_9; 
  x_3_9 = PAz * sd_0.x_0_9 - PCz * sd_1.x_0_9; 
  x_3_9 += 0.5/Zeta * 2.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 

 } 

/* PD auxilary integral, m=1 */ 
__device__ __inline__ PDint_1::PDint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 
  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=2 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=2 

  x_1_4 = PAx * sd_1.x_0_4 - PCx * sd_2.x_0_4; 
  x_1_4 += 0.5/Zeta * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_2_4 = PAy * sd_1.x_0_4 - PCy * sd_2.x_0_4; 
  x_2_4 += 0.5/Zeta * 1.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_3_4 = PAz * sd_1.x_0_4 - PCz * sd_2.x_0_4; 
  x_1_5 = PAx * sd_1.x_0_5 - PCx * sd_2.x_0_5; 
  x_2_5 = PAy * sd_1.x_0_5 - PCy * sd_2.x_0_5; 
  x_2_5 += 0.5/Zeta * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_3_5 = PAz * sd_1.x_0_5 - PCz * sd_2.x_0_5; 
  x_3_5 += 0.5/Zeta * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_1_6 = PAx * sd_1.x_0_6 - PCx * sd_2.x_0_6; 
  x_1_6 += 0.5/Zeta * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_2_6 = PAy * sd_1.x_0_6 - PCy * sd_2.x_0_6; 
  x_3_6 = PAz * sd_1.x_0_6 - PCz * sd_2.x_0_6; 
  x_3_6 += 0.5/Zeta * 1.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_1_7 = PAx * sd_1.x_0_7 - PCx * sd_2.x_0_7; 
  x_1_7 += 0.5/Zeta * 2.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_2_7 = PAy * sd_1.x_0_7 - PCy * sd_2.x_0_7; 
  x_3_7 = PAz * sd_1.x_0_7 - PCz * sd_2.x_0_7; 
  x_1_8 = PAx * sd_1.x_0_8 - PCx * sd_2.x_0_8; 
  x_2_8 = PAy * sd_1.x_0_8 - PCy * sd_2.x_0_8; 
  x_2_8 += 0.5/Zeta * 2.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_3_8 = PAz * sd_1.x_0_8 - PCz * sd_2.x_0_8; 
  x_1_9 = PAx * sd_1.x_0_9 - PCx * sd_2.x_0_9; 
  x_2_9 = PAy * sd_1.x_0_9 - PCy * sd_2.x_0_9; 
  x_3_9 = PAz * sd_1.x_0_9 - PCz * sd_2.x_0_9; 
  x_3_9 += 0.5/Zeta * 2.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 

 } 

/* PD auxilary integral, m=2 */ 
__device__ __inline__ PDint_2::PDint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=2 
  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=3 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=3 

  x_1_4 = PAx * sd_2.x_0_4 - PCx * sd_3.x_0_4; 
  x_1_4 += 0.5/Zeta * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_2_4 = PAy * sd_2.x_0_4 - PCy * sd_3.x_0_4; 
  x_2_4 += 0.5/Zeta * 1.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_3_4 = PAz * sd_2.x_0_4 - PCz * sd_3.x_0_4; 
  x_1_5 = PAx * sd_2.x_0_5 - PCx * sd_3.x_0_5; 
  x_2_5 = PAy * sd_2.x_0_5 - PCy * sd_3.x_0_5; 
  x_2_5 += 0.5/Zeta * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_3_5 = PAz * sd_2.x_0_5 - PCz * sd_3.x_0_5; 
  x_3_5 += 0.5/Zeta * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_1_6 = PAx * sd_2.x_0_6 - PCx * sd_3.x_0_6; 
  x_1_6 += 0.5/Zeta * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_2_6 = PAy * sd_2.x_0_6 - PCy * sd_3.x_0_6; 
  x_3_6 = PAz * sd_2.x_0_6 - PCz * sd_3.x_0_6; 
  x_3_6 += 0.5/Zeta * 1.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_1_7 = PAx * sd_2.x_0_7 - PCx * sd_3.x_0_7; 
  x_1_7 += 0.5/Zeta * 2.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_2_7 = PAy * sd_2.x_0_7 - PCy * sd_3.x_0_7; 
  x_3_7 = PAz * sd_2.x_0_7 - PCz * sd_3.x_0_7; 
  x_1_8 = PAx * sd_2.x_0_8 - PCx * sd_3.x_0_8; 
  x_2_8 = PAy * sd_2.x_0_8 - PCy * sd_3.x_0_8; 
  x_2_8 += 0.5/Zeta * 2.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_3_8 = PAz * sd_2.x_0_8 - PCz * sd_3.x_0_8; 
  x_1_9 = PAx * sd_2.x_0_9 - PCx * sd_3.x_0_9; 
  x_2_9 = PAy * sd_2.x_0_9 - PCy * sd_3.x_0_9; 
  x_3_9 = PAz * sd_2.x_0_9 - PCz * sd_3.x_0_9; 
  x_3_9 += 0.5/Zeta * 2.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 

 } 

/* DD true integral, m=0 */ 
__device__ __inline__ DDint_0::DDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PPint_0 pp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|p] for m=0 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=0 
  DPint_0 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=0 
  PPint_1 pp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|p] for m=1 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 
  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=1 

  x_4_4 = PBx * dp_0.x_4_2 - PCx * dp_1.x_4_2; 
  x_4_4 += 0.5/Zeta * 1.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  x_5_4 = PBy * dp_0.x_4_3 - PCy * dp_1.x_4_3; 
  x_5_4 += 0.5/Zeta * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_6_4 = PBx * dp_0.x_4_3 - PCx * dp_1.x_4_3; 
  x_6_4 += 0.5/Zeta * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_7_4 = PBx * dp_0.x_4_1 - PCx * dp_1.x_4_1; 
  x_7_4 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_7_4 += 0.5/Zeta * 1.000000 * (pp_0.x_2_1 - pp_1.x_2_1); 
  x_8_4 = PBy * dp_0.x_4_2 - PCy * dp_1.x_4_2; 
  x_8_4 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_8_4 += 0.5/Zeta * 1.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  x_9_4 = PBz * dp_0.x_4_3 - PCz * dp_1.x_4_3; 
  x_9_4 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_4_5 = PBx * dp_0.x_5_2 - PCx * dp_1.x_5_2; 
  x_5_5 = PBy * dp_0.x_5_3 - PCy * dp_1.x_5_3; 
  x_5_5 += 0.5/Zeta * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  x_6_5 = PBx * dp_0.x_5_3 - PCx * dp_1.x_5_3; 
  x_7_5 = PBx * dp_0.x_5_1 - PCx * dp_1.x_5_1; 
  x_7_5 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_8_5 = PBy * dp_0.x_5_2 - PCy * dp_1.x_5_2; 
  x_8_5 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_8_5 += 0.5/Zeta * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  x_9_5 = PBz * dp_0.x_5_3 - PCz * dp_1.x_5_3; 
  x_9_5 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_9_5 += 0.5/Zeta * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_4_6 = PBx * dp_0.x_6_2 - PCx * dp_1.x_6_2; 
  x_4_6 += 0.5/Zeta * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  x_5_6 = PBy * dp_0.x_6_3 - PCy * dp_1.x_6_3; 
  x_6_6 = PBx * dp_0.x_6_3 - PCx * dp_1.x_6_3; 
  x_6_6 += 0.5/Zeta * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  x_7_6 = PBx * dp_0.x_6_1 - PCx * dp_1.x_6_1; 
  x_7_6 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_7_6 += 0.5/Zeta * 1.000000 * (pp_0.x_3_1 - pp_1.x_3_1); 
  x_8_6 = PBy * dp_0.x_6_2 - PCy * dp_1.x_6_2; 
  x_8_6 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_9_6 = PBz * dp_0.x_6_3 - PCz * dp_1.x_6_3; 
  x_9_6 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_9_6 += 0.5/Zeta * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_4_7 = PBx * dp_0.x_7_2 - PCx * dp_1.x_7_2; 
  x_4_7 += 0.5/Zeta * 2.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  x_5_7 = PBy * dp_0.x_7_3 - PCy * dp_1.x_7_3; 
  x_6_7 = PBx * dp_0.x_7_3 - PCx * dp_1.x_7_3; 
  x_6_7 += 0.5/Zeta * 2.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_7_7 = PBx * dp_0.x_7_1 - PCx * dp_1.x_7_1; 
  x_7_7 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_7_7 += 0.5/Zeta * 2.000000 * (pp_0.x_1_1 - pp_1.x_1_1); 
  x_8_7 = PBy * dp_0.x_7_2 - PCy * dp_1.x_7_2; 
  x_8_7 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_9_7 = PBz * dp_0.x_7_3 - PCz * dp_1.x_7_3; 
  x_9_7 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_4_8 = PBx * dp_0.x_8_2 - PCx * dp_1.x_8_2; 
  x_5_8 = PBy * dp_0.x_8_3 - PCy * dp_1.x_8_3; 
  x_5_8 += 0.5/Zeta * 2.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_6_8 = PBx * dp_0.x_8_3 - PCx * dp_1.x_8_3; 
  x_7_8 = PBx * dp_0.x_8_1 - PCx * dp_1.x_8_1; 
  x_7_8 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_8_8 = PBy * dp_0.x_8_2 - PCy * dp_1.x_8_2; 
  x_8_8 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_8_8 += 0.5/Zeta * 2.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  x_9_8 = PBz * dp_0.x_8_3 - PCz * dp_1.x_8_3; 
  x_9_8 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_4_9 = PBx * dp_0.x_9_2 - PCx * dp_1.x_9_2; 
  x_5_9 = PBy * dp_0.x_9_3 - PCy * dp_1.x_9_3; 
  x_6_9 = PBx * dp_0.x_9_3 - PCx * dp_1.x_9_3; 
  x_7_9 = PBx * dp_0.x_9_1 - PCx * dp_1.x_9_1; 
  x_7_9 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_8_9 = PBy * dp_0.x_9_2 - PCy * dp_1.x_9_2; 
  x_8_9 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_9 = PBz * dp_0.x_9_3 - PCz * dp_1.x_9_3; 
  x_9_9 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_9 += 0.5/Zeta * 2.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 

 } 

/* DD auxilary integral, m=1 */ 
__device__ __inline__ DDint_1::DDint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PPint_1 pp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|p] for m=1 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 
  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=1 
  PPint_2 pp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|p] for m=2 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 
  DPint_2 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=2 

  x_4_4 = PBx * dp_1.x_4_2 - PCx * dp_2.x_4_2; 
  x_4_4 += 0.5/Zeta * 1.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
  x_5_4 = PBy * dp_1.x_4_3 - PCy * dp_2.x_4_3; 
  x_5_4 += 0.5/Zeta * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  x_6_4 = PBx * dp_1.x_4_3 - PCx * dp_2.x_4_3; 
  x_6_4 += 0.5/Zeta * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  x_7_4 = PBx * dp_1.x_4_1 - PCx * dp_2.x_4_1; 
  x_7_4 += 0.5/Zeta * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_7_4 += 0.5/Zeta * 1.000000 * (pp_1.x_2_1 - pp_2.x_2_1); 
  x_8_4 = PBy * dp_1.x_4_2 - PCy * dp_2.x_4_2; 
  x_8_4 += 0.5/Zeta * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_8_4 += 0.5/Zeta * 1.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
  x_9_4 = PBz * dp_1.x_4_3 - PCz * dp_2.x_4_3; 
  x_9_4 += 0.5/Zeta * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_4_5 = PBx * dp_1.x_5_2 - PCx * dp_2.x_5_2; 
  x_5_5 = PBy * dp_1.x_5_3 - PCy * dp_2.x_5_3; 
  x_5_5 += 0.5/Zeta * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  x_6_5 = PBx * dp_1.x_5_3 - PCx * dp_2.x_5_3; 
  x_7_5 = PBx * dp_1.x_5_1 - PCx * dp_2.x_5_1; 
  x_7_5 += 0.5/Zeta * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_8_5 = PBy * dp_1.x_5_2 - PCy * dp_2.x_5_2; 
  x_8_5 += 0.5/Zeta * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_8_5 += 0.5/Zeta * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
  x_9_5 = PBz * dp_1.x_5_3 - PCz * dp_2.x_5_3; 
  x_9_5 += 0.5/Zeta * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_9_5 += 0.5/Zeta * 1.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  x_4_6 = PBx * dp_1.x_6_2 - PCx * dp_2.x_6_2; 
  x_4_6 += 0.5/Zeta * 1.000000 * (pp_1.x_3_2 - pp_2.x_3_2); 
  x_5_6 = PBy * dp_1.x_6_3 - PCy * dp_2.x_6_3; 
  x_6_6 = PBx * dp_1.x_6_3 - PCx * dp_2.x_6_3; 
  x_6_6 += 0.5/Zeta * 1.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 
  x_7_6 = PBx * dp_1.x_6_1 - PCx * dp_2.x_6_1; 
  x_7_6 += 0.5/Zeta * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_7_6 += 0.5/Zeta * 1.000000 * (pp_1.x_3_1 - pp_2.x_3_1); 
  x_8_6 = PBy * dp_1.x_6_2 - PCy * dp_2.x_6_2; 
  x_8_6 += 0.5/Zeta * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_9_6 = PBz * dp_1.x_6_3 - PCz * dp_2.x_6_3; 
  x_9_6 += 0.5/Zeta * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_9_6 += 0.5/Zeta * 1.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  x_4_7 = PBx * dp_1.x_7_2 - PCx * dp_2.x_7_2; 
  x_4_7 += 0.5/Zeta * 2.000000 * (pp_1.x_1_2 - pp_2.x_1_2); 
  x_5_7 = PBy * dp_1.x_7_3 - PCy * dp_2.x_7_3; 
  x_6_7 = PBx * dp_1.x_7_3 - PCx * dp_2.x_7_3; 
  x_6_7 += 0.5/Zeta * 2.000000 * (pp_1.x_1_3 - pp_2.x_1_3); 
  x_7_7 = PBx * dp_1.x_7_1 - PCx * dp_2.x_7_1; 
  x_7_7 += 0.5/Zeta * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_7_7 += 0.5/Zeta * 2.000000 * (pp_1.x_1_1 - pp_2.x_1_1); 
  x_8_7 = PBy * dp_1.x_7_2 - PCy * dp_2.x_7_2; 
  x_8_7 += 0.5/Zeta * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_9_7 = PBz * dp_1.x_7_3 - PCz * dp_2.x_7_3; 
  x_9_7 += 0.5/Zeta * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_4_8 = PBx * dp_1.x_8_2 - PCx * dp_2.x_8_2; 
  x_5_8 = PBy * dp_1.x_8_3 - PCy * dp_2.x_8_3; 
  x_5_8 += 0.5/Zeta * 2.000000 * (pp_1.x_2_3 - pp_2.x_2_3); 
  x_6_8 = PBx * dp_1.x_8_3 - PCx * dp_2.x_8_3; 
  x_7_8 = PBx * dp_1.x_8_1 - PCx * dp_2.x_8_1; 
  x_7_8 += 0.5/Zeta * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_8_8 = PBy * dp_1.x_8_2 - PCy * dp_2.x_8_2; 
  x_8_8 += 0.5/Zeta * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_8_8 += 0.5/Zeta * 2.000000 * (pp_1.x_2_2 - pp_2.x_2_2); 
  x_9_8 = PBz * dp_1.x_8_3 - PCz * dp_2.x_8_3; 
  x_9_8 += 0.5/Zeta * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_4_9 = PBx * dp_1.x_9_2 - PCx * dp_2.x_9_2; 
  x_5_9 = PBy * dp_1.x_9_3 - PCy * dp_2.x_9_3; 
  x_6_9 = PBx * dp_1.x_9_3 - PCx * dp_2.x_9_3; 
  x_7_9 = PBx * dp_1.x_9_1 - PCx * dp_2.x_9_1; 
  x_7_9 += 0.5/Zeta * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_8_9 = PBy * dp_1.x_9_2 - PCy * dp_2.x_9_2; 
  x_8_9 += 0.5/Zeta * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_9_9 = PBz * dp_1.x_9_3 - PCz * dp_2.x_9_3; 
  x_9_9 += 0.5/Zeta * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_9_9 += 0.5/Zeta * 2.000000 * (pp_1.x_3_3 - pp_2.x_3_3); 

 } 

/* FS true integral, m=0 */ 
__device__ __inline__ FSint_0::FSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_0 ps_0(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=0 
  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 
  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 

  x_10_0 = PAx * ds_0.x_5_0 - PCx * ds_1.x_5_0; 
  x_11_0 = PAx * ds_0.x_4_0 - PCx * ds_1.x_4_0; 
  x_11_0 += 0.5/Zeta * 1.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_12_0 = PAx * ds_0.x_8_0 - PCx * ds_1.x_8_0; 
  x_13_0 = PAx * ds_0.x_6_0 - PCx * ds_1.x_6_0; 
  x_13_0 += 0.5/Zeta * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_14_0 = PAx * ds_0.x_9_0 - PCx * ds_1.x_9_0; 
  x_15_0 = PAy * ds_0.x_5_0 - PCy * ds_1.x_5_0; 
  x_15_0 += 0.5/Zeta * 1.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 
  x_16_0 = PAy * ds_0.x_9_0 - PCy * ds_1.x_9_0; 
  x_17_0 = PAx * ds_0.x_7_0 - PCx * ds_1.x_7_0; 
  x_17_0 += 0.5/Zeta * 2.000000 * (ps_0.x_1_0 - ps_1.x_1_0); 
  x_18_0 = PAy * ds_0.x_8_0 - PCy * ds_1.x_8_0; 
  x_18_0 += 0.5/Zeta * 2.000000 * (ps_0.x_2_0 - ps_1.x_2_0); 
  x_19_0 = PAz * ds_0.x_9_0 - PCz * ds_1.x_9_0; 
  x_19_0 += 0.5/Zeta * 2.000000 * (ps_0.x_3_0 - ps_1.x_3_0); 

 } 

/* FS auxilary integral, m=1 */ 
__device__ __inline__ FSint_1::FSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_1 ps_1(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=1 
  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 

  x_10_0 = PAx * ds_1.x_5_0 - PCx * ds_2.x_5_0; 
  x_11_0 = PAx * ds_1.x_4_0 - PCx * ds_2.x_4_0; 
  x_11_0 += 0.5/Zeta * 1.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_12_0 = PAx * ds_1.x_8_0 - PCx * ds_2.x_8_0; 
  x_13_0 = PAx * ds_1.x_6_0 - PCx * ds_2.x_6_0; 
  x_13_0 += 0.5/Zeta * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_14_0 = PAx * ds_1.x_9_0 - PCx * ds_2.x_9_0; 
  x_15_0 = PAy * ds_1.x_5_0 - PCy * ds_2.x_5_0; 
  x_15_0 += 0.5/Zeta * 1.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 
  x_16_0 = PAy * ds_1.x_9_0 - PCy * ds_2.x_9_0; 
  x_17_0 = PAx * ds_1.x_7_0 - PCx * ds_2.x_7_0; 
  x_17_0 += 0.5/Zeta * 2.000000 * (ps_1.x_1_0 - ps_2.x_1_0); 
  x_18_0 = PAy * ds_1.x_8_0 - PCy * ds_2.x_8_0; 
  x_18_0 += 0.5/Zeta * 2.000000 * (ps_1.x_2_0 - ps_2.x_2_0); 
  x_19_0 = PAz * ds_1.x_9_0 - PCz * ds_2.x_9_0; 
  x_19_0 += 0.5/Zeta * 2.000000 * (ps_1.x_3_0 - ps_2.x_3_0); 

 } 

/* FS auxilary integral, m=2 */ 
__device__ __inline__ FSint_2::FSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_2 ps_2(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=2 
  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=3 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=3 

  x_10_0 = PAx * ds_2.x_5_0 - PCx * ds_3.x_5_0; 
  x_11_0 = PAx * ds_2.x_4_0 - PCx * ds_3.x_4_0; 
  x_11_0 += 0.5/Zeta * 1.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_12_0 = PAx * ds_2.x_8_0 - PCx * ds_3.x_8_0; 
  x_13_0 = PAx * ds_2.x_6_0 - PCx * ds_3.x_6_0; 
  x_13_0 += 0.5/Zeta * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_14_0 = PAx * ds_2.x_9_0 - PCx * ds_3.x_9_0; 
  x_15_0 = PAy * ds_2.x_5_0 - PCy * ds_3.x_5_0; 
  x_15_0 += 0.5/Zeta * 1.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 
  x_16_0 = PAy * ds_2.x_9_0 - PCy * ds_3.x_9_0; 
  x_17_0 = PAx * ds_2.x_7_0 - PCx * ds_3.x_7_0; 
  x_17_0 += 0.5/Zeta * 2.000000 * (ps_2.x_1_0 - ps_3.x_1_0); 
  x_18_0 = PAy * ds_2.x_8_0 - PCy * ds_3.x_8_0; 
  x_18_0 += 0.5/Zeta * 2.000000 * (ps_2.x_2_0 - ps_3.x_2_0); 
  x_19_0 = PAz * ds_2.x_9_0 - PCz * ds_3.x_9_0; 
  x_19_0 += 0.5/Zeta * 2.000000 * (ps_2.x_3_0 - ps_3.x_3_0); 

 } 

/* FS auxilary integral, m=3 */ 
__device__ __inline__ FSint_3::FSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PSint_3 ps_3(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=3 
  PSint_4 ps_4(PAx, PAy, PAz, PCx, PCy, PCz, YVerticalTemp); // construct [p|s] for m=4 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=3 
  DSint_4 ds_4(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=4 

  x_10_0 = PAx * ds_3.x_5_0 - PCx * ds_4.x_5_0; 
  x_11_0 = PAx * ds_3.x_4_0 - PCx * ds_4.x_4_0; 
  x_11_0 += 0.5/Zeta * 1.000000 * (ps_3.x_2_0 - ps_4.x_2_0); 
  x_12_0 = PAx * ds_3.x_8_0 - PCx * ds_4.x_8_0; 
  x_13_0 = PAx * ds_3.x_6_0 - PCx * ds_4.x_6_0; 
  x_13_0 += 0.5/Zeta * 1.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  x_14_0 = PAx * ds_3.x_9_0 - PCx * ds_4.x_9_0; 
  x_15_0 = PAy * ds_3.x_5_0 - PCy * ds_4.x_5_0; 
  x_15_0 += 0.5/Zeta * 1.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 
  x_16_0 = PAy * ds_3.x_9_0 - PCy * ds_4.x_9_0; 
  x_17_0 = PAx * ds_3.x_7_0 - PCx * ds_4.x_7_0; 
  x_17_0 += 0.5/Zeta * 2.000000 * (ps_3.x_1_0 - ps_4.x_1_0); 
  x_18_0 = PAy * ds_3.x_8_0 - PCy * ds_4.x_8_0; 
  x_18_0 += 0.5/Zeta * 2.000000 * (ps_3.x_2_0 - ps_4.x_2_0); 
  x_19_0 = PAz * ds_3.x_9_0 - PCz * ds_4.x_9_0; 
  x_19_0 += 0.5/Zeta * 2.000000 * (ps_3.x_3_0 - ps_4.x_3_0); 

 } 

/* SF true integral, m=0 */ 
__device__ __inline__ SFint_0::SFint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 
  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 

  x_0_10 = PBx * sd_0.x_0_5 - PCx * sd_1.x_0_5; 
  x_0_11 = PBx * sd_0.x_0_4 - PCx * sd_1.x_0_4; 
  x_0_11 += 0.5/Zeta * 1.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_0_12 = PBx * sd_0.x_0_8 - PCx * sd_1.x_0_8; 
  x_0_13 = PBx * sd_0.x_0_6 - PCx * sd_1.x_0_6; 
  x_0_13 += 0.5/Zeta * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_0_14 = PBx * sd_0.x_0_9 - PCx * sd_1.x_0_9; 
  x_0_15 = PBy * sd_0.x_0_5 - PCy * sd_1.x_0_5; 
  x_0_15 += 0.5/Zeta * 1.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 
  x_0_16 = PBy * sd_0.x_0_9 - PCy * sd_1.x_0_9; 
  x_0_17 = PBx * sd_0.x_0_7 - PCx * sd_1.x_0_7; 
  x_0_17 += 0.5/Zeta * 2.000000 * (sp_0.x_0_1 - sp_1.x_0_1); 
  x_0_18 = PBy * sd_0.x_0_8 - PCy * sd_1.x_0_8; 
  x_0_18 += 0.5/Zeta * 2.000000 * (sp_0.x_0_2 - sp_1.x_0_2); 
  x_0_19 = PBz * sd_0.x_0_9 - PCz * sd_1.x_0_9; 
  x_0_19 += 0.5/Zeta * 2.000000 * (sp_0.x_0_3 - sp_1.x_0_3); 

 } 

/* SF auxilary integral, m=1 */ 
__device__ __inline__ SFint_1::SFint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 
  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=2 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=2 

  x_0_10 = PBx * sd_1.x_0_5 - PCx * sd_2.x_0_5; 
  x_0_11 = PBx * sd_1.x_0_4 - PCx * sd_2.x_0_4; 
  x_0_11 += 0.5/Zeta * 1.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_0_12 = PBx * sd_1.x_0_8 - PCx * sd_2.x_0_8; 
  x_0_13 = PBx * sd_1.x_0_6 - PCx * sd_2.x_0_6; 
  x_0_13 += 0.5/Zeta * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_0_14 = PBx * sd_1.x_0_9 - PCx * sd_2.x_0_9; 
  x_0_15 = PBy * sd_1.x_0_5 - PCy * sd_2.x_0_5; 
  x_0_15 += 0.5/Zeta * 1.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 
  x_0_16 = PBy * sd_1.x_0_9 - PCy * sd_2.x_0_9; 
  x_0_17 = PBx * sd_1.x_0_7 - PCx * sd_2.x_0_7; 
  x_0_17 += 0.5/Zeta * 2.000000 * (sp_1.x_0_1 - sp_2.x_0_1); 
  x_0_18 = PBy * sd_1.x_0_8 - PCy * sd_2.x_0_8; 
  x_0_18 += 0.5/Zeta * 2.000000 * (sp_1.x_0_2 - sp_2.x_0_2); 
  x_0_19 = PBz * sd_1.x_0_9 - PCz * sd_2.x_0_9; 
  x_0_19 += 0.5/Zeta * 2.000000 * (sp_1.x_0_3 - sp_2.x_0_3); 

 } 

/* SF auxilary integral, m=2 */ 
__device__ __inline__ SFint_2::SFint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_2 sp_2(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=2 
  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=3 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=3 

  x_0_10 = PBx * sd_2.x_0_5 - PCx * sd_3.x_0_5; 
  x_0_11 = PBx * sd_2.x_0_4 - PCx * sd_3.x_0_4; 
  x_0_11 += 0.5/Zeta * 1.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_0_12 = PBx * sd_2.x_0_8 - PCx * sd_3.x_0_8; 
  x_0_13 = PBx * sd_2.x_0_6 - PCx * sd_3.x_0_6; 
  x_0_13 += 0.5/Zeta * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_0_14 = PBx * sd_2.x_0_9 - PCx * sd_3.x_0_9; 
  x_0_15 = PBy * sd_2.x_0_5 - PCy * sd_3.x_0_5; 
  x_0_15 += 0.5/Zeta * 1.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 
  x_0_16 = PBy * sd_2.x_0_9 - PCy * sd_3.x_0_9; 
  x_0_17 = PBx * sd_2.x_0_7 - PCx * sd_3.x_0_7; 
  x_0_17 += 0.5/Zeta * 2.000000 * (sp_2.x_0_1 - sp_3.x_0_1); 
  x_0_18 = PBy * sd_2.x_0_8 - PCy * sd_3.x_0_8; 
  x_0_18 += 0.5/Zeta * 2.000000 * (sp_2.x_0_2 - sp_3.x_0_2); 
  x_0_19 = PBz * sd_2.x_0_9 - PCz * sd_3.x_0_9; 
  x_0_19 += 0.5/Zeta * 2.000000 * (sp_2.x_0_3 - sp_3.x_0_3); 

 } 

/* SF auxilary integral, m=3 */ 
__device__ __inline__ SFint_3::SFint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_3 sp_3(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=3 
  SPint_4 sp_4(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=4 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=3 
  SDint_4 sd_4(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=4 

  x_0_10 = PBx * sd_3.x_0_5 - PCx * sd_4.x_0_5; 
  x_0_11 = PBx * sd_3.x_0_4 - PCx * sd_4.x_0_4; 
  x_0_11 += 0.5/Zeta * 1.000000 * (sp_3.x_0_2 - sp_4.x_0_2); 
  x_0_12 = PBx * sd_3.x_0_8 - PCx * sd_4.x_0_8; 
  x_0_13 = PBx * sd_3.x_0_6 - PCx * sd_4.x_0_6; 
  x_0_13 += 0.5/Zeta * 1.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  x_0_14 = PBx * sd_3.x_0_9 - PCx * sd_4.x_0_9; 
  x_0_15 = PBy * sd_3.x_0_5 - PCy * sd_4.x_0_5; 
  x_0_15 += 0.5/Zeta * 1.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 
  x_0_16 = PBy * sd_3.x_0_9 - PCy * sd_4.x_0_9; 
  x_0_17 = PBx * sd_3.x_0_7 - PCx * sd_4.x_0_7; 
  x_0_17 += 0.5/Zeta * 2.000000 * (sp_3.x_0_1 - sp_4.x_0_1); 
  x_0_18 = PBy * sd_3.x_0_8 - PCy * sd_4.x_0_8; 
  x_0_18 += 0.5/Zeta * 2.000000 * (sp_3.x_0_2 - sp_4.x_0_2); 
  x_0_19 = PBz * sd_3.x_0_9 - PCz * sd_4.x_0_9; 
  x_0_19 += 0.5/Zeta * 2.000000 * (sp_3.x_0_3 - sp_4.x_0_3); 

 } 

/* FP true integral, m=0 */ 
__device__ __inline__ FPint_0::FPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  DSint_0 ds_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=0 
  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=0 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=1 

  x_10_1 = PBx * fs_0.x_10_0 - PCx * fs_1.x_10_0; 
  x_10_1 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_10_2 = PBy * fs_0.x_10_0 - PCy * fs_1.x_10_0; 
  x_10_2 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_10_3 = PBz * fs_0.x_10_0 - PCz * fs_1.x_10_0; 
  x_10_3 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_11_1 = PBx * fs_0.x_11_0 - PCx * fs_1.x_11_0; 
  x_11_1 += 0.5/Zeta * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_11_2 = PBy * fs_0.x_11_0 - PCy * fs_1.x_11_0; 
  x_11_2 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_11_3 = PBz * fs_0.x_11_0 - PCz * fs_1.x_11_0; 
  x_12_1 = PBx * fs_0.x_12_0 - PCx * fs_1.x_12_0; 
  x_12_1 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_12_2 = PBy * fs_0.x_12_0 - PCy * fs_1.x_12_0; 
  x_12_2 += 0.5/Zeta * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_12_3 = PBz * fs_0.x_12_0 - PCz * fs_1.x_12_0; 
  x_13_1 = PBx * fs_0.x_13_0 - PCx * fs_1.x_13_0; 
  x_13_1 += 0.5/Zeta * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_13_2 = PBy * fs_0.x_13_0 - PCy * fs_1.x_13_0; 
  x_13_3 = PBz * fs_0.x_13_0 - PCz * fs_1.x_13_0; 
  x_13_3 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_14_1 = PBx * fs_0.x_14_0 - PCx * fs_1.x_14_0; 
  x_14_1 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_14_2 = PBy * fs_0.x_14_0 - PCy * fs_1.x_14_0; 
  x_14_3 = PBz * fs_0.x_14_0 - PCz * fs_1.x_14_0; 
  x_14_3 += 0.5/Zeta * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_15_1 = PBx * fs_0.x_15_0 - PCx * fs_1.x_15_0; 
  x_15_2 = PBy * fs_0.x_15_0 - PCy * fs_1.x_15_0; 
  x_15_2 += 0.5/Zeta * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_15_3 = PBz * fs_0.x_15_0 - PCz * fs_1.x_15_0; 
  x_15_3 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_16_1 = PBx * fs_0.x_16_0 - PCx * fs_1.x_16_0; 
  x_16_2 = PBy * fs_0.x_16_0 - PCy * fs_1.x_16_0; 
  x_16_2 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_16_3 = PBz * fs_0.x_16_0 - PCz * fs_1.x_16_0; 
  x_16_3 += 0.5/Zeta * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_17_1 = PBx * fs_0.x_17_0 - PCx * fs_1.x_17_0; 
  x_17_1 += 0.5/Zeta * 3.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_17_2 = PBy * fs_0.x_17_0 - PCy * fs_1.x_17_0; 
  x_17_3 = PBz * fs_0.x_17_0 - PCz * fs_1.x_17_0; 
  x_18_1 = PBx * fs_0.x_18_0 - PCx * fs_1.x_18_0; 
  x_18_2 = PBy * fs_0.x_18_0 - PCy * fs_1.x_18_0; 
  x_18_2 += 0.5/Zeta * 3.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_18_3 = PBz * fs_0.x_18_0 - PCz * fs_1.x_18_0; 
  x_19_1 = PBx * fs_0.x_19_0 - PCx * fs_1.x_19_0; 
  x_19_2 = PBy * fs_0.x_19_0 - PCy * fs_1.x_19_0; 
  x_19_3 = PBz * fs_0.x_19_0 - PCz * fs_1.x_19_0; 
  x_19_3 += 0.5/Zeta * 3.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 

 } 

/* FP auxilary integral, m=1 */ 
__device__ __inline__ FPint_1::FPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  DSint_1 ds_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=1 
  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=1 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=2 

  x_10_1 = PBx * fs_1.x_10_0 - PCx * fs_2.x_10_0; 
  x_10_1 += 0.5/Zeta * 1.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_10_2 = PBy * fs_1.x_10_0 - PCy * fs_2.x_10_0; 
  x_10_2 += 0.5/Zeta * 1.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_10_3 = PBz * fs_1.x_10_0 - PCz * fs_2.x_10_0; 
  x_10_3 += 0.5/Zeta * 1.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_11_1 = PBx * fs_1.x_11_0 - PCx * fs_2.x_11_0; 
  x_11_1 += 0.5/Zeta * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_11_2 = PBy * fs_1.x_11_0 - PCy * fs_2.x_11_0; 
  x_11_2 += 0.5/Zeta * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_11_3 = PBz * fs_1.x_11_0 - PCz * fs_2.x_11_0; 
  x_12_1 = PBx * fs_1.x_12_0 - PCx * fs_2.x_12_0; 
  x_12_1 += 0.5/Zeta * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_12_2 = PBy * fs_1.x_12_0 - PCy * fs_2.x_12_0; 
  x_12_2 += 0.5/Zeta * 2.000000 * (ds_1.x_4_0 - ds_2.x_4_0); 
  x_12_3 = PBz * fs_1.x_12_0 - PCz * fs_2.x_12_0; 
  x_13_1 = PBx * fs_1.x_13_0 - PCx * fs_2.x_13_0; 
  x_13_1 += 0.5/Zeta * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_13_2 = PBy * fs_1.x_13_0 - PCy * fs_2.x_13_0; 
  x_13_3 = PBz * fs_1.x_13_0 - PCz * fs_2.x_13_0; 
  x_13_3 += 0.5/Zeta * 1.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_14_1 = PBx * fs_1.x_14_0 - PCx * fs_2.x_14_0; 
  x_14_1 += 0.5/Zeta * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_14_2 = PBy * fs_1.x_14_0 - PCy * fs_2.x_14_0; 
  x_14_3 = PBz * fs_1.x_14_0 - PCz * fs_2.x_14_0; 
  x_14_3 += 0.5/Zeta * 2.000000 * (ds_1.x_6_0 - ds_2.x_6_0); 
  x_15_1 = PBx * fs_1.x_15_0 - PCx * fs_2.x_15_0; 
  x_15_2 = PBy * fs_1.x_15_0 - PCy * fs_2.x_15_0; 
  x_15_2 += 0.5/Zeta * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_15_3 = PBz * fs_1.x_15_0 - PCz * fs_2.x_15_0; 
  x_15_3 += 0.5/Zeta * 1.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_16_1 = PBx * fs_1.x_16_0 - PCx * fs_2.x_16_0; 
  x_16_2 = PBy * fs_1.x_16_0 - PCy * fs_2.x_16_0; 
  x_16_2 += 0.5/Zeta * 1.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 
  x_16_3 = PBz * fs_1.x_16_0 - PCz * fs_2.x_16_0; 
  x_16_3 += 0.5/Zeta * 2.000000 * (ds_1.x_5_0 - ds_2.x_5_0); 
  x_17_1 = PBx * fs_1.x_17_0 - PCx * fs_2.x_17_0; 
  x_17_1 += 0.5/Zeta * 3.000000 * (ds_1.x_7_0 - ds_2.x_7_0); 
  x_17_2 = PBy * fs_1.x_17_0 - PCy * fs_2.x_17_0; 
  x_17_3 = PBz * fs_1.x_17_0 - PCz * fs_2.x_17_0; 
  x_18_1 = PBx * fs_1.x_18_0 - PCx * fs_2.x_18_0; 
  x_18_2 = PBy * fs_1.x_18_0 - PCy * fs_2.x_18_0; 
  x_18_2 += 0.5/Zeta * 3.000000 * (ds_1.x_8_0 - ds_2.x_8_0); 
  x_18_3 = PBz * fs_1.x_18_0 - PCz * fs_2.x_18_0; 
  x_19_1 = PBx * fs_1.x_19_0 - PCx * fs_2.x_19_0; 
  x_19_2 = PBy * fs_1.x_19_0 - PCy * fs_2.x_19_0; 
  x_19_3 = PBz * fs_1.x_19_0 - PCz * fs_2.x_19_0; 
  x_19_3 += 0.5/Zeta * 3.000000 * (ds_1.x_9_0 - ds_2.x_9_0); 

 } 

/* FP auxilary integral, m=2 */ 
__device__ __inline__ FPint_2::FPint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  DSint_2 ds_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=2 
  DSint_3 ds_3(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|s] for m=3 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=2 
  FSint_3 fs_3(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=3 

  x_10_1 = PBx * fs_2.x_10_0 - PCx * fs_3.x_10_0; 
  x_10_1 += 0.5/Zeta * 1.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
  x_10_2 = PBy * fs_2.x_10_0 - PCy * fs_3.x_10_0; 
  x_10_2 += 0.5/Zeta * 1.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
  x_10_3 = PBz * fs_2.x_10_0 - PCz * fs_3.x_10_0; 
  x_10_3 += 0.5/Zeta * 1.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
  x_11_1 = PBx * fs_2.x_11_0 - PCx * fs_3.x_11_0; 
  x_11_1 += 0.5/Zeta * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
  x_11_2 = PBy * fs_2.x_11_0 - PCy * fs_3.x_11_0; 
  x_11_2 += 0.5/Zeta * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
  x_11_3 = PBz * fs_2.x_11_0 - PCz * fs_3.x_11_0; 
  x_12_1 = PBx * fs_2.x_12_0 - PCx * fs_3.x_12_0; 
  x_12_1 += 0.5/Zeta * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
  x_12_2 = PBy * fs_2.x_12_0 - PCy * fs_3.x_12_0; 
  x_12_2 += 0.5/Zeta * 2.000000 * (ds_2.x_4_0 - ds_3.x_4_0); 
  x_12_3 = PBz * fs_2.x_12_0 - PCz * fs_3.x_12_0; 
  x_13_1 = PBx * fs_2.x_13_0 - PCx * fs_3.x_13_0; 
  x_13_1 += 0.5/Zeta * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
  x_13_2 = PBy * fs_2.x_13_0 - PCy * fs_3.x_13_0; 
  x_13_3 = PBz * fs_2.x_13_0 - PCz * fs_3.x_13_0; 
  x_13_3 += 0.5/Zeta * 1.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
  x_14_1 = PBx * fs_2.x_14_0 - PCx * fs_3.x_14_0; 
  x_14_1 += 0.5/Zeta * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
  x_14_2 = PBy * fs_2.x_14_0 - PCy * fs_3.x_14_0; 
  x_14_3 = PBz * fs_2.x_14_0 - PCz * fs_3.x_14_0; 
  x_14_3 += 0.5/Zeta * 2.000000 * (ds_2.x_6_0 - ds_3.x_6_0); 
  x_15_1 = PBx * fs_2.x_15_0 - PCx * fs_3.x_15_0; 
  x_15_2 = PBy * fs_2.x_15_0 - PCy * fs_3.x_15_0; 
  x_15_2 += 0.5/Zeta * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
  x_15_3 = PBz * fs_2.x_15_0 - PCz * fs_3.x_15_0; 
  x_15_3 += 0.5/Zeta * 1.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
  x_16_1 = PBx * fs_2.x_16_0 - PCx * fs_3.x_16_0; 
  x_16_2 = PBy * fs_2.x_16_0 - PCy * fs_3.x_16_0; 
  x_16_2 += 0.5/Zeta * 1.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 
  x_16_3 = PBz * fs_2.x_16_0 - PCz * fs_3.x_16_0; 
  x_16_3 += 0.5/Zeta * 2.000000 * (ds_2.x_5_0 - ds_3.x_5_0); 
  x_17_1 = PBx * fs_2.x_17_0 - PCx * fs_3.x_17_0; 
  x_17_1 += 0.5/Zeta * 3.000000 * (ds_2.x_7_0 - ds_3.x_7_0); 
  x_17_2 = PBy * fs_2.x_17_0 - PCy * fs_3.x_17_0; 
  x_17_3 = PBz * fs_2.x_17_0 - PCz * fs_3.x_17_0; 
  x_18_1 = PBx * fs_2.x_18_0 - PCx * fs_3.x_18_0; 
  x_18_2 = PBy * fs_2.x_18_0 - PCy * fs_3.x_18_0; 
  x_18_2 += 0.5/Zeta * 3.000000 * (ds_2.x_8_0 - ds_3.x_8_0); 
  x_18_3 = PBz * fs_2.x_18_0 - PCz * fs_3.x_18_0; 
  x_19_1 = PBx * fs_2.x_19_0 - PCx * fs_3.x_19_0; 
  x_19_2 = PBy * fs_2.x_19_0 - PCy * fs_3.x_19_0; 
  x_19_3 = PBz * fs_2.x_19_0 - PCz * fs_3.x_19_0; 
  x_19_3 += 0.5/Zeta * 3.000000 * (ds_2.x_9_0 - ds_3.x_9_0); 

 } 

/* PF true integral, m=0 */ 
__device__ __inline__ PFint_0::PFint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SDint_0 sd_0(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=0 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=1 

  x_1_10 = PAx * sf_0.x_0_10 - PCx * sf_1.x_0_10; 
  x_1_10 += 0.5/Zeta * 1.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
  x_2_10 = PAy * sf_0.x_0_10 - PCy * sf_1.x_0_10; 
  x_2_10 += 0.5/Zeta * 1.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
  x_3_10 = PAz * sf_0.x_0_10 - PCz * sf_1.x_0_10; 
  x_3_10 += 0.5/Zeta * 1.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
  x_1_11 = PAx * sf_0.x_0_11 - PCx * sf_1.x_0_11; 
  x_1_11 += 0.5/Zeta * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
  x_2_11 = PAy * sf_0.x_0_11 - PCy * sf_1.x_0_11; 
  x_2_11 += 0.5/Zeta * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
  x_3_11 = PAz * sf_0.x_0_11 - PCz * sf_1.x_0_11; 
  x_1_12 = PAx * sf_0.x_0_12 - PCx * sf_1.x_0_12; 
  x_1_12 += 0.5/Zeta * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
  x_2_12 = PAy * sf_0.x_0_12 - PCy * sf_1.x_0_12; 
  x_2_12 += 0.5/Zeta * 2.000000 * (sd_0.x_0_4 - sd_1.x_0_4); 
  x_3_12 = PAz * sf_0.x_0_12 - PCz * sf_1.x_0_12; 
  x_1_13 = PAx * sf_0.x_0_13 - PCx * sf_1.x_0_13; 
  x_1_13 += 0.5/Zeta * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
  x_2_13 = PAy * sf_0.x_0_13 - PCy * sf_1.x_0_13; 
  x_3_13 = PAz * sf_0.x_0_13 - PCz * sf_1.x_0_13; 
  x_3_13 += 0.5/Zeta * 1.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
  x_1_14 = PAx * sf_0.x_0_14 - PCx * sf_1.x_0_14; 
  x_1_14 += 0.5/Zeta * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
  x_2_14 = PAy * sf_0.x_0_14 - PCy * sf_1.x_0_14; 
  x_3_14 = PAz * sf_0.x_0_14 - PCz * sf_1.x_0_14; 
  x_3_14 += 0.5/Zeta * 2.000000 * (sd_0.x_0_6 - sd_1.x_0_6); 
  x_1_15 = PAx * sf_0.x_0_15 - PCx * sf_1.x_0_15; 
  x_2_15 = PAy * sf_0.x_0_15 - PCy * sf_1.x_0_15; 
  x_2_15 += 0.5/Zeta * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
  x_3_15 = PAz * sf_0.x_0_15 - PCz * sf_1.x_0_15; 
  x_3_15 += 0.5/Zeta * 1.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
  x_1_16 = PAx * sf_0.x_0_16 - PCx * sf_1.x_0_16; 
  x_2_16 = PAy * sf_0.x_0_16 - PCy * sf_1.x_0_16; 
  x_2_16 += 0.5/Zeta * 1.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 
  x_3_16 = PAz * sf_0.x_0_16 - PCz * sf_1.x_0_16; 
  x_3_16 += 0.5/Zeta * 2.000000 * (sd_0.x_0_5 - sd_1.x_0_5); 
  x_1_17 = PAx * sf_0.x_0_17 - PCx * sf_1.x_0_17; 
  x_1_17 += 0.5/Zeta * 3.000000 * (sd_0.x_0_7 - sd_1.x_0_7); 
  x_2_17 = PAy * sf_0.x_0_17 - PCy * sf_1.x_0_17; 
  x_3_17 = PAz * sf_0.x_0_17 - PCz * sf_1.x_0_17; 
  x_1_18 = PAx * sf_0.x_0_18 - PCx * sf_1.x_0_18; 
  x_2_18 = PAy * sf_0.x_0_18 - PCy * sf_1.x_0_18; 
  x_2_18 += 0.5/Zeta * 3.000000 * (sd_0.x_0_8 - sd_1.x_0_8); 
  x_3_18 = PAz * sf_0.x_0_18 - PCz * sf_1.x_0_18; 
  x_1_19 = PAx * sf_0.x_0_19 - PCx * sf_1.x_0_19; 
  x_2_19 = PAy * sf_0.x_0_19 - PCy * sf_1.x_0_19; 
  x_3_19 = PAz * sf_0.x_0_19 - PCz * sf_1.x_0_19; 
  x_3_19 += 0.5/Zeta * 3.000000 * (sd_0.x_0_9 - sd_1.x_0_9); 

 } 

/* PF auxilary integral, m=1 */ 
__device__ __inline__ PFint_1::PFint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SDint_1 sd_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 
  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=2 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=1 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=2 

  x_1_10 = PAx * sf_1.x_0_10 - PCx * sf_2.x_0_10; 
  x_1_10 += 0.5/Zeta * 1.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
  x_2_10 = PAy * sf_1.x_0_10 - PCy * sf_2.x_0_10; 
  x_2_10 += 0.5/Zeta * 1.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
  x_3_10 = PAz * sf_1.x_0_10 - PCz * sf_2.x_0_10; 
  x_3_10 += 0.5/Zeta * 1.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
  x_1_11 = PAx * sf_1.x_0_11 - PCx * sf_2.x_0_11; 
  x_1_11 += 0.5/Zeta * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
  x_2_11 = PAy * sf_1.x_0_11 - PCy * sf_2.x_0_11; 
  x_2_11 += 0.5/Zeta * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
  x_3_11 = PAz * sf_1.x_0_11 - PCz * sf_2.x_0_11; 
  x_1_12 = PAx * sf_1.x_0_12 - PCx * sf_2.x_0_12; 
  x_1_12 += 0.5/Zeta * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
  x_2_12 = PAy * sf_1.x_0_12 - PCy * sf_2.x_0_12; 
  x_2_12 += 0.5/Zeta * 2.000000 * (sd_1.x_0_4 - sd_2.x_0_4); 
  x_3_12 = PAz * sf_1.x_0_12 - PCz * sf_2.x_0_12; 
  x_1_13 = PAx * sf_1.x_0_13 - PCx * sf_2.x_0_13; 
  x_1_13 += 0.5/Zeta * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
  x_2_13 = PAy * sf_1.x_0_13 - PCy * sf_2.x_0_13; 
  x_3_13 = PAz * sf_1.x_0_13 - PCz * sf_2.x_0_13; 
  x_3_13 += 0.5/Zeta * 1.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
  x_1_14 = PAx * sf_1.x_0_14 - PCx * sf_2.x_0_14; 
  x_1_14 += 0.5/Zeta * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
  x_2_14 = PAy * sf_1.x_0_14 - PCy * sf_2.x_0_14; 
  x_3_14 = PAz * sf_1.x_0_14 - PCz * sf_2.x_0_14; 
  x_3_14 += 0.5/Zeta * 2.000000 * (sd_1.x_0_6 - sd_2.x_0_6); 
  x_1_15 = PAx * sf_1.x_0_15 - PCx * sf_2.x_0_15; 
  x_2_15 = PAy * sf_1.x_0_15 - PCy * sf_2.x_0_15; 
  x_2_15 += 0.5/Zeta * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
  x_3_15 = PAz * sf_1.x_0_15 - PCz * sf_2.x_0_15; 
  x_3_15 += 0.5/Zeta * 1.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
  x_1_16 = PAx * sf_1.x_0_16 - PCx * sf_2.x_0_16; 
  x_2_16 = PAy * sf_1.x_0_16 - PCy * sf_2.x_0_16; 
  x_2_16 += 0.5/Zeta * 1.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 
  x_3_16 = PAz * sf_1.x_0_16 - PCz * sf_2.x_0_16; 
  x_3_16 += 0.5/Zeta * 2.000000 * (sd_1.x_0_5 - sd_2.x_0_5); 
  x_1_17 = PAx * sf_1.x_0_17 - PCx * sf_2.x_0_17; 
  x_1_17 += 0.5/Zeta * 3.000000 * (sd_1.x_0_7 - sd_2.x_0_7); 
  x_2_17 = PAy * sf_1.x_0_17 - PCy * sf_2.x_0_17; 
  x_3_17 = PAz * sf_1.x_0_17 - PCz * sf_2.x_0_17; 
  x_1_18 = PAx * sf_1.x_0_18 - PCx * sf_2.x_0_18; 
  x_2_18 = PAy * sf_1.x_0_18 - PCy * sf_2.x_0_18; 
  x_2_18 += 0.5/Zeta * 3.000000 * (sd_1.x_0_8 - sd_2.x_0_8); 
  x_3_18 = PAz * sf_1.x_0_18 - PCz * sf_2.x_0_18; 
  x_1_19 = PAx * sf_1.x_0_19 - PCx * sf_2.x_0_19; 
  x_2_19 = PAy * sf_1.x_0_19 - PCy * sf_2.x_0_19; 
  x_3_19 = PAz * sf_1.x_0_19 - PCz * sf_2.x_0_19; 
  x_3_19 += 0.5/Zeta * 3.000000 * (sd_1.x_0_9 - sd_2.x_0_9); 

 } 

/* PF auxilary integral, m=2 */ 
__device__ __inline__ PFint_2::PFint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SDint_2 sd_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=2 
  SDint_3 sd_3(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=3 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=2 
  SFint_3 sf_3(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=3 

  x_1_10 = PAx * sf_2.x_0_10 - PCx * sf_3.x_0_10; 
  x_1_10 += 0.5/Zeta * 1.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
  x_2_10 = PAy * sf_2.x_0_10 - PCy * sf_3.x_0_10; 
  x_2_10 += 0.5/Zeta * 1.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
  x_3_10 = PAz * sf_2.x_0_10 - PCz * sf_3.x_0_10; 
  x_3_10 += 0.5/Zeta * 1.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
  x_1_11 = PAx * sf_2.x_0_11 - PCx * sf_3.x_0_11; 
  x_1_11 += 0.5/Zeta * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
  x_2_11 = PAy * sf_2.x_0_11 - PCy * sf_3.x_0_11; 
  x_2_11 += 0.5/Zeta * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
  x_3_11 = PAz * sf_2.x_0_11 - PCz * sf_3.x_0_11; 
  x_1_12 = PAx * sf_2.x_0_12 - PCx * sf_3.x_0_12; 
  x_1_12 += 0.5/Zeta * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
  x_2_12 = PAy * sf_2.x_0_12 - PCy * sf_3.x_0_12; 
  x_2_12 += 0.5/Zeta * 2.000000 * (sd_2.x_0_4 - sd_3.x_0_4); 
  x_3_12 = PAz * sf_2.x_0_12 - PCz * sf_3.x_0_12; 
  x_1_13 = PAx * sf_2.x_0_13 - PCx * sf_3.x_0_13; 
  x_1_13 += 0.5/Zeta * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
  x_2_13 = PAy * sf_2.x_0_13 - PCy * sf_3.x_0_13; 
  x_3_13 = PAz * sf_2.x_0_13 - PCz * sf_3.x_0_13; 
  x_3_13 += 0.5/Zeta * 1.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
  x_1_14 = PAx * sf_2.x_0_14 - PCx * sf_3.x_0_14; 
  x_1_14 += 0.5/Zeta * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
  x_2_14 = PAy * sf_2.x_0_14 - PCy * sf_3.x_0_14; 
  x_3_14 = PAz * sf_2.x_0_14 - PCz * sf_3.x_0_14; 
  x_3_14 += 0.5/Zeta * 2.000000 * (sd_2.x_0_6 - sd_3.x_0_6); 
  x_1_15 = PAx * sf_2.x_0_15 - PCx * sf_3.x_0_15; 
  x_2_15 = PAy * sf_2.x_0_15 - PCy * sf_3.x_0_15; 
  x_2_15 += 0.5/Zeta * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
  x_3_15 = PAz * sf_2.x_0_15 - PCz * sf_3.x_0_15; 
  x_3_15 += 0.5/Zeta * 1.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
  x_1_16 = PAx * sf_2.x_0_16 - PCx * sf_3.x_0_16; 
  x_2_16 = PAy * sf_2.x_0_16 - PCy * sf_3.x_0_16; 
  x_2_16 += 0.5/Zeta * 1.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 
  x_3_16 = PAz * sf_2.x_0_16 - PCz * sf_3.x_0_16; 
  x_3_16 += 0.5/Zeta * 2.000000 * (sd_2.x_0_5 - sd_3.x_0_5); 
  x_1_17 = PAx * sf_2.x_0_17 - PCx * sf_3.x_0_17; 
  x_1_17 += 0.5/Zeta * 3.000000 * (sd_2.x_0_7 - sd_3.x_0_7); 
  x_2_17 = PAy * sf_2.x_0_17 - PCy * sf_3.x_0_17; 
  x_3_17 = PAz * sf_2.x_0_17 - PCz * sf_3.x_0_17; 
  x_1_18 = PAx * sf_2.x_0_18 - PCx * sf_3.x_0_18; 
  x_2_18 = PAy * sf_2.x_0_18 - PCy * sf_3.x_0_18; 
  x_2_18 += 0.5/Zeta * 3.000000 * (sd_2.x_0_8 - sd_3.x_0_8); 
  x_3_18 = PAz * sf_2.x_0_18 - PCz * sf_3.x_0_18; 
  x_1_19 = PAx * sf_2.x_0_19 - PCx * sf_3.x_0_19; 
  x_2_19 = PAy * sf_2.x_0_19 - PCy * sf_3.x_0_19; 
  x_3_19 = PAz * sf_2.x_0_19 - PCz * sf_3.x_0_19; 
  x_3_19 += 0.5/Zeta * 3.000000 * (sd_2.x_0_9 - sd_3.x_0_9); 

 } 

/* FD true integral, m=0 */ 
__device__ __inline__ FDint_0::FDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  DPint_0 dp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=0 
  FSint_0 fs_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=0 
  FPint_0 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|p] for m=0 
  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=1 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=1 
  FPint_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|p] for m=1 

  x_10_4 = PBx * fp_0.x_10_2 - PCx * fp_1.x_10_2; 
  x_10_4 += 0.5/Zeta * 1.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
  x_10_5 = PBy * fp_0.x_10_3 - PCy * fp_1.x_10_3; 
  x_10_5 += 0.5/Zeta * 1.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  x_10_6 = PBx * fp_0.x_10_3 - PCx * fp_1.x_10_3; 
  x_10_6 += 0.5/Zeta * 1.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  x_10_7 = PBx * fp_0.x_10_1 - PCx * fp_1.x_10_1; 
  x_10_7 += 0.5/Zeta * (fs_0.x_10_0 - fs_1.x_10_0); 
  x_10_7 += 0.5/Zeta * 1.000000 * (dp_0.x_5_1 - dp_1.x_5_1); 
  x_10_8 = PBy * fp_0.x_10_2 - PCy * fp_1.x_10_2; 
  x_10_8 += 0.5/Zeta * (fs_0.x_10_0 - fs_1.x_10_0); 
  x_10_8 += 0.5/Zeta * 1.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
  x_10_9 = PBz * fp_0.x_10_3 - PCz * fp_1.x_10_3; 
  x_10_9 += 0.5/Zeta * (fs_0.x_10_0 - fs_1.x_10_0); 
  x_10_9 += 0.5/Zeta * 1.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  x_11_4 = PBx * fp_0.x_11_2 - PCx * fp_1.x_11_2; 
  x_11_4 += 0.5/Zeta * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
  x_11_5 = PBy * fp_0.x_11_3 - PCy * fp_1.x_11_3; 
  x_11_5 += 0.5/Zeta * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  x_11_6 = PBx * fp_0.x_11_3 - PCx * fp_1.x_11_3; 
  x_11_6 += 0.5/Zeta * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  x_11_7 = PBx * fp_0.x_11_1 - PCx * fp_1.x_11_1; 
  x_11_7 += 0.5/Zeta * (fs_0.x_11_0 - fs_1.x_11_0); 
  x_11_7 += 0.5/Zeta * 2.000000 * (dp_0.x_4_1 - dp_1.x_4_1); 
  x_11_8 = PBy * fp_0.x_11_2 - PCy * fp_1.x_11_2; 
  x_11_8 += 0.5/Zeta * (fs_0.x_11_0 - fs_1.x_11_0); 
  x_11_8 += 0.5/Zeta * 1.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
  x_11_9 = PBz * fp_0.x_11_3 - PCz * fp_1.x_11_3; 
  x_11_9 += 0.5/Zeta * (fs_0.x_11_0 - fs_1.x_11_0); 
  x_12_4 = PBx * fp_0.x_12_2 - PCx * fp_1.x_12_2; 
  x_12_4 += 0.5/Zeta * 1.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
  x_12_5 = PBy * fp_0.x_12_3 - PCy * fp_1.x_12_3; 
  x_12_5 += 0.5/Zeta * 2.000000 * (dp_0.x_4_3 - dp_1.x_4_3); 
  x_12_6 = PBx * fp_0.x_12_3 - PCx * fp_1.x_12_3; 
  x_12_6 += 0.5/Zeta * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  x_12_7 = PBx * fp_0.x_12_1 - PCx * fp_1.x_12_1; 
  x_12_7 += 0.5/Zeta * (fs_0.x_12_0 - fs_1.x_12_0); 
  x_12_7 += 0.5/Zeta * 1.000000 * (dp_0.x_8_1 - dp_1.x_8_1); 
  x_12_8 = PBy * fp_0.x_12_2 - PCy * fp_1.x_12_2; 
  x_12_8 += 0.5/Zeta * (fs_0.x_12_0 - fs_1.x_12_0); 
  x_12_8 += 0.5/Zeta * 2.000000 * (dp_0.x_4_2 - dp_1.x_4_2); 
  x_12_9 = PBz * fp_0.x_12_3 - PCz * fp_1.x_12_3; 
  x_12_9 += 0.5/Zeta * (fs_0.x_12_0 - fs_1.x_12_0); 
  x_13_4 = PBx * fp_0.x_13_2 - PCx * fp_1.x_13_2; 
  x_13_4 += 0.5/Zeta * 2.000000 * (dp_0.x_6_2 - dp_1.x_6_2); 
  x_13_5 = PBy * fp_0.x_13_3 - PCy * fp_1.x_13_3; 
  x_13_6 = PBx * fp_0.x_13_3 - PCx * fp_1.x_13_3; 
  x_13_6 += 0.5/Zeta * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  x_13_7 = PBx * fp_0.x_13_1 - PCx * fp_1.x_13_1; 
  x_13_7 += 0.5/Zeta * (fs_0.x_13_0 - fs_1.x_13_0); 
  x_13_7 += 0.5/Zeta * 2.000000 * (dp_0.x_6_1 - dp_1.x_6_1); 
  x_13_8 = PBy * fp_0.x_13_2 - PCy * fp_1.x_13_2; 
  x_13_8 += 0.5/Zeta * (fs_0.x_13_0 - fs_1.x_13_0); 
  x_13_9 = PBz * fp_0.x_13_3 - PCz * fp_1.x_13_3; 
  x_13_9 += 0.5/Zeta * (fs_0.x_13_0 - fs_1.x_13_0); 
  x_13_9 += 0.5/Zeta * 1.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  x_14_4 = PBx * fp_0.x_14_2 - PCx * fp_1.x_14_2; 
  x_14_4 += 0.5/Zeta * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
  x_14_5 = PBy * fp_0.x_14_3 - PCy * fp_1.x_14_3; 
  x_14_6 = PBx * fp_0.x_14_3 - PCx * fp_1.x_14_3; 
  x_14_6 += 0.5/Zeta * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  x_14_7 = PBx * fp_0.x_14_1 - PCx * fp_1.x_14_1; 
  x_14_7 += 0.5/Zeta * (fs_0.x_14_0 - fs_1.x_14_0); 
  x_14_7 += 0.5/Zeta * 1.000000 * (dp_0.x_9_1 - dp_1.x_9_1); 
  x_14_8 = PBy * fp_0.x_14_2 - PCy * fp_1.x_14_2; 
  x_14_8 += 0.5/Zeta * (fs_0.x_14_0 - fs_1.x_14_0); 
  x_14_9 = PBz * fp_0.x_14_3 - PCz * fp_1.x_14_3; 
  x_14_9 += 0.5/Zeta * (fs_0.x_14_0 - fs_1.x_14_0); 
  x_14_9 += 0.5/Zeta * 2.000000 * (dp_0.x_6_3 - dp_1.x_6_3); 
  x_15_4 = PBx * fp_0.x_15_2 - PCx * fp_1.x_15_2; 
  x_15_5 = PBy * fp_0.x_15_3 - PCy * fp_1.x_15_3; 
  x_15_5 += 0.5/Zeta * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  x_15_6 = PBx * fp_0.x_15_3 - PCx * fp_1.x_15_3; 
  x_15_7 = PBx * fp_0.x_15_1 - PCx * fp_1.x_15_1; 
  x_15_7 += 0.5/Zeta * (fs_0.x_15_0 - fs_1.x_15_0); 
  x_15_8 = PBy * fp_0.x_15_2 - PCy * fp_1.x_15_2; 
  x_15_8 += 0.5/Zeta * (fs_0.x_15_0 - fs_1.x_15_0); 
  x_15_8 += 0.5/Zeta * 2.000000 * (dp_0.x_5_2 - dp_1.x_5_2); 
  x_15_9 = PBz * fp_0.x_15_3 - PCz * fp_1.x_15_3; 
  x_15_9 += 0.5/Zeta * (fs_0.x_15_0 - fs_1.x_15_0); 
  x_15_9 += 0.5/Zeta * 1.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  x_16_4 = PBx * fp_0.x_16_2 - PCx * fp_1.x_16_2; 
  x_16_5 = PBy * fp_0.x_16_3 - PCy * fp_1.x_16_3; 
  x_16_5 += 0.5/Zeta * 1.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 
  x_16_6 = PBx * fp_0.x_16_3 - PCx * fp_1.x_16_3; 
  x_16_7 = PBx * fp_0.x_16_1 - PCx * fp_1.x_16_1; 
  x_16_7 += 0.5/Zeta * (fs_0.x_16_0 - fs_1.x_16_0); 
  x_16_8 = PBy * fp_0.x_16_2 - PCy * fp_1.x_16_2; 
  x_16_8 += 0.5/Zeta * (fs_0.x_16_0 - fs_1.x_16_0); 
  x_16_8 += 0.5/Zeta * 1.000000 * (dp_0.x_9_2 - dp_1.x_9_2); 
  x_16_9 = PBz * fp_0.x_16_3 - PCz * fp_1.x_16_3; 
  x_16_9 += 0.5/Zeta * (fs_0.x_16_0 - fs_1.x_16_0); 
  x_16_9 += 0.5/Zeta * 2.000000 * (dp_0.x_5_3 - dp_1.x_5_3); 
  x_17_4 = PBx * fp_0.x_17_2 - PCx * fp_1.x_17_2; 
  x_17_4 += 0.5/Zeta * 3.000000 * (dp_0.x_7_2 - dp_1.x_7_2); 
  x_17_5 = PBy * fp_0.x_17_3 - PCy * fp_1.x_17_3; 
  x_17_6 = PBx * fp_0.x_17_3 - PCx * fp_1.x_17_3; 
  x_17_6 += 0.5/Zeta * 3.000000 * (dp_0.x_7_3 - dp_1.x_7_3); 
  x_17_7 = PBx * fp_0.x_17_1 - PCx * fp_1.x_17_1; 
  x_17_7 += 0.5/Zeta * (fs_0.x_17_0 - fs_1.x_17_0); 
  x_17_7 += 0.5/Zeta * 3.000000 * (dp_0.x_7_1 - dp_1.x_7_1); 
  x_17_8 = PBy * fp_0.x_17_2 - PCy * fp_1.x_17_2; 
  x_17_8 += 0.5/Zeta * (fs_0.x_17_0 - fs_1.x_17_0); 
  x_17_9 = PBz * fp_0.x_17_3 - PCz * fp_1.x_17_3; 
  x_17_9 += 0.5/Zeta * (fs_0.x_17_0 - fs_1.x_17_0); 
  x_18_4 = PBx * fp_0.x_18_2 - PCx * fp_1.x_18_2; 
  x_18_5 = PBy * fp_0.x_18_3 - PCy * fp_1.x_18_3; 
  x_18_5 += 0.5/Zeta * 3.000000 * (dp_0.x_8_3 - dp_1.x_8_3); 
  x_18_6 = PBx * fp_0.x_18_3 - PCx * fp_1.x_18_3; 
  x_18_7 = PBx * fp_0.x_18_1 - PCx * fp_1.x_18_1; 
  x_18_7 += 0.5/Zeta * (fs_0.x_18_0 - fs_1.x_18_0); 
  x_18_8 = PBy * fp_0.x_18_2 - PCy * fp_1.x_18_2; 
  x_18_8 += 0.5/Zeta * (fs_0.x_18_0 - fs_1.x_18_0); 
  x_18_8 += 0.5/Zeta * 3.000000 * (dp_0.x_8_2 - dp_1.x_8_2); 
  x_18_9 = PBz * fp_0.x_18_3 - PCz * fp_1.x_18_3; 
  x_18_9 += 0.5/Zeta * (fs_0.x_18_0 - fs_1.x_18_0); 
  x_19_4 = PBx * fp_0.x_19_2 - PCx * fp_1.x_19_2; 
  x_19_5 = PBy * fp_0.x_19_3 - PCy * fp_1.x_19_3; 
  x_19_6 = PBx * fp_0.x_19_3 - PCx * fp_1.x_19_3; 
  x_19_7 = PBx * fp_0.x_19_1 - PCx * fp_1.x_19_1; 
  x_19_7 += 0.5/Zeta * (fs_0.x_19_0 - fs_1.x_19_0); 
  x_19_8 = PBy * fp_0.x_19_2 - PCy * fp_1.x_19_2; 
  x_19_8 += 0.5/Zeta * (fs_0.x_19_0 - fs_1.x_19_0); 
  x_19_9 = PBz * fp_0.x_19_3 - PCz * fp_1.x_19_3; 
  x_19_9 += 0.5/Zeta * (fs_0.x_19_0 - fs_1.x_19_0); 
  x_19_9 += 0.5/Zeta * 3.000000 * (dp_0.x_9_3 - dp_1.x_9_3); 

 } 

/* FD auxilary integral, m=1 */ 
__device__ __inline__ FDint_1::FDint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  DPint_1 dp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=1 
  FSint_1 fs_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=1 
  FPint_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|p] for m=1 
  DPint_2 dp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|p] for m=2 
  FSint_2 fs_2(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|s] for m=2 
  FPint_2 fp_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|p] for m=2 

  x_10_4 = PBx * fp_1.x_10_2 - PCx * fp_2.x_10_2; 
  x_10_4 += 0.5/Zeta * 1.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
  x_10_5 = PBy * fp_1.x_10_3 - PCy * fp_2.x_10_3; 
  x_10_5 += 0.5/Zeta * 1.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  x_10_6 = PBx * fp_1.x_10_3 - PCx * fp_2.x_10_3; 
  x_10_6 += 0.5/Zeta * 1.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  x_10_7 = PBx * fp_1.x_10_1 - PCx * fp_2.x_10_1; 
  x_10_7 += 0.5/Zeta * (fs_1.x_10_0 - fs_2.x_10_0); 
  x_10_7 += 0.5/Zeta * 1.000000 * (dp_1.x_5_1 - dp_2.x_5_1); 
  x_10_8 = PBy * fp_1.x_10_2 - PCy * fp_2.x_10_2; 
  x_10_8 += 0.5/Zeta * (fs_1.x_10_0 - fs_2.x_10_0); 
  x_10_8 += 0.5/Zeta * 1.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
  x_10_9 = PBz * fp_1.x_10_3 - PCz * fp_2.x_10_3; 
  x_10_9 += 0.5/Zeta * (fs_1.x_10_0 - fs_2.x_10_0); 
  x_10_9 += 0.5/Zeta * 1.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  x_11_4 = PBx * fp_1.x_11_2 - PCx * fp_2.x_11_2; 
  x_11_4 += 0.5/Zeta * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
  x_11_5 = PBy * fp_1.x_11_3 - PCy * fp_2.x_11_3; 
  x_11_5 += 0.5/Zeta * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  x_11_6 = PBx * fp_1.x_11_3 - PCx * fp_2.x_11_3; 
  x_11_6 += 0.5/Zeta * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  x_11_7 = PBx * fp_1.x_11_1 - PCx * fp_2.x_11_1; 
  x_11_7 += 0.5/Zeta * (fs_1.x_11_0 - fs_2.x_11_0); 
  x_11_7 += 0.5/Zeta * 2.000000 * (dp_1.x_4_1 - dp_2.x_4_1); 
  x_11_8 = PBy * fp_1.x_11_2 - PCy * fp_2.x_11_2; 
  x_11_8 += 0.5/Zeta * (fs_1.x_11_0 - fs_2.x_11_0); 
  x_11_8 += 0.5/Zeta * 1.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
  x_11_9 = PBz * fp_1.x_11_3 - PCz * fp_2.x_11_3; 
  x_11_9 += 0.5/Zeta * (fs_1.x_11_0 - fs_2.x_11_0); 
  x_12_4 = PBx * fp_1.x_12_2 - PCx * fp_2.x_12_2; 
  x_12_4 += 0.5/Zeta * 1.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
  x_12_5 = PBy * fp_1.x_12_3 - PCy * fp_2.x_12_3; 
  x_12_5 += 0.5/Zeta * 2.000000 * (dp_1.x_4_3 - dp_2.x_4_3); 
  x_12_6 = PBx * fp_1.x_12_3 - PCx * fp_2.x_12_3; 
  x_12_6 += 0.5/Zeta * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  x_12_7 = PBx * fp_1.x_12_1 - PCx * fp_2.x_12_1; 
  x_12_7 += 0.5/Zeta * (fs_1.x_12_0 - fs_2.x_12_0); 
  x_12_7 += 0.5/Zeta * 1.000000 * (dp_1.x_8_1 - dp_2.x_8_1); 
  x_12_8 = PBy * fp_1.x_12_2 - PCy * fp_2.x_12_2; 
  x_12_8 += 0.5/Zeta * (fs_1.x_12_0 - fs_2.x_12_0); 
  x_12_8 += 0.5/Zeta * 2.000000 * (dp_1.x_4_2 - dp_2.x_4_2); 
  x_12_9 = PBz * fp_1.x_12_3 - PCz * fp_2.x_12_3; 
  x_12_9 += 0.5/Zeta * (fs_1.x_12_0 - fs_2.x_12_0); 
  x_13_4 = PBx * fp_1.x_13_2 - PCx * fp_2.x_13_2; 
  x_13_4 += 0.5/Zeta * 2.000000 * (dp_1.x_6_2 - dp_2.x_6_2); 
  x_13_5 = PBy * fp_1.x_13_3 - PCy * fp_2.x_13_3; 
  x_13_6 = PBx * fp_1.x_13_3 - PCx * fp_2.x_13_3; 
  x_13_6 += 0.5/Zeta * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  x_13_7 = PBx * fp_1.x_13_1 - PCx * fp_2.x_13_1; 
  x_13_7 += 0.5/Zeta * (fs_1.x_13_0 - fs_2.x_13_0); 
  x_13_7 += 0.5/Zeta * 2.000000 * (dp_1.x_6_1 - dp_2.x_6_1); 
  x_13_8 = PBy * fp_1.x_13_2 - PCy * fp_2.x_13_2; 
  x_13_8 += 0.5/Zeta * (fs_1.x_13_0 - fs_2.x_13_0); 
  x_13_9 = PBz * fp_1.x_13_3 - PCz * fp_2.x_13_3; 
  x_13_9 += 0.5/Zeta * (fs_1.x_13_0 - fs_2.x_13_0); 
  x_13_9 += 0.5/Zeta * 1.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  x_14_4 = PBx * fp_1.x_14_2 - PCx * fp_2.x_14_2; 
  x_14_4 += 0.5/Zeta * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
  x_14_5 = PBy * fp_1.x_14_3 - PCy * fp_2.x_14_3; 
  x_14_6 = PBx * fp_1.x_14_3 - PCx * fp_2.x_14_3; 
  x_14_6 += 0.5/Zeta * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  x_14_7 = PBx * fp_1.x_14_1 - PCx * fp_2.x_14_1; 
  x_14_7 += 0.5/Zeta * (fs_1.x_14_0 - fs_2.x_14_0); 
  x_14_7 += 0.5/Zeta * 1.000000 * (dp_1.x_9_1 - dp_2.x_9_1); 
  x_14_8 = PBy * fp_1.x_14_2 - PCy * fp_2.x_14_2; 
  x_14_8 += 0.5/Zeta * (fs_1.x_14_0 - fs_2.x_14_0); 
  x_14_9 = PBz * fp_1.x_14_3 - PCz * fp_2.x_14_3; 
  x_14_9 += 0.5/Zeta * (fs_1.x_14_0 - fs_2.x_14_0); 
  x_14_9 += 0.5/Zeta * 2.000000 * (dp_1.x_6_3 - dp_2.x_6_3); 
  x_15_4 = PBx * fp_1.x_15_2 - PCx * fp_2.x_15_2; 
  x_15_5 = PBy * fp_1.x_15_3 - PCy * fp_2.x_15_3; 
  x_15_5 += 0.5/Zeta * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  x_15_6 = PBx * fp_1.x_15_3 - PCx * fp_2.x_15_3; 
  x_15_7 = PBx * fp_1.x_15_1 - PCx * fp_2.x_15_1; 
  x_15_7 += 0.5/Zeta * (fs_1.x_15_0 - fs_2.x_15_0); 
  x_15_8 = PBy * fp_1.x_15_2 - PCy * fp_2.x_15_2; 
  x_15_8 += 0.5/Zeta * (fs_1.x_15_0 - fs_2.x_15_0); 
  x_15_8 += 0.5/Zeta * 2.000000 * (dp_1.x_5_2 - dp_2.x_5_2); 
  x_15_9 = PBz * fp_1.x_15_3 - PCz * fp_2.x_15_3; 
  x_15_9 += 0.5/Zeta * (fs_1.x_15_0 - fs_2.x_15_0); 
  x_15_9 += 0.5/Zeta * 1.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  x_16_4 = PBx * fp_1.x_16_2 - PCx * fp_2.x_16_2; 
  x_16_5 = PBy * fp_1.x_16_3 - PCy * fp_2.x_16_3; 
  x_16_5 += 0.5/Zeta * 1.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 
  x_16_6 = PBx * fp_1.x_16_3 - PCx * fp_2.x_16_3; 
  x_16_7 = PBx * fp_1.x_16_1 - PCx * fp_2.x_16_1; 
  x_16_7 += 0.5/Zeta * (fs_1.x_16_0 - fs_2.x_16_0); 
  x_16_8 = PBy * fp_1.x_16_2 - PCy * fp_2.x_16_2; 
  x_16_8 += 0.5/Zeta * (fs_1.x_16_0 - fs_2.x_16_0); 
  x_16_8 += 0.5/Zeta * 1.000000 * (dp_1.x_9_2 - dp_2.x_9_2); 
  x_16_9 = PBz * fp_1.x_16_3 - PCz * fp_2.x_16_3; 
  x_16_9 += 0.5/Zeta * (fs_1.x_16_0 - fs_2.x_16_0); 
  x_16_9 += 0.5/Zeta * 2.000000 * (dp_1.x_5_3 - dp_2.x_5_3); 
  x_17_4 = PBx * fp_1.x_17_2 - PCx * fp_2.x_17_2; 
  x_17_4 += 0.5/Zeta * 3.000000 * (dp_1.x_7_2 - dp_2.x_7_2); 
  x_17_5 = PBy * fp_1.x_17_3 - PCy * fp_2.x_17_3; 
  x_17_6 = PBx * fp_1.x_17_3 - PCx * fp_2.x_17_3; 
  x_17_6 += 0.5/Zeta * 3.000000 * (dp_1.x_7_3 - dp_2.x_7_3); 
  x_17_7 = PBx * fp_1.x_17_1 - PCx * fp_2.x_17_1; 
  x_17_7 += 0.5/Zeta * (fs_1.x_17_0 - fs_2.x_17_0); 
  x_17_7 += 0.5/Zeta * 3.000000 * (dp_1.x_7_1 - dp_2.x_7_1); 
  x_17_8 = PBy * fp_1.x_17_2 - PCy * fp_2.x_17_2; 
  x_17_8 += 0.5/Zeta * (fs_1.x_17_0 - fs_2.x_17_0); 
  x_17_9 = PBz * fp_1.x_17_3 - PCz * fp_2.x_17_3; 
  x_17_9 += 0.5/Zeta * (fs_1.x_17_0 - fs_2.x_17_0); 
  x_18_4 = PBx * fp_1.x_18_2 - PCx * fp_2.x_18_2; 
  x_18_5 = PBy * fp_1.x_18_3 - PCy * fp_2.x_18_3; 
  x_18_5 += 0.5/Zeta * 3.000000 * (dp_1.x_8_3 - dp_2.x_8_3); 
  x_18_6 = PBx * fp_1.x_18_3 - PCx * fp_2.x_18_3; 
  x_18_7 = PBx * fp_1.x_18_1 - PCx * fp_2.x_18_1; 
  x_18_7 += 0.5/Zeta * (fs_1.x_18_0 - fs_2.x_18_0); 
  x_18_8 = PBy * fp_1.x_18_2 - PCy * fp_2.x_18_2; 
  x_18_8 += 0.5/Zeta * (fs_1.x_18_0 - fs_2.x_18_0); 
  x_18_8 += 0.5/Zeta * 3.000000 * (dp_1.x_8_2 - dp_2.x_8_2); 
  x_18_9 = PBz * fp_1.x_18_3 - PCz * fp_2.x_18_3; 
  x_18_9 += 0.5/Zeta * (fs_1.x_18_0 - fs_2.x_18_0); 
  x_19_4 = PBx * fp_1.x_19_2 - PCx * fp_2.x_19_2; 
  x_19_5 = PBy * fp_1.x_19_3 - PCy * fp_2.x_19_3; 
  x_19_6 = PBx * fp_1.x_19_3 - PCx * fp_2.x_19_3; 
  x_19_7 = PBx * fp_1.x_19_1 - PCx * fp_2.x_19_1; 
  x_19_7 += 0.5/Zeta * (fs_1.x_19_0 - fs_2.x_19_0); 
  x_19_8 = PBy * fp_1.x_19_2 - PCy * fp_2.x_19_2; 
  x_19_8 += 0.5/Zeta * (fs_1.x_19_0 - fs_2.x_19_0); 
  x_19_9 = PBz * fp_1.x_19_3 - PCz * fp_2.x_19_3; 
  x_19_9 += 0.5/Zeta * (fs_1.x_19_0 - fs_2.x_19_0); 
  x_19_9 += 0.5/Zeta * 3.000000 * (dp_1.x_9_3 - dp_2.x_9_3); 

 } 

/* DF true integral, m=0 */ 
__device__ __inline__ DFint_0::DFint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PDint_0 pd_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|d] for m=0 
  SFint_0 sf_0(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=0 
  PFint_0 pf_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|f] for m=0 
  PDint_1 pd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|d] for m=1 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=1 
  PFint_1 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|f] for m=1 

  x_4_10 = PAx * pf_0.x_2_10 - PCx * pf_1.x_2_10; 
  x_4_10 += 0.5/Zeta * 1.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
  x_5_10 = PAy * pf_0.x_3_10 - PCy * pf_1.x_3_10; 
  x_5_10 += 0.5/Zeta * 1.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  x_6_10 = PAx * pf_0.x_3_10 - PCx * pf_1.x_3_10; 
  x_6_10 += 0.5/Zeta * 1.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  x_7_10 = PAx * pf_0.x_1_10 - PCx * pf_1.x_1_10; 
  x_7_10 += 0.5/Zeta * (sf_0.x_0_10 - sf_1.x_0_10); 
  x_7_10 += 0.5/Zeta * 1.000000 * (pd_0.x_1_5 - pd_1.x_1_5); 
  x_8_10 = PAy * pf_0.x_2_10 - PCy * pf_1.x_2_10; 
  x_8_10 += 0.5/Zeta * (sf_0.x_0_10 - sf_1.x_0_10); 
  x_8_10 += 0.5/Zeta * 1.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
  x_9_10 = PAz * pf_0.x_3_10 - PCz * pf_1.x_3_10; 
  x_9_10 += 0.5/Zeta * (sf_0.x_0_10 - sf_1.x_0_10); 
  x_9_10 += 0.5/Zeta * 1.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  x_4_11 = PAx * pf_0.x_2_11 - PCx * pf_1.x_2_11; 
  x_4_11 += 0.5/Zeta * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
  x_5_11 = PAy * pf_0.x_3_11 - PCy * pf_1.x_3_11; 
  x_5_11 += 0.5/Zeta * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  x_6_11 = PAx * pf_0.x_3_11 - PCx * pf_1.x_3_11; 
  x_6_11 += 0.5/Zeta * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  x_7_11 = PAx * pf_0.x_1_11 - PCx * pf_1.x_1_11; 
  x_7_11 += 0.5/Zeta * (sf_0.x_0_11 - sf_1.x_0_11); 
  x_7_11 += 0.5/Zeta * 2.000000 * (pd_0.x_1_4 - pd_1.x_1_4); 
  x_8_11 = PAy * pf_0.x_2_11 - PCy * pf_1.x_2_11; 
  x_8_11 += 0.5/Zeta * (sf_0.x_0_11 - sf_1.x_0_11); 
  x_8_11 += 0.5/Zeta * 1.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
  x_9_11 = PAz * pf_0.x_3_11 - PCz * pf_1.x_3_11; 
  x_9_11 += 0.5/Zeta * (sf_0.x_0_11 - sf_1.x_0_11); 
  x_4_12 = PAx * pf_0.x_2_12 - PCx * pf_1.x_2_12; 
  x_4_12 += 0.5/Zeta * 1.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
  x_5_12 = PAy * pf_0.x_3_12 - PCy * pf_1.x_3_12; 
  x_5_12 += 0.5/Zeta * 2.000000 * (pd_0.x_3_4 - pd_1.x_3_4); 
  x_6_12 = PAx * pf_0.x_3_12 - PCx * pf_1.x_3_12; 
  x_6_12 += 0.5/Zeta * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  x_7_12 = PAx * pf_0.x_1_12 - PCx * pf_1.x_1_12; 
  x_7_12 += 0.5/Zeta * (sf_0.x_0_12 - sf_1.x_0_12); 
  x_7_12 += 0.5/Zeta * 1.000000 * (pd_0.x_1_8 - pd_1.x_1_8); 
  x_8_12 = PAy * pf_0.x_2_12 - PCy * pf_1.x_2_12; 
  x_8_12 += 0.5/Zeta * (sf_0.x_0_12 - sf_1.x_0_12); 
  x_8_12 += 0.5/Zeta * 2.000000 * (pd_0.x_2_4 - pd_1.x_2_4); 
  x_9_12 = PAz * pf_0.x_3_12 - PCz * pf_1.x_3_12; 
  x_9_12 += 0.5/Zeta * (sf_0.x_0_12 - sf_1.x_0_12); 
  x_4_13 = PAx * pf_0.x_2_13 - PCx * pf_1.x_2_13; 
  x_4_13 += 0.5/Zeta * 2.000000 * (pd_0.x_2_6 - pd_1.x_2_6); 
  x_5_13 = PAy * pf_0.x_3_13 - PCy * pf_1.x_3_13; 
  x_6_13 = PAx * pf_0.x_3_13 - PCx * pf_1.x_3_13; 
  x_6_13 += 0.5/Zeta * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  x_7_13 = PAx * pf_0.x_1_13 - PCx * pf_1.x_1_13; 
  x_7_13 += 0.5/Zeta * (sf_0.x_0_13 - sf_1.x_0_13); 
  x_7_13 += 0.5/Zeta * 2.000000 * (pd_0.x_1_6 - pd_1.x_1_6); 
  x_8_13 = PAy * pf_0.x_2_13 - PCy * pf_1.x_2_13; 
  x_8_13 += 0.5/Zeta * (sf_0.x_0_13 - sf_1.x_0_13); 
  x_9_13 = PAz * pf_0.x_3_13 - PCz * pf_1.x_3_13; 
  x_9_13 += 0.5/Zeta * (sf_0.x_0_13 - sf_1.x_0_13); 
  x_9_13 += 0.5/Zeta * 1.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  x_4_14 = PAx * pf_0.x_2_14 - PCx * pf_1.x_2_14; 
  x_4_14 += 0.5/Zeta * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
  x_5_14 = PAy * pf_0.x_3_14 - PCy * pf_1.x_3_14; 
  x_6_14 = PAx * pf_0.x_3_14 - PCx * pf_1.x_3_14; 
  x_6_14 += 0.5/Zeta * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  x_7_14 = PAx * pf_0.x_1_14 - PCx * pf_1.x_1_14; 
  x_7_14 += 0.5/Zeta * (sf_0.x_0_14 - sf_1.x_0_14); 
  x_7_14 += 0.5/Zeta * 1.000000 * (pd_0.x_1_9 - pd_1.x_1_9); 
  x_8_14 = PAy * pf_0.x_2_14 - PCy * pf_1.x_2_14; 
  x_8_14 += 0.5/Zeta * (sf_0.x_0_14 - sf_1.x_0_14); 
  x_9_14 = PAz * pf_0.x_3_14 - PCz * pf_1.x_3_14; 
  x_9_14 += 0.5/Zeta * (sf_0.x_0_14 - sf_1.x_0_14); 
  x_9_14 += 0.5/Zeta * 2.000000 * (pd_0.x_3_6 - pd_1.x_3_6); 
  x_4_15 = PAx * pf_0.x_2_15 - PCx * pf_1.x_2_15; 
  x_5_15 = PAy * pf_0.x_3_15 - PCy * pf_1.x_3_15; 
  x_5_15 += 0.5/Zeta * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  x_6_15 = PAx * pf_0.x_3_15 - PCx * pf_1.x_3_15; 
  x_7_15 = PAx * pf_0.x_1_15 - PCx * pf_1.x_1_15; 
  x_7_15 += 0.5/Zeta * (sf_0.x_0_15 - sf_1.x_0_15); 
  x_8_15 = PAy * pf_0.x_2_15 - PCy * pf_1.x_2_15; 
  x_8_15 += 0.5/Zeta * (sf_0.x_0_15 - sf_1.x_0_15); 
  x_8_15 += 0.5/Zeta * 2.000000 * (pd_0.x_2_5 - pd_1.x_2_5); 
  x_9_15 = PAz * pf_0.x_3_15 - PCz * pf_1.x_3_15; 
  x_9_15 += 0.5/Zeta * (sf_0.x_0_15 - sf_1.x_0_15); 
  x_9_15 += 0.5/Zeta * 1.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  x_4_16 = PAx * pf_0.x_2_16 - PCx * pf_1.x_2_16; 
  x_5_16 = PAy * pf_0.x_3_16 - PCy * pf_1.x_3_16; 
  x_5_16 += 0.5/Zeta * 1.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 
  x_6_16 = PAx * pf_0.x_3_16 - PCx * pf_1.x_3_16; 
  x_7_16 = PAx * pf_0.x_1_16 - PCx * pf_1.x_1_16; 
  x_7_16 += 0.5/Zeta * (sf_0.x_0_16 - sf_1.x_0_16); 
  x_8_16 = PAy * pf_0.x_2_16 - PCy * pf_1.x_2_16; 
  x_8_16 += 0.5/Zeta * (sf_0.x_0_16 - sf_1.x_0_16); 
  x_8_16 += 0.5/Zeta * 1.000000 * (pd_0.x_2_9 - pd_1.x_2_9); 
  x_9_16 = PAz * pf_0.x_3_16 - PCz * pf_1.x_3_16; 
  x_9_16 += 0.5/Zeta * (sf_0.x_0_16 - sf_1.x_0_16); 
  x_9_16 += 0.5/Zeta * 2.000000 * (pd_0.x_3_5 - pd_1.x_3_5); 
  x_4_17 = PAx * pf_0.x_2_17 - PCx * pf_1.x_2_17; 
  x_4_17 += 0.5/Zeta * 3.000000 * (pd_0.x_2_7 - pd_1.x_2_7); 
  x_5_17 = PAy * pf_0.x_3_17 - PCy * pf_1.x_3_17; 
  x_6_17 = PAx * pf_0.x_3_17 - PCx * pf_1.x_3_17; 
  x_6_17 += 0.5/Zeta * 3.000000 * (pd_0.x_3_7 - pd_1.x_3_7); 
  x_7_17 = PAx * pf_0.x_1_17 - PCx * pf_1.x_1_17; 
  x_7_17 += 0.5/Zeta * (sf_0.x_0_17 - sf_1.x_0_17); 
  x_7_17 += 0.5/Zeta * 3.000000 * (pd_0.x_1_7 - pd_1.x_1_7); 
  x_8_17 = PAy * pf_0.x_2_17 - PCy * pf_1.x_2_17; 
  x_8_17 += 0.5/Zeta * (sf_0.x_0_17 - sf_1.x_0_17); 
  x_9_17 = PAz * pf_0.x_3_17 - PCz * pf_1.x_3_17; 
  x_9_17 += 0.5/Zeta * (sf_0.x_0_17 - sf_1.x_0_17); 
  x_4_18 = PAx * pf_0.x_2_18 - PCx * pf_1.x_2_18; 
  x_5_18 = PAy * pf_0.x_3_18 - PCy * pf_1.x_3_18; 
  x_5_18 += 0.5/Zeta * 3.000000 * (pd_0.x_3_8 - pd_1.x_3_8); 
  x_6_18 = PAx * pf_0.x_3_18 - PCx * pf_1.x_3_18; 
  x_7_18 = PAx * pf_0.x_1_18 - PCx * pf_1.x_1_18; 
  x_7_18 += 0.5/Zeta * (sf_0.x_0_18 - sf_1.x_0_18); 
  x_8_18 = PAy * pf_0.x_2_18 - PCy * pf_1.x_2_18; 
  x_8_18 += 0.5/Zeta * (sf_0.x_0_18 - sf_1.x_0_18); 
  x_8_18 += 0.5/Zeta * 3.000000 * (pd_0.x_2_8 - pd_1.x_2_8); 
  x_9_18 = PAz * pf_0.x_3_18 - PCz * pf_1.x_3_18; 
  x_9_18 += 0.5/Zeta * (sf_0.x_0_18 - sf_1.x_0_18); 
  x_4_19 = PAx * pf_0.x_2_19 - PCx * pf_1.x_2_19; 
  x_5_19 = PAy * pf_0.x_3_19 - PCy * pf_1.x_3_19; 
  x_6_19 = PAx * pf_0.x_3_19 - PCx * pf_1.x_3_19; 
  x_7_19 = PAx * pf_0.x_1_19 - PCx * pf_1.x_1_19; 
  x_7_19 += 0.5/Zeta * (sf_0.x_0_19 - sf_1.x_0_19); 
  x_8_19 = PAy * pf_0.x_2_19 - PCy * pf_1.x_2_19; 
  x_8_19 += 0.5/Zeta * (sf_0.x_0_19 - sf_1.x_0_19); 
  x_9_19 = PAz * pf_0.x_3_19 - PCz * pf_1.x_3_19; 
  x_9_19 += 0.5/Zeta * (sf_0.x_0_19 - sf_1.x_0_19); 
  x_9_19 += 0.5/Zeta * 3.000000 * (pd_0.x_3_9 - pd_1.x_3_9); 

 } 

/* DF auxilary integral, m=1 */ 
__device__ __inline__ DFint_1::DFint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  PDint_1 pd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|d] for m=1 
  SFint_1 sf_1(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=1 
  PFint_1 pf_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|f] for m=1 
  PDint_2 pd_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|d] for m=2 
  SFint_2 sf_2(PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|f] for m=2 
  PFint_2 pf_2(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [p|f] for m=2 

  x_4_10 = PAx * pf_1.x_2_10 - PCx * pf_2.x_2_10; 
  x_4_10 += 0.5/Zeta * 1.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
  x_5_10 = PAy * pf_1.x_3_10 - PCy * pf_2.x_3_10; 
  x_5_10 += 0.5/Zeta * 1.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  x_6_10 = PAx * pf_1.x_3_10 - PCx * pf_2.x_3_10; 
  x_6_10 += 0.5/Zeta * 1.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  x_7_10 = PAx * pf_1.x_1_10 - PCx * pf_2.x_1_10; 
  x_7_10 += 0.5/Zeta * (sf_1.x_0_10 - sf_2.x_0_10); 
  x_7_10 += 0.5/Zeta * 1.000000 * (pd_1.x_1_5 - pd_2.x_1_5); 
  x_8_10 = PAy * pf_1.x_2_10 - PCy * pf_2.x_2_10; 
  x_8_10 += 0.5/Zeta * (sf_1.x_0_10 - sf_2.x_0_10); 
  x_8_10 += 0.5/Zeta * 1.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
  x_9_10 = PAz * pf_1.x_3_10 - PCz * pf_2.x_3_10; 
  x_9_10 += 0.5/Zeta * (sf_1.x_0_10 - sf_2.x_0_10); 
  x_9_10 += 0.5/Zeta * 1.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  x_4_11 = PAx * pf_1.x_2_11 - PCx * pf_2.x_2_11; 
  x_4_11 += 0.5/Zeta * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
  x_5_11 = PAy * pf_1.x_3_11 - PCy * pf_2.x_3_11; 
  x_5_11 += 0.5/Zeta * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  x_6_11 = PAx * pf_1.x_3_11 - PCx * pf_2.x_3_11; 
  x_6_11 += 0.5/Zeta * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  x_7_11 = PAx * pf_1.x_1_11 - PCx * pf_2.x_1_11; 
  x_7_11 += 0.5/Zeta * (sf_1.x_0_11 - sf_2.x_0_11); 
  x_7_11 += 0.5/Zeta * 2.000000 * (pd_1.x_1_4 - pd_2.x_1_4); 
  x_8_11 = PAy * pf_1.x_2_11 - PCy * pf_2.x_2_11; 
  x_8_11 += 0.5/Zeta * (sf_1.x_0_11 - sf_2.x_0_11); 
  x_8_11 += 0.5/Zeta * 1.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
  x_9_11 = PAz * pf_1.x_3_11 - PCz * pf_2.x_3_11; 
  x_9_11 += 0.5/Zeta * (sf_1.x_0_11 - sf_2.x_0_11); 
  x_4_12 = PAx * pf_1.x_2_12 - PCx * pf_2.x_2_12; 
  x_4_12 += 0.5/Zeta * 1.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
  x_5_12 = PAy * pf_1.x_3_12 - PCy * pf_2.x_3_12; 
  x_5_12 += 0.5/Zeta * 2.000000 * (pd_1.x_3_4 - pd_2.x_3_4); 
  x_6_12 = PAx * pf_1.x_3_12 - PCx * pf_2.x_3_12; 
  x_6_12 += 0.5/Zeta * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  x_7_12 = PAx * pf_1.x_1_12 - PCx * pf_2.x_1_12; 
  x_7_12 += 0.5/Zeta * (sf_1.x_0_12 - sf_2.x_0_12); 
  x_7_12 += 0.5/Zeta * 1.000000 * (pd_1.x_1_8 - pd_2.x_1_8); 
  x_8_12 = PAy * pf_1.x_2_12 - PCy * pf_2.x_2_12; 
  x_8_12 += 0.5/Zeta * (sf_1.x_0_12 - sf_2.x_0_12); 
  x_8_12 += 0.5/Zeta * 2.000000 * (pd_1.x_2_4 - pd_2.x_2_4); 
  x_9_12 = PAz * pf_1.x_3_12 - PCz * pf_2.x_3_12; 
  x_9_12 += 0.5/Zeta * (sf_1.x_0_12 - sf_2.x_0_12); 
  x_4_13 = PAx * pf_1.x_2_13 - PCx * pf_2.x_2_13; 
  x_4_13 += 0.5/Zeta * 2.000000 * (pd_1.x_2_6 - pd_2.x_2_6); 
  x_5_13 = PAy * pf_1.x_3_13 - PCy * pf_2.x_3_13; 
  x_6_13 = PAx * pf_1.x_3_13 - PCx * pf_2.x_3_13; 
  x_6_13 += 0.5/Zeta * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  x_7_13 = PAx * pf_1.x_1_13 - PCx * pf_2.x_1_13; 
  x_7_13 += 0.5/Zeta * (sf_1.x_0_13 - sf_2.x_0_13); 
  x_7_13 += 0.5/Zeta * 2.000000 * (pd_1.x_1_6 - pd_2.x_1_6); 
  x_8_13 = PAy * pf_1.x_2_13 - PCy * pf_2.x_2_13; 
  x_8_13 += 0.5/Zeta * (sf_1.x_0_13 - sf_2.x_0_13); 
  x_9_13 = PAz * pf_1.x_3_13 - PCz * pf_2.x_3_13; 
  x_9_13 += 0.5/Zeta * (sf_1.x_0_13 - sf_2.x_0_13); 
  x_9_13 += 0.5/Zeta * 1.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  x_4_14 = PAx * pf_1.x_2_14 - PCx * pf_2.x_2_14; 
  x_4_14 += 0.5/Zeta * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
  x_5_14 = PAy * pf_1.x_3_14 - PCy * pf_2.x_3_14; 
  x_6_14 = PAx * pf_1.x_3_14 - PCx * pf_2.x_3_14; 
  x_6_14 += 0.5/Zeta * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  x_7_14 = PAx * pf_1.x_1_14 - PCx * pf_2.x_1_14; 
  x_7_14 += 0.5/Zeta * (sf_1.x_0_14 - sf_2.x_0_14); 
  x_7_14 += 0.5/Zeta * 1.000000 * (pd_1.x_1_9 - pd_2.x_1_9); 
  x_8_14 = PAy * pf_1.x_2_14 - PCy * pf_2.x_2_14; 
  x_8_14 += 0.5/Zeta * (sf_1.x_0_14 - sf_2.x_0_14); 
  x_9_14 = PAz * pf_1.x_3_14 - PCz * pf_2.x_3_14; 
  x_9_14 += 0.5/Zeta * (sf_1.x_0_14 - sf_2.x_0_14); 
  x_9_14 += 0.5/Zeta * 2.000000 * (pd_1.x_3_6 - pd_2.x_3_6); 
  x_4_15 = PAx * pf_1.x_2_15 - PCx * pf_2.x_2_15; 
  x_5_15 = PAy * pf_1.x_3_15 - PCy * pf_2.x_3_15; 
  x_5_15 += 0.5/Zeta * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  x_6_15 = PAx * pf_1.x_3_15 - PCx * pf_2.x_3_15; 
  x_7_15 = PAx * pf_1.x_1_15 - PCx * pf_2.x_1_15; 
  x_7_15 += 0.5/Zeta * (sf_1.x_0_15 - sf_2.x_0_15); 
  x_8_15 = PAy * pf_1.x_2_15 - PCy * pf_2.x_2_15; 
  x_8_15 += 0.5/Zeta * (sf_1.x_0_15 - sf_2.x_0_15); 
  x_8_15 += 0.5/Zeta * 2.000000 * (pd_1.x_2_5 - pd_2.x_2_5); 
  x_9_15 = PAz * pf_1.x_3_15 - PCz * pf_2.x_3_15; 
  x_9_15 += 0.5/Zeta * (sf_1.x_0_15 - sf_2.x_0_15); 
  x_9_15 += 0.5/Zeta * 1.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  x_4_16 = PAx * pf_1.x_2_16 - PCx * pf_2.x_2_16; 
  x_5_16 = PAy * pf_1.x_3_16 - PCy * pf_2.x_3_16; 
  x_5_16 += 0.5/Zeta * 1.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 
  x_6_16 = PAx * pf_1.x_3_16 - PCx * pf_2.x_3_16; 
  x_7_16 = PAx * pf_1.x_1_16 - PCx * pf_2.x_1_16; 
  x_7_16 += 0.5/Zeta * (sf_1.x_0_16 - sf_2.x_0_16); 
  x_8_16 = PAy * pf_1.x_2_16 - PCy * pf_2.x_2_16; 
  x_8_16 += 0.5/Zeta * (sf_1.x_0_16 - sf_2.x_0_16); 
  x_8_16 += 0.5/Zeta * 1.000000 * (pd_1.x_2_9 - pd_2.x_2_9); 
  x_9_16 = PAz * pf_1.x_3_16 - PCz * pf_2.x_3_16; 
  x_9_16 += 0.5/Zeta * (sf_1.x_0_16 - sf_2.x_0_16); 
  x_9_16 += 0.5/Zeta * 2.000000 * (pd_1.x_3_5 - pd_2.x_3_5); 
  x_4_17 = PAx * pf_1.x_2_17 - PCx * pf_2.x_2_17; 
  x_4_17 += 0.5/Zeta * 3.000000 * (pd_1.x_2_7 - pd_2.x_2_7); 
  x_5_17 = PAy * pf_1.x_3_17 - PCy * pf_2.x_3_17; 
  x_6_17 = PAx * pf_1.x_3_17 - PCx * pf_2.x_3_17; 
  x_6_17 += 0.5/Zeta * 3.000000 * (pd_1.x_3_7 - pd_2.x_3_7); 
  x_7_17 = PAx * pf_1.x_1_17 - PCx * pf_2.x_1_17; 
  x_7_17 += 0.5/Zeta * (sf_1.x_0_17 - sf_2.x_0_17); 
  x_7_17 += 0.5/Zeta * 3.000000 * (pd_1.x_1_7 - pd_2.x_1_7); 
  x_8_17 = PAy * pf_1.x_2_17 - PCy * pf_2.x_2_17; 
  x_8_17 += 0.5/Zeta * (sf_1.x_0_17 - sf_2.x_0_17); 
  x_9_17 = PAz * pf_1.x_3_17 - PCz * pf_2.x_3_17; 
  x_9_17 += 0.5/Zeta * (sf_1.x_0_17 - sf_2.x_0_17); 
  x_4_18 = PAx * pf_1.x_2_18 - PCx * pf_2.x_2_18; 
  x_5_18 = PAy * pf_1.x_3_18 - PCy * pf_2.x_3_18; 
  x_5_18 += 0.5/Zeta * 3.000000 * (pd_1.x_3_8 - pd_2.x_3_8); 
  x_6_18 = PAx * pf_1.x_3_18 - PCx * pf_2.x_3_18; 
  x_7_18 = PAx * pf_1.x_1_18 - PCx * pf_2.x_1_18; 
  x_7_18 += 0.5/Zeta * (sf_1.x_0_18 - sf_2.x_0_18); 
  x_8_18 = PAy * pf_1.x_2_18 - PCy * pf_2.x_2_18; 
  x_8_18 += 0.5/Zeta * (sf_1.x_0_18 - sf_2.x_0_18); 
  x_8_18 += 0.5/Zeta * 3.000000 * (pd_1.x_2_8 - pd_2.x_2_8); 
  x_9_18 = PAz * pf_1.x_3_18 - PCz * pf_2.x_3_18; 
  x_9_18 += 0.5/Zeta * (sf_1.x_0_18 - sf_2.x_0_18); 
  x_4_19 = PAx * pf_1.x_2_19 - PCx * pf_2.x_2_19; 
  x_5_19 = PAy * pf_1.x_3_19 - PCy * pf_2.x_3_19; 
  x_6_19 = PAx * pf_1.x_3_19 - PCx * pf_2.x_3_19; 
  x_7_19 = PAx * pf_1.x_1_19 - PCx * pf_2.x_1_19; 
  x_7_19 += 0.5/Zeta * (sf_1.x_0_19 - sf_2.x_0_19); 
  x_8_19 = PAy * pf_1.x_2_19 - PCy * pf_2.x_2_19; 
  x_8_19 += 0.5/Zeta * (sf_1.x_0_19 - sf_2.x_0_19); 
  x_9_19 = PAz * pf_1.x_3_19 - PCz * pf_2.x_3_19; 
  x_9_19 += 0.5/Zeta * (sf_1.x_0_19 - sf_2.x_0_19); 
  x_9_19 += 0.5/Zeta * 3.000000 * (pd_1.x_3_9 - pd_2.x_3_9); 

 } 

/* FF true integral, m=0 */ 
__device__ __inline__ FFint_0::FFint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  DDint_0 dd_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|d] for m=0 
  FPint_0 fp_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|p] for m=0 
  FDint_0 fd_0(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|d] for m=0 
  DDint_1 dd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [d|d] for m=1 
  FPint_1 fp_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|p] for m=1 
  FDint_1 fd_1(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [f|d] for m=1 

  x_10_10 = PBx * fd_0.x_10_5 - PCx * fd_1.x_10_5; 
  x_10_10 += 0.5/Zeta * 1.000000 * (dd_0.x_5_5 - dd_1.x_5_5); 
  x_11_10 = PBx * fd_0.x_10_4 - PCx * fd_1.x_10_4; 
  x_11_10 += 0.5/Zeta * 1.000000 * (fp_0.x_10_2 - fp_1.x_10_2); 
  x_11_10 += 0.5/Zeta * 1.000000 * (dd_0.x_5_4 - dd_1.x_5_4); 
  x_12_10 = PBx * fd_0.x_10_8 - PCx * fd_1.x_10_8; 
  x_12_10 += 0.5/Zeta * 1.000000 * (dd_0.x_5_8 - dd_1.x_5_8); 
  x_13_10 = PBx * fd_0.x_10_6 - PCx * fd_1.x_10_6; 
  x_13_10 += 0.5/Zeta * 1.000000 * (fp_0.x_10_3 - fp_1.x_10_3); 
  x_13_10 += 0.5/Zeta * 1.000000 * (dd_0.x_5_6 - dd_1.x_5_6); 
  x_14_10 = PBx * fd_0.x_10_9 - PCx * fd_1.x_10_9; 
  x_14_10 += 0.5/Zeta * 1.000000 * (dd_0.x_5_9 - dd_1.x_5_9); 
  x_15_10 = PBy * fd_0.x_10_5 - PCy * fd_1.x_10_5; 
  x_15_10 += 0.5/Zeta * 1.000000 * (fp_0.x_10_3 - fp_1.x_10_3); 
  x_15_10 += 0.5/Zeta * 1.000000 * (dd_0.x_6_5 - dd_1.x_6_5); 
  x_16_10 = PBy * fd_0.x_10_9 - PCy * fd_1.x_10_9; 
  x_16_10 += 0.5/Zeta * 1.000000 * (dd_0.x_6_9 - dd_1.x_6_9); 
  x_17_10 = PBx * fd_0.x_10_7 - PCx * fd_1.x_10_7; 
  x_17_10 += 0.5/Zeta * 2.000000 * (fp_0.x_10_1 - fp_1.x_10_1); 
  x_17_10 += 0.5/Zeta * 1.000000 * (dd_0.x_5_7 - dd_1.x_5_7); 
  x_18_10 = PBy * fd_0.x_10_8 - PCy * fd_1.x_10_8; 
  x_18_10 += 0.5/Zeta * 2.000000 * (fp_0.x_10_2 - fp_1.x_10_2); 
  x_18_10 += 0.5/Zeta * 1.000000 * (dd_0.x_6_8 - dd_1.x_6_8); 
  x_19_10 = PBz * fd_0.x_10_9 - PCz * fd_1.x_10_9; 
  x_19_10 += 0.5/Zeta * 2.000000 * (fp_0.x_10_3 - fp_1.x_10_3); 
  x_19_10 += 0.5/Zeta * 1.000000 * (dd_0.x_4_9 - dd_1.x_4_9); 
  x_10_11 = PBx * fd_0.x_11_5 - PCx * fd_1.x_11_5; 
  x_10_11 += 0.5/Zeta * 2.000000 * (dd_0.x_4_5 - dd_1.x_4_5); 
  x_11_11 = PBx * fd_0.x_11_4 - PCx * fd_1.x_11_4; 
  x_11_11 += 0.5/Zeta * 1.000000 * (fp_0.x_11_2 - fp_1.x_11_2); 
  x_11_11 += 0.5/Zeta * 2.000000 * (dd_0.x_4_4 - dd_1.x_4_4); 
  x_12_11 = PBx * fd_0.x_11_8 - PCx * fd_1.x_11_8; 
  x_12_11 += 0.5/Zeta * 2.000000 * (dd_0.x_4_8 - dd_1.x_4_8); 
  x_13_11 = PBx * fd_0.x_11_6 - PCx * fd_1.x_11_6; 
  x_13_11 += 0.5/Zeta * 1.000000 * (fp_0.x_11_3 - fp_1.x_11_3); 
  x_13_11 += 0.5/Zeta * 2.000000 * (dd_0.x_4_6 - dd_1.x_4_6); 
  x_14_11 = PBx * fd_0.x_11_9 - PCx * fd_1.x_11_9; 
  x_14_11 += 0.5/Zeta * 2.000000 * (dd_0.x_4_9 - dd_1.x_4_9); 
  x_15_11 = PBy * fd_0.x_11_5 - PCy * fd_1.x_11_5; 
  x_15_11 += 0.5/Zeta * 1.000000 * (fp_0.x_11_3 - fp_1.x_11_3); 
  x_15_11 += 0.5/Zeta * 1.000000 * (dd_0.x_7_5 - dd_1.x_7_5); 
  x_16_11 = PBy * fd_0.x_11_9 - PCy * fd_1.x_11_9; 
  x_16_11 += 0.5/Zeta * 1.000000 * (dd_0.x_7_9 - dd_1.x_7_9); 
  x_17_11 = PBx * fd_0.x_11_7 - PCx * fd_1.x_11_7; 
  x_17_11 += 0.5/Zeta * 2.000000 * (fp_0.x_11_1 - fp_1.x_11_1); 
  x_17_11 += 0.5/Zeta * 2.000000 * (dd_0.x_4_7 - dd_1.x_4_7); 
  x_18_11 = PBy * fd_0.x_11_8 - PCy * fd_1.x_11_8; 
  x_18_11 += 0.5/Zeta * 2.000000 * (fp_0.x_11_2 - fp_1.x_11_2); 
  x_18_11 += 0.5/Zeta * 1.000000 * (dd_0.x_7_8 - dd_1.x_7_8); 
  x_19_11 = PBz * fd_0.x_11_9 - PCz * fd_1.x_11_9; 
  x_19_11 += 0.5/Zeta * 2.000000 * (fp_0.x_11_3 - fp_1.x_11_3); 
  x_10_12 = PBx * fd_0.x_12_5 - PCx * fd_1.x_12_5; 
  x_10_12 += 0.5/Zeta * 1.000000 * (dd_0.x_8_5 - dd_1.x_8_5); 
  x_11_12 = PBx * fd_0.x_12_4 - PCx * fd_1.x_12_4; 
  x_11_12 += 0.5/Zeta * 1.000000 * (fp_0.x_12_2 - fp_1.x_12_2); 
  x_11_12 += 0.5/Zeta * 1.000000 * (dd_0.x_8_4 - dd_1.x_8_4); 
  x_12_12 = PBx * fd_0.x_12_8 - PCx * fd_1.x_12_8; 
  x_12_12 += 0.5/Zeta * 1.000000 * (dd_0.x_8_8 - dd_1.x_8_8); 
  x_13_12 = PBx * fd_0.x_12_6 - PCx * fd_1.x_12_6; 
  x_13_12 += 0.5/Zeta * 1.000000 * (fp_0.x_12_3 - fp_1.x_12_3); 
  x_13_12 += 0.5/Zeta * 1.000000 * (dd_0.x_8_6 - dd_1.x_8_6); 
  x_14_12 = PBx * fd_0.x_12_9 - PCx * fd_1.x_12_9; 
  x_14_12 += 0.5/Zeta * 1.000000 * (dd_0.x_8_9 - dd_1.x_8_9); 
  x_15_12 = PBy * fd_0.x_12_5 - PCy * fd_1.x_12_5; 
  x_15_12 += 0.5/Zeta * 1.000000 * (fp_0.x_12_3 - fp_1.x_12_3); 
  x_15_12 += 0.5/Zeta * 2.000000 * (dd_0.x_4_5 - dd_1.x_4_5); 
  x_16_12 = PBy * fd_0.x_12_9 - PCy * fd_1.x_12_9; 
  x_16_12 += 0.5/Zeta * 2.000000 * (dd_0.x_4_9 - dd_1.x_4_9); 
  x_17_12 = PBx * fd_0.x_12_7 - PCx * fd_1.x_12_7; 
  x_17_12 += 0.5/Zeta * 2.000000 * (fp_0.x_12_1 - fp_1.x_12_1); 
  x_17_12 += 0.5/Zeta * 1.000000 * (dd_0.x_8_7 - dd_1.x_8_7); 
  x_18_12 = PBy * fd_0.x_12_8 - PCy * fd_1.x_12_8; 
  x_18_12 += 0.5/Zeta * 2.000000 * (fp_0.x_12_2 - fp_1.x_12_2); 
  x_18_12 += 0.5/Zeta * 2.000000 * (dd_0.x_4_8 - dd_1.x_4_8); 
  x_19_12 = PBz * fd_0.x_12_9 - PCz * fd_1.x_12_9; 
  x_19_12 += 0.5/Zeta * 2.000000 * (fp_0.x_12_3 - fp_1.x_12_3); 
  x_10_13 = PBx * fd_0.x_13_5 - PCx * fd_1.x_13_5; 
  x_10_13 += 0.5/Zeta * 2.000000 * (dd_0.x_6_5 - dd_1.x_6_5); 
  x_11_13 = PBx * fd_0.x_13_4 - PCx * fd_1.x_13_4; 
  x_11_13 += 0.5/Zeta * 1.000000 * (fp_0.x_13_2 - fp_1.x_13_2); 
  x_11_13 += 0.5/Zeta * 2.000000 * (dd_0.x_6_4 - dd_1.x_6_4); 
  x_12_13 = PBx * fd_0.x_13_8 - PCx * fd_1.x_13_8; 
  x_12_13 += 0.5/Zeta * 2.000000 * (dd_0.x_6_8 - dd_1.x_6_8); 
  x_13_13 = PBx * fd_0.x_13_6 - PCx * fd_1.x_13_6; 
  x_13_13 += 0.5/Zeta * 1.000000 * (fp_0.x_13_3 - fp_1.x_13_3); 
  x_13_13 += 0.5/Zeta * 2.000000 * (dd_0.x_6_6 - dd_1.x_6_6); 
  x_14_13 = PBx * fd_0.x_13_9 - PCx * fd_1.x_13_9; 
  x_14_13 += 0.5/Zeta * 2.000000 * (dd_0.x_6_9 - dd_1.x_6_9); 
  x_15_13 = PBy * fd_0.x_13_5 - PCy * fd_1.x_13_5; 
  x_15_13 += 0.5/Zeta * 1.000000 * (fp_0.x_13_3 - fp_1.x_13_3); 
  x_16_13 = PBy * fd_0.x_13_9 - PCy * fd_1.x_13_9; 
  x_17_13 = PBx * fd_0.x_13_7 - PCx * fd_1.x_13_7; 
  x_17_13 += 0.5/Zeta * 2.000000 * (fp_0.x_13_1 - fp_1.x_13_1); 
  x_17_13 += 0.5/Zeta * 2.000000 * (dd_0.x_6_7 - dd_1.x_6_7); 
  x_18_13 = PBy * fd_0.x_13_8 - PCy * fd_1.x_13_8; 
  x_18_13 += 0.5/Zeta * 2.000000 * (fp_0.x_13_2 - fp_1.x_13_2); 
  x_19_13 = PBz * fd_0.x_13_9 - PCz * fd_1.x_13_9; 
  x_19_13 += 0.5/Zeta * 2.000000 * (fp_0.x_13_3 - fp_1.x_13_3); 
  x_19_13 += 0.5/Zeta * 1.000000 * (dd_0.x_7_9 - dd_1.x_7_9); 
  x_10_14 = PBx * fd_0.x_14_5 - PCx * fd_1.x_14_5; 
  x_10_14 += 0.5/Zeta * 1.000000 * (dd_0.x_9_5 - dd_1.x_9_5); 
  x_11_14 = PBx * fd_0.x_14_4 - PCx * fd_1.x_14_4; 
  x_11_14 += 0.5/Zeta * 1.000000 * (fp_0.x_14_2 - fp_1.x_14_2); 
  x_11_14 += 0.5/Zeta * 1.000000 * (dd_0.x_9_4 - dd_1.x_9_4); 
  x_12_14 = PBx * fd_0.x_14_8 - PCx * fd_1.x_14_8; 
  x_12_14 += 0.5/Zeta * 1.000000 * (dd_0.x_9_8 - dd_1.x_9_8); 
  x_13_14 = PBx * fd_0.x_14_6 - PCx * fd_1.x_14_6; 
  x_13_14 += 0.5/Zeta * 1.000000 * (fp_0.x_14_3 - fp_1.x_14_3); 
  x_13_14 += 0.5/Zeta * 1.000000 * (dd_0.x_9_6 - dd_1.x_9_6); 
  x_14_14 = PBx * fd_0.x_14_9 - PCx * fd_1.x_14_9; 
  x_14_14 += 0.5/Zeta * 1.000000 * (dd_0.x_9_9 - dd_1.x_9_9); 
  x_15_14 = PBy * fd_0.x_14_5 - PCy * fd_1.x_14_5; 
  x_15_14 += 0.5/Zeta * 1.000000 * (fp_0.x_14_3 - fp_1.x_14_3); 
  x_16_14 = PBy * fd_0.x_14_9 - PCy * fd_1.x_14_9; 
  x_17_14 = PBx * fd_0.x_14_7 - PCx * fd_1.x_14_7; 
  x_17_14 += 0.5/Zeta * 2.000000 * (fp_0.x_14_1 - fp_1.x_14_1); 
  x_17_14 += 0.5/Zeta * 1.000000 * (dd_0.x_9_7 - dd_1.x_9_7); 
  x_18_14 = PBy * fd_0.x_14_8 - PCy * fd_1.x_14_8; 
  x_18_14 += 0.5/Zeta * 2.000000 * (fp_0.x_14_2 - fp_1.x_14_2); 
  x_19_14 = PBz * fd_0.x_14_9 - PCz * fd_1.x_14_9; 
  x_19_14 += 0.5/Zeta * 2.000000 * (fp_0.x_14_3 - fp_1.x_14_3); 
  x_19_14 += 0.5/Zeta * 2.000000 * (dd_0.x_6_9 - dd_1.x_6_9); 
  x_10_15 = PBx * fd_0.x_15_5 - PCx * fd_1.x_15_5; 
  x_11_15 = PBx * fd_0.x_15_4 - PCx * fd_1.x_15_4; 
  x_11_15 += 0.5/Zeta * 1.000000 * (fp_0.x_15_2 - fp_1.x_15_2); 
  x_12_15 = PBx * fd_0.x_15_8 - PCx * fd_1.x_15_8; 
  x_13_15 = PBx * fd_0.x_15_6 - PCx * fd_1.x_15_6; 
  x_13_15 += 0.5/Zeta * 1.000000 * (fp_0.x_15_3 - fp_1.x_15_3); 
  x_14_15 = PBx * fd_0.x_15_9 - PCx * fd_1.x_15_9; 
  x_15_15 = PBy * fd_0.x_15_5 - PCy * fd_1.x_15_5; 
  x_15_15 += 0.5/Zeta * 1.000000 * (fp_0.x_15_3 - fp_1.x_15_3); 
  x_15_15 += 0.5/Zeta * 2.000000 * (dd_0.x_5_5 - dd_1.x_5_5); 
  x_16_15 = PBy * fd_0.x_15_9 - PCy * fd_1.x_15_9; 
  x_16_15 += 0.5/Zeta * 2.000000 * (dd_0.x_5_9 - dd_1.x_5_9); 
  x_17_15 = PBx * fd_0.x_15_7 - PCx * fd_1.x_15_7; 
  x_17_15 += 0.5/Zeta * 2.000000 * (fp_0.x_15_1 - fp_1.x_15_1); 
  x_18_15 = PBy * fd_0.x_15_8 - PCy * fd_1.x_15_8; 
  x_18_15 += 0.5/Zeta * 2.000000 * (fp_0.x_15_2 - fp_1.x_15_2); 
  x_18_15 += 0.5/Zeta * 2.000000 * (dd_0.x_5_8 - dd_1.x_5_8); 
  x_19_15 = PBz * fd_0.x_15_9 - PCz * fd_1.x_15_9; 
  x_19_15 += 0.5/Zeta * 2.000000 * (fp_0.x_15_3 - fp_1.x_15_3); 
  x_19_15 += 0.5/Zeta * 1.000000 * (dd_0.x_8_9 - dd_1.x_8_9); 
  x_10_16 = PBx * fd_0.x_16_5 - PCx * fd_1.x_16_5; 
  x_11_16 = PBx * fd_0.x_16_4 - PCx * fd_1.x_16_4; 
  x_11_16 += 0.5/Zeta * 1.000000 * (fp_0.x_16_2 - fp_1.x_16_2); 
  x_12_16 = PBx * fd_0.x_16_8 - PCx * fd_1.x_16_8; 
  x_13_16 = PBx * fd_0.x_16_6 - PCx * fd_1.x_16_6; 
  x_13_16 += 0.5/Zeta * 1.000000 * (fp_0.x_16_3 - fp_1.x_16_3); 
  x_14_16 = PBx * fd_0.x_16_9 - PCx * fd_1.x_16_9; 
  x_15_16 = PBy * fd_0.x_16_5 - PCy * fd_1.x_16_5; 
  x_15_16 += 0.5/Zeta * 1.000000 * (fp_0.x_16_3 - fp_1.x_16_3); 
  x_15_16 += 0.5/Zeta * 1.000000 * (dd_0.x_9_5 - dd_1.x_9_5); 
  x_16_16 = PBy * fd_0.x_16_9 - PCy * fd_1.x_16_9; 
  x_16_16 += 0.5/Zeta * 1.000000 * (dd_0.x_9_9 - dd_1.x_9_9); 
  x_17_16 = PBx * fd_0.x_16_7 - PCx * fd_1.x_16_7; 
  x_17_16 += 0.5/Zeta * 2.000000 * (fp_0.x_16_1 - fp_1.x_16_1); 
  x_18_16 = PBy * fd_0.x_16_8 - PCy * fd_1.x_16_8; 
  x_18_16 += 0.5/Zeta * 2.000000 * (fp_0.x_16_2 - fp_1.x_16_2); 
  x_18_16 += 0.5/Zeta * 1.000000 * (dd_0.x_9_8 - dd_1.x_9_8); 
  x_19_16 = PBz * fd_0.x_16_9 - PCz * fd_1.x_16_9; 
  x_19_16 += 0.5/Zeta * 2.000000 * (fp_0.x_16_3 - fp_1.x_16_3); 
  x_19_16 += 0.5/Zeta * 2.000000 * (dd_0.x_5_9 - dd_1.x_5_9); 
  x_10_17 = PBx * fd_0.x_17_5 - PCx * fd_1.x_17_5; 
  x_10_17 += 0.5/Zeta * 3.000000 * (dd_0.x_7_5 - dd_1.x_7_5); 
  x_11_17 = PBx * fd_0.x_17_4 - PCx * fd_1.x_17_4; 
  x_11_17 += 0.5/Zeta * 1.000000 * (fp_0.x_17_2 - fp_1.x_17_2); 
  x_11_17 += 0.5/Zeta * 3.000000 * (dd_0.x_7_4 - dd_1.x_7_4); 
  x_12_17 = PBx * fd_0.x_17_8 - PCx * fd_1.x_17_8; 
  x_12_17 += 0.5/Zeta * 3.000000 * (dd_0.x_7_8 - dd_1.x_7_8); 
  x_13_17 = PBx * fd_0.x_17_6 - PCx * fd_1.x_17_6; 
  x_13_17 += 0.5/Zeta * 1.000000 * (fp_0.x_17_3 - fp_1.x_17_3); 
  x_13_17 += 0.5/Zeta * 3.000000 * (dd_0.x_7_6 - dd_1.x_7_6); 
  x_14_17 = PBx * fd_0.x_17_9 - PCx * fd_1.x_17_9; 
  x_14_17 += 0.5/Zeta * 3.000000 * (dd_0.x_7_9 - dd_1.x_7_9); 
  x_15_17 = PBy * fd_0.x_17_5 - PCy * fd_1.x_17_5; 
  x_15_17 += 0.5/Zeta * 1.000000 * (fp_0.x_17_3 - fp_1.x_17_3); 
  x_16_17 = PBy * fd_0.x_17_9 - PCy * fd_1.x_17_9; 
  x_17_17 = PBx * fd_0.x_17_7 - PCx * fd_1.x_17_7; 
  x_17_17 += 0.5/Zeta * 2.000000 * (fp_0.x_17_1 - fp_1.x_17_1); 
  x_17_17 += 0.5/Zeta * 3.000000 * (dd_0.x_7_7 - dd_1.x_7_7); 
  x_18_17 = PBy * fd_0.x_17_8 - PCy * fd_1.x_17_8; 
  x_18_17 += 0.5/Zeta * 2.000000 * (fp_0.x_17_2 - fp_1.x_17_2); 
  x_19_17 = PBz * fd_0.x_17_9 - PCz * fd_1.x_17_9; 
  x_19_17 += 0.5/Zeta * 2.000000 * (fp_0.x_17_3 - fp_1.x_17_3); 
  x_10_18 = PBx * fd_0.x_18_5 - PCx * fd_1.x_18_5; 
  x_11_18 = PBx * fd_0.x_18_4 - PCx * fd_1.x_18_4; 
  x_11_18 += 0.5/Zeta * 1.000000 * (fp_0.x_18_2 - fp_1.x_18_2); 
  x_12_18 = PBx * fd_0.x_18_8 - PCx * fd_1.x_18_8; 
  x_13_18 = PBx * fd_0.x_18_6 - PCx * fd_1.x_18_6; 
  x_13_18 += 0.5/Zeta * 1.000000 * (fp_0.x_18_3 - fp_1.x_18_3); 
  x_14_18 = PBx * fd_0.x_18_9 - PCx * fd_1.x_18_9; 
  x_15_18 = PBy * fd_0.x_18_5 - PCy * fd_1.x_18_5; 
  x_15_18 += 0.5/Zeta * 1.000000 * (fp_0.x_18_3 - fp_1.x_18_3); 
  x_15_18 += 0.5/Zeta * 3.000000 * (dd_0.x_8_5 - dd_1.x_8_5); 
  x_16_18 = PBy * fd_0.x_18_9 - PCy * fd_1.x_18_9; 
  x_16_18 += 0.5/Zeta * 3.000000 * (dd_0.x_8_9 - dd_1.x_8_9); 
  x_17_18 = PBx * fd_0.x_18_7 - PCx * fd_1.x_18_7; 
  x_17_18 += 0.5/Zeta * 2.000000 * (fp_0.x_18_1 - fp_1.x_18_1); 
  x_18_18 = PBy * fd_0.x_18_8 - PCy * fd_1.x_18_8; 
  x_18_18 += 0.5/Zeta * 2.000000 * (fp_0.x_18_2 - fp_1.x_18_2); 
  x_18_18 += 0.5/Zeta * 3.000000 * (dd_0.x_8_8 - dd_1.x_8_8); 
  x_19_18 = PBz * fd_0.x_18_9 - PCz * fd_1.x_18_9; 
  x_19_18 += 0.5/Zeta * 2.000000 * (fp_0.x_18_3 - fp_1.x_18_3); 
  x_10_19 = PBx * fd_0.x_19_5 - PCx * fd_1.x_19_5; 
  x_11_19 = PBx * fd_0.x_19_4 - PCx * fd_1.x_19_4; 
  x_11_19 += 0.5/Zeta * 1.000000 * (fp_0.x_19_2 - fp_1.x_19_2); 
  x_12_19 = PBx * fd_0.x_19_8 - PCx * fd_1.x_19_8; 
  x_13_19 = PBx * fd_0.x_19_6 - PCx * fd_1.x_19_6; 
  x_13_19 += 0.5/Zeta * 1.000000 * (fp_0.x_19_3 - fp_1.x_19_3); 
  x_14_19 = PBx * fd_0.x_19_9 - PCx * fd_1.x_19_9; 
  x_15_19 = PBy * fd_0.x_19_5 - PCy * fd_1.x_19_5; 
  x_15_19 += 0.5/Zeta * 1.000000 * (fp_0.x_19_3 - fp_1.x_19_3); 
  x_16_19 = PBy * fd_0.x_19_9 - PCy * fd_1.x_19_9; 
  x_17_19 = PBx * fd_0.x_19_7 - PCx * fd_1.x_19_7; 
  x_17_19 += 0.5/Zeta * 2.000000 * (fp_0.x_19_1 - fp_1.x_19_1); 
  x_18_19 = PBy * fd_0.x_19_8 - PCy * fd_1.x_19_8; 
  x_18_19 += 0.5/Zeta * 2.000000 * (fp_0.x_19_2 - fp_1.x_19_2); 
  x_19_19 = PBz * fd_0.x_19_9 - PCz * fd_1.x_19_9; 
  x_19_19 += 0.5/Zeta * 2.000000 * (fp_0.x_19_3 - fp_1.x_19_3); 
  x_19_19 += 0.5/Zeta * 3.000000 * (dd_0.x_9_9 - dd_1.x_9_9); 

 } 
