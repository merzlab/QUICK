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


/* PS true integral, m=0 */ 
__device__ __inline__ PSint_0::PSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 0) - PCx * VY(0, 0, 0);
  x_2_0 = PAy * VY(0, 0, 0) - PCy * VY(0, 0, 0);
  x_3_0 = PAz * VY(0, 0, 0) - PCz * VY(0, 0, 0);
} 


/* PS auxilary integral, m=1 */ 
__device__ __inline__ PSint_1::PSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 1) - PCx * VY(0, 0, 1);
  x_2_0 = PAy * VY(0, 0, 1) - PCy * VY(0, 0, 1);
  x_3_0 = PAz * VY(0, 0, 1) - PCz * VY(0, 0, 1);
} 


/* PS auxilary integral, m=2 */ 
__device__ __inline__ PSint_2::PSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 2) - PCx * VY(0, 0, 2);
  x_2_0 = PAy * VY(0, 0, 2) - PCy * VY(0, 0, 2);
  x_3_0 = PAz * VY(0, 0, 2) - PCz * VY(0, 0, 2);
} 


/* PS auxilary integral, m=3 */ 
__device__ __inline__ PSint_3::PSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_1_0 = PAx * VY(0, 0, 3) - PCx * VY(0, 0, 3);
  x_2_0 = PAy * VY(0, 0, 3) - PCy * VY(0, 0, 3);
  x_3_0 = PAz * VY(0, 0, 3) - PCz * VY(0, 0, 3);
} 


/* SP true integral, m=0 */ 
__device__ __inline__ SPint_0::SPint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 0) - PCx * VY(0, 0, 0);
  x_0_2 = PBy * VY(0, 0, 0) - PCy * VY(0, 0, 0);
  x_0_3 = PBz * VY(0, 0, 0) - PCz * VY(0, 0, 0);
} 


/* SP auxilary integral, m=1 */ 
__device__ __inline__ SPint_1::SPint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 1) - PCx * VY(0, 0, 1);
  x_0_2 = PBy * VY(0, 0, 1) - PCy * VY(0, 0, 1);
  x_0_3 = PBz * VY(0, 0, 1) - PCz * VY(0, 0, 1);
} 


/* SP auxilary integral, m=2 */ 
__device__ __inline__ SPint_2::SPint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 2) - PCx * VY(0, 0, 2);
  x_0_2 = PBy * VY(0, 0, 2) - PCy * VY(0, 0, 2);
  x_0_3 = PBz * VY(0, 0, 2) - PCz * VY(0, 0, 2);
} 


/* SP auxilary integral, m=3 */ 
__device__ __inline__ SPint_3::SPint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp){ 

  x_0_1 = PBx * VY(0, 0, 3) - PCx * VY(0, 0, 3);
  x_0_2 = PBy * VY(0, 0, 3) - PCy * VY(0, 0, 3);
  x_0_3 = PBz * VY(0, 0, 3) - PCz * VY(0, 0, 3);
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

/* PD true integral, m=0 */ 
__device__ __inline__ PDint_0::PDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp){ 

  SPint_0 sp_0(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=0 
  SPint_1 sp_1(PBx, PBy, PBz, PCx, PCy, PCz, YVerticalTemp); // construct [s|p] for m=1 
  SDint_0 sd_0(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=0 
  SDint_1 sd_1(PAx, PAy, PAz, PCx, PCy, PCz, Zeta, YVerticalTemp); // construct [s|d] for m=1 

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
  x_4_4 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_4_4 += 0.5/Zeta * 1.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  x_5_4 = PBy * dp_0.x_4_3 - PCy * dp_1.x_4_3; 
  x_5_4 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_5_4 += 0.5/Zeta * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_6_4 = PBx * dp_0.x_4_3 - PCx * dp_1.x_4_3; 
  x_6_4 += 0.5/Zeta * 1.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_6_4 += 0.5/Zeta * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_7_4 = PBx * dp_0.x_4_1 - PCx * dp_1.x_4_1; 
  x_7_4 += 0.5/Zeta * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_7_4 += 0.5/Zeta * 1.000000 * (pp_0.x_2_1 - pp_1.x_2_1); 
  x_8_4 = PBy * dp_0.x_4_2 - PCy * dp_1.x_4_2; 
  x_8_4 += 0.5/Zeta * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_8_4 += 0.5/Zeta * 1.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  x_9_4 = PBz * dp_0.x_4_3 - PCz * dp_1.x_4_3; 
  x_9_4 += 0.5/Zeta * 2.000000 * (ds_0.x_4_0 - ds_1.x_4_0); 
  x_4_5 = PBx * dp_0.x_5_2 - PCx * dp_1.x_5_2; 
  x_4_5 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_5_5 = PBy * dp_0.x_5_3 - PCy * dp_1.x_5_3; 
  x_5_5 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_5_5 += 0.5/Zeta * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  x_6_5 = PBx * dp_0.x_5_3 - PCx * dp_1.x_5_3; 
  x_6_5 += 0.5/Zeta * 1.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_7_5 = PBx * dp_0.x_5_1 - PCx * dp_1.x_5_1; 
  x_7_5 += 0.5/Zeta * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_8_5 = PBy * dp_0.x_5_2 - PCy * dp_1.x_5_2; 
  x_8_5 += 0.5/Zeta * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_8_5 += 0.5/Zeta * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  x_9_5 = PBz * dp_0.x_5_3 - PCz * dp_1.x_5_3; 
  x_9_5 += 0.5/Zeta * 2.000000 * (ds_0.x_5_0 - ds_1.x_5_0); 
  x_9_5 += 0.5/Zeta * 1.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_4_6 = PBx * dp_0.x_6_2 - PCx * dp_1.x_6_2; 
  x_4_6 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_4_6 += 0.5/Zeta * 1.000000 * (pp_0.x_3_2 - pp_1.x_3_2); 
  x_5_6 = PBy * dp_0.x_6_3 - PCy * dp_1.x_6_3; 
  x_5_6 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_6_6 = PBx * dp_0.x_6_3 - PCx * dp_1.x_6_3; 
  x_6_6 += 0.5/Zeta * 1.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_6_6 += 0.5/Zeta * 1.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 
  x_7_6 = PBx * dp_0.x_6_1 - PCx * dp_1.x_6_1; 
  x_7_6 += 0.5/Zeta * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_7_6 += 0.5/Zeta * 1.000000 * (pp_0.x_3_1 - pp_1.x_3_1); 
  x_8_6 = PBy * dp_0.x_6_2 - PCy * dp_1.x_6_2; 
  x_8_6 += 0.5/Zeta * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_9_6 = PBz * dp_0.x_6_3 - PCz * dp_1.x_6_3; 
  x_9_6 += 0.5/Zeta * 2.000000 * (ds_0.x_6_0 - ds_1.x_6_0); 
  x_9_6 += 0.5/Zeta * 1.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_4_7 = PBx * dp_0.x_7_2 - PCx * dp_1.x_7_2; 
  x_4_7 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_4_7 += 0.5/Zeta * 2.000000 * (pp_0.x_1_2 - pp_1.x_1_2); 
  x_5_7 = PBy * dp_0.x_7_3 - PCy * dp_1.x_7_3; 
  x_5_7 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_6_7 = PBx * dp_0.x_7_3 - PCx * dp_1.x_7_3; 
  x_6_7 += 0.5/Zeta * 1.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_6_7 += 0.5/Zeta * 2.000000 * (pp_0.x_1_3 - pp_1.x_1_3); 
  x_7_7 = PBx * dp_0.x_7_1 - PCx * dp_1.x_7_1; 
  x_7_7 += 0.5/Zeta * 2.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_7_7 += 0.5/Zeta * 2.000000 * (pp_0.x_1_1 - pp_1.x_1_1); 
  x_8_7 = PBy * dp_0.x_7_2 - PCy * dp_1.x_7_2; 
  x_8_7 += 0.5/Zeta * 2.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_9_7 = PBz * dp_0.x_7_3 - PCz * dp_1.x_7_3; 
  x_9_7 += 0.5/Zeta * 2.000000 * (ds_0.x_7_0 - ds_1.x_7_0); 
  x_4_8 = PBx * dp_0.x_8_2 - PCx * dp_1.x_8_2; 
  x_4_8 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_5_8 = PBy * dp_0.x_8_3 - PCy * dp_1.x_8_3; 
  x_5_8 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_5_8 += 0.5/Zeta * 2.000000 * (pp_0.x_2_3 - pp_1.x_2_3); 
  x_6_8 = PBx * dp_0.x_8_3 - PCx * dp_1.x_8_3; 
  x_6_8 += 0.5/Zeta * 1.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_7_8 = PBx * dp_0.x_8_1 - PCx * dp_1.x_8_1; 
  x_7_8 += 0.5/Zeta * 2.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_8_8 = PBy * dp_0.x_8_2 - PCy * dp_1.x_8_2; 
  x_8_8 += 0.5/Zeta * 2.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_8_8 += 0.5/Zeta * 2.000000 * (pp_0.x_2_2 - pp_1.x_2_2); 
  x_9_8 = PBz * dp_0.x_8_3 - PCz * dp_1.x_8_3; 
  x_9_8 += 0.5/Zeta * 2.000000 * (ds_0.x_8_0 - ds_1.x_8_0); 
  x_4_9 = PBx * dp_0.x_9_2 - PCx * dp_1.x_9_2; 
  x_4_9 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_5_9 = PBy * dp_0.x_9_3 - PCy * dp_1.x_9_3; 
  x_5_9 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_6_9 = PBx * dp_0.x_9_3 - PCx * dp_1.x_9_3; 
  x_6_9 += 0.5/Zeta * 1.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_7_9 = PBx * dp_0.x_9_1 - PCx * dp_1.x_9_1; 
  x_7_9 += 0.5/Zeta * 2.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_8_9 = PBy * dp_0.x_9_2 - PCy * dp_1.x_9_2; 
  x_8_9 += 0.5/Zeta * 2.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_9 = PBz * dp_0.x_9_3 - PCz * dp_1.x_9_3; 
  x_9_9 += 0.5/Zeta * 2.000000 * (ds_0.x_9_0 - ds_1.x_9_0); 
  x_9_9 += 0.5/Zeta * 2.000000 * (pp_0.x_3_3 - pp_1.x_3_3); 

 } 
