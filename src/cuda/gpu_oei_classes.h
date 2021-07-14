/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 14/07/2021                !
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
class PSint_0{ 
public: 
  QUICKDouble x_1_0; // Px, S 
  QUICKDouble x_2_0; // Py, S 
  QUICKDouble x_3_0; // Pz, S 
  __device__ __inline__ PSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* PS auxilary integral, m=1 */ 
class PSint_1{ 
public: 
  QUICKDouble x_1_0; // Px, S 
  QUICKDouble x_2_0; // Py, S 
  QUICKDouble x_3_0; // Pz, S 
  __device__ __inline__ PSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* PS auxilary integral, m=2 */ 
class PSint_2{ 
public: 
  QUICKDouble x_1_0; // Px, S 
  QUICKDouble x_2_0; // Py, S 
  QUICKDouble x_3_0; // Pz, S 
  __device__ __inline__ PSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* PS auxilary integral, m=3 */ 
class PSint_3{ 
public: 
  QUICKDouble x_1_0; // Px, S 
  QUICKDouble x_2_0; // Py, S 
  QUICKDouble x_3_0; // Pz, S 
  __device__ __inline__ PSint_3(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* SP true integral, m=0 */ 
class SPint_0{ 
public: 
  QUICKDouble x_0_1; // S, Px 
  QUICKDouble x_0_2; // S, Py 
  QUICKDouble x_0_3; // S, Pz 
  __device__ __inline__ SPint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* SP auxilary integral, m=1 */ 
class SPint_1{ 
public: 
  QUICKDouble x_0_1; // S, Px 
  QUICKDouble x_0_2; // S, Py 
  QUICKDouble x_0_3; // S, Pz 
  __device__ __inline__ SPint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* SP auxilary integral, m=2 */ 
class SPint_2{ 
public: 
  QUICKDouble x_0_1; // S, Px 
  QUICKDouble x_0_2; // S, Py 
  QUICKDouble x_0_3; // S, Pz 
  __device__ __inline__ SPint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* SP auxilary integral, m=3 */ 
class SPint_3{ 
public: 
  QUICKDouble x_0_1; // S, Px 
  QUICKDouble x_0_2; // S, Py 
  QUICKDouble x_0_3; // S, Pz 
  __device__ __inline__ SPint_3(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble* YVerticalTemp); 
}; 

/* PP true integral, m=0 */ 
class PPint_0{ 
public: 
  QUICKDouble x_1_1; // Px, Px 
  QUICKDouble x_1_2; // Px, Py 
  QUICKDouble x_1_3; // Px, Pz 
  QUICKDouble x_2_1; // Py, Px 
  QUICKDouble x_2_2; // Py, Py 
  QUICKDouble x_2_3; // Py, Pz 
  QUICKDouble x_3_1; // Pz, Px 
  QUICKDouble x_3_2; // Pz, Py 
  QUICKDouble x_3_3; // Pz, Pz 
  __device__ __inline__ PPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* PP auxilary integral, m=1 */ 
class PPint_1{ 
public: 
  QUICKDouble x_1_1; // Px, Px 
  QUICKDouble x_1_2; // Px, Py 
  QUICKDouble x_1_3; // Px, Pz 
  QUICKDouble x_2_1; // Py, Px 
  QUICKDouble x_2_2; // Py, Py 
  QUICKDouble x_2_3; // Py, Pz 
  QUICKDouble x_3_1; // Pz, Px 
  QUICKDouble x_3_2; // Pz, Py 
  QUICKDouble x_3_3; // Pz, Pz 
  __device__ __inline__ PPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* DS true integral, m=0 */ 
class DSint_0{ 
public: 
  QUICKDouble x_4_0; // Dxy, S 
  QUICKDouble x_5_0; // Dyz, S 
  QUICKDouble x_6_0; // Dxz, S 
  QUICKDouble x_7_0; // Dxx, S 
  QUICKDouble x_8_0; // Dyy, S 
  QUICKDouble x_9_0; // Dzz, S 
  __device__ __inline__ DSint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* DS auxilary integral, m=1 */ 
class DSint_1{ 
public: 
  QUICKDouble x_4_0; // Dxy, S 
  QUICKDouble x_5_0; // Dyz, S 
  QUICKDouble x_6_0; // Dxz, S 
  QUICKDouble x_7_0; // Dxx, S 
  QUICKDouble x_8_0; // Dyy, S 
  QUICKDouble x_9_0; // Dzz, S 
  __device__ __inline__ DSint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* DS auxilary integral, m=2 */ 
class DSint_2{ 
public: 
  QUICKDouble x_4_0; // Dxy, S 
  QUICKDouble x_5_0; // Dyz, S 
  QUICKDouble x_6_0; // Dxz, S 
  QUICKDouble x_7_0; // Dxx, S 
  QUICKDouble x_8_0; // Dyy, S 
  QUICKDouble x_9_0; // Dzz, S 
  __device__ __inline__ DSint_2(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* SD true integral, m=0 */ 
class SDint_0{ 
public: 
  QUICKDouble x_0_4; // S, Dxy 
  QUICKDouble x_0_5; // S, Dyz 
  QUICKDouble x_0_6; // S, Dxz 
  QUICKDouble x_0_7; // S, Dxx 
  QUICKDouble x_0_8; // S, Dyy 
  QUICKDouble x_0_9; // S, Dzz 
  __device__ __inline__ SDint_0(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* SD auxilary integral, m=1 */ 
class SDint_1{ 
public: 
  QUICKDouble x_0_4; // S, Dxy 
  QUICKDouble x_0_5; // S, Dyz 
  QUICKDouble x_0_6; // S, Dxz 
  QUICKDouble x_0_7; // S, Dxx 
  QUICKDouble x_0_8; // S, Dyy 
  QUICKDouble x_0_9; // S, Dzz 
  __device__ __inline__ SDint_1(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* SD auxilary integral, m=2 */ 
class SDint_2{ 
public: 
  QUICKDouble x_0_4; // S, Dxy 
  QUICKDouble x_0_5; // S, Dyz 
  QUICKDouble x_0_6; // S, Dxz 
  QUICKDouble x_0_7; // S, Dxx 
  QUICKDouble x_0_8; // S, Dyy 
  QUICKDouble x_0_9; // S, Dzz 
  __device__ __inline__ SDint_2(QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
                QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* DP true integral, m=0 */ 
class DPint_0{ 
public: 
  QUICKDouble x_4_1; // Dxy, Px 
  QUICKDouble x_4_2; // Dxy, Py 
  QUICKDouble x_4_3; // Dxy, Pz 
  QUICKDouble x_5_1; // Dyz, Px 
  QUICKDouble x_5_2; // Dyz, Py 
  QUICKDouble x_5_3; // Dyz, Pz 
  QUICKDouble x_6_1; // Dxz, Px 
  QUICKDouble x_6_2; // Dxz, Py 
  QUICKDouble x_6_3; // Dxz, Pz 
  QUICKDouble x_7_1; // Dxx, Px 
  QUICKDouble x_7_2; // Dxx, Py 
  QUICKDouble x_7_3; // Dxx, Pz 
  QUICKDouble x_8_1; // Dyy, Px 
  QUICKDouble x_8_2; // Dyy, Py 
  QUICKDouble x_8_3; // Dyy, Pz 
  QUICKDouble x_9_1; // Dzz, Px 
  QUICKDouble x_9_2; // Dzz, Py 
  QUICKDouble x_9_3; // Dzz, Pz 
  __device__ __inline__ DPint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* DP auxilary integral, m=1 */ 
class DPint_1{ 
public: 
  QUICKDouble x_4_1; // Dxy, Px 
  QUICKDouble x_4_2; // Dxy, Py 
  QUICKDouble x_4_3; // Dxy, Pz 
  QUICKDouble x_5_1; // Dyz, Px 
  QUICKDouble x_5_2; // Dyz, Py 
  QUICKDouble x_5_3; // Dyz, Pz 
  QUICKDouble x_6_1; // Dxz, Px 
  QUICKDouble x_6_2; // Dxz, Py 
  QUICKDouble x_6_3; // Dxz, Pz 
  QUICKDouble x_7_1; // Dxx, Px 
  QUICKDouble x_7_2; // Dxx, Py 
  QUICKDouble x_7_3; // Dxx, Pz 
  QUICKDouble x_8_1; // Dyy, Px 
  QUICKDouble x_8_2; // Dyy, Py 
  QUICKDouble x_8_3; // Dyy, Pz 
  QUICKDouble x_9_1; // Dzz, Px 
  QUICKDouble x_9_2; // Dzz, Py 
  QUICKDouble x_9_3; // Dzz, Pz 
  __device__ __inline__ DPint_1(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* PD true integral, m=0 */ 
class PDint_0{ 
public: 
  QUICKDouble x_1_4; // Px, Dxy 
  QUICKDouble x_2_4; // Py, Dxy 
  QUICKDouble x_3_4; // Pz, Dxy 
  QUICKDouble x_1_5; // Px, Dyz 
  QUICKDouble x_2_5; // Py, Dyz 
  QUICKDouble x_3_5; // Pz, Dyz 
  QUICKDouble x_1_6; // Px, Dxz 
  QUICKDouble x_2_6; // Py, Dxz 
  QUICKDouble x_3_6; // Pz, Dxz 
  QUICKDouble x_1_7; // Px, Dxx 
  QUICKDouble x_2_7; // Py, Dxx 
  QUICKDouble x_3_7; // Pz, Dxx 
  QUICKDouble x_1_8; // Px, Dyy 
  QUICKDouble x_2_8; // Py, Dyy 
  QUICKDouble x_3_8; // Pz, Dyy 
  QUICKDouble x_1_9; // Px, Dzz 
  QUICKDouble x_2_9; // Py, Dzz 
  QUICKDouble x_3_9; // Pz, Dzz 
  __device__ __inline__ PDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 

/* DD true integral, m=0 */ 
class DDint_0{ 
public: 
  QUICKDouble x_4_4; // Dxy, Dxy 
  QUICKDouble x_4_5; // Dxy, Dyz 
  QUICKDouble x_4_6; // Dxy, Dxz 
  QUICKDouble x_4_7; // Dxy, Dxx 
  QUICKDouble x_4_8; // Dxy, Dyy 
  QUICKDouble x_4_9; // Dxy, Dzz 
  QUICKDouble x_5_4; // Dyz, Dxy 
  QUICKDouble x_5_5; // Dyz, Dyz 
  QUICKDouble x_5_6; // Dyz, Dxz 
  QUICKDouble x_5_7; // Dyz, Dxx 
  QUICKDouble x_5_8; // Dyz, Dyy 
  QUICKDouble x_5_9; // Dyz, Dzz 
  QUICKDouble x_6_4; // Dxz, Dxy 
  QUICKDouble x_6_5; // Dxz, Dyz 
  QUICKDouble x_6_6; // Dxz, Dxz 
  QUICKDouble x_6_7; // Dxz, Dxx 
  QUICKDouble x_6_8; // Dxz, Dyy 
  QUICKDouble x_6_9; // Dxz, Dzz 
  QUICKDouble x_7_4; // Dxx, Dxy 
  QUICKDouble x_7_5; // Dxx, Dyz 
  QUICKDouble x_7_6; // Dxx, Dxz 
  QUICKDouble x_7_7; // Dxx, Dxx 
  QUICKDouble x_7_8; // Dxx, Dyy 
  QUICKDouble x_7_9; // Dxx, Dzz 
  QUICKDouble x_8_4; // Dyy, Dxy 
  QUICKDouble x_8_5; // Dyy, Dyz 
  QUICKDouble x_8_6; // Dyy, Dxz 
  QUICKDouble x_8_7; // Dyy, Dxx 
  QUICKDouble x_8_8; // Dyy, Dyy 
  QUICKDouble x_8_9; // Dyy, Dzz 
  QUICKDouble x_9_4; // Dzz, Dxy 
  QUICKDouble x_9_5; // Dzz, Dyz 
  QUICKDouble x_9_6; // Dzz, Dxz 
  QUICKDouble x_9_7; // Dzz, Dxx 
  QUICKDouble x_9_8; // Dzz, Dyy 
  QUICKDouble x_9_9; // Dzz, Dzz 
  __device__ __inline__ DDint_0(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,
                QUICKDouble Zeta, QUICKDouble* YVerticalTemp); 
}; 
