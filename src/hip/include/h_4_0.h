#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            4  B =            0
__device__ __inline__  f_4_0_t :: f_4_0_t ( f_3_0_t t_3_0_0, f_3_0_t t_3_0_1, f_2_0_t t_2_0_0, f_2_0_t t_2_0_1, QUICKDouble ABtemp, QUICKDouble CDcom, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_20_0 = Ptempx * t_3_0_0.x_12_0 + WPtempx * t_3_0_1.x_12_0 + ABtemp * ( t_2_0_0.x_8_0 -  CDcom * t_2_0_1.x_8_0 ) ;
    x_21_0 = Ptempx * t_3_0_0.x_14_0 + WPtempx * t_3_0_1.x_14_0 + ABtemp * ( t_2_0_0.x_9_0 -  CDcom * t_2_0_1.x_9_0 ) ;
    x_22_0 = Ptempy * t_3_0_0.x_16_0 + WPtempy * t_3_0_1.x_16_0 + ABtemp * ( t_2_0_0.x_9_0 -  CDcom * t_2_0_1.x_9_0 ) ;
    x_23_0 = Ptempx * t_3_0_0.x_10_0 + WPtempx * t_3_0_1.x_10_0 + ABtemp * ( t_2_0_0.x_5_0 -  CDcom * t_2_0_1.x_5_0 ) ;
    x_24_0 = Ptempx * t_3_0_0.x_15_0 + WPtempx * t_3_0_1.x_15_0 ;
    x_25_0 = Ptempx * t_3_0_0.x_16_0 + WPtempx * t_3_0_1.x_16_0 ;
    x_26_0 = Ptempx * t_3_0_0.x_13_0 + WPtempx * t_3_0_1.x_13_0 + ABtemp * 2 * ( t_2_0_0.x_6_0 -  CDcom * t_2_0_1.x_6_0 ) ;
    x_27_0 = Ptempx * t_3_0_0.x_19_0 + WPtempx * t_3_0_1.x_19_0 ;
    x_28_0 = Ptempx * t_3_0_0.x_11_0 + WPtempx * t_3_0_1.x_11_0 + ABtemp * 2 * ( t_2_0_0.x_4_0 -  CDcom * t_2_0_1.x_4_0 ) ;
    x_29_0 = Ptempx * t_3_0_0.x_18_0 + WPtempx * t_3_0_1.x_18_0 ;
    x_30_0 = Ptempy * t_3_0_0.x_15_0 + WPtempy * t_3_0_1.x_15_0 + ABtemp * 2 * ( t_2_0_0.x_5_0 -  CDcom * t_2_0_1.x_5_0 ) ;
    x_31_0 = Ptempy * t_3_0_0.x_19_0 + WPtempy * t_3_0_1.x_19_0 ;
    x_32_0 = Ptempx * t_3_0_0.x_17_0 + WPtempx * t_3_0_1.x_17_0 + ABtemp * 3 * ( t_2_0_0.x_7_0 -  CDcom * t_2_0_1.x_7_0 ) ;
    x_33_0 = Ptempy * t_3_0_0.x_18_0 + WPtempy * t_3_0_1.x_18_0 + ABtemp * 3 * ( t_2_0_0.x_8_0 -  CDcom * t_2_0_1.x_8_0 ) ;
    x_34_0 = Ptempz * t_3_0_0.x_19_0 + WPtempz * t_3_0_1.x_19_0 + ABtemp * 3 * ( t_2_0_0.x_9_0 -  CDcom * t_2_0_1.x_9_0 ) ;
}

