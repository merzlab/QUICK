#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            3  B =            0
__device__ __inline__  f_3_0_t :: f_3_0_t ( f_2_0_t t_2_0_0, f_2_0_t t_2_0_1, f_1_0_t t_1_0_0, f_1_0_t t_1_0_1, QUICKDouble ABtemp, QUICKDouble CDcom, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_10_0 = Ptempx * t_2_0_0.x_5_0 + WPtempx * t_2_0_1.x_5_0 ;
    x_11_0 = Ptempx * t_2_0_0.x_4_0 + WPtempx * t_2_0_1.x_4_0 + ABtemp * ( t_1_0_0.x_2_0 -  CDcom * t_1_0_1.x_2_0 ) ;
    x_12_0 = Ptempx * t_2_0_0.x_8_0 + WPtempx * t_2_0_1.x_8_0 ;
    x_13_0 = Ptempx * t_2_0_0.x_6_0 + WPtempx * t_2_0_1.x_6_0 + ABtemp * ( t_1_0_0.x_3_0 -  CDcom * t_1_0_1.x_3_0 ) ;
    x_14_0 = Ptempx * t_2_0_0.x_9_0 + WPtempx * t_2_0_1.x_9_0 ;
    x_15_0 = Ptempy * t_2_0_0.x_5_0 + WPtempy * t_2_0_1.x_5_0 + ABtemp * ( t_1_0_0.x_3_0 -  CDcom * t_1_0_1.x_3_0 ) ;
    x_16_0 = Ptempy * t_2_0_0.x_9_0 + WPtempy * t_2_0_1.x_9_0 ;
    x_17_0 = Ptempx * t_2_0_0.x_7_0 + WPtempx * t_2_0_1.x_7_0 + ABtemp * 2 * ( t_1_0_0.x_1_0 -  CDcom * t_1_0_1.x_1_0 ) ;
    x_18_0 = Ptempy * t_2_0_0.x_8_0 + WPtempy * t_2_0_1.x_8_0 + ABtemp * 2 * ( t_1_0_0.x_2_0 -  CDcom * t_1_0_1.x_2_0 ) ;
    x_19_0 = Ptempz * t_2_0_0.x_9_0 + WPtempz * t_2_0_1.x_9_0 + ABtemp * 2 * ( t_1_0_0.x_3_0 -  CDcom * t_1_0_1.x_3_0 ) ;
}

