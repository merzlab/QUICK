#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for B =            5  L =            0
__device__ __inline__  f_5_0_t :: f_5_0_t ( f_4_0_t t_4_0_0, f_4_0_t t_4_0_1, f_3_0_t t_3_0_0, f_3_0_t t_3_0_1, QUICKDouble ABtemp, QUICKDouble CDcom, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_35_0 = Ptempx * t_4_0_0.x_22_0 + WPtempx * t_4_0_1.x_22_0 ;
    x_36_0 = Ptempx * t_4_0_0.x_25_0 + WPtempx * t_4_0_1.x_25_0 + ABtemp * ( t_3_0_0.x_16_0 -  CDcom * t_3_0_1.x_16_0 ) ;
    x_37_0 = Ptempx * t_4_0_0.x_24_0 + WPtempx * t_4_0_1.x_24_0 + ABtemp * ( t_3_0_0.x_15_0 -  CDcom * t_3_0_1.x_15_0 ) ;
    x_38_0 = Ptempx * t_4_0_0.x_23_0 + WPtempx * t_4_0_1.x_23_0 + ABtemp * 2 * ( t_3_0_0.x_10_0 -  CDcom * t_3_0_1.x_10_0 ) ;
    x_39_0 = Ptempx * t_4_0_0.x_30_0 + WPtempx * t_4_0_1.x_30_0 ;
    x_40_0 = Ptempx * t_4_0_0.x_31_0 + WPtempx * t_4_0_1.x_31_0 ;
    x_41_0 = Ptempy * t_4_0_0.x_31_0 + WPtempy * t_4_0_1.x_31_0 + ABtemp * ( t_3_0_0.x_19_0 -  CDcom * t_3_0_1.x_19_0 ) ;
    x_42_0 = Ptempy * t_4_0_0.x_22_0 + WPtempy * t_4_0_1.x_22_0 + ABtemp * 2 * ( t_3_0_0.x_16_0 -  CDcom * t_3_0_1.x_16_0 ) ;
    x_43_0 = Ptempx * t_4_0_0.x_27_0 + WPtempx * t_4_0_1.x_27_0 + ABtemp * ( t_3_0_0.x_19_0 -  CDcom * t_3_0_1.x_19_0 ) ;
    x_44_0 = Ptempx * t_4_0_0.x_21_0 + WPtempx * t_4_0_1.x_21_0 + ABtemp * 2 * ( t_3_0_0.x_14_0 -  CDcom * t_3_0_1.x_14_0 ) ;
    x_45_0 = Ptempx * t_4_0_0.x_29_0 + WPtempx * t_4_0_1.x_29_0 + ABtemp * ( t_3_0_0.x_18_0 -  CDcom * t_3_0_1.x_18_0 ) ;
    x_46_0 = Ptempx * t_4_0_0.x_20_0 + WPtempx * t_4_0_1.x_20_0 + ABtemp * 2 * ( t_3_0_0.x_12_0 -  CDcom * t_3_0_1.x_12_0 ) ;
    x_47_0 = Ptempy * t_4_0_0.x_34_0 + WPtempy * t_4_0_1.x_34_0 ;
    x_48_0 = Ptempy * t_4_0_0.x_30_0 + WPtempy * t_4_0_1.x_30_0 + ABtemp * 3 * ( t_3_0_0.x_15_0 -  CDcom * t_3_0_1.x_15_0 ) ;
    x_49_0 = Ptempx * t_4_0_0.x_34_0 + WPtempx * t_4_0_1.x_34_0 ;
    x_50_0 = Ptempx * t_4_0_0.x_26_0 + WPtempx * t_4_0_1.x_26_0 + ABtemp * 3 * ( t_3_0_0.x_13_0 -  CDcom * t_3_0_1.x_13_0 ) ;
    x_51_0 = Ptempx * t_4_0_0.x_33_0 + WPtempx * t_4_0_1.x_33_0 ;
    x_52_0 = Ptempx * t_4_0_0.x_28_0 + WPtempx * t_4_0_1.x_28_0 + ABtemp * 3 * ( t_3_0_0.x_11_0 -  CDcom * t_3_0_1.x_11_0 ) ;
    x_53_0 = Ptempx * t_4_0_0.x_32_0 + WPtempx * t_4_0_1.x_32_0 + ABtemp * 4 * ( t_3_0_0.x_17_0 -  CDcom * t_3_0_1.x_17_0 ) ;
    x_54_0 = Ptempy * t_4_0_0.x_33_0 + WPtempy * t_4_0_1.x_33_0 + ABtemp * 4 * ( t_3_0_0.x_18_0 -  CDcom * t_3_0_1.x_18_0 ) ;
    x_55_0 = Ptempz * t_4_0_0.x_34_0 + WPtempz * t_4_0_1.x_34_0 + ABtemp * 4 * ( t_3_0_0.x_19_0 -  CDcom * t_3_0_1.x_19_0 ) ;
}
