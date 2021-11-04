#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            1  B =            4
__device__ __inline__  f_1_4_t :: f_1_4_t ( f_0_4_t t_0_4_0, f_0_4_t t_0_4_1,  f_0_3_t t_0_3_1, QUICKDouble ABCDtemp, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_1_20 = Ptempx * t_0_4_0.x_0_20 + WPtempx * t_0_4_1.x_0_20 + 2 * ABCDtemp * t_0_3_1.x_0_12 ;
    x_1_21 = Ptempx * t_0_4_0.x_0_21 + WPtempx * t_0_4_1.x_0_21 + 2 * ABCDtemp * t_0_3_1.x_0_14 ;
    x_1_22 = Ptempx * t_0_4_0.x_0_22 + WPtempx * t_0_4_1.x_0_22 ;
    x_1_23 = Ptempx * t_0_4_0.x_0_23 + WPtempx * t_0_4_1.x_0_23 + 2 * ABCDtemp * t_0_3_1.x_0_10 ;
    x_1_24 = Ptempx * t_0_4_0.x_0_24 + WPtempx * t_0_4_1.x_0_24 + ABCDtemp * t_0_3_1.x_0_15 ;
    x_1_25 = Ptempx * t_0_4_0.x_0_25 + WPtempx * t_0_4_1.x_0_25 + ABCDtemp * t_0_3_1.x_0_16 ;
    x_1_26 = Ptempx * t_0_4_0.x_0_26 + WPtempx * t_0_4_1.x_0_26 + 3 * ABCDtemp * t_0_3_1.x_0_13 ;
    x_1_27 = Ptempx * t_0_4_0.x_0_27 + WPtempx * t_0_4_1.x_0_27 + ABCDtemp * t_0_3_1.x_0_19 ;
    x_1_28 = Ptempx * t_0_4_0.x_0_28 + WPtempx * t_0_4_1.x_0_28 + 3 * ABCDtemp * t_0_3_1.x_0_11 ;
    x_1_29 = Ptempx * t_0_4_0.x_0_29 + WPtempx * t_0_4_1.x_0_29 + ABCDtemp * t_0_3_1.x_0_18 ;
    x_1_30 = Ptempx * t_0_4_0.x_0_30 + WPtempx * t_0_4_1.x_0_30 ;
    x_1_31 = Ptempx * t_0_4_0.x_0_31 + WPtempx * t_0_4_1.x_0_31 ;
    x_1_32 = Ptempx * t_0_4_0.x_0_32 + WPtempx * t_0_4_1.x_0_32 + 4 * ABCDtemp * t_0_3_1.x_0_17 ;
    x_1_33 = Ptempx * t_0_4_0.x_0_33 + WPtempx * t_0_4_1.x_0_33 ;
    x_1_34 = Ptempx * t_0_4_0.x_0_34 + WPtempx * t_0_4_1.x_0_34 ;
    x_2_20 = Ptempy * t_0_4_0.x_0_20 + WPtempy * t_0_4_1.x_0_20 + 2 * ABCDtemp * t_0_3_1.x_0_11 ;
    x_2_21 = Ptempy * t_0_4_0.x_0_21 + WPtempy * t_0_4_1.x_0_21 ;
    x_2_22 = Ptempy * t_0_4_0.x_0_22 + WPtempy * t_0_4_1.x_0_22 + 2 * ABCDtemp * t_0_3_1.x_0_16 ;
    x_2_23 = Ptempy * t_0_4_0.x_0_23 + WPtempy * t_0_4_1.x_0_23 + ABCDtemp * t_0_3_1.x_0_13 ;
    x_2_24 = Ptempy * t_0_4_0.x_0_24 + WPtempy * t_0_4_1.x_0_24 + 2 * ABCDtemp * t_0_3_1.x_0_10 ;
    x_2_25 = Ptempy * t_0_4_0.x_0_25 + WPtempy * t_0_4_1.x_0_25 + ABCDtemp * t_0_3_1.x_0_14 ;
    x_2_26 = Ptempy * t_0_4_0.x_0_26 + WPtempy * t_0_4_1.x_0_26 ;
    x_2_27 = Ptempy * t_0_4_0.x_0_27 + WPtempy * t_0_4_1.x_0_27 ;
    x_2_28 = Ptempy * t_0_4_0.x_0_28 + WPtempy * t_0_4_1.x_0_28 + ABCDtemp * t_0_3_1.x_0_17 ;
    x_2_29 = Ptempy * t_0_4_0.x_0_29 + WPtempy * t_0_4_1.x_0_29 + 3 * ABCDtemp * t_0_3_1.x_0_12 ;
    x_2_30 = Ptempy * t_0_4_0.x_0_30 + WPtempy * t_0_4_1.x_0_30 + 3 * ABCDtemp * t_0_3_1.x_0_15 ;
    x_2_31 = Ptempy * t_0_4_0.x_0_31 + WPtempy * t_0_4_1.x_0_31 + ABCDtemp * t_0_3_1.x_0_19 ;
    x_2_32 = Ptempy * t_0_4_0.x_0_32 + WPtempy * t_0_4_1.x_0_32 ;
    x_2_33 = Ptempy * t_0_4_0.x_0_33 + WPtempy * t_0_4_1.x_0_33 + 4 * ABCDtemp * t_0_3_1.x_0_18 ;
    x_2_34 = Ptempy * t_0_4_0.x_0_34 + WPtempy * t_0_4_1.x_0_34 ;
    x_3_20 = Ptempz * t_0_4_0.x_0_20 + WPtempz * t_0_4_1.x_0_20 ;
    x_3_21 = Ptempz * t_0_4_0.x_0_21 + WPtempz * t_0_4_1.x_0_21 + 2 * ABCDtemp * t_0_3_1.x_0_13 ;
    x_3_22 = Ptempz * t_0_4_0.x_0_22 + WPtempz * t_0_4_1.x_0_22 + 2 * ABCDtemp * t_0_3_1.x_0_15 ;
    x_3_23 = Ptempz * t_0_4_0.x_0_23 + WPtempz * t_0_4_1.x_0_23 + ABCDtemp * t_0_3_1.x_0_11 ;
    x_3_24 = Ptempz * t_0_4_0.x_0_24 + WPtempz * t_0_4_1.x_0_24 + ABCDtemp * t_0_3_1.x_0_12 ;
    x_3_25 = Ptempz * t_0_4_0.x_0_25 + WPtempz * t_0_4_1.x_0_25 + 2 * ABCDtemp * t_0_3_1.x_0_10 ;
    x_3_26 = Ptempz * t_0_4_0.x_0_26 + WPtempz * t_0_4_1.x_0_26 + ABCDtemp * t_0_3_1.x_0_17 ;
    x_3_27 = Ptempz * t_0_4_0.x_0_27 + WPtempz * t_0_4_1.x_0_27 + 3 * ABCDtemp * t_0_3_1.x_0_14 ;
    x_3_28 = Ptempz * t_0_4_0.x_0_28 + WPtempz * t_0_4_1.x_0_28 ;
    x_3_29 = Ptempz * t_0_4_0.x_0_29 + WPtempz * t_0_4_1.x_0_29 ;
    x_3_30 = Ptempz * t_0_4_0.x_0_30 + WPtempz * t_0_4_1.x_0_30 + ABCDtemp * t_0_3_1.x_0_18 ;
    x_3_31 = Ptempz * t_0_4_0.x_0_31 + WPtempz * t_0_4_1.x_0_31 + 3 * ABCDtemp * t_0_3_1.x_0_16 ;
    x_3_32 = Ptempz * t_0_4_0.x_0_32 + WPtempz * t_0_4_1.x_0_32 ;
    x_3_33 = Ptempz * t_0_4_0.x_0_33 + WPtempz * t_0_4_1.x_0_33 ;
    x_3_34 = Ptempz * t_0_4_0.x_0_34 + WPtempz * t_0_4_1.x_0_34 + 4 * ABCDtemp * t_0_3_1.x_0_19 ;
}
